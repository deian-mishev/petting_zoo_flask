import threading
import time
from collections import Counter
import numpy as np
import tensorflow as tf
from flask_socketio import SocketIO
from app import app
from app.config.session_state import SessionState
from app.services.rendering_service import render_frame
from app.services.frame_delivery import FrameDelivery
from app.services.ml_service import ml_service
from app.config.env_config import INACTIVITY_TO, INPUT_TIMEOUT, REWARD_NEG_FACTO, REWARD_POS_FACTOR
from app.services.session import cleanup_session
from app.services.training_metrics import TrainingMetrics
from app.config.player_state import Experience, PlayerType

class SessionRunner:
    def __init__(self, sid: str, session: SessionState, socketio: SocketIO, *, frame_ack=False):
        self.sid = sid
        self.session = session
        self.socketio = socketio
        self._stop_event = threading.Event()
        self._running_thread = None
        self._transitions = Counter()
        self._observation_history = {}
        self._match_update = None
        self._frames = FrameDelivery(
            lambda frame, **kwargs: self.socketio.emit('frame', frame, room=self.sid, **kwargs),
            require_ack=frame_ack,
        )

    def start(self):
        self._running_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._running_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._running_thread:
            self._running_thread.join()

    def _should_stop(self):
        return self._stop_event.is_set()

    def _run_loop(self):
        while self.session.current_agent is not None and not self._should_stop():
            try:
                _, _, acted = self._step_turn()
                if acted:
                    self._frames.send_if_due(lambda: render_frame(self.session))
                    time.sleep(INPUT_TIMEOUT)
            except Exception as e:
                app.logger.error(f"{self.sid}: Error emitting frame: {e}")
                time.sleep(INPUT_TIMEOUT)

        if self.session.current_agent is None:
            status = self.update_after_match(notify=True)
            cleanup_session(self.sid, False)
            self.socketio.emit('episode_end', {
                'message': 'Episode ended', 'model_update': status,
            }, room=self.sid)

    def update_after_match(self, *, notify=False):
        if self._match_update is not None:
            return self._match_update

        updates = ml_service.match_updates(self.session, self._transitions)
        if not any(updates.values()):
            self._match_update = 'skipped'
            return self._match_update

        try:
            if notify:
                self.socketio.emit('model_update_started', {}, room=self.sid)
            ml_service.train_model(self.sid, self.session, updates_by_agent=updates)
            self._match_update = 'completed'
        except Exception:
            self._match_update = 'failed'
            app.logger.exception(f"{self.sid}: Error updating model after match")
        return self._match_update

    def start_training(self, max_episodes=1):
        if self._running_thread and self._running_thread.is_alive():
            app.logger.info(f"{self.sid}: Training is already running.")
            return
        self._running_thread = threading.Thread(target=self._run_training, args=(max_episodes,), daemon=True)
        self._running_thread.start()

    def _run_training(self, max_episodes: int = 1):
        metrics = TrainingMetrics(self.session.env_config.name)
        try:
            self._run_training_episodes(max_episodes, metrics)
        finally:
            metrics.close()
            cleanup_session(self.sid, False)

    def _run_training_episodes(self, max_episodes, metrics):
        environment_steps = 0
        agent_names = {id(agent): name for name, agent in self.session.agents.items()}
        for episode in range(max_episodes):
            if self._should_stop():
                break
            for agent in self.session.agents.values():
                agent.total_reward = 0
            self._transitions.clear()
            self._observation_history.clear()
            shaped_rewards = {name: 0.0 for name in self.session.agents}
            episode_epsilon = {name: ml_service.training_progress(agent, self.session.env_config.name)["epsilon"]
                               for name, agent in self.session.agents.items()}
            started = time.monotonic()
            self.session.env.reset()
            self.session.agent_iter = iter(self.session.env.agent_iter())
            self._advance_agent()
            steps = 0
            while self.session.current_agent is not None and not self._should_stop():
                agent, shaped_reward, acted = self._step_turn()
                shaped_rewards[agent_names[id(agent)]] += shaped_reward
                if acted:
                    steps += 1
                    environment_steps += 1
                    if steps % 2000 == 0:
                        app.logger.info(
                            f"{self.sid}: Training for episode {episode + 1} still in session...")
                    time.sleep(0)

            if self.session.current_agent is not None:
                break

            updates = ml_service.match_updates(self.session, self._transitions)
            ml_service.train_model(self.sid, self.session, metrics=metrics, episode_steps=steps,
                                   updates_by_agent=updates)
            for name, agent in self.session.agents.items():
                metrics.scalar(f"episode/{name}/reward", agent.total_reward, episode + 1)
                metrics.scalar(f"episode/{name}/shaped_reward", shaped_rewards[name], episode + 1)
                metrics.scalar(f"episode/{name}/replay_size", len(agent.memory_buffer), episode + 1)
                metrics.scalar(f"episode/{name}/training_updates", metrics.updates[name], episode + 1)
                metrics.scalar(f"episode/{name}/epsilon", episode_epsilon[name], episode + 1)
                progress = ml_service.training_progress(agent, self.session.env_config.name)
                metrics.scalar(f"progress/{name}/completed_episodes", progress["completed_episodes"], episode + 1)
                metrics.scalar(f"progress/{name}/environment_steps", progress["environment_steps"], episode + 1)
            metrics.scalar("episode/length", steps, episode + 1)
            metrics.scalar("episode/duration_seconds", time.monotonic() - started, episode + 1)
            metrics.scalar("progress/environment_steps", environment_steps, episode + 1)
            metrics.scalar("progress/completed_episodes", episode + 1, environment_steps)
            metrics.flush()
            app.logger.info(f"{self.sid}: Episode {episode + 1}/{max_episodes} completed.")

    def _advance_agent(self):
        name = next(self.session.agent_iter, None)
        self.session.current_agent = self.session.agents[name] if name is not None else None

    def _step_turn(self):
        agent = self.session.current_agent
        obs, reward, terminated, truncated, _ = self.session.env.last()
        done = terminated or truncated
        shape = self.session.env_config.observation_space
        observation = ml_service.preprocess_state(shape, obs)
        frame_count = shape[-1]
        if frame_count > 1:
            history = self._observation_history.get(id(agent), ())
            # Replay converts these pixels to float32 when sampling a batch.
            history = (*history, observation.astype(np.uint8))[-frame_count:]
            padding = (history[0],) * (frame_count - len(history))
            observation = np.concatenate(padding + history, axis=-1)

        # PettingZoo requires a None step to consume a finished agent's turn.
        action = None if done else self._select_action(agent, observation)
        self.session.env.step(action)
        if frame_count > 1:
            self._observation_history[id(agent)] = history

        agent.total_reward += reward
        shaped_reward = 0.0
        experience = agent.current_experience
        if experience is not None:
            shaped_reward = self.augment_reward(reward)
            experience.reward = shaped_reward
            experience.next_state = observation
            experience.done = done
            agent.memory_buffer.append(experience)
            self._transitions[id(agent)] += 1

        agent.current_experience = None if done else Experience(state=observation, action=action)
        self._advance_agent()
        return agent, shaped_reward, not done

    def _select_action(self, agent, observation):
        if agent.type == PlayerType.HUMAN:
            with self.session.lock:
                return self.session.next_human_action

        config = self.session.env_config
        q_values = agent.q_network(tf.expand_dims(observation, 0))
        if agent.type == PlayerType.ATARI_PRO:
            q_values = q_values[config.name]
        epsilon = ml_service.training_progress(agent, config.name)["epsilon"]
        return ml_service.get_action(q_values, epsilon, config.num_actions)

    def augment_reward(self, reward):
        if reward > 0:
            return reward * REWARD_POS_FACTOR
        if reward == 0:
            return INACTIVITY_TO
        return reward * REWARD_NEG_FACTO
