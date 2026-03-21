import eventlet
import tensorflow as tf
from flask_socketio import SocketIO
from app import app
from app.config.session_state import SessionState
from app.services.rendering_service import render_frame
from app.services.ml_service import ml_service
from app.config.env_config import INACTIVITY_TO, INPUT_TIMEOUT, REWARD_NEG_FACTO, REWARD_POS_FACTOR
from app.services.session import cleanup_session
from app.config.player_state import Experience, PlayerType

class SessionRunner:
    def __init__(self, sid: str, session: SessionState, socketio: SocketIO):
        self.sid = sid
        self.session = session
        self.socketio = socketio
        self._stop_event = eventlet.event.Event()
        self._running_greenlet = None

    def start(self):
        self._running_greenlet = eventlet.spawn(self._run_loop)

    def stop(self):
        if not self._stop_event.ready():
            self._stop_event.send()
        if self._running_greenlet:
            self._running_greenlet.wait()

    def _should_stop(self):
        return self._stop_event.ready()

    def _run_loop(self):
        while not self._should_stop():
            try:
                agent_in_turn, done, reward, observation = self.set_agent_in_turn_and_current_experience()
                agent_in_turn.total_reward += reward
                reward = self.augment_reward(reward)

                if agent_in_turn.type == PlayerType.HUMAN:
                    with self.session.lock:
                        action = self.session.next_human_action
                else:
                    q_values = agent_in_turn.q_network(
                        tf.expand_dims(observation, 0))
                    if agent_in_turn.type == PlayerType.ATARI_PRO:
                        q_values = q_values[self.session.env_config.name]
                    action = ml_service.get_action(
                        q_values, self.session.env_config.epsilon, self.session.env_config.num_actions)

                agent_in_turn.current_experience = Experience(
                    state=observation,
                    action=action,
                    reward=reward,
                    done=done
                )

                if done:
                    self.end_episode(observation)
                    self.socketio.emit(
                        'episode_end', {'message': 'Episode ended'}, room=self.sid)
                    return

                self.session.env.step(action)
                self.session.current_agent = self.session.agents[next(
                    self.session.agent_iter)]

                frame = render_frame(self.session)
                self.socketio.emit('frame', frame, room=self.sid)

            except Exception as e:
                app.logger.error(f"{self.sid}: Error emitting frame: {e}")

            self.socketio.sleep(INPUT_TIMEOUT)

    def start_training(self, max_episodes=1):
        if self._running_greenlet and not self._running_greenlet.dead:
            app.logger.info(f"{self.sid}: Training is already running.")
            return
        self._running_greenlet = eventlet.spawn(
            self._run_training, max_episodes)

    def _run_training(self, max_episodes: int = 1):
        for episode in range(max_episodes):
            if self._should_stop():
                break
            self.session.env.reset()
            self.session.agent_iter = iter(self.session.env.agent_iter())
            self.session.current_agent = self.session.agents[next(self.session.agent_iter)]
            done = False
            steps = 1
            while not done and not self._should_stop():
                try:
                    agent_in_turn, done, reward, observation = self.set_agent_in_turn_and_current_experience()
                    agent_in_turn.total_reward += reward
                    reward = self.augment_reward(reward)

                    q_values = agent_in_turn.q_network(
                        tf.expand_dims(observation, 0))
                    if agent_in_turn.type == PlayerType.ATARI_PRO:
                        q_values = q_values[self.session.env_config.name]
                    action = ml_service.get_action(
                        q_values, self.session.env_config.epsilon, self.session.env_config.num_actions)

                    agent_in_turn.current_experience = Experience(
                        state=observation,
                        action=action,
                        reward=reward,
                        done=done
                    )

                    if done:
                        self.end_episode(observation)
                        ml_service.train_model(self.sid, self.session)
                        break

                    self.session.env.step(action)
                    self.session.current_agent = self.session.agents[next(
                        self.session.agent_iter)]
                    if steps % 2000 == 0:
                        app.logger.info(
                            f"{self.sid}: Training for episode {episode + 1} still in session...")
                    steps += 1
                    eventlet.sleep(0)
                except Exception as e:
                    app.logger.error(f"{self.sid}: Error in game loop: {e}")

            app.logger.info(
                f"{self.sid}: Episode {episode + 1}/{max_episodes} completed.")
        cleanup_session(self.sid, False)

    def end_episode(self, last_observation):
        for agent in self.session.agents.values():
            if agent.current_experience:
                agent.current_experience.next_state = last_observation
                agent.memory_buffer.append(agent.current_experience)
                agent.current_experience = None

        self.session.env.reset()
        self.session.agent_iter = iter(self.session.env.agent_iter())
        self.session.current_agent = self.session.agents[next(
            self.session.agent_iter)]

    def set_agent_in_turn_and_current_experience(self):
        agent_in_turn = self.session.current_agent
        obs, reward, terminated, truncated, _ = self.session.env.last()
        done = terminated or truncated

        obs = ml_service.preprocess_state(
            self.session.env_config.observation_space, obs)

        if agent_in_turn.current_experience is not None:
            agent_in_turn.current_experience.next_state = obs
            agent_in_turn.memory_buffer.append(
                agent_in_turn.current_experience)

        return agent_in_turn, done, reward, obs

    def augment_reward(self, reward):
        if reward > 0:
            return reward * REWARD_POS_FACTOR
        if reward == 0:
            return INACTIVITY_TO
        return reward * REWARD_NEG_FACTO
