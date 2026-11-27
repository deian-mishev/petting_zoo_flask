"""Checkpoint orchestration tests without TensorFlow or application startup."""
import copy
from enum import Enum
from pathlib import Path
import runpy
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class PlayerType(Enum):
    COMPUTER = "computer"
    HUMAN = "human"
    ATARI_PRO = "pro"


class StaleCheckpointError(RuntimeError):
    pass


def module(**attributes):
    result = ModuleType("stub")
    result.__dict__.update(attributes)
    return result


def identity_function(function=None, **kwargs):
    return function if function is not None else lambda value: value


class TrainingStateServiceTests(unittest.TestCase):
    def setUp(self):
        self.save = Mock(return_value=SimpleNamespace(generation="new"))
        stubs = {
            "cv2": module(), "numpy": module(ndarray=object),
            "tensorflow": module(function=identity_function, random=Mock()),
            "app": module(app=SimpleNamespace(logger=Mock())),
            "app.config.env_config": module(SEED=0, EPSILON=1.0, EPSILON_OVERRIDE=None, E_MIN=.01, E_DECAY=.95,
                MINIBATCH_SIZE=128, TRANSITIONS_PER_UPDATE=64, MAX_UPDATES_PER_PLAYER=50),
            "app.config.ml_env_config": module(ATTARI_PRO_MODEL_HEADS={}),
            "app.config.session_state": module(SessionState=object),
            "app.config.player_state": module(PlayerState=object, PlayerType=PlayerType),
            "app.services.model_checkpoints": module(load_model_checkpoint=Mock(),
                save_model_checkpoint=self.save, load_training_state=Mock()),
            "app.services.blob_checkpoints": module(StaleCheckpointError=StaleCheckpointError),
        }
        with patch.dict(sys.modules, stubs):
            loaded = runpy.run_path(str(ROOT / "app/services/ml_service.py"))
        self.service = loaded["ml_service"]
        self.service.train_step = Mock()
        self.agent = SimpleNamespace(type=PlayerType.COMPUTER, lock=threading.Lock(),
            total_reward=0, model_path="model.keras", weights_path="target.h5",
            q_network=object(), target_q_network=object(), checkpoint_generation="old",
            training_state={"schema_version": 1, "environments": {}})
        self.session = SimpleNamespace(agents={"first": self.agent},
            env_config=SimpleNamespace(name="boxing_v2"))

    def test_shared_model_advances_once_and_retains_other_games(self):
        second = copy.copy(self.agent)
        self.session.agents["second"] = second
        self.agent.training_state["environments"]["pong"] = {
            "epsilon": .2, "completed_episodes": 5, "environment_steps": 100}
        self.agent.memory_buffer = [None] * 128
        second.memory_buffer = [None] * 128
        updates = self.service.match_updates(self.session, {id(self.agent): 7000, id(second): 193})
        self.assertEqual(updates, {"first": 50, "second": 3})
        self.service.train_model("test", self.session, episode_steps=25, updates_by_agent=updates)
        self.assertEqual(self.service.train_step.call_count, 53)
        self.save.assert_called_once()
        self.assertIs(self.agent.training_state, second.training_state)
        self.assertEqual(self.service.training_progress(self.agent, "boxing_v2"), {
            "epsilon": .95, "completed_episodes": 1, "environment_steps": 25})
        self.assertEqual(self.agent.training_state["environments"]["pong"]["epsilon"], .2)
        self.assertEqual(self.agent.checkpoint_generation, "new")

    def test_failed_publication_does_not_advance_state_or_generation(self):
        for error in (StaleCheckpointError("stale"), OSError("upload failed")):
            with self.subTest(error=error):
                before = copy.deepcopy(self.agent.training_state)
                self.save.side_effect = error
                with self.assertRaises(type(error)):
                    self.service.train_model("test", self.session, episode_steps=25)
                self.assertEqual(self.agent.training_state, before)
                self.assertEqual(self.agent.checkpoint_generation, "old")

    def test_legacy_defaults_and_restored_state_drive_exploration(self):
        self.assertEqual(self.service.training_progress(self.agent, "boxing_v2")["epsilon"], 1.)
        self.agent.training_state["environments"]["boxing_v2"] = {
            "epsilon": .4, "completed_episodes": 10, "environment_steps": 200}
        self.service.train_model("test", self.session, episode_steps=25)
        self.assertEqual(self.service.training_progress(self.agent, "boxing_v2"), {
            "epsilon": .38, "completed_episodes": 11, "environment_steps": 225})

    def test_loading_shares_networks_and_does_not_take_optimizer_step(self):
        from contextlib import contextmanager
        @contextmanager
        def checkpoint(*args):
            yield ("model.keras", "target.h5"), "loaded"
        state = {"schema_version": 1, "environments": {"boxing_v2": {
            "epsilon": .2, "completed_episodes": 3, "environment_steps": 45}}}
        optimizer = Mock(built=True)
        network = Mock(optimizer=optimizer)
        target = Mock()
        self.service._load_networks = Mock(return_value=(network, target))
        self.session.agents["second"] = copy.copy(self.agent)
        with patch.dict(self.service.load_model.__globals__, {
            "load_model_checkpoint": checkpoint, "load_training_state": Mock(return_value=state)}):
            self.service.load_model("test", self.session, (52, 40, 1), 18)
        optimizer.apply_gradients.assert_not_called()
        optimizer.build.assert_not_called()
        self.service._load_networks.assert_called_once()
        self.assertIs(self.agent.training_state, self.session.agents["second"].training_state)
        self.assertIs(self.agent.q_network, network)
        self.assertEqual(self.agent.checkpoint_generation, "loaded")

    def test_override_is_applied_on_load_and_saved_with_normal_decay(self):
        from contextlib import contextmanager

        @contextmanager
        def checkpoint(*args):
            yield ("model.keras", "target.h5"), "loaded"

        state = {"schema_version": 1, "environments": {
            "boxing_v2": {"epsilon": .01, "completed_episodes": 20, "environment_steps": 40},
            "pong": {"epsilon": .2, "completed_episodes": 5, "environment_steps": 10},
        }}
        optimizer = Mock(built=True)
        network = Mock(optimizer=optimizer)
        self.service._load_networks = Mock(return_value=(network, Mock()))
        self.session.agents["second"] = copy.copy(self.agent)
        globals_patch = {"load_model_checkpoint": checkpoint,
                         "load_training_state": Mock(return_value=copy.deepcopy(state)),
                         "EPSILON_OVERRIDE": .3}
        with patch.dict(self.service.load_model.__globals__, globals_patch):
            self.service.load_model("test", self.session, (52, 40, 1), 18)
            self.assertEqual(self.service.training_progress(self.agent, "boxing_v2")["epsilon"], .3)
            self.assertIs(self.agent.training_state, self.session.agents["second"].training_state)
            self.service.train_model("test", self.session, episode_steps=25)
            saved = copy.deepcopy(self.save.call_args.kwargs["training_state"])

        self.assertAlmostEqual(saved["environments"]["boxing_v2"]["epsilon"], .285)
        self.assertEqual(saved["environments"]["pong"], state["environments"]["pong"])
        with patch.dict(self.service.load_model.__globals__, {
            "load_model_checkpoint": checkpoint, "load_training_state": Mock(return_value=saved),
            "EPSILON_OVERRIDE": None,
        }):
            self.service.load_model("test", self.session, (52, 40, 1), 18)
            self.assertAlmostEqual(self.service.training_progress(self.agent, "boxing_v2")["epsilon"], .285)


class RunnerCheckpointFailureTests(unittest.TestCase):
    def test_failed_checkpoint_stops_before_next_episode_and_cleans_up(self):
        from collections import defaultdict
        agent = SimpleNamespace(type=PlayerType.COMPUTER, total_reward=0,
            q_network=Mock(return_value=[0]), memory_buffer=[], current_experience=None)
        env = Mock()
        env.agent_iter.side_effect = lambda: iter(("first", "first"))
        env.last.side_effect = [([0], 0, False, False, {}), ([1], -1, True, False, {})]
        session = SimpleNamespace(env=env, agents={"first": agent},
            env_config=SimpleNamespace(name="boxing_v2", num_actions=1, observation_space=(1,)))
        service = Mock()
        service.get_action.return_value = 0
        service.match_updates.return_value = {"first": 0}
        service.training_progress.return_value = {"epsilon": .2}
        service.train_model.side_effect = OSError("upload failed")
        cleanup = Mock()
        metrics = Mock(updates=defaultdict(int))
        stubs = {
            "tensorflow": Mock(), "flask_socketio": module(SocketIO=object),
            "app": module(app=SimpleNamespace(logger=Mock())),
            "app.config.session_state": module(SessionState=object),
            "app.services.rendering_service": module(render_frame=Mock()),
            "app.services.frame_delivery": module(FrameDelivery=Mock()),
            "app.services.ml_service": module(ml_service=service),
            "app.config.env_config": module(INACTIVITY_TO=-1, INPUT_TIMEOUT=0,
                REWARD_NEG_FACTO=3, REWARD_POS_FACTOR=5),
            "app.services.session": module(cleanup_session=cleanup),
            "app.services.training_metrics": module(TrainingMetrics=Mock(return_value=metrics)),
            "app.config.player_state": module(Experience=SimpleNamespace, PlayerType=PlayerType),
        }
        with patch.dict(sys.modules, stubs):
            runner = runpy.run_path(str(ROOT / "app/services/session_runner.py"))["SessionRunner"]("test", session, None)
        with self.assertRaisesRegex(OSError, "upload failed"):
            runner._run_training(3)
        service.train_model.assert_called_once_with("test", session, metrics=metrics, episode_steps=1,
                                                    updates_by_agent={"first": 0})
        self.assertEqual(service.get_action.call_args.args[1], .2)
        metrics.close.assert_called_once()
        cleanup.assert_called_once_with("test", False)
