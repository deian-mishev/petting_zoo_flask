from collections import defaultdict
from enum import Enum
import logging
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
TrainingMetrics = runpy.run_path(str(ROOT / "app/services/training_metrics.py"))["TrainingMetrics"]


def read_scalars(directory, name):
    events = EventAccumulator(str(directory)).Reload()
    return [(event.step, float(tf.make_ndarray(event.tensor_proto))) for event in events.Tensors(name)]


class TrainingMetricsTests(unittest.TestCase):
    def test_real_event_files_separate_runs_and_skip_missing_updates(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "TENSORBOARD_ENABLED": "true", "TENSORBOARD_LOG_DIR": directory,
        }):
            first = TrainingMetrics("boxing_v2")
            second = TrainingMetrics("boxing_v2")
            self.assertNotEqual(first.log_dir, second.log_dir)
            first.update("first_0", None)
            first.update("first_0", tf.constant(0.5))
            first.update("first_0", tf.constant(0.25))
            first.update("second_0", tf.constant(1.0))
            first.flush()
            self.assertEqual(read_scalars(first.log_dir, "train/first_0/loss"), [(1, 0.5), (2, 0.25)])
            first.close()
            second.close()
            self.assertEqual(read_scalars(first.log_dir, "train/second_0/loss"), [(1, 1.0)])

    def test_disabled_logging_creates_no_files(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "TENSORBOARD_ENABLED": "false", "TENSORBOARD_LOG_DIR": directory,
        }):
            metrics = TrainingMetrics("boxing_v2")
            metrics.update("first_0", tf.constant(0.5))
            metrics.flush()
            metrics.close()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_writer_failure_does_not_interrupt_training(self):
        with patch.dict(os.environ, {"TENSORBOARD_ENABLED": "true"}), \
                patch.object(tf.summary, "create_file_writer", side_effect=OSError("disk unavailable")), \
                self.assertLogs(level="WARNING"):
            metrics = TrainingMetrics("boxing_v2")
        metrics.update("first_0", tf.constant(0.5))
        metrics.close()

    def test_summary_failure_closes_writer(self):
        with patch.dict(os.environ, {"TENSORBOARD_ENABLED": "false"}):
            metrics = TrainingMetrics("boxing_v2")
        writer = Mock()
        writer.as_default.side_effect = OSError("disk full")
        metrics.writer = writer
        with self.assertLogs(level="WARNING"):
            metrics.scalar("loss", 1.0, 1)
        writer.close.assert_called_once()
        self.assertIsNone(metrics.writer)


class RunnerMetricsTests(unittest.TestCase):
    def setUp(self):
        class PlayerType(Enum):
            COMPUTER = "ai_regular"
            HUMAN = "human"
            ATARI_PRO = "atari_ai_pro"

        self.agent = SimpleNamespace(type=PlayerType.COMPUTER, total_reward=100,
                                     q_network=Mock(return_value=[0]), memory_buffer=[],
                                     current_experience=None)
        self.env = Mock()
        self.env.agent_iter.side_effect = lambda: iter(("first_0", "first_0", "first_0"))
        self.session = SimpleNamespace(env=self.env, agents={"first_0": self.agent},
                                       env_config=SimpleNamespace(name="boxing_v2", num_actions=1,
                                                                  observation_space=(1,)))
        self.service = Mock()
        self.service.preprocess_state.side_effect = lambda shape, observation: observation
        self.service.get_action.return_value = 0
        self.service.training_progress.return_value = {"epsilon": 0.5, "completed_episodes": 0, "environment_steps": 0}
        self.cleanup = Mock()

        def module(**attributes):
            result = ModuleType("stub")
            result.__dict__.update(attributes)
            return result

        stubs = {
            "app": module(app=SimpleNamespace(logger=logging.getLogger(__name__))),
            "flask_socketio": module(SocketIO=object),
            "app.config.session_state": module(SessionState=object),
            "app.services.rendering_service": module(render_frame=Mock()),
            "app.services.frame_delivery": module(
                FrameDelivery=runpy.run_path(str(ROOT / "app/services/frame_delivery.py"))["FrameDelivery"]),
            "app.services.ml_service": module(ml_service=self.service),
            "app.config.env_config": module(INACTIVITY_TO=-1, INPUT_TIMEOUT=0,
                                             REWARD_NEG_FACTO=3, REWARD_POS_FACTOR=5),
            "app.services.session": module(cleanup_session=self.cleanup),
            "app.config.player_state": module(Experience=SimpleNamespace, PlayerType=PlayerType),
            "app.services.training_metrics": module(TrainingMetrics=TrainingMetrics),
        }
        with patch.dict(sys.modules, stubs):
            runner_class = runpy.run_path(str(ROOT / "app/services/session_runner.py"))["SessionRunner"]
        self.runner = runner_class("test", self.session, None)

    def test_episode_rewards_reset_and_events_are_readable(self):
        self.env.last.side_effect = [
            ([0], 0, False, False, {}), ([1], 2, False, False, {}), ([2], -1, True, False, {}),
            ([0], 0, False, False, {}), ([1], 3, False, False, {}), ([2], -1, True, False, {}),
        ]
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "TENSORBOARD_ENABLED": "true", "TENSORBOARD_LOG_DIR": directory,
        }):
            self.runner._run_training(2)
            run_dir = next((Path(directory) / "boxing_v2").iterdir())
            self.assertEqual(read_scalars(run_dir, "episode/first_0/reward"), [(1, 1.0), (2, 2.0)])
            self.assertEqual(read_scalars(run_dir, "episode/first_0/shaped_reward"), [(1, 7.0), (2, 12.0)])
            self.assertEqual(read_scalars(run_dir, "progress/environment_steps"), [(1, 2.0), (2, 4.0)])
        self.cleanup.assert_called_once_with("test", False)
        metrics = self.service.train_model.call_args.kwargs["metrics"]
        self.assertIsNone(metrics.writer)

    def test_checkpoint_failure_stops_training_and_closes_session(self):
        self.env.last.side_effect = [
            ([0], 0, False, False, {}), ([1], 2, False, False, {}), ([2], -1, True, False, {}),
        ]
        self.service.train_model.side_effect = OSError("checkpoint upload failed")
        metrics = Mock(updates=defaultdict(int))
        with patch.dict(self.runner._run_training.__globals__, {"TrainingMetrics": Mock(return_value=metrics)}):
            with self.assertRaisesRegex(OSError, "checkpoint upload failed"):
                self.runner._run_training(3)
        self.service.train_model.assert_called_once()
        metrics.close.assert_called_once()
        self.cleanup.assert_called_once_with("test", False)

    def test_interrupted_episode_is_not_reported_as_completed(self):
        self.env.last.return_value = ([0], 0, False, False, {})
        self.env.step.side_effect = lambda _: self.runner._stop_event.set()
        metrics = Mock(updates=defaultdict(int))
        self.runner._run_training_episodes(2, metrics)
        metrics.scalar.assert_not_called()
        self.service.train_model.assert_not_called()

    def test_writer_closes_and_session_cleans_up_on_unexpected_error(self):
        metrics = Mock()
        self.runner._run_training_episodes = Mock(side_effect=RuntimeError("environment failed"))
        with patch.dict(self.runner._run_training.__globals__, {"TrainingMetrics": Mock(return_value=metrics)}):
            with self.assertRaises(RuntimeError):
                self.runner._run_training(1)
        metrics.close.assert_called_once()
        self.cleanup.assert_called_once_with("test", False)
