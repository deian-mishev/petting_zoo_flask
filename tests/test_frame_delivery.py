"""Delivery pacing must never wait for the browser or stall game controls."""

from enum import Enum
from itertools import repeat
import logging
import os
from pathlib import Path
import runpy
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
delivery_module = runpy.run_path(str(ROOT / "app/services/frame_delivery.py"))
FrameDelivery = delivery_module["FrameDelivery"]


class FrameDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.settings = patch.dict(os.environ, {"STREAM_FPS": "30", "STREAM_MAX_IN_FLIGHT": "3"})
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.now = 0.0
        self.emit = Mock()
        self.render = Mock(side_effect=lambda: self.now)
        self.delivery = FrameDelivery(self.emit, clock=lambda: self.now)

    def send(self, now):
        self.now = now
        return self.delivery.send_if_due(self.render)

    def test_full_window_skips_encoding_and_resumes_with_current_state(self):
        for now in [0, .035, .07]:
            self.assertTrue(self.send(now))
        for now in [.105, .5, 10, 100]:
            self.assertFalse(self.send(now))
        self.assertEqual(self.render.call_count, 3)
        ack = self.emit.call_args_list[0].kwargs["callback"]
        ack()
        ack()  # Duplicate acknowledgements must not grant extra credit.
        self.assertTrue(self.send(101))
        self.assertEqual(self.emit.call_args.args[0], 101)
        self.assertFalse(self.send(102))

    def test_pipelining_sustains_30fps_with_80ms_round_trip(self):
        pending = []
        sent_at = []

        def emit(frame, callback):
            pending.append((self.now + .080, callback))
            sent_at.append(self.now)

        self.delivery.emit = emit
        for milliseconds in range(0, 1000, 7):
            self.now = milliseconds / 1000
            for due, ack in list(pending):
                if due <= self.now:
                    ack()
                    pending.remove((due, ack))
            self.delivery.send_if_due(self.render)
            self.assertLessEqual(len(pending), 3)
        self.assertEqual(len(sent_at), 30)
        gaps = [b - a for a, b in zip(sent_at, sent_at[1:])]
        self.assertGreaterEqual(min(gaps), .0279)
        self.assertLessEqual(max(gaps), .0351)

    def test_long_pause_does_not_create_catch_up_burst(self):
        self.send(0)
        self.emit.call_args.kwargs["callback"]()
        self.assertTrue(self.send(10))
        self.emit.call_args.kwargs["callback"]()
        self.assertFalse(self.send(10.001))

    def test_render_and_emit_failures_release_credit(self):
        with patch.dict(os.environ, {"STREAM_MAX_IN_FLIGHT": "1"}):
            self.delivery = FrameDelivery(self.emit, clock=lambda: self.now)
        self.render.side_effect = ValueError("encoding failed")
        with self.assertRaises(ValueError):
            self.send(0)
        self.render.side_effect = None
        self.emit.side_effect = OSError("send failed")
        with self.assertRaises(OSError):
            self.send(1)
        self.emit.side_effect = None
        self.assertTrue(self.send(2))

    def test_legacy_client_without_acknowledgements_does_not_freeze(self):
        self.delivery = FrameDelivery(self.emit, require_ack=False, clock=lambda: self.now)
        for index in range(10):
            self.assertTrue(self.send(index * .04))
        self.assertEqual(self.emit.call_count, 10)
        self.assertEqual(self.emit.call_args.kwargs, {})

    def test_settings_are_bounded_and_invalid_values_use_defaults(self):
        read = delivery_module["bounded_setting"]
        for value, expected in [("invalid", 30), ("0", 1), ("100000", 60), ("60", 60)]:
            with patch.dict(os.environ, {"STREAM_FPS": value}):
                self.assertEqual(read("STREAM_FPS", 30, 1, 60), expected)


def module(**attributes):
    result = ModuleType("stub")
    result.__dict__.update(attributes)
    return result


class GameLoopDeliveryTests(unittest.TestCase):
    def test_game_steps_and_human_actions_continue_when_viewer_stops_acknowledging(self):
        class PlayerType(Enum):
            HUMAN = "human"
            ATARI_PRO = "atari_ai_pro"

        agent = SimpleNamespace(type=PlayerType.HUMAN, total_reward=0, current_experience=None)
        session = SimpleNamespace(lock=threading.Lock(), next_human_action=0,
                                  env=Mock(), agents={"human": agent},
                                  agent_iter=repeat("human"), current_agent=agent)
        now = [0.0]
        render = Mock(return_value=b"jpeg")
        stubs = {
            "app": module(app=SimpleNamespace(logger=logging.getLogger(__name__))),
            "tensorflow": module(),
            "flask_socketio": module(SocketIO=object),
            "app.config.session_state": module(SessionState=object),
            "app.services.rendering_service": module(render_frame=render),
            "app.services.frame_delivery": module(FrameDelivery=FrameDelivery),
            "app.services.ml_service": module(ml_service=Mock()),
            "app.config.env_config": module(INACTIVITY_TO=-1, INPUT_TIMEOUT=.007,
                                             REWARD_NEG_FACTO=3, REWARD_POS_FACTOR=5),
            "app.services.session": module(cleanup_session=Mock()),
            "app.config.player_state": module(Experience=SimpleNamespace, PlayerType=PlayerType),
            "app.services.training_metrics": module(TrainingMetrics=Mock()),
        }
        with patch.dict(sys.modules, stubs):
            runner_class = runpy.run_path(str(ROOT / "app/services/session_runner.py"))["SessionRunner"]
        socket = Mock()
        runner = runner_class("sid", session, socket, frame_ack=True)
        runner._frames.clock = lambda: now[0]
        runner._frames.max_in_flight = 3
        runner.set_agent_in_turn_and_current_experience = Mock(return_value=(agent, False, 0, []))
        actions = []

        def step(action):
            actions.append(action)
            if len(actions) == 50:
                session.next_human_action = 7
            if len(actions) == 100:
                runner._stop_event.set()

        session.env.step.side_effect = step
        with patch("time.sleep", side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)):
            runner._run_loop()
        self.assertEqual(actions, [0] * 50 + [7] * 50)
        self.assertEqual(render.call_count, 3)
        self.assertEqual(socket.emit.call_count, 3)
        self.assertEqual(socket.emit.call_args.kwargs["room"], "sid")


class RenderingTests(unittest.TestCase):
    def load_renderer(self):
        with patch.dict(sys.modules, {
            "app.config.session_state": module(SessionState=object),
            "app.services.frame_delivery": module(**delivery_module),
        }):
            return runpy.run_path(str(ROOT / "app/services/rendering_service.py"))["render_frame"]

    def test_real_jpeg_encoding_preserves_dimensions_and_configures_quality(self):
        import cv2
        import numpy as np

        frame = np.random.default_rng(1).integers(0, 256, size=(210, 160, 3), dtype=np.uint8)
        session = SimpleNamespace(env=SimpleNamespace(render=lambda: frame))
        images = []
        for quality in [75, 60]:
            with patch.dict(os.environ, {"STREAM_JPEG_QUALITY": str(quality)}):
                images.append(self.load_renderer()(session))
        for encoded in images:
            decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape, frame.shape)
        self.assertLess(len(images[1]), len(images[0]))

    def test_encoding_failure_is_explicit(self):
        import cv2
        import numpy as np

        renderer = self.load_renderer()
        with patch.object(cv2, "imencode", return_value=(False, None)):
            with self.assertRaisesRegex(RuntimeError, "encode gameplay frame"):
                frame = np.zeros((8, 8, 3), dtype=np.uint8)
                renderer(SimpleNamespace(env=SimpleNamespace(render=lambda: frame)))

    def test_rgb_colors_survive_jpeg_roundtrip(self):
        import cv2
        import numpy as np

        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        frame[:, :32] = (255, 0, 0)  # RGB red
        frame[:, 32:] = (0, 0, 255)  # RGB blue
        renderer = self.load_renderer()
        jpeg = renderer(SimpleNamespace(env=SimpleNamespace(render=lambda: frame)))
        decoded_bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(int(decoded_bgr[16, 8, 2]), 240)
        self.assertLess(int(decoded_bgr[16, 8, 0]), 15)
        self.assertGreater(int(decoded_bgr[16, 56, 0]), 240)
        self.assertLess(int(decoded_bgr[16, 56, 2]), 15)


if __name__ == "__main__":
    unittest.main()
