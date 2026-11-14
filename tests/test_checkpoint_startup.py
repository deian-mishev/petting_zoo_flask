from enum import Enum
from pathlib import Path
import runpy
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from flask import Flask, request
from flask_socketio import SocketIO


ROOT = Path(__file__).resolve().parents[1]


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


class PlayerType(str, Enum):
    HUMAN = "human"


class CheckpointStartupTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.secret_key = "test"
        self.socketio = SocketIO(self.app, async_mode="threading")
        self.sessions = {}
        self.environment = Mock()
        self.environment.agent_iter.return_value = iter(["first_0"])
        self.config = SimpleNamespace(
            env=Mock(return_value=self.environment), num_actions=18, observation_space=(52, 40, 1),
        )
        self.service = Mock()
        self.runner = Mock()

        def new_session(**values):
            return SimpleNamespace(**values, agents={"first_0": SimpleNamespace()})

        def identity(handler):
            return handler

        def roles_required(*args, **kwargs):
            return identity

        stubs = {
            "app": module("app", app=self.app, socketio=self.socketio,
                          client_sessions=self.sessions, client_sessions_lock=threading.Lock()),
            "app.services.ml_service": module("ml_service", ml_service=self.service),
            "app.services.session_runner": module("session_runner", SessionRunner=self.runner),
            "app.config.session_state": module("session_state", SessionState=new_session),
            "app.config.player_state": module("player_state", PlayerType=PlayerType),
            "app.config.ml_env_config": module("ml_env_config", EnvironmentConfig=SimpleNamespace,
                                               ENVIRONMENTS={"Boxing": self.config}),
            "app.config.env_config": module("env_config"),
            "app.config.oauth2_config": module("oauth2_config", login_required=identity,
                                               roles_required=roles_required, token_required_api=roles_required),
            "app.validation": module("validation", validate_env=lambda _: True,
                                     validate_players=lambda players: (True, players),
                                     validate_env_players_comb=lambda *_: (True, {"first_0": "ai_regular"})),
            "app.services.session": module(
                "session", cleanup_session=Mock(), populate_session_agents=Mock(),
                get_available_environments_and_nemesis=lambda: ([], ["ai_regular"]),
            ),
        }
        with patch.dict(sys.modules, stubs):
            runpy.run_path(str(ROOT / "app/routes/api.py"))
            runpy.run_path(str(ROOT / "app/routes/admin.py"))

        @self.app.before_request
        def set_test_user():
            request.user = {"sub": "admin", "email": "admin@example.invalid"}

        self.client = self.app.test_client()
        with self.client.session_transaction() as session:
            session["user"] = {"name": "test player"}
            session["roles"] = ["User"]

    def connect(self):
        return self.socketio.test_client(
            self.app, flask_test_client=self.client, query_string="env=Boxing&players={}",
            auth={"client": "test"},
        )

    def test_anonymous_favicon_does_not_start_login_or_change_session(self):
        anonymous = self.app.test_client()
        response = anonymous.get("/favicon.ico")
        self.assertEqual(response.status_code, 204)
        self.assertNotIn("Location", response.headers)
        self.assertNotIn("Set-Cookie", response.headers)

    def test_model_failure_rejects_connection_and_closes_environment(self):
        self.service.load_model.side_effect = OSError("storage unavailable")
        client = self.connect()
        self.assertFalse(client.is_connected())
        self.environment.close.assert_called_once()
        self.service.load_model.assert_called_once()
        self.runner.assert_not_called()
        self.assertEqual(self.sessions, {})

    def test_environment_close_failure_still_rejects_connection(self):
        self.service.load_model.side_effect = OSError("storage unavailable")
        self.environment.close.side_effect = OSError("close failed")
        self.assertFalse(self.connect().is_connected())
        self.assertEqual(self.sessions, {})

    def test_successful_load_accepts_auth_and_starts_game(self):
        client = self.connect()
        self.assertTrue(client.is_connected())
        self.service.load_model.assert_called_once()
        self.runner.return_value.start.assert_called_once()
        self.environment.close.assert_not_called()
        self.assertEqual(len(self.sessions), 1)
        client.disconnect()

    def test_training_storage_failure_returns_503_and_does_not_start_runner(self):
        self.service.load_model.side_effect = OSError("private storage details")
        response = self.client.post("/train", json={"env": "Boxing", "players": {"first_0": "ai_regular"}})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private storage details", response.get_json()["error"])
        self.environment.close.assert_called_once()
        self.runner.assert_not_called()
        self.assertEqual(self.sessions, {})


if __name__ == "__main__":
    unittest.main()
