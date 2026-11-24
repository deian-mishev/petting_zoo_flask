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
resolve_human_action = runpy.run_path(str(ROOT / 'app/services/human_input.py'))['resolve_human_action']


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


class PlayerType(str, Enum):
    COMPUTER = "ai_regular"
    ATARI_PRO = "atari_ai_pro"
    HUMAN = "human"


class CheckpointStartupTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.secret_key = "test"
        self.socketio = SocketIO(self.app, async_mode="threading")
        self.sessions = {}
        self.environment = Mock()
        self.environment.agent_iter.side_effect = lambda: iter(["first_0"])
        self.config = SimpleNamespace(
            env=Mock(return_value=self.environment), num_actions=18, observation_space=(52, 40, 1),
            name="boxing", agents=["first_0"], lock=threading.Lock(),
            model_path="boxing.keras", weights_path="boxing.weights.h5",
        )
        self.players = {"first_0": "ai_regular"}
        self.service = Mock()
        self.runner = Mock()

        def new_session(**values):
            return SimpleNamespace(**values, agents={}, runner=None)

        def identity(handler):
            return handler

        def roles_required(*args, **kwargs):
            return identity

        stubs = {
            "app": module("app", app=self.app, socketio=self.socketio,
                          client_sessions=self.sessions, client_sessions_lock=threading.Lock()),
            "app.services.ml_service": module("ml_service", ml_service=self.service),
            "app.services.human_input": module("human_input", resolve_human_action=resolve_human_action),
            "app.services.session_runner": module("session_runner", SessionRunner=self.runner),
            "app.config.session_state": module("session_state", SessionState=new_session),
            "app.config.player_state": module("player_state", PlayerType=PlayerType,
                                              PlayerState=SimpleNamespace),
            "app.config.ml_env_config": module("ml_env_config", EnvironmentConfig=SimpleNamespace,
                                               ENVIRONMENTS={"Boxing": self.config}),
            "app.config.env_config": module("env_config", ATTARI_PRO_LOCK=threading.Lock(),
                                             ATTARI_PRO_MODEL="pro.keras",
                                             ATTARI_PRO_WEIGHTS_PATH="pro.weights.h5"),
            "app.config.oauth2_config": module("oauth2_config", login_required=identity,
                                               roles_required=roles_required, token_required_api=roles_required),
            "app.validation": module("validation", validate_env=lambda _: True,
                                     validate_players=lambda players: (True, players),
                                     validate_env_players_comb=lambda *_: (True, self.players)),
        }
        with patch.dict(sys.modules, stubs):
            # Exercise the production reservation/cleanup code. Stubbing these
            # hides whether a refresh releases a model before its update ends.
            lifecycle = runpy.run_path(str(ROOT / "app/services/session.py"))
            self.lifecycle = module("session", **{
                name: lifecycle[name] for name in (
                    "cleanup_session", "populate_session_agents", "reserve_session",
                    "get_available_environments_and_nemesis",
                )
            })
            sys.modules["app.services.session"] = self.lifecycle
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
        self.assertFalse(self.runner.call_args.kwargs["frame_ack"])
        client.disconnect()

    def test_socket_input_combines_buttons_and_releases_them_independently(self):
        client = self.connect()
        session = next(iter(self.sessions.values()))
        session.lock = threading.Lock()
        session.next_human_action = 0
        self.environment.unwrapped.action_mapping = tuple(range(18))
        for keys, expected in [(['s', '8', '6'], 14), (['6', 's', '8'], 14),
                                (['6', 's'], 11), (['6'], 3), ([], 0), (None, 0)]:
            client.emit('input', keys)
            self.assertEqual(session.next_human_action, expected)
        client.disconnect()

    def test_current_browser_negotiates_frame_acknowledgements(self):
        client = self.socketio.test_client(
            self.app, flask_test_client=self.client,
            query_string="env=Boxing&players={}&frame_ack=1",
        )
        self.assertTrue(client.is_connected())
        self.assertTrue(self.runner.call_args.kwargs["frame_ack"])
        client.disconnect()

    def test_same_user_refresh_keeps_models_reserved_until_update_finishes(self):
        self.players = {"first_0": "human"}
        old_client = self.connect()
        old_sid = next(iter(self.sessions))
        self.addCleanup(lambda: old_client.disconnect() if old_client.is_connected() else None)

        def while_updating():
            self.assertIn(old_sid, self.sessions)
            availability = self.client.get("/preconnect").get_json()
            self.assertEqual(availability["environments"], [])
            self.assertEqual(availability["ai_players"], ["ai_regular"])
            self.assertFalse(self.connect().is_connected())
            self.assertEqual(self.service.load_model.call_count, 1)
            self.assertEqual(self.runner.call_count, 1)

        self.runner.return_value.update_after_match.side_effect = while_updating
        old_client.disconnect()
        self.runner.return_value.stop.assert_called_once()
        self.runner.return_value.update_after_match.assert_called_once()
        self.assertEqual(self.sessions, {})
        availability = self.client.get("/preconnect").get_json()
        self.assertEqual([env["name"] for env in availability["environments"]], ["Boxing"])
        self.assertIn("human", availability["ai_players"])
        self.assertIn("atari_ai_pro", availability["ai_players"])

        self.runner.return_value.update_after_match.side_effect = None
        refreshed_client = self.connect()  # Reuse the same signed-in browser cookie.
        self.assertTrue(refreshed_client.is_connected())
        self.assertNotIn(old_sid, self.sessions)
        self.assertEqual(len(self.sessions), 1)
        refreshed_client.disconnect()

    def test_late_old_transport_disconnect_does_not_remove_same_users_new_game(self):
        old_client = self.connect()
        old_eio_sid = old_client.eio_sid
        old_sid = next(iter(self.sessions))
        old_client.disconnect()

        refreshed_client = self.connect()
        self.addCleanup(lambda: refreshed_client.disconnect() if refreshed_client.is_connected() else None)
        new_sid = next(iter(self.sessions))
        self.assertNotEqual(old_sid, new_sid)
        self.socketio.server._handle_eio_disconnect(old_eio_sid, "transport close")
        self.assertTrue(refreshed_client.is_connected())
        self.assertEqual(list(self.sessions), [new_sid])
        self.assertEqual(self.runner.return_value.update_after_match.call_count, 1)

    def test_transport_close_without_socket_disconnect_packet_cleans_up_game(self):
        client = self.connect()
        sid = next(iter(self.sessions))
        self.socketio.server._handle_eio_disconnect(client.eio_sid, "transport close")
        self.assertFalse(self.socketio.server.manager.is_connected(sid, "/"))
        self.assertEqual(self.sessions, {})
        self.runner.return_value.stop.assert_called_once()
        self.runner.return_value.update_after_match.assert_called_once()
        self.environment.close.assert_called_once()

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
