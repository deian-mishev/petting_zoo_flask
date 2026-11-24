from enum import Enum
from pathlib import Path
import runpy
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def module(**values):
    result = ModuleType("stub")
    result.__dict__.update(values)
    return result


class PlayerType(str, Enum):
    COMPUTER = "ai_regular"
    ATARI_PRO = "atari_ai_pro"
    HUMAN = "human"


class SessionAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.sessions = {}
        stubs = {
            "app": module(app=SimpleNamespace(logger=Mock()), client_sessions=self.sessions,
                          client_sessions_lock=threading.Lock()),
            "app.config.session_state": module(SessionState=object),
            "app.config.ml_env_config": module(ENVIRONMENTS={
                "Boxing": SimpleNamespace(name="boxing", agents=[]),
                "Pong": SimpleNamespace(name="pong", agents=[]),
            }, EnvironmentConfig=object),
            "app.config.env_config": module(ATTARI_PRO_LOCK=Mock(), ATTARI_PRO_MODEL="pro.keras",
                                            ATTARI_PRO_WEIGHTS_PATH="pro.weights.h5"),
            "app.config.player_state": module(PlayerState=object, PlayerType=PlayerType),
        }
        path = Path(__file__).resolve().parents[1] / "app/services/session.py"
        with patch.dict(sys.modules, stubs):
            self.service = runpy.run_path(str(path))

    def session(self, player_type, game="boxing"):
        return SimpleNamespace(agents={"player": SimpleNamespace(type=player_type)},
                               env_config=SimpleNamespace(name=game), env=Mock(), runner=Mock())

    def test_human_session_reserves_shared_model_through_disconnect_update(self):
        session = self.session(PlayerType.HUMAN)
        self.assertTrue(self.service["reserve_session"]("first", session))
        self.assertNotIn(PlayerType.ATARI_PRO.value,
                         self.service["get_available_environments_and_nemesis"]()[1])
        self.assertNotIn(PlayerType.HUMAN.value,
                         self.service["get_available_environments_and_nemesis"]()[1])

        def while_updating():
            self.assertFalse(self.service["reserve_session"]("second", self.session(PlayerType.ATARI_PRO, "pong")))
            self.assertNotIn(PlayerType.ATARI_PRO.value,
                             self.service["get_available_environments_and_nemesis"]()[1])

        session.runner.update_after_match.side_effect = while_updating
        self.service["cleanup_session"]("first")
        self.assertTrue(self.service["reserve_session"]("second", self.session(PlayerType.ATARI_PRO, "pong")))

    def test_pro_reservation_blocks_human_and_pro_but_allows_regular(self):
        self.assertTrue(self.service["reserve_session"]("first", self.session(PlayerType.ATARI_PRO)))
        self.assertFalse(self.service["reserve_session"]("second", self.session(PlayerType.HUMAN, "pong")))
        self.assertFalse(self.service["reserve_session"]("third", self.session(PlayerType.ATARI_PRO, "pong")))
        self.assertTrue(self.service["reserve_session"]("fourth", self.session(PlayerType.COMPUTER, "pong")))

    def test_regular_model_stays_reserved_during_update(self):
        session = self.session(PlayerType.COMPUTER, "boxing")
        self.assertTrue(self.service["reserve_session"]("first", session))
        self.assertFalse(self.service["reserve_session"]("second", self.session(PlayerType.COMPUTER, "boxing")))
        self.assertTrue(self.service["reserve_session"]("third", self.session(PlayerType.COMPUTER, "pong")))
        self.assertNotIn("Boxing", [env["name"] for env in
                                    self.service["get_available_environments_and_nemesis"]()[0]])

        def while_updating():
            self.assertFalse(self.service["reserve_session"]("second", self.session(PlayerType.COMPUTER, "boxing")))

        session.runner.update_after_match.side_effect = while_updating
        self.service["cleanup_session"]("first")
        self.assertTrue(self.service["reserve_session"]("second", self.session(PlayerType.COMPUTER, "boxing")))


if __name__ == "__main__":
    unittest.main()
