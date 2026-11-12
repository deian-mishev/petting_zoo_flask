import os
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

from pymongo import ASCENDING
from pymongo.errors import ServerSelectionTimeoutError


CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "app" / "config" / "persistance_config.py"
)


class PersistenceConfigTests(unittest.TestCase):
    def load_config(self):
        return runpy.run_path(str(CONFIG_PATH))

    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        dotenv = patch("dotenv.load_dotenv")
        dotenv.start()
        self.addCleanup(dotenv.stop)
        client = patch("pymongo.MongoClient")
        self.client = client.start()
        self.addCleanup(client.stop)

    def test_disabled_never_creates_client_and_rejects_database_access(self):
        for uri in (None, "not-a-valid-mongodb-uri"):
            with self.subTest(uri=uri):
                os.environ["MONGO_ENABLED"] = " false "
                if uri is not None:
                    os.environ["MONGO_DB_URI"] = uri
                with self.assertLogs("app", level="INFO") as logs:
                    config = self.load_config()
                self.assertIn("disabled", logs.output[0])
                self.client.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, "disabled"):
                    config["get_experiences_collection"]()

    def test_enabled_and_legacy_default_create_existing_index(self):
        for enabled in (None, "TRUE"):
            with self.subTest(enabled=enabled):
                self.client.reset_mock()
                os.environ["MONGO_DB_URI"] = "mongodb://example.invalid/rl_db"
                if enabled is not None:
                    os.environ["MONGO_ENABLED"] = enabled
                config = self.load_config()
                self.client.assert_called_once_with(
                    "mongodb://example.invalid/rl_db"
                )
                collection = self.client.return_value["rl_db"]["experiences"]
                collection.create_index.assert_called_once_with(
                    [("env_name", ASCENDING), ("timestamp", ASCENDING)]
                )
                self.assertIs(config["get_experiences_collection"](), collection)

    def test_invalid_flag_and_missing_enabled_uri_fail_before_connection(self):
        cases = [
            {"MONGO_ENABLED": "maybe"},
            {"MONGO_ENABLED": ""},
            {"MONGO_ENABLED": "true"},
            {"MONGO_ENABLED": "true", "MONGO_DB_URI": " "},
        ]
        for environment in cases:
            with self.subTest(environment=environment):
                with patch.dict(os.environ, environment, clear=True):
                    with self.assertRaises(ValueError):
                        self.load_config()
                self.client.assert_not_called()

    def test_enabled_connection_failure_is_not_silently_disabled(self):
        os.environ["MONGO_ENABLED"] = "true"
        os.environ["MONGO_DB_URI"] = "mongodb://example.invalid/rl_db"
        collection = self.client.return_value["rl_db"]["experiences"]
        collection.create_index.side_effect = ServerSelectionTimeoutError(
            "Database unavailable"
        )
        with self.assertRaises(ServerSelectionTimeoutError):
            self.load_config()


if __name__ == "__main__":
    unittest.main()
