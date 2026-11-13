import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf
from test_blob_checkpoints import MemoryBlobs


with patch.dict(os.environ, {"ENV": "DEV"}):
    from app.config.ml_env_config import ATTARI_PRO_MODEL_HEADS
    from app.config.player_state import PlayerState, PlayerType
    from app.services.ml_service import ml_service
    from app.services.model_checkpoints import locked_model_files
    from app.services.blob_checkpoints import (
        BlobCheckpointStore, CheckpointConflict, CheckpointNotFound, _prefix, _validate_with_keras,
    )


class MLCheckpointIntegrationTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"MODEL_CHECKPOINT_BACKEND": "local", "CONTAINER_APP_NAME": ""})
        environment.start()
        self.addCleanup(environment.stop)

    def new_network(self, player_type):
        if player_type == PlayerType.ATARI_PRO:
            return ml_service.build_multi_head_q_network((52, 40, 1), ATTARI_PRO_MODEL_HEADS)
        return ml_service.build_q_network((52, 40, 1), 18)

    def new_session(self, player_type, model_path, weights_path, model=None, target=None):
        agent = PlayerState(
            type=player_type,
            model_path=str(model_path),
            weights_path=str(weights_path),
            q_network=model,
            target_q_network=target,
            optimizer=model.optimizer if model is not None else None,
            lock=threading.Lock(),
        )
        config = SimpleNamespace(
            name="boxing_v2",
            observation_space=(52, 40, 1),
            num_actions=18,
            epsilon=1.0,
            lock=threading.Lock(),
        )
        return SimpleNamespace(agents={"first_0": agent}, env_config=config), agent

    def test_real_keras_models_are_published_and_loaded_as_pairs(self):
        for player_type in (PlayerType.COMPUTER, PlayerType.ATARI_PRO):
            with self.subTest(player_type=player_type), tempfile.TemporaryDirectory() as directory:
                model_path = Path(directory) / "legacy.keras"
                weights_path = Path(directory) / "legacy.weights.h5"
                model = self.new_network(player_type)
                target = self.new_network(player_type)
                target.set_weights(model.get_weights())
                model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
                session, _ = self.new_session(player_type, model_path, weights_path, model, target)

                ml_service.train_model("test", session)
                with locked_model_files(str(model_path), str(weights_path)) as saved_files:
                    self.assertIsNotNone(saved_files)
                    self.assertEqual(Path(saved_files[0]).parent.name, "v1")
                    _validate_with_keras(*saved_files)
                self.assertFalse(model_path.exists())
                self.assertFalse(weights_path.exists())

                loaded_session, loaded_agent = self.new_session(player_type, model_path, weights_path)
                ml_service.load_model("test", loaded_session, (52, 40, 1), 18)
                for expected, actual in zip(model.get_weights(), loaded_agent.q_network.get_weights()):
                    np.testing.assert_allclose(actual, expected)
                for expected, actual in zip(target.get_weights(), loaded_agent.target_q_network.get_weights()):
                    np.testing.assert_allclose(actual, expected)

    def test_real_legacy_pair_is_loaded_without_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "legacy.keras"
            weights_path = Path(directory) / "legacy.weights.h5"
            model = self.new_network(PlayerType.COMPUTER)
            target = self.new_network(PlayerType.COMPUTER)
            target.set_weights(model.get_weights())
            model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
            model.save(str(model_path))
            target.save_weights(str(weights_path))

            session, agent = self.new_session(PlayerType.COMPUTER, model_path, weights_path)
            ml_service.load_model("test", session, (52, 40, 1), 18)
            self.assertFalse((Path(f"{model_path}.checkpoints") / "current.json").exists())
            for expected, actual in zip(model.get_weights(), agent.q_network.get_weights()):
                np.testing.assert_allclose(actual, expected)
            for expected, actual in zip(target.get_weights(), agent.target_q_network.get_weights()):
                np.testing.assert_allclose(actual, expected)

    def test_blob_migration_and_stale_real_model_training(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "legacy.keras"
            weights_path = Path(directory) / "legacy.weights.h5"
            model = self.new_network(PlayerType.COMPUTER)
            target = self.new_network(PlayerType.COMPUTER)
            target.set_weights(model.get_weights())
            model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
            model.save(str(model_path))
            target.save_weights(str(weights_path))
            store = BlobCheckpointStore(MemoryBlobs(CheckpointNotFound, CheckpointConflict))
            store.bootstrap(str(model_path), str(weights_path), _validate_with_keras)

            with patch.dict(os.environ, {"MODEL_CHECKPOINT_BACKEND": "azure_blob", "CONTAINER_APP_NAME": "web"}), patch(
                "app.services.blob_checkpoints.blob_store_from_environment", return_value=store
            ):
                first, first_agent = self.new_session(PlayerType.COMPUTER, model_path, weights_path)
                second, second_agent = self.new_session(PlayerType.COMPUTER, model_path, weights_path)
                ml_service.load_model("first", first, (52, 40, 1), 18)
                ml_service.load_model("second", second, (52, 40, 1), 18)
                self.assertEqual(first_agent.checkpoint_generation, second_agent.checkpoint_generation)
                ml_service.train_model("first", first)
                first_generation = first_agent.checkpoint_generation
                ml_service.train_model("first", first)
                self.assertNotEqual(first_agent.checkpoint_generation, first_generation)
                ml_service.train_model("second", second)
                manifest, _ = store._manifest(_prefix(str(model_path)))
                self.assertEqual(manifest["high_water"], 3)
                self.assertEqual(manifest["generation"], first_agent.checkpoint_generation)
                self.assertNotEqual(manifest["generation"], second_agent.checkpoint_generation)
                self.assertFalse(Path(f"{model_path}.checkpoints").exists())

    def test_blob_shared_human_and_pro_generations_advance_together(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "shared.keras"
            weights_path = Path(directory) / "shared.weights.h5"
            model = self.new_network(PlayerType.ATARI_PRO)
            target = self.new_network(PlayerType.ATARI_PRO)
            target.set_weights(model.get_weights())
            model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
            model.save(str(model_path))
            target.save_weights(str(weights_path))
            store = BlobCheckpointStore(MemoryBlobs(CheckpointNotFound, CheckpointConflict))
            store.bootstrap(str(model_path), str(weights_path), _validate_with_keras)
            session, human = self.new_session(PlayerType.HUMAN, model_path, weights_path)
            _, pro = self.new_session(PlayerType.ATARI_PRO, model_path, weights_path)
            session.agents["second_0"] = pro

            with patch.dict(os.environ, {"MODEL_CHECKPOINT_BACKEND": "azure_blob"}), patch(
                "app.services.blob_checkpoints.blob_store_from_environment", return_value=store
            ):
                ml_service.load_model("shared", session, (52, 40, 1), 18)
                self.assertIs(human.q_network, pro.q_network)
                self.assertEqual(human.checkpoint_generation, pro.checkpoint_generation)
                old_generation = human.checkpoint_generation
                ml_service.train_model("shared", session)
                self.assertNotEqual(human.checkpoint_generation, old_generation)
                self.assertEqual(human.checkpoint_generation, pro.checkpoint_generation)
                ml_service.train_model("shared", session)
                manifest, _ = store._manifest(_prefix(str(model_path)))
                self.assertEqual(manifest["high_water"], 3)
                self.assertEqual(human.checkpoint_generation, pro.checkpoint_generation)


if __name__ == "__main__":
    unittest.main()
