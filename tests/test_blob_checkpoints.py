from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
local_functions = runpy.run_path(str(ROOT / "app/services/model_checkpoints.py"))
local_module = types.ModuleType("app.services.model_checkpoints")
local_module.__dict__.update(local_functions)
with patch.dict(sys.modules, {"app.services.model_checkpoints": local_module}):
    blob = runpy.run_path(str(ROOT / "app/services/blob_checkpoints.py"))

Store = blob["BlobCheckpointStore"]
Conflict = blob["CheckpointConflict"]
Stale = blob["StaleCheckpointError"]
NotFound = blob["CheckpointNotFound"]
prefix_for = blob["_prefix"]


class MemoryBlobs:
    def __init__(self, not_found=NotFound, conflict=Conflict):
        self.blobs = {}
        self.not_found = not_found
        self.conflict = conflict
        self.lock = threading.Lock()
        self.number = 0
        self.after_download = None
        self.ambiguous_replace = False
        self.delete_failure = False
        self.replace_barrier = None

    def read(self, name):
        with self.lock:
            if name not in self.blobs:
                raise self.not_found(name)
            return self.blobs[name]

    def download(self, name, destination):
        if self.after_download:
            callback, self.after_download = self.after_download, None
            callback()
        data, _ = self.read(name)
        Path(destination).write_bytes(data)

    def create(self, name, data):
        with self.lock:
            if name in self.blobs:
                raise self.conflict(name)
            self.number += 1
            self.blobs[name] = (data.read() if hasattr(data, "read") else data, str(self.number))

    def replace(self, name, data, etag):
        if self.replace_barrier:
            self.replace_barrier.wait(timeout=10)
        with self.lock:
            if name not in self.blobs or self.blobs[name][1] != etag:
                raise self.conflict(name)
            self.number += 1
            self.blobs[name] = (data, str(self.number))
            if self.ambiguous_replace:
                self.ambiguous_replace = False
                raise OSError("connection lost after publication")

    def delete(self, name):
        if self.delete_failure:
            raise OSError("delete failed")
        with self.lock:
            self.blobs.pop(name, None)


class FakeModel:
    def __init__(self, value):
        self.value = value

    def save(self, path):
        Path(path).write_bytes(self.value)

    def save_weights(self, path):
        self.save(path)


def validate(model_path, weights_path):
    if Path(model_path).read_bytes() != Path(weights_path).read_bytes():
        raise ValueError("Mismatched model and weights")


class BlobCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = str(Path(self.directory.name) / "model.keras")
        self.weights = str(Path(self.directory.name) / "model.weights.h5")
        self.blobs = MemoryBlobs()
        self.store = Store(self.blobs)
        self.prefix = prefix_for(self.model)

    def bootstrap(self, value=b"first"):
        Path(self.model).write_bytes(value)
        Path(self.weights).write_bytes(value)
        self.store.bootstrap(self.model, self.weights, validate)

    def manifest(self):
        return json.loads(self.blobs.read(f"{self.prefix}/manifest.json")[0])

    def load(self):
        with self.store.load(self.model) as (paths, generation):
            return (tuple(Path(path).read_bytes() for path in paths) if paths else None), generation

    def save(self, value, generation=None, validation=validate):
        if generation is None:
            _, generation = self.load()
        return self.store.save(self.model, FakeModel(value), FakeModel(value), validation, generation)

    def test_explicit_migration_and_retention(self):
        self.bootstrap()
        for number in range(2, 7):
            self.assertEqual(self.save(str(number).encode()), f"v{number}")
        manifest = self.manifest()
        self.assertEqual([entry["version"] for entry in manifest["history"]], ["v6", "v5", "v4", "v3"])
        self.assertEqual(manifest["cleanup"], [])
        self.assertEqual(len([name for name in self.blobs.blobs if "/objects/" in name]), 4)
        self.assertEqual(self.load()[0], (b"6", b"6"))

    def test_old_local_history_and_high_water_are_migrated(self):
        root = Path(f"{self.model}.checkpoints")
        for version in ("v8", "v3"):
            folder = root / "versions" / version
            folder.mkdir(parents=True)
            (folder / "model.keras").write_bytes(version.encode())
            (folder / "target.weights.h5").write_bytes(version.encode())
        (root / "current.json").write_text(json.dumps({
            "version": "v8", "history": ["v8", "v3"], "high_water": 12,
        }))
        self.store.bootstrap(self.model, self.weights, validate)
        self.assertEqual(self.load()[0], (b"v8", b"v8"))
        self.assertEqual(self.save(b"next"), "v13")
        self.assertEqual([item["version"] for item in self.manifest()["history"]], ["v13", "v8", "v3"])

    def test_stale_and_concurrent_sessions_cannot_overwrite(self):
        self.bootstrap()
        _, generation = self.load()
        barrier = threading.Barrier(2)
        self.blobs.replace_barrier = threading.Barrier(2)

        def validate_together(model_path, weights_path):
            validate(model_path, weights_path)
            barrier.wait(timeout=10)

        def publish(value):
            try:
                return self.save(value, generation, validate_together)
            except Stale:
                return "stale"

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(publish, [b"second-a", b"second-b"]))
        self.assertCountEqual(results, ["v2", "stale"])
        self.assertIn(self.load()[0], [(b"second-a", b"second-a"), (b"second-b", b"second-b")])
        with self.assertRaises(Stale):
            self.save(b"old", generation)

    def test_migration_does_not_reuse_an_unpublished_local_version_number(self):
        root = Path(f"{self.model}.checkpoints")
        folder = root / "versions" / "v2"
        folder.mkdir(parents=True)
        (folder / "model.keras").write_bytes(b"current")
        (folder / "target.weights.h5").write_bytes(b"current")
        (root / "versions" / "v9").mkdir()
        (root / "current.json").write_text(json.dumps({
            "version": "v2", "history": ["v2"], "high_water": 2,
        }))
        self.store.bootstrap(self.model, self.weights, validate)
        self.assertEqual(self.save(b"next"), "v10")

    def test_model_key_normalization_is_independent_of_slash_style(self):
        self.assertEqual(
            prefix_for(r".\resources\unused\..\models\game.keras"),
            prefix_for("resources/models/game.keras"),
        )

    def test_lost_publication_response_is_reconciled_and_writer_can_continue(self):
        self.bootstrap()
        self.blobs.ambiguous_replace = True
        version = self.save(b"second")
        self.assertEqual(version, "v2")
        self.assertEqual(version.generation, self.manifest()["generation"])
        self.assertEqual(self.load()[0], (b"second", b"second"))
        self.assertEqual(len([name for name in self.blobs.blobs if "/objects/" in name]), 2)
        self.assertEqual(self.save(b"third", version.generation), "v3")

    def test_rejected_candidate_is_deleted_only_before_any_publication_attempt(self):
        self.bootstrap()
        _, generation = self.load()

        def publish_other_model(model_path, weights_path):
            validate(model_path, weights_path)
            self.save(b"winner", generation)

        with self.assertRaises(Stale):
            self.save(b"loser", generation, publish_other_model)
        self.assertEqual(self.load()[0], (b"winner", b"winner"))
        self.assertEqual(len([name for name in self.blobs.blobs if "/objects/" in name]), 2)

    def test_unverifiable_publication_never_deletes_the_candidate(self):
        self.bootstrap()
        original_replace = self.blobs.replace

        def lose_response_and_storage(name, data, etag):
            original_replace(name, data, etag)
            self.blobs.read = lambda *_: (_ for _ in ()).throw(OSError("storage unavailable"))
            raise OSError("publication response lost")

        with patch.object(self.blobs, "replace", side_effect=lose_response_and_storage):
            with self.assertRaises(OSError):
                self.save(b"second")
        self.assertEqual(len([name for name in self.blobs.blobs if "/objects/" in name]), 2)

    def test_conditional_retry_after_success_is_reconciled(self):
        self.bootstrap()
        original_replace = self.blobs.replace

        def write_then_report_retry_conflict(name, data, etag):
            original_replace(name, data, etag)
            raise Conflict("conditional retry rejected after successful first attempt")

        with patch.object(self.blobs, "replace", side_effect=write_then_report_retry_conflict):
            version = self.save(b"second")
        self.assertEqual(version, "v2")
        self.assertEqual(version.generation, self.manifest()["generation"])

    def test_lost_response_does_not_adopt_another_writers_generation(self):
        self.bootstrap()
        original_replace = self.blobs.replace

        def publish_then_another_writer_wins(name, data, etag):
            original_replace(name, data, etag)
            with patch.object(self.blobs, "replace", original_replace):
                self.save(b"third")
            raise OSError("second publication response lost")

        with patch.object(self.blobs, "replace", side_effect=publish_then_another_writer_wins):
            with self.assertRaises(OSError):
                self.save(b"second")
        self.assertEqual(self.load()[0], (b"third", b"third"))

    def test_failed_staging_never_publishes(self):
        self.bootstrap()
        with self.assertRaises(ValueError):
            self.save(b"second", validation=lambda *_: (_ for _ in ()).throw(ValueError("bad model")))
        self.assertEqual(self.load()[0], (b"first", b"first"))
        self.assertEqual(len([name for name in self.blobs.blobs if "/objects/" in name]), 1)

    def test_missing_bad_and_corrupted_data_fail_closed(self):
        with self.assertRaises(NotFound):
            self.load()
        self.bootstrap()
        name = f"{self.prefix}/manifest.json"
        data, tag = self.blobs.read(name)
        self.blobs.blobs[name] = (b"not json", tag)
        with self.assertRaises(ValueError):
            self.load()
        self.blobs.blobs[name] = (data, tag)
        entry = self.manifest()["history"][0]
        object_name = f"{self.prefix}/objects/{entry['object']}.zip"
        self.blobs.blobs[object_name] = (b"broken", "0")
        with self.assertRaises(ValueError):
            self.load()
        del self.blobs.blobs[object_name]
        with self.assertRaises(NotFound):
            self.load()

    def test_empty_bootstrap_is_explicit_and_never_overwrites(self):
        self.store.bootstrap(self.model, self.weights, validate, empty=True)
        self.assertIsNone(self.load()[0])
        self.assertEqual(self.save(b"first"), "v1")
        with self.assertRaises(Conflict):
            self.store.bootstrap(self.model, self.weights, validate, empty=True)
        Path(self.model).write_bytes(b"existing")
        with self.assertRaises(ValueError):
            self.store.bootstrap(self.model, self.weights, validate, empty=True)

    def test_migration_rejects_missing_pairs_and_unpublished_versions(self):
        Path(self.model).write_bytes(b"existing")
        with self.assertRaises(FileNotFoundError):
            self.store.bootstrap(self.model, self.weights, validate)
        (Path(f"{self.model}.checkpoints") / "versions" / "v1").mkdir(parents=True)
        with self.assertRaises(FileNotFoundError):
            self.store.bootstrap(self.model, self.weights, validate, empty=True)
        self.assertFalse(self.blobs.blobs)

    def test_blob_configuration_and_unknown_backend_fail_closed(self):
        with patch.dict(os.environ, {
            "MODEL_CHECKPOINT_BACKEND": "azure_blob",
            "MODEL_CHECKPOINT_BLOB_ACCOUNT_URL": "",
            "MODEL_CHECKPOINT_BLOB_CONTAINER": "",
        }):
            with self.assertRaises(ValueError):
                blob["blob_store_from_environment"]()
        with patch.dict(os.environ, {"MODEL_CHECKPOINT_BACKEND": "misspelled"}):
            with self.assertRaises(ValueError), local_functions["load_model_checkpoint"](self.model, self.weights):
                pass

    def test_cli_bootstrap_requires_writers_stopped_confirmation(self):
        with patch.object(sys, "argv", ["checkpoints", "bootstrap", self.model, self.weights]):
            with patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as error:
                    blob["main"]()
        self.assertEqual(error.exception.code, 2)

    def test_inventory_lists_configured_paths_without_blob_access(self):
        configurations = {
            "Boxing": types.SimpleNamespace(model_path=self.model, weights_path=self.weights),
        }
        environment_module = types.ModuleType("app.config.env_config")
        environment_module.ATTARI_PRO_MODEL = ""
        environment_module.ATTARI_PRO_WEIGHTS_PATH = ""
        config_module = types.ModuleType("app.config.ml_env_config")
        config_module.ENVIRONMENTS = configurations
        with patch.dict(sys.modules, {
            "app.config.env_config": environment_module, "app.config.ml_env_config": config_module,
        }):
            with patch.object(sys, "argv", ["checkpoints", "inventory"]):
                with patch("sys.stdout", new_callable=io.StringIO) as output:
                    blob["main"]()
            inventory = json.loads(output.getvalue())
            self.assertEqual(inventory[0]["model_path"], self.model)
            self.assertEqual(inventory[0]["prefix"], self.prefix)
            environment_module.ATTARI_PRO_MODEL = self.model
            environment_module.ATTARI_PRO_WEIGHTS_PATH = "different.weights.h5"
            with self.assertRaisesRegex(ValueError, "Conflicting weights"):
                blob["configured_models"]()

    def test_storage_client_is_closed_without_masking_result_or_failure(self):
        closed = []
        self.blobs.close = lambda: closed.append(True)
        with self.store:
            self.bootstrap()
        self.assertEqual(closed, [True])
        with self.assertRaisesRegex(ValueError, "operation failed"):
            with self.store:
                raise ValueError("operation failed")
        self.assertEqual(closed, [True, True])

    def test_cleanup_retries_and_reader_reloads_retired_version(self):
        self.bootstrap()
        for number in range(2, 5):
            self.save(str(number).encode())
        self.blobs.delete_failure = True
        self.save(b"5")
        self.assertEqual(len(self.manifest()["cleanup"]), 1)
        self.blobs.delete_failure = False
        self.save(b"6")
        self.assertEqual(self.manifest()["cleanup"], [])
        self.blobs.after_download = lambda: [self.save(str(value).encode()) for value in range(7, 11)]
        self.assertEqual(self.load()[0], (b"10", b"10"))

    def test_rollback_changes_generation_and_keeps_version_counter(self):
        self.bootstrap()
        self.save(b"second")
        _, old_generation = self.load()
        self.store.rollback(self.model, "v1")
        self.assertEqual(self.load()[0], (b"first", b"first"))
        with self.assertRaises(Stale):
            self.save(b"stale", old_generation)
        self.assertEqual(self.save(b"third"), "v3")
        with self.assertRaises(ValueError):
            self.store.rollback(self.model, "v99")


if __name__ == "__main__":
    unittest.main()
