from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import tempfile
import threading
import unittest
from unittest.mock import patch


CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1] / "app" / "services" / "model_checkpoints.py"
)
checkpoint_functions = runpy.run_path(str(CHECKPOINT_PATH))
locked_model_files = checkpoint_functions["locked_model_files"]
save_model_checkpoint = checkpoint_functions["save_model_checkpoint"]


class FakeModel:
    def __init__(self, value, error=None):
        self.value = value
        self.error = error

    def save(self, path):
        if self.error:
            raise self.error
        Path(path).write_bytes(self.value)

    def save_weights(self, path):
        self.save(path)


def validate_matching_files(model_path, weights_path):
    if Path(model_path).read_bytes() != Path(weights_path).read_bytes():
        raise ValueError("Model and target do not match")


def save_from_process(model_path, value, barrier, results):
    def validate_together(saved_model, saved_weights):
        validate_matching_files(saved_model, saved_weights)
        barrier.wait(timeout=15)

    try:
        version = save_model_checkpoint(
            model_path, FakeModel(value), FakeModel(value), validate_together
        )
        results.put((version, None))
    except Exception as error:
        results.put((None, repr(error)))


class ModelCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model_path = Path(self.directory.name) / "legacy.keras"
        self.weights_path = Path(self.directory.name) / "legacy.weights.h5"
        self.root = Path(f"{self.model_path}.checkpoints")

    def save(self, value, validate=validate_matching_files):
        return save_model_checkpoint(
            str(self.model_path), FakeModel(value), FakeModel(value), validate
        )

    def resolved_contents(self):
        with locked_model_files(str(self.model_path), str(self.weights_path)) as files:
            if files is None:
                return None
            model, weights = files
            return Path(model).read_bytes(), Path(weights).read_bytes()

    def manifest(self):
        return json.loads((self.root / "current.json").read_text(encoding="utf-8"))

    def test_unconfigured_and_new_models(self):
        for model, weights in ((None, None), ("", ""), (str(self.model_path), str(self.weights_path))):
            with self.subTest(model=model), locked_model_files(model, weights) as files:
                self.assertIsNone(files)
        with self.assertRaises(ValueError), locked_model_files(str(self.model_path), None):
            pass

    def test_legacy_pair_is_used_without_modification(self):
        self.model_path.write_bytes(b"old-model")
        self.weights_path.write_bytes(b"old-target")
        self.assertEqual(self.resolved_contents(), (b"old-model", b"old-target"))
        self.assertEqual(self.save(b"new"), "v1")
        self.assertEqual(self.model_path.read_bytes(), b"old-model")
        self.assertEqual(self.weights_path.read_bytes(), b"old-target")

    def test_incomplete_legacy_pair_fails_closed(self):
        self.model_path.write_bytes(b"model")
        with self.assertRaises(FileNotFoundError):
            self.resolved_contents()

    def test_failed_first_staging_does_not_create_a_checkpoint(self):
        with self.assertRaises(OSError):
            save_model_checkpoint(
                str(self.model_path), FakeModel(b"model", OSError("save failed")),
                FakeModel(b"target"), validate_matching_files,
            )
        self.assertIsNone(self.resolved_contents())
        self.assertEqual(list(self.root.glob(".stage-*")), [])
        self.assertEqual(self.save(b"first"), "v1")

    def test_sequential_versions_retain_three_rollback_targets(self):
        for number in range(1, 7):
            self.assertEqual(self.save(str(number).encode()), f"v{number}")
        self.assertEqual(self.manifest(), {
            "version": "v6", "high_water": 6,
            "history": ["v6", "v5", "v4", "v3"],
        })
        self.assertEqual(
            {path.name for path in (self.root / "versions").iterdir()},
            {"v3", "v4", "v5", "v6"},
        )
        self.assertEqual(self.resolved_contents(), (b"6", b"6"))

    def test_failed_writes_validation_and_publication_keep_current_version(self):
        self.save(b"first")
        cases = (
            (FakeModel(b"second", OSError("model save failed")), FakeModel(b"second"), validate_matching_files),
            (FakeModel(b"second"), FakeModel(b"second", OSError("weights save failed")), validate_matching_files),
            (FakeModel(b"second"), FakeModel(b"different"), validate_matching_files),
        )
        for model, target, validate in cases:
            with self.subTest(model_error=model.error, target_error=target.error):
                with self.assertRaises((OSError, ValueError)):
                    save_model_checkpoint(str(self.model_path), model, target, validate)
                self.assertEqual(self.resolved_contents(), (b"first", b"first"))
                self.assertEqual(self.manifest()["version"], "v1")

        with patch.object(os, "replace", side_effect=OSError("publication failed")):
            with self.assertRaises(OSError):
                self.save(b"second")
        self.assertEqual(self.resolved_contents(), (b"first", b"first"))
        self.assertEqual(list(self.root.glob(".current-*.tmp")), [])
        self.assertEqual(self.save(b"third"), "v3")
        self.assertFalse((self.root / "versions" / "v2").exists())

    def test_missing_pointer_with_completed_version_fails_closed(self):
        self.model_path.write_bytes(b"legacy")
        self.weights_path.write_bytes(b"legacy")
        (self.root / "versions" / "v1").mkdir(parents=True)
        with self.assertRaises(FileNotFoundError):
            self.resolved_contents()
        with self.assertRaises(FileNotFoundError):
            self.save(b"new")

    def test_invalid_or_incomplete_published_checkpoint_does_not_use_legacy(self):
        self.model_path.write_bytes(b"legacy")
        self.weights_path.write_bytes(b"legacy")
        self.root.mkdir()
        pointer = self.root / "current.json"
        payloads = (
            "not json", "[]", '{"version": "../legacy"}',
            '{"version": "v1", "high_water": 0, "history": ["v1"]}',
            '{"version": "v1", "high_water": 1, "history": ["v1", "v1"]}',
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                pointer.write_text(payload, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.resolved_contents()
                with self.assertRaises(ValueError):
                    self.save(b"new")
        pointer.write_text(json.dumps({"version": "a" * 32}), encoding="utf-8")
        with self.assertRaises(FileNotFoundError):
            self.resolved_contents()

    def test_uuid_pointer_is_migrated_and_retained_as_a_rollback(self):
        version = "a" * 32
        old_directory = self.root / "versions" / version
        old_directory.mkdir(parents=True)
        (old_directory / "model.keras").write_bytes(b"old")
        (old_directory / "target.weights.h5").write_bytes(b"old")
        (self.root / "current.json").write_text(json.dumps({"version": version}), encoding="utf-8")
        self.assertEqual(self.resolved_contents(), (b"old", b"old"))
        self.assertEqual(self.save(b"new"), "v1")
        self.assertEqual(self.manifest()["history"], ["v1", version])
        for number in range(2, 5):
            self.save(str(number).encode())
        self.assertFalse(old_directory.exists())

    def test_failed_uuid_cleanup_is_retried_on_next_save(self):
        version = "a" * 32
        old_directory = self.root / "versions" / version
        old_directory.mkdir(parents=True)
        (old_directory / "model.keras").write_bytes(b"old")
        (old_directory / "target.weights.h5").write_bytes(b"old")
        (self.root / "current.json").write_text(json.dumps({"version": version}), encoding="utf-8")
        for number in range(1, 4):
            self.save(str(number).encode())
        with patch("shutil.rmtree", side_effect=OSError("cleanup failed")):
            self.save(b"four")
        self.assertEqual(self.manifest()["cleanup"], [version])
        self.assertTrue(old_directory.exists())
        self.save(b"five")
        self.assertFalse(old_directory.exists())

    def test_orphan_version_advances_high_water_and_is_cleaned(self):
        self.save(b"first")
        orphan = self.root / "versions" / "v9"
        orphan.mkdir()
        self.assertEqual(self.save(b"second"), "v10")
        self.assertEqual(self.manifest()["high_water"], 10)
        self.assertFalse(orphan.exists())

    def test_missing_rollback_file_blocks_new_publication(self):
        self.save(b"first")
        self.save(b"second")
        (self.root / "versions" / "v1" / "target.weights.h5").unlink()
        with self.assertRaises(FileNotFoundError):
            self.save(b"third")
        self.assertEqual(self.resolved_contents(), (b"second", b"second"))
        self.assertEqual(self.manifest()["version"], "v2")

    def test_rollback_keeps_high_water_and_previous_active_version(self):
        for number in range(1, 5):
            self.save(str(number).encode())
        manifest = self.manifest()
        manifest["version"] = "v2"
        manifest["history"] = ["v2", "v4", "v3", "v1"]
        (self.root / "current.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(self.resolved_contents(), (b"2", b"2"))
        self.assertEqual(self.save(b"five"), "v5")
        self.assertEqual(self.manifest()["history"], ["v5", "v2", "v4", "v3"])

    def test_cleanup_failure_does_not_undo_published_checkpoint(self):
        for number in range(1, 5):
            self.save(str(number).encode())
        with patch("shutil.rmtree", side_effect=OSError("cleanup failed")):
            self.assertEqual(self.save(b"five"), "v5")
        self.assertEqual(self.resolved_contents(), (b"five", b"five"))
        self.assertTrue((self.root / "versions" / "v1").exists())
        self.save(b"six")
        self.assertFalse((self.root / "versions" / "v1").exists())

    def test_reader_blocks_publication_and_cleanup_until_load_finishes(self):
        self.save(b"first")
        reader_started = threading.Event()
        release_reader = threading.Event()
        writer_ready = threading.Event()

        def reader():
            with locked_model_files(str(self.model_path), str(self.weights_path)) as files:
                reader_started.set()
                if not release_reader.wait(timeout=15):
                    raise TimeoutError("Test did not release reader")
                return tuple(Path(path).read_bytes() for path in files)

        def validate_and_signal(model_path, weights_path):
            validate_matching_files(model_path, weights_path)
            writer_ready.set()

        def writer():
            self.save(b"second", validate_and_signal)
            for value in (b"third", b"fourth", b"fifth"):
                self.save(value)

        with ThreadPoolExecutor(max_workers=2) as executor:
            reader_future = executor.submit(reader)
            try:
                self.assertTrue(reader_started.wait(timeout=10))
                writer_future = executor.submit(writer)
                self.assertTrue(writer_ready.wait(timeout=10))
                self.assertFalse(writer_future.done())
                self.assertTrue((self.root / "versions" / "v1").exists())
            finally:
                release_reader.set()
            self.assertEqual(reader_future.result(timeout=15), (b"first", b"first"))
            writer_future.result(timeout=15)
        self.assertFalse((self.root / "versions" / "v1").exists())

    def test_slow_staging_does_not_block_loads_or_other_saves(self):
        self.save(b"first")
        staging_started = threading.Event()
        release_staging = threading.Event()

        def wait_before_publish(model_path, weights_path):
            validate_matching_files(model_path, weights_path)
            staging_started.set()
            if not release_staging.wait(timeout=15):
                raise TimeoutError("Test did not release staging")

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.save, b"slow", wait_before_publish)
            try:
                self.assertTrue(staging_started.wait(timeout=10))
                self.assertEqual(self.resolved_contents(), (b"first", b"first"))
                for value in (b"second", b"third", b"fourth", b"fifth"):
                    self.save(value)
                self.assertEqual(len(list(self.root.glob(".stage-*"))), 1)
            finally:
                release_staging.set()
            self.assertEqual(future.result(timeout=15), "v6")
        self.assertEqual(self.resolved_contents(), (b"slow", b"slow"))

    def test_concurrent_processes_publish_distinct_complete_versions(self):
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(2)
        results = context.Queue()
        processes = [
            context.Process(target=save_from_process, args=(str(self.model_path), value, barrier, results))
            for value in (b"first", b"second")
        ]
        try:
            for process in processes:
                process.start()
            outcomes = [results.get(timeout=30) for process in processes]
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()
            results.close()
        self.assertEqual({version for version, error in outcomes}, {"v1", "v2"})
        self.assertEqual([error for version, error in outcomes], [None, None])
        model, weights = self.resolved_contents()
        self.assertEqual(model, weights)
        self.assertIn(model, (b"first", b"second"))


if __name__ == "__main__":
    unittest.main()
