import json
import logging
import os
import re
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


_VERSION = re.compile(r"v([1-9][0-9]*)\Z")
_LEGACY_VERSION = re.compile(r"[0-9a-f]{32}\Z")
_RETAINED_VERSIONS = 4
_LOGGER = logging.getLogger(__name__)


def _checkpoint_root(model_path):
    return Path(f"{model_path}.checkpoints")


def _valid_version(version):
    return isinstance(version, str) and bool(
        _VERSION.fullmatch(version) or _LEGACY_VERSION.fullmatch(version)
    )


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _checkpoint_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(root / "checkpoint.lock.sqlite3"), timeout=120)
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield
        connection.commit()
    finally:
        connection.close()


def _checkpoint_files(root, version):
    version_dir = root / "versions" / version
    model = version_dir / "model.keras"
    weights = version_dir / "target.weights.h5"
    if not model.is_file() or not weights.is_file():
        raise FileNotFoundError(f"Incomplete model checkpoint: {version_dir}")
    return str(model), str(weights)


def _read_manifest(root):
    pointer = root / "current.json"
    try:
        with pointer.open(encoding="utf-8") as pointer_file:
            data = json.load(pointer_file)
    except FileNotFoundError:
        return None

    if not isinstance(data, dict) or not _valid_version(data.get("version")):
        raise ValueError(f"Invalid checkpoint pointer: {pointer}")
    version = data["version"]
    if set(data) == {"version"} and _LEGACY_VERSION.fullmatch(version):
        return {"version": version, "high_water": 0, "history": [version], "cleanup": []}

    history = data.get("history")
    high_water = data.get("high_water")
    cleanup = data.get("cleanup", [])
    if (
        not isinstance(high_water, int)
        or isinstance(high_water, bool)
        or high_water < 0
        or not isinstance(history, list)
        or not 1 <= len(history) <= _RETAINED_VERSIONS
        or history[0] != version
        or any(not _valid_version(item) for item in history)
        or len(set(history)) != len(history)
        or not isinstance(cleanup, list)
        or any(not isinstance(item, str) or not _LEGACY_VERSION.fullmatch(item) for item in cleanup)
        or len(set(cleanup)) != len(cleanup)
        or any(item in history for item in cleanup)
        or any(
            int(match.group(1)) > high_water
            for item in history
            if (match := _VERSION.fullmatch(item))
        )
    ):
        raise ValueError(f"Invalid checkpoint manifest: {pointer}")
    return {"version": version, "high_water": high_water, "history": history, "cleanup": cleanup}


def _require_no_unpublished_versions(root):
    versions = root / "versions"
    if versions.is_dir() and any(versions.iterdir()):
        raise FileNotFoundError(f"Checkpoint pointer is missing: {root / 'current.json'}")


@contextmanager
def locked_model_files(model_path, weights_path):
    if not model_path and not weights_path:
        yield None
        return
    if not model_path or not weights_path:
        raise ValueError("Both model and target weights paths must be configured")

    root = _checkpoint_root(model_path)
    with _checkpoint_lock(root):
        manifest = _read_manifest(root)
        if manifest is not None:
            yield _checkpoint_files(root, manifest["version"])
            return

        _require_no_unpublished_versions(root)
        model = Path(model_path)
        weights = Path(weights_path)
        if not model.exists() and not weights.exists():
            yield None
            return
        if not model.is_file() or not weights.is_file():
            raise FileNotFoundError("Both legacy model and target weights files are required")
        yield str(model), str(weights)


def _next_version(root, manifest):
    high_water = manifest["high_water"] if manifest else 0
    for path in (root / "versions").iterdir():
        match = _VERSION.fullmatch(path.name)
        if match:
            high_water = max(high_water, int(match.group(1)))
    return high_water + 1


def _publish_manifest(root, manifest):
    temporary_pointer = root / f".current-{uuid4().hex}.tmp"
    try:
        with temporary_pointer.open("x", encoding="utf-8") as pointer_file:
            json.dump(manifest, pointer_file)
            pointer_file.flush()
            os.fsync(pointer_file.fileno())
        os.replace(temporary_pointer, root / "current.json")
        _sync_directory(root)
    finally:
        temporary_pointer.unlink(missing_ok=True)


def _cleanup_versions(root, cleanup, history):
    versions = root / "versions"
    retained = set(history)
    try:
        paths = list(versions.iterdir())
    except OSError:
        _LOGGER.warning("Could not scan old checkpoints in %s", versions, exc_info=True)
        return
    for path in paths:
        if path.name in retained:
            continue
        if not _VERSION.fullmatch(path.name) and path.name not in cleanup:
            continue
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
        except OSError:
            _LOGGER.warning("Could not remove old checkpoint %s", path, exc_info=True)


def save_model_checkpoint(model_path, model, target_model, validate):
    if not model_path:
        raise ValueError("Model path must be configured")
    root = _checkpoint_root(model_path)
    versions = root / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".stage-", dir=root))
    model_file = staging / "model.keras"
    weights_file = staging / "target.weights.h5"

    try:
        model.save(str(model_file))
        target_model.save_weights(str(weights_file))
        validate(str(model_file), str(weights_file))
        for checkpoint_file in (model_file, weights_file):
            with checkpoint_file.open("rb") as saved_file:
                os.fsync(saved_file.fileno())
        _sync_directory(staging)

        with _checkpoint_lock(root):
            manifest = _read_manifest(root)
            if manifest is None:
                _require_no_unpublished_versions(root)
                previous_history = []
                previous_cleanup = []
            else:
                previous_history = manifest["history"]
                previous_cleanup = manifest["cleanup"]
                for version in previous_history:
                    _checkpoint_files(root, version)

            number = _next_version(root, manifest)
            version = f"v{number}"
            os.rename(staging, versions / version)
            _sync_directory(versions)
            history = [version, *previous_history][:_RETAINED_VERSIONS]
            cleanup = [
                item for item in [*previous_cleanup, *previous_history]
                if _LEGACY_VERSION.fullmatch(item)
                and item not in history
                and (versions / item).exists()
            ]
            new_manifest = {
                "version": version,
                "high_water": number,
                "history": history,
            }
            if cleanup:
                new_manifest["cleanup"] = cleanup
            _publish_manifest(root, new_manifest)
            _cleanup_versions(root, cleanup, history)
            return version
    finally:
        if staging.is_dir():
            shutil.rmtree(staging)
