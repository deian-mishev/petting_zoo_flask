import argparse
import hashlib
import json
import logging
import os
import posixpath
import re
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from app.services.model_checkpoints import (
    _checkpoint_files,
    _checkpoint_root,
    _next_version,
    _read_manifest,
    _require_no_unpublished_versions,
    _valid_version,
)


_LOGGER = logging.getLogger(__name__)
_UUID = re.compile(r"[0-9a-f]{32}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_NUMERIC_VERSION = re.compile(r"v([1-9][0-9]*)\Z")
_RETAINED = 4


class CheckpointConflict(RuntimeError):
    pass


class StaleCheckpointError(CheckpointConflict):
    pass


class CheckpointNotFound(FileNotFoundError):
    pass


class PublishedVersion(str):
    def __new__(cls, version, generation):
        value = super().__new__(cls, version)
        value.generation = generation
        return value


class AzureBlobAdapter:
    def __init__(self, container, credential=None):
        self.container = container
        self.credential = credential

    def close(self):
        try:
            self.container.close()
        finally:
            if self.credential is not None:
                self.credential.close()

    def read(self, name):
        from azure.core.exceptions import ResourceNotFoundError
        try:
            download = self.container.get_blob_client(name).download_blob()
            return download.readall(), download.properties.etag
        except ResourceNotFoundError as error:
            raise CheckpointNotFound(name) from error

    def download(self, name, destination):
        from azure.core.exceptions import ResourceNotFoundError
        try:
            download = self.container.get_blob_client(name).download_blob()
            with open(destination, "wb") as output:
                download.readinto(output)
        except ResourceNotFoundError as error:
            raise CheckpointNotFound(name) from error

    def create(self, name, data):
        from azure.core.exceptions import ResourceExistsError
        try:
            self.container.get_blob_client(name).upload_blob(data, overwrite=False)
        except ResourceExistsError as error:
            raise CheckpointConflict(f"Blob already exists: {name}") from error

    def replace(self, name, data, etag):
        from azure.core import MatchConditions
        from azure.core.exceptions import HttpResponseError
        try:
            self.container.get_blob_client(name).upload_blob(
                data, overwrite=True, etag=etag, match_condition=MatchConditions.IfNotModified
            )
        except HttpResponseError as error:
            if error.status_code == 412:
                raise CheckpointConflict(f"Blob changed: {name}") from error
            raise

    def delete(self, name):
        from azure.core.exceptions import ResourceNotFoundError
        try:
            self.container.get_blob_client(name).delete_blob()
        except ResourceNotFoundError:
            pass


def blob_store_from_environment():
    account_url = os.getenv("MODEL_CHECKPOINT_BLOB_ACCOUNT_URL", "").strip()
    container_name = os.getenv("MODEL_CHECKPOINT_BLOB_CONTAINER", "").strip()
    if not account_url.startswith("https://") or not container_name:
        raise ValueError("Azure Blob account URL and container must be configured")
    from azure.identity import ManagedIdentityCredential
    from azure.storage.blob import ContainerClient
    client_id = os.getenv("MODEL_CHECKPOINT_MANAGED_IDENTITY_CLIENT_ID", "").strip()
    credential = ManagedIdentityCredential(client_id=client_id or None)
    try:
        container = ContainerClient(account_url=account_url, container_name=container_name, credential=credential)
    except Exception:
        credential.close()
        raise
    return BlobCheckpointStore(AzureBlobAdapter(container, credential))


def _prefix(model_path):
    if not model_path:
        raise ValueError("Model path must be configured")
    normalized = posixpath.normpath(os.fspath(model_path).replace("\\", "/"))
    return f"model-checkpoints/{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"


def _check_manifest(manifest):
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema", "generation", "high_water", "history", "cleanup"
    }:
        raise ValueError("Invalid Blob checkpoint manifest")
    generation = manifest["generation"]
    high_water = manifest["high_water"]
    history = manifest["history"]
    cleanup = manifest["cleanup"]
    if (
        not isinstance(manifest["schema"], int) or isinstance(manifest["schema"], bool)
        or manifest["schema"] != 1
        or not isinstance(generation, str) or not _UUID.fullmatch(generation)
        or not isinstance(high_water, int) or isinstance(high_water, bool) or high_water < 0
        or not isinstance(history, list) or len(history) > _RETAINED
        or (not history and high_water != 0)
        or not isinstance(cleanup, list)
    ):
        raise ValueError("Invalid Blob checkpoint manifest")
    versions = set()
    objects = set()
    for entry in history:
        if not isinstance(entry, dict) or set(entry) != {"version", "object", "sha256"}:
            raise ValueError("Invalid Blob checkpoint history")
        version, object_id, checksum = entry["version"], entry["object"], entry["sha256"]
        if (
            not _valid_version(version)
            or not isinstance(object_id, str) or not _UUID.fullmatch(object_id)
            or not isinstance(checksum, str) or not _HASH.fullmatch(checksum)
            or version in versions or object_id in objects
        ):
            raise ValueError("Invalid Blob checkpoint history")
        match = _NUMERIC_VERSION.fullmatch(version)
        if match and int(match.group(1)) > high_water:
            raise ValueError("Checkpoint version exceeds high-water mark")
        versions.add(version)
        objects.add(object_id)
    for object_id in cleanup:
        if not isinstance(object_id, str) or not _UUID.fullmatch(object_id) or object_id in objects:
            raise ValueError("Invalid Blob checkpoint cleanup list")
        objects.add(object_id)
    return manifest


def _digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _archive(model_file, weights_file, archive_file):
    with zipfile.ZipFile(archive_file, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(model_file, "model.keras")
        archive.write(weights_file, "target.weights.h5")
    return _digest(archive_file)


class BlobCheckpointStore:
    def __init__(self, blobs):
        self.blobs = blobs

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        close = getattr(self.blobs, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                _LOGGER.warning("Could not close checkpoint storage client", exc_info=True)

    def _manifest(self, prefix):
        try:
            data, etag = self.blobs.read(f"{prefix}/manifest.json")
        except CheckpointNotFound as error:
            raise CheckpointNotFound(
                "Blob checkpoint manifest is missing"
            ) from error
        try:
            return _check_manifest(json.loads(data)), etag
        except (ValueError, UnicodeDecodeError, TypeError) as error:
            raise ValueError("Invalid Blob checkpoint manifest") from error

    def _write(self, prefix, manifest, etag=None):
        data = json.dumps(_check_manifest(manifest), separators=(",", ":")).encode("utf-8")
        name = f"{prefix}/manifest.json"
        if etag is None:
            self.blobs.create(name, data)
        else:
            self.blobs.replace(name, data, etag)

    def _upload_pair(self, prefix, version, model_file, weights_file, directory):
        object_id = uuid4().hex
        archive_file = Path(directory) / f"{object_id}.zip"
        checksum = _archive(model_file, weights_file, archive_file)
        with archive_file.open("rb") as source:
            self.blobs.create(f"{prefix}/objects/{object_id}.zip", source)
        return {"version": version, "object": object_id, "sha256": checksum}

    def _download_pair(self, prefix, entry, directory):
        archive_file = Path(directory) / "checkpoint.zip"
        self.blobs.download(f"{prefix}/objects/{entry['object']}.zip", archive_file)
        if _digest(archive_file) != entry["sha256"]:
            raise ValueError("Blob checkpoint checksum mismatch")
        with zipfile.ZipFile(archive_file) as archive:
            if sorted(archive.namelist()) != ["model.keras", "target.weights.h5"]:
                raise ValueError("Invalid Blob checkpoint archive")
            for name in ("model.keras", "target.weights.h5"):
                with archive.open(name) as source, (Path(directory) / name).open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
        return str(Path(directory) / "model.keras"), str(Path(directory) / "target.weights.h5")

    @contextmanager
    def load(self, model_path):
        prefix = _prefix(model_path)
        with tempfile.TemporaryDirectory(prefix="model-load-") as directory:
            for attempt in range(5):
                try:
                    manifest, _ = self._manifest(prefix)
                except CheckpointNotFound:
                    # Create-only publication lets concurrent first readers share
                    # one generation without replacing an existing checkpoint.
                    try:
                        self._write(prefix, {
                            "schema": 1, "generation": uuid4().hex,
                            "high_water": 0, "history": [], "cleanup": [],
                        })
                    except CheckpointConflict:
                        pass
                    manifest, _ = self._manifest(prefix)
                if not manifest["history"]:
                    yield None, manifest["generation"]
                    return
                entry = manifest["history"][0]
                try:
                    files = self._download_pair(prefix, entry, directory)
                except CheckpointNotFound:
                    current, _ = self._manifest(prefix)
                    if any(item["object"] == entry["object"] for item in current["history"]):
                        raise
                    if attempt == 4:
                        raise CheckpointConflict("Checkpoint changed repeatedly while loading")
                    continue
                yield files, manifest["generation"]
                return

    def save(self, model_path, model, target_model, validate, expected_generation):
        if not expected_generation:
            raise ValueError("A loaded Blob checkpoint generation is required to save")
        prefix = _prefix(model_path)
        manifest, _ = self._manifest(prefix)
        if manifest["generation"] != expected_generation:
            raise StaleCheckpointError("Model changed since this session loaded; training update was not published")
        with tempfile.TemporaryDirectory(prefix="model-save-") as directory:
            model_file = str(Path(directory) / "model.keras")
            weights_file = str(Path(directory) / "target.weights.h5")
            model.save(model_file)
            target_model.save_weights(weights_file)
            validate(model_file, weights_file)
            entry = self._upload_pair(prefix, "v1", model_file, weights_file, directory)
            publication_attempted = False
            for _ in range(5):
                manifest, etag = self._manifest(prefix)
                if manifest["generation"] != expected_generation:
                    if not publication_attempted:
                        try:
                            self.blobs.delete(f"{prefix}/objects/{entry['object']}.zip")
                        except Exception:
                            _LOGGER.warning("Could not remove unpublished checkpoint %s", entry["object"], exc_info=True)
                    raise StaleCheckpointError("Model changed since this session loaded; training update was not published")
                number = manifest["high_water"] + 1
                entry["version"] = f"v{number}"
                history = [entry, *manifest["history"]]
                updated = {
                    "schema": 1,
                    "generation": uuid4().hex,
                    "high_water": number,
                    "history": history[:_RETAINED],
                    "cleanup": [*manifest["cleanup"], *(item["object"] for item in history[_RETAINED:])],
                }
                try:
                    publication_attempted = True
                    self._write(prefix, updated, etag)
                except Exception as error:
                    try:
                        current, _ = self._manifest(prefix)
                    except Exception:
                        raise error
                    if (current["generation"] != updated["generation"]
                            or not current["history"] or current["history"][0] != entry):
                        if isinstance(error, CheckpointConflict):
                            continue
                        raise
                self._cleanup(prefix)
                return PublishedVersion(entry["version"], updated["generation"])
        raise CheckpointConflict("Checkpoint manifest changed repeatedly; training update was not published")

    def _cleanup(self, prefix):
        try:
            manifest, etag = self._manifest(prefix)
            deleted = []
            for object_id in manifest["cleanup"]:
                try:
                    self.blobs.delete(f"{prefix}/objects/{object_id}.zip")
                    deleted.append(object_id)
                except Exception:
                    _LOGGER.warning("Could not remove retired checkpoint %s", object_id, exc_info=True)
            if deleted:
                updated = dict(manifest)
                updated["cleanup"] = [item for item in manifest["cleanup"] if item not in deleted]
                try:
                    self._write(prefix, updated, etag)
                except CheckpointConflict:
                    pass
        except Exception:
            _LOGGER.warning("Could not complete checkpoint cleanup for %s", prefix, exc_info=True)

    def bootstrap(self, model_path, weights_path, validate, empty=False):
        prefix = _prefix(model_path)
        root = _checkpoint_root(model_path)
        local_manifest = _read_manifest(root)
        if local_manifest is None:
            _require_no_unpublished_versions(root)
        model = Path(model_path)
        weights = Path(weights_path)
        if empty:
            if local_manifest or model.exists() or weights.exists():
                raise ValueError("Cannot initialize an empty checkpoint over existing local models")
            pairs = []
            high_water = 0
        elif local_manifest:
            pairs = [(version, *_checkpoint_files(root, version)) for version in local_manifest["history"]]
            high_water = _next_version(root, local_manifest) - 1
        else:
            if not model.is_file() or not weights.is_file():
                raise FileNotFoundError("Both legacy model and target weights files are required")
            pairs = [("v1", str(model), str(weights))]
            high_water = 1
        try:
            self._manifest(prefix)
        except CheckpointNotFound:
            pass
        else:
            raise CheckpointConflict("Blob checkpoint manifest already exists")
        history = []
        with tempfile.TemporaryDirectory(prefix="model-bootstrap-") as directory:
            for version, model_file, weights_file in pairs:
                validate(model_file, weights_file)
                history.append(self._upload_pair(prefix, version, model_file, weights_file, directory))
            self._write(prefix, {
                "schema": 1, "generation": uuid4().hex, "high_water": high_water,
                "history": history, "cleanup": [],
            })

    def rollback(self, model_path, version):
        prefix = _prefix(model_path)
        manifest, etag = self._manifest(prefix)
        entry = next((item for item in manifest["history"] if item["version"] == version), None)
        if entry is None:
            raise ValueError("Requested version is not a retained checkpoint")
        with tempfile.TemporaryDirectory(prefix="model-rollback-") as directory:
            self._download_pair(prefix, entry, directory)
        updated = dict(manifest)
        updated["generation"] = uuid4().hex
        updated["history"] = [entry, *(item for item in manifest["history"] if item != entry)]
        self._write(prefix, updated, etag)


def _validate_with_keras(model_path, weights_path):
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    if getattr(model, "optimizer", None) is None:
        raise ValueError("Loaded model has no optimizer")
    target = tf.keras.models.clone_model(model)
    target.load_weights(weights_path)
    if [tuple(weight.shape) for weight in model.weights] != [tuple(weight.shape) for weight in target.weights]:
        raise ValueError("Model and target network architectures do not match")


def configured_models():
    from app.config.env_config import ATTARI_PRO_MODEL, ATTARI_PRO_WEIGHTS_PATH
    from app.config.ml_env_config import ENVIRONMENTS
    candidates = [
        (name, config.model_path, config.weights_path)
        for name, config in ENVIRONMENTS.items()
    ]
    candidates.append(("Atari Pro / Human", ATTARI_PRO_MODEL, ATTARI_PRO_WEIGHTS_PATH))
    models = {}
    for name, model_path, weights_path in candidates:
        if not model_path and not weights_path:
            continue
        if not model_path or not weights_path:
            raise ValueError(f"Both model and weights paths must be configured for {name}")
        prefix = _prefix(model_path)
        if prefix in models and _prefix(models[prefix]["weights_path"]) != _prefix(weights_path):
            raise ValueError(f"Conflicting weights paths configured for {model_path}")
        models[prefix] = {
            "name": name, "model_path": model_path, "weights_path": weights_path, "prefix": prefix,
        }
    return list(models.values())


def _status(store, model_path):
    manifest, _ = store._manifest(_prefix(model_path))
    return {
        "model_path": model_path,
        "current": manifest["history"][0]["version"] if manifest["history"] else None,
        "history": [entry["version"] for entry in manifest["history"]],
        "high_water": manifest["high_water"],
        "pending_cleanup": len(manifest["cleanup"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Offline Azure Blob checkpoint administration")
    subparsers = parser.add_subparsers(dest="action", required=True)
    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument("model_path")
    bootstrap.add_argument("weights_path")
    bootstrap.add_argument("--empty", action="store_true")
    bootstrap.add_argument("--writers-stopped", action="store_true", required=True,
                           help="Confirm all applications and jobs writing the source models are stopped")
    rollback = subparsers.add_parser("rollback")
    rollback.add_argument("model_path")
    rollback.add_argument("version")
    status = subparsers.add_parser("status")
    status.add_argument("model_path")
    subparsers.add_parser("inventory", help="List configured model paths without contacting Blob storage")
    subparsers.add_parser("status-all", help="Check Blob manifests for every configured model")
    args = parser.parse_args()
    if args.action == "inventory":
        print(json.dumps(configured_models(), indent=2))
        return
    with blob_store_from_environment() as store:
        if args.action == "bootstrap":
            store.bootstrap(args.model_path, args.weights_path, _validate_with_keras, args.empty)
        elif args.action == "rollback":
            store.rollback(args.model_path, args.version)
        elif args.action == "status-all":
            print(json.dumps([_status(store, model["model_path"]) for model in configured_models()], indent=2))
        else:
            print(json.dumps(_status(store, args.model_path), indent=2))


if __name__ == "__main__":
    main()
