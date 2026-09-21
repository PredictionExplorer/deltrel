"""Successful manifest verification cached for a single learner's control polls.

Weight loading and recovery keep their independent full verification. This cache
only avoids rehashing unchanged, content-addressed publications to read control
metadata. All three files, including their resolved paths, remain observable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .checkpoint import (
    MODEL_MANIFEST_FORMAT,
    MODEL_POINTER_FORMAT,
    ModelManifest,
    load_model_manifest,
)


class _PublicationChanged(RuntimeError):
    """An observed replacement invalidated a verification attempt."""


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    path: Path
    resolved: Path
    signature: tuple[int, int, int, int, int]

    @classmethod
    def capture(cls, path: Path) -> _FileIdentity:
        resolved = path.resolve(strict=True)
        stat = path.stat()
        if path.resolve(strict=True) != resolved:
            raise _PublicationChanged(
                "model publication changed during path resolution"
            )
        return cls(
            path,
            resolved,
            (
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            ),
        )

    def unchanged(self) -> bool:
        try:
            return self == self.capture(self.path)
        except (OSError, RuntimeError):
            return False


@dataclass(frozen=True, slots=True)
class _Dependencies:
    source: _FileIdentity
    artifact: _FileIdentity
    checkpoint: _FileIdentity

    def unchanged(self) -> bool:
        return all(
            item.unchanged() for item in (self.source, self.artifact, self.checkpoint)
        )


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read model publication {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"model publication must be an object: {path}")
    return value


def _referenced_path(value: object, *, parent: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"model publication {name} path is invalid")
    path = Path(value)
    return path if path.is_absolute() else parent / path


def _discover_dependencies(source: Path) -> _Dependencies:
    # Discover only filenames from the small JSON files. Their content is not
    # trusted until the existing full verifier succeeds below. Capture each
    # identity BEFORE reading it, so a verified old file cannot acquire the
    # signature of a replacement installed during verification.
    source_identity = _FileIdentity.capture(source)
    payload = _json_object(source)
    if payload.get("format") == MODEL_POINTER_FORMAT:
        artifact_path = _referenced_path(
            payload.get("manifest"), parent=source.parent, name="manifest"
        )
        artifact_identity = _FileIdentity.capture(artifact_path)
        payload = _json_object(artifact_identity.resolved)
        checkpoint_parent = artifact_identity.resolved.parent
    elif payload.get("format") == MODEL_MANIFEST_FORMAT:
        artifact_identity = source_identity
        checkpoint_parent = source.parent
    else:
        raise ValueError("legacy mutable model manifests are not supported")
    checkpoint_path = _referenced_path(
        payload.get("checkpoint"), parent=checkpoint_parent, name="checkpoint"
    )
    dependencies = _Dependencies(
        source_identity, artifact_identity, _FileIdentity.capture(checkpoint_path)
    )
    if not dependencies.unchanged():
        raise _PublicationChanged(
            "model publication changed during dependency discovery"
        )
    return dependencies


class ControlManifestCache:
    """Single-threaded cache for the learner's two model publication pointers."""

    def __init__(self) -> None:
        self._entries: dict[Path, tuple[_Dependencies, ModelManifest]] = {}

    def load(self, path: str | Path) -> ModelManifest:
        source = Path(path).absolute()
        # Evict before any operation that may fail. A failed replacement must
        # never expose the formerly verified publication as a fallback.
        entry = self._entries.pop(source, None)
        if entry is not None and entry[0].unchanged():
            self._entries[source] = entry
            return entry[1]
        for attempt in range(2):
            try:
                dependencies = _discover_dependencies(source)
                manifest = load_model_manifest(source)
                artifact = manifest.artifact_manifest or manifest.path
                if (
                    not dependencies.unchanged()
                    or artifact.resolve(strict=True) != dependencies.artifact.resolved
                    or manifest.checkpoint.resolve(strict=True)
                    != dependencies.checkpoint.resolved
                ):
                    raise _PublicationChanged(
                        "model publication changed during verification"
                    )
            except _PublicationChanged:
                # An ordinary atomic promotion may overlap the first hash.
                # Verify its successor once; persistent churn still fails closed.
                if attempt == 0:
                    continue
                raise
            self._entries[source] = (dependencies, manifest)
            return manifest
        raise AssertionError("unreachable manifest verification retry")
