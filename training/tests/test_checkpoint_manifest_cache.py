from __future__ import annotations

import hashlib
import json
import os
import mmap
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import deltreltrain.checkpoint as checkpoints
import deltreltrain.checkpoint_manifest_cache as caching
from deltreltrain.checkpoint_manifest_cache import ControlManifestCache
from deltreltrain.config import PlateauConfig
from deltreltrain.contracts import FEATURE_SCHEMA_HASH, RULES_HASH_WIRE
from deltreltrain.learner import LearnerLoop
from deltreltrain.model import MODEL_SCHEMA_VERSION


def publication(root: Path, step: int, *, role: str = "candidate") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    content = f"checkpoint {step}".encode()
    checksum = hashlib.sha256(content).hexdigest()
    identity = f"sha256-{checksum}"
    checkpoint = root / f"{identity}.pt"
    checkpoint.write_bytes(content)
    payload = {
        "format": checkpoints.MODEL_MANIFEST_FORMAT,
        "schema_version": checkpoints.MODEL_MANIFEST_VERSION,
        "model_version": identity,
        "model_identity": identity,
        "model_step": step,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": checksum,
        "checkpoint_bytes": len(content),
        "weights": "ema",
        "run_id": "test-run",
        "generation_family": "test-family",
        "rules_hash": RULES_HASH_WIRE,
        "feature_schema_hash": f"{FEATURE_SCHEMA_HASH:016x}",
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "created_ns": step + 1,
    }
    encoded = json.dumps(payload).encode()
    manifest_hash = hashlib.sha256(encoded).hexdigest()
    artifact = root / f"manifest-{manifest_hash}.json"
    artifact.write_bytes(encoded)
    pointer = root / f"{role}.json"
    replacement = root / f"{role}.replacement"
    replacement.write_text(
        json.dumps(
            {
                "format": checkpoints.MODEL_POINTER_FORMAT,
                "schema_version": checkpoints.MODEL_POINTER_VERSION,
                "role": role,
                "manifest": artifact.name,
                "manifest_sha256": manifest_hash,
                "manifest_bytes": len(encoded),
                "model_identity": identity,
                "model_step": step,
                "run_id": "test-run",
                "generation_family": "test-family",
            }
        )
    )
    replacement.replace(pointer)
    return pointer


@pytest.fixture
def hashes(monkeypatch):
    calls = []
    original = checkpoints.sha256_file

    def counted(path):
        if Path(path).suffix == ".pt":
            calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(checkpoints, "sha256_file", counted)
    return calls


@pytest.mark.parametrize("direct", [False, True])
def test_unchanged_publication_reuses_metadata_but_reverifies_payload(
    tmp_path, hashes, direct
):
    pointer = publication(tmp_path, 1)
    if direct:
        pointer = tmp_path / json.loads(pointer.read_text())["manifest"]
    cache = ControlManifestCache()
    first = cache.load(pointer)
    for _ in range(10):
        assert cache.load(pointer) is first
    assert hashes == [first.checkpoint] * 12


def test_historical_planning_reverifies_payloads_even_with_unchanged_metadata(
    tmp_path, hashes
):
    from deltreltrain.historical_evaluation import load_historical_manifests
    from deltreltrain.runtime import RunIdentity

    # Match the current production manifest count without depending on timing
    # or checkpoint size: the important cost is repeated complete-file reads.
    for step in range(42):
        publication(tmp_path, step)
    identity = RunIdentity(tmp_path / "run.json", "test-run", "test-family", 1)
    cache = ControlManifestCache(max_entries=64)
    first = load_historical_manifests(tmp_path, run_identity=identity, cache=cache)
    assert len(first) == 42 and len(hashes) == 84
    for _ in range(5):
        assert (
            load_historical_manifests(tmp_path, run_identity=identity, cache=cache)
            == first
        )
    assert len(hashes) == 294
    checkpoint = next(iter(first.values())).checkpoint
    replacement = checkpoint.with_suffix(".replacement")
    replacement.write_bytes(checkpoint.read_bytes())
    replacement.replace(checkpoint)
    load_historical_manifests(tmp_path, run_identity=identity, cache=cache)
    assert len(hashes) == 337
    stat = checkpoint.stat()
    data = checkpoint.read_bytes()
    checkpoint.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    os.utime(checkpoint, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="SHA-256"):
        load_historical_manifests(tmp_path, run_identity=identity, cache=cache)
    assert len(hashes) == 338  # Failed replacement cannot reuse verified success.


def test_historical_loader_without_cache_retains_full_verification(tmp_path, hashes):
    from deltreltrain.historical_evaluation import load_historical_manifests
    from deltreltrain.runtime import RunIdentity

    publication(tmp_path, 1)
    identity = RunIdentity(tmp_path / "run.json", "test-run", "test-family", 1)
    for _ in range(2):
        assert len(load_historical_manifests(tmp_path, run_identity=identity)) == 1
    assert len(hashes) == 2


def test_historical_shortlist_verifies_only_required_checkpoint_bytes(tmp_path, hashes):
    from deltreltrain.historical_evaluation import load_historical_manifests
    from deltreltrain.runtime import RunIdentity

    for step in range(42):
        publication(tmp_path, step)
    model_identity = f"sha256-{hashlib.sha256(b'checkpoint 41').hexdigest()}"
    required = frozenset({model_identity})
    identity = RunIdentity(tmp_path / "run.json", "test-run", "test-family", 1)
    cache = ControlManifestCache()
    for _ in range(5):
        selected = load_historical_manifests(
            tmp_path, run_identity=identity, cache=cache, required_identities=required
        )
        assert set(selected) == required
    # One eligible checkpoint is read per warm pass; discovery verifies twice.
    # The other 41 checkpoints are never read.
    assert len(hashes) == 6
    manifest = selected[model_identity].artifact_manifest
    assert manifest is not None
    # A corrupt manifest cannot hide the required identity from verification.
    manifest.write_bytes(
        manifest.read_bytes().replace(b'"model_identity"', b'"other_identity"')
    )
    with pytest.raises(ValueError, match="content-address"):
        load_historical_manifests(
            tmp_path, run_identity=identity, cache=cache, required_identities=required
        )
    assert len(hashes) == 6


def test_control_manifest_cache_is_bounded_and_reverifies_evictions(tmp_path, hashes):
    paths = [publication(tmp_path / str(step), step) for step in range(3)]
    cache = ControlManifestCache(max_entries=2)
    cache.load(paths[0])
    cache.load(paths[1])
    cache.load(paths[0])  # Keep zero recent so one is the eviction victim.
    cache.load(paths[2])
    assert len(cache._entries) == 2 and len(hashes) == 7
    cache.load(paths[1])
    assert len(cache._entries) == 2 and len(hashes) == 9


@pytest.mark.parametrize("capacity", [True, 0, -1, 1.5])
def test_control_manifest_cache_rejects_invalid_capacity(capacity):
    with pytest.raises(ValueError, match="positive integer"):
        ControlManifestCache(max_entries=capacity)


def test_pointer_replacement_is_fully_verified(tmp_path, hashes):
    pointer = publication(tmp_path, 1)
    cache = ControlManifestCache()
    old = cache.load(pointer)
    publication(tmp_path, 2)
    new = cache.load(pointer)
    assert (old.model_step, new.model_step) == (1, 2)
    assert hashes == [old.checkpoint, old.checkpoint, new.checkpoint, new.checkpoint]


def test_same_content_checkpoint_replacement_reverifies(tmp_path, hashes):
    cache = ControlManifestCache()
    pointer = publication(tmp_path, 1)
    first = cache.load(pointer)
    original_stat = first.checkpoint.stat()
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(first.checkpoint.read_bytes())
    os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    replacement.replace(first.checkpoint)
    assert cache.load(pointer) == first
    assert hashes == [first.checkpoint] * 4


@pytest.mark.parametrize("file_kind", ["pointer", "manifest", "checkpoint"])
def test_same_path_corruption_fails_closed_and_recovers(tmp_path, hashes, file_kind):
    cache = ControlManifestCache()
    pointer = publication(tmp_path, 1)
    first = cache.load(pointer)
    path = {
        "pointer": pointer,
        "manifest": first.artifact_manifest,
        "checkpoint": first.checkpoint,
    }[file_kind]
    assert path is not None
    content, stat = path.read_bytes(), path.stat()
    path.write_bytes(b"!" * len(content))
    # Metadata alone is insufficient: even ctime can collide on the target host.
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    for _ in range(2):
        with pytest.raises(ValueError):
            cache.load(pointer)
        assert not cache._entries
    path.write_bytes(content)
    assert cache.load(pointer) == first
    assert len(hashes) >= 2


def freeze_stat_signatures(monkeypatch):
    """Model a filesystem returning identical metadata after byte changes."""
    original = caching._FileIdentity.capture
    frozen = {}

    def capture(cls, path, **kwargs):
        identity = original(path, **kwargs)
        signature = frozen.setdefault(identity.path, identity.signature)
        return replace(identity, signature=signature)

    monkeypatch.setattr(caching._FileIdentity, "capture", classmethod(capture))


@pytest.mark.parametrize("file_kind", ["pointer", "manifest", "checkpoint"])
def test_content_corruption_is_detected_when_every_stat_field_collides(
    tmp_path, monkeypatch, file_kind
):
    freeze_stat_signatures(monkeypatch)
    pointer = publication(tmp_path, 1)
    cache = ControlManifestCache()
    first = cache.load(pointer)
    path = {
        "pointer": pointer,
        "manifest": first.artifact_manifest,
        "checkpoint": first.checkpoint,
    }[file_kind]
    assert path is not None
    path.write_bytes(b"!" * path.stat().st_size)
    with pytest.raises(ValueError):
        cache.load(pointer)
    assert not cache._entries


def test_preexisting_writable_mapping_cannot_bypass_checkpoint_integrity(
    tmp_path, monkeypatch
):
    freeze_stat_signatures(monkeypatch)
    pointer = publication(tmp_path, 1)
    artifact = tmp_path / json.loads(pointer.read_text())["manifest"]
    checkpoint = tmp_path / json.loads(artifact.read_text())["checkpoint"]
    cache = ControlManifestCache()
    with checkpoint.open("r+b") as backing:
        with mmap.mmap(backing.fileno(), 0, access=mmap.ACCESS_WRITE) as mapped:
            # Make the page writable before admission without changing bytes.
            # Later stores can evade both stat metadata and inotify events.
            mapped[0] = mapped[0]
            cache.load(pointer)
            mapped[1] ^= 1
            # Keep the mapping and descriptor open: no flush/close side effect
            # is available to rescue a metadata or notification-only cache.
            with pytest.raises(ValueError, match="SHA-256"):
                cache.load(pointer)
            assert not cache._entries


def test_mid_verification_mmap_write_is_not_admitted_with_colliding_stats(
    tmp_path, monkeypatch
):
    freeze_stat_signatures(monkeypatch)
    pointer = publication(tmp_path, 1)
    artifact = tmp_path / json.loads(pointer.read_text())["manifest"]
    checkpoint = tmp_path / json.loads(artifact.read_text())["checkpoint"]
    cache = ControlManifestCache()
    original = caching.load_model_manifest
    with checkpoint.open("r+b") as backing:
        with mmap.mmap(backing.fileno(), 0, access=mmap.ACCESS_WRITE) as mapped:
            mapped[0] = mapped[0]

            def changed_after_load(path):
                manifest = original(path)
                mapped[1] ^= 1
                return manifest

            monkeypatch.setattr(caching, "load_model_manifest", changed_after_load)
            with pytest.raises(ValueError, match="SHA-256"):
                cache.load(pointer)
            assert not cache._entries


def test_deleted_checkpoint_is_not_a_cached_success(tmp_path):
    cache = ControlManifestCache()
    pointer = publication(tmp_path, 1)
    cache.load(pointer).checkpoint.unlink()
    with pytest.raises(OSError):
        cache.load(pointer)
    assert not cache._entries


def test_replacement_during_verification_cannot_admit_stale_manifest(
    tmp_path, monkeypatch, hashes
):
    cache = ControlManifestCache()
    pointer = publication(tmp_path, 1)
    cache.load(pointer)
    publication(tmp_path, 2)
    original = caching.load_model_manifest

    def replace_after_verification(path):
        verified = original(path)
        publication(tmp_path, 3)
        return verified

    monkeypatch.setattr(caching, "load_model_manifest", replace_after_verification)
    with pytest.raises(RuntimeError, match="changed during verification"):
        cache.load(pointer)
    assert not cache._entries
    monkeypatch.setattr(caching, "load_model_manifest", original)
    assert cache.load(pointer).model_step == 3
    assert len(hashes) == 6


@pytest.mark.parametrize("file_kind", ["manifest", "checkpoint"])
def test_dependency_change_during_verification_is_not_cached(
    tmp_path, monkeypatch, file_kind
):
    pointer = publication(tmp_path, 1)
    cache = ControlManifestCache()
    original = caching.load_model_manifest

    def corrupt_after_verification(path):
        verified = original(path)
        target = (
            verified.artifact_manifest
            if file_kind == "manifest"
            else verified.checkpoint
        )
        target.write_bytes(b"corrupt")
        return verified

    monkeypatch.setattr(caching, "load_model_manifest", corrupt_after_verification)
    with pytest.raises(ValueError):
        cache.load(pointer)
    assert not cache._entries
    monkeypatch.setattr(caching, "load_model_manifest", original)
    with pytest.raises(ValueError):
        cache.load(pointer)


def test_pointer_change_during_discovery_fails_closed(tmp_path, monkeypatch):
    pointer = publication(tmp_path, 1)
    original = caching._json_object

    def replace_after_read(path):
        payload = original(path)
        if path == pointer:
            publication(tmp_path, 2)
        return payload

    monkeypatch.setattr(caching, "_json_object", replace_after_read)
    cache = ControlManifestCache()
    with pytest.raises(RuntimeError, match="changed during dependency discovery"):
        cache.load(pointer)
    assert not cache._entries
    monkeypatch.setattr(caching, "_json_object", original)
    assert cache.load(pointer).model_step == 2


def test_one_atomic_promotion_during_verification_retries_latest(
    tmp_path, monkeypatch, hashes
):
    pointer = publication(tmp_path, 1)
    original = caching.load_model_manifest

    def replace_once(path):
        verified = original(path)
        if verified.model_step == 1:
            publication(tmp_path, 2)
        return verified

    monkeypatch.setattr(caching, "load_model_manifest", replace_once)
    cache = ControlManifestCache()
    latest = cache.load(pointer)
    assert latest.model_step == 2
    assert cache.load(pointer) is latest
    assert len(hashes) == 4


def test_symlink_pointer_replacement_is_observed(tmp_path, hashes):
    one = publication(tmp_path / "one", 1)
    two = publication(tmp_path / "two", 2)
    # Use an absolute manifest reference because pointer-relative resolution is
    # intentionally relative to the alias, matching the existing verifier.
    for pointer in (one, two):
        payload = json.loads(pointer.read_text())
        payload["manifest"] = str(pointer.parent / payload["manifest"])
        pointer.write_text(json.dumps(payload))
    alias = tmp_path / "candidate.json"
    alias.symlink_to(one)
    cache = ControlManifestCache()
    assert cache.load(alias).model_step == 1
    alias.unlink()
    alias.symlink_to(two)
    assert cache.load(alias).model_step == 2
    assert len(hashes) == 4


def test_learner_control_polls_reuse_verified_manifests(tmp_path, hashes):
    champion = publication(tmp_path, 1, role="champion")
    candidate = publication(tmp_path, 2)
    loop = object.__new__(LearnerLoop)
    loop.publisher = SimpleNamespace(champion_path=champion, candidate_path=candidate)
    loop.promotion_status_path = tmp_path / "missing-status.json"
    loop.step = 20
    loop.learner_config = SimpleNamespace(max_replay_lag_steps=100)
    loop._last_plateau_reset = None
    configured = PlateauConfig(enabled=True, action="reduce_lr_keep_weights")
    for _ in range(10):
        assert loop._rank_zero_plateau_action(configured) == {"kind": "proceed"}
    assert len(hashes) == 22
    publication(tmp_path, 3)
    assert loop._rank_zero_plateau_action(configured) == {"kind": "proceed"}
    assert len(hashes) == 25
