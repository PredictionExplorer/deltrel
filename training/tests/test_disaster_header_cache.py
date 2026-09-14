from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import os
from pathlib import Path

import pytest

from scripts import training_disaster_recovery as recovery
from test_training_disaster_recovery import _fixture, _publish_latest, _snapshot


def header(root: Path, timestamp: int, entries: int = 1) -> Path:
    run = root / "snapshots/cache-test"
    run.mkdir(parents=True, exist_ok=True)
    payload = {
        "report": recovery.SNAPSHOT_REPORT,
        "schema_version": recovery.SCHEMA_VERSION,
        "run_id": "cache-test",
        "generation_family": "cache-family",
        "created_ns": timestamp,
        "source": {
            "run_root": str(root / "source"),
            "profile": str(root / "source/profile.yaml"),
            "profile_logical_path": "profile.yaml",
            "legacy_initialized_missing": False,
            "replay_backup": {"created_ns": 1, "sha256": "0" * 64, "bytes": 1},
        },
        "catalog": {
            f"replay/shards/{index:08d}.npz": {
                "sha256": "0" * 64,
                "bytes": 1000,
                "kind": "replay-shard",
            }
            for index in range(entries)
        },
    }
    contents = recovery._canonical_json(payload)
    path = run / f"{timestamp}-{hashlib.sha256(contents).hexdigest()}.json"
    path.write_bytes(contents)
    return path


def count_parses(monkeypatch):
    counts = {}
    original = recovery._snapshot_envelope

    def counted(path, root):
        counts[path] = counts.get(path, 0) + 1
        return original(path, root)

    monkeypatch.setattr(recovery, "_snapshot_envelope", counted)
    return counts


def test_operation_reuses_deeply_immutable_headers_and_clears_nested_context(
    monkeypatch, tmp_path
):
    path = header(tmp_path, 1)
    counts = count_parses(monkeypatch)
    assert recovery._SNAPSHOT_HEADER_CACHE.get() is None
    with recovery._snapshot_header_operation():
        first = recovery._snapshot_headers(path.parent, tmp_path)[0]
        with recovery._snapshot_header_operation():
            assert recovery._snapshot_headers(path.parent, tmp_path)[0] is first
        assert counts == {path: 1}
        with pytest.raises(FrozenInstanceError):
            first.sha256 = "changed"
        with pytest.raises(TypeError):
            first.payload["source"]["run_root"] = "changed"
        with pytest.raises(TypeError):
            first.payload["catalog"]["replay/shards/00000000.npz"]["sha256"] = "changed"
        with pytest.raises(TypeError):
            first.catalog["replay/shards/00000000.npz"] = None
        assert recovery._snapshot_headers(path.parent, tmp_path)[0] is first
    assert recovery._SNAPSHOT_HEADER_CACHE.get() is None
    with recovery._snapshot_header_operation():
        assert recovery._snapshot_headers(path.parent, tmp_path)[0] == first
    assert counts == {path: 2}


def test_operation_context_always_cleans_up_after_failure(tmp_path):
    path = header(tmp_path, 1)
    with pytest.raises(RuntimeError, match="simulated failure"):
        with recovery._snapshot_header_operation():
            recovery._snapshot_headers(path.parent, tmp_path)
            raise RuntimeError("simulated failure")
    assert recovery._SNAPSHOT_HEADER_CACHE.get() is None


def test_identical_file_replacement_requires_fresh_validation(monkeypatch, tmp_path):
    path = header(tmp_path, 1)
    counts = count_parses(monkeypatch)
    with recovery._snapshot_header_operation():
        first = recovery._snapshot_headers(path.parent, tmp_path)[0]
        replacement = tmp_path / "replacement"
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
        second = recovery._snapshot_headers(path.parent, tmp_path)[0]
        assert second == first and second is not first
        assert counts == {path: 2}


def test_changed_contents_with_restored_mtime_cannot_reuse_header(
    monkeypatch, tmp_path
):
    path = header(tmp_path, 1)
    counts = count_parses(monkeypatch)
    with recovery._snapshot_header_operation():
        recovery._snapshot_headers(path.parent, tmp_path)
        prior = path.stat()
        contents = path.read_bytes().replace(b"profile.yaml", b"profilx.yaml")
        path.write_bytes(contents)
        os.utime(path, ns=(prior.st_atime_ns, prior.st_mtime_ns))
        assert path.stat().st_size == prior.st_size
        with pytest.raises(recovery.DisasterRecoveryError, match="immutable identity"):
            recovery._snapshot_headers(path.parent, tmp_path)
        assert counts == {path: 2}
        assert recovery._SNAPSHOT_HEADER_CACHE.get() == {}


def test_document_change_during_validation_never_enters_cache(monkeypatch, tmp_path):
    path = header(tmp_path, 1)
    original = recovery._snapshot_envelope

    def changed(document, root):
        result = original(document, root)
        document.write_bytes(document.read_bytes())
        return result

    monkeypatch.setattr(recovery, "_snapshot_envelope", changed)
    with recovery._snapshot_header_operation():
        with pytest.raises(
            recovery.DisasterRecoveryError, match="changed during header"
        ):
            recovery._snapshot_headers(path.parent, tmp_path)
        assert recovery._SNAPSHOT_HEADER_CACHE.get() == {}


def test_deleted_and_symlinked_documents_do_not_gain_cached_authority(tmp_path):
    path = header(tmp_path, 1)
    with recovery._snapshot_header_operation():
        recovery._snapshot_headers(path.parent, tmp_path)
        other = tmp_path / "original.json"
        path.rename(other)
        assert recovery._snapshot_headers(path.parent, tmp_path) == []
        assert recovery._SNAPSHOT_HEADER_CACHE.get() == {}
        path.symlink_to(other)
        with pytest.raises(recovery.DisasterRecoveryError, match="unexpected snapshot"):
            recovery._snapshot_headers(path.parent, tmp_path)


def test_cached_document_cannot_move_to_a_different_backup_namespace(tmp_path):
    root = tmp_path / "backup"
    path = header(root, 1)
    with recovery._snapshot_header_operation():
        recovery._snapshot_headers(path.parent, root)
        moved = tmp_path / "moved-backup"
        root.rename(moved)
        root.symlink_to(moved, target_is_directory=True)
        with pytest.raises(recovery.DisasterRecoveryError, match="backup root"):
            recovery._snapshot_headers(path.parent, root)


def test_directory_membership_latest_pointers_and_commit_markers_remain_live(tmp_path):
    first = header(tmp_path, 1)
    first_snapshot = recovery._snapshot_header(first, tmp_path)
    recovery._publish_snapshot_commit(first_snapshot)
    _publish_latest(tmp_path, first, dict(first_snapshot.payload))
    latest = tmp_path / "latest.json"
    with recovery._snapshot_header_operation():
        assert (
            recovery._resolve_latest(latest, tmp_path, verify_payload=False).path
            == first
        )
        second = header(tmp_path, 2)
        assert (
            recovery._resolve_latest(latest, tmp_path, verify_payload=False).path
            == first
        )
        second_snapshot = recovery._snapshot_header(second, tmp_path)
        recovery._publish_snapshot_commit(second_snapshot)
        with pytest.raises(recovery.DisasterRecoveryError, match="stale"):
            recovery._resolve_latest(latest, tmp_path, verify_payload=False)
        # Existing semantics ignore a malformed commit marker, but never cache
        # its previous valid state or the old/latest pointer's bytes.
        recovery._snapshot_commit_path(second).unlink()
        recovery._snapshot_commit_path(second).write_text("{}\n")
        assert (
            recovery._resolve_latest(latest, tmp_path, verify_payload=False).path
            == first
        )
        latest.write_text("{}\n")
        with pytest.raises(recovery.DisasterRecoveryError, match="fields are invalid"):
            recovery._resolve_latest(latest, tmp_path, verify_payload=False)
        latest.unlink()
        with pytest.raises(recovery.DisasterRecoveryError, match="cannot inspect"):
            recovery._resolve_latest(latest, tmp_path, verify_payload=False)


def test_public_snapshot_operation_parses_old_catalogs_once_and_still_verifies_payloads(
    tmp_path, monkeypatch
):
    run = _fixture(tmp_path)
    root = tmp_path / "backup"
    first = _snapshot(run, root)
    counts = count_parses(monkeypatch)
    second = _snapshot(run, root)
    assert counts[first] == 1
    assert recovery._SNAPSHOT_HEADER_CACHE.get() is None
    counts.clear()
    assert recovery.verify_snapshot(root / "latest.json")["snapshot"] == str(second)
    assert counts[first] == 1
    # The newest header is parsed once for selection and again for actual
    # catalog/payload verification; header reuse does not waive object checks.
    assert counts[second] == 2
    assert recovery._SNAPSHOT_HEADER_CACHE.get() is None
