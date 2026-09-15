from __future__ import annotations

import asyncio
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

import scripts.training_disaster_recovery as recovery
from test_training_disaster_recovery import _fixture, _snapshot


def _next_document(previous: Path) -> tuple[Path, bytes]:
    payload = json.loads(previous.read_bytes())
    payload["created_ns"] += 1
    data = recovery._canonical_json(payload)
    digest = hashlib.sha256(data).hexdigest()
    return previous.with_name(f"{payload['created_ns']}-{digest}.json"), data


def test_semantic_failure_never_publishes_candidate_and_preserves_last_good(tmp_path):
    case = _fixture(tmp_path)
    root = tmp_path / "backup"
    first = _snapshot(case, root)
    latest = (first.parent / "latest.json", root / "latest.json")
    retained = {
        p: p.read_bytes()
        for p in (*latest, first, recovery._snapshot_commit_path(first))
    }
    before_documents = set(first.parent.glob("*.json"))
    cadence_path = case.root / "learner/cadence.json"
    cadence = json.loads(cadence_path.read_bytes())
    cadence["candidate_examples"] = 80
    cadence_path.write_bytes(recovery._canonical_json(cadence))

    with pytest.raises(recovery.DisasterRecoveryError, match="ahead"):
        _snapshot(case, root)

    assert set(first.parent.glob("*.json")) == before_documents
    assert not list(first.parent.glob(".*.tmp"))
    for path, data in retained.items():
        assert path.read_bytes() == data
    assert recovery.verify_snapshot(root / "latest.json")["snapshot"] == str(first)
    assert recovery.garbage_collect(root)["snapshots"] == 1
    restored = recovery.restore_snapshot(root / "latest.json", tmp_path / "restored")
    assert (restored / "run.json").is_file()


@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, KeyboardInterrupt, asyncio.CancelledError, GeneratorExit],
)
def test_validation_base_exception_never_publishes_candidate(
    tmp_path, monkeypatch, error_type
):
    case = _fixture(tmp_path)
    root = tmp_path / "backup"
    first = _snapshot(case, root)
    path, data = _next_document(first)
    pointers = (first.parent / "latest.json", root / "latest.json")
    before = {p: p.read_bytes() for p in (*pointers, first)}

    def interrupt(*args, **kwargs):
        assert not path.exists()
        assert kwargs["_staged_document"].read_bytes() == data
        raise error_type("validation interrupted")

    monkeypatch.setattr(recovery, "_verify_snapshot_document", interrupt)
    with recovery._backup_lock(root), pytest.raises(error_type):
        recovery._publish_verified_snapshot(path, data, root)

    assert not path.exists()
    assert not list(first.parent.glob(".*.tmp"))
    for prior, content in before.items():
        assert prior.read_bytes() == content


@pytest.mark.parametrize("same_bytes", [True, False])
def test_snapshot_collision_cannot_delete_or_overwrite_preexisting_document(
    tmp_path, same_bytes
):
    case = _fixture(tmp_path)
    root = tmp_path / "backup"
    first = _snapshot(case, root)
    content = first.read_bytes()
    supplied = content if same_bytes else b"different content"

    with (
        recovery._backup_lock(root),
        pytest.raises(recovery.DisasterRecoveryError),
    ):
        recovery._publish_verified_snapshot(
            first,
            supplied,
            root,
        )

    assert first.read_bytes() == content
    assert recovery._snapshot_commit_path(first).is_file()
    assert recovery.verify_snapshot(root / "latest.json")["snapshot"] == str(first)


def test_final_document_is_absent_until_staging_validation_finishes(
    tmp_path, monkeypatch
):
    case = _fixture(tmp_path)
    root = tmp_path / "backup"
    first = _snapshot(case, root)
    path, data = _next_document(first)
    original = recovery._verify_snapshot_document
    validating = Event()
    resume = Event()

    def paused(candidate, *args, **kwargs):
        if candidate == path and kwargs.get("_staged_document") is not None:
            assert kwargs["_staged_document"].read_bytes() == data
            validating.set()
            assert resume.wait(timeout=5)
        return original(candidate, *args, **kwargs)

    def publish():
        with recovery._backup_lock(root):
            return recovery._publish_verified_snapshot(path, data, root)

    monkeypatch.setattr(recovery, "_verify_snapshot_document", paused)
    with ThreadPoolExecutor(max_workers=1) as workers:
        future = workers.submit(publish)
        try:
            assert validating.wait(timeout=5)
            assert not path.exists()
            assert not recovery._snapshot_commit_path(path).exists()
            assert recovery.verify_snapshot(root / "latest.json")["snapshot"] == str(
                first
            )
        finally:
            resume.set()
        verified = future.result(timeout=5)

    assert verified.path == path
    assert path.read_bytes() == data
    assert not list(first.parent.glob(".*.tmp"))
    assert recovery.verify_snapshot(path)["status"] == "ok"
    assert recovery.verify_snapshot(root / "latest.json")["snapshot"] == str(first)


def test_staging_read_does_not_relax_public_verification_or_final_identity(tmp_path):
    case = _fixture(tmp_path)
    root = tmp_path / "backup"
    first = _snapshot(case, root)
    path, data = _next_document(first)
    staging = first.parent / ".staging.tmp"
    staging.write_bytes(data)

    with pytest.raises(recovery.DisasterRecoveryError, match="cannot inspect"):
        recovery.verify_snapshot(path)
    verified = recovery._verify_snapshot_document(path, root, _staged_document=staging)
    assert verified.path == path
    with pytest.raises(recovery.DisasterRecoveryError, match="immutable identity"):
        recovery._verify_snapshot_document(first, root, _staged_document=staging)
    with pytest.raises(recovery.DisasterRecoveryError, match="immutable identity"):
        recovery._verify_snapshot_document(
            path, root / "other", _staged_document=staging
        )


def test_collision_created_during_validation_remains_untouched(tmp_path, monkeypatch):
    case = _fixture(tmp_path)
    root = tmp_path / "backup"
    first = _snapshot(case, root)
    path, data = _next_document(first)
    original = recovery._verify_snapshot_document

    def collision(*args, **kwargs):
        verified = original(*args, **kwargs)
        path.write_bytes(b"pre-existing collision")
        return verified

    monkeypatch.setattr(recovery, "_verify_snapshot_document", collision)
    with (
        recovery._backup_lock(root),
        pytest.raises(recovery.DisasterRecoveryError, match="existing snapshot"),
    ):
        recovery._publish_verified_snapshot(path, data, root)

    assert path.read_bytes() == b"pre-existing collision"
    assert first.is_file()
    assert recovery._snapshot_commit_path(first).is_file()
