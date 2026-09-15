from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Event

import pytest

import scripts.replay_manifest_backup as backup
from test_replay_manifest_backup import _database


def test_capture_blocks_concurrent_retention_until_consumer_exits(tmp_path: Path):
    root = tmp_path / "run"
    _database(root, "before")
    attempted = Event()

    def rotate() -> Path:
        with pytest.raises(backup.BackupLockBusy):
            backup.create_backup(root, retain=1, blocking=False)
        attempted.set()
        return backup.create_backup(root, retain=1)

    with ThreadPoolExecutor(max_workers=1) as workers:
        with backup.captured_backup_with_evidence(root, retain=1) as captured:
            path, evidence = captured
            content = path.read_bytes()
            future = workers.submit(rotate)
            assert attempted.wait(timeout=5)
            assert not future.done()
            assert path.read_bytes() == content
            assert evidence["sha256"] == hashlib.sha256(content).hexdigest()
            assert evidence["bytes"] == len(content)
        rotated = future.result(timeout=5)

    assert rotated.is_file()
    assert rotated != path
    assert not path.exists()


def test_capture_does_not_hold_a_training_writer_or_reader_transaction(tmp_path: Path):
    root = tmp_path / "run"
    _database(root, "before")

    with backup.captured_backup_with_evidence(root, retain=1) as (path, _):
        manifest = root / "replay" / "manifest.sqlite3"
        with closing(sqlite3.connect(manifest, timeout=0)) as writer:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute("UPDATE state SET value='after'")
            writer.commit()
            busy, _, _ = writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            assert busy == 0
        with closing(sqlite3.connect(path)) as reader:
            assert reader.execute("SELECT value FROM state").fetchone()[0] == "before"


@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, KeyboardInterrupt, asyncio.CancelledError, GeneratorExit],
)
def test_consumer_failure_releases_backup_lock(tmp_path: Path, error_type):
    root = tmp_path / "run"
    _database(root, "before")

    with pytest.raises(error_type):
        with backup.captured_backup_with_evidence(root, retain=1):
            raise error_type("consumer interrupted")

    assert backup.create_backup(root, retain=1, blocking=False).is_file()


@pytest.mark.parametrize("stage", ["_create_backup_locked", "_backup_evidence_locked"])
def test_capture_failure_releases_backup_lock(tmp_path: Path, monkeypatch, stage: str):
    root = tmp_path / "run"
    _database(root, "before")

    def fail(*args, **kwargs):
        raise RuntimeError("capture failed")

    with monkeypatch.context() as scoped:
        scoped.setattr(backup, stage, fail)
        with pytest.raises(RuntimeError, match="capture failed"):
            with backup.captured_backup_with_evidence(root, retain=1):
                pytest.fail("capture failure must precede yield")

    assert backup.create_backup(root, retain=1, blocking=False).is_file()
