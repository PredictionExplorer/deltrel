"""Deterministic replay-retirement races across ledger capture and payload copy."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import sqlite3
import threading

import pytest

import scripts.training_disaster_recovery as recovery
from startrain.replay_store import ReplayStore
from test_training_disaster_recovery import _fixture, _snapshot, _snapshot_payload


def add_newer_shard(fixture):
    with ReplayStore(fixture.root / "replay") as store:
        row = dict(
            store.connection.execute(
                "SELECT * FROM shards ORDER BY id DESC LIMIT 1"
            ).fetchone()
        )
        number = int(row.pop("id")) + 1
        row["created_ns"] += 1
        row["relative_path"] = f"shards/shard-{number:06d}.npz"
        path = fixture.root / "replay" / row["relative_path"]
        path.write_bytes(f"new immutable replay shard {number}".encode())
        row["checksum_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        store.connection.execute(
            f"INSERT INTO shards({','.join(row)}) VALUES({','.join('?' for _ in row)})",
            list(row.values()),
        )
        store.connection.execute(
            "UPDATE run_counters SET committed_samples=committed_samples+?",
            (row["sample_count"],),
        )
    return path


def retire_old_shards(fixture):
    with ReplayStore(fixture.root / "replay") as store:
        return store.collect_garbage(
            run_id="run-disaster-test",
            generation_family="family-disaster-test",
            retain_shards_per_ring=1,
            dry_run=False,
        )


def count_captures(monkeypatch):
    captures = []
    original = recovery.captured_backup_with_evidence

    @contextmanager
    def capture(*args, **kwargs):
        with original(*args, **kwargs) as result:
            captures.append(result[1]["sha256"])
            yield result

    monkeypatch.setattr(recovery, "captured_backup_with_evidence", capture)
    return captures


def intercept_old_shard(monkeypatch, fixture, action):
    original = recovery._SnapshotBuilder.add_shard
    called = False

    def add(builder, source, logical, expected_sha256):
        nonlocal called
        if not called and logical == f"replay/shards/{fixture.shard.name}":
            called = True
            action(builder, source, expected_sha256)
        return original(builder, source, logical, expected_sha256)

    monkeypatch.setattr(recovery._SnapshotBuilder, "add_shard", add)


def ledger_rows(snapshot, backup):
    entry = _snapshot_payload(snapshot)["catalog"]["replay/manifest.sqlite3"]
    path = recovery._object_path(backup, entry["sha256"])
    with sqlite3.connect(
        f"{path.as_uri()}?mode=ro&immutable=1", uri=True
    ) as connection:
        return connection.execute(
            "SELECT relative_path FROM shards WHERE state='ready' ORDER BY id"
        ).fetchall()


@pytest.mark.parametrize("cached", [False, True])
def test_real_gc_after_frozen_ledger_uses_exact_cas_or_retakes(
    tmp_path, monkeypatch, cached
):
    fixture = _fixture(tmp_path)
    backup = tmp_path / "backup"
    if cached:
        _snapshot(fixture, backup)
    newest = add_newer_shard(fixture)
    captures = count_captures(monkeypatch)

    def retire(_builder, source, _checksum):
        assert len(captures) == 1
        assert source == fixture.shard
        assert retire_old_shards(fixture)["deleted_shards"] == 1
        assert not source.exists()

    intercept_old_shard(monkeypatch, fixture, retire)
    snapshot = _snapshot(fixture, backup)
    assert recovery.verify_snapshot(snapshot)["status"] == "ok"
    catalog = _snapshot_payload(snapshot)["catalog"]
    old_logical = f"replay/shards/{fixture.shard.name}"
    new_logical = f"replay/shards/{newest.name}"
    assert new_logical in catalog
    assert (old_logical in catalog) is cached
    assert len(captures) == (1 if cached else 2)
    assert catalog["replay/manifest.sqlite3"]["sha256"] == captures[0 if cached else 1]
    expected = [(f"shards/{newest.name}",)]
    if cached:
        expected.insert(0, (f"shards/{fixture.shard.name}",))
    assert ledger_rows(snapshot, backup) == expected
    restored = recovery.restore_snapshot(
        snapshot, tmp_path / "restored", relocate_profile=True
    )
    assert (restored / new_logical).read_bytes() == newest.read_bytes()
    assert (restored / old_logical).exists() is cached


def test_retirement_between_regular_file_stat_and_descriptor_open_retakes(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path)
    backup = tmp_path / "backup"
    newest = add_newer_shard(fixture)
    captures = count_captures(monkeypatch)
    original_open = recovery.os.open
    retired = False

    def retire_before_open(path, flags, *args, **kwargs):
        nonlocal retired
        if (
            not retired
            and isinstance(path, (str, os.PathLike))
            and Path(path) == fixture.shard
        ):
            # _store_object has already passed its source lstat at this point.
            retired = True
            assert retire_old_shards(fixture)["deleted_shards"] == 1
            assert not fixture.shard.exists()
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(recovery.os, "open", retire_before_open)
    snapshot = _snapshot(fixture, backup)
    assert retired and len(captures) == 2
    assert ledger_rows(snapshot, backup) == [(f"shards/{newest.name}",)]
    assert recovery.verify_snapshot(snapshot)["status"] == "ok"


@pytest.mark.parametrize("state", ["ready", "superseded", "quarantined"])
def test_missing_payload_with_live_ledger_row_never_uses_cached_fallback(
    tmp_path, monkeypatch, state
):
    fixture = _fixture(tmp_path)
    backup = tmp_path / "backup"
    _snapshot(fixture, backup)
    latest = (backup / "latest.json").read_bytes()
    captures = count_captures(monkeypatch)

    def remove(_builder, source, _checksum):
        with sqlite3.connect(fixture.root / "replay/manifest.sqlite3") as connection:
            connection.execute("UPDATE shards SET state=?", (state,))
        source.unlink()

    intercept_old_shard(monkeypatch, fixture, remove)
    with pytest.raises(recovery.DisasterRecoveryError):
        _snapshot(fixture, backup)
    assert len(captures) == 1
    assert (backup / "latest.json").read_bytes() == latest
    with sqlite3.connect(fixture.root / "replay/manifest.sqlite3") as connection:
        assert connection.execute("SELECT state FROM shards").fetchone() == (state,)


def test_retired_source_fallback_rehashes_even_same_size_readonly_cached_objects(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path)
    backup = tmp_path / "backup"
    _snapshot(fixture, backup)
    latest = (backup / "latest.json").read_bytes()
    add_newer_shard(fixture)
    captures = count_captures(monkeypatch)

    def corrupt(builder, _source, checksum):
        assert retire_old_shards(fixture)["deleted_shards"] == 1
        cached = recovery._object_path(builder.backup_root, checksum)
        before = cached.read_bytes()
        cached.chmod(0o644)
        cached.write_bytes(bytes([before[0] ^ 1]) + before[1:])
        cached.chmod(0o444)
        assert cached.stat().st_size == len(before)

    intercept_old_shard(monkeypatch, fixture, corrupt)
    with pytest.raises(recovery.DisasterRecoveryError, match="SHA|checksum|hash"):
        _snapshot(fixture, backup)
    assert len(captures) == 1
    assert (backup / "latest.json").read_bytes() == latest


@pytest.mark.parametrize("fault", ["source_corruption", "unsafe_symlink"])
def test_corruption_and_unsafe_source_paths_are_not_retirement_retries(
    tmp_path, monkeypatch, fault
):
    fixture = _fixture(tmp_path)
    backup = tmp_path / "backup"
    captures = count_captures(monkeypatch)
    if fault == "unsafe_symlink":
        add_newer_shard(fixture)

    def break_source(_builder, source, _checksum):
        if fault == "source_corruption":
            before = source.read_bytes()
            source.write_bytes(bytes([before[0] ^ 1]) + before[1:])
        else:
            assert retire_old_shards(fixture)["deleted_shards"] == 1
            outside = tmp_path / "outside.npz"
            outside.write_bytes(b"outside payload")
            source.symlink_to(outside)

    intercept_old_shard(monkeypatch, fixture, break_source)
    with pytest.raises(recovery.DisasterRecoveryError):
        _snapshot(fixture, backup)
    assert len(captures) == 1
    assert not (backup / "latest.json").exists()


def test_slow_shard_copy_does_not_hold_a_live_ledger_writer_lock(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    backup = tmp_path / "backup"
    copying, release = threading.Event(), threading.Event()
    original = recovery._store_object

    def slow_copy(source, *args, **kwargs):
        if source == fixture.shard:
            copying.set()
            if not release.wait(10):
                raise RuntimeError("test failed to release paused copy")
        return original(source, *args, **kwargs)

    monkeypatch.setattr(recovery, "_store_object", slow_copy)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_snapshot, fixture, backup)
        try:
            assert copying.wait(10)
            # timeout=0 proves immediate writer progress while payload copying
            # is held at the barrier, not merely after the copy eventually ends.
            with sqlite3.connect(
                fixture.root / "replay/manifest.sqlite3",
                timeout=0,
                isolation_level=None,
            ) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO store_metadata(key,value) VALUES('backup-writer-test','committed')"
                )
                connection.execute("COMMIT")
            assert not future.done()
        finally:
            release.set()
        snapshot = future.result(timeout=10)
    assert recovery.verify_snapshot(snapshot)["status"] == "ok"


def test_copy_cancellation_cleans_temporary_objects_releases_locks_and_keeps_latest(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path)
    backup = tmp_path / "backup"
    _snapshot(fixture, backup)
    latest = (backup / "latest.json").read_bytes()
    newest = add_newer_shard(fixture)
    target_inode = newest.stat().st_ino
    original_fdopen = recovery.os.fdopen
    descriptors = []

    class InterruptedStream:
        def __init__(self, stream):
            self.stream = stream
            self.reads = 0

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, size):
            self.reads += 1
            if self.reads == 2:
                raise KeyboardInterrupt("cancel during replay payload copy")
            return self.stream.read(size)

    def interrupt(descriptor, *args, **kwargs):
        stream = original_fdopen(descriptor, *args, **kwargs)
        if args and args[0] == "rb" and os.fstat(descriptor).st_ino == target_inode:
            descriptors.append(descriptor)
            return InterruptedStream(stream)
        return stream

    with monkeypatch.context() as interrupted:
        interrupted.setattr(recovery.os, "fdopen", interrupt)
        with pytest.raises(
            KeyboardInterrupt, match="cancel during replay payload copy"
        ):
            _snapshot(fixture, backup)
    assert descriptors
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert not list(backup.rglob(".object-*.tmp"))
    assert not list(backup.rglob(".snapshot-bytes-*"))
    assert (backup / "latest.json").read_bytes() == latest
    # Both snapshot and replay-ledger backup locks must be reusable afterward.
    complete = _snapshot(fixture, backup)
    assert recovery.verify_snapshot(complete)["status"] == "ok"


def test_repeated_retirement_exhausts_the_bounded_capture_budget(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    backup = tmp_path / "backup"
    captures = count_captures(monkeypatch)
    original = recovery._SnapshotBuilder.add_shard

    def retire_each_time(builder, source, logical, expected_sha256):
        add_newer_shard(fixture)
        assert retire_old_shards(fixture)["deleted_shards"] == 1
        assert not source.exists()
        return original(builder, source, logical, expected_sha256)

    monkeypatch.setattr(recovery._SnapshotBuilder, "add_shard", retire_each_time)
    with pytest.raises(recovery.DisasterRecoveryError):
        _snapshot(fixture, backup)
    assert len(captures) == recovery._SNAPSHOT_CAPTURE_ATTEMPTS == 4
    assert not (backup / "latest.json").exists()
