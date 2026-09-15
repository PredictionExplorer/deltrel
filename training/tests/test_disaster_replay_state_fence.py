from __future__ import annotations

import os
import sqlite3

import pytest

from scripts import training_disaster_recovery as recovery
from test_training_disaster_recovery import _fixture, _snapshot


def fence_fixture(tmp_path):
    root = tmp_path / "run"
    replay = root / "replay"
    replay.mkdir(parents=True)
    profile = root / "profile.yaml"
    profile.write_text("{}\n")
    manifest = replay / "manifest.sqlite3"
    with sqlite3.connect(manifest) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, value TEXT)")
    return root, profile, manifest


def test_replay_fence_ignores_ordinary_wal_writes_and_main_database_growth(tmp_path):
    root, profile, manifest = fence_fixture(tmp_path)
    before = recovery._capture_state_fence(root, profile)
    with sqlite3.connect(manifest) as connection:
        connection.execute("INSERT INTO items(value) VALUES (?)", ("v" * 100_000,))
        connection.commit()
        assert recovery._capture_state_fence(root, profile) == before
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert recovery._capture_state_fence(root, profile) == before
    assert before[str(manifest)] == (manifest.stat().st_dev, manifest.stat().st_ino)


def test_same_byte_database_replacement_changes_fence_even_with_restored_mtime(
    tmp_path,
):
    root, profile, manifest = fence_fixture(tmp_path)
    before = recovery._capture_state_fence(root, profile)
    metadata = manifest.stat()
    replacement = manifest.with_name("replacement.sqlite3")
    replacement.write_bytes(manifest.read_bytes())
    os.utime(replacement, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    os.replace(replacement, manifest)
    after = recovery._capture_state_fence(root, profile)
    assert before[str(manifest)] != after[str(manifest)]


def test_restore_marker_creation_change_and_removal_are_fenced(tmp_path):
    root, profile, _ = fence_fixture(tmp_path)
    marker = root / "replay/restore-marker.json"
    absent = recovery._capture_state_fence(root, profile)
    assert absent[str(marker)] == ()
    marker.write_text('{"restored":1}\n')
    created = recovery._capture_state_fence(root, profile)
    assert created != absent
    marker.write_text('{"restored":2,"checkpoint":3}\n')
    assert recovery._capture_state_fence(root, profile) != created
    marker.unlink()
    assert recovery._capture_state_fence(root, profile) == absent


@pytest.mark.parametrize(
    "logical", ["replay", "replay/manifest.sqlite3", "replay/restore-marker.json"]
)
@pytest.mark.parametrize("broken", [False, True])
def test_replay_fence_rejects_symlinked_state_even_when_target_is_missing(
    tmp_path, logical, broken
):
    root, profile, _ = fence_fixture(tmp_path)
    path = root / logical
    target = tmp_path / "elsewhere"
    if logical == "replay":
        path.rename(target)
        if broken:
            target = tmp_path / "missing"
        path.symlink_to(target, target_is_directory=True)
    else:
        if path.exists():
            path.unlink()
        if not broken:
            target.write_text("same bytes elsewhere")
        path.symlink_to(target)
    with pytest.raises(recovery.DisasterRecoveryError, match="fence.*unsafe"):
        recovery._capture_state_fence(root, profile)


@pytest.mark.parametrize("change", ["database", "restore_marker"])
def test_snapshot_recaptures_when_replay_state_is_replaced_during_capture(
    tmp_path, monkeypatch, change
):
    run = _fixture(tmp_path)
    original_collect = recovery._collect_payloads
    calls = 0

    def changed_capture(*args, **kwargs):
        nonlocal calls
        result = original_collect(*args, **kwargs)
        calls += 1
        if calls == 1:
            if change == "database":
                manifest = run.root / "replay/manifest.sqlite3"
                replacement = manifest.with_name("replacement.sqlite3")
                replacement.write_bytes(manifest.read_bytes())
                os.replace(replacement, manifest)
            else:
                (run.root / "replay/restore-marker.json").write_text(
                    '{"restored":true}\n'
                )
        return result

    monkeypatch.setattr(recovery, "_collect_payloads", changed_capture)
    snapshot = _snapshot(run, tmp_path / "backup")
    assert calls == 2
    assert recovery.verify_snapshot(snapshot)["status"] == "ok"
