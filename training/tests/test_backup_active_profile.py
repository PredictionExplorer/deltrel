from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import backup_active_profile as backup


def _register(root: Path, name: str, contents: str) -> Path:
    path = root / name
    path.write_text(contents)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "profile.sha256").write_text(f"{digest}  {name}\n")
    return path


def test_each_backup_follows_the_latest_registered_profile(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    (root / "profile.yaml").write_text("obsolete")
    destination = tmp_path / "nfs" / "backup"
    mount = destination.parent
    observed = []

    def capture(run_root, profile, backup_root, **kwargs):
        observed.append(profile)
        assert run_root == root and backup_root == destination
        assert kwargs == {"expected_backup_mount": mount, "replay_backup_retain": 3}
        return destination / "snapshot.json"

    monkeypatch.setattr(backup, "create_snapshot", capture)
    first = _register(root, "profile-screen.yaml", "screen")
    assert backup.backup_active_profile(root, destination, expected_backup_mount=mount) == destination / "snapshot.json"
    second = _register(root, "profile-continuation.yaml", "continued")
    backup.backup_active_profile(root, destination, expected_backup_mount=mount)
    assert observed == [first, second]


def test_corrupt_authority_cannot_publish_using_legacy_fallback(tmp_path, monkeypatch):
    (tmp_path / "profile.yaml").write_text("obsolete")
    registered = _register(tmp_path, "profile-screen.yaml", "screen")
    registered.write_text("tampered")

    def unexpected(*args, **kwargs):
        pytest.fail("corrupt active profile reached snapshot publication")

    monkeypatch.setattr(backup, "create_snapshot", unexpected)
    with pytest.raises(ValueError, match="checksum"):
        backup.backup_active_profile(tmp_path, tmp_path / "backup", expected_backup_mount=tmp_path)


def test_capture_failure_is_reported_without_retrying_old_profile(tmp_path, monkeypatch, capsys):
    _register(tmp_path, "profile-screen.yaml", "screen")
    calls = []

    def changed(*args, **kwargs):
        calls.append(args)
        raise backup.DisasterRecoveryError("active profile changed during capture")

    monkeypatch.setattr(backup, "create_snapshot", changed)
    assert backup.main([
        "--run-root", str(tmp_path), "--backup-root", str(tmp_path / "backup"),
        "--expected-backup-mount", str(tmp_path),
    ]) == 2
    assert len(calls) == 1
    assert '"status": "error"' in capsys.readouterr().err
