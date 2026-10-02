"""Committed continuation inputs remain in recovery backup closure."""

import hashlib
import json
import shutil

import pytest

from deltreltrain.runtime import atomic_json
from deltreltrain.strength_recovery import FORMAT, PLAN_NAME, digest
from deltreltrain.strength_recovery_archive import (
    CONTINUATION_FORMAT,
    CONTINUATION_INPUT,
    CONTINUATION_PLAN,
    INSTALLED_PROFILE,
    PROVENANCE,
    backup_files,
    preserve_provenance,
    validate_archive,
)


def pin(path):
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def sealed(path, body):
    value = {**body, "plan_sha256": digest(body)}
    atomic_json(path, value)
    return value


def make_run(root, *, explicit=True, continuation=True):
    root.mkdir()
    source = root.parent / "source.yaml"
    source.write_text("kind: initial-recovery\n")
    binary = root.parent / "checkpoint.pt"
    binary.write_bytes(b"binary" * 200000)
    body = {
        "format": FORMAT,
        "schema_version": 1,
        "run_root": str(root),
        "source_pins": [pin(binary)],
        "implementation_pins": [],
        "profile": pin(source),
        "schedule_seconds": [7200, 21600, 43200],
    }
    if explicit:
        body["installed_profile_name"] = INSTALLED_PROFILE
    plan = sealed(root / PLAN_NAME, body)
    preserve_provenance(root, plan)
    shutil.copyfile(source, root / INSTALLED_PROFILE)
    (root / "profile-strength-continuation.yaml").write_text("kind: continuous\n")
    if continuation:
        target = root / CONTINUATION_INPUT
        target.write_text("kind: continuous\n")
        sealed(
            root / CONTINUATION_PLAN,
            {
                "format": CONTINUATION_FORMAT,
                "schema_version": 1,
                "recovery_plan_sha256": plan["plan_sha256"],
                "target_profile": pin(target),
                "target_profile_name": "profile-strength-continuation.yaml",
                "provisioned_gpus": 8,
                "candidate_interval_examples": 3000000,
                "implementation": {
                    "relative_path": "scripts/run_strength_recovery_continuation.py",
                    "sha256": "code",
                    "bytes": 1,
                },
                "source_commit": "test-source",
            },
        )
    return plan


@pytest.mark.parametrize("explicit", [False, True])
def test_continuation_keeps_original_profile_and_relocates_without_binary_reads(
    tmp_path, explicit
):
    root = tmp_path / "run"
    make_run(root, explicit=explicit)
    expected = backup_files(root)
    assert {
        PLAN_NAME,
        PROVENANCE,
        CONTINUATION_PLAN,
        CONTINUATION_INPUT,
        INSTALLED_PROFILE,
    } <= expected.keys()
    restored = tmp_path / "restore"
    shutil.copytree(root, restored)
    assert backup_files(restored) == expected

    def read(logical):
        assert logical.endswith(".json"), "binary/YAML payload must use fingerprint"
        return (restored / logical).read_bytes()

    def fingerprint(logical):
        path = restored / logical
        return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size

    assert (
        validate_archive(read=read, available=set(expected), fingerprint=fingerprint)
        == expected
    )


@pytest.mark.parametrize("name", [INSTALLED_PROFILE, CONTINUATION_INPUT])
@pytest.mark.parametrize("mutation", ["missing", "corrupt", "symlink"])
def test_rejects_missing_changed_or_symlinked_dependencies(tmp_path, name, mutation):
    root = tmp_path / "run"
    make_run(root)
    path = root / name
    path.unlink()
    if mutation == "corrupt":
        path.write_text("different: bytes\n")
    elif mutation == "symlink":
        path.symlink_to(root / "profile-strength-continuation.yaml")
    with pytest.raises(ValueError):
        backup_files(root)


@pytest.mark.parametrize(
    "change", ["hash", "recovery", "format", "escape", "other-input"]
)
def test_rejects_invalid_continuation_plan_links_and_paths(tmp_path, change):
    root = tmp_path / "run"
    make_run(root)
    path = root / CONTINUATION_PLAN
    body = json.loads(path.read_text())
    body.pop("plan_sha256")
    if change == "hash":
        body["plan_sha256"] = "invalid"
        atomic_json(path, body)
    else:
        if change == "recovery":
            body["recovery_plan_sha256"] = "wrong-recovery"
        elif change == "format":
            body["format"] = "other-format"
        else:
            body["target_profile"]["path"] = str(
                tmp_path / CONTINUATION_INPUT
                if change == "escape"
                else root / "profile-strength-continuation.yaml"
            )
        sealed(path, body)
    with pytest.raises(ValueError):
        backup_files(root)


def test_legacy_plan_keeps_old_dependency_set_until_continuation_commits(tmp_path):
    root = tmp_path / "run"
    make_run(root, explicit=False, continuation=False)
    expected = backup_files(root)
    assert INSTALLED_PROFILE not in expected
    (root / CONTINUATION_INPUT).write_text("not-yet-pinned: true\n")
    assert backup_files(root) == expected


def test_orphan_continuation_fails_closed(tmp_path):
    atomic_json(tmp_path / CONTINUATION_PLAN, {})
    with pytest.raises(ValueError, match="lack their immutable plan"):
        backup_files(tmp_path)


def test_explicit_original_profile_required_before_continuation(tmp_path):
    root = tmp_path / "run"
    make_run(root, continuation=False)
    assert INSTALLED_PROFILE in backup_files(root)
    (root / INSTALLED_PROFILE).unlink()
    with pytest.raises(ValueError, match="missing or unsafe"):
        backup_files(root)
