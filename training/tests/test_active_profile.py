import hashlib
import json
import os
from pathlib import Path

import pytest

from test_strength_recovery import source as source
from test_strength_recovery_continuation import screen as screen
from test_monitor_run import _healthy_dependencies
from scripts import active_profile as active
from scripts import monitor_run as monitor
from scripts import prepare_strength_recovery as preparation
from scripts import run_strength_recovery_continuation as continuation
from startrain.runtime import atomic_json
from startrain.strength_recovery import PLAN_NAME, digest


def register(root, profile):
    checksum = hashlib.sha256(profile.read_bytes()).hexdigest()
    (root / "profile.sha256").write_text(f"{checksum}  {profile.name}\n")


@pytest.fixture
def recovery(source, tmp_path):
    root, output = tmp_path / "recovery", tmp_path / "prepared"
    preparation.prepare(
        source_profile=source.profile,
        destination=root,
        output=output,
        muon_lr=0.0005,
        adamw_lr=0.0000075,
        warmup_steps=500,
    )
    preparation.apply(output / PLAN_NAME)
    return root


def test_legacy_profile_fallback_and_explicit_profile(tmp_path):
    legacy = tmp_path / "profile.yaml"
    legacy.write_text("learner: {}\n")
    selected = active.resolve_active_profile(tmp_path)
    assert selected.path == legacy and selected.source == "legacy_fallback"
    assert selected.expected_sha256 is None
    other = tmp_path / "alternate.yaml"
    other.write_text("model: {}\n")
    assert active.resolve_active_profile(tmp_path, other).path == other
    assert (
        active.resolve_active_profile(tmp_path / "absent", allow_missing=True).contents
        is None
    )


def test_each_resolution_follows_current_registered_profile(tmp_path):
    first, second = tmp_path / "screen.yaml", tmp_path / "continuation.yaml"
    first.write_text("stage: screen\n")
    second.write_text("stage: continuation\n")
    register(tmp_path, first)
    assert active.resolve_active_profile(tmp_path).path == first
    register(tmp_path, second)
    assert active.resolve_active_profile(tmp_path).path == second
    with pytest.raises(ValueError, match="explicit profile differs"):
        active.resolve_active_profile(tmp_path, first)


def test_invalid_authority_never_uses_legacy_fallback(tmp_path):
    (tmp_path / "profile.yaml").write_text("valid legacy bytes")
    (tmp_path / "profile.sha256").write_text("broken authority")
    with pytest.raises(ValueError, match="malformed"):
        active.resolve_active_profile(tmp_path)


def test_equal_size_rewrite_with_restored_mtime_is_detected(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_bytes(b"aaaa")
    register(tmp_path, path)
    active.resolve_active_profile(tmp_path)
    before = path.stat()
    path.write_bytes(b"bbbb")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="disagrees"):
        active.resolve_active_profile(tmp_path)


def test_authority_rejects_escape_and_symbolic_links(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    external = tmp_path / "outside.yaml"
    external.write_text("{}")
    checksum = hashlib.sha256(external.read_bytes()).hexdigest()
    authority = root / "profile.sha256"
    authority.write_text(f"{checksum} {external}\n")
    with pytest.raises(ValueError, match="escaped"):
        active.resolve_active_profile(root)
    (root / "link.yaml").symlink_to(external)
    authority.write_text(f"{checksum} link.yaml\n")
    with pytest.raises(ValueError, match="symbolic"):
        active.resolve_active_profile(root)


def test_registered_recovery_is_admitted_without_binary_provenance_reads(
    recovery, monkeypatch
):
    original = Path.open

    def guarded(path, *args, **kwargs):
        if (
            path.suffix in (".pt", ".sqlite3")
            or "strength-recovery-provenance" in path.parts
        ):
            pytest.fail("monitor attempted to read model/replay/provenance binaries")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    selected = active.resolve_active_profile(recovery)
    config, classification = active.validate_profile_for_monitor(recovery, selected)
    assert classification == "registered_recovery_screen"
    assert config.orchestration.plateau.enabled is False


@pytest.mark.parametrize("rehash", [False, True])
def test_recovery_plan_tamper_is_not_a_generic_validation_bypass(recovery, rehash):
    plan_path = recovery / PLAN_NAME
    plan = json.loads(plan_path.read_text())
    plan["wall_budget_seconds"] = 86400
    if rehash:
        plan["plan_sha256"] = digest(
            {k: v for k, v in plan.items() if k != "plan_sha256"}
        )
    plan_path.chmod(0o644)
    atomic_json(plan_path, plan)
    with pytest.raises(ValueError):
        active.validate_profile_for_monitor(
            recovery, active.resolve_active_profile(recovery)
        )


def test_profile_tamper_is_reported_as_objective_error(recovery, monkeypatch):
    path = recovery / "profile-elo-ablation.yaml"
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b"\n# unregistered change\n")
    _healthy_dependencies(monkeypatch)
    snapshot = monitor.collect_snapshot(recovery)
    assert snapshot["objective_contract"]["validated"] is False
    assert "objective_profile_invalid" in {
        warning["code"] for warning in snapshot["warnings"]
    }


def test_monitor_admits_registered_experiment_and_dynamically_switches_at_handoff(
    screen, monkeypatch
):
    _healthy_dependencies(monkeypatch)
    first = monitor.collect_snapshot(screen)
    assert first["objective_contract"]["validated"] is True
    assert first["objective_contract"]["admission"] == "registered_recovery_screen"
    assert "objective_profile_invalid" not in {row["code"] for row in first["warnings"]}
    plan = continuation.prepare(screen, source_commit="a" * 40)
    target = continuation.handoff(screen, plan)
    second = monitor.collect_snapshot(screen)
    assert second["objective_contract"]["profile"] == str(target)
    assert second["objective_contract"]["validated"] is True
    assert (
        second["objective_contract"]["admission"] == "registered_recovery_continuation"
    )
    stale = monitor.collect_snapshot(
        screen, profile_path=screen / "profile-elo-ablation.yaml"
    )
    assert "objective_profile_invalid" in {row["code"] for row in stale["warnings"]}


def test_continuation_requires_seal_and_migration_receipt(screen):
    plan = continuation.prepare(screen, source_commit="a" * 40)
    continuation.handoff(screen, plan)
    path = screen / "strength-screen-seal.json"
    path.chmod(0o644)
    seal = json.loads(path.read_text())
    seal["ablation_sha256"] = "0" * 64
    seal["sha256"] = digest({k: v for k, v in seal.items() if k != "sha256"})
    atomic_json(path, seal)
    with pytest.raises(ValueError, match="sealed"):
        active.validate_profile_for_monitor(
            screen, active.resolve_active_profile(screen)
        )


def test_cli_outputs_a_verified_path_and_optional_provenance(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "profile.yaml"
    path.write_text("{}\n")
    register(tmp_path, path)
    monkeypatch.setattr("sys.argv", ["active-profile", "--run-root", str(tmp_path)])
    active.main()
    assert capsys.readouterr().out.strip() == str(path)
    monkeypatch.setattr(
        "sys.argv", ["active-profile", "--run-root", str(tmp_path), "--json"]
    )
    active.main()
    assert json.loads(capsys.readouterr().out)["source"] == "registered_profile"
