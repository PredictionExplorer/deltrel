import base64
import json
from types import SimpleNamespace

import pytest

from test_strength_recovery import source as source
from test_strength_recovery_continuation import screen as screen
from scripts import repair_strength_screen_integrity as repair
from scripts import run_elo_ablation as runner
from scripts import run_strength_recovery_continuation as controller
from deltreltrain.runtime import atomic_json
from deltreltrain.strength_recovery import completed_screen


@pytest.fixture
def missing_integrity(screen, monkeypatch):
    controller.prepare(screen, source_commit="a" * 40)
    metadata = json.loads((screen / "ablation.json").read_text())
    metadata.update(integrity=None, integrity_status=None, measurement_attempt_count=1)
    metadata["measurement_attempts"] = [
        {
            "status": "complete",
            "completion_status": "complete",
            "outcome": "budget_completion",
            "stop_reason": "wall_budget",
            "measurement_cutoff_ns": metadata["measurement_cutoff_ns"],
            "resource_released_ns": metadata["resource_released_ns"],
            "teardown": metadata["measurement_teardown"],
            "integrity": None,
            "integrity_status": None,
            "failure": None,
        }
    ]
    atomic_json(screen / "ablation.json", metadata)
    atomic_json(
        screen / "status/coordinator.json",
        {
            "state": "stopped",
            "failure": None,
            "coordinator_pid": None,
            "workers": {
                "learner": {"state": "stopped", "last_exit_code": 0, "pid": None}
            },
        },
    )
    state = {
        "ActiveState": "failed",
        "SubState": "failed",
        "Result": "exit-code",
        "MainPID": "0",
        "ExecMainCode": "1",
        "ExecMainStatus": "78",
        "InvocationID": "a" * 32,
        "ExecStart": f"scripts.run_strength_recovery_continuation run --run-root {screen}",
    }
    monkeypatch.setattr(repair, "service_state", lambda _: dict(state))
    monkeypatch.setattr(controller, "require_idle_gpus", lambda: None)
    return screen, state


def test_missing_integrity_repair_runs_real_preflight_preserves_bytes_and_handoffs(
    missing_integrity,
):
    root, _ = missing_integrity
    original = (root / "ablation.json").read_bytes()
    with pytest.raises(ValueError, match="wall budget"):
        completed_screen(root)
    assert repair.repair(root, unit="recovery.service")["status"] == "eligible"
    assert (root / "ablation.json").read_bytes() == original
    result = repair.repair(root, unit="recovery.service", apply=True)
    assert result["status"] == "repaired"
    receipt = json.loads((root / repair.RECEIPT).read_text())
    before, after = repair.validate_receipt(receipt)
    assert before == original and after == (root / "ablation.json").read_bytes()
    assert receipt["integrity"]["state_preflight"]["status"] == "ok"
    assert completed_screen(root)["integrity"]["valid"] is True
    assert repair.repair(root, unit="recovery.service", apply=True) == result
    plan = controller.verify_plan(root)
    assert controller.handoff(root, plan).name == controller.INSTALLED
    assert json.loads((root / controller.SEAL).read_text())[
        "ablation_sha256"
    ] == repair._sha(after)


@pytest.mark.parametrize(
    "change",
    [
        "active",
        "different_exit",
        "missing_endpoint",
        "bad_profile",
        "bad_plan",
        "bad_attempt",
        "warning",
        "bad_state",
        "new_invocation",
    ],
)
def test_repair_refuses_unrelated_or_unverified_failure(missing_integrity, change):
    root, state = missing_integrity
    if change == "active":
        state.update(ActiveState="active", MainPID="1")
    elif change == "different_exit":
        state["ExecMainStatus"] = "75"
    elif change == "missing_endpoint":
        (root / "strength-recovery-snapshots/43200/snapshot.json").unlink()
    elif change == "bad_profile":
        (root / "profile-elo-ablation.yaml").chmod(0o644)
        (root / "profile-elo-ablation.yaml").write_text("changed")
    elif change == "bad_plan":
        (root / "strength-continuation-plan.json").chmod(0o644)
        (root / "strength-continuation-plan.json").write_text("{}")
    elif change in ("bad_attempt", "warning"):
        path = root / "ablation.json"
        metadata = json.loads(path.read_text())
        if change == "bad_attempt":
            metadata["measurement_attempts"][-1]["stop_reason"] = "signal_15"
        else:
            metadata["measurement_completion_status"] = "complete_with_warning"
        atomic_json(path, metadata)
    elif change == "bad_state":
        (root / "learner/utd-segment.json").write_text("{}")
    elif change == "new_invocation":
        repair.repair(root, unit="recovery.service", apply=True)
        state["InvocationID"] = "b" * 32
    before = (root / "ablation.json").read_bytes()
    with pytest.raises((ValueError, RuntimeError, OSError)):
        repair.repair(root, unit="recovery.service", apply=True, restart=True)
    assert (root / "ablation.json").read_bytes() == before


def test_interrupted_receipt_commit_recovers_only_exact_metadata(
    missing_integrity, monkeypatch
):
    root, _ = missing_integrity
    original = repair._write

    def fail(path, data, *, replace):
        if path.name == repair.RECEIPT:
            raise OSError("interrupted receipt")
        original(path, data, replace=replace)

    monkeypatch.setattr(repair, "_write", fail)
    with pytest.raises(OSError, match="interrupted receipt"):
        repair.repair(root, unit="recovery.service", apply=True)
    assert (root / repair.PENDING).is_file()
    assert completed_screen(root)["integrity"]["valid"] is True
    monkeypatch.setattr(repair, "_write", original)
    assert (
        repair.repair(root, unit="recovery.service", apply=True)["status"] == "repaired"
    )
    assert not (root / repair.PENDING).exists()
    receipt = json.loads((root / repair.RECEIPT).read_text())
    receipt["original"]["payload_base64"] = base64.b64encode(b"altered").decode()
    with pytest.raises(ValueError):
        repair.validate_receipt(receipt)


def test_restart_targets_same_registered_service_once_per_failed_invocation(
    missing_integrity, monkeypatch
):
    root, state = missing_integrity
    commands = []
    monkeypatch.setattr(
        repair.subprocess, "run", lambda cmd, **kw: commands.append(cmd)
    )
    assert repair.repair(root, unit="recovery.service", apply=True, restart=True)[
        "restart_requested"
    ]
    assert commands == [["systemctl", "start", "--no-block", "recovery.service"]]
    state["InvocationID"] = "b" * 32
    with pytest.raises(ValueError, match="invocation changed"):
        repair.repair(root, unit="recovery.service", apply=True, restart=True)
    assert len(commands) == 1


def test_clean_budget_runner_records_real_integrity_accepted_by_continuation(
    screen, monkeypatch
):
    path = screen / "ablation.json"
    metadata = json.loads(path.read_text())
    clock = [metadata["measurement_cutoff_ns"]]
    metadata.update(
        measurement_status="running",
        measurement_outcome=None,
        measurement_started_ns=clock[0] - 43200 * 10**9 + 10**9,
        measurement_attempts=[],
        measurement_attempt_count=0,
    )
    atomic_json(path, metadata)
    monkeypatch.setattr(runner.time, "time_ns", lambda: clock[0])
    monkeypatch.setattr(
        runner.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + int(seconds * 10**9)),
    )
    monkeypatch.setattr(runner, "restore_if_corrupt", lambda _: None)
    monkeypatch.setattr(runner, "_resolve_executable", lambda _: "fake-orchestrator")
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(poll=lambda: None)
    )
    monkeypatch.setattr(
        runner,
        "_terminate",
        lambda *a, **kw: {
            "clean": True,
            "process_group_released": True,
            "exit_code": 0,
            "resource_released_ns": clock[0],
            "status": "graceful",
        },
    )
    report = runner.run_elo_ablation(
        config_path=screen / "profile-elo-ablation.yaml",
        orchestrator="fake",
        poll_seconds=1.0,
    )
    assert report["status"] == "complete"
    assert report["integrity"]["valid"] is True
    assert completed_screen(screen)["integrity"]["state_preflight"]["status"] == "ok"


def test_clean_exit_with_invalid_final_state_is_failed(screen, monkeypatch):
    path = screen / "ablation.json"
    metadata = json.loads(path.read_text())
    metadata.update(
        measurement_status="running", measurement_outcome=None, measurement_attempts=[]
    )
    atomic_json(path, metadata)
    monkeypatch.setattr(runner, "restore_if_corrupt", lambda _: None)
    monkeypatch.setattr(runner, "_resolve_executable", lambda _: "unused")
    calls = []

    def preflight(*args, **kwargs):
        calls.append(kwargs["apply"])
        if not kwargs["apply"]:
            raise RuntimeError("final checkpoint corrupt")
        return {"status": "ok"}

    monkeypatch.setattr(runner, "run_state_preflight", preflight)
    report = runner.run_elo_ablation(
        config_path=screen / "profile-elo-ablation.yaml",
        orchestrator="unused",
        poll_seconds=1.0,
    )
    assert report["status"] == "failed"
    assert report["failure_domain"] == "state_integrity"
    assert calls == [True, False]
    with pytest.raises(ValueError, match="wall budget"):
        completed_screen(screen)


def test_receipt_is_durable_under_existing_r3_root_json_backup(
    missing_integrity, tmp_path
):
    from scripts.training_disaster_recovery import _SnapshotBuilder, _object_path

    root, _ = missing_integrity
    repair.repair(root, unit="recovery.service", apply=True)
    backup = tmp_path / "backup"
    backup.mkdir()
    builder = _SnapshotBuilder(root, backup)
    logical, entry = builder.add_run_file(root / repair.RECEIPT, "run-metadata")
    assert logical == repair.RECEIPT
    original, repaired = repair.validate_receipt(
        json.loads(_object_path(backup, entry.sha256).read_text())
    )
    assert json.loads(original)["integrity"] is None
    assert repaired == (root / "ablation.json").read_bytes()


def test_root_handler_preserves_run_owner_for_controller_sealing(tmp_path, monkeypatch):
    path = tmp_path / "ablation.json"
    path.write_text("original")
    owner = path.stat()
    changes = []
    monkeypatch.setattr(repair.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        repair.os, "fchown", lambda fd, uid, gid: changes.append((uid, gid))
    )
    repair._write(path, b"repaired", replace=True)
    assert changes == [(owner.st_uid, owner.st_gid)]
    assert path.read_bytes() == b"repaired"
