from dataclasses import replace
import json
import os
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

from test_strength_recovery import source as source
from scripts import prepare_strength_recovery as preparation
from scripts import run_strength_recovery_continuation as controller
from scripts.migrate_continuous_profile import _validate_profile_pair
from deltreltrain.checkpoint import load_model_manifest
from deltreltrain.config import load_config
from deltreltrain.runtime import atomic_json
from deltreltrain.strength_recovery import (
    PLAN_NAME,
    completed_screen,
    due_snapshot,
    record_snapshot,
    validate_continuation_transition,
)


@pytest.fixture
def screen(source, tmp_path):
    root, output = tmp_path / "screen", tmp_path / "plan"
    preparation.prepare(
        source_profile=source.profile,
        destination=root,
        output=output,
        muon_lr=0.0005,
        adamw_lr=0.0000075,
        warmup_steps=500,
    )
    preparation.apply(output / PLAN_NAME)
    (root / "source-commit.txt").write_text("a" * 40 + "\n")
    metadata = json.loads((root / "ablation.json").read_text())
    now = time.time_ns()
    metadata.update(
        measurement_started_ns=now - 43200 * 10**9, measurement_status="running"
    )
    atomic_json(root / "ablation.json", metadata)
    record_snapshot(
        root, load_model_manifest(root / "learner/champion.json"), now_ns=now
    )
    metadata.update(
        measurement_status="complete",
        measurement_outcome="budget_completion",
        measurement_completion_status="complete",
        measurement_stop_reason="wall_budget",
        measurement_cutoff_ns=now,
        resource_released_ns=now + 1,
        measurement_teardown={"clean": True, "process_group_released": True},
        integrity={"valid": True},
    )
    atomic_json(root / "ablation.json", metadata)
    atomic_json(
        root / "status/learner.heartbeat.json",
        {
            "schema_version": 1,
            "worker": "learner",
            "phase": "stopped",
            "pid": 999999999,
            "step": 10,
            "epoch": 0,
            "examples_consumed": 100,
            "heartbeat_ns": now,
        },
    )
    return root


def test_completed_screen_never_fabricates_late_missing_endpoints(screen):
    endpoint = screen / "strength-recovery-snapshots/43200/snapshot.json"
    endpoint.unlink()
    assert due_snapshot(screen, now_ns=time.time_ns() + 10**12) is None
    with pytest.raises(ValueError, match="retained twelve-hour"):
        completed_screen(screen)


@pytest.mark.parametrize(
    "field,value",
    [
        ("measurement_status", "failed"),
        ("measurement_outcome", "transient_crash"),
        ("measurement_completion_status", "failed"),
        ("measurement_stop_reason", "signal_15"),
        ("resource_released_ns", None),
        ("measurement_teardown", {"clean": False}),
        ("integrity", {"valid": False}),
    ],
)
def test_false_budget_completion_cannot_handoff(screen, field, value):
    metadata = json.loads((screen / "ablation.json").read_text())
    metadata[field] = value
    atomic_json(screen / "ablation.json", metadata)
    with pytest.raises(ValueError, match="wall budget"):
        completed_screen(screen)


def test_only_three_million_cadence_can_use_completed_screen_exception(screen):
    source = load_config(screen / "profile-elo-ablation.yaml")
    target = replace(
        source, learner=replace(source.learner, candidate_interval_examples=3_000_000)
    )
    assert validate_continuation_transition(source, target, screen)
    changes = _validate_profile_pair(source, target, run_root=screen)
    assert [row[0] for row in changes] == ["learner.candidate_interval_examples"]
    coupled = replace(target, train=replace(target.train, ema_decay=0.99))
    assert not validate_continuation_transition(source, coupled, screen)


def test_real_migration_is_idempotently_recovered_and_screen_stays_sealed(screen):
    plan = controller.prepare(screen, source_commit="a" * 40)
    original = (screen / "ablation.json").read_bytes()
    old_champion = (screen / "learner/champion.json").read_bytes()
    target = controller.handoff(screen, plan)
    assert target.name == controller.INSTALLED
    assert load_config(target).learner.candidate_interval_examples == 3_000_000
    assert (screen / "profile.sha256").read_text().split()[0] == plan["target_profile"][
        "sha256"
    ]
    assert controller.handoff(screen, plan) == target
    assert (screen / "ablation.json").read_bytes() == original
    assert (screen / "learner/champion.json").read_bytes() == old_champion
    assert json.loads((screen / controller.SEAL).read_text())["ablation"] == json.loads(
        original
    )


def test_occupied_gpu_blocks_start(monkeypatch):
    monkeypatch.setattr(
        controller.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="GPU-abc, 123\n"),
    )
    with pytest.raises(RuntimeError, match="release all"):
        controller.require_idle_gpus()


def test_continuation_stop_reaps_owned_process_and_accounts_separately(
    screen, monkeypatch
):
    plan = controller.prepare(screen, source_commit="a" * 40)
    target = controller.handoff(screen, plan)
    original = (screen / "ablation.json").read_bytes()
    monkeypatch.setattr(controller, "require_idle_gpus", lambda: None)
    monkeypatch.setattr(
        controller, "_resolve_executable", lambda _: "/qualified/orchestrator"
    )

    class Latch:
        def install(self):
            pass

        def is_set(self):
            return True

    monkeypatch.setattr(controller, "SignalLatch", Latch)
    process = SimpleNamespace(pid=987654, returncode=0, poll=lambda: None)
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *a, **k: process)
    terminated = []

    def terminate(owned, **kwargs):
        terminated.append(owned)
        return {"clean": True, "process_group_released": True}

    monkeypatch.setattr(controller, "_terminate", terminate)
    state = controller.supervise_continuation(screen, target, plan, "orchestrator")
    assert state["phase"] == "stopped"
    assert state["provisioned_gpus"] == 8
    assert state["continuation_provisioned_gpu_hours"] >= 0
    assert terminated == [process]
    assert (screen / "ablation.json").read_bytes() == original


def test_failed_screen_never_starts_continuation(screen, monkeypatch):
    controller.prepare(screen, source_commit="a" * 40)
    metadata = json.loads((screen / "ablation.json").read_text())
    metadata["measurement_status"] = "running"
    atomic_json(screen / "ablation.json", metadata)
    monkeypatch.setattr(controller, "require_idle_gpus", lambda: None)
    monkeypatch.setattr(
        controller, "run_screen", lambda **_: {"status": "failed", "error": "bad state"}
    )
    monkeypatch.setattr(controller, "handoff", lambda *a: pytest.fail("false handoff"))
    result = controller.run(screen, orchestrator="unused")
    assert result["status"] == "failed" and result["phase"] == "screen"


def test_continuation_transient_crashes_have_bounded_restarts(screen, monkeypatch):
    plan = controller.prepare(screen, source_commit="a" * 40)
    target = controller.handoff(screen, plan)
    monkeypatch.setattr(controller, "require_idle_gpus", lambda: None)
    monkeypatch.setattr(
        controller, "_resolve_executable", lambda _: "/qualified/orchestrator"
    )
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)
    atomic_json(screen / "coordinator.lock", {"pid": 999999999, "created_ns": 1})
    launched = []

    def spawn(*args, **kwargs):
        process = SimpleNamespace(
            pid=987654 + len(launched), returncode=75, poll=lambda: 75
        )
        launched.append(process)
        return process

    monkeypatch.setattr(controller.subprocess, "Popen", spawn)
    monkeypatch.setattr(
        controller,
        "_terminate",
        lambda *a, **k: {"clean": True, "process_group_released": True},
    )
    with pytest.raises(RuntimeError, match="restart allowance"):
        controller.supervise_continuation(screen, target, plan, "orchestrator")
    assert len(launched) == 4
    state = json.loads((screen / controller.STATE).read_text())
    assert state["phase"] == "failed" and len(state["attempts"]) == 4


def test_qualified_source_precedes_editable_environment_and_is_restored(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/older/release:/existing/libraries")
    expected = str(Path(controller.__file__).resolve().parents[1])
    with controller.qualified_sources():
        assert os.environ["PYTHONPATH"].split(os.pathsep)[0] == expected
        assert "/older/release" in os.environ["PYTHONPATH"]
    assert os.environ["PYTHONPATH"] == "/older/release:/existing/libraries"
