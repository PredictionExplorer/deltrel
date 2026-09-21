from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts import graceful_training_deploy as deploy
from deltreltrain.checkpoint import ExponentialMovingAverage, save_checkpoint
from deltreltrain.config import load_config

WORKERS = ["learner", "actor-gpu-1"]


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def reconciliation_fixture(tmp_path, *, missing=1):
    root = (tmp_path / "run").resolve()
    root.mkdir()
    config = load_config(Path(__file__).parents[1] / "configs/small.yaml")
    config = replace(
        config,
        train=replace(config.train, per_rank_batch_size=512),
        orchestration=replace(
            config.orchestration,
            run_id="reconciliation-run",
            directories=replace(config.orchestration.directories, root=str(root)),
        ),
    )
    profile = root / "source.yaml"
    profile.write_text(yaml.safe_dump(json.loads(json.dumps(config.as_dict()))))
    (root / "profile.sha256").write_text(f"{deploy.digest(profile)}  {profile}\n")
    (root / "source-commit.txt").write_text("a" * 40 + "\n")
    identity = {
        "schema_version": 1,
        "run_id": "reconciliation-run",
        "generation_family": "reconciliation-family",
        "created_ns": 1,
    }
    write(root / "run.json", identity)
    step, epoch, batch = 26, 7, 512
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    checkpoint_path = root / "learner/recovery/staged.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ExponentialMovingAverage(model),
        step=step,
        epoch=epoch,
        config=config.as_dict(),
        extra={
            "run_id": identity["run_id"],
            "generation_family": identity["generation_family"],
            "examples_consumed": step * batch,
            "global_batch_size": batch,
        },
    )
    checksum = deploy.digest(checkpoint_path)
    final_checkpoint = checkpoint_path.with_name(f"sha256-{checksum}.pt")
    checkpoint_path.rename(final_checkpoint)
    pointer = {
        "format": "deltreltrain.recovery-pointer",
        "schema_version": 1,
        "checkpoint": f"recovery/{final_checkpoint.name}",
        "checkpoint_sha256": checksum,
        "checkpoint_bytes": final_checkpoint.stat().st_size,
        "step": step,
        "epoch": epoch,
        "examples_consumed": step * batch,
        "run_id": identity["run_id"],
        "generation_family": identity["generation_family"],
        "updated_ns": 300,
    }
    heartbeat = {
        "schema_version": 1,
        "worker": "learner",
        "phase": "stopped",
        "pid": 987654321,
        "heartbeat_ns": 400,
        "step": step,
        "epoch": epoch,
        "examples_consumed": (step - missing) * batch,
        "window_batches_consumed": step - missing,
        "window_batches_allocated": 100,
        "window_selection_max_shard_id": 99,
    }
    coordinator = {
        "schema_version": 1,
        "state": "stopped",
        "failure": None,
        "timestamp_ns": 500,
        "coordinator_pid": None,
        "workers": {
            name: {"state": "stopped", "pid": None, "last_exit_code": 0}
            for name in WORKERS
        },
    }
    rows = [
        {
            "schema_version": 1,
            "worker": "learner",
            "event": "utd_wait",
            "timestamp_ns": 100,
            "step": step - missing,
            "examples_consumed": (step - missing) * batch,
        },
        {
            "schema_version": 1,
            "worker": "learner",
            "event": "replay_window_consumed",
            "timestamp_ns": 200,
            "step": step,
            "epoch": epoch,
            "window_batches_consumed_this_spin": missing,
            "window_batches_consumed": step,
            "window_batches_allocated": 100,
            "window_opened_step": 0,
            "window_selection_max_shard_id": 99,
        },
        {
            "schema_version": 1,
            "worker": "learner",
            "event": "recovery_checkpoint",
            "timestamp_ns": 350,
            "step": step,
            "checkpoint_sha256": checksum,
            "checkpoint_bytes": final_checkpoint.stat().st_size,
        },
    ]
    for path, payload in [
        (root / "learner/recovery.json", pointer),
        (root / "status/learner.heartbeat.json", heartbeat),
        (root / "status/coordinator.json", coordinator),
    ]:
        write(path, payload)
    metrics = root / "learner/metrics.jsonl"
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return SimpleNamespace(
        root=root,
        evidence=tmp_path / "evidence",
        profile=profile,
        config=config,
        checkpoint=final_checkpoint,
        pointer=pointer,
        heartbeat=heartbeat,
        coordinator=coordinator,
        metrics=metrics,
        rows=rows,
        workers=WORKERS,
    )


@pytest.mark.parametrize("missing", [1, 3, 10])
def test_verified_same_window_spin_repairs_only_telemetry_preserving_raw_proof(
    tmp_path, missing
):
    case = reconciliation_fixture(tmp_path, missing=missing)
    heartbeat = case.root / "status/learner.heartbeat.json"
    original = heartbeat.read_bytes()
    checkpoint_bytes = case.checkpoint.read_bytes()
    pointer_bytes = (case.root / "learner/recovery.json").read_bytes()
    with pytest.raises(RuntimeError, match="checkpoint examples are ahead"):
        deploy.validate_boundary(case.root, strict=True, expected_workers=WORKERS)
    result = deploy.reconcile_stopped_heartbeat(
        case.root, case.evidence, expected_workers=WORKERS
    )
    assert result["reconciled_batches"] == missing
    corrected = json.loads(heartbeat.read_text())
    assert corrected["examples_consumed"] == case.pointer["examples_consumed"]
    assert {
        key: corrected[key] for key in case.heartbeat if key != "examples_consumed"
    } == {
        key: value
        for key, value in case.heartbeat.items()
        if key != "examples_consumed"
    }
    proof = json.loads(Path(result["proof"]).read_text())
    assert Path(result["proof"]).parent == case.root / "status"
    assert (
        Path(result["audit_proof"]).read_bytes() == Path(result["proof"]).read_bytes()
    )
    assert Path(result["proof"]).stat().st_mode & 0o222 == 0
    assert proof["original_files"]["heartbeat"]["contents_utf8"].encode() == original
    assert proof["metric_evidence"]["tail_utf8"] == case.metrics.read_text()
    assert case.checkpoint.read_bytes() == checkpoint_bytes
    assert (case.root / "learner/recovery.json").read_bytes() == pointer_bytes
    assert (
        deploy.validate_boundary(case.root, strict=True, expected_workers=WORKERS)[
            "discarded_uncheckpointed_examples"
        ]
        == 0
    )
    assert (
        deploy.reconcile_stopped_heartbeat(
            case.root, case.evidence, expected_workers=WORKERS
        )
        is None
    )


@pytest.mark.parametrize("strict", [False, True])
def test_controller_stop_and_recovery_boundary_reconcile_before_unchanged_validation(
    tmp_path, monkeypatch, strict
):
    case = reconciliation_fixture(tmp_path)
    subject = object.__new__(deploy.Deployment)
    subject.root, subject.base, subject.main = (
        case.root,
        case.evidence,
        "training.service",
    )
    subject.plan = {"expected_workers": WORKERS}
    monkeypatch.setattr(subject, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(subject, "show", lambda *_: "0")
    monkeypatch.setattr(deploy.subprocess, "check_output", lambda *args, **kwargs: "")
    result = subject.stop(strict=strict)
    assert result["telemetry_reconciliation"]["reconciled_batches"] == 1
    assert result["discarded_uncheckpointed_steps"] == 0
    assert result["discarded_uncheckpointed_examples"] == 0


@pytest.mark.parametrize(
    "defect",
    [
        "dirty_stop",
        "different_epoch",
        "different_step",
        "missing_wait",
        "wrong_window",
        "wrong_publication",
        "wrong_examples",
        "later_progress",
        "wrong_order",
    ],
)
def test_unproved_reconciliation_never_changes_heartbeat_or_checkpoint(
    tmp_path, defect
):
    case = reconciliation_fixture(tmp_path)
    if defect == "dirty_stop":
        case.coordinator["workers"]["learner"]["last_exit_code"] = 1
        write(case.root / "status/coordinator.json", case.coordinator)
    elif defect in {"different_epoch", "different_step"}:
        case.heartbeat["epoch" if defect == "different_epoch" else "step"] += 1
        write(case.root / "status/learner.heartbeat.json", case.heartbeat)
    elif defect == "missing_wait":
        case.rows.pop(0)
    elif defect == "wrong_window":
        case.rows[1]["window_selection_max_shard_id"] += 1
    elif defect == "wrong_publication":
        case.rows[2]["checkpoint_sha256"] = "a" * 64
    elif defect == "wrong_examples":
        case.heartbeat["examples_consumed"] -= 1
        write(case.root / "status/learner.heartbeat.json", case.heartbeat)
    elif defect == "later_progress":
        case.rows.append({"worker": "learner", "step": 27})
    elif defect == "wrong_order":
        case.rows.reverse()
    case.metrics.write_text("".join(json.dumps(row) + "\n" for row in case.rows))
    heartbeat_path = case.root / "status/learner.heartbeat.json"
    original = heartbeat_path.read_bytes()
    checkpoint_bytes = case.checkpoint.read_bytes()
    with pytest.raises(RuntimeError):
        deploy.reconcile_stopped_heartbeat(
            case.root, case.evidence, expected_workers=WORKERS
        )
    assert heartbeat_path.read_bytes() == original
    assert case.checkpoint.read_bytes() == checkpoint_bytes
    assert not list(case.evidence.glob("*.json"))
