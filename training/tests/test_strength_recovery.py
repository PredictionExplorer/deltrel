import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts import prepare_strength_recovery as prepare_module
from scripts import run_strength_recovery as runner
from scripts import evaluate_strength_recovery as endpoint
from startrain.checkpoint import (
    ExponentialMovingAverage,
    load_model_manifest,
    write_model_pointer,
    write_recovery_checkpoint,
)
from startrain.config import load_config
from startrain.contracts import SEARCH_ALGORITHM_ID
from startrain.learner import ImmutableModelPublisher
from startrain.orchestration import RunDirectories, build_worker_specs
from startrain.model import GraphResTNet
from startrain.optim import build_optimizer
from startrain.replay_store import ReplayStore
from startrain.runtime import RunIdentity, atomic_json
from startrain.strength_recovery import (
    FORMAT,
    PLAN_NAME,
    SCHEDULE_SECONDS,
    digest,
    due_snapshot,
    load_plan,
    record_snapshot,
    recovery_config,
)
from startrain.training import build_scheduler

CONFIG = Path(__file__).parents[1] / "configs/h100-8gpu-pie-even.yaml"


def test_recovery_preserves_objective_but_declares_quality_and_topology_changes(
    tmp_path,
):
    source = load_config(CONFIG)
    target = recovery_config(
        source,
        destination=tmp_path / "fork",
        replay_watermark=99,
        muon_lr=0.0005,
        adamw_lr=0.0000075,
        warmup_steps=500,
    )
    assert target.model == source.model
    assert target.game == source.game
    assert target.loss == source.loss
    assert target.arena == source.arena
    assert target.orchestration.ring_mixture == source.orchestration.ring_mixture
    assert target.learner.segment_quotas == source.learner.segment_quotas
    assert target.learner.target_updates_per_new_sample == 1.5
    assert target.learner.minimum_replay_shard_id_exclusive == 99
    assert target.orchestration.model_refresh.selfplay_source == "champion"
    assert {gpu.gpu_id for gpu in target.orchestration.gpus} == set(range(7))
    assert target.orchestration.promotion.gpu_id == 7
    assert target.orchestration.promotion.pause_sharing_mode is False
    assert target.orchestration.promotion.finish_inflight_candidate is True
    assert target.orchestration.plateau.enabled is False
    assert target.selfplay.ring_search_allocations == ()
    assert target.selfplay.full_probability == 0.35
    assert target.train.scheduler.min_lr_ratio == 1
    assert target.optimizer.muon_lr == 0.0005
    workers = build_worker_specs(
        target,
        config_path=tmp_path / "profile.yaml",
        directories=RunDirectories.from_experiment(target),
        base_environment={},
    )
    assert {gpu for worker in workers for gpu in worker.gpu_ids} == set(range(8))
    arena = next(worker for worker in workers if worker.role == "arena")
    assert arena.gpu_ids == (7,)
    assert arena.environment["CUDA_VISIBLE_DEVICES"] == "7"
    assert all(7 not in worker.gpu_ids for worker in workers if worker.role == "actor")


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf"), True])
def test_recovery_rejects_invalid_explicit_rates(tmp_path, rate):
    with pytest.raises(ValueError):
        recovery_config(
            load_config(CONFIG),
            destination=tmp_path,
            replay_watermark=0,
            muon_lr=rate,
            adamw_lr=0.00001,
            warmup_steps=0,
        )


def test_recovery_rejects_blind_scheduler_restart(tmp_path):
    source = load_config(CONFIG)
    with pytest.raises(ValueError, match="inherit"):
        recovery_config(
            source,
            destination=tmp_path,
            replay_watermark=0,
            muon_lr=source.optimizer.muon_lr,
            adamw_lr=source.optimizer.adamw_lr,
            warmup_steps=0,
        )


def schedule(root, *, started=1_000_000_000):
    plan = {
        "format": FORMAT,
        "schema_version": 1,
        "search_algorithm": SEARCH_ALGORITHM_ID,
        "run_root": str(root),
        "schedule_seconds": list(SCHEDULE_SECONDS),
        "profile_sha256": "abc",
    }
    plan["plan_sha256"] = digest(plan)
    atomic_json(root / PLAN_NAME, plan)
    atomic_json(
        root / "ablation.json",
        {"profile_sha256": "abc", "measurement_started_ns": started},
    )
    return plan


def test_schedule_uses_actual_clock_and_reports_missing_endpoints(tmp_path):
    schedule(tmp_path)
    assert due_snapshot(tmp_path, now_ns=7200 * 10**9) is None
    assert due_snapshot(tmp_path, now_ns=7201 * 10**9) == (7200, 10**9)
    # Restart after seven hours cannot manufacture a model at hour two.
    assert due_snapshot(tmp_path, now_ns=7 * 3600 * 10**9) == (21600, 10**9)
    atomic_json(tmp_path / "strength-recovery-snapshots/21600/snapshot.json", {})
    assert due_snapshot(tmp_path, now_ns=7 * 3600 * 10**9) is None


def test_schedule_fails_closed_on_contract_or_budget_authority_change(tmp_path):
    schedule(tmp_path)
    raw = json.loads((tmp_path / PLAN_NAME).read_text())
    raw["schedule_seconds"] = [1]
    atomic_json(tmp_path / PLAN_NAME, raw)
    with pytest.raises(ValueError, match="identity or hash"):
        load_plan(tmp_path)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    raw = yaml.safe_load(CONFIG.read_text())
    raw["model"].update(width=8, rrt_groups=1, attention_heads=2, kv_heads=1)
    raw["train"].update(per_rank_batch_size=1, precision="fp32", compile=False)
    raw["data"].update(workers=0, pin_memory=False)
    raw["orchestration"]["directories"]["root"] = str(root)
    raw["orchestration"]["run_id"] = "test-strength"
    profile = root / "profile.yaml"
    profile.write_text(yaml.safe_dump(raw))
    config = load_config(profile)
    identity = RunIdentity(root / "run.json", "test-strength", "family", 1)
    atomic_json(
        identity.path,
        {
            "schema_version": 1,
            "run_id": identity.run_id,
            "generation_family": identity.generation_family,
            "created_ns": 1,
        },
    )
    with ReplayStore(root / "replay") as store:
        store.register_run(identity)
    model = GraphResTNet(config.model)
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.train.scheduler)
    ema = ExponentialMovingAverage(model, decay=config.train.ema_decay)
    with torch.no_grad():
        next(model.parameters()).add_(0.5)
    publisher = ImmutableModelPublisher(root / "learner", identity)
    champion = publisher.publish(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=10,
        epoch=0,
        config=config.as_dict(),
        examples_consumed=100,
        global_batch_size=1,
    )
    write_model_pointer(
        root / "learner/champion.json",
        champion,
        role="champion",
        promotion_result="bootstrap",
    )
    write_recovery_checkpoint(
        root / "learner",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=10,
        epoch=0,
        config=config.as_dict(),
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        examples_consumed=100,
        global_batch_size=1,
    )
    return SimpleNamespace(root=root, profile=profile, config=config, champion=champion)


def test_snapshot_hardlinks_survive_original_retention(source):
    plan = schedule(source.root)
    record_snapshot(source.root, source.champion, now_ns=7 * 3600 * 10**9)
    record = json.loads(
        (source.root / "strength-recovery-snapshots/21600/snapshot.json").read_text()
    )
    assert record["missed_earlier_endpoints"] == [7200]
    assert record["plan_sha256"] == plan["plan_sha256"]
    source.champion.checkpoint.unlink()
    frozen = load_model_manifest(record["artifacts"]["manifest"]["path"])
    assert frozen.model_identity == source.champion.model_identity


def test_prepare_and_apply_isolate_champion_and_zero_old_replay_credit(
    source, tmp_path
):
    output, destination = tmp_path / "plan", tmp_path / "fork"
    before = (source.root / "learner/champion.json").read_bytes()
    plan = prepare_module.prepare(
        source_profile=source.profile,
        destination=destination,
        output=output,
        muon_lr=0.0005,
        adamw_lr=0.0000075,
        warmup_steps=500,
    )
    assert plan["initial_replay_credit"] == 0
    result = prepare_module.apply(output / PLAN_NAME)
    assert result["status"] == "prepared"
    marker = json.loads((destination / "learner/champion-warm-start.json").read_text())
    assert marker["training_segment"]["initial_replay_credit"] == 0
    assert marker["source_model_identity"] == source.champion.model_identity
    assert marker["training_segment"]["optimizer_state"] == "fresh"
    assert marker["training_segment"]["ema_state"] == "fresh-from-champion-ema"
    assert (source.root / "learner/champion.json").read_bytes() == before
    epoch = json.loads((destination / "strength-epoch.json").read_text())
    assert epoch["anchor_identity"] == source.champion.model_identity
    assert epoch["minimum_candidate_step"] == 10
    with pytest.raises(FileExistsError):
        prepare_module.apply(output / PLAN_NAME)


def test_preparation_rejects_source_movement(source, tmp_path):
    output = tmp_path / "plan"
    prepare_module.prepare(
        source_profile=source.profile,
        destination=tmp_path / "fork",
        output=output,
        muon_lr=0.0005,
        adamw_lr=0.0000075,
        warmup_steps=500,
    )
    (source.root / "learner/champion.json").write_text("changed")
    with pytest.raises(ValueError):
        prepare_module.apply(output / PLAN_NAME)
    assert not (tmp_path / "fork").exists()


def test_runner_terminal_retry_never_restarts_workers(monkeypatch, tmp_path):
    config = load_config(CONFIG)
    plan = {"profile": {}, "implementation_pins": [], "wall_budget_seconds": 43200}
    monkeypatch.setattr(runner, "load_plan", lambda _: plan)
    monkeypatch.setattr(runner, "verify_artifact", lambda _: None)
    monkeypatch.setattr(runner, "load_config", lambda _: config)
    monkeypatch.setattr(
        runner, "validate_production_ring_allocations", lambda *a, **k: None
    )
    monkeypatch.setattr(
        runner, "run_elo_ablation", lambda **_: pytest.fail("restarted")
    )
    atomic_json(
        tmp_path / "ablation.json",
        {
            "wall_budget_seconds": 43200,
            "measurement_status": "complete",
            "resource_released_ns": 123,
        },
    )
    assert (
        runner.run(root=tmp_path, orchestrator="unused")["status"] == "already_complete"
    )


@pytest.fixture
def frozen_endpoint(source, tmp_path):
    plan = schedule(source.root)
    plan["profile"] = prepare_module.artifact(source.profile)
    plan["anchor_identity"] = source.champion.model_identity
    plan["anchor_manifest_name"] = source.champion.artifact_manifest.name
    plan["plan_sha256"] = digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    atomic_json(source.root / PLAN_NAME, plan)
    (source.root / "profile-elo-ablation.yaml").write_bytes(source.profile.read_bytes())
    record_snapshot(source.root, source.champion, now_ns=7 * 3600 * 10**9)
    output = tmp_path / "endpoint"
    receipt = endpoint.freeze(
        root=source.root,
        snapshot_seconds=21600,
        output=output,
        simulations=256,
        pairs_per_cell=4,
        wall_budget_hours=2,
    )
    return source, output, receipt


def test_endpoint_preserves_broad_production_search_and_frozen_original_anchor(
    frozen_endpoint,
):
    source, output, receipt = frozen_endpoint
    diagnostic_plan, config = endpoint.diagnostic.verify_plan(output)
    assert config.simulations == 256
    assert config.max_considered == source.config.arena.max_considered == 64
    assert receipt["anchor_identity"] == source.champion.model_identity
    (source.root / "learner/champion.json").write_text("moving pointer")
    assert endpoint.diagnostic.verify_plan(output)[0] == diagnostic_plan


def test_endpoint_rejects_profile_changed_after_snapshot(frozen_endpoint, tmp_path):
    source, _, _ = frozen_endpoint
    installed = source.root / "profile-elo-ablation.yaml"
    changed = yaml.safe_load(installed.read_text())
    changed["arena"]["seed"] += 1
    installed.write_text(yaml.safe_dump(changed))
    destination = tmp_path / "tampered-endpoint"
    with pytest.raises(ValueError):
        endpoint.freeze(
            root=source.root,
            snapshot_seconds=21600,
            output=destination,
            simulations=256,
            pairs_per_cell=4,
            wall_budget_hours=2,
        )
    assert not destination.exists()


def test_endpoint_budget_expiry_does_not_start_inference(frozen_endpoint, monkeypatch):
    _, output, receipt = frozen_endpoint
    atomic_json(
        output / "endpoint-clock.json",
        {"plan_sha256": receipt["plan_sha256"], "started_ns": 1},
    )
    monkeypatch.setattr(
        endpoint.diagnostic,
        "run_session",
        lambda **_: pytest.fail("inference after deadline"),
    )
    first = endpoint.run(output=output, device="cpu", exclusive_device=False)
    assert first["status"] == "budget_exhausted"
    assert "Incomplete" in first["interpretation"]
    assert endpoint.run(output=output, device="cpu", exclusive_device=False) == first


def test_endpoint_resume_preserves_clock_and_stops_after_ema_budget(
    frozen_endpoint, monkeypatch
):
    _, output, _ = frozen_endpoint
    calls = []

    def session(**kwargs):
        calls.append(kwargs)
        return {
            "arms": {
                "ema": {"terminal": len(calls) == 2, "completed_pairs": 8 * len(calls)}
            }
        }

    monkeypatch.setattr(endpoint.diagnostic, "run_session", session)
    first = endpoint.run(output=output, device="cpu", exclusive_device=False)
    second = endpoint.run(output=output, device="cpu", exclusive_device=False)
    assert first["status"] == "running"
    assert second["status"] == "complete"
    assert first["started_ns"] == second["started_ns"]
    assert all(call["arm"] == "ema" for call in calls)
    assert endpoint.run(output=output, device="cpu", exclusive_device=False) == second
    assert len(calls) == 2
