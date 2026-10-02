"""Immutable, elapsed-time checkpoints for an isolated recovery experiment."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from .checkpoint import ModelManifest, sha256_file, verify_file
from .config import ExperimentConfig
from .contracts import SEARCH_ALGORITHM_ID
from .runtime import atomic_json

PLAN_NAME = "strength-recovery-plan.json"
FORMAT = "startrain.strength-recovery"
SCHEDULE_SECONDS = (7200, 21600, 43200)


def digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def recovery_config(
    source: ExperimentConfig,
    *,
    destination: Path,
    replay_watermark: int,
    muon_lr: float,
    adamw_lr: float,
    warmup_steps: int,
) -> ExperimentConfig:
    """One explicitly coupled restart; preserve model/rules/loss/search targets."""
    if source.orchestration.training_objective != "ring10_pie":
        raise ValueError("strength recovery requires the existing ring10_pie objective")
    if source.learner.target_updates_per_new_sample != 1.5:
        raise ValueError("restart screen preserves reuse 1.5")
    if type(replay_watermark) is not int or replay_watermark < 0:
        raise ValueError("replay watermark must be a non-negative integer")
    if type(warmup_steps) is not int or not 0 <= warmup_steps <= 5000:
        raise ValueError("explicit warmup must be in [0, 5000]")
    for name, value in (("muon_lr", muon_lr), ("adamw_lr", adamw_lr)):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (muon_lr, adamw_lr) == (source.optimizer.muon_lr, source.optimizer.adamw_lr):
        raise ValueError(
            "select calibrated rates explicitly; do not inherit reference rates"
        )
    roles = {(gpu.gpu_id, gpu.role) for gpu in source.orchestration.gpus}
    expected = {(0, "learner"), *((gpu, "actor") for gpu in range(1, 8))}
    if roles != expected:
        raise ValueError("recovery requires learner 0 and seven existing actor GPUs")
    actor7 = next(gpu for gpu in source.orchestration.gpus if gpu.gpu_id == 7)
    orchestration = replace(
        source.orchestration,
        directories=replace(
            source.orchestration.directories, root=str(destination.resolve())
        ),
        gpus=tuple(gpu for gpu in source.orchestration.gpus if gpu.gpu_id != 7),
        model_refresh=replace(
            source.orchestration.model_refresh,
            selfplay_source="champion",
            candidate_probability=0.0,
            history_probability=0.0,
        ),
        promotion=replace(
            source.orchestration.promotion,
            gpu_id=7,
            cpu_affinity=actor7.cpu_affinity,
            pause_sharing_mode=False,
            pause_strategy="terminate",
            max_waves_per_lease=None,
            inter_wave_cooldown_seconds=0.0,
            session_seconds=3600.0,
            final_drain_timeout_seconds=300.0,
            finish_inflight_candidate=True,
        ),
        historical_evaluation=replace(
            source.orchestration.historical_evaluation,
            enabled=True,
            measure_direct_predecessor=True,
            every_promotions=1,
            simulations=1024,
            measurement_service_fraction=0.0,
            cooldown_seconds=0.0,
            session_seconds=3600.0,
        ),
        # This is a fixed-rate screen. Automatic optimizer resets would couple
        # its endpoint to asynchronous verdict timing instead of its pinned arm.
        plateau=replace(source.orchestration.plateau, enabled=False),
        shutdown=replace(
            source.orchestration.shutdown,
            terminate_grace_seconds=300.0,
            kill_grace_seconds=30.0,
        ),
    )
    return replace(
        source,
        optimizer=replace(source.optimizer, muon_lr=muon_lr, adamw_lr=adamw_lr),
        train=replace(
            source.train,
            scheduler=replace(
                source.train.scheduler, warmup_steps=warmup_steps, min_lr_ratio=1.0
            ),
        ),
        # This is an explicit higher-search-quality restart package. Old
        # reduced-full-search admission receipts do not qualify corrected pie
        # search and are deliberately not inherited into the new namespace.
        selfplay=replace(
            source.selfplay,
            ring_search_allocations=(),
            fast_probability=0.65,
            full_probability=0.35,
            fast_simulations=32,
            full_simulations=384,
            simulation_reference_rings=6,
            simulation_ring_exponent=1.0,
        ),
        learner=replace(
            source.learner,
            minimum_replay_shard_id_exclusive=replay_watermark,
            protected_champion_fraction=0.0,
            protected_champion_after_ns=None,
            reuse_clock_reference_target=None,
            # Timed publication below, not an assumed learner speed, controls
            # the predeclared endpoints. The ordinary cadence stays unreachable.
            candidate_interval_examples=10**15,
            selfplay_snapshot_interval_examples=None,
            selfplay_snapshot_warmup_examples=0,
            selfplay_snapshot_warmup_interval_examples=None,
        ),
        orchestration=orchestration,
    )


def load_plan(root: Path) -> dict[str, Any] | None:
    path = root / PLAN_NAME
    if not path.exists():
        return None
    if path.is_symlink():
        raise ValueError("strength recovery plan must not be a symlink")
    payload = json.loads(path.read_text())
    body = {key: value for key, value in payload.items() if key != "plan_sha256"}
    if (
        payload.get("format") != FORMAT
        or payload.get("schema_version") != 1
        or payload.get("plan_sha256") != digest(body)
        or payload.get("search_algorithm") != SEARCH_ALGORITHM_ID
        or payload.get("schedule_seconds") != list(SCHEDULE_SECONDS)
        or Path(payload.get("run_root", "")).resolve() != root.resolve()
    ):
        raise ValueError("strength recovery plan identity or hash differs")
    return payload


def due_snapshot(root: Path, *, now_ns: int | None = None) -> tuple[int, int] | None:
    """Return the latest due endpoint; never fabricate missed earlier models."""
    plan = load_plan(root)
    if plan is None:
        return None
    metadata = json.loads((root / "ablation.json").read_text())
    if metadata.get("profile_sha256") != plan["profile_sha256"]:
        raise ValueError("recovery budget/profile disagrees with the pinned schedule")
    started = metadata.get("measurement_started_ns")
    if started is None:
        return None
    if type(started) is not int or started <= 0:
        raise ValueError("recovery experiment start time is invalid")
    now = time.time_ns() if now_ns is None else now_ns
    elapsed = (now - started) / 1e9
    due = [seconds for seconds in SCHEDULE_SECONDS if elapsed >= seconds]
    if not due:
        return None
    endpoint = due[-1]
    record = root / "strength-recovery-snapshots" / str(endpoint) / "snapshot.json"
    return None if record.exists() else (endpoint, started)


def record_snapshot(
    root: Path, manifest: ModelManifest, *, now_ns: int | None = None
) -> None:
    now = time.time_ns() if now_ns is None else now_ns
    due = due_snapshot(root, now_ns=now)
    if due is None:
        return
    seconds, started = due
    plan = load_plan(root)
    assert plan is not None
    destination = root / "strength-recovery-snapshots" / str(seconds)
    original_manifest = manifest.artifact_manifest or manifest.path
    artifacts = {}
    for name, source in (
        ("checkpoint", manifest.checkpoint),
        ("manifest", original_manifest),
    ):
        target = (
            destination
            / ("checkpoints" if name == "checkpoint" else "manifests")
            / source.name
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = sha256_file(source)
        size = source.stat().st_size
        try:
            os.link(source, target)
        except FileExistsError:
            pass
        verify_file(target, expected_sha256=expected, expected_bytes=size)
        artifacts[name] = {"path": str(target), "sha256": expected, "bytes": size}
    missed = [
        earlier
        for earlier in SCHEDULE_SECONDS
        if earlier < seconds
        and not (
            root / "strength-recovery-snapshots" / str(earlier) / "snapshot.json"
        ).exists()
    ]
    atomic_json(
        destination / "snapshot.json",
        {
            "schema_version": 1,
            "plan_sha256": plan["plan_sha256"],
            "scheduled_seconds": seconds,
            "measurement_started_ns": started,
            "captured_ns": now,
            "actual_elapsed_seconds": (now - started) / 1e9,
            "missed_earlier_endpoints": missed,
            "model_identity": manifest.model_identity,
            "model_step": manifest.model_step,
            "artifacts": artifacts,
            "interpretation": "First publication at or after the wall-clock endpoint; no strength claim.",
        },
    )
    (destination / "snapshot.json").chmod(0o444)
