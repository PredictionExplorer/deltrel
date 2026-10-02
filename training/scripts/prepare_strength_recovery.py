#!/usr/bin/env python3
"""Freeze and prepare a stopped champion restart with a twelve-hour budget."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

import yaml

from scripts.fork_elo_ablation import fork_elo_ablation
from scripts.prepare_champion_warm_start import prepare_champion_warm_start
from startrain.balanced_evaluation import evaluation_contract
from startrain.checkpoint import load_model_manifest, sha256_file, verify_file
from startrain.config import load_config
from startrain.contracts import SEARCH_ALGORITHM_ID
from startrain.runtime import atomic_json
from startrain.search_allocation_gate import validate_production_ring_allocations
from startrain.strength_recovery import (
    FORMAT,
    PLAN_NAME,
    SCHEDULE_SECONDS,
    digest,
    recovery_config,
)


def artifact(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact must be a regular file: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def verify_artifact(pin: dict[str, Any]) -> None:
    path = Path(pin["path"])
    if path.is_symlink():
        raise ValueError("pinned artifact became a symlink")
    verify_file(path, expected_sha256=pin["sha256"], expected_bytes=pin["bytes"])


def implementation_pins() -> list[dict[str, Any]]:
    training = Path(__file__).resolve().parents[1]
    return [
        artifact(training / name)
        for name in (
            "scripts/prepare_strength_recovery.py",
            "scripts/run_strength_recovery.py",
            "scripts/run_elo_ablation.py",
            "scripts/fork_elo_ablation.py",
            "scripts/prepare_champion_warm_start.py",
            "startrain/strength_recovery.py",
            "startrain/learner.py",
            "startrain/contracts.py",
        )
    ]


def prepare(
    *,
    source_profile: Path,
    destination: Path,
    output: Path,
    muon_lr: float,
    adamw_lr: float,
    warmup_steps: int,
) -> dict[str, Any]:
    source_profile, destination, output = (
        path.expanduser().resolve() for path in (source_profile, destination, output)
    )
    source_config = load_config(source_profile)
    source = Path(source_config.orchestration.directories.root).resolve()
    for other in (destination, output):
        if other == source or source in other.parents or other in source.parents:
            raise ValueError("recovery destination/output must be separate from source")
    if (
        destination == output
        or destination in output.parents
        or output in destination.parents
    ):
        raise ValueError("plan output and destination must be separate")
    if destination.exists() or output.exists():
        raise FileExistsError("recovery output or destination already exists")
    if (source / "coordinator.lock").exists():
        raise ValueError("stop source coordinator before freezing its replay boundary")
    champion = load_model_manifest(source / "learner/champion.json")
    if champion.run_id != source_config.orchestration.run_id:
        raise ValueError("champion identity does not match source profile")
    database = source / "replay/manifest.sqlite3"
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        watermark = connection.execute(
            "SELECT COALESCE(MAX(id), 0) FROM shards"
        ).fetchone()[0]
    pins = [
        artifact(path)
        for path in (
            source_profile,
            source / "run.json",
            source / "learner/champion.json",
            champion.artifact_manifest or champion.path,
            champion.checkpoint,
            database,
        )
    ]
    target = recovery_config(
        source_config,
        destination=destination,
        replay_watermark=watermark,
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        warmup_steps=warmup_steps,
    )
    # The conservative allocation requires no inherited low-full-search waiver.
    # Any future treatment that does require one must pass the real validator.
    validate_production_ring_allocations(target, _fresh=True)
    output.mkdir(parents=True)
    profile = output / "profile.yaml"
    profile.write_text(
        yaml.safe_dump(json.loads(json.dumps(target.as_dict())), sort_keys=False)
    )
    profile.chmod(0o444)
    if load_config(profile) != target:
        raise ValueError("recovery profile did not round-trip")
    plan: dict[str, Any] = {
        "format": FORMAT,
        "schema_version": 1,
        "search_algorithm": SEARCH_ALGORITHM_ID,
        "source_run_root": str(source),
        "run_root": str(destination),
        "source_pins": pins,
        "implementation_pins": implementation_pins(),
        "profile": artifact(profile),
        "profile_sha256": sha256_file(profile),
        "schedule_seconds": list(SCHEDULE_SECONDS),
        "wall_budget_seconds": SCHEDULE_SECONDS[-1],
        "leaf_budget": 10**15,
        "initial_replay_credit": 0,
        "replay_watermark": watermark,
        "anchor_identity": champion.model_identity,
        "anchor_manifest_name": (champion.artifact_manifest or champion.path).name,
        "anchor_step": champion.model_step,
        "created_ns": time.time_ns(),
        "classification": "coupled-champion-restart-and-higher-search-quality-screen",
        "strength_improvement_established": False,
        "compute_contract": "Eight provisioned GPUs; elapsed clock includes restarts and teardown is reported separately.",
    }
    plan["plan_sha256"] = digest(plan)
    atomic_json(output / PLAN_NAME, plan)
    (output / PLAN_NAME).chmod(0o444)
    for pin in pins:
        verify_artifact(pin)
    return plan


def verify_preparation(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("recovery plan must not be a symlink")
    plan = json.loads(path.read_text())
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (
        plan.get("format") != FORMAT
        or plan.get("schema_version") != 1
        or plan.get("plan_sha256") != digest(body)
        or plan.get("search_algorithm") != SEARCH_ALGORITHM_ID
        or plan.get("schedule_seconds") != list(SCHEDULE_SECONDS)
        or plan.get("wall_budget_seconds") != SCHEDULE_SECONDS[-1]
        or plan.get("initial_replay_credit") != 0
    ):
        raise ValueError("recovery preparation identity/hash/budget differs")
    for pin in [plan["profile"], *plan["source_pins"], *plan["implementation_pins"]]:
        verify_artifact(pin)
    return plan


def apply(plan_path: Path) -> dict[str, Any]:
    plan = verify_preparation(plan_path)
    source, destination = Path(plan["source_run_root"]), Path(plan["run_root"])
    if (source / "coordinator.lock").exists():
        raise ValueError("source was restarted after recovery preparation")
    profile = Path(plan["profile"]["path"])
    config = load_config(profile)
    ablation_plan = {
        "report": "startrain-elo-ablation-plan",
        "schema_version": 1,
        "initialization": "fork",
        "source_run_root": str(source),
        "training_objective": config.orchestration.training_objective,
        "promotion_objective": "weighted_aggregate",
        "guard_rings": [],
        "per_ring_guarantees": False,
        "wall_budget_seconds": plan["wall_budget_seconds"],
        "leaf_budget": plan["leaf_budget"],
        "treatments": [
            {
                "treatment": "strength-recovery",
                "profile": str(profile),
                "profile_sha256": plan["profile_sha256"],
                "run_root": str(destination),
            }
        ],
    }
    fork_plan = plan_path.parent / "fork-plan.json"
    if fork_plan.exists():
        if json.loads(fork_plan.read_text()) != ablation_plan:
            raise ValueError("existing fork plan differs")
    else:
        atomic_json(fork_plan, ablation_plan)
    fork_elo_ablation(
        source_run_root=source, plan_path=fork_plan, treatment="strength-recovery"
    )
    # Old evaluation contracts, pending leases and plateau verdicts cannot join
    # the corrected-search experiment, even though checkpoint schemas match.
    arena = destination / "arena"
    if arena.exists():
        arena.rename(destination / "ablation-parent/arena")
    arena.mkdir()
    epoch = destination / "strength-epoch.json"
    if epoch.exists():
        epoch.rename(destination / "ablation-parent/strength-epoch.json")
    installed = destination / "profile-elo-ablation.yaml"
    warm = prepare_champion_warm_start(
        destination,
        installed,
        apply=True,
        initial_replay_credit=0,
        replace_existing=True,
    )
    champion = load_model_manifest(destination / "learner/champion.json")
    if champion.model_identity != plan["anchor_identity"]:
        raise ValueError("fork champion differs from pinned recovery anchor")
    strength = replace(config.arena, simulations=1024, allocation_policy="equal_cells")
    atomic_json(
        epoch,
        {
            "schema_version": 1,
            "started_ns": time.time_ns(),
            "minimum_candidate_step": champion.model_step,
            "anchor_identity": champion.model_identity,
            "anchor_manifest": str(champion.artifact_manifest or champion.path),
            "evaluation_contract_identity": evaluation_contract(strength)["identity"],
            "reason": "isolated_champion_strength_recovery",
        },
    )
    atomic_json(destination / PLAN_NAME, plan)
    (destination / PLAN_NAME).chmod(0o444)
    for pin in plan["source_pins"]:
        verify_artifact(pin)
    return {
        "status": "prepared",
        "run_root": str(destination),
        "warm_start": warm,
        "profile": str(installed),
        "plan_sha256": plan["plan_sha256"],
        "training_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("prepare")
    for name in ("source-profile", "destination", "output"):
        freeze.add_argument("--" + name, type=Path, required=True)
    for name in ("muon-lr", "adamw-lr"):
        freeze.add_argument("--" + name, type=float, required=True)
    freeze.add_argument("--warmup-steps", type=int, required=True)
    activate = sub.add_parser("apply")
    activate.add_argument("--plan", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = apply(args["plan"]) if command == "apply" else prepare(**args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
