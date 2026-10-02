#!/usr/bin/env python3
"""Freeze/resume an endpoint versus the original champion at 256 or 1024 search."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from scripts import compare_checkpoint_averaging as diagnostic
from scripts.prepare_strength_recovery import artifact, verify_artifact
from deltreltrain.checkpoint import load_model_manifest
from deltreltrain.config import load_config
from deltreltrain.runtime import SignalLatch, atomic_json
from deltreltrain.strength_recovery import digest, load_plan

ENDPOINT_PLAN = "endpoint-plan.json"


def freeze(
    *,
    root: Path,
    snapshot_seconds: int,
    output: Path,
    simulations: int,
    pairs_per_cell: int,
    wall_budget_hours: float,
) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    plan = load_plan(root)
    if plan is None or snapshot_seconds not in plan["schedule_seconds"]:
        raise ValueError("select a predeclared recovery endpoint")
    if simulations not in (256, 1024):
        raise ValueError("endpoint search must be 256 or 1024 simulations")
    if not math.isfinite(wall_budget_hours) or not 0 < wall_budget_hours <= 24:
        raise ValueError("endpoint wall budget must be in (0, 24] hours")
    snapshot_file = (
        root / "strength-recovery-snapshots" / str(snapshot_seconds) / "snapshot.json"
    )
    snapshot = json.loads(snapshot_file.read_text())
    if snapshot.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("snapshot belongs to a different recovery experiment")
    for pin in snapshot["artifacts"].values():
        verify_artifact(pin)
    source = load_config(root / "profile-elo-ablation.yaml")
    if source.train.precision not in ("fp32", "bf16"):
        raise ValueError("endpoint precision must be explicit, not auto")
    # Anchor the immutable starting champion even if promotion has since moved
    # champion.json. The fork preserves its immutable manifest/checkpoint bytes.
    anchor_path = root / "learner/manifests" / plan["anchor_manifest_name"]
    champion = load_model_manifest(anchor_path)
    if champion.model_identity != plan["anchor_identity"]:
        raise ValueError("independent endpoint anchor differs")
    config = diagnostic.diagnostic_config(
        pairs_per_cell=pairs_per_cell,
        simulations=simulations,
        seed=source.arena.seed,
        max_considered=source.arena.max_considered,
    )
    # The reused runner currently supports these production constants only.
    if config.c_visit != source.arena.c_visit or config.c_scale != source.arena.c_scale:
        raise ValueError(
            "endpoint evaluator cannot silently change the search constants"
        )
    frozen = diagnostic.freeze_plan(
        source_run_root=root,
        checkpoint=Path(snapshot["artifacts"]["checkpoint"]["path"]),
        champion_checkpoint=champion.checkpoint,
        output=output,
        config=config,
        precision=source.train.precision,
    )
    receipt = {
        "schema_version": 1,
        "recovery_plan_sha256": plan["plan_sha256"],
        "diagnostic_plan_sha256": frozen["plan_sha256"],
        "snapshot": artifact(snapshot_file),
        "simulations": simulations,
        "pairs_per_cell": pairs_per_cell,
        "wall_budget_seconds": wall_budget_hours * 3600,
        "anchor_identity": champion.model_identity,
        "candidate_identity": snapshot["model_identity"],
        "snapshot_seconds": snapshot_seconds,
        "selection_scope": "fixed endpoint; no automatic promotion or strength claim",
    }
    receipt["plan_sha256"] = digest(receipt)
    atomic_json(output / ENDPOINT_PLAN, receipt)
    (output / ENDPOINT_PLAN).chmod(0o444)
    return receipt


def run(
    *,
    output: Path,
    device: str,
    exclusive_device: bool,
    session_seconds: float = 300.0,
) -> dict[str, Any]:
    """Bounded resumable sessions; elapsed downtime consumes the fixed budget."""
    output = output.resolve()
    receipt = json.loads((output / ENDPOINT_PLAN).read_text())
    body = {key: value for key, value in receipt.items() if key != "plan_sha256"}
    if receipt.get("schema_version") != 1 or receipt.get("plan_sha256") != digest(body):
        raise ValueError("endpoint plan hash changed")
    frozen, _ = diagnostic.verify_plan(output)
    if frozen["plan_sha256"] != receipt["diagnostic_plan_sha256"]:
        raise ValueError("endpoint inputs differ from the frozen diagnostic")
    if not math.isfinite(session_seconds) or not 0 < session_seconds <= 3600:
        raise ValueError("session seconds must be in (0, 3600]")
    with diagnostic._exclusive_output(output):
        clock_path = output / "endpoint-clock.json"
        if not clock_path.exists():
            atomic_json(
                clock_path,
                {
                    "plan_sha256": receipt["plan_sha256"],
                    "started_ns": time.time_ns(),
                },
            )
            clock_path.chmod(0o444)
        clock = json.loads(clock_path.read_text())
        if (
            clock.get("plan_sha256") != receipt["plan_sha256"]
            or type(clock.get("started_ns")) is not int
            or not 0 < clock["started_ns"] <= time.time_ns()
        ):
            raise ValueError("endpoint budget clock is invalid")
        state_path = output / "endpoint-state.json"
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {
                "schema_version": 1,
                "plan_sha256": receipt["plan_sha256"],
                "started_ns": clock["started_ns"],
                "status": "running",
            }
        )
        if (
            state.get("plan_sha256") != receipt["plan_sha256"]
            or state.get("started_ns") != clock["started_ns"]
        ):
            raise ValueError("endpoint state belongs to another plan")
        if state.get("status") in ("complete", "budget_exhausted"):
            return state
        atomic_json(state_path, state)
        elapsed = (time.time_ns() - state["started_ns"]) / 1e9
        remaining = receipt["wall_budget_seconds"] - elapsed
        if remaining <= 0:
            state.update(
                status="budget_exhausted",
                completed_ns=time.time_ns(),
                interpretation="Incomplete evidence; budget exhaustion is not a verdict.",
            )
            atomic_json(state_path, state)
            return state
        latch = SignalLatch()
        # The existing diagnostic persists game actions at each safe boundary.
        import signal

        previous = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
        latch.install()
        try:
            summary = diagnostic.run_session(
                output=output,
                device=torch.device(device),
                exclusive_device=exclusive_device,
                session_seconds=min(session_seconds, remaining),
                arm="ema",
                stop_requested=latch.is_set,
            )
        finally:
            signal.signal(signal.SIGINT, previous[0])
            signal.signal(signal.SIGTERM, previous[1])
        terminal = summary["arms"]["ema"]["terminal"]
        state.update(
            status="complete" if terminal else "running",
            updated_ns=time.time_ns(),
            completed_pairs=summary["arms"]["ema"]["completed_pairs"],
            result=str(output / "ema.json"),
            interpretation="Complete fixed paired budget."
            if terminal
            else "Partial paired evidence; resume within the original budget.",
        )
        atomic_json(state_path, state)
        return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("prepare")
    create.add_argument("--run-root", type=Path, required=True)
    create.add_argument("--snapshot-seconds", type=int, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--simulations", type=int, choices=(256, 1024), required=True)
    create.add_argument("--pairs-per-cell", type=int, default=4)
    create.add_argument("--wall-budget-hours", type=float, required=True)
    session = sub.add_parser("run")
    session.add_argument("--output", type=Path, required=True)
    session.add_argument("--device", default="cuda:0")
    session.add_argument("--exclusive-device", action="store_true")
    session.add_argument("--session-seconds", type=float, default=300)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        args["root"] = args.pop("run_root")
        result = freeze(**args)
    else:
        result = run(**args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
