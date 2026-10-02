#!/usr/bin/env python3
"""Run a pinned recovery arm under the existing resumable hard budget runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.prepare_strength_recovery import verify_artifact
from scripts.run_elo_ablation import run_elo_ablation
from startrain.config import load_config
from startrain.search_allocation_gate import validate_production_ring_allocations
from startrain.strength_recovery import load_plan


def run(
    *, root: Path, orchestrator: str, poll_seconds: float = 1.0
) -> dict[str, object]:
    root = root.expanduser().resolve()
    plan = load_plan(root)
    if plan is None:
        raise ValueError("run has no frozen recovery plan")
    profile = root / "profile-elo-ablation.yaml"
    verify_artifact({**plan["profile"], "path": str(profile)})
    for pin in plan["implementation_pins"]:
        verify_artifact(pin)
    config = load_config(profile)
    validate_production_ring_allocations(config, _fresh=True)
    metadata = json.loads((root / "ablation.json").read_text())
    if metadata.get("wall_budget_seconds") != plan["wall_budget_seconds"]:
        raise ValueError("recovery wall budget changed")
    if metadata.get("measurement_status") == "complete":
        # An orchestration supervisor may retry after losing our final output.
        # Budget completion is idempotent and must never start another run.
        return {
            "status": "already_complete",
            "run_root": str(root),
            "resource_released_ns": metadata.get("resource_released_ns"),
        }
    return run_elo_ablation(
        config_path=profile, orchestrator=orchestrator, poll_seconds=poll_seconds
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--orchestrator", default="startrain-orchestrate")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                root=args.run_root,
                orchestrator=args.orchestrator,
                poll_seconds=args.poll_seconds,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
