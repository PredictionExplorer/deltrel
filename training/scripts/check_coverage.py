#!/usr/bin/env python3
"""Enforce risk-weighted Python coverage floors from coverage.py JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MINIMUM_TOTAL = 74.0
FILE_FLOORS = {
    "deltrelserve/app.py": 80.0,
    "deltrelserve/config.py": 50.0,
    "deltrelserve/runtime.py": 75.0,
    "deltrelserve/schemas.py": 90.0,
    "deltreltrain/actor.py": 70.0,
    "deltreltrain/arena.py": 85.0,
    "deltreltrain/checkpoint.py": 68.0,
    "deltreltrain/cli.py": 30.0,
    "deltreltrain/config.py": 70.0,
    "deltreltrain/distill.py": 65.0,
    "deltreltrain/features.py": 80.0,
    "deltreltrain/inference.py": 80.0,
    "deltreltrain/learner.py": 60.0,
    "deltreltrain/losses.py": 90.0,
    "deltreltrain/model.py": 90.0,
    "deltreltrain/native.py": 78.0,
    "deltreltrain/optim.py": 40.0,
    "deltreltrain/orchestration.py": 68.0,
    "deltreltrain/promotion.py": 65.0,
    "deltreltrain/publish.py": 78.0,
    "deltreltrain/replay.py": 77.0,
    "deltreltrain/replay_store.py": 77.0,
    "deltreltrain/runtime.py": 80.0,
    "deltreltrain/sampling.py": 80.0,
    "deltreltrain/scoring.py": 95.0,
    "deltreltrain/selfplay.py": 82.0,
    "deltreltrain/topology.py": 88.0,
    "deltreltrain/training.py": 84.0,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    arguments = parser.parse_args()
    with arguments.report.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    files = report.get("files")
    totals = report.get("totals")
    if not isinstance(files, dict) or not isinstance(totals, dict):
        raise SystemExit("coverage report has an invalid schema")

    failures = []
    total = float(totals.get("percent_covered", 0.0))
    if total < MINIMUM_TOTAL:
        failures.append(f"overall coverage {total:.2f}% is below {MINIMUM_TOTAL:.2f}%")
    for path, minimum in FILE_FLOORS.items():
        entry = files.get(path)
        if not isinstance(entry, dict) or not isinstance(entry.get("summary"), dict):
            failures.append(f"required coverage entry is missing: {path}")
            continue
        actual = float(entry["summary"].get("percent_covered", 0.0))
        if actual < minimum:
            failures.append(f"{path}: {actual:.2f}% is below {minimum:.2f}%")

    print(
        json.dumps(
            {
                "schema_version": 1,
                "overall": total,
                "minimum_overall": MINIMUM_TOTAL,
                "file_floors": len(FILE_FLOORS),
                "passed": not failures,
            },
            sort_keys=True,
        )
    )
    if failures:
        for failure in failures:
            print(f"coverage gate: {failure}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
