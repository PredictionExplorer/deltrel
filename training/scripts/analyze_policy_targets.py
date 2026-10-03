#!/usr/bin/env python3
"""Compare target-only treatments on a frozen search sweep, never claim Elo.

Run benchmark_search_budgets first. Its immutable model, input and search pins
remain part of this report. Only reference-visited actions can supply measured
Q; missing coverage is explicitly unavailable rather than assumed zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from deltreltrain.policy_targets import constrain_policy_target


def compare_target(
    record: dict, reference: dict, *, scale: float, max_kl: float | None
) -> dict:
    if record["actions"] != reference["actions"]:
        raise ValueError("reference action support differs")
    target = constrain_policy_target(
        np.asarray(record["policy_target"]),
        np.asarray(record["priors"]),
        scale=scale,
        max_kl=max_kl,
    )
    probabilities = target.probabilities.astype(np.float64)
    probabilities /= probabilities.sum()
    reference_policy = np.asarray(reference["policy_target"], dtype=np.float64)
    reference_q = np.asarray(reference["q_values"], dtype=np.float64)
    visits = np.asarray(reference["visits"])
    if any(
        v.shape != probabilities.shape for v in (reference_policy, reference_q, visits)
    ):
        raise ValueError("reference vectors have mismatched lengths")
    if (
        not np.isfinite(reference_q).all()
        or not np.isfinite(reference_policy).all()
        or (reference_policy < 0).any()
        or reference_policy.sum() <= 0
        or not np.issubdtype(visits.dtype, np.integer)
        or (visits < 0).any()
    ):
        raise ValueError("reference contains invalid measurements")
    reference_policy /= reference_policy.sum()
    measured = visits > 0
    coverage = float(probabilities[measured].sum())
    positive = probabilities > 0
    best_q = float(reference_q[measured].max()) if measured.any() else None
    return {
        "scale": scale,
        "max_kl": max_kl,
        "applied_scale": target.applied_scale,
        "kl_to_prior_nats": target.kl_nats if np.isfinite(target.kl_nats) else None,
        "entropy_nats": -float(
            np.sum(probabilities[positive] * np.log(probabilities[positive]))
        ),
        "max_probability": float(probabilities.max()),
        "l1_to_reference_target": float(np.abs(probabilities - reference_policy).sum()),
        "reference_visited_target_mass": coverage,
        "reference_expected_regret": (
            max(0.0, best_q - float(probabilities @ reference_q))
            if best_q is not None and coverage >= 1 - 1e-8
            else None
        ),
        "reference_regret_scope": "among reference-visited moves; unavailable without full target coverage",
        "selected_action_unchanged": record["selected_action"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    parser.add_argument("--max-kl", type=float, nargs="*", default=[0.5, 0.1])
    args = parser.parse_args(argv)
    if args.output.exists() or args.search_report.stat().st_size > 256 * 1024**2:
        parser.error("output must be new and input at most 256 MiB")
    raw = args.search_report.read_bytes()
    report = json.loads(raw)
    references = {}
    for row in report["results"]:
        if row["arm"] == "reference":
            key = (row["position_id"], row["repeat"])
            if key in references:
                raise ValueError("duplicate frozen reference")
            references[key] = row
    treatments = [(scale, None) for scale in args.scales] + [
        (1.0, kl) for kl in args.max_kl
    ]
    if not treatments or len(treatments) > 16:
        parser.error("require 1..16 declared treatments")
    rows = []
    for row in report["results"]:
        if row["arm"] == "reference":
            continue
        reference = references[(row["position_id"], row["repeat"])]
        for scale, limit in treatments:
            rows.append(
                {
                    "position_id": row["position_id"],
                    "repeat": row["repeat"],
                    "search_arm": row["arm"],
                    **compare_target(row, reference, scale=scale, max_kl=limit),
                }
            )
    if not rows:
        raise ValueError("search report contains no comparable measurements")
    result = {
        "format": "deltreltrain.policy-target-calibration",
        "schema_version": 1,
        "search_report_sha256": hashlib.sha256(raw).hexdigest(),
        "search_plan": report["plan"],
        "results": rows,
        "scope": "fixed-position target sensitivity; approximate deeper search is not ground truth or Elo",
        "automatic_adoption": False,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "comparisons": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
