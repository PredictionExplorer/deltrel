#!/usr/bin/env python3
"""Compare explicit-rate recovery diagnostics using paired games within cells."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random

from startrain.runtime import atomic_json
from scripts.recovery_label_compatibility import normalize_implementation, verify_bridge
from scripts.run_frozen_replay_optimizer_calibration import (
    FORMAT,
    RECOVERY_ARMS,
    RECOVERY_CELLS,
    RECOVERY_PARTITION,
    SCHEMA_VERSION,
    _digest,
    _pin,
)


def _number(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("recovery observation is not finite numeric evidence")
    return float(value)


def _observations(result: dict, *, raw: bool = False) -> dict[str, dict]:
    heldout = (
        result["recovery_diagnostics"]["raw_heldout"] if raw else result["heldout"]
    )
    if heldout.get("finite") is not True:
        raise ValueError("recovery heldout must be finite")
    rows = heldout.get("observations")
    if not isinstance(rows, list) or not rows:
        raise ValueError("recovery game observations missing")
    games = {}
    for row in rows:
        identity = row.get("game_identity")
        if not isinstance(identity, str) or not identity or identity in games:
            raise ValueError("recovery game observations must be unique")
        if (
            row.get("cell") not in RECOVERY_CELLS
            or type(row.get("samples")) is not int
            or row["samples"] <= 0
        ):
            raise ValueError("recovery game cell/sample count invalid")
        if result["partition"]["game_cells"].get(identity) != row["cell"]:
            raise ValueError("recovery observation cell differs from frozen partition")
        for side in ("reference", "candidate"):
            for key in ("policy", "value", "composite"):
                _number(row[side][key])
        games[identity] = row
    if set(games) != set(result["partition"]["holdout_game_ids"]):
        raise ValueError("observations differ from frozen holdout games")
    if {row["cell"] for row in rows} != set(RECOVERY_CELLS):
        raise ValueError("all four recovery cells are required")
    return games


def _read(path: Path) -> tuple[dict, dict]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("recovery result must be a regular non-symlink file")
    result = json.loads(path.read_text())
    unsigned = dict(result)
    digest = unsigned.pop("result_sha256", None)
    if _digest(unsigned) != digest:
        raise ValueError("recovery result semantic hash failed")
    if (
        result.get("format") != FORMAT
        or result.get("schema_version") != SCHEMA_VERSION
        or result.get("status") != "complete"
        or result.get("arm") not in RECOVERY_ARMS
        or result.get("training", {}).get("finite") is not True
    ):
        raise ValueError("completed finite recovery result required")
    partition = result["partition"]
    if (
        partition.get("method") != RECOVERY_PARTITION
        or partition.get("game_disjoint") is not True
        or partition.get("cell_weights") != RECOVERY_CELLS
        or set(partition["train_game_ids"]) & set(partition["holdout_game_ids"])
    ):
        raise ValueError("recovery partition must be cell-stratified and game-disjoint")
    training = result["training"]
    if training.get("effective_rate_contract") != "explicit-constant-no-governor-v1":
        raise ValueError("explicit constant effective-rate contract required")
    expected = [
        training["effective_muon_lr"],
        training["effective_adamw_lr"],
        training["effective_adamw_lr"],
    ]
    if result["recovery_diagnostics"]["actual_final_learning_rates"] != expected:
        raise ValueError("actual rates differ from declared effective rates")
    if any(_number(value) <= 0 for value in expected):
        raise ValueError("effective rates must be positive")
    for key in ("fresh_from_champion_ema",):
        if result["optimizer"].get(key) is not True:
            raise ValueError("recovery did not initialize from champion EMA")
    if (
        result["optimizer"].get("source_optimizer_loaded") is not False
        or result["optimizer"].get("source_scheduler_loaded") is not False
    ):
        raise ValueError("recovery inherited source optimization state")
    _observations(result)
    _observations(result, raw=True)
    return result, _pin(path).as_dict()


def _common_contract(result: dict, *, label_only_bridge: dict | None = None) -> dict:
    config = copy.deepcopy(result["config_contract"])
    config["optimizer"].pop("muon_lr")
    config["optimizer"].pop("adamw_lr")
    recovery = copy.deepcopy(result["recovery"])
    if label_only_bridge is not None:
        recovery["implementation"] = normalize_implementation(
            recovery["implementation"], label_only_bridge
        )
    return {
        "config": config,
        "partition": result["partition"],
        "recovery": recovery,
        "champion_identity": result["champion"]["model_identity"],
        "champion_checkpoint": result["champion"]["checkpoint"]["sha256"],
        "cutoff": result["replay"]["cutoff_sha256"],
        "training": {
            key: result["training"][key]
            for key in (
                "steps",
                "completed_steps",
                "batch_size",
                "seed",
                "gradient_diagnostic_rows",
            )
        },
        "evaluation": result["evaluation"],
        "runtime": {
            "device_type": result["device"]["resolved"].split(":")[0],
            "precision": result["device"]["precision"],
            "compile": result["device"]["compile"],
            "hardware_name": result["device"].get("hardware", {}).get("name"),
        },
    }


def paired_cell_comparison(
    control: dict[str, dict],
    treatment: dict[str, dict],
    *,
    alpha: float,
    samples: int,
    seed: int,
    minimum_games: int,
) -> dict:
    if set(control) != set(treatment):
        raise ValueError("recovery arms evaluated different games")
    cells = {}
    for identity, before in control.items():
        after = treatment[identity]
        if any(before[key] != after[key] for key in ("cell", "samples", "reference")):
            raise ValueError("recovery arms differ in paired game/reference evidence")
        cells.setdefault(before["cell"], []).append(
            (
                before["samples"],
                _number(before["candidate"]["composite"])
                - _number(after["candidate"]["composite"]),
            )
        )
    if set(cells) != set(RECOVERY_CELLS):
        raise ValueError("recovery comparison lacks a required cell")

    def mean(rows: list[tuple[int, float]]) -> float:
        return sum(count * delta for count, delta in rows) / sum(
            count for count, _ in rows
        )

    rng = random.Random(seed)
    distribution = sorted(
        sum(
            RECOVERY_CELLS[cell] * mean(rng.choices(rows, k=len(rows)))
            for cell, rows in sorted(cells.items())
        )
        for _ in range(samples)
    )
    lower = distribution[min(samples - 1, int(alpha * samples))]
    covered = all(len(rows) >= minimum_games for rows in cells.values())
    return {
        "weighted_composite_improvement": sum(
            RECOVERY_CELLS[cell] * mean(rows) for cell, rows in cells.items()
        ),
        "one_sided_lower_bound": lower,
        "per_cell": {
            cell: {"games": len(rows), "improvement": mean(rows)}
            for cell, rows in cells.items()
        },
        "minimum_games_per_cell": minimum_games,
        "coverage_sufficient": covered,
        "passes_diagnostic_screen": covered and lower > 0,
    }


def compare(
    paths: list[Path],
    *,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 17,
    minimum_games: int = 8,
    label_only_bridge: Path | None = None,
) -> dict:
    if (
        not 100 <= bootstrap_samples <= 100000
        or not 0.5 < confidence < 1
        or minimum_games < 2
    ):
        raise ValueError("invalid recovery bootstrap settings")
    pairs = [_read(path) for path in paths]
    results = {result["arm"]: result for result, _ in pairs}
    if len(results) != len(paths) or len(paths) < 2 or RECOVERY_ARMS[0] not in results:
        raise ValueError("unique recovery control and at least one treatment required")
    control = results[RECOVERY_ARMS[0]]
    bridge = verify_bridge(label_only_bridge) if label_only_bridge is not None else None
    if bridge is not None:
        control_pin = next(
            pin for result, pin in pairs if result["arm"] == RECOVERY_ARMS[0]
        )
        expected = bridge["document"]["control_result"]
        if any(control_pin[key] != expected[key] for key in ("sha256", "bytes")):
            raise ValueError(
                "comparison control is not the immutable control pinned by the label-only bridge"
            )
    contract = _common_contract(control, label_only_bridge=bridge)
    comparisons = {}
    alpha = (1 - confidence) / (len(results) - 1)
    for arm, result in results.items():
        if _common_contract(result, label_only_bridge=bridge) != contract:
            raise ValueError("recovery arms differ beyond explicit learning rates")
        if arm == RECOVERY_ARMS[0]:
            continue
        comparisons[arm] = {
            name: paired_cell_comparison(
                _observations(control, raw=raw),
                _observations(result, raw=raw),
                alpha=alpha,
                samples=bootstrap_samples,
                seed=seed,
                minimum_games=minimum_games,
            )
            for name, raw in (("ema", False), ("raw", True))
        }
    passing = [
        (values["ema"]["weighted_composite_improvement"], arm)
        for arm, values in comparisons.items()
        if values["ema"]["passes_diagnostic_screen"]
    ]
    best_score = max((score for score, _ in passing), default=None)
    best = [arm for score, arm in passing if score == best_score]
    selected = best[0] if len(best) == 1 else None
    if label_only_bridge is not None and verify_bridge(label_only_bridge) != bridge:
        raise ValueError("label-only source evidence changed during comparison")
    return {
        "format": "startrain.recovery-calibration-comparison",
        "schema_version": 1,
        "diagnostic_only": True,
        "production_promotion_authorized": False,
        "source_results": [pin for _, pin in pairs],
        "comparisons": comparisons,
        "suggested_strength_screen_arm": selected,
        "primary_endpoint": "EMA heldout composite, cell-weighted paired-game bootstrap",
        "confidence": confidence,
        "per_treatment_alpha": alpha,
        "bootstrap_samples": bootstrap_samples,
        **({"label_only_compatibility": bridge} if bridge is not None else {}),
        "limitations": "Exploratory fixed-data calibration; raw results are descriptive. Playing strength still requires frozen paired arena evaluation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--minimum-games-per-cell", type=int, default=8)
    parser.add_argument("--label-only-bridge", type=Path)
    args = parser.parse_args()
    result = compare(
        args.result,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        minimum_games=args.minimum_games_per_cell,
        label_only_bridge=args.label_only_bridge,
    )
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
