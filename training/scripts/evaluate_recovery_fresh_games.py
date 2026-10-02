#!/usr/bin/env python3
"""Score immutable checkpoints on whole-game, zero-PDA fresh pie positions.

No training, search, replay publication, or promotion. These are paired loss
observations on two even-game cells, not arena games or a four-cell Elo result.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time

import torch

from deltreltrain.checkpoint import load_checkpoint, load_model_manifest, verify_file
from deltreltrain.config import ExperimentConfig, load_config
from deltreltrain.contracts import SEARCH_ALGORITHM_ID, TARGET_OUTCOME
from deltreltrain.device import enable_fast_math, resolve_precision, synchronize_device
from deltreltrain.model import GraphResTNet
from deltreltrain.replay import (
    ReplayBatch,
    ReplaySample,
    collate_replay_samples,
    decode_replay_shard,
)
from deltreltrain.runtime import atomic_json, validate_identifier
from scripts.run_frozen_replay_optimizer_calibration import (
    _component_loss_totals,
    _components_from_totals,
    _digest,
    _pin,
    _recovery_implementation,
    _replay_game_identity,
)

CELLS = ("pie-classic", "pie-double")
COMPONENTS = ("policy", "value", "composite")


def _verify(pin: dict) -> None:
    verify_file(
        Path(pin["path"]), expected_sha256=pin["sha256"], expected_bytes=pin["bytes"]
    )


def _semantic(path: Path) -> dict:
    document = json.loads(path.read_text())
    unsigned = dict(document)
    digest = unsigned.pop("sha256", None)
    if _digest(unsigned) != digest:
        raise ValueError(f"fresh-data semantic pin failed: {path}")
    return document


@dataclass(frozen=True)
class FreshGame:
    identity: str
    game_id: str
    cell: str
    samples: tuple[ReplaySample, ...]


def load_fresh_games(allowlist_path: Path) -> tuple[tuple[FreshGame, ...], dict]:
    allowlist_pin = _pin(allowlist_path).as_dict()
    document = _semantic(allowlist_path)
    if (
        document.get("schema_version") != 1
        or document.get("search_algorithm") != SEARCH_ALGORITHM_ID
    ):
        raise ValueError("fresh data must use the current corrected search contract")
    games: dict[str, FreshGame] = {}
    teachers: set[str] = set()
    pins = [allowlist_pin]
    for source in document["sources"]:
        root = Path(source["root"])
        plan_path, summaries_path = root / "collection-plan.json", root / "games.json"
        plan = _semantic(plan_path)
        teachers.add(plan["champion_identity"])
        if (
            plan["sha256"] != source["plan_sha256"]
            or plan["search_algorithm"] != document["search_algorithm"]
        ):
            raise ValueError("fresh collection plan differs from allowlist")
        summary_pin = _pin(summaries_path).as_dict()
        if summary_pin["sha256"] != source["source_games_sha256"]:
            raise ValueError("fresh game summary pin changed")
        pins.extend([_pin(plan_path).as_dict(), summary_pin])
        for pin in (
            *plan["input_pins"],
            *plan["implementation_pins"],
            *plan["source_pins"],
        ):
            _verify(pin)
            pins.append(pin)
        selected = source["selected_game_ids"]
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("fresh allowlist requires unique game IDs")
        summaries = {
            row["game_id"]: row
            for row in json.loads(summaries_path.read_text())["games"]
        }
        wanted = set(selected)
        if not wanted <= set(summaries):
            raise ValueError("allowlisted games are missing completed summaries")
        for game_id in wanted:
            summary = summaries[game_id]
            if (
                summary["pda_seat0"] != 0
                or summary["pda_seat1"] != 0
                or summary["variant"] not in CELLS
                or summary["model_identity"] != plan["champion_identity"]
            ):
                raise ValueError(
                    "allowlisted summary is not a zero-PDA champion pie game"
                )
        selected_samples: dict[str, dict[int, ReplaySample]] = defaultdict(dict)
        identities: dict[str, str] = {}
        for shard_pin in source["shards"]:
            _verify(shard_pin)
            pins.append(shard_pin)
            decoded = decode_replay_shard(Path(shard_pin["path"]))
            if len(decoded) != shard_pin["all_rows"]:
                raise ValueError("fresh shard row count changed")
            selected_rows = 0
            for index in range(len(decoded)):
                if str(decoded.arrays["game_id"][index]) not in wanted:
                    continue
                sample = decoded.sample(index)
                if (
                    sample.pda != 0
                    or not sample.pie
                    or sample.handicap != 1
                    or sample.rings != 10
                    or not sample.target_mask & TARGET_OUTCOME
                    or sample.model_identity != plan["champion_identity"]
                    or sample.run_id != plan["run_id"]
                    or sample.generation_family != plan["generation_family"]
                ):
                    raise ValueError(
                        "selected row is not a finalized zero-PDA champion pie position"
                    )
                cell = "pie-" + sample.mode
                if cell != summaries[sample.game_id]["variant"]:
                    raise ValueError(
                        "fresh row variant differs from its completed game"
                    )
                identity = _replay_game_identity(decoded, index)
                if identities.setdefault(sample.game_id, identity) != identity:
                    raise ValueError("fresh immutable game identity changed")
                if sample.ply in selected_samples[sample.game_id]:
                    raise ValueError(
                        "fresh allowlist duplicates a logical game position"
                    )
                selected_samples[sample.game_id][sample.ply] = sample
                selected_rows += 1
            if selected_rows != shard_pin["selected_rows"]:
                raise ValueError("fresh allowlist selected row count changed")
        for game_id in wanted:
            rows = selected_samples[game_id]
            if len(rows) != summaries[game_id]["samples"]:
                raise ValueError(
                    "fresh holdout requires every position from each completed game"
                )
            identity = identities[game_id]
            if identity in games:
                raise ValueError("fresh games duplicate across sources")
            games[identity] = FreshGame(
                identity,
                game_id,
                summaries[game_id]["variant"],
                tuple(rows[ply] for ply in sorted(rows)),
            )
    counts = Counter(game.cell for game in games.values())
    if len(teachers) != 1:
        raise ValueError("fresh games must share one frozen champion teacher")
    positions = sum(len(game.samples) for game in games.values())
    if set(counts) != set(CELLS) or min(counts.values()) < 8:
        raise ValueError(
            "fresh evaluation requires at least eight games in both pie cells"
        )
    if (
        positions != document["totals"]["positions"]
        or len(games) != document["totals"]["games"]
    ):
        raise ValueError("fresh complete-game totals differ from allowlist")
    return tuple(games[key] for key in sorted(games)), {
        "allowlist": allowlist_pin,
        "allowlist_semantic_sha256": document["sha256"],
        "source_pins": pins,
        "games": len(games),
        "positions": positions,
        "games_by_cell": dict(counts),
        "search_algorithm": document["search_algorithm"],
        "champion_identity": next(iter(teachers)),
        "game_ids": sorted(games),
        "whole_games": True,
        "pda": 0,
        "scope": "even-game-only-corrected-search-loss-validation",
    }


def aggregate(observations: list[dict]) -> dict:
    per_cell = {}
    for cell in CELLS:
        rows = [row for row in observations if row["cell"] == cell]
        if not rows:
            raise ValueError("fresh model evaluation omitted a cell")
        total = sum(row["samples"] for row in rows)
        per_cell[cell] = {
            key: sum(row["samples"] * row["losses"][key] for row in rows) / total
            for key in COMPONENTS
        }
    return {
        "cells": per_cell,
        "even_macro": {
            key: sum(per_cell[cell][key] for cell in CELLS) / 2 for key in COMPONENTS
        },
    }


def paired_even(
    reference: list[dict],
    candidate: list[dict],
    *,
    alpha: float,
    seed: int = 17,
    repeats: int = 10000,
) -> dict:
    before, after = (
        {row["game_identity"]: row for row in rows} for rows in (reference, candidate)
    )
    if (
        len(before) != len(reference)
        or len(after) != len(candidate)
        or set(before) != set(after)
    ):
        raise ValueError("fresh comparisons require the same unique complete games")
    strata = defaultdict(list)
    for identity, row in before.items():
        other = after[identity]
        if any(row[key] != other[key] for key in ("cell", "samples")):
            raise ValueError("fresh paired game cells/counts differ")
        strata[row["cell"]].append(
            (row["samples"], row["losses"]["composite"] - other["losses"]["composite"])
        )
    if set(strata) != set(CELLS):
        raise ValueError("fresh comparisons require both even-game cells")

    def weighted(rows):
        return sum(n * delta for n, delta in rows) / sum(n for n, _ in rows)

    rng = random.Random(seed)
    distribution = sorted(
        sum(weighted(rng.choices(rows, k=len(rows))) for rows in strata.values()) / 2
        for _ in range(repeats)
    )
    return {
        "weighted_composite_improvement": sum(
            weighted(rows) for rows in strata.values()
        )
        / 2,
        "one_sided_lower_bound": distribution[int(alpha * repeats)],
        "per_endpoint_alpha": alpha,
        "bootstrap_samples": repeats,
        "unit": "paired-whole-game-within-even-cell",
    }


def score_model(
    model: GraphResTNet,
    batches: list[tuple[FreshGame, list[ReplayBatch]]],
    config: ExperimentConfig,
    *,
    device: torch.device,
    precision: str,
    check_budget: Callable[[], None],
) -> list[dict]:
    model.eval()
    observations = []
    for game, chunks in batches:
        totals = {
            key: [0.0, 0.0]
            for key in ("policy", "soft_policy", "outcome", "score_margin")
        }
        for batch in chunks:
            check_budget()
            values = _component_loss_totals(
                model, batch, config, device=device, precision=precision
            )
            for key, (numerator, denominator) in values.items():
                totals[key][0] += numerator
                totals[key][1] += denominator
        observations.append(
            {
                "game_identity": game.identity,
                "game_id": game.game_id,
                "cell": game.cell,
                "samples": len(game.samples),
                "losses": _components_from_totals(totals, config),
            }
        )
    return observations


def run(
    *,
    allowlist: Path,
    config_path: Path,
    champion: Path,
    calibration_results: list[Path],
    output_dir: Path,
    device_name: str = "cuda:0",
    batch_size: int = 64,
    budget_seconds: float = 1800,
) -> dict:
    if (
        not 1 <= batch_size <= 256
        or not math.isfinite(budget_seconds)
        or not 0 < budget_seconds <= 3300
    ):
        raise ValueError("invalid bounded fresh evaluation settings")
    started = time.monotonic()

    def check_budget():
        if time.monotonic() - started >= budget_seconds:
            raise TimeoutError("fresh evaluation exhausted its bounded budget")

    games, dataset = load_fresh_games(allowlist)
    config = load_config(config_path)
    config_pin = _pin(config_path).as_dict()
    manifest = load_model_manifest(champion)
    if manifest.model_identity != dataset["champion_identity"]:
        raise ValueError("fresh-data teacher differs from the evaluated champion")
    champion_pin = _pin(manifest.checkpoint).as_dict()
    models = [{"id": "champion-ema", "weights": "ema", "checkpoint": champion_pin}]
    result_pins = []
    for path in calibration_results:
        pin = _pin(path).as_dict()
        result = json.loads(path.read_text())
        unsigned = dict(result)
        result_hash = unsigned.pop("result_sha256", None)
        if _digest(unsigned) != result_hash or result.get("status") != "complete":
            raise ValueError(
                "fresh evaluation requires complete pinned calibration results"
            )
        validate_identifier("calibration arm", result["arm"])
        if result["champion"]["checkpoint"]["sha256"] != champion_pin["sha256"]:
            raise ValueError("calibration and fresh validation champions differ")
        if set(dataset["game_ids"]) & (
            set(result["partition"]["train_game_ids"])
            | set(result["partition"]["holdout_game_ids"])
        ):
            raise ValueError("fresh games overlap old calibration partitions")
        _verify(result["candidate_checkpoint"])
        result_pins.append(pin)
        for weights in ("ema", "raw"):
            models.append(
                {
                    "id": result["arm"] + "-" + weights,
                    "weights": weights,
                    "checkpoint": result["candidate_checkpoint"],
                }
            )
    if len({model["id"] for model in models}) != len(models):
        raise ValueError("fresh evaluation model identities must be unique")
    implementation = _recovery_implementation()
    device = torch.device(device_name)
    precision = resolve_precision(config.train.precision, device)
    contract = {
        "dataset": dataset,
        "config": config_pin,
        "models": models,
        "calibration_results": result_pins,
        "batch_size": batch_size,
        "implementation": implementation,
        "evaluator": _pin(Path(__file__)).as_dict(),
        "runtime": {
            "device_type": device.type,
            "precision": precision,
            "gpu_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != contract:
        raise ValueError("existing fresh evaluation plan differs")
    atomic_json(plan_path, contract)
    materialization_started = time.monotonic()
    batches = [
        (
            game,
            [
                collate_replay_samples(game.samples[start : start + batch_size])
                for start in range(0, len(game.samples), batch_size)
            ],
        )
        for game in games
    ]
    materialization_seconds = time.monotonic() - materialization_started
    enable_fast_math(device)
    results = {}
    for descriptor in models:
        check_budget()
        model_id = descriptor["id"]
        result_path = output_dir / (model_id + ".json")
        if result_path.exists():
            existing = json.loads(result_path.read_text())
            unsigned = dict(existing)
            digest = unsigned.pop("result_sha256", None)
            if (
                existing.get("contract_sha256") != _digest(contract)
                or existing.get("model") != descriptor
                or _digest(unsigned) != digest
            ):
                raise ValueError(
                    "cached fresh model result differs from frozen contract"
                )
            results[model_id] = existing
            continue
        model = GraphResTNet(config.model).to(device)
        pin = descriptor["checkpoint"]
        load_checkpoint(
            Path(pin["path"]),
            model=model,
            use_ema_weights=descriptor["weights"] == "ema",
            require_ema=True,
            expected_model_config=config.as_dict()["model"],
            expected_game_config=config.as_dict()["game"],
            map_location=device,
            expected_sha256=pin["sha256"],
            expected_bytes=pin["bytes"],
        )
        evaluation_started = time.monotonic()
        observations = score_model(
            model,
            batches,
            config,
            device=device,
            precision=precision,
            check_budget=check_budget,
        )
        synchronize_device(device)
        result = {
            "contract_sha256": _digest(contract),
            "model": descriptor,
            "observations": observations,
            **aggregate(observations),
            "evaluation_seconds": time.monotonic() - evaluation_started,
        }
        result["result_sha256"] = _digest(result)
        atomic_json(result_path, result)
        results[model_id] = result
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        atomic_json(
            output_dir / "progress.json",
            {"completed_models": list(results), "total_models": len(models)},
        )
    reference = results["champion-ema"]["observations"]
    comparisons = {
        name: paired_even(
            reference, result["observations"], alpha=0.05 / max(1, len(models) - 1)
        )
        for name, result in results.items()
        if name != "champion-ema"
    }
    for pin in (*dataset["source_pins"], config_pin, champion_pin, *result_pins):
        _verify(pin)
    if _recovery_implementation() != implementation:
        raise ValueError("fresh evaluator dependencies changed during evaluation")
    _verify(contract["evaluator"])
    payload = {
        "format": "deltreltrain.fresh-even-recovery-validation",
        "schema_version": 1,
        "status": "complete",
        "contract_sha256": _digest(contract),
        "contract": contract,
        "models": {
            name: {key: value for key, value in result.items() if key != "observations"}
            for name, result in results.items()
        },
        "comparisons_against_champion": comparisons,
        "elapsed_seconds": time.monotonic() - started,
        "materialization_seconds": materialization_seconds,
        "precision": precision,
        "training_performed": False,
        "production_promotion_authorized": False,
        "limitations": "Even-game-only paired loss validation; no handicap coverage, arena games, or Elo estimate. Models scored on the same 50/50 classic/double objective; all fresh games are excluded from training.",
    }
    atomic_json(output_dir / "result.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--config", dest="config_path", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument(
        "--calibration-result",
        dest="calibration_results",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", dest="device_name", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--budget-seconds", type=float, default=1800)
    print(json.dumps(run(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
