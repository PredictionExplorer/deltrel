#!/usr/bin/env python3
"""Freeze balanced real replay positions, then compare batched self-play search.

This measures search cost and disagreement, never Elo. The first raw fast/full
pair must be the profile baseline. Every arm uses exact retained history, PDA,
per-segment score utility and the same per-position random seed. Read-only replay
selection copies validated samples into a new owned directory before GPU work.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

from startrain.actor import resolve_actor_experiment
from startrain.checkpoint import load_model_manifest
from startrain.config import load_config
from startrain.contracts import FEATURE_SCHEMA_HASH, RULES_HASH, SEARCH_ALGORITHM_ID
from startrain.native import BITBOARD_WORDS, load_star_native
from startrain.promotion import load_manifest_evaluator
from startrain.replay import ReplaySample, read_replay_shard, write_replay_shard
from startrain.selfplay import SelfPlayConfig
from startrain.topology import SUPPORTED_RINGS

MODES = (
    "standard-double",
    "standard-classic",
    "handicap-double",
    "handicap-classic",
    "pie-double",
    "pie-classic",
)
PHASES = ("early", "middle", "late")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def publish_new(path: Path, data: bytes) -> None:
    """Atomic no-overwrite publication, including dangling destination symlinks."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def cell(sample: ReplaySample) -> tuple[int, str, str]:
    segment = "pie" if sample.pie else "handicap" if sample.handicap > 1 else "standard"
    phase = PHASES[
        min(2, 3 * int(np.count_nonzero(sample.stones != -1)) // len(sample.stones))
    ]
    return sample.rings, f"{segment}-{sample.mode}", phase


def sample_id(sample: ReplaySample) -> str:
    return f"{sample.run_id}/{sample.game_id}/{sample.ply}"


def samples_from_bytes(contents: bytes) -> list[ReplaySample]:
    # The production decoder intentionally accepts filesystem paths only. Read
    # from an owned immutable byte snapshot, never re-open a live GC-able path.
    with tempfile.TemporaryDirectory(prefix="star-frozen-replay-") as temporary:
        path = Path(temporary) / "snapshot.npz"
        path.write_bytes(contents)
        return read_replay_shard(path)


def freeze_positions(args: Any, config: Any, manifest: Any) -> dict[str, Any]:
    root = args.replay_root.resolve(strict=True)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    wanted = {
        (ring, mode, phase) for ring in args.rings for mode in MODES for phase in PHASES
    }
    selected: dict[tuple[int, str, str], list[ReplaySample]] = {
        key: [] for key in wanted
    }
    games: dict[tuple[int, str, str], set[str]] = {key: set() for key in wanted}
    swap_per_mode = getattr(args, "swap_per_mode", 0)
    swaps: dict[tuple[int, str], list[ReplaySample]] = {
        (ring, mode): [] for ring in args.rings for mode in ("classic", "double")
    }
    swap_games: dict[tuple[int, str], set[str]] = {key: set() for key in swaps}
    selected_ids: set[str] = set()
    excluded_games: set[str] = set()
    exclusions = []
    for directory in getattr(args, "exclude_positions", []):
        contents = (directory / "selection.json").read_bytes()
        excluded = json.loads(contents)
        if (
            excluded.get("schema_version") != 1
            or excluded.get("run_id") != manifest.run_id
            or excluded.get("generation_family") != manifest.generation_family
        ):
            raise ValueError("excluded holdout namespace does not match source")
        excluded_games.update(
            row["id"].rsplit("/", 1)[0] for row in excluded["positions"]
        )
        exclusions.append(
            {
                "selection_sha256": digest(contents),
                "positions": len(excluded["positions"]),
            }
        )
    evidence = []
    lower = max(0, manifest.model_step - config.learner.max_replay_lag_steps)
    cutoff = config.learner.minimum_replay_shard_id_exclusive or 0
    # Never instantiate ReplayStore here: initialization/reconciliation writes.
    connection = sqlite3.connect(
        (root / "manifest.sqlite3").as_uri() + "?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        rows = []
        for ring in args.rings:
            rows.extend(
                connection.execute(
                    "SELECT * FROM shards WHERE state='ready' AND ring=? AND run_id=? "
                    "AND generation_family=? AND model_step BETWEEN ? AND ? AND id>? "
                    "AND rules_hash=? AND feature_schema_hash=? ORDER BY id DESC LIMIT ?",
                    (
                        ring,
                        manifest.run_id,
                        manifest.generation_family,
                        lower,
                        manifest.model_step,
                        cutoff,
                        f"{RULES_HASH:016x}",
                        f"{FEATURE_SCHEMA_HASH:016x}",
                        args.max_shards,
                    ),
                ).fetchall()
            )
        # Only metadata is snapshotted. Payloads may race ordinary GC: missing
        # files are skipped, but checksums/context mismatch fail closed.
        connection.rollback()
    finally:
        connection.close()
    random.Random(args.seed).shuffle(rows)
    missing_files = 0
    for row in rows:
        path = (root / row["relative_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("replay payload path escaped its store")
        try:
            contents = path.read_bytes()
        except FileNotFoundError:
            missing_files += 1
            continue
        if digest(contents) != row["checksum_sha256"]:
            raise ValueError(f"replay payload checksum mismatch: shard {row['id']}")
        samples = samples_from_bytes(contents)
        if len(samples) != row["sample_count"]:
            raise ValueError("replay payload and manifest sample counts disagree")
        order = list(range(len(samples)))
        random.Random(args.seed ^ int(row["id"])).shuffle(order)
        chosen = []
        for index in order:
            sample = samples[index]
            if (
                sample.run_id,
                sample.generation_family,
                sample.rings,
                sample.model_identity,
            ) != (
                row["run_id"],
                row["generation_family"],
                row["ring"],
                row["model_identity"],
            ):
                raise ValueError("replay payload and manifest identity disagree")
            key = cell(sample)
            identifier = sample_id(sample)
            if (
                sample.terminal
                or not sample.history_known
                or key not in wanted
                or identifier in selected_ids
                or identifier.rsplit("/", 1)[0] in excluded_games
            ):
                continue
            swap_key = (sample.rings, sample.mode)
            if (
                sample.swap_available
                and len(swaps[swap_key]) < swap_per_mode
                and sample.game_id not in swap_games[swap_key]
            ):
                swaps[swap_key].append(sample)
                swap_games[swap_key].add(sample.game_id)
                kind = "swap-probe"
            elif (
                len(selected[key]) < args.per_cell and sample.game_id not in games[key]
            ):
                selected[key].append(sample)
                games[key].add(sample.game_id)
                kind = "balanced"
            else:
                continue
            selected_ids.add(identifier)
            chosen.append(
                {
                    "sample_index": index,
                    "sample_id": identifier,
                    "cell": key,
                    "kind": kind,
                }
            )
        if chosen:
            evidence.append(
                {
                    "shard_id": row["id"],
                    "sha256": row["checksum_sha256"],
                    "model_step": row["model_step"],
                    "selections": chosen,
                }
            )
        if all(len(values) == args.per_cell for values in selected.values()) and all(
            len(values) == swap_per_mode for values in swaps.values()
        ):
            break
    shortages = {
        "/".join(map(str, key)): args.per_cell - len(values)
        for key, values in selected.items()
        if len(values) < args.per_cell
    }
    shortages.update(
        {
            f"{ring}/pie-{mode}/swap": swap_per_mode - len(rows)
            for (ring, mode), rows in swaps.items()
            if len(rows) < swap_per_mode
        }
    )
    if shortages:
        raise ValueError(
            f"insufficient distinct games in balanced cells: {shortages}; increase --max-shards"
        )
    samples = [sample for key in sorted(selected) for sample in selected[key]]
    swap_samples = [sample for key in sorted(swaps) for sample in swaps[key]]
    samples.extend(swap_samples)
    if not 1 <= len(samples) <= 512:
        raise ValueError("frozen selection must contain 1..512 positions")
    args.output.mkdir()  # New owned artifact directory; never a replay mutation.
    # The replay decoder requires each file to have one exact variant. Keep
    # that contract while the inference driver later coalesces variant groups.
    variants: dict[tuple[int, str, int, bool], list[ReplaySample]] = defaultdict(list)
    for sample in samples:
        variants[sample.rings, sample.mode, sample.handicap, sample.pie].append(sample)
    payloads = []
    for index, (_, rows) in enumerate(sorted(variants.items())):
        payload = args.output / f"positions-{index:03d}.npz"
        write_replay_shard(payload, rows)
        with payload.open("rb") as stream:
            os.fsync(stream.fileno())
        payloads.append(
            {
                "name": payload.name,
                "sha256": digest(payload.read_bytes()),
                "samples": len(rows),
            }
        )
    report = {
        "schema_version": 1,
        "payloads": payloads,
        "payload_sha256": digest(json_bytes(payloads)),
        "config_sha256": digest(args.config.read_bytes()),
        "run_id": manifest.run_id,
        "generation_family": manifest.generation_family,
        "manifest_sha256": manifest.manifest_sha256,
        "model_identity": manifest.model_identity,
        "model_step": manifest.model_step,
        "seed": args.seed,
        "per_cell": args.per_cell,
        "swap_per_mode": swap_per_mode,
        "swap_position_ids": [sample_id(sample) for sample in swap_samples],
        "excluded_games": len(excluded_games),
        "excluded_selections": exclusions,
        "rings": args.rings,
        "phase_definition": "occupied fraction: [0,1/3), [1/3,2/3), [2/3,1)",
        "selection": "seeded shuffle of bounded recent ready shards; distinct game per cell; not IID whole replay",
        "minimum_model_step": lower,
        "minimum_shard_id_exclusive": cutoff,
        "missing_payloads_skipped": missing_files,
        "sources": evidence,
        "positions": [
            {
                "id": sample_id(s),
                "cell": cell(s),
                "pda": s.pda,
                "swap_available": s.swap_available,
            }
            for s in samples
        ],
    }
    publish_new(args.output / "selection.json", json_bytes(report))
    return report


def read_frozen(
    path: Path, config_sha: str, manifest: Any
) -> tuple[list[ReplaySample], dict[str, Any]]:
    metadata = json.loads((path / "selection.json").read_bytes())
    if (
        metadata["schema_version"] != 1
        or metadata["payload_sha256"] != digest(json_bytes(metadata["payloads"]))
        or metadata["config_sha256"] != config_sha
        or metadata["manifest_sha256"] != manifest.manifest_sha256
        or metadata["model_identity"] != manifest.model_identity
    ):
        raise ValueError("frozen positions do not match pinned profile/model/payload")
    samples = []
    names = set()
    for payload in metadata["payloads"]:
        name = payload["name"]
        if not isinstance(name, str) or Path(name).name != name or name in names:
            raise ValueError("invalid frozen payload path")
        names.add(name)
        contents = (path / name).read_bytes()
        if digest(contents) != payload["sha256"]:
            raise ValueError("frozen payload differs from pinned checksum")
        rows = samples_from_bytes(contents)
        if len(rows) != payload["samples"]:
            raise ValueError("frozen payload sample count disagrees")
        samples.extend(rows)
    by_id = {sample_id(sample): sample for sample in samples}
    if len(by_id) != len(samples) or set(by_id) != {
        row["id"] for row in metadata["positions"]
    }:
        raise ValueError("frozen positions contain duplicate or missing identities")
    samples = [by_id[row["id"]] for row in metadata["positions"]]
    if not 1 <= len(samples) <= 512:
        raise ValueError("benchmark requires 1..512 frozen positions")
    descriptors = [
        {
            "id": sample_id(s),
            "cell": list(cell(s)),
            "pda": s.pda,
            "swap_available": s.swap_available,
        }
        for s in samples
    ]
    if descriptors != metadata["positions"] or len(
        {s["id"] for s in descriptors}
    ) != len(samples):
        raise ValueError("frozen sample descriptors do not match payload")
    expected = {
        (ring, mode, phase): metadata["per_cell"]
        for ring in metadata["rings"]
        for mode in MODES
        for phase in PHASES
    }
    swap_ids = metadata.get("swap_position_ids", [])
    if (
        not isinstance(swap_ids, list)
        or len(set(swap_ids)) != len(swap_ids)
        or not set(swap_ids) <= set(by_id)
    ):
        raise ValueError("invalid frozen swap probe identities")
    swap_count = metadata.get("swap_per_mode", 0)
    if type(swap_count) is not int or not 0 <= swap_count <= 32:
        raise ValueError("invalid frozen swap probe quota")
    expected_swaps = {
        (ring, mode): swap_count
        for ring in metadata["rings"]
        for mode in ("classic", "double")
        if swap_count
    }
    swap_samples = [by_id[identifier] for identifier in swap_ids]
    if (
        any(not sample.swap_available or not sample.pie for sample in swap_samples)
        or Counter((sample.rings, sample.mode) for sample in swap_samples)
        != expected_swaps
    ):
        raise ValueError("frozen swap probes do not match requested quota")
    if (
        Counter(cell(sample) for sample in samples if sample_id(sample) not in swap_ids)
        != expected
    ):
        raise ValueError(
            "frozen positions are not balanced across all six modes and phases"
        )
    if any(
        s.terminal
        or not s.history_known
        or s.run_id != manifest.run_id
        or s.generation_family != manifest.generation_family
        for s in samples
    ):
        raise ValueError("frozen positions contain incompatible state or namespace")
    return samples, metadata


def semantic_states(native: Any, samples: list[ReplaySample]) -> Any:
    if not samples or len({sample.rings for sample in samples}) != 1:
        raise ValueError("native batches require nonempty homogeneous ring rows")

    def bits(kind: str | int) -> list[int]:
        output = []
        for sample in samples:
            if kind == "zero":
                indices = np.flatnonzero(sample.stones == 0)
            elif kind == "one":
                indices = np.flatnonzero(sample.stones == 1)
            else:
                assert sample.history_flags is not None and isinstance(kind, int)
                indices = np.flatnonzero(sample.history_flags & kind)
            words = [0] * BITBOARD_WORDS
            for index in indices:
                words[int(index) // 64] |= 1 << (int(index) % 64)
            output.extend(words)
        return output

    return native.StateBatch.from_semantic(
        samples[0].rings,
        bits("zero"),
        bits("one"),
        [s.to_move for s in samples],
        [s.moves_left for s in samples],
        [s.opening for s in samples],
        mode=[int(s.mode == "double") for s in samples],
        handicap=[s.handicap for s in samples],
        pie=[s.pie for s in samples],
        swap_available=[s.swap_available for s in samples],
        swapped=[s.swapped for s in samples],
        current_turn_bits=bits(1),
        own_previous_turn_bits=bits(2),
        previous_turn_bits=bits(4),
        handicap_bits=bits(8),
    )


def budgets(
    base: SelfPlayConfig, sample: ReplaySample, pair: tuple[int, int], full: bool
) -> tuple[int, int, float]:
    local = replace(
        base,
        rings=sample.rings,
        mode=sample.mode,
        handicap=sample.handicap,
        pie=sample.pie,
        fast_simulations=pair[0],
        full_simulations=pair[1],
    )
    amount = local.simulation_budget(full=full)
    high, low = local.playout_budgets(simulations=amount, pda=abs(sample.pda))
    if high != low * 2 ** abs(sample.pda):
        raise ValueError(
            f"raw pair {pair} clips the declared PDA {sample.pda} at ring {sample.rings}"
        )
    return (
        (low if sample.pda < 0 else high),
        local.considered_actions(),
        local.effective_score_utility_weight(),
    )


def seed_for(sample: ReplaySample, seed: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{sample_id(sample)}".encode()).digest()[:8], "little"
    )


def run_arm(
    native: Any,
    evaluator: Any,
    samples: list[ReplaySample],
    base: SelfPlayConfig,
    pair: tuple[int, int],
    *,
    full: bool,
    width: int,
    max_rows: int,
    seed: int,
    deadline: float,
    reset: str = "all",
) -> dict[str, Any]:
    if len({sample.rings for sample in samples}) != 1 or not samples:
        raise ValueError("each measured arm requires one common ring")
    grouped: dict[tuple[int, float], list[int]] = defaultdict(list)
    allocations = [budgets(base, s, pair, full) for s in samples]
    for index, (_, considered, weight) in enumerate(allocations):
        grouped[considered, weight].append(index)
    groups = []
    for (considered, weight), indices in grouped.items():
        for start in range(0, len(indices), max_rows):
            subset = indices[start : start + max_rows]
            states = semantic_states(native, [samples[index] for index in subset])
            seats = [
                (s.pda, -s.pda) if s.to_move == 0 else (-s.pda, s.pda)
                for s in (samples[index] for index in subset)
            ]
            search = native.SearchBatch(
                states,
                simulations=max(allocations[index][0] for index in subset),
                simulations_per_root=[allocations[index][0] for index in subset],
                max_considered=considered,
                c_visit=base.c_visit,
                c_scale=base.c_scale,
                seeds_per_root=[seed_for(samples[index], seed) for index in subset],
                pda_by_seat=seats,
                first_visit_batch_size=width,
            )
            groups.append((search, weight, subset))
    if reset == "all":
        evaluator.clear_inference_cache()
    elif reset == "predictions":
        evaluator.clear_prediction_cache()
    else:
        raise ValueError("unsupported benchmark cache reset")
    before = evaluator.metrics_snapshot()
    started = time.monotonic()
    calls = rows = 0

    def evaluate(pending: list[tuple[Any, Any]], initialize: bool) -> None:
        nonlocal calls, rows
        if time.monotonic() >= deadline:
            raise TimeoutError("batched search exceeded its deadline")
        prepared = [
            evaluator.prepare_requests(request, score_utility_weight=group[1])
            for group, request in pending
        ]
        responses = evaluator.evaluate_prepared(prepared)
        calls += 1
        rows += sum(len(request) for _, request in pending)
        if len(responses) != len(pending):
            raise RuntimeError("inference returned wrong group count")
        for (group, _), (response, _) in zip(pending, responses, strict=True):
            if not all(
                math.isfinite(value)
                for value in [*response.values, *response.policy_logits]
            ):
                raise ValueError("nonfinite neural prediction")
            method = group[0].initialize_roots if initialize else group[0].submit
            method(*response.submit_args())

    # Root rows may exceed the global call cap. Never silently overfill it.
    pending = []
    count = 0
    for group in groups:
        request = group[0].root_requests()
        if count + len(request) > max_rows:
            evaluate(pending, True)
            pending, count = [], 0
        pending.append((group, request))
        count += len(request)
    if pending:
        evaluate(pending, True)
    rotation = 0
    while not all(group[0].is_done() for group in groups):
        if time.monotonic() >= deadline:
            raise TimeoutError("batched search exceeded its deadline")
        pending, count = [], 0
        ordered = groups[rotation:] + groups[:rotation]
        rotation = (rotation + 1) % len(groups)
        for group in ordered:
            if (
                group[0].is_done()
                or count == max_rows
                or (width == 1 and max_rows - count < len(group[2]))
            ):
                continue
            request = group[0].next_requests(max_rows=max_rows - count)
            if len(request):
                pending.append((group, request))
                count += len(request)
        if pending:
            evaluate(pending, False)
    elapsed = time.monotonic() - started
    metrics = asdict(evaluator.metrics_snapshot().delta(before))
    records: dict[int, dict[str, Any]] = {}
    for search, _, indices in groups:
        result = search.results()
        offsets, actions, visits, q, policy = (
            getattr(result, name)
            for name in (
                "action_offsets",
                "actions",
                "visits",
                "q_values",
                "policy_target",
            )
        )
        chosen, values = result.selected_actions, result.selected_action_values
        for row, index in enumerate(indices):
            lo, hi = offsets[row : row + 2]
            if sum(visits[lo:hi]) != allocations[index][0]:
                raise RuntimeError("search did not consume exactly the fresh budget")
            records[index] = {
                "id": sample_id(samples[index]),
                "cell": cell(samples[index]),
                "pda": samples[index].pda,
                "swap_available": samples[index].swap_available,
                "simulations": allocations[index][0],
                "selected_action": chosen[row],
                "selected_value": values[row],
                "swap": bool(
                    samples[index].swap_available
                    and values[row] < -base.variants.swap_dead_zone
                ),
                "actions": actions[lo:hi],
                "visits": visits[lo:hi],
                "q_values": q[lo:hi],
                "policy_target": policy[lo:hi],
            }
    return {
        "raw_pair": pair,
        "full": full,
        "first_visit_batch_size": width,
        "seconds": elapsed,
        "evaluator_batches": calls,
        "requested_rows": rows,
        "simulations": sum(a[0] for a in allocations),
        "inference": metrics,
        "positions": [records[index] for index in range(len(samples))],
    }


def compare(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    if (
        candidate["id"] != reference["id"]
        or candidate["actions"] != reference["actions"]
    ):
        raise ValueError(
            "comparison requires identical position and ordered legal support"
        )
    policy = np.asarray(candidate["policy_target"], dtype=np.float64)
    target = np.asarray(reference["policy_target"], dtype=np.float64)
    index = reference["actions"].index(candidate["selected_action"])
    visited = np.asarray(reference["visits"]) > 0
    assessed = bool(visited[index])
    q = np.asarray(reference["q_values"])
    # Completed Q for unvisited actions is imputed; never label it regret.
    return {
        "id": candidate["id"],
        "cell": candidate["cell"],
        "action_matches_reference": candidate["selected_action"]
        == reference["selected_action"],
        "swap_matches_reference": candidate["swap"] == reference["swap"],
        "swap_available": candidate.get("swap_available", False),
        "candidate_keep_value": candidate["selected_value"],
        "reference_keep_value": reference["selected_value"],
        "reference_selected_q_loss": float(reference["selected_value"] - q[index])
        if assessed
        else None,
        "policy_l1": float(np.abs(policy - target).sum()),
        "policy_kl_reference_to_candidate_epsilon_1e_12": float(
            (
                target * np.log(np.maximum(target, 1e-12) / np.maximum(policy, 1e-12))
            ).sum()
        ),
        "candidate_action_reference_visited": assessed,
        "reference_visits_at_candidate": int(reference["visits"][index]),
        "reference_policy_mass_on_visited": float(target[visited].sum()),
        "reference_visited_q_gap": float(q[visited].max() - q[index])
        if assessed
        else None,
        "selected_value_abs_difference": abs(
            candidate["selected_value"] - reference["selected_value"]
        ),
    }


def summarize_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    assessed = [
        row["reference_visited_q_gap"]
        for row in rows
        if row["reference_visited_q_gap"] is not None
    ]
    return {
        "positions": len(rows),
        "action_agreement": np.mean(
            [r["action_matches_reference"] for r in rows]
        ).item(),
        "swap_agreement": np.mean([r["swap_matches_reference"] for r in rows]).item(),
        "mean_policy_l1": np.mean([r["policy_l1"] for r in rows]).item(),
        "mean_policy_kl": np.mean(
            [r["policy_kl_reference_to_candidate_epsilon_1e_12"] for r in rows]
        ).item(),
        "reference_q_assessed_fraction": len(assessed) / len(rows),
        "mean_reference_visited_q_gap": float(np.mean(assessed)) if assessed else None,
        "p95_reference_visited_q_gap": float(np.quantile(assessed, 0.95))
        if assessed
        else None,
    }


TIMING_SETUP_COUNTERS = (
    "graph_captures",
    "graph_warmup_calls",
    "graph_evictions",
    "graph_fallbacks",
    "graph_validation_failures",
    "graph_validation_replays",
)


def measure_sweep(
    native: Any,
    evaluator: Any,
    samples: list[ReplaySample],
    base: SelfPlayConfig,
    pairs: list[tuple[int, int]],
    widths: list[int],
    waves: list[str],
    *,
    max_rows: int,
    seed: int,
    deadline: float,
    repeats: int,
) -> list[dict[str, Any]]:
    """Warm each distinct search shape/trace, then time prediction-cold repeats.

    The same full-search allocation can serve multiple cap-mixture proposals.
    Such aliases explicitly share one measurement, not independent samples.
    """
    results = []
    for ring in sorted({sample.rings for sample in samples}):
        subset = [sample for sample in samples if sample.rings == ring]
        common: dict[str, Any] = dict(max_rows=max_rows, seed=seed, deadline=deadline)
        reference = run_arm(
            native, evaluator, subset, base, pairs[0], full=True, width=1, **common
        )
        reference_by_id = {row["id"]: row for row in reference["positions"]}
        results.append({"ring": ring, "stage": "cold-reference", **reference})
        aliases: dict[tuple[Any, ...], list[tuple[tuple[int, int], bool, int]]] = (
            defaultdict(list)
        )
        for pair in pairs:
            for wave in waves:
                for width in widths:
                    full = wave == "full"
                    key = (
                        width,
                        tuple(budgets(base, sample, pair, full) for sample in subset),
                    )
                    aliases[key].append((pair, full, width))
        arms = list(aliases.values())
        random.Random(seed + ring).shuffle(arms)
        for index, names in enumerate(arms):
            pair, full, width = names[0]
            canonical = f"ring-{ring}-arm-{index}"
            warm = run_arm(
                native, evaluator, subset, base, pair, full=full, width=width, **common
            )
            results.append(
                {"ring": ring, "stage": "warmup", "canonical_arm": canonical, **warm}
            )
            for repeat in range(repeats):
                record = run_arm(
                    native,
                    evaluator,
                    subset,
                    base,
                    pair,
                    full=full,
                    width=width,
                    reset="predictions",
                    **common,
                )
                if record["positions"] != warm["positions"]:
                    raise RuntimeError("warmed repeat changed the frozen search trace")
                setup = {
                    key: int(record["inference"].get(key, 0))
                    for key in TIMING_SETUP_COUNTERS
                }
                record["timing_admissible"] = not any(setup.values())
                record["timing_setup_counters"] = setup
                comparisons = [
                    compare(row, reference_by_id[row["id"]])
                    for row in record["positions"]
                ]
                record["comparisons"] = comparisons
                record["quality"] = summarize_comparisons(comparisons)
                by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in comparisons:
                    by_cell["/".join(map(str, row["cell"]))].append(row)
                record["quality_by_cell"] = {
                    key: summarize_comparisons(rows) for key, rows in by_cell.items()
                }
                for alias_pair, alias_full, _ in names:
                    results.append(
                        {
                            **record,
                            "raw_pair": alias_pair,
                            "full": alias_full,
                            "ring": ring,
                            "stage": "measured",
                            "repeat": repeat,
                            "canonical_arm": canonical,
                            "shared_measurement": len(names) > 1,
                        }
                    )
    return results


def weighted_costs(
    results: list[dict[str, Any]],
    base: SelfPlayConfig,
    pairs: list[tuple[int, int]],
    widths: list[int],
    candidate_full_probability: float | None,
    candidate_fast_policy_weight: float | None,
) -> list[dict[str, Any]]:
    costs = []
    for ring in sorted({row["ring"] for row in results}):
        for width in widths:
            for pair in pairs:
                matching = {
                    full: [
                        row
                        for row in results
                        if row["stage"] == "measured"
                        and row["ring"] == ring
                        and tuple(row["raw_pair"]) == pair
                        and row["first_visit_batch_size"] == width
                        and row["full"] == full
                    ]
                    for full in (True, False)
                }
                if not all(matching.values()):
                    continue
                full_probability = (
                    base.full_probability
                    if pair == pairs[0] or candidate_full_probability is None
                    else candidate_full_probability
                )
                fast_probability = 1 - full_probability
                fast_weight = base.fast_policy_weight
                if pair != pairs[0]:
                    if candidate_fast_policy_weight is not None:
                        fast_weight = candidate_fast_policy_weight
                    elif candidate_full_probability is not None:
                        fast_weight *= (
                            base.fast_probability / base.full_probability
                        ) * (full_probability / fast_probability)
                weights = {True: full_probability, False: fast_probability}
                seconds = {
                    full: float(
                        np.median(
                            [row["seconds"] / len(row["positions"]) for row in rows]
                        )
                    )
                    for full, rows in matching.items()
                }
                requested = {
                    full: float(
                        np.median(
                            [
                                row["requested_rows"] / len(row["positions"])
                                for row in rows
                            ]
                        )
                    )
                    for full, rows in matching.items()
                }
                admissible = all(
                    row["timing_admissible"]
                    for rows in matching.values()
                    for row in rows
                )
                expected = sum(weights[full] * seconds[full] for full in weights)
                expected_rows = sum(weights[full] * requested[full] for full in weights)
                costs.append(
                    {
                        "ring": ring,
                        "raw_pair": pair,
                        "first_visit_batch_size": width,
                        "full_probability": full_probability,
                        "fast_probability": fast_probability,
                        "fast_policy_weight": fast_weight,
                        "expected_weighted_full_policy_share": (
                            full_probability
                            / (full_probability + fast_probability * fast_weight)
                        )
                        if full_probability + fast_probability * fast_weight
                        else None,
                        "timing_admissible": admissible,
                        "repeats_per_wave": {
                            str(full): len(rows) for full, rows in matching.items()
                        },
                        "expected_seconds_per_root": expected if admissible else None,
                        "expected_requested_rows_per_root": expected_rows,
                        "expected_full_targets_per_second": full_probability / expected
                        if admissible
                        else None,
                        "expected_full_targets_per_requested_row": full_probability
                        / expected_rows,
                        "wave_median_seconds_per_root": {
                            str(full): value for full, value in seconds.items()
                        },
                        "wave_seconds_per_root": {
                            str(full): [
                                row["seconds"] / len(row["positions"]) for row in rows
                            ]
                            for full, rows in matching.items()
                        },
                        "caveat": "projection of separate fast/full waves; production mixed-wave utilization and normalized-loss variance may differ",
                    }
                )
    return costs


def resolve_benchmark_config(
    config: Any, actor_gpu_id: int | None, runtime: str
) -> tuple[Any, Any]:
    """Match ActorSupervisor's per-GPU overrides before optional eager probing.

    The actor ID chooses settings, not device allocation. CUDA visibility and
    --device remain the responsibility of the isolated benchmark launcher.
    Profiles without actor GPUs retain their standalone settings by default.
    """
    actors = [gpu for gpu in config.orchestration.gpus if gpu.role == "actor"]
    if actor_gpu_id is not None:
        selected = next((gpu for gpu in actors if gpu.gpu_id == actor_gpu_id), None)
        if selected is None:
            raise ValueError("--actor-gpu-id must identify a configured actor GPU")
    else:
        selected = actors[0] if actors else None
    effective = (
        resolve_actor_experiment(config, selected) if selected is not None else config
    )
    if selected is not None:
        # SharedModelRegistry reserves capacity for concurrently pinned models,
        # including two publication transitions beyond the active cohorts.
        refresh = effective.orchestration.model_refresh
        inference = replace(
            refresh.inference,
            cuda_graph_max_bytes=max(
                1,
                refresh.inference.cuda_graph_max_bytes // (selected.actor_cohorts + 2),
            ),
        )
        effective = replace(
            effective,
            orchestration=replace(
                effective.orchestration,
                model_refresh=replace(refresh, inference=inference),
            ),
        )
    if runtime == "eager":
        inference = replace(
            effective.orchestration.model_refresh.inference, cuda_graphs=False
        )
        refresh = replace(effective.orchestration.model_refresh, inference=inference)
        effective = replace(
            effective,
            train=replace(effective.train, compile=False),
            orchestration=replace(effective.orchestration, model_refresh=refresh),
        )
    elif runtime != "profile":
        raise ValueError("unsupported benchmark runtime")
    return effective, selected


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("operation", choices=("freeze", "run"))
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--output", type=Path)
    result.add_argument("--seed", type=int, default=1701)
    result.add_argument("--replay-root", type=Path)
    result.add_argument("--rings", type=int, nargs="+", default=[10])
    result.add_argument("--per-cell", type=int, default=3)
    result.add_argument(
        "--swap-per-mode",
        type=int,
        default=0,
        help="additional swap-available positions per classic/double pie mode and ring",
    )
    result.add_argument(
        "--exclude-positions",
        type=Path,
        nargs="*",
        default=[],
        help="exclude every game appearing in these prior frozen selection directories",
    )
    result.add_argument("--max-shards", type=int, default=1024)
    result.add_argument("--positions", type=Path)
    result.add_argument("--pairs", nargs="+", default=["32:384", "16:384", "8:384"])
    result.add_argument("--widths", type=int, nargs="+", default=[1, 8])
    result.add_argument(
        "--waves", nargs="+", choices=("fast", "full"), default=["full", "fast"]
    )
    result.add_argument("--max-rows", type=int, default=256)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--candidate-full-probability", type=float)
    result.add_argument(
        "--candidate-fast-policy-weight",
        type=float,
        help="projection only; when omitted with candidate probability, preserve expected full/fast policy weight ratio",
    )
    result.add_argument("--device", default="cpu")
    result.add_argument("--runtime", choices=("eager", "profile"), default="eager")
    result.add_argument(
        "--actor-gpu-id",
        type=int,
        help="resolve this actor GPU's pipeline settings; default first configured actor; does not select CUDA device",
    )
    result.add_argument("--timeout-seconds", type=float, default=540)
    result.add_argument("--execute", action="store_true")
    result.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--pinned-plan", help=argparse.SUPPRESS)
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = parser()
    args = command.parse_args(arguments)
    if (
        type(args.seed) is not int
        or not 0 <= args.seed < 2**64
        or not 1 <= args.per_cell <= 7
        or not 0 <= args.swap_per_mode <= 32
        or len(args.rings) * (18 * args.per_cell + 2 * args.swap_per_mode) > 512
        or not args.rings
        or len(set(args.rings)) != len(args.rings)
        or any(ring not in SUPPORTED_RINGS for ring in args.rings)
        or not 1 <= args.max_shards <= 8192
        or not 1 <= args.max_rows <= 256
        or not 2 <= args.repeats <= 5
        or (
            args.candidate_full_probability is not None
            and not 0 < args.candidate_full_probability < 1
        )
        or (
            args.candidate_fast_policy_weight is not None
            and (
                not math.isfinite(args.candidate_fast_policy_weight)
                or not 0 <= args.candidate_fast_policy_weight <= 1
            )
        )
        or not math.isfinite(args.timeout_seconds)
        or not 0 < args.timeout_seconds <= 600
        or not args.widths
        or any(width not in (1, 2, 4, 8, 16, 32, 64) for width in args.widths)
        or len(set(args.widths)) != len(args.widths)
        or len(set(args.waves)) != len(args.waves)
    ):
        command.error(
            "invalid bounded rings, seed, sampling, batch, width, wave or timeout parameters"
        )
    config = load_config(args.config)
    manifest = load_model_manifest(args.checkpoint)
    if (
        args.candidate_full_probability is not None
        and not 0 < config.selfplay.full_probability < 1
    ):
        raise ValueError(
            "candidate mixture projection requires a mixed fast/full source profile"
        )
    if (
        config.orchestration.run_id is not None
        and config.orchestration.run_id != manifest.run_id
    ):
        raise ValueError("profile and immutable model run identity disagree")
    if args.operation == "freeze":
        if args.replay_root is None or args.output is None:
            command.error("freeze requires --replay-root and a new --output directory")
        report = freeze_positions(args, config, manifest)
        print(
            json.dumps(
                {"output": str(args.output), "positions": len(report["positions"])}
            )
        )
        return 0
    if args.positions is None:
        command.error(
            "run requires --positions pointing to a completed frozen directory"
        )
    try:
        pairs = [tuple(map(int, item.split(":"))) for item in args.pairs]
    except ValueError as exc:
        raise ValueError("pairs require raw-fast:raw-full integers") from exc
    if (
        not 1 <= len(pairs) <= 5
        or len(set(pairs)) != len(pairs)
        or any(len(pair) != 2 or not 1 <= pair[0] <= pair[1] <= 8192 for pair in pairs)
        or pairs[0]
        != (config.selfplay.fast_simulations, config.selfplay.full_simulations)
    ):
        raise ValueError(
            "first pair must equal profile baseline; require 1..5 distinct bounded fast:full pairs"
        )
    samples, frozen = read_frozen(
        args.positions, digest(args.config.read_bytes()), manifest
    )
    for pair in pairs:
        for sample in samples:
            for full in (False, True):
                budgets(config.selfplay, sample, (pair[0], pair[1]), full)
    runtime_config, actor_gpu = resolve_benchmark_config(
        config, args.actor_gpu_id, args.runtime
    )
    plan = {
        "schema_version": 2,
        "config_sha256": digest(args.config.read_bytes()),
        "selection_sha256": digest((args.positions / "selection.json").read_bytes()),
        "positions_sha256": frozen["payload_sha256"],
        "model_identity": manifest.model_identity,
        "manifest_sha256": manifest.manifest_sha256,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "search_algorithm": SEARCH_ALGORITHM_ID,
        "pairs": pairs,
        "widths": args.widths,
        "waves": args.waves,
        "seed": args.seed,
        "max_rows": args.max_rows,
        "repeats": args.repeats,
        "candidate_full_probability": args.candidate_full_probability,
        "candidate_fast_policy_weight": args.candidate_fast_policy_weight,
        "swap_probe_count": len(frozen.get("swap_position_ids", [])),
        "swap_probe_waves": ["fast"] if "fast" in args.waves else ["full"],
        "device": args.device,
        "runtime": args.runtime,
        "precision": config.train.precision,
        "profile_inference": asdict(config.orchestration.model_refresh.inference),
        "actor_gpu_id": actor_gpu.gpu_id if actor_gpu is not None else None,
        "graph_registry_model_capacity": actor_gpu.actor_cohorts + 2
        if actor_gpu is not None
        else None,
        "actor_pipeline": asdict(actor_gpu.actor_pipeline)
        if actor_gpu is not None and actor_gpu.actor_pipeline is not None
        else None,
        "effective_inference": asdict(
            runtime_config.orchestration.model_refresh.inference
        ),
        "effective_compile": runtime_config.train.compile,
        "effective_inference_compile_dynamic": runtime_config.orchestration.model_refresh.inference_compile_dynamic,
        "effective_inference_compile_mode": runtime_config.orchestration.model_refresh.inference_compile_mode,
        "scope": "batched frozen-position search cost and disagreement, not Elo; no subtree reuse; cold prediction cache each arm",
        "quality_reference": "profile full budget, first-visit width 1, same seed; Q gaps only at reference-visited actions",
        "repeat_scope": "repeats and shared-arm aliases are timing observations, not independent quality samples; compare one repeat per position",
        "mixture_projection_scope": "ratio of expected policy weights; not equality of expected normalized minibatch gradients or learning objectives",
        "timing": "per-arm warmup excluded; prediction-only reset; repeated timing requires zero graph captures/warmups/evictions/fallbacks/validation work",
    }
    plan_sha = digest(json_bytes(plan))
    if args.pinned_plan is not None and plan_sha != args.pinned_plan:
        raise ValueError("benchmark plan changed before worker execution")
    if not args.execute:
        print(json_bytes({"plan": plan, "plan_sha256": plan_sha}).decode(), end="")
        return 0
    if args.output is None or args.output.exists() or args.output.is_symlink():
        command.error("--execute requires a new --output JSON path")
    if not args.worker:
        child = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                *arguments,
                "--worker",
                "--pinned-plan",
                plan_sha,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = child.communicate(timeout=args.timeout_seconds)
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.communicate()
            raise
        if child.returncode:
            raise RuntimeError(f"batched search worker failed: {stderr[-6000:]}")
        print(stdout, end="")
        return 0
    native = load_star_native(required=True)
    deadline = time.monotonic() + args.timeout_seconds
    evaluator = load_manifest_evaluator(runtime_config, manifest, device=args.device)
    try:
        swap_ids = set(frozen.get("swap_position_ids", []))
        balanced_samples = [
            sample for sample in samples if sample_id(sample) not in swap_ids
        ]
        common = dict(
            max_rows=args.max_rows,
            seed=args.seed,
            deadline=deadline,
            repeats=args.repeats,
        )
        balanced_results = measure_sweep(
            native,
            evaluator,
            balanced_samples,
            config.selfplay,
            [(pair[0], pair[1]) for pair in pairs],
            args.widths,
            args.waves,
            **common,
        )
        for record in balanced_results:
            record["dataset"] = "balanced"
        costs = weighted_costs(
            balanced_results,
            config.selfplay,
            [(pair[0], pair[1]) for pair in pairs],
            args.widths,
            args.candidate_full_probability,
            args.candidate_fast_policy_weight,
        )
        results = balanced_results
        if swap_ids:
            swap_results = measure_sweep(
                native,
                evaluator,
                [sample for sample in samples if sample_id(sample) in swap_ids],
                config.selfplay,
                [(pair[0], pair[1]) for pair in pairs],
                args.widths,
                ["fast"] if "fast" in args.waves else ["full"],
                **common,
            )
            for record in swap_results:
                record["dataset"] = "swap-probes"
            results.extend(swap_results)
        passed = all(
            row["timing_admissible"] for row in results if row["stage"] == "measured"
        )
        artifact = {
            "status": "passed" if passed else "timing-rejected",
            "plan": plan,
            "plan_sha256": plan_sha,
            "results": results,
            "weighted_costs": costs,
            "limitations": [
                "No playing-strength or training-learning-curve measurement.",
                "Small balanced stratified sample; correlated positions may share games across phases.",
                "Reference Q is noisy finite-search evidence, not ground truth.",
                "Runtime includes Python/native orchestration; warmup costs are separate and invalid measured setup is rejected.",
            ],
        }
        publish_new(args.output, json_bytes(artifact))
    finally:
        evaluator.close()
    print(
        json.dumps(
            {
                "status": "passed" if passed else "timing-rejected",
                "output": str(args.output),
                "plan_sha256": plan_sha,
                "records": len(results),
                "weighted_costs": costs,
            }
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
