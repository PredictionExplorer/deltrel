#!/usr/bin/env python3
"""CPU-only comparison of immutable arena snapshot construction.

The synthetic fixture models stored history volume, not played-game evidence.
Both arms run in one process in alternating order. The retained baseline is the
previous JSON-roundtrip implementation. Timings exclude disk persistence, native
search and inference; they establish no whole-arena or Elo/hour improvement.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
from types import SimpleNamespace
from typing import Any, Callable, cast

from deltreltrain.arena import ArenaGame, ArenaPair, ArenaRunner
from deltreltrain.config import ArenaConfig
from deltreltrain.selfplay import GameVariant
from deltreltrain.topology import get_topology


def legacy_snapshot(subject: ArenaRunner) -> dict[str, Any]:
    """Frozen pre-optimization implementation, also used for differential tests."""
    entries = [subject._resume_games[key] for key in sorted(subject._resume_games)]
    games = [entry["result"] for entry in entries if entry.get("result") is not None]
    grouped: dict[tuple[int, str, int], dict[int, dict[str, Any]]] = {}
    for game in games:
        grouped.setdefault((game["ring"], game["variant"], game["pair"]), {})[
            game["candidate_player"]
        ] = game
    pairs = []
    for seats in grouped.values():
        if set(seats) != {0, 1}:
            continue
        first, second = seats[0], seats[1]
        pairs.append(
            asdict(
                ArenaPair(
                    ring=first["ring"],
                    pair=first["pair"],
                    opening_seed=first["opening_seed"],
                    opening_action=first["opening_action"],
                    forced_opening=first["forced_opening"],
                    outcomes=(first["outcome"], second["outcome"]),
                    variant=first["variant"],
                    segment=first["segment"],
                )
            )
        )
    return json.loads(
        json.dumps(
            {
                **cast(dict[str, object], subject._resume_contract),
                "game_states": entries,
                "games": games,
                "pairs": pairs,
                "progress": {
                    "completed_games": len(games),
                    "completed_pairs": len(pairs),
                    "completed_moves": sum(len(entry["actions"]) for entry in entries),
                },
            }
        )
    )


def synthetic_fixture() -> ArenaRunner:
    subject = ArenaRunner(
        native_module=object(),
        candidate=cast(Any, SimpleNamespace(model_version="candidate")),
        baseline=cast(Any, SimpleNamespace(model_version="baseline")),
        config=ArenaConfig(rings=(10,)),
        stable_pair_seeds=True,
    )
    subject._initialize_resume(None, lambda _snapshot: None)
    variant = GameVariant()
    nodes = get_topology(10).n
    for pair in range(120):
        for _, seat, seed, opening in subject._pair_specifications(10, [pair], variant):
            actions = [action for action in range(nodes) if action != opening]
            game = (
                ArenaGame(
                    ring=10,
                    pair=pair,
                    candidate_player=seat,
                    opening_seed=seed,
                    opening_action=opening,
                    forced_opening=opening is not None,
                    winner=seat,
                    outcome=1,
                    searched_moves=len(actions),
                )
                if pair < 96
                else None
            )
            subject._resume_games[(10, variant.label, pair, seat)] = {
                "ring": 10,
                "variant": variant.label,
                "pair": pair,
                "candidate_player": seat,
                "opening_seed": seed,
                "opening_action": opening,
                "actions": actions,
                "result": asdict(game) if game is not None else None,
            }
    return subject


def benchmark(*, repeats: int, iterations: int) -> dict[str, Any]:
    if type(repeats) is not int or repeats < 2 or repeats % 2:
        raise ValueError("repeats must be an even integer of at least two")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    subject = synthetic_fixture()
    functions: dict[str, Callable[[], dict[str, Any]]] = {
        "legacy": lambda: legacy_snapshot(subject),
        "optimized": subject._resume_snapshot,
    }
    expected = json.dumps(functions["legacy"]())
    measurements: dict[str, list[float]] = {name: [] for name in functions}
    orders = []
    for repeat in range(repeats):
        order = list(functions) if repeat % 2 == 0 else list(reversed(functions))
        orders.append(order)
        for name in order:
            function = functions[name]
            if json.dumps(function()) != expected:
                raise RuntimeError("snapshot differs from legacy JSON bytes")
            start = time.perf_counter()
            for _ in range(iterations):
                function()
            measurements[name].append((time.perf_counter() - start) / iterations)
            if json.dumps(function()) != expected:
                raise RuntimeError("snapshot changed during the timed arm")
    return {
        "benchmark": "arena-resume-snapshot-v1",
        "scope": "synthetic CPU snapshot construction; excludes search/inference/disk",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "arena_source_sha256": hashlib.sha256(
            Path(__file__).parents[1].joinpath("deltreltrain/arena.py").read_bytes()
        ).hexdigest(),
        "snapshot_sha256": hashlib.sha256(expected.encode()).hexdigest(),
        "snapshot_bytes": len(expected.encode()),
        "game_states": 240,
        "completed_games": 192,
        "byte_identical": True,
        "iterations_per_arm": iterations,
        "orders": orders,
        "seconds_per_snapshot": measurements,
        "median_speedup": statistics.median(measurements["legacy"])
        / statistics.median(measurements["optimized"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(
        benchmark(repeats=args.repeats, iterations=args.iterations), indent=2
    )
    if args.output is not None:
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
