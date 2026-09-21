#!/usr/bin/env python3
"""Compare exact cached and uncached readiness checks on an owned manifest.

This is a CPU/SQLite microbenchmark. It does not read live replay, load shard
payloads, measure loader throughput, or estimate training strength improvement.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from deltreltrain.contracts import FEATURE_SCHEMA_HASH, RULES_HASH
from deltreltrain.replay_store import ReplayStore


def populate_manifest(store: ReplayStore, shards: int) -> None:
    """Populate metadata only; payloads aren't accessed by readiness queries."""

    modes = (
        ("double", "standard"),
        ("classic", "classic"),
        ("handicap-2-classic", "handicap"),
        ("handicap-2-double", "handicap"),
        ("pie-classic", "pie"),
        ("pie-double", "pie"),
    )
    store.connection.execute("BEGIN IMMEDIATE")
    try:
        store.connection.executemany(
            """INSERT INTO shards (
                relative_path, created_ns, sample_count, ring, phase_min, phase_max,
                model_version, model_step, model_identity, run_id, generation_family,
                actor_id, generation, game_count, rules_hash, feature_schema_hash,
                checksum_sha256, variant, segment
            ) VALUES (?, ?, 64, ?, 0, 63, 'model', ?, 'model', 'benchmark', 'benchmark',
                      'actor', 0, 1, ?, ?, ?, ?, ?)""",
            (
                (
                    f"shards/metadata-only-{index}.npz",
                    index,
                    (6, 8, 10)[(index // 6) % 3],
                    500 + index % 601,
                    f"{RULES_HASH:016x}",
                    f"{FEATURE_SCHEMA_HASH:016x}",
                    "0" * 64,
                    *modes[index % len(modes)],
                )
                for index in range(shards)
            ),
        )
        store.connection.execute("COMMIT")
    finally:
        if store.connection.in_transaction:
            store.connection.execute("ROLLBACK")


def benchmark(*, shards: int, checks_per_step: int, repeats: int) -> dict[str, object]:
    if any(
        type(value) is not int or value <= 0
        for value in (shards, checks_per_step, repeats)
    ):
        raise ValueError(
            "shards, checks_per_step and repeats must be positive integers"
        )
    scenarios = []
    with tempfile.TemporaryDirectory(prefix="replay-eligibility-benchmark-") as root:
        with ReplayStore(Path(root) / "replay") as store:
            populate_manifest(store, shards)

            def uncached(statement, parameters):
                # The original implementation executes these same aggregate
                # queries on every check. Only result reuse differs here.
                return tuple(store.connection.execute(statement, parameters))

            for by_segment in (False, True):
                method = (
                    store.eligible_sample_counts_by_segment
                    if by_segment
                    else store.eligible_sample_counts
                )
                durations = {"uncached": [], "cached": []}
                scans = {"uncached": 0, "cached": 0}

                def read_counts(step: int):
                    return method(
                        rings=(6, 8, 10),
                        run_id="benchmark",
                        generation_family="benchmark",
                        current_model_step=step,
                        max_model_lag_steps=400,
                    )

                # Exclude topology construction and cold SQLite page reads.
                read_counts(1000)
                store._eligible_counts_cache.clear()
                for repeat in range(repeats):
                    step = 1000 + repeat
                    observed = {}
                    arms = (
                        ("uncached", "cached")
                        if repeat % 2 == 0
                        else ("cached", "uncached")
                    )
                    for arm in arms:
                        statements = []
                        store.connection.set_trace_callback(statements.append)
                        try:
                            if arm == "uncached":
                                with patch.object(
                                    store, "_eligible_count_rows", uncached
                                ):
                                    started = time.perf_counter()
                                    values = [
                                        read_counts(step)
                                        for _ in range(checks_per_step)
                                    ]
                                    elapsed = time.perf_counter() - started
                            else:
                                started = time.perf_counter()
                                values = [
                                    read_counts(step) for _ in range(checks_per_step)
                                ]
                                elapsed = time.perf_counter() - started
                        finally:
                            store.connection.set_trace_callback(None)
                        durations[arm].append(elapsed)
                        scans[arm] += sum(
                            "SUM(sample_count)" in sql for sql in statements
                        )
                        observed[arm] = values
                    if observed["uncached"] != observed["cached"]:
                        raise RuntimeError(
                            "cached and uncached eligibility counts differ"
                        )
                medians = {
                    arm: statistics.median(samples)
                    for arm, samples in durations.items()
                }
                scenarios.append(
                    {
                        "grouping": "ring_segment" if by_segment else "ring",
                        "identical_counts": True,
                        "aggregate_scans": scans,
                        "seconds_per_step_median": medians,
                        "speedup": medians["uncached"] / medians["cached"],
                    }
                )
    return {
        "benchmark": "replay-eligibility-cache",
        "schema_version": 1,
        "metadata_shards": shards,
        "checks_per_step": checks_per_step,
        "repeats": repeats,
        "scenarios": scenarios,
        "scope": "isolated CPU/SQLite readiness queries; no training or Elo estimate",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=30_000)
    parser.add_argument("--checks-per-step", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(benchmark(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
