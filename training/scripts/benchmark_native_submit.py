#!/usr/bin/env python3
"""Alternate two exact native binaries on deterministic CPU search workloads.

Build each revision with ``maturin build --release --locked --manifest-path
crates/deltrel-py/Cargo.toml`` and extract each wheel's deltrel_native.abi3.so to a
separate directory. Pass those
paths as --baseline/--candidate; no installed extension is replaced. This uses
a cheap synthetic evaluator, not a trained network or an H100 throughput test.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_native(path: Path, threads: int) -> Any:
    spec = importlib.util.spec_from_file_location("deltrel_native", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load native extension {path}")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    native.configure_rayon_threads(threads)
    return native


def prediction(requests: Any) -> tuple[list, list, list, list]:
    values, logits = [], []
    actions = requests.legal_actions
    offsets = requests.legal_offsets
    for row, code in enumerate(requests.states.hashes):
        values.append((code % 101 - 50) / 100)
        logits.extend(
            ((action * 13 + code) % 17) / 10 - 0.8
            for action in actions[offsets[row] : offsets[row + 1]]
        )
    return (list(requests.tokens), values, offsets, logits)


def trial(
    native: Any, ring: int, roots: int, budget: int, width: int, candidates: int
) -> dict:
    states = native.StateBatch(ring, roots, mode="double", pie=True)
    # Different openings exercise independent session trees and legal-action rows.
    states.apply_many(
        list(range(roots)), [row % states.node_count for row in range(roots)]
    )
    search = native.SearchBatch(
        states,
        simulations=budget,
        max_considered=min(candidates, budget),
        first_visit_batch_size=width,
        seeds_per_root=[17 + row for row in range(roots)],
    )
    root_requests = search.root_requests()
    search.initialize_roots(*prediction(root_requests))
    submit_ns = next_ns = neural_rows = calls = 0
    trace = hashlib.sha256()
    while not search.is_done():
        start = time.perf_counter_ns()
        requests = search.next_requests(max_rows=max(256, roots))
        next_ns += time.perf_counter_ns() - start
        if not len(requests):
            continue
        # Opaque tokens are allocated concurrently; semantic request order matters.
        trace.update(
            digest(
                [
                    requests.tree_indices,
                    requests.states.hashes,
                    requests.legal_offsets,
                    requests.legal_actions,
                ]
            ).encode()
        )
        args = prediction(requests)
        start = time.perf_counter_ns()
        search.submit(*args)
        submit_ns += time.perf_counter_ns() - start
        neural_rows += len(requests)
        calls += 1
    result = search.results()
    trace.update(
        digest(
            {
                field: getattr(result, field)
                for field in (
                    "selected_actions",
                    "terminal",
                    "terminal_values",
                    "root_values",
                    "selected_action_values",
                    "action_offsets",
                    "actions",
                    "visits",
                    "q_values",
                    "priors",
                    "policy_target",
                )
            }
        ).encode()
    )
    return {
        "submit_ns": submit_ns,
        "next_ns": next_ns,
        "native_ns": submit_ns + next_ns,
        "calls": calls,
        "neural_rows": neural_rows,
        "trace_sha256": trace.hexdigest(),
    }


def worker(args: argparse.Namespace) -> dict:
    native = load_native(args.worker, args.threads)
    cases = {}
    for ring in args.rings:
        for roots in args.roots:
            for budget in args.budgets:
                key = f"r{ring}-roots{roots}-budget{budget}-width{args.width}"
                trial(native, ring, roots, budget, args.width, args.candidates)
                samples = [
                    trial(native, ring, roots, budget, args.width, args.candidates)
                    for _ in range(args.repeats)
                ]
                if len({row["trace_sha256"] for row in samples}) != 1:
                    raise RuntimeError(f"nondeterministic semantic trace: {key}")
                cases[key] = samples
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rings", nargs="+", type=int, default=[4, 10])
    parser.add_argument("--roots", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--budgets", nargs="+", type=int, default=[27, 64])
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=4)
    args = parser.parse_args()
    if (
        min(
            args.threads,
            args.width,
            args.candidates,
            args.repeats,
            args.pairs,
            *args.roots,
            *args.budgets,
        )
        < 1
    ):
        parser.error("dimensions must be positive")
    if args.worker:
        print(json.dumps(worker(args)))
        return
    if not args.baseline or not args.candidate or not args.output:
        parser.error("--baseline, --candidate and --output are required")
    runs = []
    for pair in range(args.pairs):
        for arm in (
            ["baseline", "candidate"] if pair % 2 == 0 else ["candidate", "baseline"]
        ):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(getattr(args, arm).resolve()),
                "--threads",
                str(args.threads),
                "--width",
                str(args.width),
                "--candidates",
                str(args.candidates),
                "--repeats",
                str(args.repeats),
            ]
            for name in ("rings", "roots", "budgets"):
                command += [f"--{name}", *map(str, getattr(args, name))]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            runs.append({"pair": pair, "arm": arm, "cases": json.loads(result.stdout)})
            args.output.with_suffix(".partial.json").write_text(
                json.dumps({"complete": False, "runs": runs}, indent=2) + "\n"
            )
    summary = {}
    for case in runs[0]["cases"]:
        arms = {
            arm: [
                sample
                for run in runs
                if run["arm"] == arm
                for sample in run["cases"][case]
            ]
            for arm in ("baseline", "candidate")
        }
        if (
            len(
                {
                    sample["trace_sha256"]
                    for samples in arms.values()
                    for sample in samples
                }
            )
            != 1
        ):
            raise RuntimeError(f"baseline/candidate semantic trace mismatch: {case}")
        summary[case] = {
            name + "_ratio": statistics.median(row[name] for row in arms["baseline"])
            / statistics.median(row[name] for row in arms["candidate"])
            for name in ("submit_ns", "native_ns")
        }
        summary[case]["trace_sha256"] = arms["baseline"][0]["trace_sha256"]
    report = {
        "scope": "CPU native search with synthetic evaluation; not H100 or Elo throughput",
        "platform": platform.platform(),
        "python": sys.version,
        "settings": {
            name: getattr(args, name)
            for name in (
                "threads",
                "rings",
                "roots",
                "budgets",
                "width",
                "candidates",
                "repeats",
                "pairs",
            )
        },
        "binaries": {
            arm: {
                "path": str(getattr(args, arm).resolve()),
                "sha256": hashlib.sha256(getattr(args, arm).read_bytes()).hexdigest(),
            }
            for arm in ("baseline", "candidate")
        },
        "summary": summary,
        "runs": runs,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
