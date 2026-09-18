#!/usr/bin/env python3
"""Compare trusted losses and cached native returns against a pinned source ref.

CPU component measurements only: no server access, GPU work, optimizer updates,
or Elo claim. Arms use identical inputs and alternate order on every repeat.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import nn

from startrain import inference, losses
from startrain.model import StarModelOutput

ROOT = Path(__file__).resolve().parents[2]


def baseline_module(ref: str, name: str) -> tuple[ModuleType, str]:
    source = subprocess.check_output(
        ["git", "show", f"{ref}:training/startrain/{name}.py"], cwd=ROOT
    )
    module_name = f"startrain._benchmark_baseline_{name}"
    module = ModuleType(module_name)
    sys.modules[module_name] = module
    exec(compile(source, f"{ref}:{name}.py", "exec"), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def paired_times(functions, *, repeats: int, iterations: int) -> dict:
    timings: dict[str, list[float]] = {name: [] for name in functions}
    for fn in functions.values():
        for _ in range(5):
            fn()
    for repeat in range(repeats):
        names = list(functions)
        if repeat % 2:
            names.reverse()
        for name in names:
            start = time.perf_counter()
            for _ in range(iterations):
                functions[name]()
            timings[name].append((time.perf_counter() - start) / iterations)
    medians = {name: statistics.median(values) for name, values in timings.items()}
    return {
        "seconds_per_call": timings,
        "median_seconds": medians,
        "median_latency_ratio": medians["baseline"] / medians["candidate"],
    }


def loss_fixture(batch: int, nodes: int):
    generator = torch.Generator().manual_seed(1729)
    shapes = (
        (batch, nodes),
        (batch, 2),
        (batch, 303),
        (batch, nodes, 3),
        (batch, nodes),
        (batch, nodes),
    )
    output = StarModelOutput(
        *(torch.randn(shape, generator=generator).requires_grad_() for shape in shapes)
    )
    mask = torch.arange(batch) % 3 != 0
    target = losses.TrainingTargets(
        policy=torch.full((batch, nodes), 1 / nodes),
        outcome=torch.arange(batch) % 2,
        score_margin=torch.arange(batch) % 303 - 151,
        ownership=torch.arange(batch * nodes).reshape(batch, nodes) % 3,
        alive=(torch.arange(batch * nodes).reshape(batch, nodes) % 2).float(),
        soft_policy=torch.full((batch, nodes), 1 / nodes),
        policy_mask=torch.ones_like(mask),
        outcome_mask=mask,
        score_margin_mask=mask,
        ownership_mask=mask,
        alive_mask=mask,
        soft_policy_mask=torch.ones_like(mask),
    )
    options: dict[str, Any] = dict(
        legal_action_mask=torch.ones(batch, nodes, dtype=torch.bool),
        node_mask=torch.ones(batch, nodes, dtype=torch.bool),
        validate_targets=False,
    )
    return output, target, options


def benchmark_losses(baseline, *, repeats, iterations) -> dict:
    output, target, options = loss_fixture(512, 275)
    functions = {
        "baseline": lambda: baseline.compute_losses(output, target, **options),
        "candidate": lambda: losses.compute_losses(output, target, **options),
    }
    expected, actual = functions["baseline"](), functions["candidate"]()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    tensors = tuple(value for value in output if value is not None)
    left = torch.autograd.grad(expected["total"], tensors)
    right = torch.autograd.grad(actual["total"], tensors)
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    counts = {}
    for name, fn in functions.items():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as trace:
            fn()
        counts[name] = sum(
            event.count for event in trace.key_averages() if event.key == "aten::index"
        )
    return {
        "batch": 512,
        "nodes": 275,
        "scope": "trusted loss forward only",
        "losses_and_gradients_exactly_equal": True,
        "boolean_index_calls": counts,
        **paired_times(functions, repeats=repeats, iterations=iterations),
    }


class FixtureNetwork(nn.Module):
    """Only populates caches before timing; never runs during timed returns."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.calls = 0

    def forward(self, node_features, global_features, *args, **kwargs):
        self.calls += 1
        batch, nodes = node_features.shape[:2]
        policy = node_features.sum(dim=-1) + self.weight
        value = global_features.sum(dim=-1)
        return StarModelOutput(
            policy,
            torch.stack((-value, value), dim=-1),
            torch.arange(303, dtype=torch.float32).expand(batch, -1) / 303,
            torch.zeros(batch, nodes, 3),
            torch.zeros(batch, nodes),
            policy,
        )


def benchmark_returns(baseline, *, rows, repeats, iterations) -> dict:
    star_native = importlib.import_module("star_native")

    states = star_native.StateBatch(10, rows)
    for action in range(15):
        indices = [row for row in range(rows) if row % 16 > action]
        if indices:
            states.apply_many(indices, [action] * len(indices))
    search = star_native.SearchBatch(
        states,
        simulations=1,
        pda_by_seat=[(row % 7 - 3, 3 - row % 7) for row in range(rows)],
    )
    request = search.root_requests()
    adapters, functions = [], {}
    for name, module in (("baseline", baseline), ("candidate", inference)):
        adapter = module.GraphInferenceAdapter(
            FixtureNetwork(),
            model_identity="cpu-return-fixture",
            config=module.InferenceConfig(
                cache_max_entries=rows,
                cache_max_bytes=16_000_000,
                deduplicate=True,
                score_utility_weight=0.05,
            ),
        )
        adapter.evaluate(request)
        prepared = adapter.prepare_requests(request)
        functions[name] = lambda adapter=adapter, prepared=prepared: (
            adapter.evaluate_prepared((prepared,))
        )
        adapters.append(adapter)
    expected, actual = functions["baseline"]()[0][0], functions["candidate"]()[0][0]
    for name in ("tokens", "values", "policy_offsets", "policy_logits"):
        assert getattr(expected, name) == getattr(actual, name), name
    calls = [adapter.model.calls for adapter in adapters]
    result = paired_times(functions, repeats=repeats, iterations=iterations)
    assert calls == [adapter.model.calls for adapter in adapters], (
        "timing included neural work"
    )
    for adapter in adapters:
        adapter.close()
    return {
        "rows": rows,
        "scope": "warm cached native evaluate_prepared",
        "responses_exactly_equal": True,
        "timed_neural_calls": 0,
        **result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 2 or args.iterations < 1:
        parser.error("repeats must be at least 2 and iterations must be positive")
    if args.output.exists():
        parser.error("output must be a new file")
    ref = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.baseline_ref}^{{commit}}"],
        cwd=ROOT,
        text=True,
    ).strip()
    baseline_losses, loss_hash = baseline_module(ref, "losses")
    baseline_inference, inference_hash = baseline_module(ref, "inference")
    torch.set_num_threads(1)
    options = dict(repeats=args.repeats, iterations=args.iterations)
    report = {
        "schema_version": 1,
        "benchmark": "cpu-hotpaths",
        "baseline_commit": ref,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cpu_threads": 1,
        "repeats": args.repeats,
        "iterations": args.iterations,
        "baseline_sha256": {"losses.py": loss_hash, "inference.py": inference_hash},
        "candidate_sha256": {
            name: hashlib.sha256(
                (ROOT / "training/startrain" / name).read_bytes()
            ).hexdigest()
            for name in ("losses.py", "inference.py")
        },
        "losses": benchmark_losses(baseline_losses, **options),
        "returns": [
            benchmark_returns(baseline_inference, rows=rows, **options)
            for rows in (1, 8, 32, 128)
        ],
        "limitations": "CPU synthetic component timings; no H100, end-to-end training, or Elo/hour measurement.",
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
