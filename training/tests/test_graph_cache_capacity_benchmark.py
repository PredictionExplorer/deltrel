from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
import hashlib
import itertools
import json
import os
from types import SimpleNamespace

import pytest

from scripts import benchmark_graph_cache_capacity as benchmark
from startrain.graph_cache_evidence import validate_graph_cache_report
from startrain.inference import InferenceMetrics, InferenceResponse

SNAPSHOT_STAMPS = itertools.count(1)


def response(rows):
    return InferenceResponse(
        list(range(rows)), [0.25] * rows, list(range(rows + 1)), [0.5] * rows
    )


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def snapshot_factory():
    def snapshot():
        return {
            "observed_ns": next(SNAPSHOT_STAMPS),
            "gpu_uuid": "GPU-test",
            "verified": True,
            "owners": [{"pid": os.getpid(), "start_ticks": 123}],
        }

    return snapshot


class Adapter:
    def __init__(self, clock, entries, *, byte_slots=64, corrupt=False):
        self.clock, self.entries, self.byte_slots = clock, entries, byte_slots
        self.corrupt = corrupt
        self.cache = OrderedDict()
        self.metrics = InferenceMetrics()
        self.resets = 0

    def clear_prediction_cache(self):
        self.resets += 1

    def metrics_snapshot(self):
        return self.metrics

    def graph_residency_snapshot(self):
        return {"entries": list(self.cache)}

    def evaluate_prepared(self, requests):
        key = requests[0]
        rows = key[1]
        capture, eviction = 0, 0
        if key not in self.cache:
            capture = 1
            if len(self.cache) >= min(self.entries, self.byte_slots):
                self.cache.popitem(last=False)
                eviction = 1
            self.cache[key] = True
        self.cache.move_to_end(key)
        self.clock.now += 1 + capture * 10
        previous = self.metrics
        self.metrics = replace(
            previous,
            evaluator_calls=previous.evaluator_calls + 1,
            evaluator_rows=previous.evaluator_rows + rows,
            neural_calls=previous.neural_calls + 1,
            neural_rows=previous.neural_rows + rows,
            graph_replays=previous.graph_replays + 1,
            graph_captures=previous.graph_captures + capture,
            graph_evictions=previous.graph_evictions + eviction,
            graph_warmup_calls=previous.graph_warmup_calls + 4 * capture,
            graph_warmup_rows=previous.graph_warmup_rows + 4 * capture * rows,
            graph_validation_replays=previous.graph_validation_replays + capture,
        )
        result = response(rows)
        if self.corrupt:
            result.values[0] += 0.125
        return [(result, None)]


def run_arm(scenario, entries, *, byte_slots=64, corrupt=False):
    clock = Clock()
    adapter = Adapter(clock, entries, byte_slots=byte_slots, corrupt=corrupt)
    trace = benchmark.build_trace(scenario)
    prepared = {key: key for key in trace}
    reference = {
        key: benchmark.response_digest(response(key[1]), key[1]) for key in trace
    }
    measured = benchmark.measure_arm(
        adapter,
        trace,
        prepared,
        reference,
        cycles=3,
        sync=lambda: None,
        snapshot=snapshot_factory(),
        memory=lambda: {
            "peak_allocated_bytes": 64 * 1024**2,
            "peak_reserved_bytes": 128 * 1024**2,
            "graph_retained_bytes": len(adapter.cache) * 1024**2,
        },
        clock=clock,
    )
    return measured, adapter


def test_traces_cover_real_buckets_and_both_capacity_and_residual_pressure():
    assert len(set(benchmark.build_trace("ring10-control"))) == 16
    assert len(set(benchmark.build_trace("mixed-6-10"))) == 32
    assert len(set(benchmark.build_trace("mixed-8-10"))) == 32
    assert len(set(benchmark.build_trace("mixed-6-8-10"))) == 48
    trace = benchmark.build_trace("mixed-6-8-10")
    assert sum(ring == 10 for ring, _ in trace) == 5 * sum(
        ring != 10 for ring, _ in trace
    )
    assert set(rows for _, rows in trace) == set(benchmark.BUCKETS)
    with pytest.raises(ValueError, match="unique"):
        benchmark.build_trace("mixed-6-10", (3, 3))


def test_cold_capture_and_recapture_are_charged_without_resetting_graphs():
    old, old_adapter = run_arm("mixed-6-10", 16)
    new, new_adapter = run_arm("mixed-6-10", 32)
    assert old["execution_valid"] and new["execution_valid"]
    assert new["cold_cycle"]["metrics"]["graph_captures"] == 32
    assert new["steady_totals"]["graph_captures"] == 0
    assert old["steady_totals"]["graph_captures"] > 0
    assert new["seconds"] < old["seconds"]
    assert (
        old_adapter.resets
        == new_adapter.resets
        == len(benchmark.build_trace("mixed-6-10")) * 3
    )
    assert new["metrics"]["graph_warmup_calls"] == 128
    assert new["seconds"] == sum(cycle["seconds"] for cycle in new["cycles"])


def test_byte_pressure_is_measured_even_with_32_entry_allowance():
    record, _ = run_arm("mixed-6-10", 32, byte_slots=24)
    assert record["steady_totals"]["graph_captures"] > 0
    assert record["metrics"]["graph_evictions"] > 0
    assert all(
        cycle["graph_retained_bytes"] <= 24 * 1024**2 for cycle in record["cycles"]
    )


def test_changed_predictions_cannot_pass_execution_gate():
    record, _ = run_arm("ring10-control", 32, corrupt=True)
    assert not record["output_parity_exact"] and not record["execution_valid"]
    assert all(len(cycle["output_digests"]) == 16 for cycle in record["cycles"])


def test_nonfinite_or_bad_routing_is_rejected():
    output = response(1)
    output.values[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        benchmark.response_digest(output, 1)
    with pytest.raises(ValueError, match="routing"):
        benchmark.response_digest(response(1), 2)


def fixture_report():
    profile = {
        "cuda_graph_max_entries": 16,
        "cuda_graphs": True,
        "small_batch_graph_buckets": True,
    }
    hashes = {
        "source_config_sha256": "a" * 64,
        "source_config_canonical_sha256": "b" * 64,
        "model_identity": "sha256-" + "c" * 64,
        "manifest_sha256": "d" * 64,
        "checkpoint_sha256": "e" * 64,
    }
    plan = {
        "schema_version": 1,
        "benchmark": "graph-cache-capacity",
        **hashes,
        "config_sha256": hashes["source_config_sha256"],
        "profile_inference": profile,
        "inference_configuration": profile,
        "control_entries": 16,
        "treatment_entries": 32,
        "entries": [16, 32],
        "graph_cache_bytes_per_arm": 8 * 1024**3,
        "production_graph_cache_bytes_per_model": 8 * 1024**3,
        "traces": {
            name: [list(key) for key in benchmark.build_trace(name)]
            for name in benchmark.SCENARIOS
        },
        "cycles": 3,
        "repeats": 2,
        "memory_fraction": 0.5,
        "actor_gpu_id": 1,
    }
    records = []
    for scenario in benchmark.SCENARIOS:
        for repeat in range(2):
            for entries in (16, 32) if repeat == 0 else (32, 16):
                record, _ = run_arm(scenario, entries)
                records.append(
                    {
                        "scenario": scenario,
                        "repeat": repeat,
                        "entries": entries,
                        "graph_cache_bytes": 8 * 1024**3,
                        **record,
                    }
                )
    keys = {tuple(key) for trace in plan["traces"].values() for key in trace}
    report = {
        "plan": plan,
        "plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True).encode()
        ).hexdigest(),
        "pid": os.getpid(),
        "gpu_uuid": "GPU-test",
        "device_name": "NVIDIA H100 80GB HBM3",
        "device_total_memory_bytes": 80 * 1024**3,
        "reference_response_sha256": {
            f"{ring}:{rows}": benchmark.response_digest(response(rows), rows)
            for ring, rows in keys
        },
        "records": records,
        "assessment": {"eligible_for_controlled_activation": False},
    }
    expected = {
        **hashes,
        "profile_inference": profile,
        "graph_cache_bytes": 8 * 1024**3,
        "actor_gpu_id": 1,
    }
    return report, expected


@pytest.fixture(scope="module")
def evidence():
    return fixture_report()


def test_evidence_recomputes_gate_instead_of_trusting_saved_success(evidence):
    report, expected = evidence
    assert validate_graph_cache_report(report, **expected)[
        "eligible_for_controlled_activation"
    ]


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_arm",
        "wrong_digest",
        "cache_hits",
        "memory",
        "owner",
        "speed",
        "reorder",
        "partial_trace",
        "changed_model",
        "warm_hidden",
    ],
)
def test_corrupt_or_incomplete_evidence_is_never_admitted(evidence, corruption):
    original, expected = evidence
    report = deepcopy(original)
    row = report["records"][1]
    if corruption == "missing_arm":
        report["records"].pop()
    elif corruption == "wrong_digest":
        row["cycles"][0]["output_digests"][0] = "f" * 64
    elif corruption == "cache_hits":
        row["cycles"][0]["metrics"]["cache_hits"] = 1
    elif corruption == "memory":
        row["cycles"][0]["graph_retained_bytes"] = 9 * 1024**3
    elif corruption == "owner":
        row["gpu_observations"][1]["owners"].append({"pid": 1234, "start_ticks": 99})
    elif corruption == "speed":
        for candidate in report["records"]:
            if candidate["entries"] == 32:
                for cycle in candidate["cycles"]:
                    cycle["seconds"] *= 100
                candidate["seconds"] *= 100
    elif corruption == "reorder":
        report["records"][0], report["records"][1] = (
            report["records"][1],
            report["records"][0],
        )
    elif corruption == "partial_trace":
        report["plan"]["traces"].pop("mixed-6-8-10")
        report["plan_sha256"] = hashlib.sha256(
            json.dumps(report["plan"], sort_keys=True).encode()
        ).hexdigest()
    elif corruption == "changed_model":
        expected = {**expected, "model_identity": "sha256-other"}
    elif corruption == "warm_hidden":
        row["cycles"][0]["metrics"]["graph_captures"] = 0
        row["cycles"][0]["metrics"]["graph_warmup_calls"] = 0
        row["cycles"][0]["metrics"]["graph_warmup_rows"] = 0
        row["cycles"][0]["metrics"]["graph_validation_replays"] = 0
    report["assessment"]["eligible_for_controlled_activation"] = True
    with pytest.raises(ValueError):
        validate_graph_cache_report(report, **expected)


def test_cli_bounds_and_manifest_alias():
    args = benchmark.parser().parse_args(
        ["--config", "config.yaml", "--manifest", "model.json"]
    )
    assert str(args.checkpoint) == "model.json"
    benchmark.validate(args)
    for field, value in (
        ("repeats", 3),
        ("cycles", 1),
        ("graph_cache_bytes", 9 * 1024**3),
        ("memory_fraction", float("nan")),
        ("batch_sizes", [3, 3]),
    ):
        invalid = SimpleNamespace(**vars(args))
        setattr(invalid, field, value)
        with pytest.raises(ValueError):
            benchmark.validate(invalid)
