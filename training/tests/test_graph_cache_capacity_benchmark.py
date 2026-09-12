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
from startrain.graph_cache_evidence import (
    ACTOR_INFERENCE_MATH,
    BOUNDED_NONINFERIORITY_POLICY,
    STRICT_PERFORMANCE_POLICY,
    actor_inference_math_expected,
    performance_policy_contract,
    validate_actor_inference_math,
    validate_graph_cache_execution_report,
    validate_graph_cache_report,
)
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
        self.config = SimpleNamespace(cuda_graphs=True, cuda_graph_max_entries=entries)
        self.closed = False

    def prepare_requests(self, request):
        return request

    def close(self):
        self.closed = True
        self.cache.clear()

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


def run_arm(scenario, entries, *, byte_slots=64, corrupt=False, cycles=3):
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
        cycles=cycles,
        sync=lambda: None,
        snapshot=snapshot_factory(),
        memory=lambda: {
            "peak_allocated_bytes": 64 * 1024**2,
            "peak_reserved_bytes": 128 * 1024**2,
            "graph_retained_bytes": len(adapter.cache) * 1024**2,
        },
        clock=clock,
        math_state=actor_inference_math_expected,
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


def fixture_report(
    *, cycles=2, repeats=2, performance_policy=STRICT_PERFORMANCE_POLICY
):
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
        "actor_inference_math": actor_inference_math_expected(),
        "reference_execution": "production-graph-16",
        "performance_policy": performance_policy,
        "performance_policy_contract": performance_policy_contract(performance_policy),
        "graph_cache_bytes_per_arm": 8 * 1024**3,
        "production_graph_cache_bytes_per_model": 8 * 1024**3,
        "traces": {
            name: [list(key) for key in benchmark.build_trace(name)]
            for name in benchmark.SCENARIOS
        },
        "cycles": cycles,
        "repeats": repeats,
        "memory_fraction": 0.5,
        "actor_gpu_id": 1,
    }
    records = []
    for scenario in benchmark.SCENARIOS:
        for repeat in range(repeats):
            for entries in (16, 32) if repeat % 2 == 0 else (32, 16):
                record, _ = run_arm(scenario, entries, cycles=cycles)
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
    clock = Clock()
    _, priming = benchmark.prime_reference(
        lambda entries: Adapter(clock, entries),
        {key: key for key in keys},
        sync=lambda: None,
        math_state=actor_inference_math_expected,
        clock=clock,
    )
    report = {
        "plan": plan,
        "plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True).encode()
        ).hexdigest(),
        "pid": os.getpid(),
        "gpu_uuid": "GPU-test",
        "device_name": "NVIDIA H100 80GB HBM3",
        "device_total_memory_bytes": 80 * 1024**3,
        "worker_initial_math": actor_inference_math_expected(),
        "priming_math_before": actor_inference_math_expected(),
        "priming_math_after": actor_inference_math_expected(),
        "reference_priming": priming,
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
    assert args.cycles == 2 and args.repeats == 2
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


def test_two_cycles_retain_cold_cost_and_complete_capacity_revisit():
    old, _ = run_arm("mixed-6-10", 16, cycles=2)
    new, _ = run_arm("mixed-6-10", 32, cycles=2)
    assert new["cold_cycle"]["metrics"]["graph_captures"] == 32
    assert new["steady_totals"]["graph_captures"] == 0
    assert old["steady_totals"]["graph_captures"] > 0
    assert len(old["cycles"]) == len(new["cycles"]) == 2
    assert old["useful_rows"] == new["useful_rows"]


def test_reference_uses_graph16_captures_every_shape_and_closes_before_timing():
    clock = Clock()
    created = []

    def make_adapter(entries):
        adapter = Adapter(clock, entries)
        created.append(adapter)
        return adapter

    keys = {key for name in benchmark.SCENARIOS for key in benchmark.build_trace(name)}
    reference, priming = benchmark.prime_reference(
        make_adapter,
        {key: key for key in keys},
        sync=lambda: None,
        math_state=actor_inference_math_expected,
        clock=clock,
    )
    assert len(created) == 1 and created[0].entries == 16
    assert created[0].closed and not created[0].cache
    assert priming["execution"] == "production-graph-16"
    assert priming["metrics"]["graph_captures"] == 48
    assert priming["metrics"]["graph_replays"] == 48
    assert priming["cache_empty_after_close"]
    assert len(reference) == 48
    # A measured control is a fresh object: its first cycle must still capture.
    control, _ = run_arm("ring10-control", 16, cycles=2)
    assert control["cold_cycle"]["metrics"]["graph_captures"] == 16


@pytest.mark.parametrize(
    "corruption",
    ["graph_off", "missing_capture", "retained_graphs", "incomplete_shapes"],
)
def test_nonproduction_reference_or_hidden_priming_is_rejected(evidence, corruption):
    original, expected = evidence
    report = deepcopy(original)
    priming = report["reference_priming"]
    if corruption == "graph_off":
        priming["cuda_graphs"] = False
    elif corruption == "missing_capture":
        priming["metrics"]["graph_captures"] -= 1
    elif corruption == "retained_graphs":
        priming["cache_empty_after_close"] = False
    else:
        priming["requested_shapes"].pop()
    with pytest.raises(ValueError):
        validate_graph_cache_report(report, **expected)


def test_math_reader_observes_flags_and_environment_without_normalizing_them():
    torch = SimpleNamespace(
        get_float32_matmul_precision=lambda: "high",
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=False),
        ),
    )
    environment = {"NVIDIA_TF32_OVERRIDE": "1"}
    actual = benchmark.read_actor_inference_math(torch, environment)
    assert actual == {
        "float32_matmul_precision": "high",
        "cuda_matmul_allow_tf32": True,
        "cudnn_allow_tf32": False,
        "environment": {
            "NVIDIA_TF32_OVERRIDE": "1",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": None,
        },
    }
    assert environment == {"NVIDIA_TF32_OVERRIDE": "1"}
    with pytest.raises(ValueError, match="actor inference math"):
        validate_actor_inference_math(actual, phase="test")


@pytest.mark.parametrize(
    "phase",
    [
        "plan",
        "worker_initial_math",
        "priming_math_before",
        "priming_math_after",
        "math_before",
        "math_after",
    ],
)
@pytest.mark.parametrize(
    "alteration", ["missing", "precision", "cuda", "cudnn", "environment"]
)
def test_admission_requires_actor_math_at_every_phase(evidence, phase, alteration):
    original, expected = evidence
    report = deepcopy(original)
    if phase == "plan":
        owner, key = report["plan"], "actor_inference_math"
    elif phase in ("math_before", "math_after"):
        owner, key = report["records"][0], phase
    else:
        owner, key = report, phase
    if alteration == "missing":
        owner.pop(key)
    elif alteration == "precision":
        owner[key]["float32_matmul_precision"] = "high"
    elif alteration == "cuda":
        owner[key]["cuda_matmul_allow_tf32"] = True
    elif alteration == "cudnn":
        owner[key]["cudnn_allow_tf32"] = False
    else:
        owner[key]["environment"]["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
    report["plan_sha256"] = hashlib.sha256(
        json.dumps(report["plan"], sort_keys=True).encode()
    ).hexdigest()
    report["assessment"]["eligible_for_controlled_activation"] = True
    with pytest.raises(ValueError):
        validate_graph_cache_report(report, **expected)


def test_math_contract_copies_and_untyped_flags_are_not_accepted():
    state = actor_inference_math_expected()
    state["environment"]["NVIDIA_TF32_OVERRIDE"] = "0"
    assert ACTOR_INFERENCE_MATH["environment"]["NVIDIA_TF32_OVERRIDE"] is None
    with pytest.raises(ValueError):
        validate_actor_inference_math(state, phase="override")
    state = actor_inference_math_expected()
    state["cuda_matmul_allow_tf32"] = 0
    with pytest.raises(ValueError):
        validate_actor_inference_math(state, phase="untyped")


def rehash_plan(report):
    report["plan_sha256"] = hashlib.sha256(
        json.dumps(report["plan"], sort_keys=True).encode()
    ).hexdigest()


def set_paired_ratio(report, scenario, repeat, ratio):
    pair = {
        row["entries"]: row
        for row in report["records"]
        if row["scenario"] == scenario and row["repeat"] == repeat
    }
    treatment = pair[32]
    factor = pair[16]["seconds"] / (ratio * treatment["seconds"])
    for cycle in treatment["cycles"]:
        cycle["seconds"] *= factor
    treatment["seconds"] = sum(cycle["seconds"] for cycle in treatment["cycles"])
    treatment["steady_totals"]["seconds"] = sum(
        cycle["seconds"] for cycle in treatment["cycles"][1:]
    )


@pytest.fixture(scope="module")
def bounded_evidence():
    return fixture_report(
        repeats=4, cycles=2, performance_policy=BOUNDED_NONINFERIORITY_POLICY
    )


def test_prospective_bounded_policy_accepts_small_observed_noninferiority(
    bounded_evidence,
):
    source, expected = bounded_evidence
    report = deepcopy(source)
    set_paired_ratio(report, "ring10-control", 1, 0.9951)
    set_paired_ratio(report, "mixed-6-8-10", 3, 0.997)
    report["assessment"] = {
        "eligible_for_controlled_activation": False,
        "conservative_min_speedup": 99,
    }
    result = validate_graph_cache_report(report, **expected)
    assert result["performance_policy"] == BOUNDED_NONINFERIORITY_POLICY
    assert result["eligible_for_controlled_activation"]
    assert result["conservative_min_speedup"] == pytest.approx(0.9951)
    assert (
        result["performance_policy_contract"]["statistical_confidence_claim"] is False
    )


@pytest.mark.parametrize("scenario", list(benchmark.SCENARIOS))
def test_each_individual_pair_must_respect_margin_even_when_median_improves(
    bounded_evidence, scenario
):
    source, expected = bounded_evidence
    report = deepcopy(source)
    set_paired_ratio(report, scenario, 2, 0.9949)
    with pytest.raises(ValueError, match="nonregression"):
        validate_graph_cache_report(report, **expected)


@pytest.mark.parametrize("scenario", ["mixed-6-10", "mixed-8-10"])
def test_both_two_board_medians_need_five_percent_improvement(
    bounded_evidence, scenario
):
    source, expected = bounded_evidence
    report = deepcopy(source)
    for repeat in range(4):
        set_paired_ratio(report, scenario, repeat, 1.049)
    with pytest.raises(ValueError, match="nonregression"):
        validate_graph_cache_report(report, **expected)


def test_v2_report_cannot_be_relabelled_as_prospective_confirmation(evidence):
    source, expected = evidence
    report = deepcopy(source)
    report["plan"]["performance_policy"] = BOUNDED_NONINFERIORITY_POLICY
    report["plan"]["performance_policy_contract"] = performance_policy_contract(
        BOUNDED_NONINFERIORITY_POLICY
    )
    rehash_plan(report)
    with pytest.raises(ValueError, match="four prospective repeats"):
        validate_graph_cache_report(report, **expected)
    report["plan"]["repeats"] = 4
    rehash_plan(report)
    with pytest.raises(ValueError, match="records are incomplete"):
        validate_graph_cache_report(report, **expected)


@pytest.mark.parametrize(
    "change",
    [
        "missing_contract",
        "changed_margin",
        "wrong_cycles",
        "wrong_repeats",
        "unknown_policy",
    ],
)
def test_bounded_contract_is_explicit_fixed_and_complete(bounded_evidence, change):
    source, expected = bounded_evidence
    report = deepcopy(source)
    plan = report["plan"]
    if change == "missing_contract":
        plan.pop("performance_policy_contract")
    elif change == "changed_margin":
        plan["performance_policy_contract"]["every_individual_pair_min"] = 0.99
    elif change == "wrong_cycles":
        plan["cycles"] = 3
    elif change == "wrong_repeats":
        plan["repeats"] = 6
    else:
        plan["performance_policy"] = "accept-anything"
    rehash_plan(report)
    with pytest.raises(ValueError):
        validate_graph_cache_report(report, **expected)


def test_missing_policy_retains_strict_legacy_semantics(evidence):
    source, expected = evidence
    report = deepcopy(source)
    report["plan"].pop("performance_policy")
    report["plan"].pop("performance_policy_contract")
    rehash_plan(report)
    assert (
        validate_graph_cache_report(report, **expected)["performance_policy"]
        == STRICT_PERFORMANCE_POLICY
    )
    for repeat in (0, 1):
        set_paired_ratio(report, "ring10-control", repeat, 0.999)
    with pytest.raises(ValueError, match="nonregression"):
        validate_graph_cache_report(report, **expected)


def test_bounded_cli_cannot_silently_reduce_confirmation_work():
    args = benchmark.parser().parse_args(
        [
            "--config",
            "profile.yaml",
            "--manifest",
            "model.json",
            "--performance-policy",
            BOUNDED_NONINFERIORITY_POLICY,
            "--repeats",
            "4",
            "--cycles",
            "2",
        ]
    )
    benchmark.validate(args)
    default = benchmark.parser().parse_args(
        ["--config", "profile.yaml", "--manifest", "model.json"]
    )
    assert default.performance_policy == STRICT_PERFORMANCE_POLICY
    for key, value in (
        ("repeats", 2),
        ("cycles", 3),
        ("batch_sizes", [64]),
        ("scenarios", ["mixed-6-10"]),
        ("minimum_speedup", 1.05),
    ):
        altered = SimpleNamespace(**vars(args))
        setattr(altered, key, value)
        with pytest.raises(ValueError, match="bounded noninferiority"):
            benchmark.validate(altered)


def test_conservative_factor_never_credits_apparent_speedups(bounded_evidence):
    source, expected = bounded_evidence
    report = deepcopy(source)
    for scenario in benchmark.SCENARIOS:
        for repeat in range(4):
            set_paired_ratio(report, scenario, repeat, 1.10)
    assert (
        validate_graph_cache_report(report, **expected)["conservative_min_speedup"]
        == 1.0
    )


def test_execution_only_validation_preserves_failed_performance_and_input(
    bounded_evidence,
):
    source, expected = bounded_evidence
    report = deepcopy(source)
    set_paired_ratio(report, "ring10-control", 0, 0.98)
    before = json.dumps(report, sort_keys=True)
    result = validate_graph_cache_execution_report(report, **expected)
    assert result["valid_complete_comparison"]
    assert not result["performance_gate_passed"]
    assert not result["eligible_for_controlled_activation"]
    assert result["conservative_min_speedup"] == pytest.approx(0.98)
    assert json.dumps(report, sort_keys=True) == before
    with pytest.raises(ValueError, match="fails recomputed nonregression/capture gate"):
        validate_graph_cache_report(report, **expected)


def test_execution_only_and_strict_use_identical_validation_for_passing_report(
    evidence,
):
    report, expected = evidence
    assert validate_graph_cache_execution_report(
        report, **expected
    ) == validate_graph_cache_report(report, **expected)


@pytest.mark.parametrize(
    "failure",
    [
        "math",
        "oracle",
        "parity",
        "memory",
        "ownership",
        "ordering",
        "counters",
        "model",
        "missing_arm",
        "plan_hash",
    ],
)
def test_execution_only_does_not_bypass_any_execution_guard(evidence, failure):
    source, expected = evidence
    report = deepcopy(source)
    first = report["records"][0]
    if failure == "math":
        report["worker_initial_math"]["cuda_matmul_allow_tf32"] = True
    elif failure == "oracle":
        report["reference_priming"]["cuda_graphs"] = False
    elif failure == "parity":
        first["cycles"][0]["output_digests"][0] = "f" * 64
    elif failure == "memory":
        first["cycles"][0]["graph_retained_bytes"] = 9 * 1024**3
    elif failure == "ownership":
        first["gpu_observations"][0]["owners"][0]["pid"] += 1
    elif failure == "ordering":
        report["records"][0], report["records"][1] = (
            report["records"][1],
            report["records"][0],
        )
    elif failure == "counters":
        first["cycles"][0]["metrics"]["neural_calls"] += 1
    elif failure == "model":
        expected = {**expected, "model_identity": "sha256-other-model"}
    elif failure == "missing_arm":
        report["records"].pop()
    else:
        report["plan_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        validate_graph_cache_execution_report(report, **expected)
