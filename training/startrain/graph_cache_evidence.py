"""Pure validation of frozen-model graph-capacity benchmark evidence.

This module never imports benchmark scripts, PyTorch, or native/GPU code. Saved
success flags are informational: admission is recomputed from the bound plan,
complete per-cycle counters, response fingerprints, memory and owner records.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
import statistics
from typing import Any


BUCKETS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
SCENARIOS = {
    "ring10-control": (10,),
    "mixed-6-10": (6, 10),
    "mixed-8-10": (8, 10),
    "mixed-6-8-10": (6, 8, 10),
}
STRICT_PERFORMANCE_POLICY = "strict-v1"
BOUNDED_NONINFERIORITY_POLICY = "bounded-noninferiority-v1"
PERFORMANCE_POLICIES = (STRICT_PERFORMANCE_POLICY, BOUNDED_NONINFERIORITY_POLICY)


def performance_policy_contract(policy: str) -> dict[str, Any]:
    """A prospective engineering rule, never a fitted confidence interval."""
    if policy == STRICT_PERFORMANCE_POLICY:
        return {"every_scenario_median_min": 1.0, "requires_capture_reduction": True}
    if policy == BOUNDED_NONINFERIORITY_POLICY:
        return {
            "counterbalanced_repeats": 4,
            "cycles": 2,
            "every_individual_pair_min": 0.995,
            "two_board_median_min": 1.05,
            "requires_capture_reduction": True,
            "statistical_confidence_claim": False,
        }
    raise ValueError("unknown graph cache performance policy")


# Actors start independent Python processes and do not enable the learner's
# fast FP32 math. Evidence for changing only cache capacity must preserve these
# settings, including the absence of environment-level TF32 overrides.
ACTOR_INFERENCE_MATH = {
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": True,
    "environment": {
        "NVIDIA_TF32_OVERRIDE": None,
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": None,
    },
}


def actor_inference_math_expected() -> dict[str, Any]:
    """Return an independently owned copy for plans and report fixtures."""
    return deepcopy(ACTOR_INFERENCE_MATH)


def validate_actor_inference_math(state: Any, *, phase: str) -> None:
    _equal(state, ACTOR_INFERENCE_MATH, f"actor inference math at {phase}")


def build_trace(
    scenario: str, buckets: tuple[int, ...] = BUCKETS
) -> list[tuple[int, int]]:
    """Reproducible capacity/byte-pressure stress, not an observed frequency mix."""
    rings = SCENARIOS[scenario]
    if (
        not buckets
        or len(set(buckets)) != len(buckets)
        or not set(buckets) <= set(BUCKETS)
    ):
        raise ValueError("buckets must be unique production graph sizes")
    if len(rings) == 1:
        return [(10, rows) for rows in buckets]
    trace = []
    for index, rows in enumerate(buckets):
        for ring in rings[:-1]:
            trace.extend(
                (10, buckets[(index + offset) % len(buckets)]) for offset in range(5)
            )
            trace.append((ring, rows))
    return trace


def assess(
    records: list[dict[str, Any]],
    scenarios: list[str],
    repeats: int,
    min_speedup: float,
    *,
    performance_policy: str = STRICT_PERFORMANCE_POLICY,
) -> dict[str, Any]:
    """Summarize benchmark observations; strict admission uses validation below."""
    policy_contract = performance_policy_contract(performance_policy)
    comparisons = []
    valid = len(records) == len(scenarios) * repeats * 2
    for scenario in scenarios:
        ratios = []
        captures = {16: 0, 32: 0}
        for repeat in range(repeats):
            rows = [
                row
                for row in records
                if row["scenario"] == scenario and row["repeat"] == repeat
            ]
            pair = {row["entries"]: row for row in rows}
            if len(rows) != 2 or set(pair) != {16, 32}:
                valid = False
                continue
            for entries, row in pair.items():
                valid = (
                    valid
                    and row["execution_valid"]
                    and row["load_assessment"]["comparison_adoptable"]
                )
                valid = valid and all(
                    cycle["graph_retained_bytes"] <= row["graph_cache_bytes"]
                    for cycle in row["cycles"]
                )
                captures[entries] += row["metrics"]["graph_captures"]
            if pair[16]["useful_rows"] != pair[32]["useful_rows"]:
                valid = False
            ratios.append(pair[16]["seconds"] / pair[32]["seconds"])
        median = statistics.median(ratios) if ratios else None
        comparisons.append(
            {
                "scenario": scenario,
                "paired_speedup_ratios": ratios,
                "median_speedup": median,
                "graph_captures": captures,
            }
        )
    mixed = [row for row in comparisons if row["scenario"] != "ring10-control"]
    all_ratios = [
        ratio for row in comparisons for ratio in row["paired_speedup_ratios"]
    ]
    capture_reduction = any(
        row["graph_captures"][32] < row["graph_captures"][16] for row in mixed
    )
    if performance_policy == BOUNDED_NONINFERIORITY_POLICY:
        two_board = [
            row
            for row in comparisons
            if row["scenario"] in ("mixed-6-10", "mixed-8-10")
        ]
        valid = (
            valid
            and repeats == 4
            and scenarios == list(SCENARIOS)
            and all(len(row["cycles"]) == 2 for row in records)
        )
        performance = (
            len(all_ratios) == 16
            and all(ratio >= 0.995 for ratio in all_ratios)
            and len(two_board) == 2
            and all(
                row["median_speedup"] is not None and row["median_speedup"] >= 1.05
                for row in two_board
            )
            and capture_reduction
        )
        description = "prospective observed noninferiority: every individual paired ratio>=0.995; both two-board medians>=1.05; four counterbalanced repeats and two cycles; no statistical confidence claim"
    else:
        performance = (
            bool(mixed)
            and all(
                row["median_speedup"] is not None and row["median_speedup"] >= 1.0
                for row in comparisons
            )
            and all(
                row["median_speedup"] is not None
                and row["median_speedup"] >= min_speedup
                for row in mixed
            )
            and capture_reduction
        )
        description = f"exact outputs, exclusive ownership, complete work, retained-byte bound; mixed-case median>={min_speedup}x; fewer captures in at least one mixed case; every-case median>=1x"
    return {
        "performance_policy": performance_policy,
        "performance_policy_contract": policy_contract,
        # Never credit apparent speedups to make otherwise inadequate original
        # teacher-rate evidence pass. Only measured slowdowns reduce its bound.
        "conservative_min_speedup": min(1.0, *all_ratios) if all_ratios else None,
        "comparisons": comparisons,
        "valid_complete_comparison": valid,
        "performance_gate_passed": performance,
        "eligible_for_controlled_activation": valid and performance,
        "gate": description,
        "limitation": "Synthetic adapter traces do not establish live self-play throughput or Elo/hour; ownership observations are point samples. The engineering margin is an observed bound, not statistical confidence.",
    }


def _integer(
    value: Any, name: str, *, minimum: int = 0, maximum: int | None = None
) -> int:
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"graph cache evidence has invalid {name}")
    return value


def _positive(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"graph cache evidence has invalid {name}")
    return float(value)


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"graph cache evidence has invalid {name}")
    return value


def _equal(actual: Any, expected: Any, name: str) -> None:
    # JSON equality distinguishes bool from int and rejects nonfinite values.
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise ValueError(f"graph cache evidence {name} differs from its authority")


def _owner_samples(samples: Any, *, pid: int, uuid: str, count: int) -> int:
    if not isinstance(samples, list) or len(samples) != count:
        raise ValueError("graph cache evidence ownership samples are incomplete")
    starts = set()
    previous = -1
    for sample in samples:
        if (
            not isinstance(sample, dict)
            or sample.get("verified") is not True
            or sample.get("gpu_uuid") != uuid
        ):
            raise ValueError("graph cache evidence has unverified GPU ownership")
        owners = sample.get("owners")
        if (
            not isinstance(owners, list)
            or len(owners) != 1
            or owners[0].get("pid") != pid
        ):
            raise ValueError("graph cache evidence GPU was shared")
        starts.add(_integer(owners[0].get("start_ticks"), "owner start", minimum=1))
        stamp = _integer(sample.get("observed_ns"), "observation time", minimum=1)
        if stamp <= previous:
            raise ValueError("graph cache evidence ownership samples are not ordered")
        previous = stamp
    if len(starts) != 1:
        raise ValueError("graph cache evidence owner identity changed")
    return next(iter(starts))


def validate_graph_cache_report(
    report: Mapping[str, Any],
    *,
    source_config_sha256: str,
    source_config_canonical_sha256: str,
    model_identity: str,
    manifest_sha256: str,
    checkpoint_sha256: str,
    profile_inference: Mapping[str, Any],
    graph_cache_bytes: int,
    actor_gpu_id: int | None = None,
) -> dict[str, Any]:
    """Validate execution and require the originally planned performance gate."""
    return _validate_graph_cache_report(
        report,
        source_config_sha256=source_config_sha256,
        source_config_canonical_sha256=source_config_canonical_sha256,
        model_identity=model_identity,
        manifest_sha256=manifest_sha256,
        checkpoint_sha256=checkpoint_sha256,
        profile_inference=profile_inference,
        graph_cache_bytes=graph_cache_bytes,
        actor_gpu_id=actor_gpu_id,
        require_performance=True,
    )


def validate_graph_cache_execution_report(
    report: Mapping[str, Any],
    *,
    source_config_sha256: str,
    source_config_canonical_sha256: str,
    model_identity: str,
    manifest_sha256: str,
    checkpoint_sha256: str,
    profile_inference: Mapping[str, Any],
    graph_cache_bytes: int,
    actor_gpu_id: int | None = None,
) -> dict[str, Any]:
    """Validate all execution evidence without authorizing an adoption or trial.

    Only the final performance rejection is returned rather than raised. The
    original recomputed performance result, including false eligibility, remains
    unchanged. Callers need a separately verified, explicitly scoped authority to
    act on a report whose planned performance gate failed. Model/config/math,
    oracle, exact outputs, counters, memory, ownership and ordering checks are
    identical to the strict API; malformed evidence still raises.
    """
    return _validate_graph_cache_report(
        report,
        source_config_sha256=source_config_sha256,
        source_config_canonical_sha256=source_config_canonical_sha256,
        model_identity=model_identity,
        manifest_sha256=manifest_sha256,
        checkpoint_sha256=checkpoint_sha256,
        profile_inference=profile_inference,
        graph_cache_bytes=graph_cache_bytes,
        actor_gpu_id=actor_gpu_id,
        require_performance=False,
    )


def _validate_graph_cache_report(
    report: Mapping[str, Any],
    *,
    source_config_sha256: str,
    source_config_canonical_sha256: str,
    model_identity: str,
    manifest_sha256: str,
    checkpoint_sha256: str,
    profile_inference: Mapping[str, Any],
    graph_cache_bytes: int,
    actor_gpu_id: int | None = None,
    require_performance: bool,
) -> dict[str, Any]:
    """Return independently recomputed admission, or raise on invalid evidence.

    The caller supplies trusted frozen baseline and model identities and checks
    the report file's content hash. A passing report supports only the exact
    16-to-32 entry change with the original byte cap and inference configuration.
    """
    try:
        plan = report["plan"]
        if not isinstance(plan, dict):
            raise ValueError("graph cache evidence plan must be an object")
        _equal(plan["schema_version"], 1, "schema")
        _equal(plan["benchmark"], "graph-cache-capacity", "benchmark")
        actual_plan_hash = hashlib.sha256(
            json.dumps(plan, sort_keys=True).encode()
        ).hexdigest()
        _equal(_sha(report["plan_sha256"], "plan hash"), actual_plan_hash, "plan hash")
        for name, expected in (
            ("source_config_sha256", source_config_sha256),
            ("source_config_canonical_sha256", source_config_canonical_sha256),
            ("model_identity", model_identity),
            ("manifest_sha256", manifest_sha256),
            ("checkpoint_sha256", checkpoint_sha256),
        ):
            _equal(plan[name], expected, name)
        _equal(plan["config_sha256"], source_config_sha256, "raw config alias")
        _equal(plan["profile_inference"], dict(profile_inference), "profile inference")
        _equal(
            plan["inference_configuration"], dict(profile_inference), "inference alias"
        )
        _equal(plan["control_entries"], 16, "control entry count")
        _equal(plan["treatment_entries"], 32, "treatment entry count")
        _equal(plan["entries"], [16, 32], "entry choices")
        _equal(
            plan["reference_execution"], "production-graph-16", "reference execution"
        )
        validate_actor_inference_math(plan["actor_inference_math"], phase="plan")
        for phase in (
            "worker_initial_math",
            "priming_math_before",
            "priming_math_after",
        ):
            validate_actor_inference_math(report[phase], phase=phase)
        _equal(
            profile_inference.get("cuda_graph_max_entries"),
            16,
            "baseline entry capacity",
        )
        _equal(profile_inference.get("cuda_graphs"), True, "baseline graphs")
        _equal(
            profile_inference.get("small_batch_graph_buckets"), True, "baseline buckets"
        )
        byte_limit = _integer(
            graph_cache_bytes,
            "trusted byte limit",
            minimum=1024**2,
            maximum=8 * 1024**3,
        )
        _equal(plan["graph_cache_bytes_per_arm"], byte_limit, "byte cap")
        _equal(
            plan["production_graph_cache_bytes_per_model"],
            byte_limit,
            "production byte cap",
        )
        if actor_gpu_id is not None:
            _equal(plan["actor_gpu_id"], actor_gpu_id, "actor GPU")
        expected_traces = {
            name: [list(key) for key in build_trace(name)] for name in SCENARIOS
        }
        _equal(plan["traces"], expected_traces, "complete production traces")
        repeats = _integer(plan["repeats"], "repeats", minimum=2, maximum=6)
        cycles = _integer(plan["cycles"], "cycles", minimum=2, maximum=8)
        if repeats % 2:
            raise ValueError("graph cache evidence is not counterbalanced")
        policy = plan.get("performance_policy", STRICT_PERFORMANCE_POLICY)
        contract = performance_policy_contract(policy)
        if (
            "performance_policy_contract" in plan
            or policy == BOUNDED_NONINFERIORITY_POLICY
        ):
            _equal(
                plan["performance_policy_contract"],
                contract,
                "prospective performance contract",
            )
        if policy == BOUNDED_NONINFERIORITY_POLICY and (repeats != 4 or cycles != 2):
            raise ValueError(
                "bounded noninferiority requires four prospective repeats and two cycles"
            )
        if "H100" not in report["device_name"]:
            raise ValueError("graph cache evidence was not measured on H100")
        pid = _integer(report["pid"], "PID", minimum=1)
        uuid = report["gpu_uuid"]
        if not isinstance(uuid, str) or not uuid.startswith("GPU-"):
            raise ValueError("graph cache evidence GPU UUID is invalid")
        total_memory = _integer(
            report["device_total_memory_bytes"], "device memory", minimum=1
        )
        fraction = _positive(plan["memory_fraction"], "memory fraction")
        if not 0.1 <= fraction <= 0.75:
            raise ValueError("graph cache evidence memory fraction is out of bounds")
        references = report["reference_response_sha256"]
        reference_keys = {
            f"{ring}:{rows}"
            for trace in expected_traces.values()
            for ring, rows in trace
        }
        if not isinstance(references, dict) or set(references) != reference_keys:
            raise ValueError("graph cache evidence reference responses are incomplete")
        for digest in references.values():
            _sha(digest, "reference response")
        priming = report["reference_priming"]
        expected_shapes = sorted(
            {tuple(key) for trace in expected_traces.values() for key in trace}
        )
        _equal(priming["execution"], "production-graph-16", "reference execution")
        _equal(priming["entries"], 16, "reference cache capacity")
        _equal(priming["cuda_graphs"], True, "reference graphs")
        _equal(
            priming["requested_shapes"],
            [list(key) for key in expected_shapes],
            "reference capture shapes",
        )
        _equal(priming["cache_empty_after_close"], True, "discarded priming graphs")
        for phase in ("math_before", "math_after"):
            validate_actor_inference_math(priming[phase], phase=f"reference/{phase}")
        _positive(priming["seconds"], "reference priming duration")
        priming_metrics = priming["metrics"]
        for name, expected in (
            ("evaluator_calls", len(expected_shapes)),
            ("neural_calls", len(expected_shapes)),
            ("graph_captures", len(expected_shapes)),
            ("graph_replays", len(expected_shapes)),
            ("graph_validation_replays", len(expected_shapes)),
            ("graph_warmup_calls", 4 * len(expected_shapes)),
            ("evaluator_rows", sum(size for _, size in expected_shapes)),
            ("neural_rows", sum(size for _, size in expected_shapes)),
        ):
            _equal(priming_metrics[name], expected, f"reference {name}")
        for name in (
            "cache_hits",
            "deduplicated_rows",
            "neural_padding_rows",
            "graph_fallbacks",
            "graph_validation_failures",
        ):
            _equal(priming_metrics[name], 0, f"reference {name}")
        records = report["records"]
        expected_sequence = [
            (scenario, repeat, entries)
            for scenario in SCENARIOS
            for repeat in range(repeats)
            for entries in ((16, 32) if repeat % 2 == 0 else (32, 16))
        ]
        if not isinstance(records, list) or len(records) != len(expected_sequence):
            raise ValueError("graph cache evidence arm records are incomplete")
        verified_records = []
        owner_starts = set()
        previous_arm_end = -1
        for record, (scenario, repeat, entries) in zip(
            records, expected_sequence, strict=True
        ):
            _equal(
                [record["scenario"], record["repeat"], record["entries"]],
                [scenario, repeat, entries],
                "arm order",
            )
            _equal(record["graph_cache_bytes"], byte_limit, "arm byte cap")
            for phase in ("math_before", "math_after"):
                validate_actor_inference_math(
                    record[phase], phase=f"{scenario}/{repeat}/{entries}/{phase}"
                )
            trace = expected_traces[scenario]
            calls, rows = len(trace), sum(key[1] for key in trace)
            cycle_records = record["cycles"]
            if not isinstance(cycle_records, list) or len(cycle_records) != cycles:
                raise ValueError("graph cache evidence cycle records are incomplete")
            owner_starts.add(
                _owner_samples(
                    record["gpu_observations"], pid=pid, uuid=uuid, count=cycles + 1
                )
            )
            if record["gpu_observations"][0]["observed_ns"] <= previous_arm_end:
                raise ValueError("graph cache evidence arms were not measured serially")
            previous_arm_end = record["gpu_observations"][-1]["observed_ns"]
            sums: dict[str, int] = {}
            elapsed = 0.0
            for index, cycle in enumerate(cycle_records):
                _equal(cycle["cycle"], index, "cycle index")
                elapsed += _positive(cycle["seconds"], "cycle duration")
                _equal(
                    cycle["output_digests"],
                    [references[f"{ring}:{size}"] for ring, size in trace],
                    "exact response parity",
                )
                allocated = _integer(
                    cycle["peak_allocated_bytes"], "peak allocated", minimum=1
                )
                reserved = _integer(
                    cycle["peak_reserved_bytes"], "peak reserved", minimum=allocated
                )
                if allocated > total_memory * fraction or reserved > total_memory:
                    raise ValueError("graph cache evidence memory limit was exceeded")
                _integer(
                    cycle["graph_retained_bytes"], "retained bytes", maximum=byte_limit
                )
                metrics = cycle["metrics"]
                for name, expected in (
                    ("evaluator_rows", rows),
                    ("neural_rows", rows),
                    ("evaluator_calls", calls),
                    ("neural_calls", calls),
                    ("graph_replays", calls),
                ):
                    _equal(metrics[name], expected, name)
                for name in (
                    "cache_hits",
                    "deduplicated_rows",
                    "neural_padding_rows",
                    "graph_fallbacks",
                    "graph_validation_failures",
                ):
                    _equal(metrics[name], 0, name)
                captures = _integer(
                    metrics["graph_captures"], "captures", maximum=calls
                )
                _integer(metrics["graph_evictions"], "evictions", maximum=calls)
                _equal(
                    metrics["graph_warmup_calls"],
                    captures * 4,
                    "capture/warmup accounting",
                )
                _equal(
                    metrics["graph_validation_replays"],
                    captures,
                    "capture validation accounting",
                )
                _integer(
                    metrics["graph_warmup_rows"],
                    "warmup rows",
                    minimum=captures * 4,
                    maximum=captures * 4 * max(BUCKETS),
                )
                if index == 0 and captures < min(entries, len(set(map(tuple, trace)))):
                    raise ValueError(
                        "graph cache evidence did not start with an empty cache"
                    )
                for name, value in metrics.items():
                    if type(value) is int:
                        _integer(value, name)
                        sums[name] = sums.get(name, 0) + value
            for name, value in sums.items():
                _equal(record["metrics"][name], value, f"total {name}")
            if not math.isclose(
                _positive(record["seconds"], "arm duration"),
                elapsed,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "graph cache evidence arm time differs from its cycles"
                )
            _equal(record["useful_rows"], rows * cycles, "useful rows")
            verified_records.append(
                {
                    **record,
                    "seconds": elapsed,
                    "execution_valid": True,
                    "load_assessment": {"comparison_adoptable": True},
                }
            )
        if len(owner_starts) != 1:
            raise ValueError("graph cache evidence owner identity changed between arms")
        assessment = assess(
            verified_records, list(SCENARIOS), repeats, 1.0, performance_policy=policy
        )
        if require_performance and not assessment["eligible_for_controlled_activation"]:
            raise ValueError(
                "graph cache evidence fails recomputed nonregression/capture gate"
            )
        return assessment
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("graph cache evidence is malformed or incomplete") from error
