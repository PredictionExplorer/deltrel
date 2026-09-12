"""Narrow authority for an explicitly requested, evidence-backed cache trial.

This does not rewrite benchmark policies, results, or success flags. Ordinary
graph admission remains separate. The caller verifies content-pinned files and
all graph execution evidence before passing recomputed assessments here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import PurePath
import re
from typing import Any

from .graph_cache_evidence import BOUNDED_NONINFERIORITY_POLICY, SCENARIOS


CONTROLLED_ACTIVATION_FORMAT = "startrain.graph-cache-controlled-activation"
LIVE_WORKLOAD_FORMAT = "startrain.graph-cache-live-workload"
ACTIVATION_SCOPE = "controlled-production-trial-not-established-throughput-or-Elo-gain"
GRAPH_CACHE_BYTES = 8 * 1024**3
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MODEL = re.compile(r"sha256-[0-9a-f]{64}\Z")


def _fail(message: str) -> None:
    raise ValueError("graph cache controlled activation: " + message)


def _equal(actual: Any, expected: Any, name: str) -> None:
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        _fail(f"{name} differs from pinned authority")


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 8192:
        _fail(f"{name} must contain bounded nonempty text")


def _sha(value: Any, name: str) -> None:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        _fail(f"{name} must be a lowercase SHA256")


def _reference(value: Any, name: str) -> None:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        _fail(f"{name} must be an exact content reference")
    path = value["path"]
    if (
        not isinstance(path, str)
        or not path
        or PurePath(path).is_absolute()
        or ".." in PurePath(path).parts
    ):
        _fail(f"{name} path must remain inside the run")
    _sha(value["sha256"], name)


def validate_controlled_activation_receipt(
    receipt: Mapping[str, Any],
    *,
    run_id: str,
    source_config_sha256: str,
    target_config_sha256: str,
    baseline_profile: Mapping[str, Any],
    graph_cache_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind a conditional deployment request without inventing a gate waiver.

    The immutable receipt records the verbatim request and the separate
    engineering rationale. No time expiry may invalidate running training.
    """
    required = {
        "format",
        "schema_version",
        "run_id",
        "source_config_sha256",
        "target_config_sha256",
        "baseline_profile",
        "graph_cache_reports",
        "authorization",
        "rationale",
        "prior_policy_failures_acknowledged",
        "activation_scope",
        "live_workload_evidence",
    }
    if set(receipt) != required:
        _fail("receipt fields are incomplete or unknown")
    for name, expected in (
        ("format", CONTROLLED_ACTIVATION_FORMAT),
        ("schema_version", 1),
        ("run_id", run_id),
        ("source_config_sha256", source_config_sha256),
        ("target_config_sha256", target_config_sha256),
        ("baseline_profile", dict(baseline_profile)),
        ("graph_cache_reports", list(graph_cache_reports)),
        ("prior_policy_failures_acknowledged", True),
        ("activation_scope", ACTIVATION_SCOPE),
    ):
        _equal(receipt[name], expected, name)
    _sha(receipt["source_config_sha256"], "source config")
    _sha(receipt["target_config_sha256"], "target config")
    if source_config_sha256 == target_config_sha256:
        _fail("trial requires a distinct cache32 target")
    authorization = receipt["authorization"]
    if not isinstance(authorization, dict) or set(authorization) != {"kind", "request"}:
        _fail("authorization must record its kind and verbatim request")
    _equal(authorization["kind"], "explicit-user-request", "authorization kind")
    _text(authorization["request"], "verbatim request")
    _text(receipt["rationale"], "engineering rationale")
    _reference(receipt["baseline_profile"], "original baseline")
    for reference in receipt["graph_cache_reports"]:
        _reference(reference, "original graph report")
    _reference(receipt["live_workload_evidence"], "live workload evidence")
    return dict(receipt["live_workload_evidence"])


def validate_live_workload_evidence(
    evidence: Mapping[str, Any],
    *,
    run_id: str,
    source_config_sha256: str,
    actor_gpu_ids: set[int],
) -> None:
    """Require actual process counter deltas and corroborating mixed residency.

    GPU-process counters aggregate its models. The model/rings describe resident
    context, not model-specific attribution of captures or elapsed performance.
    Additional embedded provenance is retained without inventing confidence.
    """
    required = {
        "format",
        "schema_version",
        "run_id",
        "source_config_sha256",
        "observed_from_ns",
        "observed_until_ns",
        "gpu_id",
        "process_pid",
        "model_identity",
        "rings",
        "counter_scope",
        "graph_captures_delta",
        "graph_evictions_delta",
        "useful_neural_rows_delta",
    }
    if not required <= set(evidence):
        _fail("live workload evidence is incomplete")
    for name, expected in (
        ("format", LIVE_WORKLOAD_FORMAT),
        ("schema_version", 1),
        ("run_id", run_id),
        ("source_config_sha256", source_config_sha256),
        ("counter_scope", "gpu_process"),
    ):
        _equal(evidence[name], expected, "live " + name)
    for name in (
        "observed_from_ns",
        "observed_until_ns",
        "process_pid",
        "graph_captures_delta",
        "graph_evictions_delta",
        "useful_neural_rows_delta",
    ):
        if type(evidence[name]) is not int or evidence[name] <= 0:
            _fail(f"live {name} must be a positive integer")
    if evidence["observed_until_ns"] <= evidence["observed_from_ns"]:
        _fail("live evidence interval is not ordered")
    if type(evidence["gpu_id"]) is not int or evidence["gpu_id"] not in actor_gpu_ids:
        _fail("live GPU is not an actor in the bound source configuration")
    if (
        not isinstance(evidence["model_identity"], str)
        or _MODEL.fullmatch(evidence["model_identity"]) is None
    ):
        _fail("live resident model identity is invalid")
    rings = evidence["rings"]
    if (
        not isinstance(rings, list)
        or len(rings) != 2
        or any(type(ring) is not int or ring not in (6, 8, 10) for ring in rings)
        or len(set(rings)) != len(rings)
        or set(rings) not in ({6, 10}, {8, 10})
    ):
        _fail("live evidence must corroborate a beneficial benchmarked two-board mix")


def validate_controlled_activation_assessments(
    assessments: Mapping[str, Mapping[str, Any]],
    full_target_rate_ratios: Mapping[str, float],
) -> dict[str, Any]:
    """Apply trial safeguards to freshly recomputed execution evidence.

    Retain the original failed policy assessments. This separately authorizes
    the controlled activation; it never labels those measurements as passes.
    """
    if len(assessments) < 2 or not set(assessments) <= set(full_target_rate_ratios):
        _fail("both frozen models and original teacher rates are required")
    floors = {}
    original_eligibility = {}
    for identity, assessment in assessments.items():
        if (
            assessment.get("performance_policy") != BOUNDED_NONINFERIORITY_POLICY
            or assessment.get("valid_complete_comparison") is not True
        ):
            _fail("trial requires complete original four-repeat confirmation evidence")
        eligibility = assessment.get("eligible_for_controlled_activation")
        if type(eligibility) is not bool:
            _fail("original eligibility is missing")
        original_eligibility[identity] = eligibility
        comparisons = assessment["comparisons"]
        if len(comparisons) != len(SCENARIOS) or {
            row["scenario"] for row in comparisons
        } != set(SCENARIOS):
            _fail("original comparison scenarios are incomplete")
        ratios = []
        reduced = False
        for row in comparisons:
            values = row["paired_speedup_ratios"]
            if len(values) != 4 or any(
                type(value) not in (float, int)
                or not math.isfinite(value)
                or value <= 0
                for value in values
            ):
                _fail("original paired timing ratios are invalid")
            ratios.extend(values)
            if row["scenario"] in ("mixed-6-10", "mixed-8-10"):
                ordered = sorted(values)
                if (ordered[1] + ordered[2]) / 2 < 1.05:
                    _fail("both two-board median gains must still exceed five percent")
            if row["scenario"] != "ring10-control":
                captures = row["graph_captures"]
                reduced |= captures[32] < captures[16]
        if not reduced:
            _fail("trial requires observed mixed-board capture reduction")
        original_rate = full_target_rate_ratios[identity]
        factor = min(1.0, *ratios)
        if (
            type(original_rate) not in (float, int)
            or not math.isfinite(original_rate)
            or original_rate < 1
            or original_rate * factor < 1
        ):
            _fail("original full-target production floor would be weakened")
        floors[identity] = {
            "original_full_target_rate_ratio": original_rate,
            "observed_execution_factor_capped_at_one": factor,
            "product": original_rate * factor,
        }
    if all(original_eligibility.values()):
        _fail(
            "ordinary passing evidence does not require a failed-policy trial receipt"
        )
    return {
        "activation_scope": ACTIVATION_SCOPE,
        "original_policy_eligibility": original_eligibility,
        "teacher_floor_calculations": floors,
        "qualification": "controlled-activation-authorized-original-policy-results-unchanged",
        "limitation": "Observed adapter and workload evidence supports expected benefit, not a proven fleet throughput or Elo gain or statistical lower bound.",
    }
