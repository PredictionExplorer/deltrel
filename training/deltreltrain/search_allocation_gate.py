"""Offline evidence required to waive a production per-ring search floor.

The raw global floor remains the responsibility of the production validator.
This module never grants an exception to it or changes profile serialization.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections import OrderedDict
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, NoReturn

from .config import ExperimentConfig, load_config
from .config_compatibility import without_arena_clinch_default
from .contracts import SEARCH_ALGORITHM_ID


GATE_FORMAT = "deltreltrain.ring-search-allocation-gate"
POLICY_TRANSITION_FORMAT = "deltreltrain.search-allocation-policy-transition"
POLICY_TRANSITION_CLASS = "unchanged-execution-policy-transition"
POLICY_TRANSITION_SCOPE = "original-frozen-six-variant-workload"
PROMOTION_TRANSITION_FORMAT = "deltreltrain.search-allocation-promotion-transition"
PROMOTION_TRANSITION_CLASS = (
    "unchanged-selfplay-execution-promotion-allocation-transition"
)
PROMOTION_TRANSITION_SCOPE = POLICY_TRANSITION_SCOPE
AUXILIARY_TRANSITION_FORMAT = "deltreltrain.search-allocation-auxiliary-transition"
AUXILIARY_TRANSITION_CLASS = (
    "unchanged-search-execution-auxiliary-prediction-transition"
)
AUXILIARY_TRANSITION_SCOPE = POLICY_TRANSITION_SCOPE
FULL_PROBABILITY_FLOOR = 0.35
MAX_INCREMENTAL_REGRET_UPPER95 = 0.02
TIMING_SETUP_COUNTERS = (
    "graph_captures",
    "graph_warmup_calls",
    "graph_evictions",
    "graph_fallbacks",
    "graph_validation_failures",
    "graph_validation_replays",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_REPORT_BYTES = 256 * 1024**2
_ANALYSES: OrderedDict[tuple[object, ...], dict[str, Any]] = OrderedDict()
_VERIFIED_GATES: OrderedDict[str, dict[str, tuple[int, ...]]] = OrderedDict()


def canonical_config_sha256(config: ExperimentConfig) -> str:
    # Adding a disabled arena optimization must not orphan existing search
    # admission artifacts. Enabled treatment values retain distinct authority.
    payload = without_arena_clinch_default(config.as_dict())
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def allocation_gate_path(config: ExperimentConfig) -> Path:
    return (
        Path(config.orchestration.directories.root).expanduser().resolve()
        / "status"
        / "search-allocation-gates"
        / f"{canonical_config_sha256(config)}.json"
    )


def _validate_graph_capacity_reports(
    root: Path,
    gate: dict[str, Any],
    baseline: ExperimentConfig,
    target: ExperimentConfig,
    verified: dict[str, tuple[int, ...]],
    full_target_rate_ratios: Mapping[str, float],
) -> None:
    """Extend measured search authority only for a proven cache-entry increase.

    Original search-quality and target-production evidence remains mandatory.
    These separately pinned H100 comparisons cover the same frozen models and
    preserve prediction settings, graph byte limits, and producer topology.
    """
    from .graph_cache_evidence import (
        BOUNDED_NONINFERIORITY_POLICY,
        validate_graph_cache_execution_report,
        validate_graph_cache_report,
    )
    from .graph_cache_controlled_activation import (
        GRAPH_CACHE_BYTES,
        validate_controlled_activation_assessments,
        validate_controlled_activation_receipt,
        validate_live_workload_evidence,
    )

    before = baseline.orchestration.model_refresh.inference
    after = target.orchestration.model_refresh.inference
    references = gate.get("graph_cache_reports")
    controlled = "graph_cache_controlled_activation" in gate
    if before == after:
        if "graph_cache_reports" in gate or controlled:
            _fail("graph cache reports require a cache-capacity treatment")
        return
    if (
        before.cuda_graph_max_entries != 16
        or after.cuda_graph_max_entries != 32
        or replace(after, cuda_graph_max_entries=16) != before
        or baseline.orchestration.gpus != target.orchestration.gpus
        or baseline.orchestration.model_refresh.inference_compile_dynamic
        != target.orchestration.model_refresh.inference_compile_dynamic
        or baseline.orchestration.model_refresh.inference_compile_mode
        != target.orchestration.model_refresh.inference_compile_mode
    ):
        _fail("only measured graph capacity 16-to-32 may extend inference authority")
    if not isinstance(references, list) or not references:
        _fail("changed graph capacity requires pinned H100 reports")
    if controlled:
        run_id = target.orchestration.run_id
        if not isinstance(run_id, str) or not run_id:
            _fail("controlled graph activation requires an explicit run identity")
        _, contents = _read_ref(
            root, gate["graph_cache_controlled_activation"], verified
        )
        receipt = _json(contents)
        # This is the current cache16 configuration, not the older baseline
        # before the measured search-allocation change. Every other field stays.
        control = replace(
            target,
            orchestration=replace(
                target.orchestration,
                model_refresh=replace(
                    target.orchestration.model_refresh,
                    inference=replace(after, cuda_graph_max_entries=16),
                ),
            ),
        )
        control_sha = canonical_config_sha256(control)
        live_reference = validate_controlled_activation_receipt(
            receipt,
            run_id=run_id,
            source_config_sha256=control_sha,
            target_config_sha256=canonical_config_sha256(target),
            baseline_profile=gate["baseline_profile"],
            graph_cache_reports=references,
        )
        _, live_contents = _read_ref(root, live_reference, verified)
        validate_live_workload_evidence(
            _json(live_contents),
            run_id=run_id,
            source_config_sha256=control_sha,
            actor_gpu_ids={gpu.gpu_id for gpu in target.orchestration.actor_gpus},
        )
    models: dict[str, tuple[str, str]] = {}
    for group in gate["groups"]:
        _, contents = _read_ref(root, group["model_manifest"], verified)
        model = _json(contents)
        model_identity = model["model_identity"]
        expected = (group["model_manifest"]["sha256"], model["checkpoint_sha256"])
        if model_identity in models and models[model_identity] != expected:
            _fail("frozen graph benchmark model references disagree")
        models[model_identity] = expected
    if len(references) != len(models):
        _fail("graph cache reports must cover every frozen search model once")
    seen = set()
    assessments = {}
    source_canonical = hashlib.sha256(
        json.dumps(baseline.as_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    for reference in references:
        _, contents = _read_ref(root, reference, verified)
        report = _json(contents)
        plan = report.get("plan")
        if not isinstance(plan, dict):
            _fail("graph cache report plan is missing")
        identity = plan.get("model_identity")
        if not isinstance(identity, str) or identity not in models or identity in seen:
            _fail("graph cache report frozen model is unexpected or duplicated")
        gpu_id = plan.get("actor_gpu_id")
        actor = next(
            (
                gpu
                for gpu in baseline.orchestration.actor_gpus
                if type(gpu_id) is int and gpu.gpu_id == gpu_id
            ),
            None,
        )
        if (
            actor is None
            or actor.actor_cohorts < 2
            or actor.actor_pipeline is None
            or actor.actor_pipeline.cuda_graphs is not True
            or plan.get("actor_configuration") != asdict(actor)
        ):
            _fail("graph cache report producer topology differs from baseline")
        refresh = baseline.orchestration.model_refresh
        for name, expected_value in (
            ("precision", baseline.train.precision),
            ("compile", baseline.train.compile),
            ("compile_dynamic", refresh.inference_compile_dynamic),
            ("compile_mode", refresh.inference_compile_mode),
        ):
            if (
                type(plan.get(name)) is not type(expected_value)
                or plan[name] != expected_value
            ):
                _fail(f"graph cache report {name} differs from baseline")
        manifest_sha, checkpoint_sha = models[identity]
        byte_cap = max(1, before.cuda_graph_max_bytes // (actor.actor_cohorts + 2))
        if controlled and byte_cap != GRAPH_CACHE_BYTES:
            _fail(
                "controlled graph activation requires the original eight-GiB byte cap"
            )
        validator = (
            validate_graph_cache_execution_report
            if controlled
            else validate_graph_cache_report
        )
        assessment = validator(
            report,
            source_config_sha256=gate["baseline_profile"]["sha256"],
            source_config_canonical_sha256=source_canonical,
            model_identity=identity,
            manifest_sha256=manifest_sha,
            checkpoint_sha256=checkpoint_sha,
            profile_inference=asdict(replace(before, cuda_graphs=True)),
            graph_cache_bytes=byte_cap,
            actor_gpu_id=actor.gpu_id,
        )
        if assessment["performance_policy"] == BOUNDED_NONINFERIORITY_POLICY:
            # A bounded execution regression may consume some existing timing
            # headroom, but may never waive the original teacher-production
            # floor. Gains are capped at one; they cannot repair weak evidence.
            execution_floor = _number(
                assessment["conservative_min_speedup"],
                "graph execution floor",
                positive=True,
            )
            if execution_floor > 1 or (
                full_target_rate_ratios[identity] * execution_floor < 1
            ):
                _fail("graph execution margin violates original full-target rate floor")
        seen.add(identity)
        assessments[identity] = assessment
    if controlled:
        validate_controlled_activation_assessments(assessments, full_target_rate_ratios)


def _fail(message: str) -> NoReturn:
    raise ValueError(f"search allocation gate: {message}")


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{name} must be a lowercase SHA256")
    assert isinstance(value, str)
    return value


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (positive and value <= 0)
    ):
        _fail(f"{name} must be a finite {'positive ' if positive else ''}number")
    assert isinstance(value, (int, float))
    return float(value)


def _safe_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        _fail("artifact path must be a relative path inside the run")
    assert isinstance(relative, str)
    parts = Path(relative).parts
    if Path(relative).is_absolute() or ".." in parts or not parts:
        _fail("artifact path escaped the run")
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            _fail("artifact path contains a symlink")
    if not current.is_file():
        _fail(f"artifact is missing: {relative}")
    if current.stat().st_size > _MAX_REPORT_BYTES:
        _fail("artifact exceeds the bounded report size")
    return current


def _signature(path: Path) -> tuple[int, ...]:
    value = path.stat()
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _unchanged(root: Path, files: dict[str, tuple[int, ...]]) -> bool:
    return all(
        _signature(_safe_path(root, path)) == value for path, value in files.items()
    )


def _read_ref(
    root: Path, reference: object, verified: dict[str, tuple[int, ...]] | None = None
) -> tuple[Path, bytes]:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        _fail("artifact reference requires exactly path and sha256")
    assert isinstance(reference, dict)
    expected = _sha(reference["sha256"], "artifact hash")
    path = _safe_path(root, reference["path"])
    before = _signature(path)
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected:
        _fail(f"artifact hash mismatch: {reference['path']}")
    if _signature(path) != before:
        _fail("artifact changed during validation")
    if verified is not None:
        verified[str(path.relative_to(root))] = before
    return path, contents


def _json(contents: bytes) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in rows:
            if name in result:
                _fail(f"duplicate JSON key: {name}")
            result[name] = value
        return result

    def invalid(value: str) -> None:
        _fail(f"nonfinite JSON value: {value}")

    try:
        value = json.loads(contents, object_pairs_hook=pairs, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("search allocation gate: invalid JSON") from exc
    if not isinstance(value, dict):
        _fail("artifact must be a JSON object")
    return value


def _validate_measured_report(report: dict[str, Any]) -> None:
    if report.get("status") != "passed" or not isinstance(report.get("plan"), dict):
        _fail("benchmark report did not complete")
    plan = report["plan"]
    plan_bytes = (
        json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    if hashlib.sha256(plan_bytes).hexdigest() != report.get("plan_sha256"):
        _fail("benchmark plan hash mismatch")
    if (
        plan.get("schema_version") != 2
        or plan.get("search_algorithm") != SEARCH_ALGORITHM_ID
    ):
        _fail("benchmark search contract is incompatible")
    results = report.get("results")
    if not isinstance(results, list) or not results:
        _fail("benchmark results are empty")
    measured = [
        row
        for row in results
        if isinstance(row, dict) and row.get("stage") == "measured"
    ]
    if not measured:
        _fail("benchmark has no measured repeats")
    for row in measured:
        counters = row.get("timing_setup_counters")
        inference = row.get("inference")
        if not isinstance(counters, dict) or not isinstance(inference, dict):
            _fail("measured timing counters are missing")
        if row.get("timing_admissible") is not True or any(
            type(counters.get(key)) is not int
            or counters[key] != 0
            or type(inference.get(key)) is not int
            or inference[key] != 0
            for key in TIMING_SETUP_COUNTERS
        ):
            _fail("measured timing contains setup, fallback or validation work")
        _number(row.get("seconds"), "measured seconds", positive=True)
        if type(row.get("repeat")) is not int or row["repeat"] < 0:
            _fail("measured repeat index is invalid")
        warm = [
            prior
            for prior in results
            if isinstance(prior, dict)
            and prior.get("stage") == "warmup"
            and prior.get("ring") == row.get("ring")
            and prior.get("dataset") == row.get("dataset")
            and prior.get("canonical_arm") == row.get("canonical_arm")
        ]
        if (
            len(warm) != 1
            or not row.get("positions")
            or warm[0].get("positions") != row["positions"]
        ):
            _fail("measured trace does not match its warmup")


def _position_ids(selection: Mapping[str, Any]) -> set[str]:
    rows = selection.get("positions")
    if not isinstance(rows, list) or not rows:
        _fail("frozen selection has no positions")
    identifiers = []
    for row in rows:
        value = row.get("id") if isinstance(row, dict) else None
        if not isinstance(value, str) or len(value.split("/")) != 3:
            _fail("frozen position identity is invalid")
        identifiers.append(value)
    if len(set(identifiers)) != len(identifiers):
        _fail("frozen selection duplicates positions")
    return set(identifiers)


def _validate_group_inputs(
    root: Path,
    group: dict[str, Any],
    baseline: ExperimentConfig,
    baseline_profile_sha256: str,
    verified: dict[str, tuple[int, ...]],
) -> tuple[dict[str, Any], str]:
    if set(group) != {
        "role",
        "rings",
        "report",
        "selection",
        "model_manifest",
        "excluded_selections",
    }:
        _fail("model group fields are invalid")
    _, contents = _read_ref(root, group["report"], verified)
    report = _json(contents)
    _validate_measured_report(report)
    plan = report["plan"]
    if plan.get("config_sha256") != baseline_profile_sha256:
        _fail("benchmark did not use the pinned baseline profile")
    if plan.get("runtime") != "profile" or not str(plan.get("device", "")).startswith(
        "cuda"
    ):
        _fail("production gate requires measured profile-runtime CUDA costs")
    if plan.get("precision") != baseline.train.precision or plan.get(
        "profile_inference"
    ) != asdict(baseline.orchestration.model_refresh.inference):
        _fail("benchmark inference configuration differs from the pinned profile")
    pairs = plan.get("pairs")
    if (
        not isinstance(pairs, list)
        or not pairs
        or pairs[0]
        != [baseline.selfplay.fast_simulations, baseline.selfplay.full_simulations]
    ):
        _fail("benchmark baseline search caps differ from the pinned profile")
    _, manifest_contents = _read_ref(root, group["model_manifest"], verified)
    manifest = _json(manifest_contents)
    checkpoint_sha = _sha(manifest.get("checkpoint_sha256"), "model checkpoint hash")
    identity = f"sha256-{checkpoint_sha}"
    if (
        manifest.get("model_identity") != identity
        or plan.get("model_identity") != identity
        or manifest.get("run_id") != baseline.orchestration.run_id
        or plan.get("manifest_sha256") != group["model_manifest"]["sha256"]
        or plan.get("checkpoint_sha256") != checkpoint_sha
    ):
        _fail("frozen model identity or hashes do not match the measured report")
    _, selected_contents = _read_ref(root, group["selection"], verified)
    selected = _json(selected_contents)
    if (
        selected.get("schema_version") != 1
        or selected.get("run_id") != baseline.orchestration.run_id
        or selected.get("generation_family") != manifest.get("generation_family")
        or selected.get("model_identity") != identity
        or selected.get("manifest_sha256") != group["model_manifest"]["sha256"]
        or selected.get("config_sha256") != baseline_profile_sha256
        or plan.get("selection_sha256") != group["selection"]["sha256"]
        or plan.get("positions_sha256") != selected.get("payload_sha256")
        or group["rings"] not in selected.get("rings", [])
    ):
        _fail("held-out selection does not match the measured model/profile")
    identifiers = _position_ids(selected)
    observed = {
        position["id"]
        for row in report["results"]
        for position in row.get("positions", [])
    }
    if observed != identifiers:
        _fail("measured positions do not match the frozen selection")
    exclusions = group["excluded_selections"]
    if not isinstance(exclusions, list) or not exclusions:
        _fail("independent holdout requires the preliminary frozen selections")
    prior_hashes = set()
    prior_games: set[str] = set()
    for reference in exclusions:
        _, previous_contents = _read_ref(root, reference, verified)
        previous = _json(previous_contents)
        if previous.get("run_id") != selected.get("run_id") or previous.get(
            "generation_family"
        ) != selected.get("generation_family"):
            _fail("preliminary selection namespace differs from holdout")
        prior_hashes.add(reference["sha256"])
        prior_games.update(value.rsplit("/", 1)[0] for value in _position_ids(previous))
    if prior_hashes != {
        row.get("selection_sha256") for row in selected.get("excluded_selections", [])
    }:
        _fail("holdout exclusions do not match the pinned preliminary selections")
    if prior_games & {value.rsplit("/", 1)[0] for value in identifiers}:
        _fail("held-out and preliminary selections share games")
    return report, group["report"]["sha256"]


def _analysis(
    report: dict[str, Any],
    digest: str,
    baseline: ExperimentConfig,
    target: ExperimentConfig,
    ring: int,
) -> dict[str, Any]:
    # Lazy import keeps ordinary profile validation free of analysis work.
    from .search_allocation_evidence import analyze_search_allocation_report

    settings = replace(target.selfplay, rings=ring).resolved_search_allocation()
    parameters = {
        "ring": ring,
        "candidate_fast_simulations": settings.fast_simulations,
        "full_probability": settings.full_probability,
        "fast_policy_weight": settings.fast_policy_weight,
        "first_visit_batch_size": settings.search_execution.first_visit_batch_size,
        "baseline_full_probability": baseline.selfplay.full_probability,
        "baseline_first_visit_batch_size": baseline.selfplay.search_execution.first_visit_batch_size,
        "swap_dead_zone": settings.variants.swap_dead_zone,
    }
    key = (digest, *parameters.values())
    if key not in _ANALYSES:
        _ANALYSES[key] = analyze_search_allocation_report(report, **parameters)
        while len(_ANALYSES) > 16:
            _ANALYSES.popitem(last=False)
    _ANALYSES.move_to_end(key)
    return _ANALYSES[key]


def _validate_policy_transition_gate(
    root: Path,
    gate: dict[str, Any],
    target: ExperimentConfig,
    verified: dict[str, tuple[int, ...]],
    *,
    _fresh: bool = False,
) -> None:
    """Continue an admitted execution plan through one explicit policy change.

    This is not a new optimization qualification: the original reports remain
    evidence about their original frozen workload. No new objective throughput
    or playing-strength assertion is admitted by this receipt.
    """
    from .pie_policy import validate_pie_policy_transition

    _, contents = _read_ref(root, gate["training_policy_transition"], verified)
    receipt = _json(contents)
    if (
        set(receipt)
        != {
            "format",
            "schema_version",
            "classification",
            "run_id",
            "source_profile",
            "source_gate",
            "source_config_sha256",
            "target_config_sha256",
            "measurement_scope",
            "new_objective_performance_qualified",
        }
        or receipt.get("format") != POLICY_TRANSITION_FORMAT
        or type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt.get("classification") != POLICY_TRANSITION_CLASS
        or receipt.get("run_id") != target.orchestration.run_id
        or receipt.get("target_config_sha256") != canonical_config_sha256(target)
        or receipt.get("measurement_scope") != POLICY_TRANSITION_SCOPE
        or receipt.get("new_objective_performance_qualified") is not False
    ):
        _fail("policy transition receipt identity or evidence scope is invalid")
    source_path, _ = _read_ref(root, receipt["source_profile"], verified)
    source = load_config(source_path)
    if receipt["source_config_sha256"] != canonical_config_sha256(source):
        _fail("policy transition source configuration hash differs")
    validate_pie_policy_transition(source, target)
    source_gate_path, source_contents = _read_ref(
        root, receipt["source_gate"], verified
    )
    if source_gate_path != allocation_gate_path(source):
        _fail("policy transition must pin the source configuration's original gate")
    source_gate = _json(source_contents)
    if (
        "training_policy_transition" in source_gate
        or "promotion_allocation_transition" in source_gate
        or "auxiliary_prediction_transition" in source_gate
    ):
        _fail("policy transition receipts cannot be chained")
    expected = dict(source_gate)
    expected["target_config_sha256"] = canonical_config_sha256(target)
    expected["training_policy_transition"] = gate["training_policy_transition"]
    if gate != expected:
        _fail("policy transition must preserve the complete original gate and reports")
    # Original quality, rate and separately controlled cache-capacity admission
    # are all revalidated under their exact source configuration and identities.
    validate_production_ring_allocations(source, _fresh=_fresh)
    inherited = _VERIFIED_GATES.get(str(source_gate_path))
    if inherited is None:
        _fail("policy transition source has no verified allocation evidence")
    for path, signature in inherited.items():
        if path in verified and verified[path] != signature:
            _fail("source evidence changed while inheriting its admission")
        verified[path] = signature


def _validate_promotion_transition_gate(
    root: Path,
    gate: dict[str, Any],
    target: ExperimentConfig,
    verified: dict[str, tuple[int, ...]],
) -> None:
    """Carry existing admission through exactly one promotion-only change.

    The source may already carry its one training-policy transition. It may
    never carry another promotion transition. Exact config transforms make
    the inheritance order finite: original -> pie training -> adaptive screen.
    All original artifact references and qualification limits remain intact.
    """
    from .pie_promotion import validate_pie_promotion_transition

    reference = gate["promotion_allocation_transition"]
    _, contents = _read_ref(root, reference, verified)
    receipt = _json(contents)
    if (
        set(receipt)
        != {
            "format",
            "schema_version",
            "classification",
            "run_id",
            "source_profile",
            "source_gate",
            "source_config_sha256",
            "target_config_sha256",
            "measurement_scope",
            "new_objective_performance_qualified",
        }
        or receipt.get("format") != PROMOTION_TRANSITION_FORMAT
        or type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt.get("classification") != PROMOTION_TRANSITION_CLASS
        or receipt.get("run_id") != target.orchestration.run_id
        or receipt.get("target_config_sha256") != canonical_config_sha256(target)
        or receipt.get("measurement_scope") != PROMOTION_TRANSITION_SCOPE
        or receipt.get("new_objective_performance_qualified") is not False
    ):
        _fail("promotion transition receipt identity or evidence scope is invalid")
    source_path, _ = _read_ref(root, receipt["source_profile"], verified)
    source = load_config(source_path)
    if receipt["source_config_sha256"] != canonical_config_sha256(source):
        _fail("promotion transition source configuration hash differs")
    validate_pie_promotion_transition(source, target)
    source_gate_path, source_contents = _read_ref(
        root, receipt["source_gate"], verified
    )
    if source_gate_path != allocation_gate_path(source):
        _fail("promotion transition must pin the source configuration's original gate")
    original = _json(source_contents)
    if (
        "promotion_allocation_transition" in original
        or "auxiliary_prediction_transition" in original
    ):
        _fail("promotion transition receipts cannot be chained")
    expected = dict(original)
    expected["target_config_sha256"] = canonical_config_sha256(target)
    expected["promotion_allocation_transition"] = reference
    if gate != expected:
        _fail("promotion transition must preserve the complete source gate and reports")
    # This new wrapper obtains fresh cryptographic checks of the complete
    # source closure, including any earlier training-policy inheritance.
    validate_production_ring_allocations(source, _fresh=True)
    inherited = _VERIFIED_GATES.get(str(source_gate_path))
    if inherited is None:
        _fail("promotion transition source has no verified allocation evidence")
    for path, signature in inherited.items():
        if path in verified and verified[path] != signature:
            _fail("source evidence changed while inheriting promotion admission")
        verified[path] = signature


def _validate_auxiliary_transition_gate(
    root: Path,
    gate: dict[str, Any],
    target: ExperimentConfig,
    verified: dict[str, tuple[int, ...]],
) -> None:
    """Retain the complete prior gate for a single additive training extension.

    New heads do not run in search leaves or change scalar utility. The receipt
    grants no new throughput, training-efficiency, or strength qualification.
    Original -> policy -> promotion -> auxiliary is a finite inheritance order.
    """
    from .auxiliary_policy import validate_auxiliary_prediction_transition

    reference = gate["auxiliary_prediction_transition"]
    _, contents = _read_ref(root, reference, verified)
    receipt = _json(contents)
    if (
        set(receipt)
        != {
            "format",
            "schema_version",
            "classification",
            "run_id",
            "source_profile",
            "source_gate",
            "source_config_sha256",
            "target_config_sha256",
            "measurement_scope",
            "new_objective_performance_qualified",
        }
        or receipt.get("format") != AUXILIARY_TRANSITION_FORMAT
        or type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt.get("classification") != AUXILIARY_TRANSITION_CLASS
        or receipt.get("run_id") != target.orchestration.run_id
        or receipt.get("target_config_sha256") != canonical_config_sha256(target)
        or receipt.get("measurement_scope") != AUXILIARY_TRANSITION_SCOPE
        or receipt.get("new_objective_performance_qualified") is not False
    ):
        _fail("auxiliary transition receipt identity or evidence scope is invalid")
    source_path, _ = _read_ref(root, receipt["source_profile"], verified)
    source = load_config(source_path)
    if receipt["source_config_sha256"] != canonical_config_sha256(source):
        _fail("auxiliary transition source configuration hash differs")
    validate_auxiliary_prediction_transition(source, target)
    source_gate_path, source_contents = _read_ref(
        root, receipt["source_gate"], verified
    )
    if source_gate_path != allocation_gate_path(source):
        _fail("auxiliary transition must pin the source configuration's original gate")
    original = _json(source_contents)
    if "auxiliary_prediction_transition" in original:
        _fail("auxiliary transition receipts cannot be chained")
    expected = dict(original)
    expected["target_config_sha256"] = canonical_config_sha256(target)
    expected["auxiliary_prediction_transition"] = reference
    if gate != expected:
        _fail("auxiliary transition must preserve the complete source gate and reports")
    validate_production_ring_allocations(source, _fresh=True)
    inherited = _VERIFIED_GATES.get(str(source_gate_path))
    if inherited is None:
        _fail("auxiliary transition source has no verified allocation evidence")
    for path, signature in inherited.items():
        if path in verified and verified[path] != signature:
            _fail("source evidence changed while inheriting auxiliary admission")
        verified[path] = signature


def _cache_verified_gate(
    root: Path, key: str, verified: dict[str, tuple[int, ...]]
) -> None:
    if not _unchanged(root, verified):
        _fail("evidence changed before validation completed")
    _VERIFIED_GATES[key] = verified
    while len(_VERIFIED_GATES) > 4:
        _VERIFIED_GATES.popitem(last=False)


def validate_production_ring_allocations(
    config: ExperimentConfig, *, _fresh: bool = False
) -> None:
    allocations = config.selfplay.ring_search_allocations
    for allocation in allocations:
        if not 0 < allocation.fast_policy_weight <= 0.25:
            _fail("production fast_policy_weight must be positive and at most 0.25")
    required = {
        row.rings
        for row in allocations
        if row.full_probability < FULL_PROBABILITY_FLOOR
    }
    if not required:
        return
    root = Path(config.orchestration.directories.root).expanduser().resolve()
    path = allocation_gate_path(config)
    try:
        path = _safe_path(root, str(path.relative_to(root)))
        cache_key = str(path)
        cached = _VERIFIED_GATES.get(cache_key)
        fresh = (
            _fresh
            or config.arena.allocation_policy == "adaptive_pie"
            or config.model.auxiliary_predictions
        )
        if cached is not None and not fresh and _unchanged(root, cached):
            _VERIFIED_GATES.move_to_end(cache_key)
            return
        _VERIFIED_GATES.pop(cache_key, None)
        before = _signature(path)
        gate = _json(path.read_bytes())
        if _signature(path) != before:
            _fail("gate changed during validation")
        verified = {str(path.relative_to(root)): before}
        if (
            set(gate)
            - {
                "graph_cache_reports",
                "graph_cache_controlled_activation",
                "training_policy_transition",
                "promotion_allocation_transition",
                "auxiliary_prediction_transition",
            }
            != {
                "format",
                "schema_version",
                "run_id",
                "target_config_sha256",
                "baseline_profile",
                "groups",
            }
            or gate.get("format") != GATE_FORMAT
            or type(gate.get("schema_version")) is not int
            or gate["schema_version"] != 1
            or gate.get("run_id") != config.orchestration.run_id
            or gate.get("target_config_sha256") != canonical_config_sha256(config)
        ):
            _fail("gate identity/configuration fields are invalid")
        if "auxiliary_prediction_transition" in gate:
            _validate_auxiliary_transition_gate(root, gate, config, verified)
            _cache_verified_gate(root, cache_key, verified)
            return
        if "promotion_allocation_transition" in gate:
            _validate_promotion_transition_gate(root, gate, config, verified)
            _cache_verified_gate(root, cache_key, verified)
            return
        if "training_policy_transition" in gate:
            _validate_policy_transition_gate(root, gate, config, verified, _fresh=fresh)
            _cache_verified_gate(root, cache_key, verified)
            return
        baseline_path, _ = _read_ref(root, gate["baseline_profile"], verified)
        baseline = load_config(baseline_path)
        if baseline.orchestration.run_id != config.orchestration.run_id:
            _fail("baseline run differs from target")
        if (
            baseline.selfplay.ring_search_allocations
            or baseline.selfplay.full_probability < FULL_PROBABILITY_FLOOR
        ):
            _fail("baseline must retain the original full-search probability floor")
        before, after = asdict(baseline.selfplay), asdict(config.selfplay)
        for payload in (before, after):
            del payload["ring_search_allocations"]
            del payload["search_execution"]["first_visit_batch_size"]
        if before != after:
            _fail(
                "global search caps, allocation, candidates, PDA or other search settings changed"
            )
        for name in ("game", "model", "train", "loss", "optimizer", "learner"):
            if getattr(baseline, name) != getattr(config, name):
                _fail(f"baseline {name} differs from target")
        if (
            baseline.orchestration.model_refresh.inference
            != (config.orchestration.model_refresh.inference)
            and "graph_cache_reports" not in gate
        ):
            _fail("target inference settings were not measured by the pinned profile")
        groups = gate["groups"]
        if not isinstance(groups, list) or any(
            not isinstance(group, dict) for group in groups
        ):
            _fail("model groups must be a list")
        actual = [(group.get("role"), group.get("rings")) for group in groups]
        expected = {(role, ring) for ring in required for role in ("actor", "champion")}
        if (
            len(actual) != len(expected)
            or set(actual) != expected
            or any(type(group.get("rings")) is not int for group in groups)
        ):
            _fail(
                "each affected ring requires exactly one frozen actor and champion group"
            )
        full_target_rate_ratios: dict[str, float] = {}
        for group in groups:
            report, digest = _validate_group_inputs(
                root, group, baseline, gate["baseline_profile"]["sha256"], verified
            )
            analysis = _analysis(report, digest, baseline, config, group["rings"])
            if analysis.get("full_budget_preserved") is not True:
                _fail("measured allocation changed full-search budgets")
            for dataset in ("balanced", "swaps"):
                quality = analysis.get(dataset)
                if (
                    not isinstance(quality, dict)
                    or _number(quality.get("coverage"), f"{dataset} coverage") != 1
                ):
                    _fail(f"{group['role']} {dataset} has unassessed actions")
                if (
                    _number(
                        quality.get("upper95"), f"{dataset} incremental regret upper95"
                    )
                    > MAX_INCREMENTAL_REGRET_UPPER95
                ):
                    _fail(
                        f"{group['role']} {dataset} incremental regret upper95 exceeds 0.02"
                    )
                if (
                    type(quality.get("new_swap_failures")) is not int
                    or quality["new_swap_failures"] != 0
                ):
                    _fail(f"{group['role']} {dataset} has new swap failures")
            if (
                analysis.get("new_swap_failures") != 0
                or type(analysis.get("new_swap_failures")) is not int
            ):
                _fail(f"{group['role']} has new swap failures")
            cost = analysis.get("cost")
            if (
                not isinstance(cost, dict)
                or _number(
                    cost.get("conservative_full_target_rate_ratio"),
                    "conservative full-target rate",
                    positive=True,
                )
                < 1
            ):
                _fail(
                    f"{group['role']} conservative full-target rate falls below baseline"
                )
            identity = report["plan"]["model_identity"]
            ratio = float(cost["conservative_full_target_rate_ratio"])
            full_target_rate_ratios[identity] = min(
                full_target_rate_ratios.get(identity, ratio), ratio
            )
        _validate_graph_capacity_reports(
            root, gate, baseline, config, verified, full_target_rate_ratios
        )
        _cache_verified_gate(root, cache_key, verified)
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError(
            f"search allocation gate is missing or malformed: {exc}"
        ) from exc
