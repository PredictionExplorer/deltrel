from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest
import yaml

from scripts.validate_continuous_profile import validate_continuous_config
from deltreltrain.config import load_config
from deltreltrain.contracts import SEARCH_ALGORITHM_ID
from deltreltrain import search_allocation_gate as gate
from deltreltrain.selfplay import RingSearchAllocation


GOOD = {
    "full_budget_preserved": True,
    "new_swap_failures": 0,
    "balanced": {
        "coverage": 1.0,
        "upper95": 0.01,
        "positions": 18,
        "games": 18,
        "new_swap_failures": 0,
    },
    "swaps": {
        "coverage": 1.0,
        "upper95": 0.01,
        "positions": 4,
        "games": 4,
        "new_swap_failures": 0,
    },
    "cost": {"conservative_full_target_rate_ratio": 1.01},
}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def reference(root, path):
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def update_plan(report):
    report["plan_sha256"] = hashlib.sha256(
        (json.dumps(report["plan"], sort_keys=True, indent=2) + "\n").encode()
    ).hexdigest()


def fixture(
    tmp_path,
    monkeypatch,
    *,
    stub_analysis=True,
    root=None,
    run_id="gate-run",
    family="gate-family",
):
    base = load_config(
        Path(__file__).parents[1] / "configs/h100-8gpu-largest-board-priority.yaml"
    )
    root = root or tmp_path / "run"
    root.mkdir(exist_ok=True)
    base = replace(
        base,
        orchestration=replace(
            base.orchestration,
            run_id=run_id,
            directories=replace(base.orchestration.directories, root=str(root)),
        ),
    )
    target = replace(
        base,
        selfplay=replace(
            base.selfplay,
            ring_search_allocations=(RingSearchAllocation(10, 8, 0.125, 0.053),),
        ),
    )
    baseline_path = root / "profile-baseline.yaml"
    baseline_path.write_text(yaml.safe_dump(base.as_dict()))
    envelope = {
        "format": gate.GATE_FORMAT,
        "schema_version": 1,
        "run_id": run_id,
        "target_config_sha256": gate.canonical_config_sha256(target),
        "baseline_profile": reference(root, baseline_path),
        "groups": [],
    }
    for role, letter in (("actor", "a"), ("champion", "b")):
        parent = root / "status/search-allocation-gates/inputs" / role
        parent.mkdir(parents=True)
        manifest = {
            "format": "deltreltrain.model-manifest",
            "schema_version": 3,
            "weights": "ema",
            "model_identity": "sha256-" + letter * 64,
            "checkpoint_sha256": letter * 64,
            "run_id": run_id,
            "generation_family": family,
        }
        manifest_path = parent / "model.json"
        write_json(manifest_path, manifest)
        prior = {
            "schema_version": 1,
            "run_id": run_id,
            "generation_family": family,
            "positions": [{"id": f"{run_id}/preliminary-game/1"}],
        }
        prior_path = parent / "preliminary.json"
        write_json(prior_path, prior)
        positions = [{"id": f"{run_id}/holdout-{role}-{index}/1"} for index in range(2)]
        selected = {
            "schema_version": 1,
            "run_id": run_id,
            "generation_family": family,
            "model_identity": manifest["model_identity"],
            "manifest_sha256": reference(root, manifest_path)["sha256"],
            "config_sha256": envelope["baseline_profile"]["sha256"],
            "payload_sha256": "c" * 64,
            "rings": [10],
            "positions": positions,
            "excluded_selections": [
                {"selection_sha256": reference(root, prior_path)["sha256"]}
            ],
        }
        selected_path = parent / "selection.json"
        write_json(selected_path, selected)
        zeros = dict.fromkeys(gate.TIMING_SETUP_COUNTERS, 0)
        plan = {
            "schema_version": 2,
            "search_algorithm": SEARCH_ALGORITHM_ID,
            "runtime": "profile",
            "device": "cuda:0",
            "precision": base.train.precision,
            "profile_inference": asdict(base.orchestration.model_refresh.inference),
            "config_sha256": envelope["baseline_profile"]["sha256"],
            "model_identity": manifest["model_identity"],
            "manifest_sha256": reference(root, manifest_path)["sha256"],
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "selection_sha256": reference(root, selected_path)["sha256"],
            "positions_sha256": selected["payload_sha256"],
            "pairs": [
                [base.selfplay.fast_simulations, base.selfplay.full_simulations],
                [8, base.selfplay.full_simulations],
            ],
        }
        report = {
            "status": "passed",
            "plan": plan,
            "results": [
                {
                    "ring": 10,
                    "dataset": "balanced",
                    "stage": "warmup",
                    "canonical_arm": "a",
                    "positions": positions,
                },
                {
                    "ring": 10,
                    "dataset": "balanced",
                    "stage": "measured",
                    "canonical_arm": "a",
                    "positions": positions,
                    "repeat": 0,
                    "seconds": 1.0,
                    "inference": zeros.copy(),
                    "timing_setup_counters": zeros.copy(),
                    "timing_admissible": True,
                },
            ],
        }
        update_plan(report)
        report_path = parent / "report.json"
        write_json(report_path, report)
        envelope["groups"].append(
            {
                "role": role,
                "rings": 10,
                "report": reference(root, report_path),
                "selection": reference(root, selected_path),
                "model_manifest": reference(root, manifest_path),
                "excluded_selections": [reference(root, prior_path)],
            }
        )
    write_json(gate.allocation_gate_path(target), envelope)
    if stub_analysis:
        monkeypatch.setattr(gate, "_analysis", lambda *_args: deepcopy(GOOD))
    return base, target, root, envelope


def test_current_profile_needs_no_artifact_and_retains_canonical_hash(
    tmp_path, monkeypatch
):
    base, _, _, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        gate,
        "_safe_path",
        lambda *_args: pytest.fail("ordinary profile must not read evidence"),
    )
    validate_continuous_config(base)
    assert "ring_search_allocations" not in base.as_dict()["selfplay"]


def test_valid_hashed_gate_and_shared_production_validator(tmp_path, monkeypatch):
    _, target, _, _ = fixture(tmp_path, monkeypatch)
    validate_continuous_config(target)


@pytest.mark.parametrize("value", [0, 0.250001, 1])
def test_production_override_weight_floor_applies_even_without_probability_waiver(
    tmp_path, monkeypatch, value
):
    base, _, _, _ = fixture(tmp_path, monkeypatch)
    target = replace(
        base,
        selfplay=replace(
            base.selfplay,
            ring_search_allocations=(RingSearchAllocation(10, 8, 0.4, value),),
        ),
    )
    with pytest.raises(ValueError, match="fast_policy_weight"):
        validate_continuous_config(target)


def test_gate_never_waives_the_raw_global_floor(tmp_path, monkeypatch):
    _, target, _, _ = fixture(tmp_path, monkeypatch)
    target = replace(
        target,
        selfplay=replace(
            target.selfplay, full_probability=0.125, fast_probability=0.875
        ),
    )
    monkeypatch.setattr(
        gate,
        "_safe_path",
        lambda *_args: pytest.fail("global floor must be checked first"),
    )
    with pytest.raises(ValueError, match="large-board search"):
        validate_continuous_config(target)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "config",
        "run",
        "actor_only",
        "duplicate_actor",
        "unknown",
        "bool_schema",
        "symlink",
        "escaped_path",
        "report_hash",
    ],
)
def test_gate_identity_scope_and_file_guards(tmp_path, monkeypatch, fault):
    _, target, root, envelope = fixture(tmp_path, monkeypatch)
    path = gate.allocation_gate_path(target)
    if fault == "missing":
        path.unlink()
    else:
        if fault == "config":
            envelope["target_config_sha256"] = "f" * 64
        elif fault == "run":
            envelope["run_id"] = "other"
        elif fault == "actor_only":
            envelope["groups"].pop()
        elif fault == "duplicate_actor":
            envelope["groups"][1]["role"] = "actor"
        elif fault == "unknown":
            envelope["passed"] = True
        elif fault == "bool_schema":
            envelope["schema_version"] = True
        elif fault == "escaped_path":
            envelope["groups"][0]["report"]["path"] = "../elsewhere.json"
        elif fault == "report_hash":
            envelope["groups"][0]["report"]["sha256"] = "f" * 64
        elif fault == "symlink":
            original = root / envelope["groups"][0]["report"]["path"]
            link = original.with_name("symlink.json")
            link.symlink_to(original)
            envelope["groups"][0]["report"]["path"] = str(link.relative_to(root))
        write_json(path, envelope)
    with pytest.raises(ValueError, match="search allocation gate"):
        validate_continuous_config(target)


@pytest.mark.parametrize(
    "fault",
    [
        "counter",
        "missing_counter",
        "bool_counter",
        "cold",
        "trace",
        "plan_hash",
        "model",
        "manifest",
        "profile",
        "wrong_caps",
        "nonfinite_time",
    ],
)
def test_measured_report_cannot_hide_setup_or_change_inputs(
    tmp_path, monkeypatch, fault
):
    _, target, root, envelope = fixture(tmp_path, monkeypatch)
    group = envelope["groups"][0]
    path = root / group["report"]["path"]
    report = json.loads(path.read_text())
    row = report["results"][1]
    if fault == "counter":
        row["inference"]["graph_captures"] = 1
    elif fault == "missing_counter":
        del row["timing_setup_counters"]["graph_captures"]
    elif fault == "bool_counter":
        row["timing_setup_counters"]["graph_captures"] = False
    elif fault == "cold":
        row["timing_admissible"] = False
    elif fault == "trace":
        row["positions"] = [{"id": "gate-run/changed/1"}]
    elif fault == "plan_hash":
        report["plan_sha256"] = "f" * 64
    elif fault == "model":
        report["plan"]["model_identity"] = "sha256-" + "f" * 64
    elif fault == "manifest":
        report["plan"]["manifest_sha256"] = "f" * 64
    elif fault == "profile":
        report["plan"]["config_sha256"] = "f" * 64
    elif fault == "wrong_caps":
        report["plan"]["pairs"][0][1] = 160
    elif fault == "nonfinite_time":
        row["seconds"] = float("nan")
    if fault not in ("plan_hash", "nonfinite_time"):
        update_plan(report)
    write_json(path, report)
    group["report"] = reference(root, path)
    write_json(gate.allocation_gate_path(target), envelope)
    with pytest.raises(ValueError, match="search allocation gate"):
        validate_continuous_config(target)


def test_holdout_is_disjoint_by_game_not_only_position(tmp_path, monkeypatch):
    _, target, root, envelope = fixture(tmp_path, monkeypatch)
    group = envelope["groups"][0]
    path = root / group["excluded_selections"][0]["path"]
    previous = json.loads(path.read_text())
    previous["positions"][0]["id"] = "gate-run/holdout-actor-0/9"
    write_json(path, previous)
    group["excluded_selections"] = [reference(root, path)]
    selection_path = root / group["selection"]["path"]
    selected = json.loads(selection_path.read_text())
    selected["excluded_selections"] = [
        {"selection_sha256": reference(root, path)["sha256"]}
    ]
    write_json(selection_path, selected)
    group["selection"] = reference(root, selection_path)
    report_path = root / group["report"]["path"]
    report = json.loads(report_path.read_text())
    report["plan"]["selection_sha256"] = group["selection"]["sha256"]
    update_plan(report)
    write_json(report_path, report)
    group["report"] = reference(root, report_path)
    write_json(gate.allocation_gate_path(target), envelope)
    with pytest.raises(ValueError, match="share games"):
        validate_continuous_config(target)


@pytest.mark.parametrize(
    "dataset,key,value",
    [
        ("balanced", "upper95", 0.020001),
        ("swaps", "upper95", 0.020001),
        ("balanced", "coverage", 0.99),
        ("swaps", "coverage", 0.99),
        ("balanced", "upper95", None),
        ("swaps", "upper95", float("nan")),
        ("swaps", "new_swap_failures", 1),
        ("swaps", "new_swap_failures", False),
        ("cost", "conservative_full_target_rate_ratio", 0.9999),
        ("cost", "conservative_full_target_rate_ratio", float("inf")),
    ],
)
def test_gate_recomputes_numeric_criteria_for_each_model_group(
    tmp_path, monkeypatch, dataset, key, value
):
    _, target, _, _ = fixture(tmp_path, monkeypatch)
    responses = [deepcopy(GOOD), deepcopy(GOOD)]
    responses[1][dataset][key] = value
    monkeypatch.setattr(gate, "_analysis", lambda *_args: responses.pop(0))
    with pytest.raises(ValueError, match="search allocation gate"):
        validate_continuous_config(target)


def test_analysis_cache_reuses_only_verified_report_hash_and_parameters(
    tmp_path, monkeypatch
):
    _, target, root, envelope = fixture(tmp_path, monkeypatch, stub_analysis=False)
    fake = ModuleType("deltreltrain.search_allocation_evidence")
    calls = []

    def analyze(report, **parameters):
        calls.append(parameters)
        return deepcopy(GOOD)

    fake.analyze_search_allocation_report = analyze
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    gate._ANALYSES.clear()
    validate_continuous_config(target)
    validate_continuous_config(target)
    assert len(calls) == 2
    with monkeypatch.context() as cached:
        cached.setattr(
            gate,
            "_json",
            lambda *_args: pytest.fail("unchanged gate must not reparse reports"),
        )
        validate_continuous_config(target)
    assert calls[0]["full_probability"] == 0.125
    report_path = root / envelope["groups"][0]["report"]["path"]
    report_path.write_text(report_path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_continuous_config(target)
    assert len(calls) == 2
    envelope["groups"][0]["report"] = reference(root, report_path)
    write_json(gate.allocation_gate_path(target), envelope)
    validate_continuous_config(target)
    assert len(calls) == 3
    gate._ANALYSES.clear()


def install_real_reports(root, target, envelope, *, loss=0.01, width8_full_scale=1):
    from test_search_allocation_evidence import valid_report

    for group in envelope["groups"]:
        report_path = root / group["report"]["path"]
        bindings = json.loads(report_path.read_text())["plan"]
        report = valid_report(loss=loss, width8_full_scale=width8_full_scale)
        report["plan"].update(bindings)
        warmed = set()
        for row in list(report["results"]):
            for position in row["positions"]:
                position["id"] = position["id"].replace(
                    "run/", target.orchestration.run_id + "/", 1
                )
            if row["stage"] != "measured":
                continue
            key = row["dataset"], row["canonical_arm"]
            if key not in warmed:
                warmed.add(key)
                report["results"].append({**deepcopy(row), "stage": "warmup"})
        selected_path = root / group["selection"]["path"]
        selected = json.loads(selected_path.read_text())
        selected["positions"] = [
            {"id": value}
            for value in sorted(
                {
                    position["id"]
                    for row in report["results"]
                    for position in row["positions"]
                }
            )
        ]
        write_json(selected_path, selected)
        group["selection"] = reference(root, selected_path)
        report["plan"]["selection_sha256"] = group["selection"]["sha256"]
        update_plan(report)
        write_json(report_path, report)
        group["report"] = reference(root, report_path)
    write_json(gate.allocation_gate_path(target), envelope)


@pytest.mark.parametrize(
    "loss,scale,width,passes",
    [(0.01, 1, 1, True), (0.03, 1, 1, False), (0.01, 3, 8, False)],
)
def test_real_raw_reports_are_analyzed_before_production_admission(
    tmp_path, monkeypatch, loss, scale, width, passes
):
    _, target, root, envelope = fixture(tmp_path, monkeypatch, stub_analysis=False)
    target = replace(
        target,
        selfplay=replace(
            target.selfplay,
            search_execution=replace(
                target.selfplay.search_execution, first_visit_batch_size=width
            ),
        ),
    )
    envelope["target_config_sha256"] = gate.canonical_config_sha256(target)
    install_real_reports(root, target, envelope, loss=loss, width8_full_scale=scale)
    gate._ANALYSES.clear()
    if passes:
        validate_continuous_config(target)
        assert len(gate._ANALYSES) == 2
    else:
        with pytest.raises(ValueError, match="incremental regret|full-target rate"):
            validate_continuous_config(target)
