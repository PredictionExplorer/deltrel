from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from startrain import search_allocation_gate as gate
from startrain.graph_cache_controlled_activation import (
    ACTIVATION_SCOPE,
    CONTROLLED_ACTIVATION_FORMAT,
    LIVE_WORKLOAD_FORMAT,
)
from startrain.graph_cache_evidence import BOUNDED_NONINFERIORITY_POLICY
from test_graph_cache_capacity_benchmark import set_paired_ratio
from test_graph_capacity_admission import graph_admission_fixture
from test_search_allocation_gate import GOOD, reference, write_json


USER_REQUEST = "Can we enable the third improvement on the server if it will improve the performance. Deploy it gracefully."


def controlled_admission_fixture(
    tmp_path, monkeypatch, *, original_rate=1.04, **kwargs
):
    """Fully bound failed reports plus a separately authorized controlled trial."""
    base, target, root, envelope = graph_admission_fixture(
        tmp_path,
        monkeypatch,
        report_options={
            "repeats": 4,
            "cycles": 2,
            "performance_policy": BOUNDED_NONINFERIORITY_POLICY,
        },
        **kwargs,
    )
    # Match the live source situation: exact clinches are already enabled.
    target = replace(target, arena=replace(target.arena, exact_clinch_termination=True))
    control = replace(
        target,
        orchestration=replace(
            target.orchestration,
            model_refresh=replace(
                target.orchestration.model_refresh,
                inference=replace(
                    target.orchestration.model_refresh.inference,
                    cuda_graph_max_entries=16,
                ),
            ),
        ),
    )
    quality = deepcopy(GOOD)
    quality["cost"]["conservative_full_target_rate_ratio"] = original_rate
    monkeypatch.setattr(gate, "_analysis", lambda *_: deepcopy(quality))
    for index, ref in enumerate(envelope["graph_cache_reports"]):
        path = root / ref["path"]
        report = json.loads(path.read_text())
        set_paired_ratio(report, "ring10-control", 0, 0.98)
        for row in report["records"]:
            row["cold_cycle"] = deepcopy(row["cycles"][0])
        write_json(path, report)
        envelope["graph_cache_reports"][index] = reference(root, path)
    model = json.loads(
        (root / envelope["groups"][0]["model_manifest"]["path"]).read_text()
    )
    live_path = root / "status/graph-cache-live-workload.json"
    write_json(
        live_path,
        {
            "format": LIVE_WORKLOAD_FORMAT,
            "schema_version": 1,
            "run_id": target.orchestration.run_id,
            "source_config_sha256": gate.canonical_config_sha256(control),
            "observed_from_ns": 1_000_000_000,
            "observed_until_ns": 601_000_000_000,
            "gpu_id": 2,
            "process_pid": 1234,
            "model_identity": model["model_identity"],
            "rings": [8, 10],
            "counter_scope": "gpu_process",
            "graph_captures_delta": 329,
            "graph_evictions_delta": 329,
            "useful_neural_rows_delta": 500_000,
        },
    )
    receipt_path = root / "status/graph-cache-controlled-activation.json"
    write_json(
        receipt_path,
        {
            "format": CONTROLLED_ACTIVATION_FORMAT,
            "schema_version": 1,
            "run_id": target.orchestration.run_id,
            "source_config_sha256": gate.canonical_config_sha256(control),
            "target_config_sha256": gate.canonical_config_sha256(target),
            "baseline_profile": envelope["baseline_profile"],
            "graph_cache_reports": envelope["graph_cache_reports"],
            "authorization": {"kind": "explicit-user-request", "request": USER_REQUEST},
            "rationale": "Observed mixed-board churn and exact execution evidence support expected benefit from a controlled activation; original benchmark failures remain unchanged.",
            "prior_policy_failures_acknowledged": True,
            "activation_scope": ACTIVATION_SCOPE,
            "live_workload_evidence": reference(root, live_path),
        },
    )
    envelope["target_config_sha256"] = gate.canonical_config_sha256(target)
    envelope["graph_cache_controlled_activation"] = reference(root, receipt_path)
    write_json(gate.allocation_gate_path(target), envelope)
    return base, target, root, envelope


def test_explicit_controlled_activation_preserves_failed_report_bytes(
    tmp_path, monkeypatch
):
    _, target, root, envelope = controlled_admission_fixture(tmp_path, monkeypatch)
    reports_before = {
        (root / ref["path"]): (root / ref["path"]).read_bytes()
        for ref in envelope["graph_cache_reports"]
    }
    gate.validate_production_ring_allocations(target)
    assert all(
        path.read_bytes() == contents for path, contents in reports_before.items()
    )
    for path in reports_before:
        assert (
            json.loads(path.read_text())["assessment"][
                "eligible_for_controlled_activation"
            ]
            is False
        )
    # Removing only separate trial authority restores the original failed gate.
    envelope.pop("graph_cache_controlled_activation")
    write_json(gate.allocation_gate_path(target), envelope)
    with pytest.raises(ValueError, match="nonregression"):
        gate.validate_production_ring_allocations(target)


@pytest.mark.parametrize("which", ["receipt", "live", "report"])
def test_controlled_authority_dependencies_invalidate_a_cached_pass(
    tmp_path, monkeypatch, which
):
    _, target, root, envelope = controlled_admission_fixture(tmp_path, monkeypatch)
    gate.validate_production_ring_allocations(target)
    receipt_path = root / envelope["graph_cache_controlled_activation"]["path"]
    receipt = json.loads(receipt_path.read_text())
    path = {
        "receipt": receipt_path,
        "live": root / receipt["live_workload_evidence"]["path"],
        "report": root / envelope["graph_cache_reports"][0]["path"],
    }[which]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        gate.validate_production_ring_allocations(target)


@pytest.mark.parametrize(
    "change",
    [
        "request",
        "acknowledgment",
        "source",
        "target",
        "baseline",
        "reports",
        "scope",
        "expiry",
        "missing_live",
    ],
)
def test_receipt_cannot_change_or_omit_its_authority(tmp_path, monkeypatch, change):
    base, target, root, envelope = controlled_admission_fixture(tmp_path, monkeypatch)
    path = root / envelope["graph_cache_controlled_activation"]["path"]
    receipt = json.loads(path.read_text())
    if change == "request":
        receipt["authorization"]["request"] = ""
    elif change == "acknowledgment":
        receipt["prior_policy_failures_acknowledged"] = False
    elif change == "source":
        receipt["source_config_sha256"] = gate.canonical_config_sha256(base)
    elif change == "target":
        receipt["target_config_sha256"] = "f" * 64
    elif change == "baseline":
        receipt["baseline_profile"]["sha256"] = "f" * 64
    elif change == "reports":
        receipt["graph_cache_reports"].reverse()
    elif change == "scope":
        receipt["activation_scope"] = "permanent-proven-elo-gain"
    elif change == "expiry":
        receipt["expires_at_ns"] = 1
    elif change == "missing_live":
        receipt.pop("live_workload_evidence")
    write_json(path, receipt)
    envelope["graph_cache_controlled_activation"] = reference(root, path)
    write_json(gate.allocation_gate_path(target), envelope)
    with pytest.raises(ValueError, match="controlled activation"):
        gate.validate_production_ring_allocations(target)


@pytest.mark.parametrize(
    "change",
    [
        "timestamps",
        "no_churn",
        "single_board",
        "unbenchmarked_pair",
        "three_boards",
        "counter_scope",
        "source",
        "pid",
    ],
)
def test_live_evidence_is_bound_and_does_not_invent_model_attribution(
    tmp_path, monkeypatch, change
):
    _, target, root, envelope = controlled_admission_fixture(tmp_path, monkeypatch)
    receipt_path = root / envelope["graph_cache_controlled_activation"]["path"]
    receipt = json.loads(receipt_path.read_text())
    live_path = root / receipt["live_workload_evidence"]["path"]
    live = json.loads(live_path.read_text())
    if change == "timestamps":
        live["observed_until_ns"] = live["observed_from_ns"]
    elif change == "no_churn":
        live["graph_evictions_delta"] = 0
    elif change == "single_board":
        live["rings"] = [10]
    elif change == "unbenchmarked_pair":
        live["rings"] = [6, 8]
    elif change == "three_boards":
        live["rings"] = [6, 8, 10]
    elif change == "counter_scope":
        live["counter_scope"] = "resident_model_only"
    elif change == "source":
        live["source_config_sha256"] = "e" * 64
    elif change == "pid":
        live["process_pid"] = True
    write_json(live_path, live)
    receipt["live_workload_evidence"] = reference(root, live_path)
    write_json(receipt_path, receipt)
    envelope["graph_cache_controlled_activation"] = reference(root, receipt_path)
    write_json(gate.allocation_gate_path(target), envelope)
    with pytest.raises(ValueError, match="controlled activation"):
        gate.validate_production_ring_allocations(target)


def test_controlled_activation_still_preserves_original_teacher_floor(
    tmp_path, monkeypatch
):
    _, target, _, _ = controlled_admission_fixture(
        tmp_path, monkeypatch, original_rate=1.01
    )
    with pytest.raises(ValueError, match="original full-target rate floor"):
        gate.validate_production_ring_allocations(target)


@pytest.mark.parametrize(
    "change", ["parity", "two_board_gain", "bytes", "other_config"]
)
def test_controlled_activation_is_not_a_general_gate_bypass(
    tmp_path, monkeypatch, change
):
    _, target, root, envelope = controlled_admission_fixture(tmp_path, monkeypatch)
    path = root / envelope["graph_cache_reports"][0]["path"]
    report = json.loads(path.read_text())
    if change == "parity":
        report["records"][0]["cycles"][0]["output_digests"][0] = "0" * 64
    elif change == "two_board_gain":
        for repeat in range(4):
            set_paired_ratio(report, "mixed-8-10", repeat, 1.01)
    elif change == "bytes":
        report["records"][0]["graph_cache_bytes"] = 4 * 1024**3
    elif change == "other_config":
        target = replace(target, arena=replace(target.arena, simulations=1024))
    write_json(path, report)
    envelope["graph_cache_reports"][0] = reference(root, path)
    receipt_path = root / envelope["graph_cache_controlled_activation"]["path"]
    receipt = json.loads(receipt_path.read_text())
    receipt["graph_cache_reports"] = envelope["graph_cache_reports"]
    write_json(receipt_path, receipt)
    envelope["graph_cache_controlled_activation"] = reference(root, receipt_path)
    envelope["target_config_sha256"] = gate.canonical_config_sha256(target)
    write_json(gate.allocation_gate_path(target), envelope)
    with pytest.raises(ValueError):
        gate.validate_production_ring_allocations(target)
