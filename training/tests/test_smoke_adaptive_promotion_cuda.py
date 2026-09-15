import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy

import pytest
import torch

from scripts import smoke_adaptive_promotion_cuda as smoke
from startrain.config import ArenaConfig, load_config
from startrain.inference import GraphInferenceAdapter, InferenceConfig
from startrain.model import GraphResTNet, ModelConfig


def stopped_run(tmp_path, monkeypatch):
    from scripts import migrate_continuous_profile as migration

    root = tmp_path / "run"
    status = root / "status"
    status.mkdir(parents=True)
    checkpoint = root / "learner" / "recovery" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    selected = checkpoint.with_name(f"sha256-{digest}.pt")
    checkpoint.rename(selected)
    (root / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "run",
                "generation_family": "family",
                "created_ns": 1,
            }
        )
    )
    (root / "learner" / "recovery.json").write_text(
        json.dumps(
            {
                "format": "startrain.recovery-pointer",
                "schema_version": 1,
                "run_id": "run",
                "generation_family": "family",
                "step": 1,
                "epoch": 0,
                "examples_consumed": 2,
                "updated_ns": 1,
                "checkpoint": f"recovery/{selected.name}",
                "checkpoint_sha256": digest,
                "checkpoint_bytes": selected.stat().st_size,
            }
        )
    )
    heartbeat = status / "learner.heartbeat.json"
    heartbeat.write_text(
        json.dumps({"schema_version": 1, "worker": "learner", "pid": 123})
    )
    coordinator = {
        "state": "stopped",
        "coordinator_pid": 124,
        "failure": None,
        "workers": {
            "learner": {
                "state": "drained",
                "pid": None,
                "failure_reason": None,
                "heartbeat": str(heartbeat),
            }
        },
    }
    (status / "coordinator.json").write_text(json.dumps(coordinator))
    monkeypatch.setattr(migration, "_pid_is_live", lambda pid: pid == 999)
    config = SimpleNamespace(
        orchestration=SimpleNamespace(
            run_id="run", directories=SimpleNamespace(root=str(root), status="status")
        )
    )
    return config, selected, status, coordinator


def test_stopped_guard_uses_current_inventory_and_ignores_obsolete_live_pid(
    tmp_path, monkeypatch
):
    config, checkpoint, status, _ = stopped_run(tmp_path, monkeypatch)
    (status / "actor-obsolete.heartbeat.json").write_text(json.dumps({"pid": 999}))
    result = smoke.require_stopped_run(config, checkpoint)
    assert result["stopped_worker_pids"] == {"learner": 123}
    assert result["checkpoint_step"] == 1


@pytest.mark.parametrize(
    "fault", ["coordinator", "failure", "worker", "heartbeat", "path"]
)
def test_stopped_guard_rejects_live_or_unbound_current_workers(
    tmp_path, monkeypatch, fault
):
    config, checkpoint, status, coordinator = stopped_run(tmp_path, monkeypatch)
    if fault == "coordinator":
        coordinator["coordinator_pid"] = 999
    elif fault == "failure":
        coordinator["failure"] = "failed"
    elif fault == "worker":
        coordinator["workers"]["learner"]["pid"] = 999
    elif fault == "heartbeat":
        (status / "learner.heartbeat.json").write_text(
            json.dumps({"worker": "learner", "pid": 999})
        )
    else:
        coordinator["workers"]["learner"]["heartbeat"] = str(
            status / "obsolete.heartbeat.json"
        )
    (status / "coordinator.json").write_text(json.dumps(coordinator))
    with pytest.raises(ValueError):
        smoke.require_stopped_run(config, checkpoint)


def test_report_cannot_replace_existing_evidence(tmp_path):
    path = tmp_path / "report.json"
    smoke.publish_report(path, {"status": "passed"})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        smoke.publish_report(path, {"status": "failed"})
    assert path.read_bytes() == before


def test_native_config_preserves_production_padding_and_graph_shadow_is_separate():
    config = load_config(
        Path(__file__).parents[1] / "configs/h100-8gpu-largest-board-priority.yaml"
    )
    refresh = config.orchestration.model_refresh
    config = replace(
        config,
        orchestration=replace(
            config.orchestration,
            model_refresh=replace(
                refresh,
                inference=replace(
                    refresh.inference, cuda_graphs=False, small_batch_graph_buckets=True
                ),
            ),
        ),
    )
    production, shadow = smoke.smoke_inference_configs(config)
    assert production.cuda_graphs is False
    assert (
        production.deduplicate
        == config.orchestration.model_refresh.inference.deduplicate
    )
    assert production.cache_max_entries == production.cache_max_bytes == 0
    assert replace(production, cuda_graphs=True) == shadow
    device = torch.device("cuda:0")
    assert (
        GraphInferenceAdapter._inference_batch_rows(
            SimpleNamespace(device=device, config=production), 18
        )
        == 32
    )
    assert (
        GraphInferenceAdapter._inference_batch_rows(
            SimpleNamespace(device=device, config=shadow), 18
        )
        == 24
    )


@pytest.mark.parametrize(
    ("production_status", "graphs", "shadow_status", "boundary_changed", "expected"),
    [
        ("failed", False, "passed", False, "failed"),
        ("passed", False, "failed", False, "passed"),
        ("passed", True, "failed", False, "failed"),
        ("passed", True, "passed", False, "passed"),
        ("passed", False, "failed", True, "failed"),
        ("passed", False, "passed", True, "failed"),
    ],
)
def test_required_production_and_optional_shadow_boundaries_are_explicit(
    monkeypatch, production_status, graphs, shadow_status, boundary_changed, expected
):
    production = {
        "status": production_status,
        "production_inference": {"configuration": {"cuda_graphs": graphs}},
    }
    shadow = {
        "status": shadow_status,
        "error": "bitwise mismatch" if shadow_status == "failed" else None,
    }
    stages = []

    def run_stage(command, timeout):
        stages.append(("--shadow-only" in command, timeout))
        return deepcopy(shadow if "--shadow-only" in command else production)

    checks = []

    def revalidate(*_args):
        checks.append(True)
        if boundary_changed:
            raise RuntimeError("checkpoint changed")

    monkeypatch.setattr(smoke, "run_owned_stage", run_stage)
    monkeypatch.setattr(smoke, "revalidate_boundary", revalidate)
    result = smoke.run_stages(
        SimpleNamespace(profile="profile", checkpoint="checkpoint", output="report")
    )
    assert result["status"] == expected
    if production_status == "failed":
        assert stages == [(False, 180)]
        assert result["shadow_graph_check"]["status"] == "not_run"
        assert checks == []
    else:
        assert stages == [(False, 180), (True, 60)]
        assert checks == [True]
        assert result["shadow_graph_check"]["status"] == shadow_status
        assert result["shadow_graph_parity_qualified"] is (shadow_status == "passed")
        if boundary_changed:
            assert "checkpoint changed" in result["boundary_revalidation_error"]


def test_reused_shadow_failure_stays_failed_after_successful_cold_retry(monkeypatch):
    from scripts import validate_cuda_graph_runtime as graph_checks

    class Graph:
        def __init__(self):
            self._graphs = SimpleNamespace(_entries={})
            self._graphs.clear = self._graphs._entries.clear
            self.logical_rows = {18}
            self.comparisons = 0
            self.calls = self.captures = 0

        def efficiency_snapshot(self):
            return {"graph_captures": self.captures, "graph_replays": self.calls}

        def evaluate(self, mode):
            self.calls += 1
            if not self._graphs._entries:
                self.captures += 1
                self._graphs._entries["shape"] = True
            if self.calls == 3:
                assert mode == "double"
                raise ValueError("policy_logits max_abs=0.125")
            self.comparisons += 1

    monkeypatch.setattr(smoke, "root_probe", lambda _native, _config, mode: mode)
    monkeypatch.setattr(
        graph_checks, "check_health", lambda graph: graph.efficiency_snapshot()
    )
    monkeypatch.setattr(graph_checks, "entry_records", lambda graph: [])
    result = smoke.shadow_checks(None, Graph(), None, None)
    assert result["status"] == "failed"
    failed = next(row for row in result["attempts"] if row["status"] == "failed")
    assert (failed["mode"], failed["phase"], failed["comparison_index"]) == (
        "double",
        "reused-entry",
        2,
    )
    retry = result["attempts"][3]
    assert (
        retry["mode"] == "double" and retry["phase"] == "cold-retry-after-reuse-failure"
    )
    assert retry["status"] == "passed"


@pytest.mark.native
@pytest.mark.parametrize("mode", ["classic", "double"])
def test_eighteen_native_slots_stop_after_one_wave_and_resume_history(mode):
    native = pytest.importorskip("star_native")
    model = GraphResTNet(
        ModelConfig(width=8, rrt_groups=1, attention_heads=2, kv_heads=1)
    ).eval()
    settings = InferenceConfig(precision="fp32")
    adapter = smoke.ObservedAdapter(
        model,
        config=settings,
        homogeneous_relational_bias=True,
        model_version="checkpoint",
    )
    config = ArenaConfig(
        rings=(10,),
        balanced_cells=True,
        variant_policy="pie_even",
        allocation_policy="adaptive_pie",
        pairs_per_ring=4,
        minimum_pairs_per_ring=4,
        simulations=2,
        max_considered=2,
    )
    try:
        result = smoke.exercise_prefix_resume(native, adapter, config, f"pie-{mode}")
        assert result["pairs"] == 9
        assert [wave["game_slots"] for wave in result["waves"]] == [18, 18]
        assert [wave["searched_moves"] for wave in result["waves"]] == [18, 36]
        assert result["resumed_histories_verified"]
        assert adapter.logical_rows and adapter.physical_rows
        assert all(
            wave["shared_inference"]["completed_requests"] > 0
            for wave in result["waves"]
        )
    finally:
        adapter.close()
