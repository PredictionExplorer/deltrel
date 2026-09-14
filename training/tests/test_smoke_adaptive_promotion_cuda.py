import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
