from types import SimpleNamespace
from dataclasses import replace
import hashlib
from unittest.mock import patch

import pytest

from scripts import benchmark_local_message_adapter as benchmark
from deltreltrain.config import ActorPipelineConfig, load_config
from pathlib import Path


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--batch-sizes", "257"),
        ("--timeout-seconds", "481"),
        ("--max-memory-gib", "49"),
        ("--repeats", "1"),
    ],
)
def test_invalid_resource_bounds(flag, value):
    args = benchmark._parser().parse_args(
        ["--config", "profile.yaml", "--checkpoint", "model.json", flag, value]
    )
    with pytest.raises(ValueError):
        benchmark._validate(args)


def test_probability_gate_and_ties_are_reported_independently():
    reference = SimpleNamespace(
        policy_offsets=[0, 2, 4], policy_logits=[0.0, 0.0, 1.0, 3.0]
    )
    assert benchmark.compare_policy(reference, reference)["passed"]
    shifted = SimpleNamespace(
        policy_offsets=reference.policy_offsets, policy_logits=[5.0, 5.0, 9.0, 11.0]
    )
    assert benchmark.compare_policy(shifted, reference)["passed"]
    broken = SimpleNamespace(
        policy_offsets=reference.policy_offsets, policy_logits=[0.0, 0.0, 3.0, 1.0]
    )
    assert not benchmark.compare_policy(broken, reference)["passed"]


def test_adapter_preserves_production_transport_settings_but_measures_misses():
    config = load_config(
        Path(__file__).parents[1] / "configs/h100-8gpu-largest-board-priority.yaml"
    )
    settings = benchmark.adapter_config(config)
    effective, _ = benchmark.actor_experiment(config)
    source = effective.orchestration.model_refresh.inference
    assert settings.precision == "bf16"
    assert settings.cache_max_entries == settings.cache_max_bytes == 0
    assert not settings.deduplicate
    for name in (
        "pinned_transfers",
        "cuda_graphs",
        "preserve_broadcast_topology",
        "small_batch_graph_buckets",
        "compact_inference_gather",
    ):
        assert getattr(settings, name) == getattr(source, name)


def pipeline_profile():
    config = load_config(
        Path(__file__).parents[1] / "configs/h100-8gpu-largest-board-priority.yaml"
    )
    pipeline = ActorPipelineConfig(
        compatible_work=True,
        stream_completed_games=True,
        rolling_game_slots=True,
        seed_contract="game-v1",
        cohort_search_budgets=True,
        cuda_graphs=True,
    )
    return replace(
        config,
        orchestration=replace(
            config.orchestration,
            gpus=tuple(
                replace(gpu, actor_pipeline=pipeline) if gpu.role == "actor" else gpu
                for gpu in config.orchestration.gpus
            ),
        ),
    )


def test_gpu_pipeline_overrides_global_graph_and_plan_preserves_raw_pin(tmp_path):
    config = pipeline_profile()
    assert not config.orchestration.model_refresh.inference.cuda_graphs
    selected = config.orchestration.actor_gpus[0]
    assert selected.gpu_id == 1 and selected.actor_pipeline.cuda_graphs
    effective, gpu = benchmark.actor_experiment(config, 1)
    assert effective.orchestration.model_refresh.inference.cuda_graphs
    assert effective.selfplay.seed_contract == selected.actor_pipeline.seed_contract
    assert (
        effective.orchestration.model_refresh.compatible_cohort_work
        == selected.actor_pipeline.compatible_work
    )
    assert benchmark.adapter_config(config, actor_gpu_id=1).cuda_graphs
    assert benchmark.adapter_config(config, actor_gpu_id=1).cuda_graph_max_bytes == (
        config.orchestration.model_refresh.inference.cuda_graph_max_bytes
        // (selected.actor_cohorts + 2)
    )
    assert not config.orchestration.model_refresh.inference.cuda_graphs
    raw = tmp_path / "profile.yaml"
    raw.write_text("frozen profile input")
    args = benchmark._parser().parse_args(
        ["--config", str(raw), "--checkpoint", "manifest.json", "--actor-gpu-id", "1"]
    )
    manifest = SimpleNamespace(
        model_identity="identity",
        manifest_sha256="manifest",
        checkpoint_sha256="checkpoint",
    )
    with (
        patch("deltreltrain.config.load_config", return_value=config),
        patch("deltreltrain.checkpoint.load_model_manifest", return_value=manifest),
    ):
        plan = benchmark.plan(args)
    assert plan["actor_gpu_id"] == 1
    assert plan["effective_actor_runtime"]["cuda_graphs"] is True
    assert plan["effective_adapter_config"]["cuda_graphs"] is True
    assert (
        plan["effective_adapter_config"]["cuda_graph_max_bytes"]
        == plan["effective_actor_runtime"]["per_model_cuda_graph_max_bytes"]
    )
    assert plan["config_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert plan["effective_config_sha256"]


def test_explicit_actor_selection_does_not_apply_another_gpu_pipeline():
    config = pipeline_profile()
    other = config.orchestration.actor_gpus[1]
    changed = replace(
        other, actor_pipeline=replace(other.actor_pipeline, cuda_graphs=False)
    )
    config = replace(
        config,
        orchestration=replace(
            config.orchestration,
            gpus=tuple(
                changed if gpu.gpu_id == other.gpu_id else gpu
                for gpu in config.orchestration.gpus
            ),
        ),
    )
    assert benchmark.adapter_config(config, actor_gpu_id=1).cuda_graphs
    assert not benchmark.adapter_config(config, actor_gpu_id=other.gpu_id).cuda_graphs
    with pytest.raises(ValueError, match="not a configured actor"):
        benchmark.actor_experiment(config, 0)


@pytest.mark.native
def test_benchmark_fixtures_are_real_distinct_native_requests():
    import deltrel_native

    first = benchmark.requests(deltrel_native, 10, 64, 91)
    second = benchmark.requests(deltrel_native, 10, 64, 91)
    assert type(first) is deltrel_native.EvalBatch
    assert first is not second
    assert first.inference_keys() == second.inference_keys()
    assert len(set(first.inference_keys())) == 64
    assert len(first) == 64
