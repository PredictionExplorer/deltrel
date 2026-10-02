import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from scripts import benchmark_search_budgets as benchmark


def test_standalone_evaluator_accepts_shared_production_actor_topology():
    from dataclasses import replace
    from deltreltrain.config import load_config

    source = load_config("configs/h100-8gpu.yaml")
    refresh = source.orchestration.model_refresh
    source = replace(
        source,
        orchestration=replace(
            source.orchestration,
            gpus=tuple(
                replace(gpu, actor_cohorts=2) if gpu.role == "actor" else gpu
                for gpu in source.orchestration.gpus
            ),
            model_refresh=replace(
                refresh,
                compatible_cohort_work=True,
                work_scheduling=replace(refresh.work_scheduling, enabled=True),
                inference=replace(
                    refresh.inference,
                    shared_batching=True,
                    cache_max_entries=100,
                    cache_max_bytes=1000000,
                    cuda_graphs=True,
                ),
            ),
        ),
    )
    result = benchmark.standalone_evaluation_config(source)
    inference = result.orchestration.model_refresh.inference
    assert result.game == source.game
    assert result.model == source.model
    assert result.selfplay == source.selfplay
    assert result.orchestration.gpus == source.orchestration.gpus
    assert inference.shared_batching
    assert not inference.cuda_graphs and not inference.cache_max_entries
    assert not result.train.compile


def write_positions(path: Path, **changes):
    value = {"id": "fixed", "rings": 4, "actions": [], "seed": 7}
    value.update(changes)
    path.write_text(json.dumps({"schema_version": 1, "positions": [value]}))
    return path


def arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "load_model_manifest",
        lambda _path: SimpleNamespace(
            model_identity="immutable",
            manifest_sha256="manifest-digest",
            checkpoint_sha256="checkpoint-digest",
        ),
    )
    return [
        "--config",
        str(Path("configs/small.yaml").resolve()),
        "--checkpoint",
        str(tmp_path / "checkpoint.json"),
        "--positions",
        str(write_positions(tmp_path / "positions.json")),
    ]


def test_default_benchmark_only_reports_immutable_plan(tmp_path, monkeypatch, capsys):
    args = arguments(tmp_path, monkeypatch)
    monkeypatch.setattr(
        benchmark,
        "load_deltrel_native",
        lambda **_: pytest.fail("native execution needs --execute"),
    )
    assert benchmark.main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["model_identity"] == "immutable"
    assert plan["positions"][0]["seed"] == 7
    assert plan["reference_cap"] > max(plan["caps"])
    assert "not Elo" in plan["scope"]
    assert "prediction cache" in plan["runtime"]
    assert len(plan["positions_sha256"]) == len(plan["plan_sha256"]) == 64


@pytest.mark.parametrize(
    "changes",
    [
        {"rings": 5},
        {"seed": -1},
        {"pda": 4},
        {"actions": [True]},
        {"actions": [51]},
        {"unknown": 1},
    ],
)
def test_frozen_position_inputs_reject_invalid_or_unknown_semantics(tmp_path, changes):
    with pytest.raises((TypeError, ValueError)):
        benchmark.read_positions(
            write_positions(tmp_path / "positions.json", **changes)
        )


def test_budget_comparison_uses_common_legal_support_and_independent_differences():
    reference = {
        "actions": [1, 3],
        "selected_action": 1,
        "selected_value": 0.5,
        "policy_target": [0.75, 0.25],
        "q_values": [0.5, -0.25],
    }
    candidate = {
        "actions": [1, 3],
        "selected_action": 3,
        "selected_value": 0.2,
        "policy_target": [0.4, 0.6],
        "q_values": [0.1, 0.2],
    }
    result = benchmark.compare_search(candidate, reference)
    assert result["action_matches_reference"] is False
    assert result["policy_target_l1"] == pytest.approx(0.7)
    assert result["selected_value_absolute_difference"] == pytest.approx(0.3)
    assert result["reference_selected_q_minus_candidate_action_q"] == pytest.approx(
        0.75
    )
    with pytest.raises(ValueError, match="action spaces"):
        benchmark.compare_search({**candidate, "actions": [3, 1]}, reference)


def test_execution_rejects_changed_frozen_plan_and_existing_outputs(
    tmp_path, monkeypatch
):
    args = arguments(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="inputs changed"):
        benchmark.main([*args, "--execute", "--worker", "--pinned-plan", "wrong"])
    output = tmp_path / "existing.json"
    output.write_text("keep")
    with pytest.raises(SystemExit):
        benchmark.main([*args, "--execute", "--output", str(output)])
    assert output.read_text() == "keep"


@pytest.mark.parametrize("interrupted", [False, True])
def test_process_timeout_kills_only_owned_worker_group(
    tmp_path, monkeypatch, interrupted
):
    args = arguments(tmp_path, monkeypatch)
    killed = []
    calls = []

    class Child:
        pid = 56789

        def communicate(self, timeout=None):
            calls.append(timeout)
            if timeout is not None:
                if interrupted:
                    raise KeyboardInterrupt()
                raise subprocess.TimeoutExpired("benchmark", timeout)
            return "", ""

    monkeypatch.setattr(
        benchmark.subprocess, "Popen", lambda *_args, **_kwargs: Child()
    )
    monkeypatch.setattr(
        benchmark.os, "killpg", lambda pid, signal: killed.append((pid, signal))
    )
    with pytest.raises(KeyboardInterrupt if interrupted else TimeoutError):
        benchmark.main(
            [
                *args,
                "--execute",
                "--output",
                str(tmp_path / "new.json"),
                "--timeout-seconds",
                "0.1",
            ]
        )
    assert killed == [(56789, benchmark.signal.SIGKILL)]
    assert calls == [0.1, None]
    assert not (tmp_path / "new.json").exists()


@pytest.mark.native
def test_native_frozen_position_sweep_records_real_rows_and_reference_differences():
    import time
    from deltreltrain.config import load_config
    from deltreltrain.search_options import FullSearchBudgetConfig
    from test_inference_efficiency import cached_adapter

    native = pytest.importorskip("deltrel_native")
    evaluator = cached_adapter()
    position = benchmark.FrozenPosition("late", 4, tuple(range(42)), seed=19)
    config = load_config("configs/small.yaml").selfplay
    deadline = time.monotonic() + 10
    try:
        reference = benchmark.search_position(
            native,
            evaluator,
            position,
            config,
            16,
            FullSearchBudgetConfig(),
            deadline=deadline,
        )
        candidate = benchmark.search_position(
            native,
            evaluator,
            position,
            config,
            8,
            FullSearchBudgetConfig(mode="root-entropy", entropy_threshold=1),
            first_visit_batch_size=2,
            deadline=deadline,
        )
    finally:
        evaluator.close()
    assert 1 <= candidate["actual_simulations"] <= candidate["base_cap"]
    assert candidate["inference"]["evaluator_rows"] > 0
    assert candidate["inference"]["neural_rows"] > 0
    assert sum(candidate["visits"]) == candidate["actual_simulations"]
    assert sum(candidate["policy_target"]) == pytest.approx(1)
    assert 0 <= benchmark.compare_search(candidate, reference)["policy_target_l1"] <= 2
