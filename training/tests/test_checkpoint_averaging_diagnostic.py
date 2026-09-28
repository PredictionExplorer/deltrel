from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import compare_checkpoint_averaging as diagnostic
from scripts.run_lineage_arena import load_candidate
from deltreltrain.checkpoint import ExponentialMovingAverage, save_checkpoint
from deltreltrain.config import GameConfig
from deltreltrain.model import GraphResTNet, ModelConfig


def test_diagnostic_has_bounded_complete_four_cell_contract():
    config = diagnostic.diagnostic_config()
    assert config.rings == (10,)
    assert (
        config.pairs_per_ring
        == config.minimum_pairs_per_ring
        == config.max_pairs_per_ring
        == 8
    )
    assert config.continuation_pairs_per_ring is None
    assert diagnostic.balanced_cells(config) == (
        "r10/classic-pie",
        "r10/double-pie",
        "r10/classic-handicap",
        "r10/double-handicap",
    )
    with pytest.raises(ValueError, match="complete"):
        diagnostic.diagnostic_config(pairs_per_cell=5)


def test_raw_and_ema_diagnostic_load_distinct_weights_without_mutating_checkpoint(
    tmp_path,
):
    config = ModelConfig(width=8, rrt_groups=1, attention_heads=2, kv_heads=1)
    model = GraphResTNet(config)
    ema = ExponentialMovingAverage(model)
    original = next(model.parameters()).detach().clone()
    with torch.no_grad():
        next(model.parameters()).add_(1)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        step=10,
        ema=ema,
        config={"model": asdict(config), "game": asdict(GameConfig())},
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    raw, raw_meta = load_candidate(path, device=torch.device("cpu"), weights="raw")
    averaged, ema_meta = load_candidate(path, device=torch.device("cpu"), weights="ema")
    torch.testing.assert_close(next(raw.model.parameters()), original + 1)
    torch.testing.assert_close(next(averaged.model.parameters()), original)
    assert raw_meta["identity"] != ema_meta["identity"]
    assert raw_meta["checkpoint_sha256"] == ema_meta["checkpoint_sha256"] == digest
    assert raw_meta["weights"] == "raw" and ema_meta["weights"] == "ema"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


@pytest.fixture
def frozen(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    config = ModelConfig(width=8, rrt_groups=1, attention_heads=2, kv_heads=1)
    model = GraphResTNet(config)
    ema = ExponentialMovingAverage(model)
    with torch.no_grad():
        next(model.parameters()).add_(1)
    path = root / "recovery.pt"
    save_checkpoint(
        path,
        model=model,
        step=10,
        ema=ema,
        config={"model": asdict(config), "game": asdict(GameConfig())},
    )
    champion = root / "champion.pt"
    champion.write_bytes(path.read_bytes())
    output = tmp_path / "diagnostic"
    config = diagnostic.diagnostic_config(pairs_per_cell=4, simulations=1)
    plan = diagnostic.freeze_plan(
        source_run_root=root,
        checkpoint=path,
        champion_checkpoint=champion,
        output=output,
        config=config,
    )
    return root, output, plan, config


def test_plan_freezes_both_sources_and_never_writes_to_run(frozen):
    root, output, plan, config = frozen
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    verified, restored = diagnostic.verify_plan(output)
    assert restored == config
    assert verified == plan
    assert before == {path.name: path.read_bytes() for path in root.iterdir()}
    # A moving/replaced source checkpoint cannot redirect a resumed experiment.
    (root / "recovery.pt").unlink()
    (root / "champion.pt").write_text("new live checkpoint")
    diagnostic.verify_plan(output)
    assert (output / "checkpoint.pt").read_bytes() == before["recovery.pt"]


def test_plan_rejects_frozen_checkpoint_and_contract_corruption(frozen):
    _, output, _, _ = frozen
    checkpoint = output / "checkpoint.pt"
    checkpoint.chmod(0o644)
    checkpoint.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="(SHA|sha|byte|size)"):
        diagnostic.verify_plan(output)
    path = output / diagnostic.PLAN_NAME
    plan = json.loads(path.read_text())
    plan["arena"]["simulations"] += 1
    path.chmod(0o644)
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="plan or implementation hash"):
        diagnostic.verify_plan(output)


def test_source_output_isolation_rejected_before_any_write(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    output = root / "arena"
    assert (
        diagnostic.main(
            ["--source-run-root", str(root), "--output-dir", str(output), "--plan-only"]
        )
        == 2
    )
    assert list(root.iterdir()) == []


def test_gpu_gate_requires_reserved_idle_device(monkeypatch):
    cuda = torch.device("cuda:0")
    with pytest.raises(ValueError, match="reserve"):
        diagnostic.require_exclusive_device(cuda, acknowledged=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(ValueError, match="full GPU UUID"):
        diagnostic.require_exclusive_device(cuda, acknowledged=True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")

    def query(args, **kwargs):
        return SimpleNamespace(
            stdout="GPU-test\n" if "--query-gpu=uuid" in args else "1234\n"
        )

    monkeypatch.setattr(diagnostic.subprocess, "run", query)
    with pytest.raises(ValueError, match="compute processes: 1234"):
        diagnostic.require_exclusive_device(cuda, acknowledged=True)
    monkeypatch.setattr(
        diagnostic.subprocess,
        "run",
        lambda args, **kwargs: SimpleNamespace(
            stdout="GPU-test\n" if "--query-gpu=uuid" in args else ""
        ),
    )
    diagnostic.require_exclusive_device(cuda, acknowledged=True)


def test_resume_keeps_same_champion_seeds_budget_and_incomplete_games(
    frozen, monkeypatch
):
    root, output, plan, config = frozen
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    runs = []

    class Runner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            runs.append(self)

        def run(self, *, resume_state, checkpoint, stop_requested):
            assert self.kwargs["config"] == config
            self.resume = resume_state
            snapshot = {
                "pairs": [],
                "games": [],
                "game_states": [{"marker": "in-flight"}],
                "candidate": self.kwargs["candidate"].model_version,
                "baseline": self.kwargs["baseline"].model_version,
            }
            checkpoint(snapshot)
            assert (output / "raw.json").exists()
            return {
                "interrupted": True,
                "resume_state": snapshot,
                "pairs": [],
                "games": [],
                "promotion": {"decision": "promote"},
            }

    monkeypatch.setattr(diagnostic, "ArenaRunner", Runner)
    options = dict(output=output, device=torch.device("cpu"), native_module=object())
    first = diagnostic.run_session(**options, arm="raw", stop_requested=lambda: True)
    second = diagnostic.run_session(**options, arm="raw")
    third = diagnostic.run_session(**options)  # auto chooses the less sampled EMA arm
    assert all(result["terminal"] is False for result in (first, second, third))
    assert runs[0].resume is None
    assert runs[1].resume["game_states"] == [{"marker": "in-flight"}]
    assert (
        runs[0].kwargs["candidate"].model_version
        != runs[2].kwargs["candidate"].model_version
    )
    assert len({run.kwargs["baseline"].model_version for run in runs}) == 1
    for name in diagnostic.ARMS:
        saved = json.loads((output / f"{name}.json").read_text())
        assert saved["plan_sha256"] == plan["plan_sha256"]
        assert "promotion" not in saved
    assert before == {path.name: path.read_bytes() for path in root.iterdir()}


def test_session_time_limit_is_passed_to_arena_stop_callback(frozen, monkeypatch):
    _, output, _, _ = frozen
    clock = iter((0.0, 301.0))
    monkeypatch.setattr(diagnostic.time, "monotonic", lambda: next(clock))

    class Runner:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            assert kwargs["stop_requested"]() is True
            return {"interrupted": True, "pairs": [], "games": []}

    monkeypatch.setattr(diagnostic, "ArenaRunner", Runner)
    diagnostic.run_session(
        output=output, device=torch.device("cpu"), native_module=object()
    )


def test_short_resumed_wave_keeps_all_previously_completed_evidence(
    frozen, monkeypatch
):
    _, output, _, _ = frozen
    pair = asdict(
        diagnostic.ArenaPair(
            ring=10,
            pair=0,
            opening_seed=123,
            opening_action=None,
            forced_opening=False,
            outcomes=(1, -1),
            variant="pie-classic",
            segment="pie",
        )
    )

    class Runner:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            return {
                "interrupted": True,
                "pairs": [],
                "games": [],
                "resume_state": {"pairs": [pair], "games": [{"saved": True}]},
            }

    monkeypatch.setattr(diagnostic, "ArenaRunner", Runner)
    summary = diagnostic.run_session(
        output=output,
        device=torch.device("cpu"),
        native_module=object(),
    )
    assert summary["arms"]["raw"]["completed_pairs"] == 1
    result = json.loads((output / "raw.json").read_text())
    assert len(result["pairs"]) == 1
    assert result["games"] == [{"saved": True}]
    assert "promotion" not in result
