from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    options: dict[str, Any] = dict(
        output=output, device=torch.device("cpu"), native_module=object()
    )
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


@pytest.fixture
def raw_frozen(tmp_path):
    """Distinct tiny CPU inputs: each also has different RAW and EMA tensors."""
    source = tmp_path / "raw-inputs"
    source.mkdir()
    config = ModelConfig(width=8, rrt_groups=1, attention_heads=2, kv_heads=1)
    for index, name in enumerate(("candidate", "baseline")):
        model = GraphResTNet(config)
        with torch.no_grad():
            next(model.parameters()).fill_(index)
        ema = ExponentialMovingAverage(model)
        with torch.no_grad():
            next(model.parameters()).add_(10)
        save_checkpoint(
            source / f"{name}.pt",
            model=model,
            step=1000,
            ema=ema,
            config={"model": asdict(config), "game": asdict(GameConfig())},
        )
    output = tmp_path / "raw-output"
    config = diagnostic.diagnostic_config(pairs_per_cell=4, simulations=1)
    plan = diagnostic.freeze_plan(
        source_run_root=source,
        checkpoint=source / "candidate.pt",
        champion_checkpoint=source / "baseline.pt",
        output=output,
        config=config,
        baseline_weights="raw",
    )
    assert (
        plan["checkpoints"]["checkpoint"]["sha256"]
        != plan["checkpoints"]["champion"]["sha256"]
    )
    return source, output, plan, config


@pytest.fixture
def raw_saved(raw_frozen):
    _, output, plan, config = raw_frozen
    # Real Runner constructs its resume contract; stopping before search does
    # not play games or qualify a strength result.
    diagnostic.run_session(
        output=output,
        device=torch.device("cpu"),
        arm="raw",
        native_module=object(),
        stop_requested=lambda: True,
    )
    saved = json.loads((output / "raw.json").read_text())
    assert saved["terminal"] is False and isinstance(saved["resume_state"], dict)
    return output, plan, config, saved


def test_explicit_ema_has_identical_default_v1_plan(frozen, tmp_path):
    source, _, default, config = frozen
    explicit = diagnostic.freeze_plan(
        source_run_root=source,
        checkpoint=source / "recovery.pt",
        champion_checkpoint=source / "champion.pt",
        output=tmp_path / "explicit-ema",
        config=config,
        baseline_weights="ema",
    )
    assert explicit == default
    assert default["schema_version"] == 1 and "baseline_weights" not in default


@pytest.mark.parametrize("weights", [None, True, [], "RAW", "other"])
def test_invalid_baseline_selection_refuses_before_output_or_copy(
    tmp_path, monkeypatch, weights
):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid selector reached checkpoint copy")

    monkeypatch.setattr(diagnostic, "_freeze_checkpoint", unexpected)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="baseline weights"):
        diagnostic.freeze_plan(
            source_run_root=tmp_path,
            checkpoint=tmp_path / "missing.pt",
            champion_checkpoint=tmp_path / "missing2.pt",
            output=output,
            config=diagnostic.diagnostic_config(),
            baseline_weights=weights,
        )
    assert not output.exists()


def test_raw_plan_loads_real_raw_baseline_and_distinct_adapter_identity(
    raw_frozen, monkeypatch
):
    source, output, plan, config = raw_frozen
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    assert diagnostic.verify_plan(output) == (plan, config)
    assert plan["schema_version"] == 2 and plan["baseline_weights"] == "raw"
    runner = diagnostic.ArenaRunner
    observed = []

    def inspected(**kwargs):
        observed.append(kwargs)
        torch.testing.assert_close(
            next(kwargs["candidate"].model.parameters()),
            torch.full_like(next(kwargs["candidate"].model.parameters()), 10),
        )
        torch.testing.assert_close(
            next(kwargs["baseline"].model.parameters()),
            torch.full_like(next(kwargs["baseline"].model.parameters()), 11),
        )
        return runner(**kwargs)

    monkeypatch.setattr(diagnostic, "ArenaRunner", inspected)
    summary = diagnostic.run_session(
        output=output,
        device=torch.device("cpu"),
        arm="raw",
        native_module=object(),
        stop_requested=lambda: True,
    )
    saved = json.loads((output / "raw.json").read_text())
    assert saved["baseline_weights"] == summary["baseline_weights"] == "raw"
    assert observed[0]["baseline_metadata"]["kind"] == "frozen_raw_checkpoint"
    for slot, key in (("checkpoint", "candidate"), ("champion", "baseline")):
        metadata = saved[key + "_metadata"]
        expected = (
            "sha256-"
            + hashlib.sha256(
                f"deltreltrain-weights:raw:{plan['checkpoints'][slot]['sha256']}".encode()
            ).hexdigest()
        )
        assert metadata["weights"] == "raw"
        assert metadata["identity"] == observed[0][key].model_version == expected
        assert metadata["checkpoint_sha256"] == plan["checkpoints"][slot]["sha256"]
        assert metadata["checkpoint_bytes"] == plan["checkpoints"][slot]["bytes"]
        assert saved["resume_state"][key] == expected
    assert "promotion" not in saved
    assert summary["terminal"] is False
    assert not (output / "ema.json").exists()
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}


@pytest.mark.parametrize(
    "version,weights",
    [
        (2, None),
        (2, "ema"),
        (2, True),
        (2, []),
        (2, "other"),
        (1, "raw"),
        (1, "ema"),
        (3, "raw"),
    ],
)
def test_rehashed_invalid_baseline_plan_is_not_admitted(raw_frozen, version, weights):
    _, output, plan, _ = raw_frozen
    bad = deepcopy(plan)
    bad["schema_version"] = version
    if weights is None:
        bad.pop("baseline_weights")
    else:
        bad["baseline_weights"] = weights
    bad["plan_sha256"] = diagnostic._digest(
        {k: v for k, v in bad.items() if k != "plan_sha256"}
    )
    path = output / diagnostic.PLAN_NAME
    path.chmod(0o644)
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="baseline weights or version"):
        diagnostic.verify_plan(output)


@pytest.mark.parametrize("field", ["baseline_weights", "implementation_hashes"])
def test_raw_plan_cannot_bypass_digest_or_source_hash(raw_frozen, field):
    _, output, plan, _ = raw_frozen
    bad = deepcopy(plan)
    bad[field] = "ema" if field == "baseline_weights" else {}
    if field == "implementation_hashes":
        bad["plan_sha256"] = diagnostic._digest(
            {k: v for k, v in bad.items() if k != "plan_sha256"}
        )
    path = output / diagnostic.PLAN_NAME
    path.chmod(0o644)
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="plan or implementation hash"):
        diagnostic.verify_plan(output)


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_body",
        "null_body",
        "nonmapping_body",
        "candidate",
        "baseline",
        "missing_candidate",
        "missing_baseline",
        "identity_type",
        "swapped",
        "checkpoint_sha_as_identity",
        "config",
        "rules_hash",
        "seed_stream_policy",
        "search_algorithm",
        "candidate_search",
        "baseline_search",
        "metadata_weights",
        "metadata_bytes",
        "metadata_identity",
        "weights",
        "output_candidate",
        "output_baseline",
        "missing_output_candidate",
        "missing_output_baseline",
    ],
)
def test_raw_saved_contract_refuses_before_terminal_or_resume_return(
    raw_saved, monkeypatch, terminal, mutation
):
    output, plan, config, saved = raw_saved
    saved["terminal"] = terminal  # Synthetic flag tests the fast-return boundary.
    resume = saved["resume_state"]
    if mutation == "missing_body":
        del saved["resume_state"]
    elif mutation == "null_body":
        saved["resume_state"] = None
    elif mutation == "nonmapping_body":
        saved["resume_state"] = []
    elif mutation in ("candidate", "baseline"):
        resume[mutation] = "sha256-" + "0" * 64
    elif mutation in ("missing_candidate", "missing_baseline"):
        del resume[mutation.removeprefix("missing_")]
    elif mutation == "identity_type":
        resume["candidate"] = True
    elif mutation == "swapped":
        resume["candidate"], resume["baseline"] = (
            resume["baseline"],
            resume["candidate"],
        )
    elif mutation == "checkpoint_sha_as_identity":
        resume["baseline"] = "sha256-" + plan["checkpoints"]["champion"]["sha256"]
    elif mutation == "config":
        resume["config"]["seed"] += 1
    elif mutation in ("rules_hash", "seed_stream_policy", "search_algorithm"):
        resume[mutation] = "changed"
    elif mutation in ("candidate_search", "baseline_search"):
        resume[mutation]["simulations"] += 1
    elif mutation == "metadata_weights":
        saved["baseline_metadata"]["weights"] = "ema"
    elif mutation == "metadata_bytes":
        saved["baseline_metadata"]["checkpoint_bytes"] = True
    elif mutation == "metadata_identity":
        saved["baseline_metadata"]["identity"] = resume["candidate"]
    elif mutation.startswith("output_"):
        saved[mutation.removeprefix("output_")] = "sha256-" + "0" * 64
    elif mutation.startswith("missing_output_"):
        del saved[mutation.removeprefix("missing_output_")]
    else:
        saved["baseline_weights"] = "ema"
    (output / "raw.json").write_text(json.dumps(saved))

    def unexpected(*args, **kwargs):
        pytest.fail("invalid saved contract reached model load or ArenaRunner")

    monkeypatch.setattr(diagnostic, "load_candidate", unexpected)
    monkeypatch.setattr(diagnostic, "ArenaRunner", unexpected)
    with pytest.raises(ValueError):
        diagnostic.run_session(output=output, device=torch.device("cpu"), arm="raw")
    with pytest.raises(ValueError):
        diagnostic._summary(output, plan, config)


def test_valid_raw_terminal_fast_path_keeps_summary_completion_separate(
    raw_saved, monkeypatch
):
    output, plan, config, saved = raw_saved
    saved["terminal"] = True  # No games are claimed by this synthetic flag fixture.
    (output / "raw.json").write_text(json.dumps(saved))

    def unexpected(*args, **kwargs):
        pytest.fail("terminal contract check loaded another model")

    monkeypatch.setattr(diagnostic, "load_candidate", unexpected)
    monkeypatch.setattr(diagnostic, "ArenaRunner", unexpected)
    summary = diagnostic.run_session(
        output=output, device=torch.device("cpu"), arm="raw"
    )
    assert summary["arms"]["raw"]["terminal"] is True
    assert summary["arms"]["ema"]["terminal"] is False
    assert summary["terminal"] is False and summary["baseline_weights"] == "raw"
    assert all(
        row["raw_minus_ema_score"] is None
        for row in summary["matched_comparison"].values()
    )
    assert diagnostic._load_arm(output, "ema", plan) == {}
    assert diagnostic._summary(output, plan, config) == summary


def test_raw_resume_uses_real_existing_contract_and_preserves_clinch_normalization(
    raw_saved,
):
    output, plan, _, saved = raw_saved
    saved["resume_state"]["config"].pop("exact_clinch_termination", None)
    (output / "raw.json").write_text(json.dumps(saved))
    diagnostic._load_arm(output, "raw", plan)
    diagnostic.run_session(
        output=output,
        device=torch.device("cpu"),
        arm="raw",
        native_module=object(),
        stop_requested=lambda: True,
    )
    assert json.loads((output / "raw.json").read_text())["session_count"] == 2


def test_terminal_raw_arm_cannot_drop_both_output_identities(raw_saved):
    output, plan, _, saved = raw_saved
    saved["terminal"] = True
    del saved["candidate"], saved["baseline"]
    (output / "raw.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="output identity"):
        diagnostic._load_arm(output, "raw", plan)


def test_real_interim_persist_has_no_invented_output_identity(raw_frozen, monkeypatch):
    _, output, plan, _ = raw_frozen

    def stopped_after_checkpoint(self, *, resume_state, checkpoint, stop_requested):
        self._initialize_resume(resume_state, checkpoint)
        checkpoint(self._resume_snapshot())
        raise InterruptedError("simulated interruption after real persist")

    monkeypatch.setattr(diagnostic.ArenaRunner, "run", stopped_after_checkpoint)
    with pytest.raises(InterruptedError, match="simulated interruption"):
        diagnostic.run_session(
            output=output, device=torch.device("cpu"), arm="raw", native_module=object()
        )
    saved = diagnostic._load_arm(output, "raw", plan)
    assert saved["terminal"] is False
    assert "candidate" not in saved and "baseline" not in saved
    assert saved["resume_state"]["candidate"] != saved["resume_state"]["baseline"]


@pytest.mark.parametrize("mutation", ["metadata", "adapter"])
def test_raw_loaded_baseline_must_join_weights_and_adapter_before_runner(
    raw_frozen, monkeypatch, mutation
):
    _, output, _, _ = raw_frozen
    load = diagnostic.load_candidate

    def changed(path, **kwargs):
        evaluator, metadata = load(path, **kwargs)
        if path.name == "champion.pt":
            if mutation == "metadata":
                metadata["weights"] = "ema"
            else:
                evaluator.model_version = "sha256-" + "0" * 64
        return evaluator, metadata

    def unexpected(*args, **kwargs):
        pytest.fail("mismatched loaded baseline reached ArenaRunner")

    monkeypatch.setattr(diagnostic, "load_candidate", changed)
    monkeypatch.setattr(diagnostic, "ArenaRunner", unexpected)
    with pytest.raises(ValueError, match="(selected-weight metadata|adapter identity)"):
        diagnostic.run_session(output=output, device=torch.device("cpu"), arm="raw")
    assert not (output / "raw.json").exists()


def test_baseline_cli_is_freeze_only_and_absent_arm_file_allows_first_run(
    raw_frozen,
    tmp_path,
):
    source, _, _, _ = raw_frozen
    output = tmp_path / "cli-output"
    assert (
        diagnostic.main(
            [
                "--source-run-root",
                str(source),
                "--checkpoint",
                str(source / "candidate.pt"),
                "--champion-checkpoint",
                str(source / "baseline.pt"),
                "--output-dir",
                str(output),
                "--baseline-weights",
                "raw",
                "--plan-only",
            ]
        )
        == 0
    )
    plan, _ = diagnostic.verify_plan(output)
    assert diagnostic._load_arm(output, "raw", plan) == {}
    for weights in ("raw", "ema"):
        assert (
            diagnostic.main(
                [
                    "--output-dir",
                    str(output),
                    "--baseline-weights",
                    weights,
                    "--plan-only",
                ]
            )
            == 2
        )
    assert diagnostic.main(["--output-dir", str(output), "--plan-only"]) == 0
    diagnostic.run_session(
        output=output,
        device=torch.device("cpu"),
        arm="raw",
        native_module=object(),
        stop_requested=lambda: True,
    )
    assert (output / "raw.json").is_file()
