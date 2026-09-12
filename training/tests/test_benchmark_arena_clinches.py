from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest

import scripts.benchmark_arena_clinches as benchmark
from startrain.arena import ArenaRunner
from startrain.config import ArenaConfig
from startrain.inference import InferenceResponse
from startrain.selfplay import GameVariant


class Evaluator:
    def __init__(self, identity):
        self.model_version = identity
        self.evaluator_calls = 0
        self.evaluator_rows = 0

    def evaluate(self, requests):
        self.evaluator_calls += 1
        self.evaluator_rows += len(requests)
        return InferenceResponse(
            list(requests.tokens),
            [-1.0] * len(requests),
            list(requests.legal_offsets),
            [0.0] * len(requests.legal_actions),
        )


@pytest.fixture
def native_inputs():
    native = pytest.importorskip("star_native")
    cfg = ArenaConfig(
        rings=(4,),
        balanced_cells=True,
        pairs_per_ring=4,
        minimum_pairs_per_ring=4,
        max_pairs_per_ring=16,
        simulations=2,
        max_considered=2,
        bootstrap_samples=200,
    )
    subject = ArenaRunner(
        native_module=native,
        candidate=Evaluator("candidate"),
        baseline=Evaluator("baseline"),
        config=cfg,
    )
    subject._initialize_resume(None, lambda _snapshot: None)
    for mode in ("classic", "double"):
        for pie, handicap in ((False, 1), (True, 1), (False, 9)):
            variant = GameVariant(mode=mode, pie=pie, handicap=handicap)
            pairs = [3, 7, 11, 15] if handicap == 9 else [0, 1, 2, 3]
            with subject._inference_owner() as executor:
                subject._play_ring_batch(
                    4,
                    subject._pair_specifications(4, pairs, variant),
                    variant=variant,
                    progress=None,
                    inference_executor=executor,
                    stop_requested=lambda: False,
                )
    saved = subject._resume_snapshot()
    proofs = []
    for entry in saved["game_states"]:
        variant = GameVariant.parse(entry["variant"])
        states = native.StateBatch(
            4, 1, mode=variant.mode, pie=variant.pie, handicap=variant.handicap
        )
        if entry["opening_action"] is not None:
            states.apply_many([0], [entry["opening_action"]])
        first = None
        for index, action in enumerate(entry["actions"]):
            if first is None and subject._proven_clinch_winners(states):
                first = index
            states.apply_many([0], [action])
        if first is None:
            continue
        proofs.append(
            {
                "variant": variant.label,
                "pair": entry["pair"],
                "seat": entry["candidate_player"],
                "completed": True,
                "clinch_move": first,
                "searched_moves": len(entry["actions"]),
                "clinch_winner": entry["result"]["winner"],
            }
        )
    return native, cfg, saved, {"games": proofs}


@pytest.mark.native
def test_selected_six_mode_native_tails_keep_proof_seeds_and_resume(native_inputs):
    native, cfg, saved, proof = native_inputs
    cases = benchmark.select_cases(saved, proof, 3)
    assert len(cases) == 6
    benchmark.validate_cases(cases, native, cfg)
    for case in cases:
        assert all(0 < row["original_tail_moves"] <= 3 for row in case["seats"])
        if "handicap" in case["variant"]:
            assert case["variant"].startswith("handicap-9-")
            assert {row["pda"] for row in case["seats"]} == {3}
        candidate, baseline = Evaluator("candidate"), Evaluator("baseline")
        config = SimpleNamespace(arena=cfg)
        _, control = benchmark.play_case(
            case, config, native, candidate, baseline, enabled=False
        )
        before = candidate.evaluator_rows + baseline.evaluator_rows
        _, early = benchmark.play_case(
            case, config, native, candidate, baseline, enabled=True
        )
        assert candidate.evaluator_rows + baseline.evaluator_rows == before
        assert control["pairs"] == early["pairs"]
        assert all(
            entry["actions"] == original["actions"]
            for entry, original in zip(
                early["game_states"], case["resume"]["game_states"], strict=True
            )
        )
        _, restored = benchmark.play_case(
            case, config, native, candidate, baseline, enabled=False, saved=early
        )
        assert restored["pairs"] == early["pairs"]
        assert candidate.evaluator_rows + baseline.evaluator_rows == before


@pytest.mark.native
def test_benchmark_rejects_mismatched_proof_or_contract(native_inputs):
    native, cfg, saved, proof = native_inputs
    forged = json.loads(json.dumps(proof))
    forged["games"][0]["clinch_winner"] ^= 1
    with pytest.raises(ValueError, match="evidence disagrees"):
        benchmark.select_cases(saved, forged, 3)
    cases = benchmark.select_cases(saved, proof, 3)
    cases[0]["expected_winners"][0] ^= 1
    with pytest.raises(ValueError, match="winner.*proof"):
        benchmark.validate_cases(cases, native, cfg)
    with pytest.raises(ValueError, match="evaluation contract"):
        benchmark.validate_cases(
            benchmark.select_cases(saved, proof, 3), native, replace(cfg, simulations=4)
        )


def test_controller_uses_bounded_private_child_and_records_timeout(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        benchmark,
        "prepare",
        lambda _args: (
            {"plan": "fixed"},
            [],
            SimpleNamespace(
                orchestration=SimpleNamespace(
                    directories=SimpleNamespace(root=tmp_path / "run")
                )
            ),
            None,
            None,
            None,
        ),
    )
    output = tmp_path / "new-output"

    def timed_out(command, *, env, timeout):
        assert "--worker" in command
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-reserved"
        assert env["TORCHINDUCTOR_CACHE_DIR"].startswith(str(output))
        assert timeout == 30
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(benchmark, "_run_process", timed_out)
    with pytest.raises(subprocess.TimeoutExpired):
        benchmark.main(
            [
                "--profile",
                "profile.yaml",
                "--resume",
                "resume.json",
                "--proof",
                "proof.json",
                "--execute",
                "--gpu-uuid",
                "GPU-reserved",
                "--output-dir",
                str(output),
                "--timeout-seconds",
                "30",
            ]
        )
    assert json.loads((output / "failure.json").read_text())["status"] == "failed"
    assert (output / "plan.json").is_file()
