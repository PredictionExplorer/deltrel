from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from startrain.contracts import SEARCH_ALGORITHM_ID, TARGET_OUTCOME
from startrain.checkpoint import load_model_manifest
from startrain.replay import write_replay_shard
from scripts import evaluate_recovery_fresh_games as fresh
from scripts.run_frozen_replay_optimizer_calibration import _digest, _pin
from test_frozen_replay_optimizer_calibration import FAMILY, RUN_ID, _sample
from test_recovery_calibration import _fixture as _calibration_fixture


def _semantic(path, document):
    document = {**document, "sha256": _digest(document)}
    path.write_text(json.dumps(document))
    return document


def _allowlist(
    tmp_path: Path, *, bad_pda=False, missing_row=False, teacher="source-model"
):
    sources = []
    for mode in ("classic", "double"):
        root = tmp_path / mode
        root.mkdir(parents=True)
        samples, summaries = [], []
        for index in range(8):
            sample = _sample(index)
            sample = replace(
                sample,
                mode=mode,
                pie=True,
                game_id=f"fresh-{mode}-{index}",
                actor_id=f"fresh-{mode}",
                target_mask=sample.target_mask | TARGET_OUTCOME,
                outcome=1 if sample.to_move == 0 else 0,
                final_scores=np.array([1, 0], dtype=np.int16),
                model_identity=teacher,
            )
            samples.append(sample)
            summaries.append(
                {
                    "game_id": sample.game_id,
                    "samples": 2 if missing_row and index == 0 else 1,
                    "pda_seat0": 1 if bad_pda and index == 0 else 0,
                    "pda_seat1": 0,
                    "model_identity": sample.model_identity,
                    "variant": "pie-" + mode,
                }
            )
        # An unselected row must never be scored, even if present in a pinned shard.
        excluded = replace(samples[0], game_id=f"excluded-{mode}")
        shard = write_replay_shard(root / "samples.npz", [*samples, excluded])
        plan = _semantic(
            root / "collection-plan.json",
            {
                "search_algorithm": SEARCH_ALGORITHM_ID,
                "champion_identity": teacher,
                "run_id": RUN_ID,
                "generation_family": FAMILY,
                "input_pins": [],
                "implementation_pins": [],
                "source_pins": [],
            },
        )
        summary_path = root / "games.json"
        summary_path.write_text(json.dumps({"games": summaries}))
        sources.append(
            {
                "root": str(root),
                "plan_sha256": plan["sha256"],
                "source_games_sha256": _pin(summary_path).sha256,
                "selected_game_ids": [sample.game_id for sample in samples],
                "shards": [
                    {**_pin(shard).as_dict(), "all_rows": 9, "selected_rows": 8}
                ],
            }
        )
    path = tmp_path / "allowlist.json"
    _semantic(
        path,
        {
            "schema_version": 1,
            "search_algorithm": SEARCH_ALGORITHM_ID,
            "sources": sources,
            "totals": {"games": 16, "positions": 16},
        },
    )
    return path


def test_fresh_loader_uses_only_complete_allowlisted_zero_pda_games(tmp_path):
    allowlist = _allowlist(tmp_path)
    before = allowlist.read_bytes()
    games, evidence = fresh.load_fresh_games(allowlist)
    assert len(games) == 16
    assert evidence["games_by_cell"] == {"pie-classic": 8, "pie-double": 8}
    assert sum(len(game.samples) for game in games) == 16
    assert all(not game.game_id.startswith("excluded") for game in games)
    assert allowlist.read_bytes() == before


@pytest.mark.parametrize(
    "option,reason", [("bad_pda", "zero-PDA"), ("missing_row", "every position")]
)
def test_fresh_loader_refuses_pda_or_partial_games(tmp_path, option, reason):
    allowlist = _allowlist(tmp_path, **{option: True})
    with pytest.raises(ValueError, match=reason):
        fresh.load_fresh_games(allowlist)


def test_fresh_loader_refuses_mutated_summary(tmp_path):
    allowlist = _allowlist(tmp_path)
    (tmp_path / "classic/games.json").write_text("{}")
    with pytest.raises(ValueError, match="summary pin"):
        fresh.load_fresh_games(allowlist)


def test_even_macro_and_bootstrap_keep_equal_mode_weight():
    reference, candidate = [], []
    for mode, count, delta in (("classic", 100, 1.0), ("double", 1, -0.5)):
        for index in range(8):
            row = {
                "game_identity": f"{mode}-{index}",
                "cell": "pie-" + mode,
                "samples": count,
                "losses": {"policy": 2.0, "value": 1.0, "composite": 3.0},
            }
            reference.append(row)
            candidate.append(
                {
                    **row,
                    "losses": {
                        "policy": 2.0 - delta,
                        "value": 1.0,
                        "composite": 3.0 - delta,
                    },
                }
            )
    assert fresh.aggregate(candidate)["even_macro"]["composite"] == pytest.approx(2.75)
    comparison = fresh.paired_even(reference, candidate, alpha=0.05, repeats=100)
    assert comparison["weighted_composite_improvement"] == pytest.approx(0.25)
    assert comparison["one_sided_lower_bound"] == pytest.approx(0.25)
    candidate[0]["samples"] += 1
    with pytest.raises(ValueError, match="counts differ"):
        fresh.paired_even(reference, candidate, alpha=0.05, repeats=100)


def test_fresh_evaluator_scores_raw_ema_and_reuses_only_pinned_results(tmp_path):
    from scripts.run_frozen_replay_optimizer_calibration import run_calibration

    base = tmp_path / "old"
    base.mkdir()
    settings, _, _, _, _ = _calibration_fixture(base)
    run_calibration(settings)
    identity = load_model_manifest(settings.champion).model_identity
    allowlist = _allowlist(tmp_path / "fresh", teacher=identity)
    calibration_result = settings.output_dir / "result.json"
    old_result_bytes = calibration_result.read_bytes()
    kwargs = dict(
        allowlist=allowlist,
        config_path=settings.config,
        champion=settings.champion,
        calibration_results=[calibration_result],
        output_dir=tmp_path / "validation",
        device_name="cpu",
        batch_size=8,
        budget_seconds=120.0,
    )
    result = fresh.run(**kwargs)
    assert result["status"] == "complete"
    assert result["training_performed"] is False
    assert len(result["models"]) == 3
    assert result["contract"]["dataset"]["games"] == 16
    assert (
        fresh.run(**kwargs)["comparisons_against_champion"]
        == result["comparisons_against_champion"]
    )
    assert calibration_result.read_bytes() == old_result_bytes
    cached = kwargs["output_dir"] / "champion-ema.json"
    altered = json.loads(cached.read_text())
    altered["even_macro"]["composite"] += 1
    cached.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="cached fresh model result"):
        fresh.run(**kwargs)
