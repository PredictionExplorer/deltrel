"""Fault injection for durable adaptive decisions and asynchronous completion."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import time

import pytest
import torch

import startrain.promotion as promotion_module
from startrain.adaptive_promotion import next_allocation
from startrain.arena import ArenaGame, summarize_completed_arena_pairs
from startrain.balanced_evaluation import evaluation_contract
from startrain.checkpoint import collect_model_garbage, write_model_pointer
from scripts.training_disaster_recovery import _manifest_references
from test_adaptive_promotion import _case, pairs_for


def allocation_path(case):
    result = case.supervisor._result_path(case.candidate, case.champion)
    return result.with_name(f"{result.stem}.allocation.json")


def save_resume_pairs(case, cfg, pairs):
    """Supply structurally valid completed seats, isolating persistence logic.

    Native replay/winner proof and interruption behavior have separate runner
    tests; these fixtures do not perform or qualify playing-strength searches.
    """
    entries = []
    for pair in pairs:
        for seat, outcome in enumerate(pair.outcomes):
            game = ArenaGame(
                ring=pair.ring,
                pair=pair.pair,
                candidate_player=seat,
                opening_seed=pair.opening_seed,
                opening_action=pair.opening_action,
                forced_opening=pair.forced_opening,
                winner=seat if outcome == 1 else 1 - seat,
                outcome=outcome,
                searched_moves=0,
                variant=pair.variant,
                segment=pair.segment,
            )
            entries.append(
                {
                    "ring": game.ring,
                    "variant": game.variant,
                    "pair": game.pair,
                    "candidate_player": seat,
                    "opening_seed": game.opening_seed,
                    "opening_action": game.opening_action,
                    "actions": [],
                    "result": asdict(game),
                }
            )
    state = {
        "config": asdict(cfg),
        "candidate": case.candidate.model_version,
        "baseline": case.champion.model_version,
        "game_states": entries,
        "pairs": [asdict(pair) for pair in pairs],
    }
    subject = case.supervisor
    subject._resume_writer(
        subject._result_path(case.candidate, case.champion),
        case.candidate,
        case.champion,
    )(state)
    return state


@pytest.mark.parametrize(
    "fault",
    [
        "schema_boolean",
        "index_boolean",
        "extra_boolean",
        "extra_negative",
        "extra_field",
        "target_boolean",
        "target_budget",
        "summary",
        "evidence_hash",
        "contract",
    ],
)
def test_corrupt_allocation_is_rejected_without_rewriting_it(
    tmp_path, monkeypatch, fault
):
    case = _case(tmp_path, monkeypatch)
    subject = case.supervisor
    subject._adaptive_allocation(case.candidate, case.champion, [])
    path = allocation_path(case)
    payload = json.loads(path.read_text())
    record = payload["plan_history"][0]
    if fault == "schema_boolean":
        payload["schema_version"] = True
    elif fault == "index_boolean":
        record["plan_index"] = False
    elif fault == "extra_boolean":
        record["extra_handicap_pairs_total"] = False
    elif fault == "extra_negative":
        record["extra_handicap_pairs_total"] = -1
    elif fault == "extra_field":
        payload["ignore_contract"] = True
    elif fault == "target_boolean":
        record["cell_pair_targets"]["r10/classic-pie"] = True
    elif fault == "target_budget":
        record["cell_pair_targets"]["r10/classic-pie"] = 161
    elif fault == "summary":
        record["decision_summary"]["promotion"]["decision"] = "promote"
    elif fault == "evidence_hash":
        record["decision_pairs_sha256"] = "0" * 64
    else:
        payload["evaluation_contract"]["identity"] = "sha256-" + "0" * 64
    encoded = json.dumps(payload, sort_keys=True)
    path.write_text(encoded)
    with pytest.raises(ValueError, match="adaptive"):
        subject._adaptive_allocation(case.candidate, case.champion, [])
    assert path.read_text() == encoded


def test_recomputed_forged_plan_cannot_extend_an_incomplete_previous_plan(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch)
    subject = case.supervisor
    cfg = subject._arena_config(case.candidate, case.champion)
    first = subject._adaptive_allocation(case.candidate, case.champion, [])
    partial = pairs_for(cfg, first["cell_pair_targets"])[:1]
    summary = summarize_completed_arena_pairs(partial, cfg)
    forged = next_allocation(cfg, previous_plan=first, summary=summary)
    evidence = [asdict(pair) for pair in partial]
    forged.update(
        {
            "plan_index": 1,
            "decision_summary": summary,
            "decision_pairs": evidence,
            "decision_pairs_sha256": hashlib.sha256(
                json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    )
    path = allocation_path(case)
    payload = json.loads(path.read_text())
    payload["plan_history"].append(forged)
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="adaptive"):
        subject._adaptive_allocation(case.candidate, case.champion, partial)
    assert path.read_bytes() == before


def test_plan_commit_before_result_commit_recovers_from_completed_resume_seats(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch)
    subject = case.supervisor
    cfg = subject._arena_config(case.candidate, case.champion)
    first = subject._adaptive_allocation(case.candidate, case.champion, [])
    completed = pairs_for(cfg, first["cell_pair_targets"])
    save_resume_pairs(case, cfg, completed)
    second = subject._adaptive_allocation(case.candidate, case.champion, completed)
    path = allocation_path(case)
    before = path.read_bytes()
    # The allocation is durable but the top-level result is still the empty
    # prior wave. Its completed evidence remains durable in the resume sidecar.
    resumed = subject._adaptive_allocation(case.candidate, case.champion, [])
    assert resumed == second
    assert path.read_bytes() == before


def test_oversized_extension_cannot_replace_a_readable_committed_ledger(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch)
    subject = case.supervisor
    cfg = subject._arena_config(case.candidate, case.champion)
    first = subject._adaptive_allocation(case.candidate, case.champion, [])
    completed = pairs_for(cfg, first["cell_pair_targets"])
    path = allocation_path(case)
    before = path.read_bytes()
    monkeypatch.setattr(
        promotion_module, "_MAX_ADAPTIVE_ALLOCATION_BYTES", len(before) + 1
    )
    with pytest.raises(ValueError, match="adaptive.*size|adaptive.*limit"):
        subject._adaptive_allocation(case.candidate, case.champion, completed)
    assert path.read_bytes() == before


def test_allocation_alone_protects_its_models_from_gc_and_exposes_backup_references(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch)
    subject = case.supervisor
    subject._adaptive_allocation(case.candidate, case.champion, [])
    ledger = json.loads(allocation_path(case).read_text())
    expected = {
        str((manifest.artifact_manifest or manifest.path).resolve())
        for manifest in (case.candidate, case.champion)
    }
    assert set(_manifest_references(ledger)) == expected
    # Simulate newer learner outputs moving the ordinary pointers and retention
    # window on before the allocated evaluation has written its first result.
    with torch.no_grad():
        next(case.model.parameters()).add_(0.01)
    case.ema.update(case.model)
    newest = case.publisher.publish(
        model=case.model,
        optimizer=case.optimizer,
        scheduler=case.scheduler,
        ema=case.ema,
        step=2,
        epoch=0,
        config=case.experiment.as_dict(),
    )
    write_model_pointer(
        case.publisher.champion_path, newest, role="champion", promotion_result="test"
    )
    without_ledger = collect_model_garbage(
        case.publisher.root, retain_candidate_manifests=1, dry_run=True
    )
    assert without_ledger["candidate_manifests"] >= 2
    protected = collect_model_garbage(
        case.publisher.root,
        retain_candidate_manifests=1,
        dry_run=True,
        referenced_result_directory=subject.results_directory,
    )
    assert protected["candidate_manifests"] == 0
    assert protected["candidate_checkpoints"] == 0


@pytest.mark.parametrize("incomplete", [False, True])
def test_resume_games_without_their_committed_plan_fail_closed(
    tmp_path, monkeypatch, incomplete
):
    case = _case(tmp_path, monkeypatch)
    subject = case.supervisor
    cfg = subject._arena_config(case.candidate, case.champion)
    first = subject._adaptive_allocation(case.candidate, case.champion, [])
    completed = pairs_for(cfg, first["cell_pair_targets"])[:1]
    state = save_resume_pairs(case, cfg, completed)
    if incomplete:
        state["game_states"] = state["game_states"][:1]
        state["game_states"][0]["result"] = None
        state["pairs"] = []
        subject._resume_writer(
            subject._result_path(case.candidate, case.champion),
            case.candidate,
            case.champion,
        )(state)
    path = allocation_path(case)
    path.unlink()
    with pytest.raises(ValueError, match="adaptive"):
        subject._adaptive_allocation(case.candidate, case.champion, [])
    assert not path.exists()


def test_uncommitted_pending_seat_cannot_be_absorbed_by_a_new_plan(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch)
    subject = case.supervisor
    cfg = subject._arena_config(case.candidate, case.champion)
    first = subject._adaptive_allocation(case.candidate, case.champion, [])
    completed = pairs_for(cfg, first["cell_pair_targets"])
    state = save_resume_pairs(case, cfg, completed)
    ghost = deepcopy(state["game_states"][0])
    ghost["pair"] = first["cell_pair_targets"]["r10/classic-pie"]
    ghost["result"] = None
    state["game_states"].append(ghost)
    subject._resume_writer(
        subject._result_path(case.candidate, case.champion),
        case.candidate,
        case.champion,
    )(state)
    path = allocation_path(case)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="adaptive"):
        subject._adaptive_allocation(case.candidate, case.champion, completed)
    assert path.read_bytes() == before


@pytest.mark.parametrize("verdict", ["promote", "reject"])
def test_global_verdict_cannot_stop_an_incomplete_committed_allocation(
    tmp_path, monkeypatch, verdict
):
    case = _case(tmp_path, monkeypatch)
    subject = case.supervisor
    cfg = subject._arena_config(case.candidate, case.champion)
    plan = subject._adaptive_allocation(case.candidate, case.champion, [])
    partial = pairs_for(cfg, plan["cell_pair_targets"])[:1]
    observed_summary = summarize_completed_arena_pairs(partial, cfg)
    observed_summary["promotion"]["decision"] = verdict
    monkeypatch.setattr(
        promotion_module,
        "summarize_completed_arena_pairs",
        lambda *_args, **_kwargs: deepcopy(observed_summary),
    )
    monkeypatch.setattr(
        subject,
        "_adaptive_allocation",
        lambda *_args, **_kwargs: pytest.fail(
            "partial global verdict tried to extend the committed plan"
        ),
    )
    monkeypatch.setattr(
        promotion_module,
        "write_model_pointer",
        lambda *_args, **_kwargs: pytest.fail("partial allocation promoted a model"),
    )
    result = {
        "pairs": [asdict(pair) for pair in partial],
        "games": [],
        "evaluation_contract": evaluation_contract(cfg),
        **deepcopy(observed_summary),
    }
    decision, terminal = subject._persist_wave(
        candidate=case.candidate,
        champion=case.champion,
        previous=None,
        accumulated=[],
        result=result,
        arena_config=cfg,
        round_started=time.perf_counter(),
        metric_device=torch.device("cpu"),
        collect_cuda_metrics=False,
        progress=None,
        wave_index=0,
        pair_starts={10: 0},
        pair_counts={10: 4},
        cell_pair_targets=plan["cell_pair_targets"],
        allocation_plan=plan,
    )
    assert decision == "continue" and not terminal
    assert result["promotion"]["decision"] == "continue"
    assert not result["terminal"]
