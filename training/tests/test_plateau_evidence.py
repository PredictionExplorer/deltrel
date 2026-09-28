from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from startrain.checkpoint import (
    ExponentialMovingAverage,
    load_checkpoint,
    save_checkpoint,
)
from startrain.config import PlateauConfig, SchedulerConfig
from startrain.lr_governor import LearningRateGovernorState
from startrain.plateau_evidence import (
    PLATEAU_RECEIPTS_KEY,
    PlateauVerdict,
    PlateauVerdictStore,
    receipt_for_recovery,
    receipts_from_checkpoint_extra,
    record_verdict,
)
from startrain.training import build_scheduler


def verdict(step: int, **changes: object) -> PlateauVerdict:
    return replace(
        PlateauVerdict(
            candidate_identity=f"candidate-{step}",
            candidate_step=step,
            champion_identity="champion",
            champion_step=100,
            evaluation_contract_identity="contract",
            decision="reject",
            completed_ns=step,
        ),
        **changes,
    )


def pending(store: PlateauVerdictStore, **changes: object):
    args = dict(
        champion_identity="champion",
        champion_step=100,
        contract_identity="contract",
        learner_step=1_000,
        receipts=None,
        required_rejections=2,
        count_inconclusive=False,
        poll_seconds=0,
    )
    args.update(changes)
    return store.pending_recovery(**args)


def test_verdict_survives_status_replacement_and_duplicate_write(tmp_path):
    first, second = verdict(200), verdict(300)
    record_verdict(tmp_path, first)
    record_verdict(tmp_path, replace(first, completed_ns=900))
    record_verdict(tmp_path, second)
    (tmp_path / "promotion-status.json").write_text(
        json.dumps({"decision": "continue", "candidate_identity": "newer-candidate"})
    )
    evidence = pending(PlateauVerdictStore(tmp_path))
    assert evidence["verdict_ids"] == [first.identity, second.identity]
    assert evidence["candidate_identity"] == second.candidate_identity
    assert len(list((tmp_path / "plateau-verdicts").glob("*.json"))) == 2


def test_conflicting_durable_verdict_is_rejected(tmp_path):
    item = verdict(200)
    record_verdict(tmp_path, item)
    with pytest.raises(ValueError, match="persisted plateau verdict"):
        record_verdict(tmp_path, replace(item, candidate_step=201))


@pytest.mark.parametrize(
    "changes",
    [
        {"champion_identity": "old-champion"},
        {"champion_step": 99},
        {"evaluation_contract_identity": "retired-contract"},
        {"candidate_step": 99},
        {"candidate_step": 1_001},
        {"decision": "reject_max_pairs"},
    ],
)
def test_scope_and_eligible_candidate_boundaries(tmp_path, changes):
    record_verdict(tmp_path, verdict(200))
    record_verdict(tmp_path, verdict(300, **changes))
    assert pending(PlateauVerdictStore(tmp_path)) is None


def test_inconclusive_rejections_only_count_when_enabled(tmp_path):
    record_verdict(tmp_path, verdict(200, decision="reject_max_pairs"))
    record_verdict(tmp_path, verdict(300, decision="reject_max_pairs"))
    store = PlateauVerdictStore(tmp_path)
    assert pending(store) is None
    assert len(pending(store, count_inconclusive=True)["verdict_ids"]) == 2


def test_backfill_validates_legacy_result_manifests_and_excludes_crossplay(tmp_path):
    for step in (200, 300):
        item = verdict(step)
        manifests = {}
        for role in ("candidate", "champion"):
            manifest = tmp_path / f"{role}-{step}.json"
            manifest.write_text(
                json.dumps(
                    {
                        "model_identity": getattr(item, f"{role}_identity"),
                        "model_step": getattr(item, f"{role}_step"),
                    }
                )
            )
            manifests[f"{role}_manifest"] = str(manifest)
        result = {
            "result_kind": "promotion",
            "candidate": item.candidate_identity,
            "baseline": item.champion_identity,
            "evaluation_contract": {"identity": "contract"},
            "completed_ns": item.completed_ns,
            "terminal": True,
            "promotion": {"decision": "reject"},
            **manifests,
        }
        (tmp_path / f"candidate-{step}-vs-champion.json").write_text(json.dumps(result))
        (tmp_path / f"crossplay-{step}-vs-champion.json").write_text(
            json.dumps({**result, "result_kind": "historical_crossplay"})
        )
    evidence = pending(PlateauVerdictStore(tmp_path))
    assert evidence["candidate_step"] == 300
    assert len(evidence["verdict_ids"]) == 2
    # Backfilled compact records survive removal of old bulky arena artifacts.
    for path in tmp_path.glob("*-vs-*.json"):
        path.unlink()
    assert pending(PlateauVerdictStore(tmp_path)) == evidence


def test_mismatched_legacy_manifest_is_not_evidence(tmp_path):
    item = verdict(200)
    wrong_manifest = tmp_path / "wrong.json"
    wrong_manifest.write_text(
        json.dumps({"model_identity": "other", "model_step": 200})
    )
    (tmp_path / "candidate-vs-champion.json").write_text(
        json.dumps(
            {
                "result_kind": "promotion",
                "candidate": item.candidate_identity,
                "baseline": item.champion_identity,
                "candidate_manifest": str(wrong_manifest),
                "champion_step": 100,
                "evaluation_contract": {"identity": "contract"},
                "completed_ns": 200,
                "terminal": True,
                "promotion": {"decision": "reject"},
            }
        )
    )
    assert pending(PlateauVerdictStore(tmp_path), required_rejections=1) is None


def test_receipt_deduplicates_backlog_restart_and_late_old_candidates(
    tmp_path, monkeypatch
):
    for step in (200, 300, 400):
        record_verdict(tmp_path, verdict(step))
    evidence = pending(PlateauVerdictStore(tmp_path))
    monkeypatch.setattr("startrain.plateau_evidence.time.time_ns", lambda: 1_000)
    receipts = receipt_for_recovery(None, evidence, learner_step=500)
    receipts = receipts_from_checkpoint_extra(
        json.loads(json.dumps({PLATEAU_RECEIPTS_KEY: receipts}))
    )
    restarted = PlateauVerdictStore(tmp_path)
    assert pending(restarted, receipts=receipts) is None
    # A late verdict for weights trained before the cut is not fresh evidence.
    record_verdict(tmp_path, verdict(450, completed_ns=2_000))
    assert pending(restarted, receipts=receipts) is None
    record_verdict(tmp_path, verdict(600, completed_ns=2_100))
    assert pending(restarted, receipts=receipts) is None
    record_verdict(tmp_path, verdict(700, completed_ns=2_200))
    fresh = pending(restarted, receipts=receipts)
    assert fresh["verdict_ids"] == [verdict(600).identity, verdict(700).identity]


def test_new_champion_or_contract_does_not_reuse_old_receipts(tmp_path, monkeypatch):
    for step in (200, 300):
        record_verdict(tmp_path, verdict(step))
    evidence = pending(PlateauVerdictStore(tmp_path))
    monkeypatch.setattr("startrain.plateau_evidence.time.time_ns", lambda: 1_000)
    receipts = receipt_for_recovery(None, evidence, learner_step=500)
    for step in (600, 700):
        record_verdict(
            tmp_path,
            verdict(step, champion_identity="new-champion", completed_ns=2_000 + step),
        )
    assert (
        pending(
            PlateauVerdictStore(tmp_path),
            receipts=receipts,
            champion_identity="new-champion",
        )
        is not None
    )
    assert (
        pending(
            PlateauVerdictStore(tmp_path),
            receipts=receipts,
            contract_identity="new-contract",
        )
        is None
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"schema_version": 2, "scopes": {}},
        {"schema_version": 1, "scopes": {"bad": {}}},
    ],
)
def test_corrupt_receipt_metadata_fails_closed(payload):
    with pytest.raises(ValueError, match="plateau verdict receipt"):
        receipts_from_checkpoint_extra({PLATEAU_RECEIPTS_KEY: payload})


def test_recovery_checkpoint_couples_receipt_and_rate_without_replacing_live_arena(
    tmp_path, monkeypatch
):
    from test_pipeline_core import _plateau_policy_fixture

    learner, status, status_path = _plateau_policy_fixture(tmp_path, monkeypatch)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler_config = SchedulerConfig(warmup_steps=0, total_steps=100)
    scheduler = build_scheduler(optimizer, scheduler_config)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    weights_before = copy.deepcopy(model.state_dict())
    learning_rate_before = optimizer.param_groups[0]["lr"]
    learner.optimizer = optimizer
    learner.scheduler = scheduler
    learner.gradient_clipper = None
    learner._lr_governor = LearningRateGovernorState.from_scheduler(scheduler)
    learner.step = 180_000
    learner.rank = 0
    learner._last_recovery_step = 0
    learner._broadcast_object = lambda value: value
    learner._distributed_barrier = lambda: None
    learner.metrics = SimpleNamespace(append=lambda _event: None)
    configured = PlateauConfig(
        enabled=True,
        action="reduce_lr_keep_weights",
        reset_learning_rate_scale=0.5,
        consecutive_terminal_rejections=2,
        poll_seconds=0.01,
    )
    learner._plateau_config = lambda: configured
    learner.expected_promotion_contract_identity = "contract"
    for step in (120_000, 130_000, 140_000):
        record_verdict(
            status_path.parent,
            verdict(step, champion_identity="sha256-champion", champion_step=100_000),
        )
    status(decision="continue", terminal=False, consecutive_conclusive_rejections=3)
    original_status = status_path.read_bytes()
    checkpoint = tmp_path / "recovery.pt"

    def save_recovery(**_kwargs):
        save_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ExponentialMovingAverage(model, decay=0.9),
            step=learner.step,
            extra=learner._checkpoint_extra(),
        )
        return SimpleNamespace(checkpoint=checkpoint)

    learner._maybe_write_recovery_checkpoint = save_recovery
    assert learner._plateau_control(stop_requested=lambda: False, progress=None)
    assert learner._lr_governor.multiplier == 0.5
    assert optimizer.param_groups[0]["lr"] == pytest.approx(learning_rate_before * 0.5)
    assert not optimizer.state
    assert status_path.read_bytes() == original_status
    assert not (tmp_path / "learner" / "resume-cutover.json").exists()
    for key, weight in weights_before.items():
        torch.testing.assert_close(model.state_dict()[key], weight)

    fresh_model = torch.nn.Linear(2, 1)
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=0.1)
    fresh_scheduler = build_scheduler(fresh_optimizer, scheduler_config)
    metadata = load_checkpoint(
        checkpoint,
        model=fresh_model,
        optimizer=fresh_optimizer,
        scheduler=fresh_scheduler,
        ema=ExponentialMovingAverage(fresh_model, decay=0.9),
    )
    restarted = copy.copy(learner)
    restarted.optimizer = fresh_optimizer
    restarted.scheduler = fresh_scheduler
    restarted._last_plateau_reset = None
    del restarted._plateau_verdict_store
    restarted._adopt_checkpoint_governor(metadata["extra"])
    assert restarted._lr_governor.multiplier == 0.5
    # Even a terminal mutable status with the old accumulated streak cannot
    # bypass the receipt and cause a second cut after restart.
    status(decision="reject", terminal=True, consecutive_conclusive_rejections=4)
    assert restarted._rank_zero_plateau_action(configured) == {"kind": "proceed"}
    assert (
        len(
            next(iter(restarted._plateau_verdict_receipts["scopes"].values()))[
                "consumed_verdict_ids"
            ]
        )
        == 3
    )


def test_promoter_records_once_before_status_changes_to_new_candidate(tmp_path):
    from test_promotion_contract_status import experiment
    from startrain.promotion import PromotionSupervisor

    supervisor = object.__new__(PromotionSupervisor)
    supervisor.experiment = experiment()
    supervisor.status_path = tmp_path / "promotion-status.json"
    supervisor._resume_cutover = lambda: None
    champion = SimpleNamespace(model_identity="champion", model_step=100)
    for step in (200, 300):
        candidate = SimpleNamespace(model_identity=f"candidate-{step}", model_step=step)
        for _ in range(2):
            supervisor._write_status(
                candidate=candidate, champion=champion, decision="reject", terminal=True
            )
    supervisor._write_status(
        candidate=SimpleNamespace(model_identity="candidate-400", model_step=400),
        champion=champion,
        decision="continue",
        terminal=False,
    )
    contract = json.loads(supervisor.status_path.read_text())[
        "evaluation_contract_identity"
    ]
    assert (
        len(
            pending(PlateauVerdictStore(tmp_path), contract_identity=contract)[
                "verdict_ids"
            ]
        )
        == 2
    )
