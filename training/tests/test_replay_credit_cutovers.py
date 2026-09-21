"""Operational cutovers preserve the same eligible-position credit as training."""

from dataclasses import replace
import hashlib
import json

import pytest
import yaml

from scripts import training_disaster_recovery as recovery
from scripts.fork_elo_ablation import _prepare_utd_segment
from scripts.prepare_champion_warm_start import (
    WarmStartError,
    prepare_champion_warm_start,
)
from scripts.prepare_pie_training_profile import pie_training_config
from deltreltrain.config import RingWeightStage
from deltreltrain.learner import LearnerLoop, UTDSegmentState
from deltreltrain.replay_store import ReplayStore
from test_replay_six_mode_selection import append_mode
from test_run_state_preflight import _fixture as warm_fixture
from test_training_disaster_recovery import _fixture as disaster_fixture, _snapshot


def pie_fixture(tmp_path):
    fixture = warm_fixture(tmp_path)
    source = fixture.experiment
    source = replace(
        source,
        data=replace(source.data, ring_stratified=True),
        orchestration=replace(
            source.orchestration,
            ring_mixture=replace(
                source.orchestration.ring_mixture,
                step_weights=(RingWeightStage(0, (0.05, 0.05, 0.05, 0.85)),),
            ),
        ),
    )
    config = pie_training_config(source)
    fixture.profile.write_text(
        yaml.safe_dump(json.loads(json.dumps(config.as_dict())), sort_keys=False)
    )
    with ReplayStore(fixture.root / "replay") as store:
        store.lease_generation(fixture.identity, "actor-test")
        append_mode(store, fixture.identity, "classic", 4, pie=True, step=0)
        append_mode(store, fixture.identity, "double", 20, step=0)
    # Model weights remain from the old profile; the stopped-run migration
    # establishes the new scoped boundary before any operational warm start.
    (fixture.root / "learner" / "utd-segment.json").write_text(
        json.dumps(
            UTDSegmentState(
                fixture.identity.run_id,
                fixture.identity.generation_family,
                1.0,
                100,
                4,
                "ring10_pie",
            ).as_dict()
        )
    )
    return fixture, config


def test_warm_start_checkpoint_and_initial_allowance_use_scoped_replay(tmp_path):
    fixture, _ = pie_fixture(tmp_path)
    with pytest.raises(WarmStartError, match="initial replay credit"):
        prepare_champion_warm_start(
            fixture.root, fixture.profile, initial_replay_credit=5
        )
    plan = prepare_champion_warm_start(fixture.root, fixture.profile, prepare_only=True)
    assert plan["committed_replay_samples"] == plan["initial_replay_credit"] == 4
    assert plan["utd_segment"]["training_objective"] == "ring10_pie"
    assert plan["utd_segment"]["baseline_committed_replay_samples"] == 0
    assert plan["training_segment"]["training_objective"] == "ring10_pie"
    assert plan["preflight"]["replay"]["committed_samples"] == 24
    marker = json.loads(
        (fixture.root / "learner" / "champion-warm-start.json").read_text()
    )
    assert marker["utd_segment"] == plan["utd_segment"]
    assert (
        LearnerLoop._parse_utd_segment_state(marker["utd_segment"]).training_objective
        == "ring10_pie"
    )


def test_ablation_fork_rebases_legacy_credit_and_is_idempotent(tmp_path):
    fixture, config = pie_fixture(tmp_path)
    segment_path = fixture.root / "learner" / "utd-segment.json"
    segment_path.write_text(
        json.dumps(
            UTDSegmentState(
                fixture.identity.run_id,
                fixture.identity.generation_family,
                1.0,
                100,
                24,
            ).as_dict()
        )
    )
    kwargs = dict(
        experiment=config,
        run_id=fixture.identity.run_id,
        generation_family=fixture.identity.generation_family,
    )
    segment = _prepare_utd_segment(fixture.root, **kwargs)
    assert segment["training_objective"] == "ring10_pie"
    assert segment["baseline_committed_replay_samples"] == 4
    assert segment["baseline_examples_consumed"] == 100
    assert _prepare_utd_segment(fixture.root, **kwargs) == segment
    assert json.loads(segment_path.read_text()) == segment


@pytest.mark.parametrize(
    "scope,baseline,error",
    [
        ("ring10_pie", 0, None),
        ("ring10_pie", 1, "UTD baseline is ahead of replay cutoff"),
        (None, 0, "training objective disagrees"),
    ],
)
def test_disaster_snapshot_checks_scoped_credit_not_global_total(
    tmp_path, scope, baseline, error
):
    fixture = disaster_fixture(tmp_path)
    profile = yaml.safe_load(fixture.profile.read_text())
    profile["orchestration"]["training_objective"] = "ring10_pie"
    fixture.profile.write_text(yaml.safe_dump(profile))
    digest = hashlib.sha256(fixture.profile.read_bytes()).hexdigest()
    (fixture.root / "profile.sha256").write_text(f"{digest}  {fixture.profile.name}\n")
    path = fixture.root / "learner" / "utd-segment.json"
    utd = json.loads(path.read_text())
    utd["baseline_committed_replay_samples"] = baseline
    if scope is not None:
        utd["training_objective"] = scope
    path.write_text(json.dumps(utd))
    # The fixture has four aggregate positions, all legacy no-pie even games.
    if error:
        with pytest.raises(recovery.DisasterRecoveryError, match=error):
            _snapshot(fixture, tmp_path / "backup")
    else:
        snapshot = _snapshot(fixture, tmp_path / "backup")
        assert recovery.verify_snapshot(snapshot)["status"] == "ok"


def test_scoped_utd_metrics_do_not_mix_historical_examples_with_filtered_credit():
    learner = object.__new__(LearnerLoop)
    learner.serialized_config = {"orchestration": {"training_objective": "ring10_pie"}}
    learner.examples_consumed = 102
    learner._latest_total_replay_samples = 8
    learner._utd_segment_state = UTDSegmentState(
        "run", "family", 1.0, 100, 4, "ring10_pie"
    )
    values = learner._utd_metric_values()
    assert values["lifetime_updates_per_new_sample"] is None
    assert values["updates_per_new_sample"] is None
    assert values["segment_updates_per_new_sample"] == 0.5
    assert values["utd_segment_training_objective"] == "ring10_pie"
