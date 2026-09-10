"""Real native games feed the real learner before outcomes exist, exactly once."""

import json

import pytest
import torch

from startrain import learner as learner_module
from startrain.config import DataConfig, LearnerConfig, SchedulerConfig, TrainConfig
from startrain.replay import (
    TARGET_OUTCOME,
    TARGET_POLICY,
    TARGET_SOFT_POLICY,
    read_replay_shard,
)
from startrain.replay_store import ReplayStore
from startrain.runtime import RunIdentity
from startrain.selfplay import (
    PolicyPublicationConfig,
    SelfPlayActor,
    SelfPlayConfig,
    SelfPlayIdentity,
)
from test_pipeline_core import make_test_learner
from test_selfplay_streaming import Evaluator, VARIANTS

MODEL = "sha256-" + "1" * 64


class FixedEvaluator(Evaluator):
    model_identity = model_version = MODEL
    model_step = 0


def setup(tmp_path, variant=VARIANTS[1], *, games=2, rolling=False):
    native = pytest.importorskip("star_native")
    identity = RunIdentity(tmp_path / "run.json", "live-run", "live-family", 1)
    store = ReplayStore(tmp_path / "replay")
    generation = store.lease_generation(identity, "live-actor")
    config = SelfPlayConfig(
        rings=4,
        games=games,
        batch_size=2,
        stream_completed_games=True,
        rolling_game_slots=rolling,
        seed_contract="game-v1",
        mode=variant.mode,
        handicap=variant.handicap,
        pie=variant.pie,
        fast_probability=0.5,
        full_probability=0.5,
        fast_simulations=1,
        full_simulations=2,
        simulation_ring_exponent=0,
        max_considered=2,
        record_fast_policy_targets=True,
        policy_surprise_weight=0.5,
        policy_publication=PolicyPublicationConfig(True, 2, 8),
    )
    actor = SelfPlayActor(
        native,
        FixedEvaluator(),
        store,
        config,
        SelfPlayIdentity(
            identity.run_id, identity.generation_family, "live-actor", generation
        ),
        source_role="candidate",
    )
    return actor, store, identity


def selection(store, identity, *, step=0):
    return store.select_recent_spans(
        rings=[4],
        per_ring_quota=10000,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        current_model_step=step,
        max_model_lag_steps=100,
    )


def credit(store, identity):
    return store.total_committed_sample_count(
        run_id=identity.run_id, generation_family=identity.generation_family
    )


@pytest.mark.native
@pytest.mark.parametrize("variant", VARIANTS)
def test_live_prefixes_and_completed_games_preserve_original_rows_and_credit(
    tmp_path, variant
):
    actor, store, identity = setup(tmp_path, variant)
    publications = []

    def progress(**fields):
        if fields.get("phase") == "selfplay_policy_published":
            selected = selection(store, identity)
            if not actor.completed_games:
                assert credit(store, identity) == actor.persisted_decisions
                for span in selected.spans:
                    rows = read_replay_shard(span.record.path)
                    assert [s.ply for s in rows] == list(range(len(rows)))
                    assert all(
                        s.target_mask == TARGET_POLICY | TARGET_SOFT_POLICY
                        for s in rows
                    )
                    assert all(
                        ":final=pending-policy:" in s.search_provenance for s in rows
                    )
            publications.append(fields)

    try:
        summaries = actor.run(progress=progress)
        assert publications and publications[0]["completed_games"] == 0
        assert len(summaries) == 2
        count = sum(game.samples for game in summaries)
        assert credit(store, identity) == actor.persisted_decisions == count
        assert actor.completed_decisions == count
        assert (
            actor.source_candidate_samples == count
            and actor.source_candidate_games == 2
        )
        assert actor.policy_published_decisions == actor.enriched_decisions > 0
        assert actor.retained_incomplete_decisions == 0
        assert actor.replay_written_decisions > count
        selected = selection(store, identity)
        assert selected.sample_count == count and len(selected.spans) == 2
        for span in selected.spans:
            rows = read_replay_shard(span.record.path)
            assert all(s.target_mask & TARGET_OUTCOME for s in rows)
            assert all(":final=board-full:" in s.search_provenance for s in rows)
        assert (
            actor.replay_written_decisions
            == store.connection.execute(
                "SELECT SUM(sample_count) FROM shards"
            ).fetchone()[0]
        )
        assert store.connection.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 2
    finally:
        store.close()


@pytest.mark.native
def test_rolling_games_and_clean_stop_keep_published_prefixes_without_recredit(
    tmp_path,
):
    actor, store, identity = setup(tmp_path, games=7, rolling=True)
    try:
        summaries = actor.run(
            stop_requested=lambda: (
                actor.refilled_games > 0
                and actor.full_decisions + actor.fast_decisions
                > actor.completed_decisions
            )
        )
        m = actor.metrics_snapshot()
        assert summaries and m.dropped_games
        assert (
            m.completed_decisions + m.retained_incomplete_decisions
            == actor.persisted_decisions
        )
        assert credit(store, identity) == actor.persisted_decisions
        assert m.retained_incomplete_decisions >= m.salvaged_policy_decisions
        assert m.source_candidate_samples == actor.persisted_decisions
        assert m.source_candidate_games == len(summaries)
        assert m.policy_published_decisions > m.enriched_decisions
        assert all(
            s.target_mask & TARGET_POLICY
            for span in selection(store, identity).spans
            for s in read_replay_shard(span.record.path)
        )
    finally:
        store.close()


@pytest.mark.native
def test_real_learner_updates_on_policy_before_any_game_completes(
    tmp_path, monkeypatch
):
    actor, store, identity = setup(tmp_path)
    learner = make_test_learner(
        store,
        identity,
        tmp_path / "learner",
        learner_config=LearnerConfig(
            minimum_replay_samples=1,
            recent_samples_per_ring=128,
            steps_per_window=10,
            target_updates_per_new_sample=1.5,
            device="cpu",
            metrics_interval=1,
            replay_poll_seconds=0.001,
        ),
        train_config=TrainConfig(
            per_rank_batch_size=2,
            scheduler=SchedulerConfig(warmup_steps=0, total_steps=100),
        ),
        data_config=DataConfig(workers=0, d5_augmentation=False, ring_stratified=False),
    )
    metrics = []
    original = learner_module.train_step

    def train(*args, **kwargs):
        result = original(*args, **kwargs)
        metrics.append(
            {
                name: float(value.detach())
                for name, value in result.loss_tensors.items()
                if value.numel() == 1
            }
        )
        return result

    monkeypatch.setattr(learner_module, "train_step", train)
    trained = False

    def progress(**fields):
        nonlocal trained
        if fields.get("phase") == "selfplay_policy_published" and not trained:
            assert actor.completed_games == 0
            assert learner.run(steps=1) == 1
            assert metrics[-1]["policy"] > 0
            for name in ("outcome", "score_margin", "ownership", "alive"):
                assert metrics[-1][name] == 0
            trained = True

    try:
        summaries = actor.run(progress=progress)
        assert trained and summaries
        assert credit(store, identity) == sum(game.samples for game in summaries)
        assert learner.run(steps=1) == 2
        assert metrics[-1]["outcome"] > 0
        assert all(
            torch.isfinite(parameter).all() for parameter in learner.model.parameters()
        )
        assert learner.examples_consumed == 4
        state = json.loads((tmp_path / "learner/utd-segment.json").read_text())
        assert state["target_updates_per_new_sample"] == 1.5
    finally:
        learner._shutdown_loader_pool()
        store.close()


def test_policy_stream_requires_policy_targets_and_revision_aware_sink():
    with pytest.raises(ValueError, match="fast policy"):
        SelfPlayConfig(policy_publication=PolicyPublicationConfig(enabled=True))
