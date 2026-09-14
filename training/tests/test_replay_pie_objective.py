"""Pie-only even training excludes legacy variants throughout replay and credit."""

from collections import Counter
from dataclasses import replace
import sqlite3

import pytest

from startrain.config import LearnerConfig, RingMixtureConfig, RingWeightStage
from startrain.learner import (
    LazyShardReplayDataset,
    LearnerLoop,
    UniqueReplayBatchSampler,
    UTDSegmentState,
)
from startrain.replay_store import (
    ReplayStore,
    training_committed_sample_count,
    _strict_variant_targets,
)
from startrain.variant_training import training_segment_quotas
from test_pipeline_core import make_test_learner, run_identity
from test_replay_game_revisions import _samples, publication as publication
from test_replay_six_mode_selection import append_mode, mode_counts

WEIGHTS = {4: 0.05, 6: 0.05, 8: 0.05, 10: 0.85}
QUOTAS = {"standard": 0.0, "classic": 0.0, "pie": 0.9, "handicap": 0.1}


def options(identity):
    return dict(run_id=identity.run_id, generation_family=identity.generation_family)


def select(store, identity, *, rings=(4, 6, 8, 10), quota=170, **overrides):
    args = dict(
        rings=rings,
        per_ring_quota=quota,
        current_model_step=100,
        max_model_lag_steps=100,
        training_objective="ring10_pie",
        segment_quotas=QUOTAS,
        ring_segment_quotas={
            ring: training_segment_quotas(ring, WEIGHTS) for ring in rings
        },
        within_segment_classic_shares={"handicap": 0.5, "pie": 0.5},
    )
    args.update(overrides)
    return store.select_recent_spans(**options(identity), **args)


def credit(store, identity):
    return store.total_committed_sample_count(
        **options(identity), training_objective="ring10_pie"
    )


def test_mixed_legacy_replay_uses_board_conditional_mix_and_equal_modes(tmp_path):
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        for ring in WEIGHTS:
            for mode in ("classic", "double"):
                append_mode(store, identity, mode, 100, pie=True, ring=ring)
                append_mode(store, identity, mode, 100, handicap=9, ring=ring)
                append_mode(store, identity, mode, 100, ring=ring)
        selection = select(store, identity)
        by_ring = {}
        for ring in WEIGHTS:
            by_ring[ring] = mode_counts(
                replace(
                    selection,
                    spans=tuple(s for s in selection.spans if s.record.ring == ring),
                )
            )
        for ring in (4, 6, 8):
            assert by_ring[ring] == {"pie-classic": 85, "pie-double": 85}
        assert by_ring[10] == {
            "pie-classic": 75,
            "pie-double": 75,
            "handicap-classic": 10,
            "handicap-double": 10,
        }
        # The board-weighted allocation is 90% even and 10% handicap globally.
        assert sum(
            WEIGHTS[r] * by_ring[r].get("handicap-classic", 0) / 170 for r in WEIGHTS
        ) == pytest.approx(0.05)
        ring10 = replace(
            selection,
            spans=tuple(span for span in selection.spans if span.record.ring == 10),
            samples_by_ring={10: 170},
        )
        dataset = LazyShardReplayDataset(
            ring10, seed=5, epoch=0, augmentation_enabled=False, shard_cache_size=4
        )
        sampler = UniqueReplayBatchSampler(
            dataset,
            batch_size=17,
            batches=10,
            seed=5,
            epoch=0,
            ring_stratified=True,
            shards_per_batch=4,
        )
        indices = [index for batch in sampler for index in batch]
        assert len(indices) == len(set(indices)) == 170
        rows = [dataset[index] for index in indices]
        assert Counter((row.segment, row.mode) for row in rows) == {
            ("pie", "classic"): 75,
            ("pie", "double"): 75,
            ("handicap", "classic"): 10,
            ("handicap", "double"): 10,
        }
        assert selection.committed_samples == credit(store, identity) == 1000
        counts = store.eligible_sample_counts(
            tuple(WEIGHTS),
            **options(identity),
            current_model_step=100,
            max_model_lag_steps=100,
            training_objective="ring10_pie",
        )
        assert counts == {4: 200, 6: 200, 8: 200, 10: 400}
        segments = store.eligible_sample_counts_by_segment(
            tuple(WEIGHTS),
            **options(identity),
            current_model_step=100,
            max_model_lag_steps=100,
            training_objective="ring10_pie",
        )
        assert segments[(4, "handicap")] == segments[(10, "standard")] == 0
        assert store.available_sample_count(**options(identity)) == 2400


def test_shortages_cannot_borrow_forbidden_data_even_without_quotas(tmp_path):
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        append_mode(store, identity, "classic", 3, pie=True)
        append_mode(store, identity, "double", 100, handicap=9)
        append_mode(store, identity, "double", 100)
        append_mode(store, identity, "classic", 4, handicap=9, ring=10)
        append_mode(store, identity, "double", 100, ring=10)
        for quota_overrides in (
            {},
            {
                "segment_quotas": None,
                "ring_segment_quotas": None,
                "within_segment_classic_shares": None,
            },
        ):
            selected = select(store, identity, **quota_overrides)
            if quota_overrides:
                assert mode_counts(selected) == {
                    "pie-classic": 3,
                    "handicap-classic": 4,
                }
                assert selected.samples_by_ring == {4: 3, 6: 0, 8: 0, 10: 4}
            else:
                assert selected.sample_count == 0
                counts = store.eligible_sample_counts(
                    tuple(WEIGHTS),
                    **options(identity),
                    current_model_step=100,
                    max_model_lag_steps=100,
                    training_objective="ring10_pie",
                    ring_segment_quotas={
                        ring: training_segment_quotas(ring, WEIGHTS) for ring in WEIGHTS
                    },
                )
                assert counts == dict.fromkeys(WEIGHTS, 0)


def test_scoped_credit_survives_gc_reopen_and_concurrent_actor(tmp_path):
    identity = run_identity(tmp_path)
    root = tmp_path / "replay"
    with ReplayStore(root) as store:
        store.lease_generation(identity, "actor-test")
        append_mode(store, identity, "classic", 3, pie=True)
        append_mode(store, identity, "double", 5, pie=True)
        append_mode(store, identity, "double", 20)
        assert credit(store, identity) == 8
        store.collect_garbage(
            **options(identity), retain_shards_per_ring=1, dry_run=False
        )
        assert credit(store, identity) == 8
        with ReplayStore(root) as actor:
            append_mode(actor, identity, "double", 7, pie=True)
            assert credit(store, identity) == 15
    with ReplayStore(root) as store:
        assert credit(store, identity) == 15
        assert store.total_committed_sample_count(**options(identity)) == 35


def test_readonly_bootstrap_matches_initialization_without_changing_old_manifest(
    tmp_path,
):
    identity = run_identity(tmp_path)
    root = tmp_path / "replay"
    with ReplayStore(root) as store:
        store.lease_generation(identity, "actor-test")
        append_mode(store, identity, "classic", 3, pie=True)
        append_mode(store, identity, "double", 20, handicap=9)
        store.connection.execute("DROP TRIGGER ring10_pie_fresh_credit")
        store.connection.execute("DROP TABLE training_objective_counters")
        manifest = store.manifest_path
    with sqlite3.connect(f"file:{manifest}?mode=ro", uri=True) as connection:
        assert (
            training_committed_sample_count(
                connection, **options(identity), training_objective="ring10_pie"
            )
            == 3
        )
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name='training_objective_counters'"
            ).fetchone()
            is None
        )
    with ReplayStore(root) as store:
        assert credit(store, identity) == 3


def test_live_policy_prefix_and_enrichment_credit_only_allowed_new_positions(
    publication,
):
    store, identity, generation, kwargs = publication

    def samples(count, *, final=False, pie=True, game="allowed"):
        return [
            replace(s, mode="classic", pie=pie)
            for s in _samples(identity, generation, count, final=final, game=game)
        ]

    store.append_game_revision(samples(2), finalized=False, **kwargs)
    store.append_game_revision(samples(3), finalized=False, **kwargs)
    assert credit(store, identity) == 3
    before = select(
        store,
        identity,
        rings=(4,),
        quota=20,
        within_segment_classic_shares={"pie": 1.0},
    )
    assert before.sample_count == 3
    store.append_game_revision(samples(3, final=True), finalized=True, **kwargs)
    after = select(
        store,
        identity,
        rings=(4,),
        quota=20,
        within_segment_classic_shares={"pie": 1.0},
    )
    assert before.committed_samples == after.committed_samples == 3
    assert after.publication_revision > before.publication_revision
    assert after.enriched_samples - before.enriched_samples == 3
    store.append_game_revision(
        samples(9, pie=False, game="forbidden"), finalized=False, **kwargs
    )
    assert credit(store, identity) == 3
    assert (
        select(
            store,
            identity,
            rings=(4,),
            quota=20,
            within_segment_classic_shares={"pie": 1.0},
        ).sample_count
        == 3
    )
    store.append_game_revision(samples(3, final=True), finalized=True, **kwargs)
    assert credit(store, identity) == 3
    # A schema-six bootstrap counts the latest logical prefix exactly once.
    store.connection.execute("DROP TRIGGER ring10_pie_fresh_credit")
    store.connection.execute("DROP TABLE training_objective_counters")
    assert (
        training_committed_sample_count(
            store.connection, **options(identity), training_objective="ring10_pie"
        )
        == 3
    )
    store._initialize_training_objective_counter()
    assert credit(store, identity) == 3


def test_learner_readiness_window_reuse_and_utd_share_objective(tmp_path):
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        append_mode(store, identity, "double", 20, step=0)
        learner = make_test_learner(
            store,
            identity,
            tmp_path / "learner",
            learner_config=LearnerConfig(
                minimum_replay_samples=2,
                recent_samples_per_ring=16,
                target_updates_per_new_sample=1.0,
            ),
            ring_mixture_config=RingMixtureConfig(
                step_weights=(RingWeightStage(0, tuple(WEIGHTS.values())),)
            ),
        )
        legacy = learner._rank_zero_select_replay_spans()
        window = learner._open_replay_window(
            legacy, batches=1, refresh_reason="initial"
        )
        learner.learner_config = replace(learner.learner_config, segment_quotas=QUOTAS)
        learner.serialized_config["orchestration"] = {
            "training_objective": "ring10_pie"
        }
        try:
            assert not learner._replay_is_ready(learner._eligible_replay_counts())
            assert learner._utd_step_budget() == 0
            assert (
                learner._window_refresh_reason(window, target=None)
                == "training_objective_change"
            )
            with pytest.raises(ValueError, match="excluded by the training objective"):
                learner._open_replay_window(legacy, batches=1, refresh_reason="reuse")
            append_mode(store, identity, "classic", 4, pie=True, step=0)
            assert not learner._replay_is_ready(learner._eligible_replay_counts())
            assert learner._rank_zero_select_replay_spans().sample_count == 0
            append_mode(store, identity, "double", 4, pie=True, step=0)
            assert learner._replay_is_ready(learner._eligible_replay_counts())
            assert learner._utd_step_budget() == 8
            assert learner._rank_zero_select_replay_spans().sample_count == 8
            assert learner._utd_segment_state.training_objective == "ring10_pie"
        finally:
            window.shutdown(strict=False)
            learner._shutdown_loader_pool()


def test_legacy_utd_segment_requires_explicit_scope_cutover(tmp_path):
    legacy = UTDSegmentState("run", "family", 1.0, 5, 10)
    assert "training_objective" not in legacy.as_dict()
    assert LearnerLoop._parse_utd_segment_state(legacy.as_dict()) == legacy
    learner = object.__new__(LearnerLoop)
    learner.serialized_config = {"orchestration": {"training_objective": "ring10_pie"}}
    learner.examples_consumed = 5
    with pytest.raises(ValueError, match="prepared scoped UTD segment"):
        learner._validate_utd_segment_boundary(legacy)
    scoped = replace(legacy, training_objective="ring10_pie")
    learner._validate_utd_segment_boundary(scoped)
    assert LearnerLoop._parse_utd_segment_state(scoped.as_dict()) == scoped


def test_scoped_gc_preserves_excluded_history_without_consuming_active_retention(
    publication,
):
    store, identity, generation, kwargs = publication

    def publish(game, count, *, pie, finalized):
        samples = [
            replace(sample, mode="classic", pie=pie)
            for sample in _samples(
                identity, generation, count, game=game, final=finalized
            )
        ]
        return store.append_game_revision(samples, finalized=finalized, **kwargs).record

    excluded_prefix = publish("excluded", 2, pie=False, finalized=False)
    excluded_final = publish("excluded", 3, pie=False, finalized=True)
    allowed_prefix = publish("allowed", 2, pie=True, finalized=False)
    allowed_final = publish("allowed", 3, pie=True, finalized=True)
    allowed_latest = publish("allowed-latest", 4, pie=True, finalized=True)
    before = credit(store, identity)
    arguments = dict(
        **options(identity), retain_shards_per_ring=1, training_objective="ring10_pie"
    )
    dry = store.collect_garbage(**arguments, dry_run=True)
    assert dry["candidate_shards"] == 2
    assert dry["preserved_excluded_shards"] == 2
    assert dry["preserved_excluded_payload_rows"] == 5
    assert dry["preserved_excluded_ready_samples"] == 3
    assert all(
        record.path.is_file()
        for record in (
            excluded_prefix,
            excluded_final,
            allowed_prefix,
            allowed_final,
            allowed_latest,
        )
    )
    applied = store.collect_garbage(**arguments, dry_run=False)
    assert applied["deleted_shards"] == 2
    assert excluded_prefix.path.is_file() and excluded_final.path.is_file()
    assert allowed_latest.path.is_file()
    assert not allowed_prefix.path.exists() and not allowed_final.path.exists()
    assert credit(store, identity) == before == 7
    assert (
        store.connection.execute(
            "SELECT state FROM shards WHERE id=?", (excluded_prefix.shard_id,)
        ).fetchone()[0]
        == "superseded"
    )
    assert mode_counts(
        select(store, identity, rings=(4,), within_segment_classic_shares={"pie": 1.0})
    ) == {"pie-classic": 4}
    assert store.collect_garbage(**arguments, dry_run=False)["candidate_shards"] == 0


def test_production_sized_shortage_caps_handicap_at_the_declared_fraction():
    capacities = {
        "pie": {"classic": 167_500, "double": 167_500},
        "handicap": {"classic": 167_500, "double": 167_500},
    }
    targets = _strict_variant_targets(
        1_000_000,
        training_segment_quotas(10, WEIGHTS),
        {"pie": 0.5, "handicap": 0.5},
        capacities,
    )
    assert targets == {
        ("pie", "classic"): 167_500,
        ("pie", "double"): 167_500,
        ("handicap", "classic"): 22_333,
        ("handicap", "double"): 22_333,
    }
    total = sum(targets.values())
    assert total == 379_666
    assert WEIGHTS[10] * (
        targets[("handicap", "classic")] + targets[("handicap", "double")]
    ) / total == pytest.approx(0.1, abs=2e-6)


def test_strict_window_capacity_matches_readiness_and_preserves_unused_rows(tmp_path):
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        for mode in ("classic", "double"):
            append_mode(store, identity, mode, 100, pie=True, ring=10)
            append_mode(store, identity, mode, 100, handicap=9, ring=10)
        selected = select(store, identity, rings=(10,), quota=1_000_000)
        assert mode_counts(selected) == {
            "pie-classic": 100,
            "pie-double": 100,
            "handicap-classic": 13,
            "handicap-double": 13,
        }
        counts = store.eligible_sample_counts(
            (10,),
            **options(identity),
            current_model_step=100,
            max_model_lag_steps=100,
            training_objective="ring10_pie",
            ring_segment_quotas={10: training_segment_quotas(10, WEIGHTS)},
        )
        assert counts == selected.samples_by_ring == {10: 226}
        assert store.available_sample_count(**options(identity)) == 400
        assert credit(store, identity) == 400
