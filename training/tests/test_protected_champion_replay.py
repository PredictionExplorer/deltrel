"""A stale champion can teach a bounded share without refreshing old data."""

from collections import Counter
from dataclasses import replace
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from startrain.actor import ActorSupervisor
from startrain.checkpoint import write_model_pointer
from startrain.config import ConfigError, LearnerConfig, load_config
from startrain.learner import (
    LazyShardReplayDataset,
    PersistentReplayWorkerDataset,
    replay_selection_diagnostics,
)
from startrain.protected_replay import (
    ProtectedChampionReplay,
    replay_commit_eligibility_metrics,
)
from startrain.replay import read_replay_shard
from startrain.replay_store import ReplayStore, validate_game_publications
from test_pipeline_core import make_replay_sample, make_test_learner, run_identity
from test_replay_game_revisions import _samples, publication as publication
from test_replay_pie_objective import select as select_pie

NOW = 2_000_000_000_000_000_000
CHAMPION = "sha256-" + "1" * 64
CANDIDATE = "sha256-" + "2" * 64
OTHER = "sha256-" + "3" * 64


def protection(identity=CHAMPION, *, step=10, minimum=NOW - 10**12):
    return ProtectedChampionReplay(identity, step, minimum, NOW, 0.25)


def append(
    store,
    identity,
    count,
    *,
    model=CHAMPION,
    step=10,
    ring=4,
    mode="double",
    pie=False,
    handicap=1,
):
    serial = store.connection.execute(
        "SELECT COALESCE(MAX(id),0) FROM shards"
    ).fetchone()[0]
    samples = [
        replace(
            make_replay_sample(
                ring,
                identity=identity,
                game_id=f"protected-{serial}-{i}",
                model_identity=model,
            ),
            mode=mode,
            pie=pie,
            handicap=handicap,
            moves_left=handicap,
        )
        for i in range(count)
    ]
    return store.append(
        samples,
        phase_min=0,
        phase_max=0,
        model_version=model,
        model_identity=model,
        model_step=step,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        actor_id="actor-test",
        generation=0,
    )


def select(store, identity, *, protected=None, quota=40, **kwargs):
    return store.select_recent_spans(
        rings=(4,),
        per_ring_quota=quota,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        current_model_step=100,
        max_model_lag_steps=20,
        protected_champion=protected,
        **kwargs,
    )


def source_counts(selection):
    counts = Counter()
    for span in selection.spans:
        counts[span.record.model_identity] += span.sample_count
    return counts


def test_only_new_current_champion_is_admitted_and_credit_does_not_change(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("startrain.replay_store.time.time_ns", lambda: NOW)
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        append(store, identity, 60, model=CANDIDATE, step=95)
        champ = append(store, identity, 30)
        append(store, identity, 30, model=OTHER)
        # A mismatched step on the same identity is also ineligible.
        append(store, identity, 30, step=11)
        before = store.total_committed_sample_count(
            run_id=identity.run_id, generation_family=identity.generation_family
        )
        assert source_counts(select(store, identity)) == {CANDIDATE: 40}
        selected = select(store, identity, protected=protection())
        assert source_counts(selected) == {CANDIDATE: 30, CHAMPION: 10}
        diagnostics = replay_selection_diagnostics(
            selected, batch_size=4, current_model_step=100
        )
        teacher = diagnostics["protected_champion_replay"]
        assert teacher["model_identity"] == CHAMPION
        assert teacher["model_step"] == 10
        assert teacher["selected_rows"] == 10
        assert teacher["selected_fraction"] == 0.25
        assert teacher["selected_rows_by_ring_and_six_mode"] == {"r4/double": 10}
        assert teacher["cap_respected"] is True
        assert diagnostics["selected_rows_by_model_identity"] == {
            CANDIDATE: 30,
            CHAMPION: 10,
        }
        # Both worker paths consume the pinned selection without applying the
        # ordinary lag filter a second time.
        dataset = LazyShardReplayDataset(
            selected, seed=17, epoch=0, augmentation_enabled=False, shard_cache_size=2
        )
        worker = PersistentReplayWorkerDataset(
            augmentation_enabled=False, shard_cache_size=2
        )
        assert Counter(dataset[i].model_identity for i in range(len(dataset))) == {
            CANDIDATE: 30,
            CHAMPION: 10,
        }
        assert Counter(
            worker[dataset.reference(i)].model_identity for i in range(len(dataset))
        ) == {CANDIDATE: 30, CHAMPION: 10}
        assert selected.committed_samples == before
        assert {
            s.record.model_step
            for s in selected.spans
            if s.record.model_identity == CHAMPION
        } == {10}
        assert all(
            row.model_identity == CHAMPION for row in read_replay_shard(champ.path)
        )
        assert source_counts(select(store, identity, protected=protection(OTHER))) == {
            CANDIDATE: 30,
            OTHER: 10,
        }
        counts = store.eligible_sample_counts(
            (4,),
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            current_model_step=100,
            max_model_lag_steps=20,
            protected_champion=protection(),
        )
        assert counts == {4: 80}  # 60 ordinary + at most 20 protected.


@pytest.mark.parametrize(
    "ordinary,champion,expected",
    [
        (0, 100, {}),
        (3, 100, {CANDIDATE: 3, CHAMPION: 1}),
        (60, 0, {CANDIDATE: 40}),
        (60, 2, {CANDIDATE: 38, CHAMPION: 2}),
    ],
)
def test_shortages_preserve_ordinary_supply_and_hard_fraction_cap(
    tmp_path, monkeypatch, ordinary, champion, expected
):
    monkeypatch.setattr("startrain.replay_store.time.time_ns", lambda: NOW)
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        if ordinary:
            append(store, identity, ordinary, model=CANDIDATE, step=95)
        if champion:
            append(store, identity, champion)
        assert (
            source_counts(select(store, identity, protected=protection())) == expected
        )


def test_age_unknown_origin_future_and_other_run_are_excluded(tmp_path, monkeypatch):
    monkeypatch.setattr("startrain.replay_store.time.time_ns", lambda: NOW)
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        append(store, identity, 40, model=CANDIDATE, step=95)
        for first_time in (None, NOW - 2 * 10**12, NOW + 1):
            record = append(store, identity, 20)
            store.connection.execute(
                "UPDATE shards SET first_published_ns=? WHERE id=?",
                (first_time, record.shard_id),
            )
        other_run = replace(
            identity, run_id="other-run", generation_family="other-family"
        )
        store.lease_generation(other_run, "actor-test")
        append(store, other_run, 20)
        assert source_counts(select(store, identity, protected=protection())) == {
            CANDIDATE: 40
        }


def test_strict_pie_ring_and_mode_quotas_survive_protection(tmp_path, monkeypatch):
    monkeypatch.setattr("startrain.replay_store.time.time_ns", lambda: NOW)
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        for ring in (4, 6, 8, 10):
            for mode in ("classic", "double"):
                for model, step in ((CANDIDATE, 95), (CHAMPION, 10)):
                    append(
                        store,
                        identity,
                        100,
                        model=model,
                        step=step,
                        ring=ring,
                        mode=mode,
                        pie=True,
                    )
                    append(
                        store,
                        identity,
                        100,
                        model=model,
                        step=step,
                        ring=ring,
                        mode=mode,
                        handicap=9,
                    )
        selection = select_pie(
            store,
            identity,
            quota=136,
            max_model_lag_steps=20,
            protected_champion=protection(),
        )
        cells = Counter()
        protected_cells = Counter()
        for span in selection.spans:
            cell = (
                span.record.ring,
                span.record.segment,
                span.record.variant.rsplit("-", 1)[-1],
            )
            cells[cell] += span.sample_count
            if span.record.model_identity == CHAMPION:
                protected_cells[cell] += span.sample_count
        for ring in (4, 6, 8):
            for mode in ("classic", "double"):
                assert cells[ring, "pie", mode] == 68
                assert protected_cells[ring, "pie", mode] == 17
                assert cells[ring, "handicap", mode] == 0
        for mode in ("classic", "double"):
            assert cells[10, "pie", mode] == 60
            assert cells[10, "handicap", mode] == 8
            assert protected_cells[10, "pie", mode] == 15
            assert protected_cells[10, "handicap", mode] == 2


@pytest.mark.parametrize("legacy", [False, True])
def test_revision_keeps_original_publication_age_and_never_recredits(
    publication, monkeypatch, legacy
):
    store, identity, generation, options = publication
    monkeypatch.setattr(
        "startrain.replay_publication.time.time_ns", lambda: NOW - 2 * 10**12
    )
    prefix = store.append_game_revision(
        _samples(identity, generation, count=2, final=False), finalized=False, **options
    )
    if legacy:
        store.connection.execute("UPDATE game_publications SET first_published_ns=NULL")
        store.connection.execute("UPDATE shards SET first_published_ns=NULL")
    monkeypatch.setattr("startrain.replay_publication.time.time_ns", lambda: NOW)
    final = store.append_game_revision(
        _samples(identity, generation, count=4, final=True), finalized=True, **options
    )
    expected = None if legacy else NOW - 2 * 10**12
    first_times = store.connection.execute(
        "SELECT first_published_ns FROM shards ORDER BY id"
    ).fetchall()
    assert [row[0] for row in first_times] == [expected, expected]
    assert (
        store.connection.execute(
            "SELECT first_published_ns FROM game_publications"
        ).fetchone()[0]
        == expected
    )
    assert prefix.new_samples == final.new_samples == 2
    assert final.enriched_samples == 2
    repeat = store.append_game_revision(
        _samples(identity, generation, count=4, final=True), finalized=True, **options
    )
    assert repeat.new_samples == 0
    # The newly finalized shard cannot enter the protected source.
    assert (
        store.recent_shards(
            sample_window=100,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            current_model_step=100,
            max_model_lag_steps=20,
            protected_champion=protection(options["model_identity"]),
            protected_only=True,
        )
        == []
    )


def test_learner_verifies_champion_and_follows_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr("startrain.learner.time.time_ns", lambda: NOW)
    identity = run_identity(tmp_path)
    config = LearnerConfig(
        max_replay_lag_steps=20,
        recent_samples_per_ring=40,
        protected_champion_fraction=0.25,
        protected_champion_after_ns=NOW - 10**12,
    )
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        learner = make_test_learner(
            store, identity, tmp_path / "learner", learner_config=config
        )
        learner.step = 10
        champion = learner._publish()
        write_model_pointer(learner.publisher.champion_path, champion, role="champion")
        learner.step = 100
        assert (
            learner._protected_champion_replay().model_identity
            == champion.model_identity
        )
        append(store, identity, 60, model=CANDIDATE, step=95)
        append(store, identity, 30, model=champion.model_identity)
        assert source_counts(learner._rank_zero_select_replay_spans()) == {
            CANDIDATE: 30,
            champion.model_identity: 10,
        }
        assert learner._eligible_replay_counts()[4] == 80
        learner.step = 30
        replacement = learner._publish()
        write_model_pointer(
            learner.publisher.champion_path, replacement, role="champion"
        )
        learner.step = 100
        assert (
            learner._protected_champion_replay().model_identity
            == replacement.model_identity
        )
        assert source_counts(learner._rank_zero_select_replay_spans()) == {
            CANDIDATE: 40
        }
        with monkeypatch.context() as scoped:
            scoped.setattr(
                learner,
                "_control_model_manifest",
                lambda _: replace(replacement, role="champion", run_id="wrong-run"),
            )
            with pytest.raises(ValueError, match="active run"):
                learner._protected_champion_replay()
        # A changed pointer must be checked again, never trusted by cached identity.
        learner.publisher.champion_path.write_text("{}")
        with pytest.raises((ValueError, KeyError)):
            learner._protected_champion_replay()


def test_publication_origin_corruption_is_rejected(publication):
    store, identity, generation, options = publication
    store.append_game_revision(
        _samples(identity, generation, count=2, final=False),
        finalized=False,
        **options,
    )
    validate_game_publications(store.connection)
    store.connection.execute(
        "UPDATE game_publications SET first_published_ns=first_published_ns-1"
    )
    with pytest.raises(ValueError, match="origin metadata"):
        validate_game_publications(store.connection)


def test_legacy_database_upgrade_never_guesses_historical_origins(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("startrain.replay_store.time.time_ns", lambda: NOW)
    identity = run_identity(tmp_path)
    directory = tmp_path / "replay"
    with ReplayStore(directory) as store:
        store.lease_generation(identity, "actor-test")
        append(store, identity, 40, model=CANDIDATE, step=95)
        append(store, identity, 20)
        manifest = store.manifest_path
    with sqlite3.connect(manifest) as connection:
        connection.execute("DROP INDEX shards_protected_champion")
        connection.execute("ALTER TABLE shards DROP COLUMN first_published_ns")
        connection.execute(
            "ALTER TABLE game_publications DROP COLUMN first_published_ns"
        )
    with ReplayStore(directory) as store:
        assert all(
            row[0] is None
            for row in store.connection.execute("SELECT first_published_ns FROM shards")
        )
        assert source_counts(select(store, identity, protected=protection())) == {
            CANDIDATE: 40
        }


def test_protection_config_is_explicit_and_default_profiles_keep_their_payload():
    profile = load_config(
        Path(__file__).parents[1] / "configs/h100-8gpu-largest-board-priority.yaml"
    )
    assert not any(
        key.startswith("protected_champion") for key in profile.as_dict()["learner"]
    )
    with pytest.raises(ConfigError, match="activation"):
        LearnerConfig(protected_champion_fraction=0.25)
    for fraction in (-1, True, 0.51, float("nan")):
        with pytest.raises(ConfigError):
            LearnerConfig(protected_champion_fraction=fraction)


def test_actor_preserves_requested_champion_only_after_protection_activation(
    monkeypatch,
):
    supervisor = object.__new__(ActorSupervisor)
    supervisor.experiment = SimpleNamespace(
        learner=LearnerConfig(max_replay_lag_steps=20)
    )
    assert supervisor._champion_outside_replay_window(
        champion_step=10, learner_step=100
    )
    supervisor.experiment.learner = replace(
        supervisor.experiment.learner,
        protected_champion_fraction=0.25,
        protected_champion_after_ns=NOW,
    )
    monkeypatch.setattr("startrain.actor.time.time_ns", lambda: NOW - 1)
    assert supervisor._champion_outside_replay_window(
        champion_step=10, learner_step=100
    )
    monkeypatch.setattr("startrain.actor.time.time_ns", lambda: NOW)
    assert not supervisor._champion_outside_replay_window(
        champion_step=10, learner_step=100
    )


def test_actor_commit_metrics_distinguish_protected_pending_from_ineligible():
    protected = replay_commit_eligibility_metrics(
        ordinary_eligible=False, protected_candidate=True, samples=100
    )
    assert protected["ordinary_replay_eligible_at_commit"] is False
    assert protected["replay_eligible_at_commit"] is None
    assert protected["ineligible_samples_at_commit"] is None
    assert protected["eligible_samples_at_commit"] is None
    assert protected["protected_pending_selection_samples"] == 100
    assert protected["replay_eligibility_status"] == "protected_pending_selection"
    ordinary = replay_commit_eligibility_metrics(
        ordinary_eligible=True, protected_candidate=False, samples=100
    )
    assert ordinary["replay_eligible_at_commit"] is True
    assert ordinary["eligible_samples_at_commit"] == 100
    expired = replay_commit_eligibility_metrics(
        ordinary_eligible=False, protected_candidate=False, samples=100
    )
    assert expired["replay_eligible_at_commit"] is False
    assert expired["ineligible_samples_at_commit"] == 100


def test_queued_cohort_keeps_protected_champion_identity(tmp_path, monkeypatch):
    from test_cohort_work import fake_actor

    actor, store, _, _ = fake_actor(tmp_path, monkeypatch)
    actor.experiment = replace(
        actor.experiment,
        learner=replace(actor.experiment.learner, max_replay_lag_steps=2),
    )
    selection = ("champion", 4, "double-standard")
    normal = actor._new_work_bundle(None, store, selection=selection)
    assert normal.metadata["model_role"] == "candidate"
    assert normal.metadata["fallback_reason"] == "champion_outside_replay_window"
    actor.experiment = replace(
        actor.experiment,
        learner=replace(
            actor.experiment.learner,
            protected_champion_fraction=0.25,
            protected_champion_after_ns=1,
        ),
    )
    protected = actor._new_work_bundle(None, store, selection=selection)
    assert protected.metadata["requested_model_role"] == "champion"
    assert protected.metadata["model_role"] == "champion"
    assert protected.metadata["model_identity"] == "champion"
    assert protected.metadata["model_step"] == 1
    assert protected.metadata["fallback_reason"] is None


def test_retention_preserves_the_next_bounded_protected_window(tmp_path, monkeypatch):
    monkeypatch.setattr("startrain.replay_store.time.time_ns", lambda: NOW)
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        teacher = append(store, identity, 20)
        append(store, identity, 40, model=CANDIDATE, step=95)
        append(store, identity, 40, model=CANDIDATE, step=95)
        metrics = store.collect_garbage(
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            retain_shards_per_ring=1,
            dry_run=False,
            minimum_samples_per_ring=40,
            current_model_step=100,
            max_model_lag_steps=20,
            protected_champion=protection(),
        )
        assert metrics["deleted_shards"] == 1
        assert teacher.path.is_file()
        assert source_counts(select(store, identity, protected=protection())) == {
            CANDIDATE: 30,
            CHAMPION: 10,
        }
