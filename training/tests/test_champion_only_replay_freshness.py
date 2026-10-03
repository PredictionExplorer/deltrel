"""A champion-only source stays usable past model-step age without relabeling."""

from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from deltreltrain.actor import ActorSupervisor
from deltreltrain.checkpoint import write_model_pointer
from deltreltrain.config import ConfigError, LearnerConfig, load_config
from deltreltrain.learner import replay_selection_diagnostics
from deltreltrain.protected_replay import ProtectedChampionReplay
from deltreltrain.replay_store import ReplayStore, validate_game_publications
from test_pipeline_core import make_test_learner, run_identity
from test_protected_champion_replay import (
    NOW,
    CHAMPION,
    CANDIDATE,
    OTHER,
    append,
    select,
    source_counts,
)
from test_replay_game_revisions import _samples, _credit, publication as publication
from test_replay_pie_objective import select as select_pie


def freshness(identity=CHAMPION, *, step=10, after=NOW - 10**12):
    return ProtectedChampionReplay(
        identity, step, after, NOW, 1.0, champion_only_freshness=True
    )


@pytest.mark.parametrize(
    "ordinary,protected,expected",
    [
        (0, 100, {CHAMPION: 40}),
        (3, 100, {CHAMPION: 40}),
        (60, 2, {CANDIDATE: 38, CHAMPION: 2}),
        (60, 0, {CANDIDATE: 40}),
    ],
)
def test_explicit_full_mode_has_no_ordinary_supply_requirement(
    tmp_path, monkeypatch, ordinary, protected, expected
):
    monkeypatch.setattr("deltreltrain.replay_store.time.time_ns", lambda: NOW)
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        if ordinary:
            append(store, identity, ordinary, model=CANDIDATE, step=95)
        if protected:
            append(store, identity, protected)
        before = _credit(store, identity)
        selection = select(store, identity, protected=freshness())
        assert source_counts(selection) == expected
        assert _credit(store, identity) == before == selection.committed_samples
        assert store.eligible_sample_counts(
            (4,),
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            current_model_step=100,
            max_model_lag_steps=20,
            protected_champion=freshness(),
        ) == {4: ordinary + protected}
        diagnostics = replay_selection_diagnostics(
            selection, batch_size=4, current_model_step=100
        )
        assert (
            diagnostics["protected_champion_replay"]["champion_only_freshness"] is True
        )
        assert diagnostics["protected_champion_replay"]["cap_respected"] is True


def test_full_cap_requires_explicit_mode_and_keeps_legacy_caps():
    assert freshness().capacity(0, 100) == 100
    assert freshness().capacity(7, 11) == 18
    assert freshness().quota(19) == 19
    legacy = replace(freshness(), max_fraction=0.25, champion_only_freshness=False)
    assert legacy.capacity(0, 100) == 0
    assert legacy.capacity(60, 100) == 80
    with pytest.raises(ValueError, match="fraction"):
        replace(freshness(), champion_only_freshness=False)
    with pytest.raises(ValueError, match="fraction"):
        replace(legacy, champion_only_freshness=True)
    with pytest.raises(ValueError, match="boolean"):
        replace(freshness(), champion_only_freshness=1)


def test_full_mode_rejects_bad_origin_identity_step_future_and_old_watermark(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("deltreltrain.replay_store.time.time_ns", lambda: NOW)
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        old = append(store, identity, 40)
        expected = append(store, identity, 12)
        append(store, identity, 30, model=OTHER)
        append(store, identity, 30, step=11)
        append(store, identity, 30, step=101)
        for origin in (None, NOW - 2 * 10**12, NOW + 1, str(NOW)):
            row = append(store, identity, 30)
            # Non-numeric text avoids SQLite INTEGER affinity coercing it back.
            if isinstance(origin, str):
                origin = "unknown"
            store.connection.execute(
                "UPDATE shards SET first_published_ns=? WHERE id=?",
                (origin, row.shard_id),
            )
        inconsistent = append(store, identity, 30)
        store.connection.execute(
            "UPDATE shards SET created_ns=? WHERE id=?",
            (NOW - 1, inconsistent.shard_id),
        )
        other_run = replace(
            identity, run_id="other-run", generation_family="other-family"
        )
        store.lease_generation(other_run, "actor-test")
        append(store, other_run, 30)
        selected = select(
            store,
            identity,
            protected=freshness(),
            minimum_shard_id_exclusive=old.shard_id,
        )
        assert source_counts(selected) == {CHAMPION: 12}
        assert {span.record.shard_id for span in selected.spans} == {expected.shard_id}


def test_full_mode_keeps_strict_pie_cells_with_only_aged_champion_rows(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("deltreltrain.replay_store.time.time_ns", lambda: NOW)
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        for ring in (4, 6, 8, 10):
            for mode in ("classic", "double"):
                append(store, identity, 100, ring=ring, mode=mode, pie=True)
                append(store, identity, 100, ring=ring, mode=mode, handicap=9)
        selection = select_pie(
            store,
            identity,
            quota=136,
            max_model_lag_steps=20,
            protected_champion=freshness(),
        )
        assert selection.samples_by_ring == {4: 136, 6: 136, 8: 136, 10: 136}
        assert all(span.record.model_identity == CHAMPION for span in selection.spans)
        assert all(
            span.record.segment == "pie"
            for span in selection.spans
            if span.record.ring < 10
        )
        diagnostics = replay_selection_diagnostics(
            selection, batch_size=8, current_model_step=100
        )
        protected = diagnostics["protected_champion_replay"]
        assert protected["selected_fraction"] == 1
        assert protected["cap_respected"] is True


@pytest.mark.parametrize("unknown_origin", [False, True])
def test_final_revision_cannot_resurrect_pre_activation_prefix_or_double_credit(
    publication, monkeypatch, unknown_origin
):
    store, identity, generation, options = publication
    monkeypatch.setattr(
        "deltreltrain.replay_publication.time.time_ns", lambda: NOW - 2 * 10**12
    )
    initial = store.append_game_revision(
        _samples(identity, generation, 2), finalized=False, **options
    )
    if unknown_origin:
        store.connection.execute("UPDATE game_publications SET first_published_ns=NULL")
        store.connection.execute("UPDATE shards SET first_published_ns=NULL")
    monkeypatch.setattr("deltreltrain.replay_publication.time.time_ns", lambda: NOW)
    final = store.append_game_revision(
        _samples(identity, generation, 4, final=True), finalized=True, **options
    )
    assert final.record.shard_id > initial.record.shard_id
    assert (
        final.new_samples == 2
        and final.enriched_samples == 2
        and _credit(store, identity) == 4
    )
    selected = select(
        store,
        identity,
        protected=freshness(options["model_identity"]),
        minimum_shard_id_exclusive=initial.record.shard_id,
    )
    assert not selected.spans
    repeated = store.append_game_revision(
        _samples(identity, generation, 4, final=True), finalized=True, **options
    )
    assert repeated.new_samples == 0 and _credit(store, identity) == 4
    validate_game_publications(store.connection)


def test_learner_follows_verified_champion_promotions_and_rejects_tampering(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("deltreltrain.learner.time.time_ns", lambda: NOW)
    monkeypatch.setattr("deltreltrain.replay_store.time.time_ns", lambda: NOW)
    identity = run_identity(tmp_path)
    config = LearnerConfig(
        max_replay_lag_steps=20,
        recent_samples_per_ring=40,
        champion_only_replay_freshness=True,
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
        append(store, identity, 40, model=champion.model_identity)
        assert source_counts(learner._rank_zero_select_replay_spans()) == {
            champion.model_identity: 40
        }
        assert learner._eligible_replay_counts()[4] == 40
        learner.step = 30
        newer = learner._publish()
        write_model_pointer(learner.publisher.champion_path, newer, role="champion")
        learner.step = 100
        assert (
            learner._protected_champion_replay().model_identity == newer.model_identity
        )
        assert not learner._rank_zero_select_replay_spans().spans
        append(store, identity, 40, model=newer.model_identity, step=30)
        assert source_counts(learner._rank_zero_select_replay_spans()) == {
            newer.model_identity: 40
        }
        with monkeypatch.context() as scoped:
            scoped.setattr(
                learner,
                "_control_model_manifest",
                lambda _: replace(newer, role="champion", run_id="wrong-run"),
            )
            with pytest.raises(ValueError, match="active run"):
                learner._protected_champion_replay()
        learner.publisher.champion_path.write_text("{}")
        with pytest.raises((ValueError, KeyError)):
            learner._protected_champion_replay()


def test_activation_timestamp_controls_actor_fallback_without_legacy_fraction(
    monkeypatch,
):
    supervisor = object.__new__(ActorSupervisor)
    supervisor.experiment = SimpleNamespace(
        learner=LearnerConfig(
            max_replay_lag_steps=20,
            champion_only_replay_freshness=True,
            protected_champion_after_ns=NOW,
        )
    )
    monkeypatch.setattr("deltreltrain.actor.time.time_ns", lambda: NOW - 1)
    assert supervisor._champion_outside_replay_window(
        champion_step=10, learner_step=100
    )
    monkeypatch.setattr("deltreltrain.actor.time.time_ns", lambda: NOW)
    assert not supervisor._champion_outside_replay_window(
        champion_step=10, learner_step=100
    )


@pytest.mark.parametrize(
    "source", ["candidate", "candidate_champion_mix", "candidate_champion_history_mix"]
)
def test_joint_config_rejects_other_selfplay_sources(source):
    base = load_config(
        Path(__file__).parents[1] / "configs/h100-8gpu-largest-board-priority.yaml"
    )
    model_refresh = replace(
        base.orchestration.model_refresh,
        selfplay_source=source,
        history_probability=0.25 if source == "candidate_champion_history_mix" else 0.0,
    )
    with pytest.raises(ConfigError, match="requires champion-only"):
        replace(
            base,
            orchestration=replace(base.orchestration, model_refresh=model_refresh),
            learner=replace(
                base.learner,
                champion_only_replay_freshness=True,
                protected_champion_after_ns=NOW,
            ),
        )


def test_mode_defaults_do_not_change_old_config_serialization():
    base = load_config(
        Path(__file__).parents[1] / "configs/h100-8gpu-largest-board-priority.yaml"
    )
    assert "champion_only_replay_freshness" not in base.as_dict()["learner"]
    assert (
        replace(
            base, learner=replace(base.learner, champion_only_replay_freshness=False)
        ).as_dict()
        == base.as_dict()
    )
    enabled = replace(
        base,
        orchestration=replace(
            base.orchestration,
            model_refresh=replace(
                base.orchestration.model_refresh,
                selfplay_source="champion",
                history_probability=0.0,
            ),
        ),
        learner=replace(
            base.learner,
            champion_only_replay_freshness=True,
            protected_champion_after_ns=NOW,
        ),
    )
    assert enabled.as_dict()["learner"]["champion_only_replay_freshness"] is True
    with pytest.raises(ConfigError, match="activation"):
        LearnerConfig(champion_only_replay_freshness=True)
    with pytest.raises(ConfigError, match="cannot combine"):
        LearnerConfig(
            champion_only_replay_freshness=True,
            protected_champion_fraction=0.25,
            protected_champion_after_ns=NOW,
        )
    with pytest.raises(ConfigError, match="boolean"):
        LearnerConfig(champion_only_replay_freshness=1)


def test_retention_keeps_a_full_recent_champion_window_without_ordinary_rows(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("deltreltrain.replay_store.time.time_ns", lambda: NOW)
    identity = run_identity(tmp_path)
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        teacher = append(store, identity, 40)
        append(store, identity, 40, model=OTHER)
        append(store, identity, 40, model=OTHER)
        before = _credit(store, identity)
        store.collect_garbage(
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            retain_shards_per_ring=1,
            dry_run=False,
            minimum_samples_per_ring=40,
            current_model_step=100,
            max_model_lag_steps=20,
            protected_champion=freshness(),
        )
        assert teacher.path.is_file()
        assert source_counts(select(store, identity, protected=freshness())) == {
            CHAMPION: 40
        }
        assert _credit(store, identity) == before


def test_queued_actor_does_not_substitute_candidate_in_full_freshness_mode(
    tmp_path, monkeypatch
):
    from test_cohort_work import fake_actor

    actor, store, _, _ = fake_actor(tmp_path, monkeypatch)
    actor.experiment = replace(
        actor.experiment,
        orchestration=replace(
            actor.experiment.orchestration,
            model_refresh=replace(
                actor.experiment.orchestration.model_refresh,
                selfplay_source="champion",
                history_probability=0.0,
            ),
        ),
        learner=replace(
            actor.experiment.learner,
            max_replay_lag_steps=2,
            champion_only_replay_freshness=True,
            protected_champion_after_ns=1,
        ),
    )
    bundle = actor._new_work_bundle(
        None, store, selection=("champion", 4, "double-standard")
    )
    assert bundle.metadata["model_role"] == "champion"
    assert bundle.metadata["model_identity"] == "champion"
    assert bundle.metadata["fallback_reason"] is None


def test_aged_champion_only_window_can_perform_optimizer_updates(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr("deltreltrain.learner.time.time_ns", lambda: NOW)
    monkeypatch.setattr("deltreltrain.replay_store.time.time_ns", lambda: NOW)
    identity = run_identity(tmp_path)
    config = LearnerConfig(
        max_replay_lag_steps=20,
        recent_samples_per_ring=40,
        replay_wait_timeout_seconds=1,
        use_ring_mixture_curriculum=True,
        champion_only_replay_freshness=True,
        protected_champion_after_ns=NOW - 10**12,
    )
    with ReplayStore(tmp_path / "replay") as store:
        store.lease_generation(identity, "actor-test")
        learner = make_test_learner(
            store,
            identity,
            tmp_path / "learner",
            learner_config=config,
        )
        learner.step = 10
        teacher = learner._publish()
        write_model_pointer(learner.publisher.champion_path, teacher, role="champion")
        append(store, identity, 40, model=teacher.model_identity)
        learner.step = 30
        assert (
            learner._protected_champion_replay() is None
        )  # Inclusive ordinary boundary.
        assert learner._eligible_replay_counts()[4] == 40
        learner.step = 31
        assert learner._protected_champion_replay().max_fraction == 1
        learner.step = 100
        before = _credit(store, identity)
        weights_before = {
            name: tensor.clone() for name, tensor in learner.model.state_dict().items()
        }
        assert learner.run(steps=2) == 102
        assert any(
            not torch.equal(weights_before[name], tensor)
            for name, tensor in learner.model.state_dict().items()
        )
        assert learner.ema.num_updates == 2
        assert _credit(store, identity) == before


def test_legacy_positional_learner_configuration_remains_compatible():
    config = LearnerConfig(minimum_replay_shard_id_exclusive=12, steps_per_window=7)
    previous_arguments = [
        getattr(config, field.name)
        for field in fields(config)
        if field.name != "champion_only_replay_freshness"
    ]
    assert LearnerConfig(*previous_arguments) == config
