from dataclasses import replace
import sqlite3
from types import SimpleNamespace

import pytest

from deltreltrain.actor_publication import PublicationProgress
from deltreltrain.replay import read_replay_shard
from test_live_policy_publication import MODEL, credit, setup


class LostCommitAcknowledgement:
    def __init__(self, connection):
        self.connection = connection

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def execute(self, statement, *arguments):
        result = self.connection.execute(statement, *arguments)
        if statement == "COMMIT":
            raise OSError("durable commit acknowledgement lost")
        return result


def _publication(records):
    return PublicationProgress(
        metadata={},
        base_games=0,
        base_samples=0,
        base_evaluator_rows=0,
        base_wall_seconds=0,
        task_started=0,
        evaluator_rows=lambda: 0,
        heartbeat=lambda **_: None,
        emit=records.append,
        interval_seconds=0,
    )


@pytest.mark.native
@pytest.mark.parametrize("stage", ["first", "extension", "final"])
def test_acknowledgement_loss_reconciles_actor_but_never_recredits_store(
    tmp_path, stage
):
    actor, store, identity = setup(tmp_path)
    original, connection = store.append_game_revision, store.connection
    failed = False

    def publish(samples, **kwargs):
        nonlocal failed
        previous = actor._published_prefixes.get(samples[0].game_id, 0)
        matches = (
            (stage == "first" and not kwargs["finalized"] and not previous)
            or (stage == "extension" and not kwargs["finalized"] and previous > 0)
            or (stage == "final" and kwargs["finalized"])
        )
        if matches and not failed:
            failed = True
            store.connection = LostCommitAcknowledgement(connection)
            try:
                return original(samples, **kwargs)
            finally:
                store.connection = connection
        return original(samples, **kwargs)

    store.append_game_revision = publish
    records = []
    publication = _publication(records)
    try:
        summaries = actor.run(progress=publication.progress)
        publication.finish()
        assert failed
        expected = sum(game.samples for game in summaries)
        assert actor.persisted_decisions == credit(store, identity) == expected
        assert actor.source_candidate_samples == expected
        assert actor.policy_published_decisions == actor.enriched_decisions > 0
        assert (
            actor.replay_written_decisions
            == connection.execute("SELECT SUM(sample_count) FROM shards").fetchone()[0]
        )
        revision, enriched = store.replay_revision_counts(
            run_id=identity.run_id, generation_family=identity.generation_family
        )
        assert actor.replay_revisions == revision
        assert actor.replay_append_calls == revision + 1
        assert actor.enriched_decisions == enriched
        assert sum(row["published_samples"] for row in records) == expected
        assert sum(row["published_enriched_samples"] for row in records) == enriched
        assert (
            sum(row["published_written_samples"] for row in records)
            == actor.replay_written_decisions
        )
        assert sum(row["published_games"] for row in records) == len(summaries)
    finally:
        store.close()


@pytest.mark.native
@pytest.mark.parametrize("error_type", [OSError, sqlite3.OperationalError])
def test_one_transient_failure_before_commit_is_retried(tmp_path, error_type):
    actor, store, identity = setup(tmp_path)
    original = store.append_game_revision
    calls = 0

    def publish(samples, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error_type("temporary storage failure")
        return original(samples, **kwargs)

    store.append_game_revision = publish
    try:
        summaries = actor.run()
        assert (
            actor.persisted_decisions
            == credit(store, identity)
            == sum(game.samples for game in summaries)
        )
        assert actor.replay_append_calls == actor.replay_revisions + 1 == calls
    finally:
        store.close()


@pytest.mark.native
@pytest.mark.parametrize(
    "error_type", [ValueError, KeyboardInterrupt, sqlite3.IntegrityError]
)
def test_semantic_lease_interrupt_and_integrity_failures_are_not_retried(
    tmp_path, error_type
):
    actor, store, identity = setup(tmp_path)
    calls = 0

    def publish(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise error_type("lease or semantic failure")

    store.append_game_revision = publish
    try:
        with pytest.raises(error_type):
            actor.run()
        assert calls == 1
        assert actor.persisted_decisions == credit(store, identity) == 0
    finally:
        store.close()


@pytest.mark.native
def test_repeated_commit_ack_failure_stops_after_two_attempts_and_retains_data(
    tmp_path,
):
    actor, store, identity = setup(tmp_path)
    original, connection = store.append_game_revision, store.connection
    calls = 0

    def publish(samples, **kwargs):
        nonlocal calls
        calls += 1
        store.connection = LostCommitAcknowledgement(connection)
        try:
            return original(samples, **kwargs)
        finally:
            store.connection = connection

    store.append_game_revision = publish
    try:
        with pytest.raises(OSError, match="acknowledgement"):
            actor.run()
        assert calls == 2
        assert credit(store, identity) == 2 and actor.persisted_decisions == 0
        paths = list(store.shard_directory.glob("*.npz"))
        assert len(paths) == 1 and len(read_replay_shard(paths[0])) == 2
    finally:
        store.close()


@pytest.mark.native
def test_unsolicited_replayed_receipt_with_local_gap_still_fails(tmp_path):
    actor, store, identity = setup(tmp_path)
    original = store.append_game_revision

    def publish(samples, **kwargs):
        original(samples, **kwargs)
        return original(samples, **kwargs)

    store.append_game_revision = publish
    try:
        with pytest.raises(RuntimeError, match="credit disagrees"):
            actor.run()
        assert credit(store, identity) == 2 and actor.persisted_decisions == 0
    finally:
        store.close()


@pytest.mark.native
def test_already_acknowledged_pending_prefix_replay_is_a_local_noop(tmp_path):
    actor, store, identity = setup(tmp_path)
    tested = False

    def progress(**fields):
        nonlocal tested
        if tested or fields.get("phase") != "selfplay_policy_published":
            return
        tested = True
        row = store.connection.execute(
            "SELECT * FROM shards WHERE state='ready' ORDER BY id LIMIT 1"
        ).fetchone()
        samples = read_replay_shard(store.root / row["relative_path"])
        decisions = [
            SimpleNamespace(phase=int((sample.stones != -1).sum()))
            for sample in samples
        ]
        before = (
            actor.persisted_decisions,
            actor.policy_published_decisions,
            actor.enriched_decisions,
            actor.replay_written_decisions,
            actor.replay_revisions,
        )
        assert (
            actor._write_game_revision(
                samples, decisions, (MODEL, 0, MODEL), finalized=False
            )
            == 0
        )
        assert before == (
            actor.persisted_decisions,
            actor.policy_published_decisions,
            actor.enriched_decisions,
            actor.replay_written_decisions,
            actor.replay_revisions,
        )

    try:
        actor.run(progress=progress)
        assert tested and actor.persisted_decisions == credit(store, identity)
    finally:
        store.close()


@pytest.mark.native
@pytest.mark.parametrize("invalid", ["model", "counts", "untyped"])
def test_retry_reconciliation_does_not_bypass_receipt_validation(tmp_path, invalid):
    actor, store, identity = setup(tmp_path)
    original, connection = store.append_game_revision, store.connection
    calls = 0

    def publish(samples, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            store.connection = LostCommitAcknowledgement(connection)
            try:
                return original(samples, **kwargs)
            finally:
                store.connection = connection
        receipt = original(samples, **kwargs)
        if invalid == "model":
            return replace(receipt, record=replace(receipt.record, model_step=99))
        if invalid == "counts":
            return replace(receipt, new_samples=1)
        return object()

    store.append_game_revision = publish
    try:
        with pytest.raises(RuntimeError):
            actor.run()
        assert calls == 2
        assert credit(store, identity) == 2 and actor.persisted_decisions == 0
    finally:
        store.close()
