"""Replay readiness stays exact while unchanged manifests avoid repeated scans."""

import sqlite3

import pytest

from startrain.replay_store import ReplayStore
from test_replay_game_revisions import publication as publication, _samples


@pytest.fixture(params=[False, True], ids=["ring", "segment"])
def counts(request, publication):
    store, identity, _generation, _kwargs = publication
    method = (
        store.eligible_sample_counts_by_segment
        if request.param
        else store.eligible_sample_counts
    )

    def read(**changes):
        options = dict(
            rings=(4,),
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            current_model_step=10,
            max_model_lag_steps=10,
        )
        options.update(changes)
        return method(**options)

    return read


def publish(publication, rows, *, final=False):
    store, identity, generation, kwargs = publication
    return store.append_game_revision(
        _samples(identity, generation, rows, final=final),
        finalized=final,
        **kwargs,
    ).record


def test_repeated_readiness_checks_do_one_scan_and_return_independent_dicts(
    publication, counts
):
    store = publication[0]
    publish(publication, 2)
    statements = []
    store.connection.set_trace_callback(statements.append)
    first = counts()
    assert sum(first.values()) == 2
    first.clear()
    for _ in range(10):
        assert sum(counts().values()) == 2
    assert len([sql for sql in statements if "SUM(sample_count)" in sql]) == 1
    # The same fast path handles an empty selection, including requested zeros.
    assert sum(counts(current_model_step=9).values()) == 0
    assert sum(counts(current_model_step=9).values()) == 0
    assert len([sql for sql in statements if "SUM(sample_count)" in sql]) == 2


def test_local_publication_supersession_and_quarantine_invalidate_counts(
    publication, counts
):
    store = publication[0]
    publish(publication, 2)
    assert sum(counts().values()) == 2
    publish(publication, 3)
    assert sum(counts().values()) == 3  # Superseded prefixes aren't double counted.
    final = publish(publication, 3, final=True)
    assert sum(counts().values()) == 3
    store.connection.execute(
        "UPDATE shards SET state='quarantined' WHERE id=?", (final.shard_id,)
    )
    assert sum(counts().values()) == 0


def test_other_connection_publication_is_visible_immediately(publication, counts):
    store, identity, generation, kwargs = publication
    publish(publication, 2)
    # Open the actor before priming the cache, to exclude initialization changes.
    with ReplayStore(store.root) as actor:
        assert sum(counts().values()) == 2
        before = store.connection.total_changes
        actor.append_game_revision(
            _samples(identity, generation, 3), finalized=False, **kwargs
        )
        assert store.connection.total_changes == before
        assert sum(counts().values()) == 3


@pytest.mark.parametrize("finish", ["COMMIT", "ROLLBACK"])
def test_local_transaction_results_never_leak_across_transaction_boundary(
    publication, counts, finish
):
    store = publication[0]
    record = publish(publication, 2)
    assert sum(counts().values()) == 2
    store.connection.execute("BEGIN IMMEDIATE")
    try:
        store.connection.execute("DELETE FROM shards WHERE id=?", (record.shard_id,))
        assert sum(counts().values()) == 0
        store.connection.execute(finish)
    finally:
        if store.connection.in_transaction:
            store.connection.execute("ROLLBACK")
    assert sum(counts().values()) == (2 if finish == "ROLLBACK" else 0)


def test_wal_reader_keeps_snapshot_until_commit_then_sees_external_gc(
    publication, counts
):
    store = publication[0]
    record = publish(publication, 2)
    assert sum(counts().values()) == 2
    store.connection.execute("BEGIN")
    try:
        assert sum(counts().values()) == 2
        with sqlite3.connect(store.manifest_path) as writer:
            writer.execute("DELETE FROM shards WHERE id=?", (record.shard_id,))
        assert sum(counts().values()) == 2
        store.connection.execute("COMMIT")
    finally:
        if store.connection.in_transaction:
            store.connection.execute("ROLLBACK")
    assert sum(counts().values()) == 0


def test_every_eligibility_constraint_is_part_of_cached_query(publication, counts):
    record = publish(publication, 2)
    assert sum(counts().values()) == 2
    for change in (
        {"current_model_step": 9},
        {"current_model_step": 21},
        {"current_model_step": 11, "max_model_lag_steps": 0},
        {"minimum_shard_id_exclusive": record.shard_id},
        {"rings": (6,)},
        {"run_id": "another-run"},
        {"generation_family": "another-family"},
    ):
        assert sum(counts(**change).values()) == 0
        assert sum(counts().values()) == 2
    assert sum(counts(current_model_step=20).values()) == 2
    # Long-running learners must not retain one cache entry per training step.
    for step in range(100):
        counts(current_model_step=step)
    assert len(publication[0]._eligible_counts_cache) <= 8


def test_mutable_ring_input_and_ring_order_cannot_change_cached_results(
    publication, counts
):
    publish(publication, 2)
    rings = [4, 6]
    expected = counts(rings=rings)
    assert sum(expected.values()) == 2
    rings.reverse()
    assert counts(rings=rings) == expected
    rings[:] = [6]
    assert sum(counts(rings=rings).values()) == 0
    assert counts(rings=[4, 6]) == expected
