from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import multiprocessing
import os
import sqlite3
import time
from unittest.mock import patch

import pytest

from startrain.replay import ReplaySample, read_replay_shard
from startrain.replay_store import (
    DuplicateGameError,
    ReplayStore,
    prove_legacy_committed_sample_history,
    validate_game_publications,
)
from startrain.runtime import RunIdentity
from test_replay_concurrency import MODEL_IDENTITY, _sample


def _samples(identity, generation, count, *, game="game-one", final=False, tag=None):
    rows = []
    for ply in range(count):
        row = _sample(
            identity, actor_id="actor-one", generation=generation, game_id=game
        )
        if not final:
            row = ReplaySample.from_position(
                row.to_position(),
                policy=row.policy,
                final_score=None,
                search_provenance="pending",
                policy_provenance="completed-q",
                run_id=identity.run_id,
                generation_family=identity.generation_family,
                actor_id="actor-one",
                generation=generation,
                game_id=game,
                model_identity=MODEL_IDENTITY,
            )
        rows.append(
            replace(
                row,
                ply=ply,
                search_provenance=f"mcts:seed={ply}:final={tag or ('board-full' if final else 'pending-policy')}:algorithm=test",
            )
        )
    return rows


@pytest.fixture
def publication(tmp_path):
    identity = RunIdentity(tmp_path / "run.json", "run-revision", "family-revision", 1)
    store = ReplayStore(tmp_path / "replay")
    generation = store.lease_generation(identity, "actor-one")
    kwargs = dict(
        phase_min=0,
        phase_max=4,
        model_version=MODEL_IDENTITY,
        model_step=10,
        model_identity=MODEL_IDENTITY,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        actor_id="actor-one",
        generation=generation,
    )
    try:
        yield store, identity, generation, kwargs
    finally:
        store.close()


def _credit(store, identity):
    return store.total_committed_sample_count(
        run_id=identity.run_id, generation_family=identity.generation_family
    )


def _revision(store, identity):
    return store.replay_revision_counts(
        run_id=identity.run_id, generation_family=identity.generation_family
    )


def _selection(store, identity):
    return store.select_recent_spans(
        rings=[4],
        per_ring_quota=100,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        current_model_step=10,
        max_model_lag_steps=100,
    )


def test_prefix_extension_and_final_enrichment_credit_positions_once(publication):
    store, identity, generation, kwargs = publication
    assert (
        store.connection.execute(
            "SELECT value FROM store_metadata WHERE key='manifest_schema_version'"
        ).fetchone()[0]
        == "5"
    )
    first = store.append_game_revision(
        _samples(identity, generation, 2), finalized=False, **kwargs
    )
    second = store.append_game_revision(
        _samples(identity, generation, 3), finalized=False, **kwargs
    )
    final = store.append_game_revision(
        _samples(identity, generation, 3, final=True), finalized=True, **kwargs
    )
    assert (
        first.new_samples,
        first.enriched_samples,
        first.completed_games,
        first.revision,
    ) == (2, 0, 0, 1)
    assert (
        second.new_samples,
        second.enriched_samples,
        second.completed_games,
        second.revision,
    ) == (1, 0, 0, 2)
    assert (
        final.new_samples,
        final.enriched_samples,
        final.completed_games,
        final.revision,
    ) == (0, 3, 1, 3)
    assert _credit(store, identity) == 3
    assert _revision(store, identity) == (3, 3)
    assert store.connection.execute(
        "SELECT SUM(sample_count),SUM(fresh_sample_count) FROM shards"
    ).fetchone()[:] == (8, 3)
    selection = _selection(store, identity)
    assert selection.sample_count == 3
    assert selection.publication_revision == selection.enriched_samples == 3
    assert [s.record.shard_id for s in selection.spans] == [final.record.shard_id]
    assert all(row.outcome >= 0 for row in read_replay_shard(final.record.path))
    assert [record.shard_id for record, _ in store.iter_after_cursor("new-reader")] == [
        final.record.shard_id
    ]
    validate_game_publications(store.connection)


@pytest.mark.parametrize("final", [False, True])
def test_exact_retry_has_no_side_effects(publication, final):
    store, identity, generation, kwargs = publication
    samples = _samples(identity, generation, 2, final=final)
    first = store.append_game_revision(samples, finalized=final, **kwargs)
    replay = store.append_game_revision(samples, finalized=final, **kwargs)
    assert replay.record == first.record
    assert replay.replayed and not first.replayed
    assert (replay.new_samples, replay.enriched_samples, replay.completed_games) == (
        0,
        0,
        0,
    )
    assert _credit(store, identity) == 2
    assert _revision(store, identity) == (1, 0)
    assert len(list(store.shard_directory.glob("*.npz"))) == 1


def test_final_with_new_tail_credits_only_tail_and_permits_final_weights(publication):
    store, identity, generation, kwargs = publication
    store.append_game_revision(
        _samples(identity, generation, 2), finalized=False, **kwargs
    )
    final = [
        replace(row, weight=1.5)
        for row in _samples(identity, generation, 4, final=True)
    ]
    result = store.append_game_revision(final, finalized=True, **kwargs)
    assert (result.new_samples, result.enriched_samples, result.completed_games) == (
        2,
        2,
        1,
    )
    assert _credit(store, identity) == 4
    assert _revision(store, identity) == (2, 2)


def test_clinch_finalization_can_enrich_outcome_without_inventing_auxiliary_labels(
    publication,
):
    from startrain.contracts import TARGET_OUTCOME, TARGET_POLICY, TARGET_SOFT_POLICY

    store, identity, generation, kwargs = publication
    store.append_game_revision(
        _samples(identity, generation, 2), finalized=False, **kwargs
    )
    final = []
    for row in _samples(identity, generation, 2, final=True, tag="clinch-loser-fill"):
        owners, alive = row.final_ownership.copy(), row.final_alive.copy()
        owners.fill(-100)
        alive.fill(255)
        final.append(
            replace(
                row,
                target_mask=TARGET_POLICY | TARGET_SOFT_POLICY | TARGET_OUTCOME,
                final_ownership=owners,
                final_alive=alive,
            )
        )
    receipt = store.append_game_revision(final, finalized=True, **kwargs)
    assert (receipt.new_samples, receipt.enriched_samples) == (0, 2)
    restored = read_replay_shard(receipt.record.path)
    assert all(
        row.target_mask == TARGET_POLICY | TARGET_SOFT_POLICY | TARGET_OUTCOME
        for row in restored
    )


def test_first_revision_cannot_upgrade_incomplete_credit_authority(publication):
    store, identity, generation, kwargs = publication
    store.connection.execute("UPDATE run_counters SET history_complete=0")
    with pytest.raises(ValueError, match="complete durable credit history"):
        store.append_game_revision(
            _samples(identity, generation, 2), finalized=False, **kwargs
        )
    assert (
        store.connection.execute(
            "SELECT value FROM store_metadata WHERE key='manifest_schema_version'"
        ).fetchone()[0]
        == "5"
    )
    assert _credit(store, identity) == 0
    assert not list(store.shard_directory.glob("*.npz"))


@pytest.mark.parametrize(
    "column,value",
    [("model_step", 11), ("variant", "classic"), ("checksum_sha256", "0" * 64)],
)
def test_head_descriptor_is_checked_against_live_shard(publication, column, value):
    store, identity, generation, kwargs = publication
    receipt = store.append_game_revision(
        _samples(identity, generation, 2), finalized=False, **kwargs
    )
    store.connection.execute(
        f"UPDATE shards SET {column}=? WHERE id=?", (value, receipt.record.shard_id)
    )
    with pytest.raises(ValueError, match="invalid live shard"):
        validate_game_publications(store.connection)


@pytest.mark.parametrize(
    "change", ["policy_weight", "policy", "provenance", "policy_provenance", "state"]
)
def test_final_cannot_rewrite_published_policy_identity(publication, change):
    store, identity, generation, kwargs = publication
    store.append_game_revision(
        _samples(identity, generation, 2), finalized=False, **kwargs
    )
    final = _samples(identity, generation, 3, final=True)
    row = final[0]
    if change == "policy_weight":
        final[0] = replace(row, policy_weight=0.3)
    elif change == "policy":
        policy = row.policy.copy()
        policy[0] += 0.01
        policy[1] -= 0.01
        from startrain.replay import katago_soft_policy_target

        final[0] = replace(
            row,
            policy=policy,
            soft_policy=katago_soft_policy_target(policy, row.stones == -1),
        )
    elif change == "provenance":
        final[0] = replace(
            row, search_provenance=row.search_provenance.replace("seed=0", "seed=9")
        )
    elif change == "policy_provenance":
        final[0] = replace(row, policy_provenance="another-policy")
    else:
        final[0] = replace(row, pda=1)
    with pytest.raises(ValueError, match="semantic state or policy changed"):
        store.append_game_revision(final, finalized=True, **kwargs)
    assert _credit(store, identity) == 2
    assert len(list(store.shard_directory.glob("*.npz"))) == 1


def test_final_cannot_reopen_or_change_result(publication):
    store, identity, generation, kwargs = publication
    store.append_game_revision(
        _samples(identity, generation, 2, final=True), finalized=True, **kwargs
    )
    with pytest.raises(ValueError, match="reopen"):
        store.append_game_revision(
            _samples(identity, generation, 2), finalized=False, **kwargs
        )
    final = _samples(identity, generation, 2, final=True)
    final[0] = replace(final[0], weight=1.1)
    with pytest.raises(ValueError, match="finalized"):
        store.append_game_revision(final, finalized=True, **kwargs)


@pytest.mark.parametrize(
    "invalid",
    ["mixed_game", "gap", "targets", "no_policy", "final_tag", "duplicate_tag"],
)
def test_invalid_pending_prefix_is_rejected_before_credit(publication, invalid):
    store, identity, generation, kwargs = publication
    rows = _samples(identity, generation, 2)
    if invalid == "mixed_game":
        rows[1] = replace(rows[1], game_id="different")
    elif invalid == "gap":
        rows[1] = replace(rows[1], ply=2)
    elif invalid == "targets":
        rows = _samples(identity, generation, 2, final=True, tag="pending-policy")
    elif invalid == "no_policy":
        rows[0] = ReplaySample.from_position(
            rows[0].to_position(),
            policy=None,
            final_score=None,
            search_provenance=rows[0].search_provenance,
            policy_provenance="none",
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            actor_id="actor-one",
            generation=generation,
            game_id=rows[0].game_id,
            model_identity=MODEL_IDENTITY,
        )
    elif invalid == "final_tag":
        rows[0] = replace(rows[0], search_provenance="mcts:final=unknown")
    else:
        rows[0] = replace(
            rows[0],
            search_provenance=rows[0].search_provenance + ":final=pending-policy",
        )
    with pytest.raises(ValueError):
        store.append_game_revision(rows, finalized=False, **kwargs)
    assert _credit(store, identity) == 0
    assert (
        store.connection.execute(
            "SELECT value FROM store_metadata WHERE key='manifest_schema_version'"
        ).fetchone()[0]
        == "5"
    )


def test_pending_weights_and_model_context_are_immutable(publication):
    store, identity, generation, kwargs = publication
    store.append_game_revision(
        _samples(identity, generation, 2), finalized=False, **kwargs
    )
    rows = _samples(identity, generation, 3)
    rows[0] = replace(rows[0], weight=0.9)
    with pytest.raises(ValueError, match="pending publication payload changed"):
        store.append_game_revision(rows, finalized=False, **kwargs)
    with pytest.raises(ValueError, match="context changed"):
        store.append_game_revision(
            _samples(identity, generation, 3),
            finalized=False,
            **(kwargs | {"model_step": 11}),
        )
    with pytest.raises(ValueError, match="shrink"):
        store.append_game_revision(
            _samples(identity, generation, 1), finalized=False, **kwargs
        )


def test_lease_rollover_fences_finalization(publication):
    store, identity, generation, kwargs = publication
    store.append_game_revision(
        _samples(identity, generation, 2), finalized=False, **kwargs
    )
    assert store.lease_generation(identity, "actor-one") == generation + 1
    with pytest.raises(ValueError, match="active lease"):
        store.append_game_revision(
            _samples(identity, generation, 3, final=True), finalized=True, **kwargs
        )
    assert _credit(store, identity) == 2


@pytest.mark.parametrize("ordinary_first", [False, True])
def test_ordinary_append_duplicate_contract_and_credit_are_preserved(
    publication, ordinary_first
):
    store, identity, generation, kwargs = publication
    final = _samples(identity, generation, 2, final=True)
    if ordinary_first:
        store.append(final, **kwargs)
        with pytest.raises(DuplicateGameError):
            store.append_game_revision(final, finalized=True, **kwargs)
    else:
        store.append_game_revision(final, finalized=True, **kwargs)
        with pytest.raises(DuplicateGameError):
            store.append(final, **kwargs)
    store.append(
        _samples(identity, generation, 1, game="normal-other", final=True), **kwargs
    )
    assert _credit(store, identity) == 3
    validate_game_publications(store.connection)


def test_transaction_failure_rolls_back_file_head_schema_and_credit(publication):
    store, identity, generation, kwargs = publication
    store.connection.execute(
        "CREATE TRIGGER fail_counter BEFORE UPDATE ON run_counters BEGIN SELECT RAISE(ABORT,'injected'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        store.append_game_revision(
            _samples(identity, generation, 2), finalized=False, **kwargs
        )
    assert _credit(store, identity) == 0
    assert store.connection.execute("SELECT COUNT(*) FROM shards").fetchone()[0] == 0
    assert (
        store.connection.execute("SELECT COUNT(*) FROM game_publications").fetchone()[0]
        == 0
    )
    assert (
        store.connection.execute(
            "SELECT value FROM store_metadata WHERE key='manifest_schema_version'"
        ).fetchone()[0]
        == "5"
    )
    assert not list(store.shard_directory.glob("*.npz"))


def test_lost_acknowledgement_after_commit_retries_without_credit(publication):
    store, identity, generation, kwargs = publication
    rows = _samples(identity, generation, 2)
    with patch(
        "startrain.replay_store.GameRevisionReceipt",
        side_effect=RuntimeError("lost acknowledgement"),
    ):
        with pytest.raises(RuntimeError, match="acknowledgement"):
            store.append_game_revision(rows, finalized=False, **kwargs)
    assert _credit(store, identity) == 2
    retry = store.append_game_revision(rows, finalized=False, **kwargs)
    assert retry.replayed and retry.new_samples == 0
    assert retry.record.path.is_file()


def test_commit_ack_failure_never_unlinks_a_durably_committed_payload(publication):
    store, identity, generation, kwargs = publication
    connection = store.connection

    class LostCommitAcknowledgement:
        def __getattr__(self, name):
            return getattr(connection, name)

        def execute(self, statement, *arguments):
            result = connection.execute(statement, *arguments)
            if statement == "COMMIT":
                raise RuntimeError("commit acknowledgement lost")
            return result

    rows = _samples(identity, generation, 2)
    store.connection = LostCommitAcknowledgement()
    try:
        with pytest.raises(RuntimeError, match="acknowledgement lost"):
            store.append_game_revision(rows, finalized=False, **kwargs)
    finally:
        store.connection = connection
    assert _credit(store, identity) == 2
    retry = store.append_game_revision(rows, finalized=False, **kwargs)
    assert retry.replayed and retry.new_samples == 0
    assert retry.record.path.is_file()
    assert len(read_replay_shard(retry.record.path)) == 2


def _process_publish(root, rows, kwargs, result_queue, crash=False):
    with ReplayStore(root) as store:
        if crash:
            with patch(
                "startrain.replay_store._sha256", side_effect=lambda _: os._exit(17)
            ):
                store.append_game_revision(rows, finalized=False, **kwargs)
        else:
            receipt = store.append_game_revision(rows, finalized=False, **kwargs)
            result_queue.put((receipt.new_samples, receipt.replayed))


def test_cross_process_identical_publication_is_credited_once(publication):
    store, identity, generation, kwargs = publication
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    rows = _samples(identity, generation, 2)
    processes = [
        context.Process(target=_process_publish, args=(store.root, rows, kwargs, queue))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(20)
        assert process.exitcode == 0
    assert sorted(queue.get(timeout=2) for _ in processes) == [(0, True), (2, False)]
    queue.close()
    assert _credit(store, identity) == 2
    assert _revision(store, identity) == (1, 0)
    validate_game_publications(store.connection)


def test_process_crash_after_immutable_file_before_manifest_is_not_credit(publication):
    store, identity, generation, kwargs = publication
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_process_publish,
        args=(store.root, _samples(identity, generation, 2), kwargs, None, True),
    )
    process.start()
    process.join(20)
    assert process.exitcode == 17
    assert _credit(store, identity) == 0
    orphans = list(store.shard_directory.glob("*.npz"))
    assert len(orphans) == 1
    os.utime(orphans[0], (time.time() - 400, time.time() - 400))
    with ReplayStore(store.root) as reopened:
        assert _credit(reopened, identity) == 0
        assert not orphans[0].exists()


def test_concurrent_conflicting_prefix_cannot_replace_first_commit(publication):
    store, identity, generation, kwargs = publication
    rows = _samples(identity, generation, 2)
    changed = deepcopy(rows)
    changed[0] = replace(changed[0], policy_weight=0.4)

    def publish(batch):
        with ReplayStore(store.root) as connection:
            try:
                return connection.append_game_revision(
                    batch, finalized=False, **kwargs
                ).new_samples
            except ValueError:
                return "conflict"

    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(publish, [rows, changed]))
    assert set(results) == {2, "conflict"}
    assert _credit(store, identity) == 2
    validate_game_publications(store.connection)


def test_gc_respects_pinned_old_versions_and_retains_credit_tombstones(publication):
    store, identity, generation, kwargs = publication
    first = store.append_game_revision(
        _samples(identity, generation, 2), finalized=False, **kwargs
    )
    store.set_gc_watermark("loader", _selection(store, identity))
    second = store.append_game_revision(
        _samples(identity, generation, 3), finalized=False, **kwargs
    )
    gc_args = dict(
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        retain_shards_per_ring=1,
        dry_run=False,
    )
    store.collect_garbage(**gc_args)
    assert first.record.path.exists()
    store.clear_gc_watermark("loader")
    store.collect_garbage(**gc_args)
    assert not first.record.path.exists() and second.record.path.exists()
    store.append_game_revision(
        _samples(identity, generation, 1, game="other-game", final=True),
        finalized=True,
        **kwargs,
    )
    store.collect_garbage(**gc_args)
    assert not second.record.path.exists()
    validate_game_publications(store.connection)
    retry = store.append_game_revision(
        _samples(identity, generation, 3), finalized=False, **kwargs
    )
    assert retry.replayed and retry.new_samples == 0
    assert retry.record.path == second.record.path
    final = store.append_game_revision(
        _samples(identity, generation, 3, final=True), finalized=True, **kwargs
    )
    assert final.new_samples == 0 and final.enriched_samples == 3
    assert _credit(store, identity) == 4
    validate_game_publications(store.connection)


def test_selection_counter_and_ready_versions_share_snapshot(publication):
    store, identity, generation, kwargs = publication
    first = store.append_game_revision(
        _samples(identity, generation, 2), finalized=False, **kwargs
    )
    original = store._select_recent_spans_snapshot
    with ReplayStore(store.root) as writer:

        def concurrent_publish(**arguments):
            writer.append_game_revision(
                _samples(identity, generation, 3, final=True), finalized=True, **kwargs
            )
            return original(**arguments)

        with patch.object(
            store, "_select_recent_spans_snapshot", side_effect=concurrent_publish
        ):
            selection = _selection(store, identity)
    assert selection.publication_revision == 1 and selection.enriched_samples == 0
    assert selection.committed_samples == 2
    assert (
        selection.sample_count == 2
        and selection.spans[0].record.shard_id == first.record.shard_id
    )
    current = _selection(store, identity)
    assert current.publication_revision == 2 and current.enriched_samples == 2
    assert current.committed_samples == 3
    assert current.sample_count == 3
    store.connection.execute("BEGIN")
    _selection(store, identity)
    assert store.connection.in_transaction
    store.connection.execute("ROLLBACK")


def test_window_open_uses_selected_credit_even_if_more_rows_commit_after_selection(
    publication,
):
    from startrain.replay_store import ReplaySelection

    store, identity, generation, kwargs = publication
    store.append_game_revision(
        _samples(identity, generation, 2), finalized=False, **kwargs
    )
    selected = _selection(store, identity)
    with ReplayStore(store.root) as writer:
        writer.append_game_revision(
            _samples(identity, generation, 3), finalized=False, **kwargs
        )
    assert selected.committed_samples == 2
    assert _credit(store, identity) == 3
    assert _selection(store, identity).committed_samples == 3
    assert ReplaySelection((), {}, 0).committed_samples is None


def test_actual_selection_pins_before_concurrent_supersede_and_gc(publication):
    import threading

    store, identity, generation, kwargs = publication
    first = store.append_game_revision(
        _samples(identity, generation, 2), finalized=False, **kwargs
    )
    started = threading.Event()
    gc_args = dict(
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        retain_shards_per_ring=1,
        dry_run=False,
    )

    def publish_and_collect():
        started.set()
        with ReplayStore(store.root) as writer:
            writer.append_game_revision(
                _samples(identity, generation, 3), finalized=False, **kwargs
            )
            writer.collect_garbage(**gc_args)

    original = store._select_recent_spans_snapshot
    with ThreadPoolExecutor(1) as executor:
        futures = []

        def concurrent_attempt(**arguments):
            futures.append(executor.submit(publish_and_collect))
            assert started.wait(2)
            return original(**arguments)

        with patch.object(
            store, "_select_recent_spans_snapshot", side_effect=concurrent_attempt
        ):
            selection = store.select_recent_spans(
                rings=[4],
                per_ring_quota=100,
                run_id=identity.run_id,
                generation_family=identity.generation_family,
                current_model_step=10,
                max_model_lag_steps=100,
                gc_watermark_name="actual-loader",
            )
        futures[0].result(timeout=20)
    assert selection.spans[0].record.shard_id == first.record.shard_id
    assert first.record.path.is_file()
    assert len(read_replay_shard(first.record.path)) == 2
    store.clear_gc_watermark("actual-loader")
    store.collect_garbage(**gc_args)
    assert not first.record.path.exists()


@pytest.mark.parametrize(
    "corruption", ["missing_counters", "downgrade", "counter_mismatch"]
)
def test_schema_six_corruption_fails_closed_before_reconstruction(
    publication, corruption
):
    store, identity, generation, kwargs = publication
    store.append_game_revision(
        _samples(identity, generation, 2), finalized=False, **kwargs
    )
    if corruption == "missing_counters":
        store.connection.execute("DROP TABLE run_counters")
    elif corruption == "downgrade":
        store.connection.execute(
            "UPDATE store_metadata SET value='5' WHERE key='manifest_schema_version'"
        )
    else:
        store.connection.execute("UPDATE run_counters SET committed_samples=0")
    with pytest.raises(ValueError):
        ReplayStore(store.root)
    if corruption == "missing_counters":
        assert (
            store.connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name='run_counters'"
            ).fetchone()
            is None
        )


def test_schema_six_legacy_history_proof_never_uses_physical_totals(publication):
    store, identity, generation, kwargs = publication
    store.append_game_revision(
        _samples(identity, generation, 2), finalized=False, **kwargs
    )
    proof = prove_legacy_committed_sample_history(
        store.connection,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
    )
    assert not proof.complete
    assert any("logical fresh credit" in reason for reason in proof.failures)


def test_read_only_legacy_probe_defaults_revision_counts_to_zero():
    connection = sqlite3.connect(":memory:")
    store = ReplayStore.__new__(ReplayStore)
    store.connection = connection
    try:
        assert store.replay_revision_counts(run_id="old", generation_family="old") == (
            0,
            0,
        )
    finally:
        connection.close()
