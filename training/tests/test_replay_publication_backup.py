"""Revision-aware replay and durable work assignment survive backup boundaries."""

from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import pytest

import scripts.benchmark_learner_batches as learner_benchmark
import scripts.preflight_run_state as preflight
import scripts.replay_manifest_backup as ledger_backup
import scripts.run_frozen_replay_optimizer_calibration as frozen_benchmark
from startrain.cohort_work import PersistentWorkSchedule
from startrain.replay import ReplaySample
from startrain.replay_store import ReplayStore
from startrain.runtime import load_run_identity
from test_pipeline_core import make_replay_sample
from test_training_disaster_recovery import _fixture, _snapshot
import scripts.training_disaster_recovery as recovery


def revision_fixture(tmp_path):
    fixture = _fixture(tmp_path)
    identity = load_run_identity(fixture.root / "run.json")
    with ReplayStore(fixture.root / "replay") as store:
        generation = store.lease_generation(identity, "actor-revision")
    final = [
        replace(
            make_replay_sample(
                10,
                identity=identity,
                actor_id="actor-revision",
                generation=generation,
                game_id="publication-game",
                ply=ply,
            ),
            search_provenance="fixture:final=board-full:search=exact",
        )
        for ply in range(3)
    ]
    pending = [
        ReplaySample.from_position(
            row.to_position(),
            policy=row.policy,
            final_score=None,
            search_provenance="fixture:final=pending-policy:search=exact",
            policy_provenance=row.policy_provenance,
            run_id=row.run_id,
            generation_family=row.generation_family,
            actor_id=row.actor_id,
            generation=row.generation,
            game_id=row.game_id,
            ply=row.ply,
            model_identity=row.model_identity,
        )
        for row in final
    ]
    return fixture, identity, pending, final


def publish(store, samples, *, finalized=False):
    first = samples[0]
    return store.append_game_revision(
        samples,
        finalized=finalized,
        phase_min=0,
        phase_max=len(samples) - 1,
        model_version=first.model_identity,
        model_identity=first.model_identity,
        model_step=10,
        run_id=first.run_id,
        generation_family=first.generation_family,
        actor_id=first.actor_id,
        generation=first.generation,
    )


def ledger_state(connection):
    return {
        table: connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
        for table in ("game_publications", "games", "run_counters")
    }


def test_online_backup_restores_logical_credit_and_exact_revision_retry(tmp_path):
    fixture, identity, pending, final = revision_fixture(tmp_path)
    manifest = fixture.root / "replay/manifest.sqlite3"
    with ReplayStore(fixture.root / "replay") as store:
        initial = store.total_committed_sample_count(
            run_id=identity.run_id, generation_family=identity.generation_family
        )
        assert publish(store, pending[:2]).new_samples == 2
        assert publish(store, pending).new_samples == 1
        completed = publish(store, final, finalized=True)
        assert completed.new_samples == 0 and completed.enriched_samples == 3
        assert (
            store.connection.execute(
                "SELECT SUM(sample_count) FROM shards WHERE fresh_sample_count IS NOT NULL"
            ).fetchone()[0]
            == 8
        )
        assert (
            store.total_committed_sample_count(
                run_id=identity.run_id, generation_family=identity.generation_family
            )
            == initial + 3
        )
    before = ledger_backup.create_backup(fixture.root, retain=2)
    with sqlite3.connect(before) as connection:
        expected = ledger_state(connection)
        connection.row_factory = sqlite3.Row
        learner_benchmark._validate_manifest_metadata(connection)
        assert (
            frozen_benchmark._replay_metadata(connection)["manifest_schema_version"]
            == "6"
        )
    # The ordinary local backup restore must retain head tombstones and counters.
    manifest.write_bytes(b"corrupt database")
    assert ledger_backup.restore_if_corrupt(fixture.root) == before
    with sqlite3.connect(manifest) as connection:
        assert ledger_state(connection) == expected
    with ReplayStore(fixture.root / "replay") as store:
        receipt = publish(store, final, finalized=True)
        assert receipt.replayed and receipt.new_samples == receipt.enriched_samples == 0
        assert (
            store.total_committed_sample_count(
                run_id=identity.run_id, generation_family=identity.generation_family
            )
            == initial + 3
        )
        assert store.replay_revision_counts(
            run_id=identity.run_id, generation_family=identity.generation_family
        ) == (3, 3)


@pytest.mark.parametrize("gc_head", [False, True])
def test_disaster_restore_preserves_retired_revision_heads_without_old_payloads(
    tmp_path, gc_head
):
    fixture, identity, pending, final = revision_fixture(tmp_path)
    with ReplayStore(fixture.root / "replay") as store:
        first = publish(store, pending[:2]).record
        second = publish(store, pending).record
        head = publish(store, final, finalized=True).record
        if gc_head:
            later = make_replay_sample(
                10,
                identity=identity,
                actor_id="actor-revision",
                generation=final[0].generation,
                game_id="later-complete-game",
            )
            store.append(
                [later],
                phase_min=0,
                phase_max=0,
                model_version=later.model_identity,
                model_identity=later.model_identity,
                model_step=10,
                run_id=later.run_id,
                generation_family=later.generation_family,
                actor_id=later.actor_id,
                generation=later.generation,
            )
            store.collect_garbage(
                run_id=identity.run_id,
                generation_family=identity.generation_family,
                retain_shards_per_ring=1,
                dry_run=False,
            )
            assert not head.path.exists()
        credit = store.total_committed_sample_count(
            run_id=identity.run_id, generation_family=identity.generation_family
        )
    snapshot = _snapshot(fixture, tmp_path / "backup")
    catalog = json.loads(snapshot.read_text())["catalog"]
    for retired in (first, second):
        assert f"replay/shards/{retired.path.name}" not in catalog
    assert (f"replay/shards/{head.path.name}" in catalog) is not gc_head
    restored = Path(recovery.restore_snapshot(snapshot, tmp_path / "restored"))
    with ReplayStore(restored / "replay") as store:
        assert (
            store.total_committed_sample_count(
                run_id=identity.run_id, generation_family=identity.generation_family
            )
            == credit
        )
        replayed = publish(store, final, finalized=True)
        assert (
            replayed.replayed and replayed.new_samples == replayed.enriched_samples == 0
        )
        assert replayed.record.path.parent == restored / "replay/shards"
        assert replayed.record.path.exists() is not gc_head
        assert (
            store.total_committed_sample_count(
                run_id=identity.run_id, generation_family=identity.generation_family
            )
            == credit
        )
        assert store.replay_revision_counts(
            run_id=identity.run_id, generation_family=identity.generation_family
        ) == (3, 3)
    # Revisioned stores require their original durable counter even if every
    # currently retained physical file is intact.
    report, _ = preflight._validate_replay(
        restored,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        created_ns=identity.created_ns,
    )
    assert report["committed_samples"] == credit
    assert report["history_complete"] is True
    assert report["history_reconciliable"] is False


@pytest.mark.parametrize(
    "damage",
    [
        "missing_publications",
        "wrong_head",
        "wrong_revision",
        "wrong_enrichment",
        "missing_counter",
        "incomplete_history",
        "schema_downgrade",
    ],
)
def test_backup_and_preflight_fail_closed_on_damaged_revision_authority(
    tmp_path, damage
):
    fixture, identity, pending, final = revision_fixture(tmp_path)
    with ReplayStore(fixture.root / "replay") as store:
        publish(store, pending)
        publish(store, final, finalized=True)
    manifest = fixture.root / "replay/manifest.sqlite3"
    with sqlite3.connect(manifest) as connection:
        if damage == "missing_publications":
            connection.execute("DROP TABLE game_publications")
        elif damage == "wrong_head":
            connection.execute("UPDATE game_publications SET latest_shard_id=999999")
        elif damage == "wrong_revision":
            connection.execute(
                "UPDATE run_counters SET replay_revision=replay_revision+1"
            )
        elif damage == "wrong_enrichment":
            connection.execute("UPDATE run_counters SET enriched_rows=enriched_rows+1")
        elif damage == "missing_counter":
            connection.execute("DELETE FROM run_counters")
        elif damage == "incomplete_history":
            connection.execute("UPDATE run_counters SET history_complete=0")
        else:
            connection.execute(
                "UPDATE store_metadata SET value='5' WHERE key='manifest_schema_version'"
            )
    valid, reason = ledger_backup._integrity_ok(manifest, run_root=fixture.root)
    assert not valid and reason != "ok"
    with pytest.raises(preflight.StatePreflightError):
        preflight._validate_replay(
            fixture.root,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            created_ns=identity.created_ns,
        )


@pytest.mark.parametrize("damage", ["missing_counter", "incomplete_history"])
def test_revisioned_credit_is_never_reconciled_from_physical_shards(tmp_path, damage):
    fixture, identity, pending, final = revision_fixture(tmp_path)
    with ReplayStore(fixture.root / "replay") as store:
        publish(store, pending[:2])
        publish(store, pending)
        publish(store, final, finalized=True)
    manifest = fixture.root / "replay/manifest.sqlite3"
    with sqlite3.connect(manifest) as connection:
        if damage == "missing_counter":
            connection.execute("DELETE FROM run_counters")
        else:
            connection.execute("UPDATE run_counters SET history_complete=0")
    with sqlite3.connect(manifest) as connection:
        before = ledger_state(connection)
    with pytest.raises(preflight.StatePreflightError, match="physical shard totals"):
        preflight._apply_history_reconciliation(
            fixture.root,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
        )
    with sqlite3.connect(manifest) as connection:
        assert ledger_state(connection) == before


def test_disaster_snapshot_preserves_work_schedule_without_its_lock(tmp_path):
    fixture = _fixture(tmp_path)
    path = fixture.root / "status/work-schedule.json"
    owner = PersistentWorkSchedule(path, namespace="run-disaster-test:family", seed=17)
    roles = {"candidate": 0.5, "champion": 0.25, "history": 0.25}
    rings = {4: 0.05, 6: 0.05, 8: 0.05, 10: 0.85}
    for _ in range(5):
        owner.choose_role_and_ring(roles, rings, units=64)
    before = path.read_bytes()
    snapshot = _snapshot(fixture, tmp_path / "backup")
    next_expected = owner.choose_role_and_ring(roles, rings, units=64)
    owner.close()
    restored = recovery.restore_snapshot(snapshot, tmp_path / "restored")
    restored_path = Path(restored) / "status/work-schedule.json"
    assert restored_path.read_bytes() == before
    assert not restored_path.with_suffix(".json.lock").exists()
    restored_owner = PersistentWorkSchedule(
        restored_path, namespace="run-disaster-test:family", seed=17
    )
    assert restored_owner.choose_role_and_ring(roles, rings, units=64) == next_expected
    restored_owner.close()
