from deltreltrain import actor_publication as publication
from deltreltrain.actor_publication import PublicationProgress
import pytest
import time


def test_publications_are_rate_limited_and_final_flush_preserves_all_counters(
    monkeypatch,
):
    clock = {"now": 10.0, "rows": 20}
    records = []
    heartbeats = []
    monkeypatch.setattr(publication.time, "monotonic", lambda: clock["now"])
    callback = publication.PublicationProgress(
        metadata={"worker": "cohort", "process_started_ns": 1},
        base_games=5,
        base_samples=50,
        base_evaluator_rows=100,
        base_wall_seconds=20,
        task_started=10,
        evaluator_rows=lambda: clock["rows"],
        heartbeat=lambda **fields: heartbeats.append(fields),
        emit=records.append,
    )
    callback.progress(
        phase="selfplay_completed", completed_games=1, persisted_decisions=10
    )
    clock.update(now=10.1, rows=40)
    callback.progress(
        phase="selfplay_completed", completed_games=2, persisted_decisions=30
    )
    callback.progress(
        phase="selfplay_refill",
        started_games=4,
        completed_games=2,
        persisted_decisions=30,
    )
    assert len(records) == 1
    assert (
        heartbeats[-1]["cumulative_games"] == 7
        and heartbeats[-1]["cumulative_samples"] == 80
    )
    callback.finish()
    callback.finish()
    assert len(records) == 2
    assert [r["published_games"] for r in records] == [1, 1]
    assert [r["published_samples"] for r in records] == [10, 20]
    assert records[-1]["cumulative_evaluator_rows"] == 140
    assert records[-1]["cumulative_batch_wall_seconds"] == 20.1
    assert all("games" not in row and "samples" not in row for row in records)


def test_no_completed_games_means_no_publication_record():
    records = []
    callback = publication.PublicationProgress(
        metadata={},
        base_games=0,
        base_samples=0,
        base_evaluator_rows=0,
        base_wall_seconds=0,
        task_started=0,
        evaluator_rows=lambda: 7,
        heartbeat=lambda **_: None,
        emit=records.append,
    )
    callback.progress(
        phase="selfplay_refill",
        started_games=4,
        completed_games=0,
        persisted_decisions=0,
    )
    callback.finish()
    assert records == []


def test_policy_only_publication_counts_new_samples_without_relabeling_old_ones():
    records = []
    callback = publication.PublicationProgress(
        metadata={},
        base_games=0,
        base_samples=0,
        base_evaluator_rows=0,
        base_wall_seconds=0,
        task_started=0,
        evaluator_rows=lambda: 0,
        heartbeat=lambda **_: None,
        emit=records.append,
    )
    callback.progress(
        phase="selfplay_completed", completed_games=1, persisted_decisions=10
    )
    with pytest.raises(ValueError, match="counters"):
        callback.progress(
            phase="selfplay_policy_salvaged",
            completed_games=1,
            persisted_decisions=10,
            salvaged_policy_decisions=1,
        )
    callback.progress(
        phase="selfplay_policy_salvaged",
        completed_games=1,
        persisted_decisions=13,
        salvaged_policy_decisions=3,
    )
    callback.finish()
    assert sum(row["published_games"] for row in records) == 1
    assert sum(row["published_samples"] for row in records) == 13
    assert sum(row["published_policy_only_samples"] for row in records) == 3
    callback.progress(
        phase="selfplay_policy_salvaged",
        completed_games=1,
        persisted_decisions=15,
        salvaged_policy_decisions=5,
    )
    callback.finish()
    assert sum(row["published_samples"] for row in records) == 15
    assert sum(row["published_policy_only_samples"] for row in records) == 5


def test_revision_metrics_separate_fresh_credit_from_enrichment_and_physical_rows():
    records = []
    progress = PublicationProgress(
        metadata={},
        base_games=0,
        base_samples=0,
        base_evaluator_rows=0,
        base_wall_seconds=0,
        task_started=time.monotonic(),
        evaluator_rows=lambda: 0,
        heartbeat=lambda **_fields: None,
        emit=records.append,
        interval_seconds=0,
    )
    progress.progress(
        phase="selfplay_policy_published",
        completed_games=0,
        persisted_decisions=4,
        policy_published_decisions=4,
        enriched_decisions=0,
        replay_written_decisions=4,
        retained_incomplete_decisions=0,
    )
    progress.progress(
        phase="selfplay_policy_published",
        completed_games=0,
        persisted_decisions=6,
        policy_published_decisions=6,
        enriched_decisions=0,
        replay_written_decisions=10,
        retained_incomplete_decisions=0,
    )
    progress.progress(
        phase="selfplay_completed",
        completed_games=1,
        persisted_decisions=6,
        policy_published_decisions=6,
        enriched_decisions=6,
        replay_written_decisions=16,
        retained_incomplete_decisions=0,
    )
    progress.finish()
    assert sum(row["published_samples"] for row in records) == 6
    assert sum(row["published_written_samples"] for row in records) == 16
    assert sum(row["published_enriched_samples"] for row in records) == 6
    assert sum(row["published_games"] for row in records) == 1
    assert records[-1]["published_samples"] == 0
    assert records[-1]["pending_policy_rows"] == 0


def test_abandoned_already_published_prefix_is_classified_without_new_credit():
    records = []
    progress = PublicationProgress(
        metadata={},
        base_games=0,
        base_samples=0,
        base_evaluator_rows=0,
        base_wall_seconds=0,
        task_started=time.monotonic(),
        evaluator_rows=lambda: 0,
        heartbeat=lambda **_fields: None,
        emit=records.append,
        interval_seconds=0,
    )
    common = dict(
        completed_games=0,
        persisted_decisions=4,
        policy_published_decisions=4,
        enriched_decisions=0,
        replay_written_decisions=4,
        salvaged_policy_decisions=0,
    )
    progress.progress(
        phase="selfplay_policy_published", retained_incomplete_decisions=0, **common
    )
    progress.progress(
        phase="selfplay_policy_salvaged", retained_incomplete_decisions=4, **common
    )
    progress.finish()
    assert sum(row["published_samples"] for row in records) == 4
    assert sum(row["published_policy_only_samples"] for row in records) == 4
    assert (
        records[-1]["published_samples"]
        == records[-1]["published_written_samples"]
        == 0
    )
    assert records[-1]["pending_policy_rows"] == 4
