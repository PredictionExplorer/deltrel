"""Late outcomes refresh real immutable loaders without minting UTD credit."""

import time

import pytest

from startrain import learner as module
from startrain.config import DataConfig, LearnerConfig, SchedulerConfig, TrainConfig
from startrain.replay import read_replay_shard, TARGET_OUTCOME
from startrain.replay_store import ReplayStore
from test_pipeline_core import make_test_learner
from test_replay_game_revisions import publication as publication, _samples, _credit


def make_learner(store, identity, tmp_path, *, workers=0, ratio=0.5):
    return make_test_learner(
        store,
        identity,
        tmp_path / "learner",
        learner_config=LearnerConfig(
            minimum_replay_samples=1,
            recent_samples_per_ring=4,
            steps_per_window=100,
            target_updates_per_new_sample=ratio,
            device="cpu",
            replay_poll_seconds=0.001,
            metrics_interval=1,
            replay_refresh_seconds=0.05,
        ),
        train_config=TrainConfig(
            per_rank_batch_size=2,
            scheduler=SchedulerConfig(warmup_steps=0, total_steps=100),
        ),
        data_config=DataConfig(
            workers=workers,
            min_batches_for_workers=1,
            d5_augmentation=False,
            ring_stratified=False,
        ),
    )


@pytest.mark.parametrize("workers", [0, 1])
def test_enrichment_only_reopens_loader_but_does_not_earn_training_credit(
    publication, tmp_path, monkeypatch, workers
):
    store, identity, generation, kwargs = publication
    kwargs = {**kwargs, "model_step": 0}
    prefix = store.append_game_revision(
        _samples(identity, generation, 4), finalized=False, **kwargs
    )
    learner = make_learner(store, identity, tmp_path, workers=workers)
    opened = []
    losses = []
    final = None
    added = False
    checks = []
    original_open = learner._open_replay_window
    original_step = module.train_step

    def open_window(*args, **options):
        window = original_open(*args, **options)
        opened.append(window)
        return window

    def train(*args, **options):
        result = original_step(*args, **options)
        losses.append(float(result.loss_tensors["outcome"].detach()))
        return result

    monkeypatch.setattr(learner, "_open_replay_window", open_window)
    monkeypatch.setattr(module, "train_step", train)

    def progress(**fields):
        nonlocal final, added
        if fields.get("phase") == "training" and learner.step == 1 and final is None:
            assert losses == [0.0]
            final = store.append_game_revision(
                _samples(identity, generation, 4, final=True), finalized=True, **kwargs
            )
            assert final.new_samples == 0 and final.enriched_samples == 4
            opened[-1].opened_monotonic = time.monotonic() - 1
        if (
            fields.get("phase") == "update_to_data_wait"
            and len(opened) == 2
            and not added
        ):
            checks.append(True)
            assert _credit(store, identity) == 4
            assert learner.examples_consumed == 2 and learner._utd_step_budget() == 0
            assert opened[0].closed and opened[0].batches_consumed == 1
            assert opened[1].refresh_reason == "fresh_replay"
            assert opened[1].selection.spans[0].record.shard_id == final.record.shard_id
            assert opened[1].opened_enriched_samples == 4
            assert all(
                row.target_mask & TARGET_OUTCOME
                for row in read_replay_shard(final.record.path)
            )
            # Supply legitimate new credit, after proving enrichment supplied none.
            # Keep this newly opened selection so the second update reads its outcomes.
            store.append_game_revision(
                _samples(identity, generation, 4, game="new-game", final=True),
                finalized=True,
                **kwargs,
            )
            opened[-1].opened_monotonic = time.monotonic() + 30
            added = True

    try:
        assert learner.run(steps=2, progress=progress) == 2
        assert checks and added and losses[0] == 0 and losses[1] > 0
        assert _credit(store, identity) == 8 and learner.examples_consumed == 4
        assert learner._utd_step_budget() == 0
        assert opened[0].selection.spans[0].record.shard_id == prefix.record.shard_id
        assert (
            store.connection.execute("SELECT COUNT(*) FROM gc_watermarks").fetchone()[0]
            == 0
        )
    finally:
        learner._shutdown_loader_pool()


def test_real_window_acquisition_pins_before_concurrent_revision_and_gc(
    publication, tmp_path, monkeypatch
):
    store, identity, generation, kwargs = publication
    kwargs = {**kwargs, "model_step": 0}
    prefix = store.append_game_revision(
        _samples(identity, generation, 4), finalized=False, **kwargs
    )
    learner = make_learner(store, identity, tmp_path, ratio=1.0)
    original = learner._open_replay_window
    seen = []

    def opened(selection, **options):
        assert (
            store.connection.execute("SELECT COUNT(*) FROM gc_watermarks").fetchone()[0]
            == 1
        )
        with ReplayStore(store.root) as writer:
            writer.append_game_revision(
                _samples(identity, generation, 5), finalized=False, **kwargs
            )
            writer.collect_garbage(
                run_id=identity.run_id,
                generation_family=identity.generation_family,
                retain_shards_per_ring=1,
                dry_run=False,
            )
        assert prefix.record.path.is_file()
        seen.append(selection.committed_samples)
        result = original(selection, **options)
        assert result.opened_committed_samples == 4
        return result

    monkeypatch.setattr(learner, "_open_replay_window", opened)
    try:
        assert learner.run(steps=1) == 1
        assert seen == [4] and _credit(store, identity) == 5
        assert (
            store.connection.execute("SELECT COUNT(*) FROM gc_watermarks").fetchone()[0]
            == 0
        )
        store.collect_garbage(
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            retain_shards_per_ring=1,
            dry_run=False,
        )
        assert not prefix.record.path.exists()
    finally:
        learner._shutdown_loader_pool()


def test_allocation_failure_releases_pending_atomic_selection_pin(
    publication, tmp_path, monkeypatch
):
    store, identity, generation, kwargs = publication
    kwargs = {**kwargs, "model_step": 0}
    store.append_game_revision(
        _samples(identity, generation, 4), finalized=False, **kwargs
    )
    learner = make_learner(store, identity, tmp_path)

    def fail(*args, **kwargs):
        assert (
            store.connection.execute("SELECT COUNT(*) FROM gc_watermarks").fetchone()[0]
            == 1
        )
        raise RuntimeError("allocation failed")

    monkeypatch.setattr(learner, "_window_allocation_budget", fail)
    with pytest.raises(RuntimeError, match="allocation failed"):
        learner.run(steps=1)
    assert (
        store.connection.execute("SELECT COUNT(*) FROM gc_watermarks").fetchone()[0]
        == 0
    )
