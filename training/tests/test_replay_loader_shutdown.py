from pathlib import Path
from types import SimpleNamespace
import time

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import deltreltrain.learner as learner_module
from deltreltrain.learner import RebindableReplayBatchSampler, SpawnedReplayLoaderPool


def pool_with_loader(loader):
    pool = object.__new__(SpawnedReplayLoaderPool)
    pool.batch_sampler = RebindableReplayBatchSampler()
    pool.loader = loader
    pool.closed = False
    pool.shutdown_count = 0
    return pool


class PendingIterator:
    def __init__(self, *, failure=None, clock=None):
        self._sampler_iter = iter(["must not be scheduled"])
        self._timeout = 120.0
        self._shutdown = False
        self.pending = 3
        self.events = []
        self.failure = failure
        self.clock = clock

    def __next__(self):
        assert list(self._sampler_iter) == []
        assert not self._shutdown
        assert 0 < self._timeout <= learner_module.REPLAY_LOADER_SHUTDOWN_DRAIN_SECONDS
        if self.failure is not None:
            raise self.failure
        if not self.pending:
            raise StopIteration
        self.pending -= 1
        self.events.append("tensor_received")
        if self.clock is not None:
            self.clock[0] += 1
        return torch.ones(2)

    def _shutdown_workers(self):
        self.events.extend(["stop_pin_thread", "join_workers"])
        self._shutdown = True


def test_prefetched_tensors_are_received_before_pinning_and_worker_stop():
    iterator = PendingIterator()
    pool = pool_with_loader(SimpleNamespace(_iterator=iterator))
    assert pool.shutdown() is None
    assert iterator.events == ["tensor_received"] * 3 + [
        "stop_pin_thread",
        "join_workers",
    ]
    assert iterator.pending == 0
    assert iterator._timeout == 120.0
    assert pool.loader._iterator is None
    assert pool.closed and pool.shutdown_count == 1
    assert pool.shutdown() is None
    assert pool.shutdown_count == 1


@pytest.mark.parametrize("strict", [False, True])
def test_worker_abort_remains_an_error_and_still_reaps_workers(strict):
    error = RuntimeError("DataLoader worker (pid 123) is killed by signal: Aborted.")
    iterator = PendingIterator(failure=error)
    pool = pool_with_loader(SimpleNamespace(_iterator=iterator))
    if strict:
        with pytest.raises(RuntimeError, match="Aborted") as caught:
            pool.shutdown()
        assert caught.value is error
    else:
        assert pool.shutdown(strict=False) is error
    assert iterator.events == ["stop_pin_thread", "join_workers"]
    assert pool.closed
    assert iterator._timeout == 120.0


def test_drain_has_one_total_deadline_and_always_runs_final_cleanup(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(learner_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(learner_module, "REPLAY_LOADER_SHUTDOWN_DRAIN_SECONDS", 0.5)
    iterator = PendingIterator(clock=clock)
    pool = pool_with_loader(SimpleNamespace(_iterator=iterator))
    error = pool.shutdown(strict=False)
    assert isinstance(error, TimeoutError)
    assert iterator.events == ["tensor_received", "stop_pin_thread", "join_workers"]
    assert iterator._timeout == 120.0
    assert pool.closed


class TensorProducer(Dataset):
    """Real spawned workers transfer tensors and record exactly what they read."""

    def __init__(self, directory, *, fail_at=None):
        self.directory = str(directory)
        self.fail_at = fail_at

    def __len__(self):
        return 640

    def __getitem__(self, index):
        time.sleep(0.005)
        if index == self.fail_at:
            raise ValueError("pending replay decode failure")
        result = torch.full((32768,), float(index))
        (Path(self.directory) / str(index)).write_text("read")
        return result


def exercise_real_shutdown(tmp_path, *, pin_memory=False, fail_at=None):
    loader = DataLoader(
        TensorProducer(tmp_path, fail_at=fail_at),
        batch_size=8,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True,
        multiprocessing_context="spawn",
        pin_memory=pin_memory,
        timeout=20,
    )
    pool = pool_with_loader(loader)
    try:
        iterator = iter(loader)
        first = next(iterator)
        assert first.shape == (8, 32768)
        assert first[:, 0].tolist() == list(range(8))
        if pin_memory:
            assert first.is_pinned()
        workers = tuple(getattr(iterator, "_workers"))
        assert all(worker.is_alive() for worker in workers)
        queued = getattr(iterator, "_send_idx")
        assert queued == 5  # four prefetched batches, plus the consumed replacement
        failure = pool.shutdown(strict=False)
        assert all(not worker.is_alive() for worker in workers)
        if fail_at is None:
            assert failure is None
            assert all(worker.exitcode == 0 for worker in workers)
            # Shutdown finishes only the already-issued work, not the remaining
            # 75 batches of the epoch. This also proves tensors were drained.
            assert {int(path.name) for path in tmp_path.iterdir()} == set(
                range(queued * 8)
            )
        else:
            assert isinstance(failure, ValueError)
            assert "pending replay decode failure" in str(failure)
        assert pool.closed and pool.loader._iterator is None
    finally:
        pool.shutdown(strict=False)


def test_real_spawned_tensor_producers_drain_only_the_prefetched_tail(tmp_path):
    exercise_real_shutdown(tmp_path)


def test_real_pending_worker_exception_is_not_hidden_by_terminal_drain(tmp_path):
    exercise_real_shutdown(tmp_path, fail_at=16)


@pytest.mark.cuda
def test_real_pinned_tensor_producers_drain_before_pin_thread_shutdown(tmp_path):
    exercise_real_shutdown(tmp_path, pin_memory=True)
