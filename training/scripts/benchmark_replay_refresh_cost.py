#!/usr/bin/env python3
"""Measure real persistent-loader fetch and partial-window drain costs.

Preparation reads the live manifest without opening ReplayStore and hard-links
immutable payloads into a new owned directory. Execution uses only that frozen
selection. It performs no optimizer updates and grants no replay credit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import time

from startrain.config import load_config
from startrain.learner import (
    LazyShardReplayDataset,
    SpawnedReplayLoaderPool,
    UniqueReplayBatchSampler,
)
from startrain.replay_store import ReplaySelection, ReplaySpan, ShardRecord

if __package__:
    from .benchmark_learner_batches import (
        CheckpointDescriptor,
        select_recent_ready_spans,
    )
else:
    from benchmark_learner_batches import (
        CheckpointDescriptor,
        select_recent_ready_spans,
    )


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_arm(value: str) -> tuple[int, int]:
    try:
        workers, prefetch = (int(part) for part in value.split(":"))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("arm must be workers:prefetch") from error
    if not 1 <= workers <= 32 or not 1 <= prefetch <= 8:
        raise argparse.ArgumentTypeError("workers must be 1..32, prefetch 1..8")
    return workers, prefetch


def freeze_selection(config_path: Path, replay_root: Path, output: Path, rows: int):
    replay_root, output = replay_root.resolve(), output.absolute()
    if rows <= 0 or rows > 262_144:
        raise ValueError("frozen selection rows must be in 1..262144")
    if output.resolve().is_relative_to(replay_root) or output.is_symlink():
        raise ValueError("snapshot must be outside the live replay root")
    if os.path.lexists(output):
        raise ValueError("snapshot directory must be new")
    config = load_config(config_path)
    identity = json.loads((replay_root.parent / "run.json").read_text())
    checkpoint = json.loads((replay_root.parent / "learner/recovery.json").read_text())
    descriptor = CheckpointDescriptor(
        step=int(checkpoint["step"]),
        epoch=int(checkpoint["epoch"]),
        run_id=identity["run_id"],
        generation_family=identity["generation_family"],
        config={},
        optimizer_compatible=False,
        scheduler_compatible=False,
    )
    ring, spans = select_recent_ready_spans(
        replay_root,
        config=config,
        descriptor=descriptor,
        minimum_samples=rows,
        target_samples=rows,
    )
    output.mkdir()
    try:
        (output / "shards").mkdir()
        records = []
        for span in spans:
            destination = output / "shards" / f"{span.record.shard_id}.npz"
            # Hard links keep an immutable payload alive through production GC.
            # A concurrent unlink before link() aborts the entire preparation.
            os.link(span.record.path, destination, follow_symlinks=False)
            if (
                destination.is_symlink()
                or digest(destination) != span.record.checksum_sha256
            ):
                raise ValueError("frozen replay payload failed checksum validation")
            record = asdict(replace(span.record, path=destination))
            record["path"] = str(destination)
            records.append(
                {
                    "record": record,
                    "sample_start": span.sample_start,
                    "sample_count": span.sample_count,
                }
            )
        plan = {
            "schema_version": 1,
            "benchmark": "persistent-loader-refresh-cost",
            "created_ns": time.time_ns(),
            "config_sha256": digest(config_path),
            "run_id": identity["run_id"],
            "generation_family": identity["generation_family"],
            "ring": ring,
            "samples": sum(span.sample_count for span in spans),
            "spans": records,
        }
        (output / "selection.json").write_text(json.dumps(plan, indent=2) + "\n")
        (output / "selection.json").chmod(0o444)
        return plan
    except BaseException:
        shutil.rmtree(output)
        raise


def load_selection(directory: Path) -> tuple[ReplaySelection, dict]:
    directory = directory.resolve()
    plan = json.loads((directory / "selection.json").read_text())
    if (
        plan.get("schema_version") != 1
        or plan.get("benchmark") != "persistent-loader-refresh-cost"
    ):
        raise ValueError("invalid frozen loader selection")
    spans = []
    for row in plan["spans"]:
        values = dict(row["record"])
        path = Path(values["path"])
        if path.is_symlink() or path.resolve().parent != directory / "shards":
            raise ValueError("frozen payload path escapes its owned directory")
        if digest(path) != values["checksum_sha256"]:
            raise ValueError("frozen payload checksum changed")
        values["path"] = path
        spans.append(
            ReplaySpan(ShardRecord(**values), row["sample_start"], row["sample_count"])
        )
    if not spans or sum(span.sample_count for span in spans) != plan["samples"]:
        raise ValueError("frozen selection counts disagree")
    return ReplaySelection(
        tuple(spans),
        {int(plan["ring"]): plan["samples"]},
        max(span.record.shard_id for span in spans),
    ), plan


class CountedSampler(UniqueReplayBatchSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.issued = 0

    def __iter__(self):
        for indices in super().__iter__():
            self.issued += 1
            yield indices


def measure_arm(
    selection,
    *,
    workers,
    prefetch,
    batch_size,
    batches,
    consume,
    cycles,
    pace_seconds,
    pin_memory,
):
    if not 0 < consume < batches or batches * batch_size > selection.sample_count:
        raise ValueError("partial-window benchmark requires enough distinct rows")
    pool = SpawnedReplayLoaderPool(
        num_workers=workers,
        augmentation_enabled=True,
        shard_cache_size=16,
        pin_memory=pin_memory,
        prefetch_factor=prefetch,
    )
    results = []
    pids = None
    try:
        for cycle in range(cycles):
            dataset = LazyShardReplayDataset(
                selection,
                seed=17,
                epoch=cycle,
                augmentation_enabled=True,
                shard_cache_size=16,
            )
            sampler = CountedSampler(
                dataset,
                batch_size=batch_size,
                batches=batches,
                seed=17,
                epoch=cycle,
                ring_stratified=True,
                shards_per_batch=4,
            )
            started = time.perf_counter()
            pool.rebind(dataset, sampler)
            iterator = iter(pool.loader)
            construction = time.perf_counter() - started
            fetch = []
            first_digest = None
            for index in range(consume):
                started = time.perf_counter()
                batch = next(iterator)
                fetch.append(time.perf_counter() - started)
                if index == 0:
                    tensor = batch.inputs.node_features.contiguous()
                    first_digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
                if pace_seconds:
                    time.sleep(pace_seconds)
            current_pids = pool.worker_pids
            if pids is not None and current_pids != pids:
                raise RuntimeError("persistent workers unexpectedly changed identity")
            pids = current_pids
            issued = sampler.issued
            started = time.perf_counter()
            pool.quiesce()
            drain = time.perf_counter() - started
            results.append(
                {
                    "cycle": cycle,
                    "construction_seconds": construction,
                    "first_fetch_seconds": fetch[0],
                    "subsequent_fetch_median_seconds": statistics.median(fetch[1:])
                    if len(fetch) > 1
                    else None,
                    "fetch_seconds": sum(fetch),
                    "quiesce_seconds": drain,
                    "issued_batches": issued,
                    "consumed_batches": consume,
                    "unconsumed_issued_batches": issued - consume,
                    "first_features_sha256": first_digest,
                    "worker_pids": pids,
                }
            )
        return {
            "workers": workers,
            "prefetch_factor": prefetch,
            "cycles": results,
            "pin_memory": pin_memory,
        }
    finally:
        pool.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--replay-root", type=Path)
    parser.add_argument(
        "--arms", nargs="+", type=parse_arm, default=[(16, 4), (8, 2), (4, 2)]
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--consume", type=int, default=16)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--pace-seconds", type=float, default=0.407)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not (
        1 <= args.cycles <= 5
        and 1 <= args.batch_size <= 1024
        and 2 <= args.batches <= 512
        and math.isfinite(args.pace_seconds)
        and 0 <= args.pace_seconds <= 2
    ):
        parser.error("benchmark bounds are invalid")
    if args.prepare:
        if args.config is None or args.replay_root is None:
            parser.error("preparation requires --config and --replay-root")
        plan = freeze_selection(
            args.config,
            args.replay_root,
            args.snapshot_dir,
            args.batch_size * args.batches,
        )
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "rows": plan["samples"],
                    "shards": len(plan["spans"]),
                    "ring": plan["ring"],
                }
            )
        )
        return
    if args.output is None or os.path.lexists(args.output):
        parser.error("execution requires a new --output path")
    selection, plan = load_selection(args.snapshot_dir)
    reports = [
        measure_arm(
            selection,
            workers=workers,
            prefetch=prefetch,
            batch_size=args.batch_size,
            batches=args.batches,
            consume=args.consume,
            cycles=args.cycles,
            pace_seconds=args.pace_seconds,
            pin_memory=args.pin_memory,
        )
        for workers, prefetch in args.arms
    ]
    signatures = [
        [cycle["first_features_sha256"] for cycle in arm["cycles"]] for arm in reports
    ]
    if any(value != signatures[0] for value in signatures):
        raise RuntimeError("loader arms consumed different augmented starting data")
    result = {
        "status": "passed",
        "benchmark": "persistent-loader-refresh-cost",
        "plan_sha256": digest(args.snapshot_dir / "selection.json"),
        "samples": plan["samples"],
        "pace_seconds": args.pace_seconds,
        "arms": reports,
        "counts_are_training_credit": False,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
