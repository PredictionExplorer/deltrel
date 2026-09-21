#!/usr/bin/env python3
"""Bounded checkpoint, allowed-variant, CUDA-graph and BF16 learner smoke.

Uses only synthetic legal positions and an in-memory copy of the checkpoint.
It writes no training replay or weights. Run on an exclusive GPU at cutover.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import cast

import numpy as np
import torch

from scripts.validate_cuda_graph_runtime import (
    check_health,
    compare_predictions,
    regular_adapter,
)
from deltreltrain.checkpoint import load_checkpoint
from deltreltrain.config import load_config
from deltreltrain.inference import GraphInferenceAdapter, InferenceConfig
from deltreltrain.model import GraphResTNet
from deltreltrain.native import load_deltrel_native, positions_from_native
from deltreltrain.optim import build_optimizer
from deltreltrain.replay import ReplaySample, collate_replay_samples
from deltreltrain.runtime import atomic_json
from deltreltrain.selfplay import GameVariant
from deltreltrain.training import train_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("smoke output must be a new file")
    started = time.time_ns()
    config = load_config(args.profile)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    model = GraphResTNet(config.model).cuda()
    optimizer = build_optimizer(model, config.optimizer)
    metadata = load_checkpoint(
        args.checkpoint,
        model=model,
        optimizer=optimizer,
        expected_model_config=asdict(config.model),
        expected_game_config=asdict(config.game),
    )
    compiled = cast(torch.nn.Module, torch.compile(model, dynamic=True, mode="default"))
    graph = GraphInferenceAdapter(
        compiled,
        device="cuda:0",
        config=InferenceConfig(
            precision="bf16",
            preserve_broadcast_topology=True,
            cuda_graphs=True,
            cuda_graph_max_entries=32,
            cuda_graph_max_bytes=8 * 1024**3,
            small_batch_graph_buckets=True,
        ),
        homogeneous_relational_bias=True,
        model_version="sha256-" + checkpoint_hash,
    )
    reference = regular_adapter(graph)
    native = load_deltrel_native(required=True)
    if native is None:
        raise RuntimeError("native extension unavailable")
    results, training_samples = [], []
    for ring in (4, 6, 8, 10):
        labels = ["pie-classic", "pie-double"]
        if ring == 10:
            labels += [
                f"handicap-{n}-{mode}" for n in (2, 9) for mode in ("classic", "double")
            ]
        for label in labels:
            variant = GameVariant.parse(label)
            states = native.StateBatch(
                ring, 2, mode=variant.mode, handicap=variant.handicap, pie=variant.pie
            )
            states.apply_many([0, 1], [0, 1])
            advantage = config.selfplay.variants.pda_for_handicap(variant.handicap)
            search = native.SearchBatch(
                states,
                simulations=2,
                max_considered=2,
                deterministic_seed=17,
                pda_by_seat=[(-advantage, advantage)] * 2,
            )
            request = search.root_requests()
            expected = reference.evaluate(request)
            first = graph.evaluate(request)
            repeated = graph.evaluate(request)
            compare_predictions(expected, first)
            difference = compare_predictions(expected, repeated)
            search.initialize_roots(*repeated.submit_args())
            for _ in range(16):
                if search.is_done():
                    break
                leaves = search.next_requests()
                if len(leaves):
                    search.submit(*graph.evaluate(leaves).submit_args())
            if not search.is_done():
                raise RuntimeError("native search failed to complete")
            result = search.results()
            if len(result.selected_actions) != 2 or any(
                int(a) < 0 for a in result.selected_actions
            ):
                raise RuntimeError("native search returned invalid actions")
            results.append({"ring": ring, "variant": label, "difference": difference})
            if ring == 10:
                position = positions_from_native(states.data())[0]
                policy = np.asarray(position.stones == -1, dtype=np.float32)
                policy /= policy.sum()
                training_samples.append(
                    ReplaySample.from_position(
                        position,
                        policy=policy,
                        final_score=None,
                        search_provenance="synthetic deployment smoke; never published",
                        policy_provenance="uniform legal policy for finite-backward check",
                    )
                )
    metrics = check_health(graph)
    if metrics.get("graph_replays", 0) <= 0:
        raise RuntimeError("CUDA graph was never replayed")
    reference.close()
    graph.close()
    model.train()
    step = train_step(
        model,
        collate_replay_samples(training_samples),
        optimizer,
        precision="bf16",
        loss_weights=config.loss,
        gradient_clip_norm=config.train.gradient_clip_norm,
        share_homogeneous_geometry=True,
    )
    if not all(np.isfinite(value) for value in step.losses.values()):
        raise FloatingPointError("nonfinite smoke learner loss")
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
        raise RuntimeError("smoke changed checkpoint bytes")
    report = {
        "status": "passed",
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": metadata["step"],
        "variants": results,
        "graph_metrics": metrics,
        "learner_losses": step.losses,
        "started_ns": started,
        "completed_ns": time.time_ns(),
        "training_artifacts_written": False,
        "timing_claim": "correctness smoke only",
    }
    atomic_json(args.output, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
