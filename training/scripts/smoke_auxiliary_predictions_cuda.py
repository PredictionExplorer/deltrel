#!/usr/bin/env python3
"""Bounded, exclusive-GPU qualification of an additive auxiliary-head upgrade.

The run must already be cleanly stopped. Original training state and committed
replay are read only; one learner update affects only the worker's in-memory
model. The parent owns the worker process group and imposes a 900-second limit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch

from scripts.smoke_adaptive_promotion_cuda import publish_report, require_stopped_run
from scripts.validate_cuda_graph_runtime import validate_output
from startrain.auxiliary_upgrade import AUXILIARY_LOSSES, is_auxiliary_parameter
from startrain.checkpoint import (
    ExponentialMovingAverage,
    load_checkpoint,
    normalize_model_config,
    sha256_file,
)
from startrain.config import load_config
from startrain.contracts import TARGET_OUTCOME, TARGET_POLICY
from startrain.features import encode_batch
from startrain.gradient_clipping import GradientClipper
from startrain.inference import GraphInferenceAdapter, InferenceConfig
from startrain.model import GraphResTNet, ModelConfig
from startrain.native import load_star_native, positions_from_native
from startrain.optim import build_optimizer, optimizer_checkpoint_contract
from startrain.replay import (
    ReplayBatch,
    ReplaySample,
    collate_replay_samples,
    read_replay_shard,
)
from startrain.selfplay import GameVariant, _Decision, _future_policy_targets
from startrain.training import build_scheduler, maybe_compile_model, train_step


TIMEOUT_SECONDS = 900
MAX_REPLAY_SHARDS = 32
REPLAY_QUERY_TIMEOUT_SECONDS = 5.0
MAX_RUNTIME_BATCH_SIZE = 1024
PRIMARY_OUTPUTS = (
    "policy_logits",
    "outcome_logits",
    "score_margin_logits",
    "ownership_logits",
    "alive_logits",
    "soft_policy_logits",
)


def assert_equal_tree(actual: Any, expected: Any, context: str) -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype:
            raise AssertionError(f"{context}: tensor type changed")
        torch.testing.assert_close(
            actual.detach().cpu(), expected.detach().cpu(), rtol=0, atol=0
        )
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f"{context}: mapping keys changed")
        for key, value in expected.items():
            assert_equal_tree(actual[key], value, f"{context}.{key}")
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise AssertionError(f"{context}: sequence changed")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_equal_tree(left, right, f"{context}[{index}]")
    elif actual != expected:
        raise AssertionError(f"{context}: value changed")


def optimizer_states_by_name(state: dict, routing: dict) -> dict:
    return {
        name: state["state"].get(index, {})
        for route, group in zip(routing["groups"], state["param_groups"], strict=True)
        for name, index in zip(route["parameter_names"], group["params"], strict=True)
    }


def load_training_state(config: Any, checkpoint: Path, device: str) -> tuple:
    """Use the production resume path and prove all old state survived exactly."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_values: dict[str, Any] = normalize_model_config(payload["config"]["model"])
    old_config = ModelConfig(**model_values)
    if old_config.auxiliary_predictions or not config.model.auxiliary_predictions:
        raise ValueError(
            "smoke requires an old checkpoint and an auxiliary-enabled target"
        )
    reference = GraphResTNet(old_config).to(device).eval()
    reference.load_state_dict(payload["model"], strict=True)
    model = GraphResTNet(config.model).to(device)
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.train.scheduler)
    ema = ExponentialMovingAverage(model, decay=config.train.resolved_ema_decay(1))
    clipper = (
        GradientClipper(
            model.named_parameters(),
            config=config.train.gradient_clipping,
            max_norm=config.train.gradient_clip_norm,
        )
        if payload.get("gradient_clipping") is not None
        else None
    )
    if clipper is None and config.train.gradient_clipping.mode != "global":
        raise ValueError("smoke cannot cold-start missing adaptive clipping state")
    metadata = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        gradient_clipper=clipper,
        allow_auxiliary_upgrade=True,
        expected_model_config=asdict(config.model),
        expected_game_config=asdict(config.game),
        map_location=device,
    )
    for name, tensor in payload["model"].items():
        assert_equal_tree(model.state_dict()[name], tensor, f"model.{name}")
    for name, tensor in payload["ema"]["shadow"].items():
        assert_equal_tree(ema.shadow[name], tensor, f"ema.{name}")
    assert_equal_tree(ema.num_updates, payload["ema"]["num_updates"], "EMA clock")
    assert_equal_tree(ema.decay, payload["ema"]["decay"], "EMA decay")
    assert_equal_tree(scheduler.state_dict(), payload["scheduler"], "scheduler")
    routing = optimizer_checkpoint_contract(optimizer)
    if routing is None or payload.get("optimizer_routing") is None:
        raise ValueError("smoke requires named optimizer routing")
    old_states = optimizer_states_by_name(
        payload["optimizer"], payload["optimizer_routing"]
    )
    new_states = optimizer_states_by_name(optimizer.state_dict(), routing)
    for name, state in old_states.items():
        assert_equal_tree(new_states[name], state, f"optimizer.{name}")
    for name in new_states.keys() - old_states.keys():
        if not is_auxiliary_parameter(name) or new_states[name]:
            raise AssertionError(
                "new optimizer parameter state is not a fresh auxiliary head"
            )
    for old_group, new_group in zip(
        payload["optimizer"]["param_groups"],
        optimizer.state_dict()["param_groups"],
        strict=True,
    ):
        assert_equal_tree(
            {k: v for k, v in new_group.items() if k != "params"},
            {k: v for k, v in old_group.items() if k != "params"},
            "optimizer group",
        )
    if clipper is not None:
        before, after = payload["gradient_clipping"], clipper.state_dict()
        for key, value in before.items():
            if key == "parameters":
                current = {v["name"]: v for v in after[key]}
                for parameter in value:
                    assert_equal_tree(
                        current[parameter["name"]], parameter, "clipping parameter"
                    )
            elif key == "ema_norms":
                for name, history in value.items():
                    assert_equal_tree(after[key][name], history, f"clipping.{name}")
            else:
                assert_equal_tree(after[key], value, f"clipping.{key}")
    for key, value in payload["extra"].items():
        assert_equal_tree(metadata["extra"][key], value, f"progress.{key}")
    for key in ("step", "epoch"):
        assert_equal_tree(metadata[key], payload[key], key)
    proof = {
        "checkpoint_step": metadata["step"],
        "checkpoint_epoch": metadata["epoch"],
        "old_parameter_tensors_preserved": len(payload["model"]),
        "old_optimizer_parameter_states_preserved": len(old_states),
        "new_auxiliary_parameter_tensors": len(new_states.keys() - old_states.keys()),
        "ema_updates_preserved": ema.num_updates,
        "scheduler_age_preserved": scheduler.last_epoch,
        "gradient_clipping_state_preserved": clipper is not None,
        "progress_metadata_preserved": True,
    }
    return model, reference, optimizer, scheduler, ema, clipper, proof


def reconstruct_observed_future_targets(rows: list[ReplaySample]) -> list[ReplaySample]:
    """Read-only smoke enrichment of historical contiguous same-game decisions."""
    groups: dict[tuple, list[ReplaySample]] = {}
    for sample in rows:
        key = (
            sample.run_id,
            sample.generation_family,
            sample.actor_id,
            sample.generation,
            sample.game_id,
            sample.model_identity,
            sample.rings,
            sample.variant_label,
        )
        groups.setdefault(key, []).append(sample)
    result = []
    for group in groups.values():
        group.sort(key=lambda sample: sample.ply)
        if any(b.ply != a.ply + 1 for a, b in zip(group, group[1:])):
            continue
        decisions = [
            _Decision(
                s.to_position(),
                s.policy if s.target_mask & TARGET_POLICY else None,
                True,
                0,
                0,
                0,
                s.ply,
                s.policy_weight,
                0.0,
                swapped="swap=taken" in s.search_provenance.split(":"),
            )
            for s in group
        ]
        result.extend(
            replace(sample, **future)
            for sample, future in zip(
                group, _future_policy_targets(decisions), strict=True
            )
        )
    return result


def load_smoke_replay(run_root: Path) -> tuple[list[ReplaySample], list[dict]]:
    replay_root = (run_root / "replay").resolve()
    identity = json.loads((run_root / "run.json").read_text())
    database = replay_root / "manifest.sqlite3"
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='game_publications'"
            ).fetchone()
            is None
        ):
            raise ValueError("smoke requires a finalized replay publication ledger")
        candidates: dict[str, list[dict]] = {}
        for mode in ("classic", "double"):
            deadline = time.monotonic() + REPLAY_QUERY_TIMEOUT_SECONDS
            connection.set_progress_handler(
                lambda: int(time.monotonic() > deadline), 10000
            )
            # The run/family publication index and shard primary key identify
            # complete game heads. Shutdown-flushed pending prefixes never
            # consume this bounded per-mode candidate allowance.
            rows = connection.execute(
                "SELECT p.game_id,p.sample_count,s.relative_path,s.checksum_sha256,"
                "s.variant,s.actor_id,s.generation,s.model_identity "
                "FROM game_publications p JOIN shards s ON s.id=p.latest_shard_id "
                "WHERE p.run_id=? AND p.generation_family=? AND p.finalized=1 "
                "AND s.run_id=p.run_id AND s.generation_family=p.generation_family "
                "AND s.state='ready' AND s.ring=10 AND s.sample_count=p.sample_count "
                "AND (s.variant=? OR s.variant LIKE ?) ORDER BY s.created_ns DESC LIMIT ?",
                (
                    identity["run_id"],
                    identity["generation_family"],
                    mode,
                    "%-" + mode,
                    MAX_REPLAY_SHARDS // 2,
                ),
            ).fetchall()
            candidates[mode] = [dict(row) for row in rows]
    except sqlite3.OperationalError as error:
        raise ValueError(f"bounded finalized replay query failed: {error}") from error
    finally:
        connection.close()
    selected: dict[str, list[ReplaySample]] = {"classic": [], "double": []}
    evidence = []
    for mode in ("classic", "double"):
        for candidate in candidates[mode]:
            path = replay_root / candidate["relative_path"]
            if path.is_symlink() or not path.resolve().is_relative_to(replay_root):
                raise ValueError("smoke replay path escapes run root")
            if sha256_file(path) != candidate["checksum_sha256"]:
                raise ValueError("smoke replay shard checksum mismatch")
            stored_rows = read_replay_shard(path)
            # Both append paths require an entire contiguous single-game
            # sequence in its authoritative final revision. Never combine
            # superseded versions or guess across an absent decision.
            if len(stored_rows) != candidate["sample_count"] or any(
                sample.ply != index
                or sample.game_id != candidate["game_id"]
                or sample.mode != mode
                or sample.rings != 10
                or sample.run_id != identity["run_id"]
                or sample.generation_family != identity["generation_family"]
                or sample.actor_id != candidate["actor_id"]
                or sample.generation != candidate["generation"]
                or sample.model_identity != candidate["model_identity"]
                or sample.variant_label != candidate["variant"]
                or not sample.target_mask & TARGET_OUTCOME
                or [
                    part
                    for part in sample.search_provenance.split(":")
                    if part.startswith("final=")
                ]
                not in (
                    ["final=board-full"],
                    ["final=clinch-loser-fill"],
                    ["final=exact-endgame"],
                )
                for index, sample in enumerate(stored_rows)
            ):
                raise ValueError(
                    "finalized replay head disagrees with its complete game sequence"
                )
            rows = reconstruct_observed_future_targets(stored_rows)
            added = False
            for sample in rows:
                if (
                    sample.target_mask & TARGET_POLICY
                    and sample.final_peries is not None
                    and sample.final_stars is not None
                    and sample.opponent_reply is not None
                ):
                    destination = selected[mode]
                    if len(destination) < 4:
                        destination.append(sample)
                        added = True
                    elif (
                        mode == "double"
                        and sample.second_stone is not None
                        and not any(s.second_stone is not None for s in destination)
                    ):
                        destination[-1] = sample
                        added = True
            if added:
                evidence.append(
                    {
                        "path": str(path),
                        "sha256": candidate["checksum_sha256"],
                        "source": "finalized-publication",
                        "mode": mode,
                        "variant": candidate["variant"],
                        "game_id": candidate["game_id"],
                        "stored_game_rows": len(stored_rows),
                        "ply_first": 0,
                        "ply_last": len(stored_rows) - 1,
                        "reconstructed_opponent_reply_rows": sum(
                            s.opponent_reply is not None for s in rows
                        ),
                        "reconstructed_second_stone_rows": sum(
                            s.second_stone is not None for s in rows
                        ),
                        "selected_fixture_rows": len(selected[mode]),
                    }
                )
            if len(selected[mode]) == 4 and (
                mode != "double"
                or any(s.second_stone is not None for s in selected[mode])
            ):
                break
    samples = selected["classic"] + selected["double"]
    if not all(len(values) == 4 for values in selected.values()) or not any(
        s.second_stone is not None for s in samples
    ):
        raise ValueError(
            "bounded replay scan found insufficient real single/double future targets"
        )
    return samples, evidence


def check_primary_parity(
    model: GraphResTNet,
    reference: GraphResTNet,
    native: Any,
    device: str,
    precision: str,
) -> list[dict]:
    results = []
    model.eval()
    for ring in (4, 6, 8, 10):
        variants = ["pie-classic", "pie-double"]
        if ring == 10:
            variants += [
                f"handicap-{n}-{mode}" for n in (2, 9) for mode in ("classic", "double")
            ]
        for label in variants:
            variant = GameVariant.parse(label)
            states = native.StateBatch(
                ring, 2, mode=variant.mode, handicap=variant.handicap, pie=variant.pie
            )
            states.apply_many([0, 1], [0, 1])
            inputs = encode_batch(positions_from_native(states.data())).to(device)
            with (
                torch.inference_mode(),
                torch.autocast(
                    torch.device(device).type,
                    dtype=torch.bfloat16,
                    enabled=precision == "bf16",
                ),
            ):
                before = reference(*inputs.model_args())
                after = model(*inputs.model_args())
                skipped = model(*inputs.model_args(), include_auxiliary=False)
            for name in PRIMARY_OUTPUTS:
                assert_equal_tree(
                    getattr(after, name), getattr(before, name), f"{label}.{name}"
                )
                assert_equal_tree(
                    getattr(skipped, name),
                    getattr(before, name),
                    f"{label}.leaf.{name}",
                )
            if any(
                getattr(after, name + "_logits") is None for name in AUXILIARY_LOSSES
            ):
                raise AssertionError("auxiliary forward omitted a required head")
            if any(
                getattr(skipped, name + "_logits") is not None
                for name in AUXILIARY_LOSSES
            ):
                raise AssertionError("leaf forward evaluated auxiliary heads")
            results.append(
                {
                    "ring": ring,
                    "variant": label,
                    "precision": precision,
                    "primary_outputs_bitwise_equal": True,
                }
            )
    return results


def check_native_inference(
    model: GraphResTNet, native: Any, device: str, precision: str
) -> list[dict]:
    calls = []
    hook = model.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(kwargs.get("include_auxiliary")),
        with_kwargs=True,
    )
    adapter = GraphInferenceAdapter(
        model,
        device=device,
        config=InferenceConfig(
            precision=precision, cache_max_entries=0, cache_max_bytes=0
        ),
        model_version="auxiliary-deployment-smoke",
        model_identity="auxiliary-deployment-smoke",
    )
    reports = []
    try:
        for mode in ("classic", "double"):
            states = native.StateBatch(4, 1, mode=mode, pie=True)
            states.apply_many([0], [0])
            search = native.SearchBatch(
                states, simulations=2, max_considered=2, deterministic_seed=17
            )
            requests = search.root_requests()
            calls.clear()
            ordinary = adapter.evaluate(requests)
            if not calls or any(value is not False for value in calls):
                raise AssertionError("search leaf path did not skip auxiliary heads")
            calls.clear()
            detailed = adapter.evaluate_detailed(requests)
            if not calls or any(value is not True for value in calls):
                raise AssertionError("detailed inference omitted auxiliary heads")
            assert_equal_tree(
                asdict(detailed.response), asdict(ordinary), "detailed primary response"
            )
            if (
                not detailed.auxiliary_predictions
                or len(detailed.auxiliary_predictions) != 1
            ):
                raise AssertionError("detailed inference omitted predictions")
            predictions = asdict(detailed.auxiliary_predictions[0])
            if not all(np.isfinite(values).all() for values in predictions.values()):
                raise FloatingPointError("nonfinite detailed auxiliary prediction")
            search.initialize_roots(*ordinary.submit_args())
            calls.clear()
            for _ in range(16):
                if search.is_done():
                    break
                pending = search.next_requests()
                if len(pending):
                    search.submit(*adapter.evaluate(pending).submit_args())
            if not search.is_done() or any(value is not False for value in calls):
                raise AssertionError(
                    "native search failed or evaluated auxiliary leaves"
                )
            action = int(search.results().selected_actions[0])
            if (
                action < 0
                or action >= states.node_count
                or positions_from_native(states.data())[0].stones[action] != -1
            ):
                raise AssertionError("native search returned an illegal placement")
            reports.append(
                {
                    "mode": mode,
                    "simulations": 2,
                    "selected_action": action,
                    "leaf_auxiliary_skipped": True,
                    "detailed_auxiliary_present": True,
                }
            )
    finally:
        hook.remove()
        adapter.close()
    return reports


def check_auxiliary_gradients(model: GraphResTNet) -> dict[str, float]:
    """Require connected, finite heads without rejecting shift-invariant biases."""
    gradients: dict[str, float] = {}
    weight_names = []
    for name, parameter in model.named_parameters():
        if not is_auxiliary_parameter(name):
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"invalid auxiliary gradient: {name}")
        norm = float(parameter.grad.float().norm())
        if not math.isfinite(norm):
            raise FloatingPointError(f"nonfinite auxiliary gradient norm: {name}")
        gradients[name] = norm
        if name.endswith(".weight") and parameter.ndim == 2:
            weight_names.append(name)
            if norm <= 0:
                raise AssertionError(
                    f"auxiliary head weight received no training gradient: {name}"
                )
    if not gradients or not weight_names:
        raise AssertionError("model has no auxiliary head gradients")
    return gradients


def validate_runtime_batch_size(size: int) -> None:
    if type(size) is not int or not 1 <= size <= MAX_RUNTIME_BATCH_SIZE:
        raise ValueError(
            f"smoke runtime batch size must be an integer in 1..{MAX_RUNTIME_BATCH_SIZE}"
        )


def build_runtime_smoke_batch(
    samples: list[ReplaySample],
    size: int,
) -> tuple[ReplayBatch, dict[str, int | bool]]:
    """Repeat read-only fixtures to exercise the configured learner shape."""
    validate_runtime_batch_size(size)
    if not samples:
        raise ValueError("smoke requires at least one real replay fixture")
    runtime_samples = [samples[index % len(samples)] for index in range(size)]
    distinct = len(
        {
            (
                sample.run_id,
                sample.generation_family,
                sample.actor_id,
                sample.generation,
                sample.game_id,
                sample.ply,
            )
            for sample in runtime_samples
        }
    )
    return collate_replay_samples(runtime_samples), {
        "distinct_fixture_samples": distinct,
        "runtime_batch_size": size,
        "repeated_smoke_fixtures": size > distinct,
    }


def run_smoke(args: Any) -> dict:
    from scripts.benchmark_actor_throughput import _gpu_ownership

    started = time.time_ns()
    config = load_config(args.profile)
    validate_output(args.output, config, args.profile, args.checkpoint)
    pinned = require_stopped_run(config, args.checkpoint)
    profile_hash = sha256_file(args.profile)
    if config.train.precision != "bf16" or not config.model.auxiliary_predictions:
        raise ValueError("smoke requires the auxiliary-enabled BF16 deployment profile")
    if any(getattr(config.loss, name) <= 0 for name in AUXILIARY_LOSSES):
        raise ValueError("smoke requires all five auxiliary losses enabled")
    validate_runtime_batch_size(config.train.per_rank_batch_size)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU tests do not qualify deployment")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    device = "cuda:0"
    torch.cuda.set_device(device)
    context = torch.empty(1, device=device)
    gpu_uuid = str(torch.cuda.get_device_properties(device).uuid)
    if not gpu_uuid.startswith("GPU-"):
        gpu_uuid = "GPU-" + gpu_uuid
    ownership_before = _gpu_ownership(gpu_uuid)
    if ownership_before.get("verified") is not True:
        raise RuntimeError(f"smoke requires an exclusive GPU: {ownership_before}")
    state = load_training_state(config, args.checkpoint, device)
    model, reference, optimizer, scheduler, ema, clipper, preservation = state
    native = load_star_native(required=True)
    if native is None:
        raise RuntimeError("native extension is unavailable")
    parity = check_primary_parity(model, reference, native, device, "bf16")
    native_checks = check_native_inference(model, native, device, "bf16")
    # Proof records contain only scalars. Release the comparison model before
    # compiling and exercising the complete production learner batch.
    del reference, state
    gc.collect()
    torch.cuda.empty_cache()
    samples, replay_evidence = load_smoke_replay(
        Path(config.orchestration.directories.root)
    )
    batch, fixture_report = build_runtime_smoke_batch(
        samples, config.train.per_rank_batch_size
    )
    masks = {
        name: int(getattr(batch.targets, name + "_mask").sum())
        for name in AUXILIARY_LOSSES
    }
    if not all(masks.values()):
        raise AssertionError("smoke batch lacks an auxiliary supervision class")
    torch.cuda.reset_peak_memory_stats(device)
    compiled = maybe_compile_model(
        model,
        enabled=config.train.compile,
        dynamic=False,
        recompile_limit=len(config.orchestration.ring_mixture.rings),
        isolate_recompiles=True,
    )
    compiled.train()
    step = train_step(
        compiled,
        batch,
        optimizer,
        precision="bf16",
        loss_weights=config.loss,
        gradient_clip_norm=config.train.gradient_clip_norm,
        scheduler=scheduler,
        ema=ema,
        gradient_clipper=clipper,
        share_homogeneous_geometry=config.train.share_homogeneous_geometry,
    )
    torch.cuda.synchronize()
    losses = step.losses
    if not all(np.isfinite(value) for value in losses.values()):
        raise FloatingPointError("nonfinite smoke training loss")
    gradients = check_auxiliary_gradients(model)
    for artifact in replay_evidence:
        if sha256_file(Path(artifact["path"])) != artifact["sha256"]:
            raise RuntimeError("smoke replay bytes changed")
    if (
        require_stopped_run(config, args.checkpoint) != pinned
        or sha256_file(args.profile) != profile_hash
    ):
        raise RuntimeError("smoke stopped-run boundary or profile changed")
    if sha256_file(args.checkpoint) != pinned["checkpoint_sha256"]:
        raise RuntimeError("smoke changed source checkpoint")
    ownership_after = _gpu_ownership(gpu_uuid)
    if ownership_after.get("verified") is not True:
        raise RuntimeError("smoke lost exclusive GPU ownership")
    del context
    return {
        "status": "passed",
        **pinned,
        "profile_sha256": profile_hash,
        "state_preservation": preservation,
        "primary_parity": parity,
        "native_search_checks": native_checks,
        "replay_sources": replay_evidence,
        "training_samples": config.train.per_rank_batch_size,
        **fixture_report,
        "fixture_note": "Replay fixtures are repeated only for runtime-shape qualification; no replay or training state is written.",
        "target_masks": masks,
        "learner_losses": losses,
        "auxiliary_gradient_norms": gradients,
        "compile_configuration": {
            "enabled": config.train.compile,
            "dynamic": False,
            "isolate_recompiles": True,
        },
        "precision": "bf16",
        "training_steps": 1,
        "gpu_ownership_before": ownership_before,
        "gpu_ownership_after": ownership_after,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "started_ns": started,
        "completed_ns": time.time_ns(),
        "training_artifacts_written": False,
        "qualification": "correctness only; not strength or throughput",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_config(args.profile)
    validate_output(args.output, config, args.profile, args.checkpoint)
    # Refuse before creating a CUDA worker or a compile-cache directory.
    require_stopped_run(config, args.checkpoint)
    if args.worker:
        print(json.dumps(run_smoke(args), allow_nan=False))
        return
    from scripts.benchmark_actor_throughput import (
        _controller_signals,
        _gpu_ownership,
        _run_owned,
    )

    command = [
        sys.executable,
        "-m",
        "scripts.smoke_auxiliary_predictions_cuda",
        "--profile",
        str(args.profile),
        "--checkpoint",
        str(args.checkpoint),
        "--output",
        str(args.output),
        "--worker",
    ]
    with (
        _controller_signals(),
        tempfile.TemporaryDirectory(prefix="startrain-auxiliary-smoke-") as cache,
    ):
        environment = dict(
            os.environ,
            TORCHINDUCTOR_CACHE_DIR=cache,
            TRITON_CACHE_DIR=str(Path(cache) / "triton"),
        )
        try:
            result = _run_owned(command, env=environment, timeout=TIMEOUT_SECONDS)
            if result.returncode:
                raise RuntimeError(result.stderr[-6000:])
            report = json.loads(result.stdout.strip().splitlines()[-1])
            if report.get("status") != "passed":
                raise ValueError("smoke worker did not pass")
            current = require_stopped_run(config, args.checkpoint)
            if (
                any(report.get(k) != v for k, v in current.items())
                or sha256_file(args.profile) != report["profile_sha256"]
            ):
                raise RuntimeError(
                    "stopped-run boundary changed after worker completion"
                )
            if (
                _gpu_ownership(report["gpu_ownership_after"]["gpu_uuid"]).get(
                    "owner_pids"
                )
                != []
            ):
                raise RuntimeError("smoke worker left GPU processes running")
            report["parent_boundary_revalidated"] = True
        except (
            subprocess.TimeoutExpired,
            RuntimeError,
            ValueError,
            IndexError,
        ) as error:
            report = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "training_artifacts_written": False,
            }
        report["child_timeout_seconds"] = TIMEOUT_SECONDS
    publish_report(args.output, report)
    print(json.dumps(report, allow_nan=False))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
