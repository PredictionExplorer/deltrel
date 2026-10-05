#!/usr/bin/env python3
"""Observe an isolated checkpoint restore/step/search; never authorize migration.

The caller owns GPU fencing, the process deadline and read-only input mounts.
Only its independent guard may attest that workers/GPUs have been released.
CPU validation restores and checks inputs without executing a production batch.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, fields, replace
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import torch

from deltreltrain import replay
from deltreltrain.checkpoint import ExponentialMovingAverage, load_ema_checkpoint
from deltreltrain.config import ExperimentConfig, load_config
from deltreltrain.contracts import (
    FEATURE_SCHEMA_HASH,
    RULES_HASH_WIRE,
    SEARCH_ALGORITHM_ID,
)
from deltreltrain.features import EncodedBatch
from deltreltrain.learner import LearnerLoop
from deltreltrain.losses import TrainingTargets
from deltreltrain.model import GraphResTNet
from deltreltrain.optim import build_optimizer
from deltreltrain.replay_store import ReplayStore
from deltreltrain.runtime import RunIdentity
from deltreltrain.training import (
    build_scheduler,
    configure_isolated_compile_cache,
    maybe_compile_model,
    train_step,
)

CHALLENGE_FORMAT = "strength-freshness-probe-challenge-v1"
RESULT_FORMAT = "strength-freshness-probe-result-v1"
BATCH_FORMAT = "strength-freshness-probe-batch-v1"
ADMISSION_FIELDS = {
    "plan_sha256",
    "source_commit",
    "source_manifest_sha256",
    "profile_sha256",
    "recovery_pointer_sha256",
    "execution_pins",
    "pyvenv",
    "training_module",
    "native_file",
    "native_binaries",
}
CHECKS = (
    "cuda_available",
    "learner_resume_step",
    "ema_preserved",
    "native_cuda_search",
)
MAX_EVIDENCE = 16 * 1024**2


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def encoded(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def read_metadata(path: Path) -> bytes:
    if (
        path.resolve() != path
        or not path.is_file()
        or path.stat().st_size > 2 * 1024**2
    ):
        raise ValueError("unsafe or oversized probe metadata")
    return path.read_bytes()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(read_metadata(path))
    if not isinstance(value, dict):
        raise ValueError("probe metadata must be an object")
    return value


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def write_new(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Observations:
    def __init__(self, output: Path, deadline: float):
        self.path = output / "observations.jsonl"
        self.deadline = deadline
        self.sequence = 0
        self.bytes = 0
        self.stream = self.path.open("xb")

    def check_time(self) -> None:
        if time.monotonic() >= self.deadline:
            raise TimeoutError("probe deadline expired")

    def record(self, stage: str, value: object) -> None:
        self.sequence += 1
        data = encoded(
            {
                "sequence": self.sequence,
                "monotonic": time.monotonic(),
                "stage": stage,
                "observed": value,
            }
        )
        self.bytes += len(data)
        if self.bytes > MAX_EVIDENCE:
            raise ValueError("probe evidence byte budget exhausted")
        self.stream.write(data)
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def append(self, value: object) -> None:
        # Learner resume may report governor adoption; this is an isolated sink.
        self.record("resume-runtime-event", value)

    def close(self) -> None:
        self.stream.close()


def pinned(pin: dict[str, Any], observations: Observations, *, maximum: int) -> Path:
    path = Path(pin["path"])
    if not path.is_absolute() or path.resolve() != path or not path.is_file():
        raise ValueError("probe input is not a canonical regular file")
    size = path.stat().st_size
    if type(pin.get("bytes")) is not int or not 0 < size <= maximum:
        raise ValueError("probe input size is invalid")
    observations.check_time()
    actual = {"path": str(path), "bytes": size, "sha256": digest(path)}
    observations.record("input-artifact", {"actual": actual, "expected": pin})
    if actual["bytes"] != pin["bytes"] or actual["sha256"] != pin["sha256"]:
        raise ValueError("probe input differs from its pin")
    observations.check_time()
    return path


def tree_digest(value: object) -> str:
    """Device-independent tensor/value identity, without saving model payloads."""
    result = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            result.update(
                encoded({"tensor": str(tensor.dtype), "shape": list(tensor.shape)})
            )
            result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            result.update(b"mapping")
            for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            result.update(encoded(item))

    visit(value)
    return result.hexdigest()


def tree_finite(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(tree_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(tree_finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def _tensor_map(
    value: object, names: set[str], *, optional: set[str]
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError("collated tensor field set differs from the runtime dataclass")
    for name, tensor in value.items():
        if name in optional and tensor is None:
            continue
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.layout != torch.strided
            or tensor.numel() > 16_000_000
        ):
            raise ValueError("collated batch must contain bounded dense CPU tensors")
    return value


def load_batch(
    pin: dict[str, Any], observations: Observations
) -> tuple[replay.ReplayBatch, dict[str, Any]]:
    manifest = read_json(pinned(pin, observations, maximum=2 * 1024**2))
    if (
        manifest.get("format") != BATCH_FORMAT
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
        or manifest.get("rules_hash") != RULES_HASH_WIRE
        or manifest.get("feature_schema_hash") != FEATURE_SCHEMA_HASH
        or type(manifest.get("rows")) is not int
        or not 1 <= manifest["rows"] <= 512
        or manifest.get("rings") not in (4, 6, 8, 10)
    ):
        raise ValueError("collated batch contract is incompatible")
    path = pinned(manifest["artifact"], observations, maximum=128 * 1024**2)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {"schema_version", "inputs", "targets", "variant_labels", "feature_path"}
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
    ):
        raise ValueError("invalid collated tensor-map schema")
    inputs = EncodedBatch(
        **_tensor_map(
            payload["inputs"], {f.name for f in fields(EncodedBatch)}, optional=set()
        )
    )
    targets = TrainingTargets(
        **_tensor_map(
            payload["targets"],
            {f.name for f in fields(TrainingTargets)},
            optional={f.name for f in fields(TrainingTargets) if f.default is None},
        )
    )
    geometry = replay._validate_homogeneous_geometry(inputs)
    labels = payload["variant_labels"]
    if (
        inputs.batch_size != manifest["rows"]
        or geometry is None
        or geometry.ring != manifest["rings"]
        or payload["feature_path"] not in ("rust", "python")
        or labels is not None
        and (
            not isinstance(labels, list)
            or len(labels) != inputs.batch_size
            or not all(isinstance(s, str) for s in labels)
        )
    ):
        raise ValueError("collated batch size, labels or geometry differs")
    batch = replay.ReplayBatch(
        inputs,
        targets,
        feature_path=payload["feature_path"],
        variant_labels=tuple(labels) if labels is not None else None,
        _homogeneous_geometry=geometry,
    )
    provenance = manifest["provenance"]
    rows = provenance["ordered_rows"]
    shards = provenance["shards"]
    if (
        not isinstance(rows, list)
        or len(rows) != inputs.batch_size
        or not isinstance(shards, list)
        or not 1 <= len(shards) <= 32
    ):
        raise ValueError("real replay row provenance is incomplete")
    by_hash = {pin["sha256"]: pin for pin in shards}
    if len(by_hash) != len(shards) or provenance["collator_module_sha256"] != digest(
        Path(replay.__file__).resolve()
    ):
        raise ValueError("source shards or collator identity differ")
    seen = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"shard_sha256", "row_index"}
            or row["shard_sha256"] not in by_hash
        ):
            raise ValueError("source row has no pinned shard")
        index = row["row_index"]
        pin = by_hash[row["shard_sha256"]]
        if (
            type(index) is not int
            or index < 0
            or ("sample_count" in pin and index >= pin["sample_count"])
        ):
            raise ValueError("source row index is invalid")
        key = (row["shard_sha256"], index)
        if key in seen:
            raise ValueError("probe batch must preserve unique real source rows")
        seen.add(key)
    observations.record(
        "collated-batch",
        {
            "rows": inputs.batch_size,
            "rings": batch.homogeneous_ring,
            "feature_path": batch.feature_path,
            "tensor_digest": tree_digest(payload),
            "finite": tree_finite(payload),
            "target_masks": {
                f.name: int(getattr(targets, f.name).sum())
                for f in fields(targets)
                if f.name.endswith("_mask")
                and isinstance(getattr(targets, f.name), torch.Tensor)
            },
        },
    )
    if not tree_finite(payload):
        raise ValueError("collated input contains nonfinite values")
    return batch, manifest


def verify_collation(
    batch: replay.ReplayBatch, manifest: dict[str, Any], observations: Observations
) -> None:
    """Preparatory CPU verification only; never rebuild a live ReplayStore."""
    by_hash = {}
    provenance = manifest["provenance"]
    if provenance["collator_module_sha256"] != digest(Path(replay.__file__).resolve()):
        raise ValueError("batch was collated by a different runtime source")
    for pin in provenance["shards"]:
        path = pinned(pin, observations, maximum=64 * 1024**2)
        by_hash[pin["sha256"]] = replay.read_replay_shard(path)
    samples = []
    for row in provenance["ordered_rows"]:
        index = row["row_index"]
        if type(index) is not int or index < 0:
            raise ValueError("invalid source replay row index")
        samples.append(by_hash[row["shard_sha256"]][index])
    observations.check_time()
    rebuilt = replay.collate_replay_samples(
        samples, prefer_native=batch.feature_path == "rust"
    )

    def tensors(value):
        return {f.name: getattr(value, f.name) for f in fields(value)}

    expected = tree_digest(
        {
            "inputs": tensors(rebuilt.inputs),
            "targets": tensors(rebuilt.targets),
            "labels": rebuilt.variant_labels,
        }
    )
    actual = tree_digest(
        {
            "inputs": tensors(batch.inputs),
            "targets": tensors(batch.targets),
            "labels": batch.variant_labels,
        }
    )
    observations.record(
        "source-row-recollation", {"expected": expected, "actual": actual}
    )
    if expected != actual or rebuilt.feature_path != batch.feature_path:
        raise ValueError("tensor batch differs from its real source rows")


def restore(
    config: ExperimentConfig,
    checkpoint: Path,
    checkpoint_pin: dict[str, Any],
    identity: dict[str, Any],
    device: str,
    observations: Observations,
) -> tuple[LearnerLoop, dict[str, Any]]:
    """Use the production learner constructor/resume on an empty scratch store.

    Build explicitly to avoid from_experiment.seed_all touching an available MPS
    backend during CPU-only validation. No loader or learner.run is invoked.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = GraphResTNet(config.model).to(device)
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.train.scheduler)
    ema = ExponentialMovingAverage(model, decay=config.train.resolved_ema_decay(1))
    scratch = observations.path.parent / "scratch"
    scratch.mkdir(exist_ok=False)
    run = RunIdentity(
        scratch / "run.json",
        identity["run_id"],
        identity["generation_family"],
        identity["created_ns"],
    )
    observations.record(
        "checkpoint-before-restore",
        {
            "step": payload.get("step"),
            "epoch": payload.get("epoch"),
            "extra": payload.get("extra"),
            "device": device,
        },
    )
    observations.check_time()
    with ReplayStore(scratch / "replay") as store:
        state = LearnerLoop(
            store=store,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            output_directory=scratch / "learner",
            learner_config=replace(config.learner, device=device),
            train_config=replace(config.train, compile=False)
            if device == "cpu"
            else config.train,
            data_config=replace(config.data, workers=0, pin_memory=False),
            loss_weights=config.loss,
            seed=config.train.seed,
            serialized_config=config.as_dict(),
            run_identity=run,
            ring_mixture_config=config.orchestration.ring_mixture,
        )
        state.resume(
            checkpoint,
            expected_sha256=checkpoint_pin["sha256"],
            expected_bytes=checkpoint_pin["bytes"],
        )
    actual = {
        "model": model.state_dict(),
        "optimizer": state.optimizer.state_dict(),
        "scheduler": state.scheduler.state_dict(),
        "ema": state.ema.state_dict(),
        "gradient_clipping": state.gradient_clipper.state_dict()
        if state.gradient_clipper
        else None,
    }
    comparison = {
        name: {
            "expected": tree_digest(payload.get(name)),
            "actual": tree_digest(value),
            "finite": tree_finite(value),
        }
        for name, value in actual.items()
    }
    observations.record(
        "restored-state",
        {
            "comparisons": comparison,
            "step": state.step,
            "epoch": state.epoch,
            "examples_consumed": state.examples_consumed,
            "learning_rates": [group["lr"] for group in state.optimizer.param_groups],
            "ema_decay": state.ema.decay,
            "ema_updates": state.ema.num_updates,
            "model_device": str(next(state.model.parameters()).device),
            "runtime_compile": state.train_config.compile,
            "serialized_profile_compile": config.train.compile,
        },
    )
    if any(
        row["expected"] != row["actual"] or not row["finite"]
        for row in comparison.values()
    ):
        raise ValueError(
            "restored model/optimizer/scheduler/EMA/clipping state differs"
        )
    continuation_observations(state, payload, observations)
    observations.check_time()
    return state, payload


def continuation_observations(
    state: LearnerLoop, payload: dict[str, Any], observations: Observations
) -> None:
    from deltreltrain.lr_governor import governor_from_checkpoint_extra
    from deltreltrain.plateau_evidence import receipts_from_checkpoint_extra

    metadata = {name: payload[name] for name in ("step", "epoch", "config", "extra")}
    metadata.update(
        scheduler_step=payload["scheduler"]["last_epoch"],
        ema_decay=payload["ema"]["decay"],
    )
    extra = payload["extra"]

    def materialize(value):
        return value.as_dict() if value is not None else None

    expected = {
        "utd_target": state._checkpoint_utd_target(metadata),
        "utd_segment": materialize(state._checkpoint_utd_segment(metadata)),
        "segment_baseline_examples": state._checkpoint_segment_baseline(
            metadata, examples_consumed=state._resume_examples_consumed(metadata)
        ),
        "reuse_clock": materialize(state._prospective_reuse_clock(metadata)),
        "governor": governor_from_checkpoint_extra(extra, state.scheduler).as_dict(),
        "plateau_receipts": receipts_from_checkpoint_extra(extra),
        "auxiliary_upgrade": extra.get("auxiliary_upgrade"),
        "auxiliary_supervision": extra.get("auxiliary_supervision"),
    }
    actual = {
        "utd_target": state._resume_utd_target,
        "utd_segment": materialize(state._resume_utd_segment_state),
        "segment_baseline_examples": state._segment_baseline_examples,
        "reuse_clock": materialize(state._reuse_clock),
        "governor": state._lr_governor.as_dict(),
        "plateau_receipts": state._plateau_verdict_receipts,
        "auxiliary_upgrade": getattr(state, "_auxiliary_upgrade", None),
        "auxiliary_supervision": getattr(state, "_auxiliary_supervision", None),
    }
    observations.record(
        "restored-continuation",
        {
            "expected": expected,
            "actual": actual,
            "replay_credit_scope": "Live credit is bound by the guard; scratch-store totals are not continuation evidence.",
        },
    )
    if tree_digest(actual) != tree_digest(expected):
        raise ValueError("learner did not adopt exact continuation metadata")


def backend_math_flags() -> dict[str, Any]:
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def restore_math_flags(flags: dict[str, Any]) -> None:
    torch.backends.cuda.matmul.allow_tf32 = flags["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = flags["cudnn_allow_tf32"]
    torch.set_float32_matmul_precision(flags["float32_matmul_precision"])
    if backend_math_flags() != flags:
        raise ValueError("actor-equivalent backend math flags did not restore")


def training_observations(
    state: LearnerLoop,
    config: ExperimentConfig,
    batch: replay.ReplayBatch,
    observations: Observations,
) -> dict[str, Any]:
    before_parameters = tree_digest(dict(state.model.named_parameters()))
    before_scheduler = state.scheduler.last_epoch
    before_updates = state.ema.num_updates
    reference = ExponentialMovingAverage(state.model, decay=state.ema.decay)
    reference.load_state_dict(state.ema.state_dict())
    observations.record(
        "train-step-start",
        {
            "restored_step": state.step,
            "rows": batch.inputs.batch_size,
            "rings": batch.homogeneous_ring,
            "precision": config.train.precision,
            "compile": config.train.compile,
        },
    )
    observations.check_time()
    result = train_step(
        state.compiled_model,
        batch,
        state.optimizer,
        loss_weights=config.loss,
        precision=config.train.precision,
        gradient_clip_norm=config.train.gradient_clip_norm,
        scheduler=state.scheduler,
        ema=state.ema,
        trusted_batch=False,
        gradient_clipper=state.gradient_clipper,
        share_homogeneous_geometry=config.train.share_homogeneous_geometry,
    )
    if next(state.model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()
    reference.update(state.model)
    after_parameters = tree_digest(dict(state.model.named_parameters()))
    value = {
        "model_changed": after_parameters != before_parameters,
        "parameter_hash_before": before_parameters,
        "parameter_hash_after": after_parameters,
        "losses": result.to_host().losses,
        "gradient_norm": float(result.gradient_norm_tensor),
        "model_finite": tree_finite(state.model.state_dict()),
        "optimizer_finite": tree_finite(state.optimizer.state_dict()),
        "scheduler_before": before_scheduler,
        "scheduler_after": state.scheduler.last_epoch,
        "ema_before": before_updates,
        "ema_after": state.ema.num_updates,
        "ema_expected": tree_digest(reference.state_dict()),
        "ema_actual": tree_digest(state.ema.state_dict()),
        "restored_step": state.step,
        "diagnostic_optimizer_steps": 1,
        "live_updates": 0,
        "model_device": str(next(state.model.parameters()).device),
    }
    observations.record("train-step-result", value)
    if (
        not value["model_changed"]
        or not value["model_finite"]
        or not value["optimizer_finite"]
        or not math.isfinite(value["gradient_norm"])
        or not tree_finite(value["losses"])
        or state.scheduler.last_epoch != before_scheduler + 1
        or state.ema.num_updates != before_updates + 1
        or value["ema_actual"] != value["ema_expected"]
    ):
        raise ValueError("diagnostic production training/EMA step failed")
    observations.check_time()
    return value


def native_observations(
    config: ExperimentConfig,
    checkpoint: Path,
    checkpoint_pin: dict[str, Any],
    identity: dict[str, Any],
    device: str,
    observations: Observations,
) -> dict[str, Any]:
    from deltreltrain.actor import resolve_actor_experiment
    from deltreltrain.inference import GraphInferenceAdapter, InferenceConfig
    from deltreltrain.native import validate_native_module

    package = ExperimentConfig.__module__.split(".")[0]
    native = importlib.import_module(package.removesuffix("train") + "_native")
    validate_native_module(native)
    gpu = next(gpu for gpu in config.orchestration.gpus if gpu.role == "actor")
    actor = resolve_actor_experiment(config, gpu)
    execution = actor.orchestration.model_refresh
    inference = execution.inference
    model = GraphResTNet(config.model).to(device).eval()
    load_ema_checkpoint(
        checkpoint,
        model=model,
        expected_model_config=config.as_dict()["model"],
        expected_game_config=config.as_dict()["game"],
        expected_run_id=identity["run_id"],
        expected_generation_family=identity["generation_family"],
        expected_sha256=checkpoint_pin["sha256"],
        expected_bytes=checkpoint_pin["bytes"],
        map_location=device,
    )
    compiled = maybe_compile_model(
        model,
        enabled=config.train.compile,
        dynamic=execution.inference_compile_dynamic,
        fullgraph=True,
        mode=execution.inference_compile_mode,
        recompile_limit=None
        if execution.inference_compile_dynamic
        else len(config.game.rings),
        isolate_recompiles=not execution.inference_compile_dynamic,
    )
    settings = {
        f.name: getattr(inference, f.name)
        for f in fields(InferenceConfig)
        if hasattr(inference, f.name)
    }
    settings.update(
        precision=config.train.precision,
        score_utility_weight=config.selfplay.score_utility_weight,
    )
    evaluator = GraphInferenceAdapter(
        compiled,
        device=device,
        model_identity="sha256-" + checkpoint_pin["sha256"],
        config=InferenceConfig(**settings),
        homogeneous_relational_bias=inference.homogeneous_relational_bias,
    )
    searches = 0
    try:
        # Deliberately bounded diagnostic searches, never a production profile edit.
        for ring in (4, 6, 8, 10):
            for mode in ("classic", "double"):
                for phase in ("pie-opening", "pie-responder", "handicap"):
                    observations.check_time()
                    states = native.StateBatch(
                        ring,
                        1,
                        mode=mode,
                        handicap=9 if phase == "handicap" else 1,
                        pie=phase != "handicap",
                    )
                    if phase == "pie-responder":
                        states.apply_many([0], [0])
                    search = native.SearchBatch(
                        states,
                        simulations=16,
                        max_considered=4,
                        c_visit=config.selfplay.c_visit,
                        c_scale=config.selfplay.c_scale,
                        deterministic_seed=17 + searches,
                        pda_by_seat=[(0, 0)],
                        first_visit_batch_size=config.selfplay.search_execution.first_visit_batch_size,
                    )
                    evaluator.clear_prediction_cache()
                    search.initialize_roots(
                        *evaluator.evaluate(search.root_requests()).submit_args()
                    )
                    iterations = 0
                    while not search.is_done():
                        observations.check_time()
                        iterations += 1
                        if iterations > 80:
                            raise ValueError(
                                "native probe search did not make bounded progress"
                            )
                        requests = search.next_requests(max_rows=8)
                        if len(requests):
                            search.submit(*evaluator.evaluate(requests).submit_args())
                    result = search.results()
                    selected = int(result.selected_actions[0])
                    actions = list(result.actions)
                    counts = list(search.completed_simulations)
                    policy = list(result.policy_target)
                    raw = {
                        "ring": ring,
                        "mode": mode,
                        "phase": phase,
                        "simulations": counts,
                        "selected": selected,
                        "actions": actions,
                        "policy": policy,
                        "root_values": list(result.root_values),
                        "q_values": list(result.q_values),
                        "visits": list(result.visits),
                    }
                    observations.record("native-search", raw)
                    if (
                        counts != [16]
                        or selected not in actions
                        or not policy
                        or len(policy) != len(actions)
                        or not tree_finite(raw)
                        or abs(sum(policy) - 1) > 1e-5
                        or sum(result.visits) != 16
                    ):
                        raise ValueError(
                            "native probe result violates its search contract"
                        )
                    searches += 1
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        metrics = asdict(evaluator.metrics_snapshot())
        value = {
            "searches": searches,
            "device": str(evaluator.device),
            "native_file": str(Path(native.__file__ or "").resolve()),
            "metrics": metrics,
            "cuda_graphs_configured": inference.cuda_graphs,
        }
        observations.record("native-summary", value)
        if device.startswith("cuda") and (
            not inference.cuda_graphs
            or metrics["graph_captures"] <= 0
            or metrics["graph_replays"] <= 0
            or metrics["graph_validation_failures"]
            or metrics["graph_fallbacks"]
            or metrics["neural_calls"] <= 0
        ):
            raise ValueError(
                "native CUDA execution/capture/replay was not demonstrated"
            )
        return value
    finally:
        evaluator.close()


def runtime_observations(challenge: dict[str, Any], observations: Observations) -> None:
    """Observe the child imports/binary bytes, rather than echo a parent claim."""
    expected = challenge["runtime"]
    root = Path(challenge["runtime_root"])
    package_name = ExperimentConfig.__module__.split(".")[0]
    package = importlib.import_module(package_name)
    native_name = package_name.removesuffix("train") + "_native"
    native = importlib.import_module(native_name)
    from deltreltrain.native import validate_native_module

    validate_native_module(native)
    if not isinstance(package.__file__, str) or not isinstance(native.__file__, str):
        raise ValueError("runtime imports lack concrete source paths")
    marker = pinned(expected["source_commit_file"], observations, maximum=1024)
    source_manifest = pinned(
        expected["source_manifest"], observations, maximum=2 * 1024**2
    )
    pyvenv = pinned(expected["pyvenv"], observations, maximum=65536)
    training = pinned(expected["training_module"], observations, maximum=2 * 1024**2)
    wrapper = pinned(expected["native_wrapper"], observations, maximum=2 * 1024**2)
    interpreter = Path(sys.executable).absolute()
    python = expected["python"]
    actual_python = {
        "path": str(interpreter),
        "resolved_path": str(interpreter.resolve()),
        "sha256": digest(interpreter.resolve()),
        "bytes": interpreter.stat().st_size,
    }
    loaded = {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if (
            (name == native_name or name.startswith(native_name + "."))
            and isinstance(filename, str)
            and Path(filename).suffix in (".so", ".pyd")
        ):
            path = Path(filename).resolve()
            loaded[str(path)] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
    expected_native = {pin["path"]: pin for pin in expected["native_binaries"]}
    actual = {
        "source_commit": marker.read_text().strip(),
        "source_manifest_sha256": digest(source_manifest),
        "python": actual_python,
        "training_module": str(Path(package.__file__).resolve()),
        "native_file": str(Path(native.__file__).resolve()),
        "native_binaries": list(loaded.values()),
        "rules_hash": native.native_rules_hash_tag(),
        "feature_schema_hash": native.native_feature_schema_hash(),
        "search_algorithm": native.native_search_algorithm_id(),
        "torch_version": importlib.metadata.version("torch"),
        "torch_runtime_version": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "torch_import": str(Path(torch.__file__).resolve()),
        "cuda_initialized": torch.cuda.is_initialized(),
        "sys_prefix": str(Path(sys.prefix).resolve()),
    }
    source_hashes = dict(
        line.split(maxsplit=1)[::-1]
        for line in source_manifest.read_text().splitlines()
    )
    imports = []
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if (name == package_name or name.startswith(package_name + ".")) and isinstance(
            filename, str
        ):
            path = Path(filename).resolve()
            if not path.is_relative_to(root):
                raise ValueError("runtime submodule import escaped qualified source")
            relative = str(path.relative_to(root.parent))
            actual_hash = digest(path)
            imports.append(
                {
                    "module": name,
                    "path": str(path),
                    "sha256": actual_hash,
                    "expected_sha256": source_hashes.get(relative),
                }
            )
    actual["runtime_imports"] = imports
    observations.record("runtime-observed", actual)
    if (
        actual["source_commit"] != expected["source_commit"]
        or actual_python != python
        or actual["training_module"] != str(training)
        or actual["native_file"] != str(wrapper)
        or pyvenv != Path(sys.prefix).resolve() / "pyvenv.cfg"
        or not loaded
        or loaded != expected_native
        or actual["rules_hash"] != expected["rules_hash"]
        or actual["rules_hash"] != RULES_HASH_WIRE
        or actual["feature_schema_hash"] != expected["feature_schema_hash"]
        or actual["feature_schema_hash"] != FEATURE_SCHEMA_HASH
        or actual["search_algorithm"] != expected["search_algorithm"]
        or actual["search_algorithm"] != SEARCH_ALGORITHM_ID
        or actual["torch_version"] != expected["torch_version"]
        or actual["torch_cuda_build"] != expected["torch_cuda_build"]
        or not Path(actual["torch_import"]).is_relative_to(Path(sys.prefix).resolve())
        or any(row["sha256"] != row["expected_sha256"] for row in imports)
    ):
        raise ValueError(
            "actual child runtime/import/native identity differs from pinned inputs"
        )
    if challenge["mode"] == "cuda":
        admission = challenge["admission"]
        if (
            marker != root.parent / "SOURCE_COMMIT"
            or source_manifest != root.parent / "SOURCE_SHA256SUMS"
            or pyvenv != root / ".venv/pyvenv.cfg"
            or not wrapper.is_relative_to(root / ".venv")
            or any(not Path(path).is_relative_to(root / ".venv") for path in loaded)
            or admission["source_commit"] != actual["source_commit"]
            or admission["source_manifest_sha256"] != actual["source_manifest_sha256"]
            or admission["training_module"] != actual["training_module"]
            or admission["native_file"] != actual["native_file"]
            or admission["native_binaries"] != actual["native_binaries"]
            or admission["pyvenv"] != expected["pyvenv"]
        ):
            raise ValueError("observed runtime does not match exact CUDA admission")
        pins = admission["execution_pins"]
        if {p["path"] for p in pins} != {
            str(root / ".venv/bin/python"),
            str(root / ".venv/bin" / (package_name + "-orchestrate")),
        }:
            raise ValueError("CUDA execution entrypoint pins differ")
        for pin in pins:
            path = Path(pin["path"])
            observed = {
                "path": str(path),
                "resolved_path": str(path.resolve()),
                "sha256": digest(path.resolve()),
                "bytes": path.stat().st_size,
            }
            observations.record("runtime-executable", observed)
            if observed != pin:
                raise ValueError("CUDA executable differs from admission")
    elif actual["cuda_initialized"]:
        raise ValueError("CPU validation must not initialize CUDA")
    observations.check_time()


def validate_challenge(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    raw = read_metadata(path)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("probe challenge differs from its explicit hash")
    value = json.loads(raw)
    if (
        not isinstance(value, dict)
        or value.get("format") != CHALLENGE_FORMAT
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
    ):
        raise ValueError("unknown probe challenge format")
    if value.get("mode") not in ("cpu_validation", "cuda"):
        raise ValueError("unknown probe execution mode")
    if value.get("boot_id") != boot_id():
        raise ValueError("probe challenge belongs to another boot")
    for name in ("issued_monotonic", "deadline_monotonic"):
        if type(value.get(name)) not in (float, int) or not math.isfinite(value[name]):
            raise ValueError("probe clock bound is invalid")
    now = time.monotonic()
    if (
        not value["issued_monotonic"]
        <= now
        < value["deadline_monotonic"]
        <= value["issued_monotonic"] + 600
    ):
        raise ValueError("probe challenge expired or exceeds its600second bound")
    if not re.fullmatch(r"[0-9a-f]{64}", value.get("nonce", "")):
        raise ValueError("probe nonce is invalid")
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", value.get("attempt_id", "")
    ):
        raise ValueError("probe attempt identity is invalid")
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", value.get("probe_unit", "")):
        raise ValueError("probe unit identity is invalid")
    invocation = os.environ.get("INVOCATION_ID", "")
    if not re.fullmatch(r"[0-9a-f]{32}", invocation):
        raise ValueError("probe requires its real systemd invocation identity")
    if value.get("probe_source_sha256") != digest(Path(__file__).resolve()):
        raise ValueError("probe source differs from its challenge")
    runtime = Path(value["runtime_root"])
    package = importlib.import_module(ExperimentConfig.__module__.split(".")[0])
    if (
        runtime.resolve() != runtime
        or not isinstance(package.__file__, str)
        or Path(package.__file__).resolve().parent.parent != runtime
    ):
        raise ValueError("probe imported an unqualified runtime root")
    output = Path(value["output_dir"])
    run = Path(value["run_root"])
    if not run.is_absolute() or run.resolve() != run or not run.is_dir():
        raise ValueError("probe run root is not a canonical existing directory")
    release = runtime.parent
    if (
        not output.is_absolute()
        or output.resolve() != output
        or output.exists()
        or output.is_relative_to(run)
        or run.is_relative_to(output)
        or output.is_relative_to(release)
        or release.is_relative_to(output)
    ):
        raise ValueError("probe output overlaps live/runtime inputs or already exists")
    for name in ("profile", "checkpoint", "batch_manifest", "recovery_pointer"):
        pin = value.get(name)
        if (
            pin is None
            and value["mode"] == "cpu_validation"
            and name == "recovery_pointer"
        ):
            continue
        if not isinstance(pin, dict) or Path(pin["path"]).is_relative_to(output):
            raise ValueError("probe inputs overlap output or are missing")
    admission = value.get("admission")
    if value["mode"] == "cuda":
        if not isinstance(admission, dict) or set(admission) != ADMISSION_FIELDS:
            raise ValueError("CUDA probe requires exact admission bindings")
        if (
            admission["profile_sha256"] != value["profile"]["sha256"]
            or admission["recovery_pointer_sha256"]
            != value["recovery_pointer"]["sha256"]
        ):
            raise ValueError("CUDA probe input/admission bindings disagree")
        if str(Path(sys.executable).absolute()) != str(runtime / ".venv/bin/python"):
            raise ValueError(
                "CUDA probe must execute the literal qualified venv interpreter"
            )
        if not re.fullmatch(r"[0-9]+", os.environ.get("CUDA_VISIBLE_DEVICES", "")):
            raise ValueError("CUDA probe needs one explicit visible device")
    elif admission is not None:
        raise ValueError("CPU validation cannot carry a CUDA admission claim")
    elif os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU validation requires CUDA visibility explicitly disabled")
    for name in ("expected_step", "expected_examples_consumed"):
        if type(value.get(name)) is not int or value[name] < 0:
            raise ValueError("probe expected restored boundary is invalid")
    identity = value["run_identity"]
    if not isinstance(identity, dict) or set(identity) != {
        "run_id",
        "generation_family",
        "created_ns",
    }:
        raise ValueError("probe run identity is incomplete")
    if type(identity["created_ns"]) is not int or identity["created_ns"] <= 0:
        raise ValueError("probe run creation identity is invalid")
    if not isinstance(value.get("runtime"), dict):
        raise ValueError("probe requires explicit runtime pins in every mode")
    return value, invocation


def run_probe(path: Path, expected_sha256: str) -> dict[str, Any]:
    challenge, invocation = validate_challenge(path, expected_sha256)
    output = Path(challenge["output_dir"])
    output.mkdir(mode=0o700)
    started = time.monotonic()
    bindings = {
        name: challenge[name]
        for name in ("attempt_id", "boot_id", "nonce", "probe_unit", "mode")
    }
    bindings.update(
        invocation_id=invocation,
        started_monotonic=started,
        challenge_sha256=expected_sha256,
    )
    write_new(
        output / "started.json",
        encoded(
            {
                "format": "strength-freshness-probe-started-v1",
                "schema_version": 1,
                **bindings,
            }
        ),
    )
    observations = Observations(output, challenge["deadline_monotonic"])
    checks = dict.fromkeys(CHECKS, False)
    work: dict[str, Any] = {
        "optimizer_steps": 0,
        "native_searches": 0,
        "native_neural_calls": 0,
    }
    result: dict[str, Any] = {
        "format": RESULT_FORMAT,
        "schema_version": 1,
        **bindings,
        "admission": challenge.get("admission"),
        "status": "failed",
        "checks": checks,
        "work": work,
    }
    initial_math = None
    try:
        runtime_observations(challenge, observations)
        profile = pinned(challenge["profile"], observations, maximum=2 * 1024**2)
        checkpoint = pinned(challenge["checkpoint"], observations, maximum=2 * 1024**3)
        config = load_config(profile)
        if config.orchestration.run_id != challenge["run_identity"]["run_id"]:
            raise ValueError("profile/run identity differs")
        batch, manifest = load_batch(challenge["batch_manifest"], observations)
        if batch.inputs.batch_size != config.train.per_rank_batch_size:
            raise ValueError("probe batch differs from production per-rank batch size")
        if challenge["mode"] == "cpu_validation":
            verify_collation(batch, manifest, observations)
        else:
            from deltreltrain.checkpoint import load_recovery_pointer

            pointer = pinned(
                challenge["recovery_pointer"], observations, maximum=1024**2
            )
            if pointer != Path(challenge["run_root"]) / "learner/recovery.json":
                raise ValueError(
                    "CUDA pointer is not the stopped live recovery authority"
                )
            info = load_recovery_pointer(
                pointer,
                expected_run_id=challenge["run_identity"]["run_id"],
                expected_generation_family=challenge["run_identity"][
                    "generation_family"
                ],
            )
            if (
                info.checkpoint_sha256 != challenge["checkpoint"]["sha256"]
                or info.checkpoint_bytes != challenge["checkpoint"]["bytes"]
                or batch.inputs.batch_size != 512
                or batch.homogeneous_ring != 10
            ):
                raise ValueError(
                    "CUDA probe does not bind the production stopped checkpoint/batch"
                )
            available = torch.cuda.is_available() and torch.cuda.device_count() == 1
            observations.record("cuda-availability", {"available": available})
            if not available:
                raise ValueError("actual exclusive CUDA device unavailable")
            checks["cuda_available"] = True
            initial_math = backend_math_flags()
            observations.record("initial-backend-math", initial_math)
            cache = configure_isolated_compile_cache(output / "compiler")
            observations.record("isolated-compile-cache", cache.as_dict())
        device = "cuda:0" if challenge["mode"] == "cuda" else "cpu"
        state, payload = restore(
            config,
            checkpoint,
            challenge["checkpoint"],
            challenge["run_identity"],
            device,
            observations,
        )
        observations.record(
            "resume-boundary",
            {
                "step": state.step,
                "epoch": state.epoch,
                "examples_consumed": state.examples_consumed,
                "expected_step": challenge["expected_step"],
                "expected_examples_consumed": challenge["expected_examples_consumed"],
            },
        )
        if (
            state.step != challenge["expected_step"]
            or state.examples_consumed != challenge["expected_examples_consumed"]
        ):
            raise ValueError("restored checkpoint boundary differs from challenge")
        checks["learner_resume_step"] = True
        if challenge["mode"] == "cuda":
            observations.record("learner-backend-math", backend_math_flags())
            training_observations(state, config, batch, observations)
            work["optimizer_steps"] = 1
            checks["ema_preserved"] = True
            del state, payload
            assert initial_math is not None
            restore_math_flags(initial_math)
            observations.record("actor-backend-math", backend_math_flags())
            native = native_observations(
                config,
                checkpoint,
                challenge["checkpoint"],
                challenge["run_identity"],
                device,
                observations,
            )
            work.update(
                native_searches=native["searches"],
                native_neural_calls=native["metrics"]["neural_calls"],
            )
            checks["native_cuda_search"] = True
        else:
            checks["ema_preserved"] = True
        for name in ("profile", "checkpoint", "batch_manifest"):
            pinned(challenge[name], observations, maximum=2 * 1024**3)
        pinned(manifest["artifact"], observations, maximum=128 * 1024**2)
        if challenge["mode"] == "cuda":
            pinned(challenge["recovery_pointer"], observations, maximum=1024**2)
        runtime_observations(challenge, observations)
        observations.check_time()
        result["status"] = (
            "cuda_observed" if challenge["mode"] == "cuda" else "cpu_validated"
        )
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)[:2000]}
        try:
            observations.record("failure", result["error"])
        except (OSError, ValueError):
            result["failure_record_unavailable"] = True
    finally:
        if initial_math is not None:
            restore_math_flags(initial_math)
        observations.close()
    result["completed_monotonic"] = time.monotonic()
    result["evidence"] = [
        {
            "path": "observations.jsonl",
            "sha256": digest(observations.path),
            "bytes": observations.path.stat().st_size,
        }
    ]
    write_new(output / "result.json", encoded(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--challenge-sha256", required=True)
    args = parser.parse_args()
    try:
        result = run_probe(args.challenge.absolute(), args.challenge_sha256)
        print(json.dumps({"status": result["status"], "output": str(args.challenge)}))
        return 0 if result["status"] in ("cuda_observed", "cpu_validated") else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "refused", "error": str(error)[:2000]}))
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
