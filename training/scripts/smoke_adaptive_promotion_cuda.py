#!/usr/bin/env python3
"""Bounded stopped-run smoke for 18-game adaptive promotion cohorts.

Loads only an in-memory EMA model. Native searches retain the profile's arena
budget; two search waves exercise checkpoint/resume without completing games.
The parent owns and bounds the child process group, then publishes one report.
No replay, model, profile, pointer or controller state is written.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from deltreltrain.inference import GraphInferenceAdapter, InferenceConfig


TIMEOUT_SECONDS = 180
SHADOW_TIMEOUT_SECONDS = 60
PAIRS_PER_MODE = 9


def require_stopped_run(config, checkpoint: Path) -> dict:
    from scripts.migrate_continuous_profile import (
        _coordinator_lock_status,
        _pid_is_live,
        _read_json,
        _validate_recovery,
        _validate_run_identity,
    )
    from deltreltrain.checkpoint import sha256_file

    root = Path(config.orchestration.directories.root).expanduser().resolve()
    identity, _ = _read_json(root / "run.json", "run identity")
    run_id, family, _ = _validate_run_identity(identity)
    if run_id != config.orchestration.run_id:
        raise ValueError("smoke profile belongs to another run")
    lock_status, _, _ = _coordinator_lock_status(root)
    status_root = (root / config.orchestration.directories.status).resolve()
    coordinator_path = status_root / "coordinator.json"
    coordinator, _ = _read_json(coordinator_path, "stopped coordinator")
    pid = coordinator.get("coordinator_pid")
    workers = coordinator.get("workers")
    if (
        coordinator.get("state") != "stopped"
        or coordinator.get("failure") is not None
        or type(pid) is not int
        or pid <= 0
        or _pid_is_live(pid)
        or not isinstance(workers, dict)
        or not workers
    ):
        raise ValueError("smoke requires a cleanly stopped coordinator inventory")
    heartbeats = {}
    # Old topologies leave heartbeat files behind. Only the stopped
    # coordinator's authoritative worker inventory belongs to this cutover.
    for name, worker in sorted(workers.items()):
        if (
            not isinstance(name, str)
            or not isinstance(worker, dict)
            or worker.get("pid") is not None
            or worker.get("state") not in ("stopped", "drained", "completed", "paused")
            or worker.get("failure_reason") is not None
            or not isinstance(worker.get("heartbeat"), str)
        ):
            raise ValueError("smoke requires stopped coordinator workers")
        path = Path(worker["heartbeat"])
        if (
            path.is_symlink()
            or path.resolve() != status_root / f"{name}.heartbeat.json"
        ):
            raise ValueError("coordinator worker heartbeat path is incompatible")
        payload, _ = _read_json(path, "worker heartbeat")
        pid = payload.get("pid")
        if (
            payload.get("worker") != name
            or type(pid) is not int
            or pid <= 0
            or _pid_is_live(pid)
        ):
            raise ValueError(f"smoke requires a stopped worker: {path.name}")
        heartbeats[name] = pid
    pointer_path = root / "learner" / "recovery.json"
    pointer, _ = _read_json(pointer_path, "stopped recovery pointer")
    step, _, digest, size, selected = _validate_recovery(
        pointer, path=pointer_path, run_id=run_id, generation_family=family
    )
    if checkpoint.resolve() != selected:
        raise ValueError("smoke checkpoint must match the stopped recovery pointer")
    return {
        "run_id": run_id,
        "generation_family": family,
        "coordinator_lock": lock_status,
        "coordinator_status_sha256": sha256_file(coordinator_path),
        "stopped_worker_pids": heartbeats,
        "checkpoint_sha256": digest,
        "checkpoint_bytes": size,
        "checkpoint_step": step,
        "recovery_pointer_sha256": sha256_file(pointer_path),
    }


class ObservedAdapter(GraphInferenceAdapter):
    """Record the actual production broker's logical and physical batch sizes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logical_rows: set[int] = set()
        self.physical_rows: set[int] = set()

    def evaluate_prepared(self, requests, *, include_details=None):
        before = self.metrics_snapshot()
        actual = super().evaluate_prepared(requests, include_details=include_details)
        delta = self.metrics_snapshot().delta(before)
        rows = sum(len(request.tokens) for request in requests)
        if rows:
            self.logical_rows.add(rows)
        if delta.neural_calls:
            if delta.neural_calls != 1:
                raise RuntimeError("smoke observed an unexpected physical batch split")
            self.physical_rows.add(delta.neural_rows)
        return actual


class ComparedGraphAdapter(ObservedAdapter):
    """Check only the separate shadow graph probe against same-shape inference."""

    reference: GraphInferenceAdapter

    def install_reference(self, reference: GraphInferenceAdapter) -> None:
        self.reference = reference
        self.comparisons = 0

    def evaluate_prepared(self, requests, *, include_details=None):
        from scripts.validate_cuda_graph_runtime import compare_predictions

        expected = self.reference.evaluate_prepared(
            requests, include_details=include_details
        )
        actual = super().evaluate_prepared(requests, include_details=include_details)
        if len(expected) != len(actual):
            raise RuntimeError("graph changed merged request routing")
        for (before, _), (after, _) in zip(expected, actual, strict=True):
            compare_predictions(before, after)
            self.comparisons += 1
        return actual


def smoke_inference_configs(config) -> tuple[InferenceConfig, InferenceConfig]:
    """Retain production execution; enable graphs only in the shadow probe."""
    runtime = config.orchestration.model_refresh.inference
    production = InferenceConfig(
        precision=config.train.precision,
        score_utility_weight=config.selfplay.score_utility_weight,
        cache_max_entries=0,
        cache_max_bytes=0,
        deduplicate=runtime.deduplicate,
        pinned_transfers=runtime.pinned_transfers,
        pinned_buffer_slots=runtime.pinned_buffer_slots,
        preserve_broadcast_topology=runtime.preserve_broadcast_topology,
        cuda_graphs=runtime.cuda_graphs,
        cuda_graph_max_entries=runtime.cuda_graph_max_entries,
        cuda_graph_max_bytes=runtime.cuda_graph_max_bytes,
        compact_inference_gather=runtime.compact_inference_gather,
        small_batch_graph_buckets=runtime.small_batch_graph_buckets,
    )
    return production, replace(production, cuda_graphs=True)


def exercise_prefix_resume(native, adapter, config, label: str) -> dict:
    from deltreltrain.arena import ArenaRunner
    from deltreltrain.selfplay import GameVariant

    variant = GameVariant.parse(label)
    initial = max(config.minimum_pairs_per_ring, len(config.handicap_severity_cycle))
    saved = None
    waves = []
    for _ in range(2):
        snapshots = []
        subject = ArenaRunner(
            native_module=native, candidate=adapter, baseline=adapter, config=config
        )
        subject._initialize_resume(saved, snapshots.append)
        specifications = subject._pair_specifications(
            10, list(range(initial, initial + PAIRS_PER_MODE)), variant
        )
        with subject._inference_owner() as executor:
            games = subject._play_ring_batch(
                10,
                specifications,
                variant=variant,
                progress=None,
                inference_executor=executor,
                stop_requested=lambda: bool(snapshots),
            )
        if games or not snapshots:
            raise RuntimeError(
                "bounded smoke must checkpoint before any game completes"
            )
        current = snapshots[-1]
        states = current["game_states"]
        if len(states) != 2 * PAIRS_PER_MODE:
            raise RuntimeError("smoke did not retain all 18 game slots")
        previous = {
            (row["pair"], row["candidate_player"]): row["actions"]
            for row in (saved or {}).get("game_states", [])
        }
        for row in states:
            prefix = previous.get((row["pair"], row["candidate_player"]), [])
            if row["actions"][:-1] != prefix or row["result"] is not None:
                raise RuntimeError(
                    "resumed smoke lost history or advanced beyond one move"
                )
        shared = subject._shared_inference_metrics
        if (
            not shared
            or shared["pending_requests"]
            or shared["failed_requests"]
            or shared["submitted_requests"] != shared["completed_requests"]
        ):
            raise RuntimeError("shared inference broker did not drain cleanly")
        waves.append(
            {
                "game_slots": len(states),
                "searched_moves": sum(len(row["actions"]) for row in states),
                "shared_inference": shared,
            }
        )
        saved = json.loads(json.dumps(current))
    return {
        "variant": label,
        "pairs": PAIRS_PER_MODE,
        "simulations_per_root": config.simulations,
        "waves": waves,
        "resumed_histories_verified": True,
    }


def record_input_layouts(adapter) -> dict[str, object]:
    """Observe already transferred inputs; never materialize a lazy request."""
    layouts = {}
    original = adapter._to_device

    def observed(host):
        encoded = original(host)
        signature = [
            {
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
            }
            for tensor in encoded.model_args()
        ]
        layouts[json.dumps(signature, sort_keys=True)] = signature
        return encoded

    adapter._to_device = observed
    return layouts


def current_math_policy() -> dict:
    import torch

    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    }


def root_probe(native, config, mode):
    states = native.StateBatch(10, 18, mode=mode, pie=True)
    states.apply_many(list(range(18)), list(range(18)))
    search = native.SearchBatch(
        states,
        simulations=config.simulations,
        max_considered=config.max_considered,
        deterministic_seed=17,
    )
    return search.root_requests()


def shadow_checks(native, graph, regular, config) -> dict:
    from scripts.validate_cuda_graph_runtime import check_health, entry_records

    attempts = []

    def attempt(request, mode, phase):
        before = graph.efficiency_snapshot()
        record = {
            "mode": mode,
            "phase": phase,
            "comparison_index": len(attempts),
            "captures_before": before["graph_captures"],
            "replays_before": before["graph_replays"],
        }
        print(
            json.dumps({"event": "shadow-comparison-start", **record}),
            file=sys.stderr,
            flush=True,
        )
        try:
            graph.evaluate(request)
            record["status"] = "passed"
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
        record["graph_metrics"] = graph.efficiency_snapshot()
        attempts.append(record)
        return record

    for mode in ("classic", "double"):
        request = root_probe(native, config, mode)
        for _ in range(2):
            reused = bool(graph._graphs is not None and graph._graphs._entries)
            result = attempt(request, mode, "reused-entry" if reused else "cold-entry")
            if result["status"] == "failed" and reused:
                assert graph._graphs is not None
                graph._graphs.clear()
                attempt(request, mode, "cold-retry-after-reuse-failure")
    health_error = None
    try:
        health = check_health(graph)
        if health.get("graph_replays", 0) <= 0 or 18 not in graph.logical_rows:
            raise RuntimeError("shadow probe did not replay the 18-row graph geometry")
    except Exception as error:
        health_error = f"{type(error).__name__}: {error}"
    return {
        "status": "passed"
        if health_error is None and all(item["status"] == "passed" for item in attempts)
        else "failed",
        "attempts": attempts,
        "health_error": health_error,
        "metrics": graph.efficiency_snapshot(),
        "capture_entries": entry_records(graph),
        "same_shape_prediction_comparisons": graph.comparisons,
        "comparison_tolerance": "bitwise equality; original mismatches remain failures even if cold retries pass",
        "initialization_order": "reference-first, matching the prior smoke; graph backend retains its own warmup policy",
        "ordinary_reference_extra_warmup_calls_per_mode": 0,
    }


def run_smoke(args) -> dict:
    import torch
    from scripts.benchmark_actor_throughput import _gpu_ownership
    from scripts.validate_cuda_graph_runtime import (
        check_health,
        native_artifacts,
        regular_adapter,
        validate_output,
    )
    from deltreltrain.checkpoint import load_ema_checkpoint, sha256_file
    from deltreltrain.config import load_config
    from deltreltrain.model import GraphResTNet
    from deltreltrain.native import load_deltrel_native
    from deltreltrain.training import maybe_compile_model

    started = time.time_ns()
    config = load_config(args.profile)
    if (
        config.arena.allocation_policy != "adaptive_pie"
        or config.arena.simulations != 256
    ):
        raise ValueError("smoke requires the adaptive 256-simulation promotion profile")
    validate_output(args.output, config, args.profile, args.checkpoint)
    pinned = require_stopped_run(config, args.checkpoint)
    profile_hash = sha256_file(args.profile)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.shadow_only:
        # Match the earlier graph diagnostic in this isolated process only.
        # Actual production arithmetic policy is left untouched.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    context = torch.empty(1, device=device)
    gpu_uuid = str(torch.cuda.get_device_properties(device).uuid)
    if not gpu_uuid.startswith("GPU-"):
        gpu_uuid = "GPU-" + gpu_uuid
    ownership_before = _gpu_ownership(gpu_uuid)
    if ownership_before.get("verified") is not True:
        raise RuntimeError(f"smoke requires an exclusive GPU: {ownership_before}")
    torch.cuda.reset_peak_memory_stats(device)
    model = GraphResTNet(config.model).to(device)
    metadata = load_ema_checkpoint(
        args.checkpoint,
        model=model,
        expected_model_config=asdict(config.model),
        expected_game_config=asdict(config.game),
        expected_run_id=pinned["run_id"],
        expected_generation_family=pinned["generation_family"],
        expected_sha256=pinned["checkpoint_sha256"],
        expected_bytes=pinned["checkpoint_bytes"],
    )
    model.eval()
    refresh = config.orchestration.model_refresh
    compiled = maybe_compile_model(
        model,
        enabled=config.train.compile,
        dynamic=refresh.inference_compile_dynamic,
        fullgraph=True,
        mode=refresh.inference_compile_mode,
    )
    runtime = refresh.inference
    if config.train.precision != "bf16" or runtime.cuda_graph_max_entries != 32:
        raise ValueError(
            "smoke requires production BF16 and the existing 32-entry graph capacity"
        )
    production_config, shadow_config = smoke_inference_configs(config)
    adapters = []
    native = load_deltrel_native(required=True)
    if native is None:
        raise RuntimeError("native extension unavailable")
    try:
        common = dict(
            device=device,
            homogeneous_relational_bias=runtime.homogeneous_relational_bias,
            model_version="sha256-" + pinned["checkpoint_sha256"],
            model_step=metadata["step"],
        )
        if args.shadow_only:
            graph = ComparedGraphAdapter(compiled, config=shadow_config, **common)
            regular = regular_adapter(graph)
            adapters.extend((regular, graph))
            graph.install_reference(regular)
            graph_layouts = record_input_layouts(graph)
            regular_layouts = record_input_layouts(regular)
            detail = shadow_checks(native, graph, regular, config.arena)
            detail.update(
                configuration=asdict(shadow_config),
                reference_configuration=asdict(regular.config),
                logical_row_counts=sorted(graph.logical_rows),
                physical_row_buckets=sorted(graph.physical_rows),
                graph_input_layouts=list(graph_layouts.values()),
                reference_input_layouts=list(regular_layouts.values()),
                native_search_and_resume=False,
            )
            specific = {"shadow_graph_check": detail}
            status = detail["status"]
        else:
            production = ObservedAdapter(compiled, config=production_config, **common)
            adapters.append(production)
            layouts = record_input_layouts(production)
            variants = []
            for mode in ("classic", "double"):
                production.evaluate(root_probe(native, config.arena, mode))
                variants.append(
                    exercise_prefix_resume(
                        native, production, config.arena, f"pie-{mode}"
                    )
                )
            check_health(production)
            if (
                18 not in production.logical_rows
                or production._inference_batch_rows(18) not in production.physical_rows
            ):
                raise RuntimeError(
                    "smoke did not execute the actual production 18-row geometry"
                )
            specific = {
                "variants": variants,
                "production_inference": {
                    "configuration": asdict(production_config),
                    "logical_row_counts": sorted(production.logical_rows),
                    "physical_row_buckets": sorted(production.physical_rows),
                    "input_layouts": list(layouts.values()),
                    "metrics": production.efficiency_snapshot(),
                    "native_search_and_resume": True,
                },
            }
            status = "passed"
        torch.cuda.synchronize(device)
        ownership_after = _gpu_ownership(gpu_uuid)
        if ownership_after.get("verified") is not True:
            raise RuntimeError("GPU became shared during the stopped-run smoke")
        if (
            require_stopped_run(config, args.checkpoint) != pinned
            or sha256_file(args.profile) != profile_hash
        ):
            raise RuntimeError("stopped-run artifacts changed during the smoke")
        return {
            "status": status,
            "format": "deltreltrain.adaptive-promotion-cuda-smoke",
            "schema_version": 2,
            "stage": "shadow" if args.shadow_only else "production",
            "profile_sha256": profile_hash,
            **pinned,
            **specific,
            "math_policy": current_math_policy(),
            "gpu_ownership_before": ownership_before,
            "gpu_ownership_after": ownership_after,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "native": native_artifacts(native),
            "smoke_source_sha256": sha256_file(Path(__file__)),
            "model_copies": 1,
            "started_ns": started,
            "completed_ns": time.time_ns(),
            "training_artifacts_written": False,
            "timing_claim": "correctness and memory smoke only; no throughput or Elo claim",
        }
    finally:
        for adapter in adapters:
            adapter.close()
        del context


def publish_report(path: Path, report: dict) -> None:
    descriptor, name = tempfile.mkstemp(prefix=".adaptive-smoke-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def run_owned_stage(command: list[str], timeout: int) -> dict:
    from scripts.benchmark_actor_throughput import _run_owned

    try:
        child = _run_owned(command, env=dict(os.environ), timeout=timeout)
        if child.returncode:
            raise RuntimeError(child.stderr[-6000:])
        report = json.loads(child.stdout.strip().splitlines()[-1])
        if not isinstance(report, dict) or report.get("status") not in (
            "passed",
            "failed",
        ):
            raise ValueError("smoke worker returned an invalid report")
    except (subprocess.TimeoutExpired, RuntimeError, ValueError, IndexError) as error:
        report = {
            "status": "failed",
            "error": str(error),
            "training_artifacts_written": False,
        }
        stderr = getattr(error, "stderr", None)
        if stderr:
            report["worker_stderr_tail"] = (
                stderr.decode(errors="replace")
                if isinstance(stderr, bytes)
                else str(stderr)
            )[-6000:]
    report["child_timeout_seconds"] = timeout
    return report


def revalidate_boundary(args, production: dict) -> None:
    from deltreltrain.config import load_config
    from deltreltrain.checkpoint import sha256_file
    from scripts.benchmark_actor_throughput import _gpu_ownership

    current = require_stopped_run(load_config(args.profile), args.checkpoint)
    if any(
        production.get(key) != value for key, value in current.items()
    ) or sha256_file(args.profile) != production.get("profile_sha256"):
        raise RuntimeError(
            "parent found changed stopped-run/profile/checkpoint evidence"
        )
    ownership = _gpu_ownership(production["gpu_ownership_after"]["gpu_uuid"])
    if ownership.get("owner_pids") != []:
        raise RuntimeError(
            "parent could not verify the GPU is idle after smoke workers"
        )


def run_stages(args) -> dict:
    command = [
        sys.executable,
        "-m",
        "scripts.smoke_adaptive_promotion_cuda",
        "--profile",
        str(args.profile),
        "--checkpoint",
        str(args.checkpoint),
        "--output",
        str(args.output),
        "--worker",
    ]
    production = run_owned_stage(command, TIMEOUT_SECONDS)
    if production.get("status") != "passed":
        return {
            **production,
            "qualification_scope": "required production check failed",
            "shadow_graph_check": {"status": "not_run"},
        }
    shadow = run_owned_stage([*command, "--shadow-only"], SHADOW_TIMEOUT_SECONDS)
    required = production["production_inference"]["configuration"]["cuda_graphs"]
    report = {
        **production,
        "status": "passed"
        if not required or shadow.get("status") == "passed"
        else "failed",
        "qualification_scope": "actual production runtime; shadow graph diagnostic is separate",
        "production_check_status": "passed",
        "shadow_graph_check": shadow,
        "shadow_required_for_production": required,
        "shadow_graph_parity_qualified": shadow.get("status") == "passed",
    }
    try:
        revalidate_boundary(args, production)
    except Exception as error:
        report.update(
            status="failed",
            boundary_revalidation_error=f"{type(error).__name__}: {error}",
        )
    else:
        report["parent_boundary_revalidated"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shadow-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("smoke output must be a new file")
    if args.worker:
        print(json.dumps(run_smoke(args)))
        return
    from deltreltrain.config import load_config
    from scripts.validate_cuda_graph_runtime import validate_output

    validate_output(
        args.output, load_config(args.profile), args.profile, args.checkpoint
    )
    from scripts.benchmark_actor_throughput import _controller_signals

    with _controller_signals():
        report = run_stages(args)
    publish_report(args.output, report)
    print(json.dumps(report))
    if report.get("status") != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
