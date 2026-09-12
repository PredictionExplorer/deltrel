#!/usr/bin/env python3
"""Compare 16/32 graph entries on bounded, mixed-board frozen-model traces.

The default only prints a pinned plan. Execution requires an already exclusive
GPU and a new output path; this tool never controls other workloads. Each arm
starts with an empty graph cache and times every capture, recapture, validation,
transfer and normal inference response. Compilation and the reference outputs are
primed separately through the production 16-entry graph path; those graphs are
discarded before timing. Prediction caches alone are cleared between calls so
repeated fixtures cannot hide graph-cache thrashing behind prediction hits.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, cast

from scripts.benchmark_graph_buckets import _gpu_snapshot, _load_assessment, _nvml_uuid
from scripts.benchmark_training_shared_geometry import _run_process
from scripts.benchmark_local_message_adapter import actor_experiment
from startrain.graph_cache_evidence import (
    BOUNDED_NONINFERIORITY_POLICY,
    BUCKETS,
    SCENARIOS,
    PERFORMANCE_POLICIES,
    STRICT_PERFORMANCE_POLICY,
    assess,
    actor_inference_math_expected,
    build_trace,
    performance_policy_contract,
    validate_actor_inference_math,
    validate_graph_cache_report,
)


def read_actor_inference_math(
    torch_module: Any = None, environment: Any = None
) -> dict[str, Any]:
    """Observe actual process settings without changing any precision flags."""
    if torch_module is None:
        import torch as torch_module
    if environment is None:
        environment = os.environ
    return {
        "float32_matmul_precision": torch_module.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch_module.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch_module.backends.cudnn.allow_tf32,
        "environment": {
            name: environment.get(name)
            for name in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")
        },
    }


def response_digest(response: Any, rows: int) -> str:
    """Bitwise FP32 search-response oracle; routing and finite values included."""
    import numpy as np

    if (
        len(response.tokens) != rows
        or len(response.values) != rows
        or len(response.policy_offsets) != rows + 1
        or response.policy_offsets[0] != 0
        or response.policy_offsets[-1] != len(response.policy_logits)
        or any(
            a > b for a, b in zip(response.policy_offsets, response.policy_offsets[1:])
        )
    ):
        raise ValueError("inference response routing or valid row count changed")
    digest = hashlib.sha256()
    for name, values, dtype in (
        ("tokens", response.tokens, "<u8"),
        ("offsets", response.policy_offsets, "<u8"),
        ("values", response.values, "<f4"),
        ("policy", response.policy_logits, "<f4"),
    ):
        array = np.asarray(values, dtype=dtype)
        if not np.isfinite(array).all():
            raise ValueError("nonfinite inference response")
        digest.update(name.encode())
        digest.update(array.size.to_bytes(8, "little"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def prime_reference(
    make_adapter: Any,
    requests: dict[tuple[int, int], Any],
    *,
    sync: Any,
    math_state: Any = read_actor_inference_math,
    clock: Any = time.perf_counter,
) -> tuple[dict[tuple[int, int], str], dict[str, Any]]:
    """Warm capture-specific compiler paths with the actual production oracle.

    This adapter is never reused for measurements. Every timed arm still starts
    empty and pays its own capture, validation and recapture costs.
    """
    prime = make_adapter(16)
    reference = {}
    before_math = math_state()
    validate_actor_inference_math(before_math, phase="priming start")
    tick = clock()
    try:
        if not prime.config.cuda_graphs or prime.config.cuda_graph_max_entries != 16:
            raise ValueError("reference requires the production 16-entry graph path")
        before = prime.metrics_snapshot()
        for key in sorted(requests):
            prime.clear_prediction_cache()
            prepared = prime.prepare_requests(requests[key])
            response = prime.evaluate_prepared([prepared])[0][0]
            reference[key] = response_digest(response, key[1])
        sync()
        seconds = clock() - tick
        metrics = asdict(prime.metrics_snapshot().delta(before))
        if any(
            metrics[name] != len(requests)
            for name in (
                "evaluator_calls",
                "neural_calls",
                "graph_captures",
                "graph_replays",
            )
        ) or any(
            metrics[name] != 0
            for name in (
                "cache_hits",
                "deduplicated_rows",
                "neural_padding_rows",
                "graph_fallbacks",
                "graph_validation_failures",
            )
        ):
            raise ValueError(
                "production graph reference priming did not complete valid captures"
            )
    finally:
        prime.close()
        sync()
    after_math = math_state()
    validate_actor_inference_math(after_math, phase="priming end")
    closed = prime.graph_residency_snapshot()
    return reference, {
        "execution": "production-graph-16",
        "entries": 16,
        "cuda_graphs": True,
        "requested_shapes": [list(key) for key in sorted(requests)],
        "seconds": seconds,
        "metrics": metrics,
        "cache_empty_after_close": isinstance(closed, dict)
        and closed.get("entries") == [],
        "math_before": before_math,
        "math_after": after_math,
    }


def measure_arm(
    adapter: Any,
    trace: list[tuple[int, int]],
    prepared: dict[tuple[int, int], Any],
    reference: dict[tuple[int, int], str],
    *,
    cycles: int,
    sync: Any,
    snapshot: Any,
    memory: Any,
    math_state: Any = read_actor_inference_math,
    clock: Any = time.perf_counter,
) -> dict[str, Any]:
    """Keep graphs across cycles, including the first empty-cache cycle in time."""
    math_before = math_state()
    validate_actor_inference_math(math_before, phase="arm start")
    snapshots = [snapshot()]
    if not _load_assessment(snapshots, declared="isolated", worker_pid=os.getpid())[
        "comparison_adoptable"
    ]:
        raise RuntimeError("exclusive GPU ownership was not verified")
    before = adapter.metrics_snapshot()
    records = []
    parity = True
    for cycle in range(cycles):
        sync()
        start_metrics = adapter.metrics_snapshot()
        elapsed = 0.0
        digests = []
        for key in trace:
            # Deliberately do not clear graphs, topology, or model caches here.
            adapter.clear_prediction_cache()
            tick = clock()
            response = adapter.evaluate_prepared([prepared[key]])[0][0]
            sync()
            elapsed += clock() - tick
            digest = response_digest(response, key[1])
            digests.append(digest)
            parity = parity and digest == reference[key]
        delta = asdict(adapter.metrics_snapshot().delta(start_metrics))
        records.append(
            {
                "cycle": cycle,
                "seconds": elapsed,
                "metrics": delta,
                "output_digests": digests,
                **memory(),
            }
        )
        snapshots.append(snapshot())
    measured = asdict(adapter.metrics_snapshot().delta(before))
    expected_rows = cycles * sum(rows for _, rows in trace)
    expected_calls = cycles * len(trace)
    valid = (
        parity
        and measured["evaluator_rows"] == expected_rows
        and measured["neural_rows"] == expected_rows
        and measured["neural_calls"] == expected_calls
        and measured["graph_replays"] == expected_calls
        and measured["cache_hits"] == 0
        and measured["deduplicated_rows"] == 0
        and measured["neural_padding_rows"] == 0
        and measured["graph_fallbacks"] == 0
        and measured["graph_validation_failures"] == 0
    )
    seconds = sum(record["seconds"] for record in records)
    load = _load_assessment(snapshots, declared="isolated", worker_pid=os.getpid())
    math_after = math_state()
    validate_actor_inference_math(math_after, phase="arm end")
    return {
        "math_before": math_before,
        "math_after": math_after,
        "cycles": records,
        "cold_cycle": records[0],
        "steady_totals": {
            "seconds": sum(record["seconds"] for record in records[1:]),
            "useful_rows": (cycles - 1) * sum(rows for _, rows in trace),
            "graph_captures": sum(
                record["metrics"]["graph_captures"] for record in records[1:]
            ),
            "graph_evictions": sum(
                record["metrics"]["graph_evictions"] for record in records[1:]
            ),
            "graph_warmup_calls": sum(
                record["metrics"]["graph_warmup_calls"] for record in records[1:]
            ),
        },
        "seconds": seconds,
        "useful_rows": expected_rows,
        "useful_rows_per_second": expected_rows / seconds,
        "metrics": measured,
        "output_parity_exact": parity,
        "execution_valid": valid,
        "load_assessment": load,
        "gpu_observations": snapshots,
        "graph_residency": adapter.graph_residency_snapshot(),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument(
        "--checkpoint",
        "--manifest",
        dest="checkpoint",
        type=Path,
        required=True,
        help="immutable verified EMA manifest",
    )
    result.add_argument("--device", default="cuda:0")
    result.add_argument(
        "--actor-gpu-id",
        type=int,
        default=1,
        help="production actor configuration to reproduce",
    )
    result.add_argument(
        "--scenarios", nargs="+", choices=tuple(SCENARIOS), default=list(SCENARIOS)
    )
    result.add_argument("--batch-sizes", type=int, nargs="+", default=list(BUCKETS))
    result.add_argument("--repeats", type=int, default=2)
    result.add_argument("--cycles", type=int, default=2)
    result.add_argument("--graph-cache-bytes", type=int, default=8 * 1024**3)
    result.add_argument("--memory-fraction", type=float, default=0.50)
    result.add_argument("--minimum-speedup", type=float, default=1.0)
    result.add_argument(
        "--performance-policy",
        choices=PERFORMANCE_POLICIES,
        default=STRICT_PERFORMANCE_POLICY,
    )
    result.add_argument("--timeout-seconds", type=float, default=1800)
    result.add_argument("--output", type=Path)
    result.add_argument("--execute", action="store_true")
    result.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--pinned-plan", help=argparse.SUPPRESS)
    return result


def validate(args: argparse.Namespace) -> None:
    if (
        not args.device.startswith("cuda:")
        or not args.device[5:].isdigit()
        or len(set(args.scenarios)) != len(args.scenarios)
        or not 2 <= args.repeats <= 6
        or args.repeats % 2
        or not 2 <= args.cycles <= 8
        or not 1024**2 <= args.graph_cache_bytes <= 8 * 1024**3
        or not math.isfinite(args.memory_fraction)
        or not 0.1 <= args.memory_fraction <= 0.75
        or not math.isfinite(args.minimum_speedup)
        or not 1 <= args.minimum_speedup <= 2
        or not math.isfinite(args.timeout_seconds)
        or not 1 <= args.timeout_seconds <= 3600
    ):
        raise ValueError(
            "invalid bounded GPU, trace, repetition, memory or deadline settings"
        )
    for scenario in args.scenarios:
        build_trace(scenario, tuple(args.batch_sizes))
    if args.performance_policy == BOUNDED_NONINFERIORITY_POLICY and (
        args.repeats != 4
        or args.cycles != 2
        or args.scenarios != list(SCENARIOS)
        or tuple(args.batch_sizes) != BUCKETS
        or args.minimum_speedup != 1.0
    ):
        raise ValueError(
            "bounded noninferiority requires four repeats, two cycles, every default scenario/bucket, and its fixed prospective thresholds"
        )
    if args.worker and (not args.execute or not args.pinned_plan):
        raise ValueError("worker requires an executed pinned plan")


def plan(args: argparse.Namespace) -> tuple[dict[str, Any], Any, Any, Any]:
    from startrain.checkpoint import load_model_manifest
    from startrain.config import load_config
    from startrain import (
        inference,
        inference_graphs,
        model,
        native,
        graph_cache_evidence,
    )

    raw_config = load_config(args.config)
    config, actor = actor_experiment(raw_config, args.actor_gpu_id)
    manifest = load_model_manifest(args.checkpoint)
    refresh = config.orchestration.model_refresh
    inference_config = refresh.inference
    if (
        not inference_config.cuda_graphs
        or not inference_config.small_batch_graph_buckets
    ):
        raise ValueError(
            "frozen profile must enable graphs and production small buckets"
        )
    traces = {
        name: build_trace(name, tuple(args.batch_sizes)) for name in args.scenarios
    }
    if not {ring for trace in traces.values() for ring, _ in trace} <= set(
        config.game.rings
    ):
        raise ValueError("trace boards are absent from the frozen model configuration")
    result = {
        "schema_version": 1,
        "benchmark": "graph-cache-capacity",
        "model_identity": manifest.model_identity,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "source_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "source_config_canonical_sha256": hashlib.sha256(
            json.dumps(
                raw_config.as_dict(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "effective_config_canonical_sha256": hashlib.sha256(
            json.dumps(config.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            module.__name__: hashlib.sha256(
                Path(cast(str, module.__file__)).read_bytes()
            ).hexdigest()
            for module in (
                inference,
                inference_graphs,
                model,
                native,
                graph_cache_evidence,
            )
        },
        "device": args.device,
        "actor_gpu_id": args.actor_gpu_id,
        "actor_configuration": asdict(actor),
        "inference_configuration": asdict(inference_config),
        "precision": config.train.precision,
        "profile_inference": asdict(inference_config),
        "control_entries": 16,
        "treatment_entries": 32,
        "compile": config.train.compile,
        "compile_dynamic": refresh.inference_compile_dynamic,
        "compile_mode": refresh.inference_compile_mode,
        "traces": traces,
        "repeats": args.repeats,
        "cycles": args.cycles,
        "entries": [16, 32],
        "actor_inference_math": actor_inference_math_expected(),
        "performance_policy": args.performance_policy,
        "performance_policy_contract": performance_policy_contract(
            args.performance_policy
        ),
        "reference_execution": "production-graph-16",
        "graph_cache_bytes_per_arm": args.graph_cache_bytes,
        "production_graph_cache_bytes_per_model": inference_config.cuda_graph_max_bytes
        // (actor.actor_cohorts + 2 if actor.actor_cohorts > 1 else 1),
        "memory_fraction": args.memory_fraction,
        "minimum_speedup": args.minimum_speedup,
        "scope": "serial empty-cache arms; all graph construction charged; compiled model priming and producer request preparation reported separately; native search budget/precision unchanged",
    }
    return result, config, manifest, actor


def native_request(native: Any, ring: int, rows: int) -> Any:
    # All six modes and three depths occur across shapes, while shape identity is
    # independent of mode. Distinct row permutations prevent prediction dedup.
    index = BUCKETS.index(rows)
    mode, handicap, pie = (
        ("classic", 1, False),
        ("double", 1, False),
        ("classic", 1, True),
        ("double", 1, True),
        ("classic", 3, False),
        ("double", 3, False),
    )[index % 6]
    states = native.StateBatch(ring, rows, mode=mode, handicap=handicap, pie=pie)
    nodes = states.node_count
    depth = (3, nodes // 2, nodes * 3 // 4)[index % 3]
    permutations = []
    for row in range(rows):
        order = list(range(nodes))
        random.Random(7127 + ring * 100000 + rows * 1000 + row).shuffle(order)
        permutations.append(order)
    for ply in range(depth):
        states.apply_many(list(range(rows)), [order[ply] for order in permutations])
    return native.SearchBatch(
        states, simulations=1, max_considered=2, deterministic_seed=7
    ).root_requests()


def worker(
    args: argparse.Namespace,
    pinned: dict[str, Any],
    config: Any,
    manifest: Any,
    actor: Any,
) -> dict[str, Any]:
    import torch
    from startrain.checkpoint import load_ema_checkpoint
    from startrain.inference import GraphInferenceAdapter, InferenceConfig
    from startrain.model import GraphResTNet
    from startrain.native import load_star_native
    from startrain.training import maybe_compile_model

    worker_initial_math = read_actor_inference_math()
    validate_actor_inference_math(worker_initial_math, phase="fresh worker")
    if not torch.cuda.is_available():
        raise ValueError("execution requires CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.set_num_threads(actor.blas_threads or actor.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.empty(1, device=device)
    uuid = _nvml_uuid(torch.cuda.get_device_properties(device).uuid)
    initial = _gpu_snapshot(uuid)
    if not _load_assessment([initial], declared="isolated", worker_pid=os.getpid())[
        "comparison_adoptable"
    ]:
        raise RuntimeError(f"GPU is not exclusive: {initial}")
    raw = GraphResTNet(config.model).eval().to(device)
    load_ema_checkpoint(
        manifest.checkpoint,
        model=raw,
        expected_model_config=asdict(config.model),
        expected_game_config=asdict(config.game),
        expected_run_id=manifest.run_id,
        expected_generation_family=manifest.generation_family,
        expected_sha256=manifest.checkpoint_sha256,
        expected_bytes=manifest.checkpoint_bytes,
        map_location=device,
    )
    refresh = config.orchestration.model_refresh
    model = maybe_compile_model(
        raw,
        enabled=config.train.compile,
        dynamic=refresh.inference_compile_dynamic,
        fullgraph=True,
        mode=refresh.inference_compile_mode,
        recompile_limit=None
        if refresh.inference_compile_dynamic
        else len(config.game.rings),
        isolate_recompiles=not refresh.inference_compile_dynamic,
    )
    source = asdict(refresh.inference)
    registry_entries = actor.actor_cohorts + 2 if actor.actor_cohorts > 1 else 1
    fields = InferenceConfig.__dataclass_fields__
    settings = {name: value for name, value in source.items() if name in fields}
    settings.update(
        precision=config.train.precision,
        score_utility_weight=config.selfplay.score_utility_weight,
        cache_max_entries=source["cache_max_entries"] // registry_entries,
        cache_max_bytes=source["cache_max_bytes"] // registry_entries,
        cuda_graph_max_bytes=args.graph_cache_bytes,
    )
    base = InferenceConfig(**settings)

    def adapter(entries: int) -> Any:
        return GraphInferenceAdapter(
            model,
            device=device,
            model_identity=manifest.model_identity,
            model_version=manifest.model_version,
            model_step=manifest.model_step,
            homogeneous_relational_bias=refresh.inference.homogeneous_relational_bias,
            config=replace(base, cuda_graphs=True, cuda_graph_max_entries=entries),
        )

    native = load_star_native(required=True)
    if native is None:
        raise ValueError("native engine unavailable")
    native.configure_rayon_threads(actor.native_threads or actor.cpu_threads)
    keys = sorted({tuple(key) for trace in pinned["traces"].values() for key in trace})
    requests = {key: native_request(native, *key) for key in keys}
    key_digest = hashlib.sha256()
    for request in requests.values():
        for semantic in request.inference_keys():
            key_digest.update(len(semantic).to_bytes(8, "little"))
            key_digest.update(semantic)
    reference, priming = prime_reference(
        adapter, requests, sync=lambda: torch.cuda.synchronize(device)
    )
    records = []
    for scenario in args.scenarios:
        trace = [tuple(key) for key in pinned["traces"][scenario]]
        for repeat in range(args.repeats):
            for entries in (16, 32) if repeat % 2 == 0 else (32, 16):
                runner = adapter(entries)
                try:
                    tick = time.perf_counter()
                    prepared = {
                        key: runner.prepare_requests(requests[key])
                        for key in set(trace)
                    }
                    preparation_seconds = time.perf_counter() - tick
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)

                    def memory(current: Any = runner) -> dict[str, int]:
                        graphs = current._graphs
                        return {
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                                device
                            ),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(
                                device
                            ),
                            "graph_retained_bytes": graphs.retained_bytes
                            if graphs is not None
                            else 0,
                        }

                    record = measure_arm(
                        runner,
                        trace,
                        prepared,
                        reference,
                        cycles=args.cycles,
                        sync=lambda: torch.cuda.synchronize(device),
                        snapshot=lambda: _gpu_snapshot(uuid),
                        memory=memory,
                    )
                    records.append(
                        {
                            "scenario": scenario,
                            "repeat": repeat,
                            "entries": entries,
                            "graph_cache_bytes": args.graph_cache_bytes,
                            "preparation_seconds": preparation_seconds,
                            **record,
                        }
                    )
                finally:
                    runner.close()
                    del runner
                    gc.collect()
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
    assessment = assess(
        records,
        args.scenarios,
        args.repeats,
        args.minimum_speedup,
        performance_policy=args.performance_policy,
    )
    production_scope = (
        args.graph_cache_bytes == pinned["production_graph_cache_bytes_per_model"]
        and set(args.scenarios) == set(SCENARIOS)
        and tuple(args.batch_sizes) == BUCKETS
        and pinned["profile_inference"]["cuda_graph_max_entries"] == 16
    )
    assessment["production_scope_complete"] = production_scope
    assessment["eligible_for_controlled_activation"] &= production_scope
    return {
        "plan": pinned,
        "worker_initial_math": worker_initial_math,
        "priming_math_before": priming["math_before"],
        "priming_math_after": priming["math_after"],
        "reference_priming": priming,
        "pid": os.getpid(),
        "gpu_uuid": uuid,
        "device_name": torch.cuda.get_device_name(device),
        "device_total_memory_bytes": torch.cuda.get_device_properties(
            device
        ).total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "compiled_priming_seconds": priming["seconds"],
        "native_request_keys_sha256": key_digest.hexdigest(),
        "reference_response_sha256": {
            f"{ring}:{rows}": digest for (ring, rows), digest in reference.items()
        },
        "records": records,
        "assessment": assessment,
    }


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    argument_parser = parser()
    args = argument_parser.parse_args(arguments)
    try:
        validate(args)
    except ValueError as error:
        argument_parser.error(str(error))
    pinned, config, manifest, actor = plan(args)
    identity = hashlib.sha256(json.dumps(pinned, sort_keys=True).encode()).hexdigest()
    if args.pinned_plan is not None and args.pinned_plan != identity:
        raise ValueError("benchmark inputs changed after planning")
    if not args.execute:
        print(json.dumps({**pinned, "plan_sha256": identity}, indent=2))
        return 0
    if args.output is None or args.output.exists():
        argument_parser.error("execution requires a new output path")
    if not args.worker:
        completed = _run_process(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                *arguments,
                "--worker",
                "--pinned-plan",
                identity,
            ],
            env=os.environ | {"TORCHINDUCTOR_COMPILE_THREADS": "1"},
            timeout=args.timeout_seconds,
        )
        if completed.returncode:
            raise RuntimeError(
                "graph capacity benchmark failed: " + completed.stderr[-8000:]
            )
        print(completed.stdout, end="")
        return 0
    report = worker(args, pinned, config, manifest, actor)
    report["plan_sha256"] = identity
    if report["assessment"]["production_scope_complete"]:
        try:
            recomputed = validate_graph_cache_report(
                report,
                source_config_sha256=pinned["source_config_sha256"],
                source_config_canonical_sha256=pinned["source_config_canonical_sha256"],
                model_identity=manifest.model_identity,
                manifest_sha256=manifest.manifest_sha256,
                checkpoint_sha256=manifest.checkpoint_sha256,
                profile_inference=pinned["profile_inference"],
                graph_cache_bytes=pinned["production_graph_cache_bytes_per_model"],
                actor_gpu_id=actor.gpu_id,
            )
            report["assessment"]["independent_recomputation"] = recomputed
        except ValueError as error:
            report["assessment"]["eligible_for_controlled_activation"] = False
            report["assessment"]["validation_error"] = str(error)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "plan_sha256": identity,
                "assessment": report["assessment"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
