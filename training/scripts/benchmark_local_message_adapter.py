#!/usr/bin/env python3
"""Bounded whole-adapter benchmark of exact local-message execution options.

Plan mode is read-only. --execute owns one child with a hard total deadline and
private compile caches. It never stops services or changes training artifacts.
The GPU must already be isolated. Use a frozen manifest and production profile.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import threading
import time
from typing import Any

from scripts.benchmark_training_shared_geometry import _run_process, gpu_ownership


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument(
        "--actor-gpu-id",
        type=int,
        help="Actor whose runtime overrides to benchmark; defaults to the first configured actor GPU",
    )
    parser.add_argument("--rings", type=int, nargs="+", default=[10])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["baseline", "project-first", "source-class"],
        default=["baseline", "source-class"],
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=971)
    parser.add_argument("--timeout-seconds", type=float, default=450)
    parser.add_argument("--max-memory-gib", type=float, default=32)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plan-sha256", help=argparse.SUPPRESS)
    return parser


def _validate(args: argparse.Namespace) -> None:
    if args.actor_gpu_id is not None and args.actor_gpu_id < 0:
        raise ValueError("actor GPU id must be nonnegative")
    if (
        not args.rings
        or len(set(args.rings)) != len(args.rings)
        or any(r not in (4, 6, 8, 10) for r in args.rings)
    ):
        raise ValueError("invalid rings")
    if (
        not args.batch_sizes
        or len(set(args.batch_sizes)) != len(args.batch_sizes)
        or len(args.batch_sizes) > 4
        or any(not 1 <= n <= 256 for n in args.batch_sizes)
    ):
        raise ValueError("batch sizes must be unique1..256, atmost4")
    if (
        not args.arms
        or args.arms[0] != "baseline"
        or len(args.arms) < 2
        or len(set(args.arms)) != len(args.arms)
    ):
        raise ValueError(
            "arms must begin with baseline and contain a unique alternative"
        )
    if (
        not 1 <= args.warmups <= 5
        or not 2 <= args.repeats <= 8
        or not 1 <= args.iterations <= 4
    ):
        raise ValueError("warmups1..5,repeats2..8,iterations1..4 required")
    if not 1 <= args.timeout_seconds <= 480 or not 1 <= args.max_memory_gib <= 48:
        raise ValueError("deadline1..480s and memory1..48GiB required")


def actor_experiment(config: Any, actor_gpu_id: int | None = None) -> tuple[Any, Any]:
    from startrain.actor import resolve_actor_experiment

    actors = config.orchestration.actor_gpus
    if not actors:
        raise ValueError("benchmark profile has no actor GPU")
    selected = actors[0].gpu_id if actor_gpu_id is None else actor_gpu_id
    gpu = next((gpu for gpu in actors if gpu.gpu_id == selected), None)
    if gpu is None:
        raise ValueError(f"GPU {selected} is not a configured actor")
    return resolve_actor_experiment(config, gpu), gpu


def runtime_flags(config: Any, gpu: Any) -> dict[str, Any]:
    refresh = config.orchestration.model_refresh
    registry_entries = gpu.actor_cohorts + 2 if gpu.actor_cohorts > 1 else 1
    return {
        "actor_gpu_id": gpu.gpu_id,
        "actor_pipeline": asdict(gpu.actor_pipeline)
        if gpu.actor_pipeline is not None
        else None,
        "cuda_graphs": refresh.inference.cuda_graphs,
        "compatible_cohort_work": refresh.compatible_cohort_work,
        "stream_completed_games": config.selfplay.stream_completed_games,
        "rolling_game_slots": config.selfplay.rolling_game_slots,
        "seed_contract": config.selfplay.seed_contract,
        "cohort_search_budgets": config.selfplay.cohort_search_budgets,
        "registry_max_entries": registry_entries,
        "per_model_cuda_graph_max_bytes": max(
            1, refresh.inference.cuda_graph_max_bytes // registry_entries
        ),
    }


def plan(args: argparse.Namespace) -> dict[str, Any]:
    from startrain.checkpoint import load_model_manifest
    from startrain.config import load_config
    import startrain.model as model_module

    config, gpu = actor_experiment(load_config(args.config), args.actor_gpu_id)
    manifest = load_model_manifest(args.checkpoint)
    source = Path(model_module.__file__).parent
    files = (
        "model.py",
        "local_message_inference.py",
        "local_message_triton.py",
        "inference.py",
    )
    return {
        "benchmark": "local-message-whole-adapter-v1",
        "model_identity": manifest.model_identity,
        "manifest_sha256": manifest.manifest_sha256,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "effective_config_sha256": hashlib.sha256(
            json.dumps(config.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "actor_gpu_id": gpu.gpu_id,
        "effective_actor_runtime": runtime_flags(config, gpu),
        "effective_adapter_config": asdict(
            adapter_config(config, actor_gpu_id=gpu.gpu_id)
        ),
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in files
        },
        "model_config": asdict(config.model),
        "precision": "bf16",
        "device": args.device,
        "rings": args.rings,
        "batch_sizes": args.batch_sizes,
        "arms": args.arms,
        "seed": args.seed,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "iterations": args.iterations,
        "timeout_seconds": args.timeout_seconds,
        "max_memory_gib": args.max_memory_gib,
        "scope": "GraphInferenceAdapter.evaluate fresh native requests, CPU features/H2D/neural/D2H/response included; predictor cache and dedup disabled equally to measure misses; no Elo claim",
    }


def requests(
    native: Any,
    ring: int,
    size: int,
    seed: int,
    variant: tuple[str, int, bool] = ("double", 1, False),
) -> Any:
    states = native.StateBatch(
        ring, size, mode=variant[0], handicap=variant[1], pie=variant[2]
    )
    nodes = states.node_count
    depths = [8 + row % 5 for row in range(size)]
    for ply in range(max(depths)):
        rows = [row for row, depth in enumerate(depths) if ply < depth]
        states.apply_many(rows, [(seed + row * 37 + ply * 19) % nodes for row in rows])
    return native.SearchBatch(
        states, simulations=1, max_considered=2, deterministic_seed=seed
    ).root_requests()


def adapter_config(config: Any, *, actor_gpu_id: int | None = None) -> Any:
    from startrain.inference import InferenceConfig

    config, gpu = actor_experiment(config, actor_gpu_id)
    source = config.orchestration.model_refresh.inference
    names = (
        "pinned_transfers",
        "pinned_buffer_slots",
        "preserve_broadcast_topology",
        "cuda_graphs",
        "cuda_graph_max_entries",
        "cuda_graph_max_bytes",
        "compact_inference_gather",
        "small_batch_graph_buckets",
    )
    values = {name: getattr(source, name) for name in names}
    # SharedModelRegistry reserves this share for each immutable model; using
    # the undivided global cap would misrepresent production graph fallbacks.
    values["cuda_graph_max_bytes"] = runtime_flags(config, gpu)[
        "per_model_cuda_graph_max_bytes"
    ]
    return InferenceConfig(
        precision="bf16",
        score_utility_weight=config.selfplay.score_utility_weight,
        cache_max_entries=0,
        cache_max_bytes=0,
        deduplicate=False,
        **values,
    )


def _heads(runner: Any, raw: Any, encoded: Any, ring: int) -> dict[str, Any]:
    import torch

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        bias = raw.prepare_inference_relational_bias(ring, dtype=torch.bfloat16)
        output = runner(
            *encoded.model_args(), homogeneous_ring=ring, inference_relation_bias=bias
        )
    result = {}
    for name, value in output._asdict().items():
        if value is None:
            continue
        if "policy" in name:
            value = value[encoded.legal_action_mask]
        elif name in ("ownership_logits", "alive_logits"):
            value = value[encoded.node_mask]
        elif name == "second_stone_logits":
            value = value[encoded.legal_action_mask]
        elif name in (
            "opponent_reply_logits",
            "final_peries_logits",
            "final_stars_logits",
        ):
            value = value[value > torch.finfo(value.dtype).min]
        result[name] = value.detach().float().cpu()
    return result


def compare_values(candidate: Any, reference: Any) -> dict[str, Any]:
    import torch

    a, b = torch.as_tensor(candidate).float(), torch.as_tensor(reference).float()
    if a.shape != b.shape or not bool(
        torch.isfinite(a).all() and torch.isfinite(b).all()
    ):
        raise ValueError("nonfinite or mismatched prediction")
    absolute = float((a - b).abs().max())
    relative = float((a - b).norm() / b.norm().clamp_min(1e-8))
    return {
        "max_absolute": absolute,
        "relative_l2": relative,
        "passed": bool(torch.allclose(a, b, rtol=0.01, atol=0.01)) and relative <= 0.01,
    }


def compare_policy(candidate: Any, reference: Any) -> dict[str, Any]:
    import torch

    if candidate.policy_offsets != reference.policy_offsets:
        raise ValueError("policy layouts differ")
    kl, maximum, agreement = [], 0.0, 0
    for start, end in zip(
        reference.policy_offsets[:-1], reference.policy_offsets[1:], strict=True
    ):
        p = torch.tensor(
            reference.policy_logits[start:end], dtype=torch.float64
        ).softmax(0)
        q = torch.tensor(
            candidate.policy_logits[start:end], dtype=torch.float64
        ).softmax(0)
        kl.append(
            float((p * (p.clamp_min(1e-300).log() - q.clamp_min(1e-300).log())).sum())
        )
        maximum = max(maximum, float((p - q).abs().max()))
        agreement += int(p.argmax() == q.argmax())
    return {
        "mean_kl": statistics.mean(kl),
        "max_probability_difference": maximum,
        "top_action_agreement": agreement / len(kl),
        "passed": max(kl) <= 1e-5 and maximum <= 1e-3,
    }


def _save(args: argparse.Namespace, report: dict[str, Any]) -> None:
    assert args.output is not None
    temporary = args.output.with_name("." + args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)


def worker(args: argparse.Namespace, pinned: dict[str, Any]) -> dict[str, Any]:
    import torch
    from startrain.checkpoint import load_ema_checkpoint, load_model_manifest
    from startrain.config import load_config
    from startrain.device import enable_fast_math
    from startrain.inference import GraphInferenceAdapter
    from startrain.model import GraphResTNet
    from startrain.native import encode_native_feature_data, load_star_native
    from startrain.training import maybe_compile_model

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("CUDA H100 required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.cuda.set_device(args.device)
    torch.cuda.mem_get_info(args.device)
    enable_fast_math(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    uuid = str(properties.uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    ownership = [gpu_ownership(uuid)]
    if not ownership[0]["verified"]:
        raise RuntimeError(f"GPU is not isolated: {ownership[0]}")
    torch.cuda.set_per_process_memory_fraction(
        min(args.max_memory_gib * 1024**3 / properties.total_memory, 0.8), args.device
    )
    stopped = threading.Event()

    def observe():
        while not stopped.wait(2):
            ownership.append(gpu_ownership(uuid))

    monitor = threading.Thread(target=observe, daemon=True)
    monitor.start()
    native = load_star_native(required=True)
    assert native is not None
    assert native.__file__ is not None
    binary = Path(native.__file__).resolve().parent / "star_native.abi3.so"
    config, gpu = actor_experiment(load_config(args.config), args.actor_gpu_id)
    manifest = load_model_manifest(args.checkpoint)
    inference = adapter_config(config, actor_gpu_id=gpu.gpu_id)
    refresh = config.orchestration.model_refresh
    adapters, models, runners = [], [], []
    report = {
        "plan": pinned,
        "status": "incomplete",
        "adoptable": False,
        "accuracy": [],
        "timing": [],
        "gpu": properties.name,
        "torch": torch.__version__,
        "native_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "inference_config": asdict(inference),
        "effective_actor_runtime": runtime_flags(config, gpu),
        "compile": {
            "enabled": config.train.compile,
            "dynamic": refresh.inference_compile_dynamic,
            "mode": refresh.inference_compile_mode,
        },
        "math": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
        },
    }
    started = time.monotonic()

    def record_process_peak() -> None:
        report["whole_process_peak_allocated_bytes"] = max(
            report.get("whole_process_peak_allocated_bytes", 0),
            torch.cuda.max_memory_allocated(args.device),
        )
        report["whole_process_peak_reserved_bytes"] = max(
            report.get("whole_process_peak_reserved_bytes", 0),
            torch.cuda.max_memory_reserved(args.device),
        )

    try:
        for mode in args.arms:
            model = GraphResTNet(config.model).eval()
            load_ema_checkpoint(
                manifest.checkpoint,
                model=model,
                map_location="cpu",
                expected_model_config=asdict(config.model),
                expected_game_config=asdict(config.game),
                expected_run_id=manifest.run_id,
                expected_generation_family=manifest.generation_family,
                expected_sha256=manifest.checkpoint_sha256,
                expected_bytes=manifest.checkpoint_bytes,
            )
            model.to(args.device)
            model.set_local_message_execution(mode)
            runner = maybe_compile_model(
                model,
                enabled=config.train.compile,
                dynamic=refresh.inference_compile_dynamic,
                fullgraph=True,
                mode=refresh.inference_compile_mode,
                recompile_limit=max(4, len(args.rings) * (len(args.batch_sizes) + 1)),
                isolate_recompiles=not refresh.inference_compile_dynamic,
            )
            adapter = GraphInferenceAdapter(
                runner,
                device=args.device,
                config=inference,
                model_version=manifest.model_version,
                model_step=manifest.model_step,
                model_identity=manifest.model_identity,
                homogeneous_relational_bias=refresh.inference.homogeneous_relational_bias,
            )
            models.append(model)
            runners.append(runner)
            adapters.append(adapter)
        if models[0].parameter_count() != 17_402_775:
            raise ValueError("benchmark requires production17,402,775parameter model")
        variants = (
            ("classic", 1, False),
            ("double", 1, False),
            ("classic", 2, False),
            ("double", 2, False),
            ("classic", 1, True),
            ("double", 1, True),
        )
        for ring in args.rings:
            # Same physical size as the first timing cell avoids an extra
            # expensive production compile while still exercising all six modes.
            size = args.batch_sizes[0]
            for variant in variants:
                request = requests(native, ring, size, args.seed, variant)
                encoded = encode_native_feature_data(
                    request.features, source="native_request"
                ).to(args.device)
                expected = _heads(runners[0], models[0], encoded, ring)
                reference_response = adapters[0].evaluate_detailed(request)
                for index in range(1, len(args.arms)):
                    actual = _heads(runners[index], models[index], encoded, ring)
                    heads = {
                        name: compare_values(actual[name], reference)
                        for name, reference in expected.items()
                    }
                    response = adapters[index].evaluate_detailed(
                        requests(native, ring, size, args.seed, variant)
                    )
                    utility = compare_values(
                        response.response.values, reference_response.response.values
                    )
                    legal_policy = compare_values(
                        response.response.policy_logits,
                        reference_response.response.policy_logits,
                    )
                    policy_distribution = compare_policy(
                        response.response, reference_response.response
                    )
                    passed = (
                        all(value["passed"] for value in heads.values())
                        and utility["passed"]
                        and legal_policy["passed"]
                        and policy_distribution["passed"]
                    )
                    report["accuracy"].append(
                        {
                            "ring": ring,
                            "variant": variant,
                            "arm": args.arms[index],
                            "heads": heads,
                            "utility": utility,
                            "legal_policy": legal_policy,
                            "policy_distribution": policy_distribution,
                            "passed": passed,
                        }
                    )
                    _save(args, report)
                    if not passed:
                        raise ValueError("whole-model/adapter numerical gate failed")
                del encoded
            for size in args.batch_sizes:
                warm = time.monotonic()
                for adapter in adapters:
                    for repeat in range(args.warmups):
                        adapter.evaluate(
                            requests(native, ring, size, args.seed + repeat + 1)
                        )
                torch.cuda.synchronize(args.device)
                warm_seconds = time.monotonic() - warm
                for index in range(1, len(args.arms)):
                    baseline, treatment = adapters[0], adapters[index]
                    times = [[], []]
                    phase_before = [
                        asdict(a.metrics_snapshot()) for a in (baseline, treatment)
                    ]
                    allocated_peaks, reserved_peaks = [], []
                    for repeat in range(args.repeats):
                        order = (0, 1, 1, 0) if repeat % 2 == 0 else (1, 0, 0, 1)
                        for arm in order:
                            # Fresh state/request objects force CPU feature generation
                            # on every miss. Native request construction is untimed.
                            samples = [
                                requests(
                                    native,
                                    ring,
                                    size,
                                    args.seed + 100 + repeat * args.iterations + k,
                                )
                                for k in range(args.iterations)
                            ]
                            torch.cuda.synchronize(args.device)
                            record_process_peak()
                            torch.cuda.reset_peak_memory_stats(args.device)
                            tick = time.perf_counter()
                            for sample in samples:
                                (baseline if arm == 0 else treatment).evaluate(sample)
                            torch.cuda.synchronize(args.device)
                            times[arm].append(
                                (time.perf_counter() - tick) / args.iterations
                            )
                            allocated_peaks.append(
                                torch.cuda.max_memory_allocated(args.device)
                            )
                            reserved_peaks.append(
                                torch.cuda.max_memory_reserved(args.device)
                            )
                            record_process_peak()
                    delta = [
                        {
                            key: value - before[key]
                            for key, value in asdict(adapter.metrics_snapshot()).items()
                        }
                        for adapter, before in zip(
                            (baseline, treatment), phase_before, strict=True
                        )
                    ]
                    medians = list(map(statistics.median, times))
                    record = {
                        "ring": ring,
                        "batch_size": size,
                        "arm": args.arms[index],
                        "baseline_seconds": times[0],
                        "treatment_seconds": times[1],
                        "median_baseline_seconds": medians[0],
                        "median_treatment_seconds": medians[1],
                        "speedup": medians[0] / medians[1],
                        "whole_process_peak_allocated_bytes_during_timing": max(
                            allocated_peaks
                        ),
                        "whole_process_peak_reserved_bytes_during_timing": max(
                            reserved_peaks
                        ),
                        "adapter_graph_memory": {
                            arm_name: {
                                "retained_bytes": adapter.efficiency_snapshot().get(
                                    "graph_retained_bytes", 0
                                ),
                                "configured_max_bytes": adapter.config.cuda_graph_max_bytes,
                            }
                            for arm_name, adapter in (
                                ("baseline", baseline),
                                (args.arms[index], treatment),
                            )
                        },
                        "memory_scope": "Both adapters share this process; allocator peaks are not per-arm memory comparisons.",
                        "adapter_metrics_delta": delta,
                        "warmup_and_compile_seconds": warm_seconds,
                        "scope": "complete adapter cache-miss calls; ABBA/BAAB balanced order",
                    }
                    expected_calls = args.repeats * 2 * args.iterations
                    if any(
                        d["neural_calls"] != expected_calls
                        or d["neural_rows"] != expected_calls * size
                        or d["cache_hits"] != 0
                        or d["graph_validation_failures"]
                        for d in delta
                    ):
                        raise RuntimeError(
                            "timing did not execute the expected neural workload"
                        )
                    report["timing"].append(record)
                    _save(args, report)
        ownership.append(gpu_ownership(uuid))
        report["ownership"] = ownership
        report["isolation_verified"] = all(o["verified"] for o in ownership)
        report["status"] = "passed"
        report["adoptable_by_arm"] = {
            arm: bool(
                report["isolation_verified"]
                and config.train.compile
                and report["timing"]
                and all(
                    r["speedup"] >= 1.05 for r in report["timing"] if r["arm"] == arm
                )
            )
            for arm in args.arms[1:]
        }
        report["adoptable"] = report["adoptable_by_arm"].get("source-class", False)
        return report
    finally:
        stopped.set()
        monitor.join(timeout=6)
        report["elapsed_seconds"] = time.monotonic() - started
        report["ownership"] = ownership
        record_process_peak()
        report["adapter_graph_memory_at_exit"] = {
            name: {
                "retained_bytes": adapter.efficiency_snapshot().get(
                    "graph_retained_bytes", 0
                ),
                "configured_max_bytes": adapter.config.cuda_graph_max_bytes,
            }
            for name, adapter in zip(args.arms, adapters)
        }
        for adapter in adapters:
            adapter.close()
        _save(args, report)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(arguments)
    try:
        _validate(args)
    except ValueError as error:
        parser.error(str(error))
    pinned = plan(args)
    digest = hashlib.sha256(json.dumps(pinned, sort_keys=True).encode()).hexdigest()
    if args.plan_sha256 is not None and args.plan_sha256 != digest:
        raise ValueError("benchmark source/profile/checkpoint changed after planning")
    if not args.execute:
        print(json.dumps(pinned | {"plan_sha256": digest}, indent=2))
        return 0
    if (
        args.output is None
        or (args.output.exists() and not args.worker)
        or not args.output.parent.is_dir()
    ):
        parser.error("execution requires a new output in an existing directory")
    profile_directory = args.config.resolve().parent
    if args.output.resolve().is_relative_to(profile_directory) and (
        (profile_directory / "status/coordinator.json").is_file()
        or (profile_directory / "replay/manifest.sqlite3").is_file()
    ):
        parser.error("benchmark output must be outside the profile/run directory")
    if args.worker:
        print(json.dumps(worker(args, pinned), indent=2))
        return 0
    with tempfile.TemporaryDirectory(prefix="local-message-benchmark-") as folder:
        cache = Path(folder)
        env = os.environ | {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "RAYON_NUM_THREADS": "2",
            "TORCHINDUCTOR_COMPILE_THREADS": "2",
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "CUDA_CACHE_PATH": str(cache / "cuda"),
        }
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *arguments,
            "--worker",
            "--plan-sha256",
            digest,
        ]
        report: dict[str, Any]
        try:
            process = _run_process(
                command,
                env=env,
                timeout=args.timeout_seconds - 5
                if args.timeout_seconds > 5
                else args.timeout_seconds,
            )
        except BaseException as error:
            report = (
                json.loads(args.output.read_text())
                if args.output.exists()
                else {"plan": pinned}
            )
            report.update(status="incomplete", adoptable=False, error=str(error))
            _save(args, report)
            raise
        if process.returncode:
            report = (
                json.loads(args.output.read_text())
                if args.output.exists()
                else {"plan": pinned}
            )
            report.update(
                status="failed", adoptable=False, error=process.stderr[-12000:]
            )
            _save(args, report)
            print(json.dumps(report, indent=2))
            return 1
        print(process.stdout)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
