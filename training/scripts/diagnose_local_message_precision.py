#!/usr/bin/env python3
"""Numerical oracle for the ring10/classic/B64/seed971 local-message failure.

No production state is changed. An owned CUDA child has a hard deadline and
private compiler caches. The original BF16 pairwise gate is reported unchanged.
Optional timing is diagnostic only and cannot make this report adoptable.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from scripts.benchmark_local_message_adapter import (
    actor_experiment,
    adapter_config,
    compare_policy,
    compare_values,
    requests,
    runtime_flags,
)
from scripts.benchmark_training_shared_geometry import _run_process, gpu_ownership

CRITERIA = {
    "fp32": {"rtol": 1e-4, "atol": 1e-5, "relative_l2": 1e-5},
    "identical_table": {"rtol": 1e-5, "atol": 1e-6, "relative_l2": 1e-6},
    "bf16_envelope": {
        "rmse_factor": 1.10,
        "max_absolute_factor": 1.25,
        "head_floor_scale": 1e-6,
        "policy_kl_factor": 1.10,
        "policy_kl_floor": 1e-7,
        "policy_max_probability_factor": 1.25,
        "policy_max_probability_floor": 1e-5,
    },
    "meaning": "A predeclared numerical-comparability screen, not a strength or deployment gate. Original BF16 gate is unchanged.",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--actor-gpu-id", type=int, default=1)
    p.add_argument("--primary-batch-sizes", nargs="+", type=int, default=[64, 128, 256])
    p.add_argument("--small-batch-size", type=int, default=16)
    p.add_argument("--skip-sweep", action="store_true")
    p.add_argument("--compiled-bf16", action="store_true")
    p.add_argument("--measure-timing", action="store_true")
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--timeout-seconds", type=float, default=450)
    p.add_argument("--max-memory-gib", type=float, default=32)
    p.add_argument("--output", type=Path)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--plan-sha256", help=argparse.SUPPRESS)
    return p


def validate(args: argparse.Namespace) -> None:
    if (
        not args.primary_batch_sizes
        or args.primary_batch_sizes[0] != 64
        or len(set(args.primary_batch_sizes)) != len(args.primary_batch_sizes)
        or any(b not in (64, 128, 256) for b in args.primary_batch_sizes)
    ):
        raise ValueError("primary batches must begin with 64 and be unique 64/128/256")
    if (
        not 1 <= args.small_batch_size <= 32
        or not 1 <= args.timeout_seconds <= 600
        or not 1 <= args.max_memory_gib <= 48
    ):
        raise ValueError(
            "small batch 1..32, deadline 1..600s, memory 1..48GiB required"
        )
    if (
        not 1 <= args.warmups <= 5
        or not 2 <= args.repeats <= 8
        or not 1 <= args.iterations <= 4
    ):
        raise ValueError("invalid bounded timing counts")


def cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    result = [
        {"ring": 10, "rows": b, "variant": ["classic", 1, False], "seed": 971}
        for b in args.primary_batch_sizes
    ]
    if not args.skip_sweep:
        variants = [
            ["classic", 1, False],
            ["double", 1, False],
            ["classic", 2, False],
            ["double", 2, False],
            ["classic", 1, True],
            ["double", 1, True],
        ]
        result += [
            {"ring": ring, "rows": args.small_batch_size, "variant": v, "seed": 971}
            for ring in (4, 6, 8, 10)
            for v in variants
        ]
    return result


def configure_math(fp32: bool, device: str) -> dict[str, Any]:
    import torch
    from deltreltrain.device import enable_fast_math

    if fp32:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        enable_fast_math(device)
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }


def metrics(
    actual: Any, expected: Any, criterion: dict[str, float] | None = None
) -> dict[str, Any]:
    import torch

    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    if a.shape != b.shape or not bool(
        torch.isfinite(a).all() and torch.isfinite(b).all()
    ):
        return {"finite": False, "passed": False}
    delta = a - b
    reference_rms = float(b.square().mean().sqrt())
    rmse = float(delta.square().mean().sqrt())
    maximum = float(delta.abs().max())
    relative = float(delta.norm() / b.norm().clamp_min(1e-12))
    result = {
        "finite": True,
        "rmse": rmse,
        "max_absolute": maximum,
        "reference_rms": reference_rms,
        "relative_l2": relative,
    }
    if criterion is not None:
        result["passed"] = (
            bool(torch.allclose(a, b, rtol=criterion["rtol"], atol=criterion["atol"]))
            and relative <= criterion["relative_l2"]
        )
    return result


def utility(outcome: Any, margin: Any, weight: float) -> Any:
    import torch
    from deltreltrain.contracts import SCORE_MARGIN_MIN, SCORE_MARGIN_MAX

    probabilities = outcome.float().softmax(-1)
    value = probabilities[:, 1] - probabilities[:, 0]
    if weight:
        support = torch.arange(
            SCORE_MARGIN_MIN, SCORE_MARGIN_MAX + 1, dtype=torch.float32
        )
        expected = (margin.float().softmax(-1) * support).sum(-1)
        value = (
            value + weight * expected / max(abs(SCORE_MARGIN_MIN), SCORE_MARGIN_MAX)
        ).clamp(-1, 1)
    return value


def capture(
    model: Any,
    encoded: Any,
    ring: int,
    precision: str,
    weight: float,
    *,
    runner: Any = None,
) -> dict[str, Any]:
    import torch

    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    with (
        torch.inference_mode(),
        torch.autocast(
            encoded.node_features.device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16",
        ),
    ):
        bias = model.prepare_inference_relational_bias(ring, dtype=dtype)
        outputs = (model if runner is None else runner)(
            *encoded.model_args(), homogeneous_ring=ring, inference_relation_bias=bias
        )
    raw = {}
    for name, value in outputs._asdict().items():
        if value is None:
            continue
        if name in (
            "opponent_reply_logits",
            "second_stone_logits",
            "final_shores_logits",
            "final_networks_logits",
        ):
            value = value[value > torch.finfo(value.dtype).min]
        raw[name] = value.detach().float().cpu()
    legal, node = encoded.legal_action_mask.cpu(), encoded.node_mask.cpu()
    heads = {
        name: (
            value[legal]
            if "policy" in name
            else value[node]
            if name in ("ownership_logits", "alive_logits")
            else value
        )
        for name, value in raw.items()
    }
    heads["utility"] = utility(
        raw["outcome_logits"], raw["score_margin_logits"], weight
    )
    counts = legal.sum(1)
    policy = SimpleNamespace(
        policy_offsets=[0, *counts.cumsum(0).tolist()],
        policy_logits=raw["policy_logits"][legal].tolist(),
    )
    return {"heads": heads, "policy": policy}


def compare_capture(
    actual: dict[str, Any], expected: dict[str, Any], *, fp32: bool
) -> dict[str, Any]:
    heads = {
        name: metrics(
            value, expected["heads"][name], CRITERIA["fp32"] if fp32 else None
        )
        for name, value in actual["heads"].items()
    }
    original = {
        name: compare_values(value, expected["heads"][name])
        for name, value in actual["heads"].items()
    }
    policy = compare_policy(actual["policy"], expected["policy"])
    return {
        "heads": heads,
        "policy": policy,
        "original_pairwise_gate": all(m["passed"] for m in original.values())
        and policy["passed"],
        "original_head_checks": original,
        "fp32_equivalent": all(m.get("passed", False) for m in heads.values())
        if fp32
        else None,
    }


def envelope(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    limits = CRITERIA["bf16_envelope"]
    heads = {}
    for name, c in candidate["heads"].items():
        b = baseline["heads"][name]
        if not c["finite"] or not b["finite"]:
            heads[name] = {"passed": False}
            continue
        floor = limits["head_floor_scale"] * max(b["reference_rms"], 1.0)
        rms_limit = limits["rmse_factor"] * b["rmse"] + floor
        maximum_limit = limits["max_absolute_factor"] * b["max_absolute"] + floor
        heads[name] = {
            "passed": c["rmse"] <= rms_limit and c["max_absolute"] <= maximum_limit,
            "baseline_rmse": b["rmse"],
            "candidate_rmse": c["rmse"],
            "rmse_limit": rms_limit,
            "baseline_max_absolute": b["max_absolute"],
            "candidate_max_absolute": c["max_absolute"],
            "max_absolute_limit": maximum_limit,
        }
    c, b = candidate["policy"], baseline["policy"]
    kl_limit = (
        limits["policy_kl_factor"] * max(b["mean_kl"], 0.0) + limits["policy_kl_floor"]
    )
    probability_limit = (
        limits["policy_max_probability_factor"] * b["max_probability_difference"]
        + limits["policy_max_probability_floor"]
    )
    policy = {
        "passed": c["mean_kl"] <= kl_limit
        and c["max_probability_difference"] <= probability_limit,
        "baseline_mean_kl": b["mean_kl"],
        "candidate_mean_kl": c["mean_kl"],
        "kl_limit": kl_limit,
        "candidate_max_probability_error": c["max_probability_difference"],
        "max_probability_error_limit": probability_limit,
    }
    return {
        "passed": all(h["passed"] for h in heads.values()) and policy["passed"],
        "heads": heads,
        "policy": policy,
    }


def _save(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def plan(args: argparse.Namespace) -> dict[str, Any]:
    from deltreltrain.config import load_config
    from deltreltrain.checkpoint import load_model_manifest
    import deltreltrain.model as model_module

    config, gpu = actor_experiment(load_config(args.config), args.actor_gpu_id)
    manifest = load_model_manifest(args.checkpoint)
    parent = Path(model_module.__file__).parent
    return {
        "diagnostic": "local-message-precision-oracle-v1",
        "criteria": CRITERIA,
        "cases": cases(args),
        "model_identity": manifest.model_identity,
        "manifest_sha256": manifest.manifest_sha256,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "raw_profile_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "runtime": runtime_flags(config, gpu),
        "effective_adapter_config": asdict(
            adapter_config(config, actor_gpu_id=gpu.gpu_id)
        ),
        "compile": {
            "enabled": config.train.compile,
            "dynamic": config.orchestration.model_refresh.inference_compile_dynamic,
            "mode": config.orchestration.model_refresh.inference_compile_mode,
            "fullgraph": True,
            "recompile_limit": 8,
            "isolate_recompiles": True,
        },
        "source_sha256": {
            name: hashlib.sha256((parent / name).read_bytes()).hexdigest()
            for name in (
                "model.py",
                "local_message_inference.py",
                "local_message_triton.py",
                "inference.py",
            )
        },
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compiled_bf16": args.compiled_bf16,
        "diagnostic_timing_requested": args.measure_timing,
        "device": args.device,
        "max_memory_gib": args.max_memory_gib,
        "timeout_seconds": args.timeout_seconds,
        "timing_counts": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "iterations": args.iterations,
        },
    }


def diagnostic_timing(
    args: argparse.Namespace,
    config: Any,
    gpu: Any,
    manifest: Any,
    weights: Any,
    *,
    progress: Any = None,
) -> list[dict[str, Any]]:
    import torch
    from deltreltrain.inference import GraphInferenceAdapter
    from deltreltrain.model import GraphResTNet
    from deltreltrain.native import load_deltrel_native
    from deltreltrain.training import maybe_compile_model

    native = load_deltrel_native(required=True)
    configure_math(False, args.device)
    refresh = config.orchestration.model_refresh
    adapters = []
    try:
        for mode in ("baseline", "source-class"):
            model = GraphResTNet(config.model).eval()
            model.load_state_dict(weights)
            model.to(args.device)
            model.set_local_message_execution(mode)
            runner = maybe_compile_model(
                model,
                enabled=config.train.compile,
                dynamic=refresh.inference_compile_dynamic,
                fullgraph=True,
                mode=refresh.inference_compile_mode,
                recompile_limit=8,
                isolate_recompiles=True,
            )
            adapters.append(
                GraphInferenceAdapter(
                    runner,
                    device=args.device,
                    config=adapter_config(config, actor_gpu_id=gpu.gpu_id),
                    homogeneous_relational_bias=refresh.inference.homogeneous_relational_bias,
                    model_version=manifest.model_version,
                    model_step=manifest.model_step,
                    model_identity=manifest.model_identity,
                )
            )
        records = []
        for size in args.primary_batch_sizes:
            started = time.monotonic()
            for adapter in adapters:
                for index in range(args.warmups):
                    adapter.evaluate(
                        requests(native, 10, size, 971 + index, ("classic", 1, False))
                    )
            torch.cuda.synchronize(args.device)
            warmup = time.monotonic() - started
            before = [asdict(a.metrics_snapshot()) for a in adapters]
            times = [[], []]
            for repeat in range(args.repeats):
                for arm in (0, 1, 1, 0) if repeat % 2 == 0 else (1, 0, 0, 1):
                    rows = [
                        requests(
                            native,
                            10,
                            size,
                            1071 + repeat * args.iterations + i,
                            ("classic", 1, False),
                        )
                        for i in range(args.iterations)
                    ]
                    torch.cuda.synchronize(args.device)
                    tick = time.perf_counter()
                    for request in rows:
                        adapters[arm].evaluate(request)
                    torch.cuda.synchronize(args.device)
                    times[arm].append((time.perf_counter() - tick) / args.iterations)
            deltas = [
                {
                    key: value - old[key]
                    for key, value in asdict(a.metrics_snapshot()).items()
                }
                for a, old in zip(adapters, before, strict=True)
            ]
            expected = 2 * args.repeats * args.iterations
            if any(
                d["neural_calls"] != expected
                or d["neural_rows"] != expected * size
                or d["cache_hits"]
                or d["graph_validation_failures"]
                for d in deltas
            ):
                raise ValueError(
                    "diagnostic timing did not execute expected neural work"
                )
            medians = list(map(statistics.median, times))
            records.append(
                {
                    "ring": 10,
                    "rows": size,
                    "variant": ["classic", 1, False],
                    "diagnostic_only": True,
                    "baseline_seconds": times[0],
                    "source_class_seconds": times[1],
                    "median_baseline_seconds": medians[0],
                    "median_source_class_seconds": medians[1],
                    "speedup": medians[0] / medians[1],
                    "warmup_compile_seconds": warmup,
                    "adapter_metrics_delta": deltas,
                    "adapter_graph_memory": [
                        a.efficiency_snapshot().get("graph_retained_bytes", 0)
                        for a in adapters
                    ],
                }
            )
            if progress is not None:
                progress(records)
        return records
    finally:
        for adapter in adapters:
            adapter.close()


def worker(args: argparse.Namespace, pinned: dict[str, Any]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    import deltreltrain.model as model_module
    from deltreltrain.checkpoint import load_ema_checkpoint, load_model_manifest
    from deltreltrain.config import load_config
    from deltreltrain.local_message_inference import _reference_mean, _source_class_mean
    from deltreltrain.native import encode_native_feature_data, load_deltrel_native
    from deltreltrain.training import maybe_compile_model

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("isolated CUDA required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.cuda.set_device(args.device)
    torch.cuda.mem_get_info(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    uuid = str(properties.uuid)
    uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
    ownership = [gpu_ownership(uuid)]
    if not ownership[0]["verified"]:
        raise ValueError("GPU is not isolated")
    torch.cuda.set_per_process_memory_fraction(
        min(args.max_memory_gib * 1024**3 / properties.total_memory, 0.8), args.device
    )
    config, gpu = actor_experiment(load_config(args.config), args.actor_gpu_id)
    manifest = load_model_manifest(args.checkpoint)
    native = load_deltrel_native(required=True)
    assert native is not None and native.__file__ is not None
    model = model_module.GraphResTNet(config.model).eval()
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
    weights = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    model.to(args.device)
    report = {
        "plan": pinned,
        "status": "incomplete",
        "adoptable": False,
        "cases": [],
        "timing": [],
        "gpu": properties.name,
        "torch": torch.__version__,
        "parameter_count": model.parameter_count(),
        "native_sha256": hashlib.sha256(
            (
                Path(native.__file__).resolve().parent / "deltrel_native.abi3.so"
            ).read_bytes()
        ).hexdigest(),
        "fp32_gate_passed": False,
        "original_bf16_gate_passed": False,
        "precision_envelope_passed": False,
    }
    if model.parameter_count() != 17_402_775:
        raise ValueError("production model required")
    stopped = threading.Event()

    def watch():
        while not stopped.wait(2):
            ownership.append(gpu_ownership(uuid))

    monitor = threading.Thread(target=watch, daemon=True)
    monitor.start()
    started = time.monotonic()
    try:
        for case_index, case in enumerate(pinned["cases"]):
            request = requests(
                native, case["ring"], case["rows"], case["seed"], tuple(case["variant"])
            )
            host = encode_native_feature_data(request.features, source="native_request")
            fixture_hash = hashlib.sha256()
            for tensor in host.model_args():
                fixture_hash.update(tensor.contiguous().numpy().tobytes())
            encoded = host.to(args.device)
            captures = {}
            tables = []
            math_modes = {}
            check_tables = case["variant"] == ["classic", 1, False]
            for precision in ("fp32", "bf16"):
                math_modes[precision] = configure_math(precision == "fp32", args.device)
                for label, mode, reducer in (
                    ("baseline", "baseline", "triton"),
                    ("project_first", "project-first", "triton"),
                    ("source_class_torch", "source-class", "torch"),
                    ("source_class_triton", "source-class", "triton"),
                ):
                    model.set_local_message_execution(
                        cast(model_module.LocalMessageExecution, mode)
                    )
                    block_index = [0]

                    def torch_aggregate(projected, embedding, indices, kinds, mask):
                        table = functional.silu(
                            projected.unsqueeze(2) + embedding[None, None, :, :]
                        )
                        expected = _reference_mean(table, indices, kinds, mask)
                        if check_tables:
                            actual = _source_class_mean(table, indices, kinds, mask)
                            tables.append(
                                {
                                    "precision": precision,
                                    "block": block_index[0],
                                    "table_dtype": str(table.dtype),
                                    **metrics(
                                        actual, expected, CRITERIA["identical_table"]
                                    ),
                                }
                            )
                        block_index[0] += 1
                        return expected

                    with (
                        patch.object(
                            model_module, "source_class_aggregate", torch_aggregate
                        )
                        if reducer == "torch"
                        else nullcontext()
                    ):
                        captures[precision + "_" + label] = capture(
                            model,
                            encoded,
                            case["ring"],
                            precision,
                            config.selfplay.score_utility_weight,
                        )
            if args.compiled_bf16 and case_index == 0:
                configure_math(False, args.device)
                for label, mode in (
                    ("baseline", "baseline"),
                    ("source_class_triton", "source-class"),
                ):
                    model.set_local_message_execution(
                        cast(model_module.LocalMessageExecution, mode)
                    )
                    runner = maybe_compile_model(
                        model,
                        enabled=True,
                        dynamic=config.orchestration.model_refresh.inference_compile_dynamic,
                        fullgraph=True,
                        mode=config.orchestration.model_refresh.inference_compile_mode,
                        recompile_limit=8,
                        isolate_recompiles=True,
                    )
                    captures["compiled_bf16_" + label] = capture(
                        model,
                        encoded,
                        case["ring"],
                        "bf16",
                        config.selfplay.score_utility_weight,
                        runner=runner,
                    )
                    del runner
            oracle = captures["fp32_baseline"]
            comparisons = {
                label: compare_capture(value, oracle, fp32=label.startswith("fp32"))
                for label, value in captures.items()
                if label != "fp32_baseline"
            }
            pairs = {
                label: compare_capture(
                    captures[label], captures["bf16_baseline"], fp32=False
                )
                for label in (
                    "bf16_project_first",
                    "bf16_source_class_torch",
                    "bf16_source_class_triton",
                )
            }
            envelopes = {
                label: envelope(comparisons[label], comparisons["bf16_baseline"])
                for label in (
                    "bf16_project_first",
                    "bf16_source_class_torch",
                    "bf16_source_class_triton",
                )
            }
            if "compiled_bf16_baseline" in captures:
                pairs["compiled_bf16_source_class_triton"] = compare_capture(
                    captures["compiled_bf16_source_class_triton"],
                    captures["compiled_bf16_baseline"],
                    fp32=False,
                )
                envelopes["compiled_bf16_source_class_triton"] = envelope(
                    comparisons["compiled_bf16_source_class_triton"],
                    comparisons["compiled_bf16_baseline"],
                )
            fp32_ok = all(
                value["fp32_equivalent"]
                for label, value in comparisons.items()
                if label.startswith("fp32")
            ) and all(t["passed"] for t in tables)
            record = {
                "fixture": case,
                "fixture_sha256": fixture_hash.hexdigest(),
                "math": math_modes,
                "against_fp32": comparisons,
                "against_bf16_baseline": pairs,
                "precision_envelopes": envelopes,
                "identical_table_checks": tables,
                "fp32_gate_passed": fp32_ok,
            }
            report["cases"].append(record)
            _save(args.output, report)
            del captures, encoded, host
            if not fp32_ok:
                break
        report["fp32_gate_passed"] = len(report["cases"]) == len(
            pinned["cases"]
        ) and all(c["fp32_gate_passed"] for c in report["cases"])
        report["original_bf16_gate_passed"] = all(
            p["original_pairwise_gate"]
            for c in report["cases"]
            for name, p in c["against_bf16_baseline"].items()
            if "source_class_triton" in name
        )
        report["precision_envelope_passed"] = all(
            p["passed"]
            for c in report["cases"]
            for name, p in c["precision_envelopes"].items()
            if "source_class_triton" in name
        )
        del model
        torch.cuda.empty_cache()
        ownership.append(gpu_ownership(uuid))
        _save(args.output, report)
        if (
            args.measure_timing
            and report["fp32_gate_passed"]
            and all(o["verified"] for o in ownership)
        ):

            def save_timing(records):
                report["timing"] = records
                _save(args.output, report)

            report["timing"] = diagnostic_timing(
                args, config, gpu, manifest, weights, progress=save_timing
            )
        elif args.measure_timing:
            report["timing_skip_reason"] = (
                "FP32 equivalence/kernel or isolation prerequisite failed"
            )
        report["status"] = "complete"
        return report
    finally:
        stopped.set()
        monitor.join(timeout=6)
        ownership.append(gpu_ownership(uuid))
        report["ownership"] = ownership
        report["isolation_verified"] = all(o["verified"] for o in ownership)
        report["elapsed_seconds"] = time.monotonic() - started
        report["whole_process_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
            args.device
        )
        report["whole_process_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(
            args.device
        )
        report["adoptable"] = False
        _save(args.output, report)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    args = p.parse_args(arguments)
    try:
        validate(args)
    except ValueError as error:
        p.error(str(error))
    pinned = plan(args)
    digest = hashlib.sha256(json.dumps(pinned, sort_keys=True).encode()).hexdigest()
    if args.plan_sha256 is not None and args.plan_sha256 != digest:
        raise ValueError("oracle inputs changed after planning")
    if not args.execute:
        print(json.dumps(pinned | {"plan_sha256": digest}, indent=2))
        return 0
    if (
        args.output is None
        or (args.output.exists() and not args.worker)
        or not args.output.parent.is_dir()
    ):
        p.error("new output in existing directory required")
    profile_directory = args.config.resolve().parent
    if args.output.resolve().is_relative_to(profile_directory) and (
        (profile_directory / "status/coordinator.json").is_file()
        or (profile_directory / "replay/manifest.sqlite3").is_file()
    ):
        p.error("diagnostic output must be outside the profile/run directory")
    if args.worker:
        result = worker(args, pinned)
        print(json.dumps(result, indent=2))
        return 0 if result["fp32_gate_passed"] and result["isolation_verified"] else 1
    with tempfile.TemporaryDirectory(prefix="message-oracle-") as folder:
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
        try:
            process = _run_process(
                command, env=env, timeout=max(1, args.timeout_seconds - 5)
            )
        except subprocess.TimeoutExpired:
            result: dict[str, Any] = (
                json.loads(args.output.read_text())
                if args.output.exists()
                else {"plan": pinned}
            )
            result.update(status="timeout", adoptable=False)
            _save(args.output, result)
            return 1
        if process.returncode:
            result = (
                json.loads(args.output.read_text())
                if args.output.exists()
                else {"plan": pinned}
            )
            result.update(
                status="failed", adoptable=False, error=process.stderr[-12000:]
            )
            _save(args.output, result)
        print(process.stdout)
        return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
