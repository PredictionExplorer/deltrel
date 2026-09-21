#!/usr/bin/env python3
"""Bounded, isolated frozen-model smoke for selected already-decided arena tails.

Without --execute, verify and print a CPU-only plan. Execution requires a GPU
reserved by the caller; this tool never stops another process or edits a run.
Default tails retain at most eight original searched moves per seat. These
selected late-game timings do not estimate complete-arena throughput or Elo.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, cast

if __package__:
    from .benchmark_graph_buckets import _gpu_snapshot, _load_assessment, _nvml_uuid
    from .benchmark_training_shared_geometry import _run_process
    from .prepare_arena_occupancy_benchmark import _native_extension_path
else:
    from benchmark_graph_buckets import _gpu_snapshot, _load_assessment, _nvml_uuid
    from benchmark_training_shared_geometry import _run_process
    from prepare_arena_occupancy_benchmark import _native_extension_path


REPORT = "deltreltrain-arena-exact-clinch-tail-smoke"
TRAINING_ROOT = Path(__file__).resolve().parents[1]


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_cases(saved: dict, proof: dict, max_tail_moves: int) -> list[dict]:
    """Select fixed pair indices, never selecting by win/loss outcome.

    Handicap cases explicitly retain severity nine and its original PDA. A
    capped prefix can be later than the first clinch; that bias is intentional
    for a bounded deployment smoke and is recorded per seat.
    """
    from deltreltrain.selfplay import GameVariant

    if type(max_tail_moves) is not int or not 1 <= max_tail_moves <= 128:
        raise ValueError("max_tail_moves must be an integer in 1..128")
    evidence = {
        (row["variant"], row["pair"], row["seat"]): row
        for row in proof["games"]
        if row["completed"]
    }
    entries = {
        (row["variant"], row["pair"], row["candidate_player"]): row
        for row in saved["game_states"]
        if row.get("result") is not None
    }
    if len(evidence) != sum(bool(row["completed"]) for row in proof["games"]):
        raise ValueError("duplicate completed proof identities")
    completed_entries = [
        row for row in saved["game_states"] if row.get("result") is not None
    ]
    if (
        len(entries) != len(completed_entries)
        or len({row["ring"] for row in completed_entries}) != 1
    ):
        raise ValueError("benchmark requires unique completed games on one board size")
    cases = []
    for mode in ("classic", "double"):
        for pie, handicap in ((False, 1), (True, 1), (False, 9)):
            variant = GameVariant(mode=mode, pie=pie, handicap=handicap)
            pairs = sorted({key[1] for key in entries if key[0] == variant.label})
            selected = next(
                (
                    pair
                    for pair in pairs
                    if all(
                        (variant.label, pair, seat) in evidence
                        and (variant.label, pair, seat) in entries
                        for seat in (0, 1)
                    )
                ),
                None,
            )
            if selected is None:
                raise ValueError(f"missing completed proof pair for {variant.label}")
            starts, expected, metadata = [], [], []
            for seat in (0, 1):
                key = variant.label, selected, seat
                original, witness = entries[key], evidence[key]
                first = witness["clinch_move"]
                length = len(original["actions"])
                if (
                    type(first) is not int
                    or not 0 <= first < length
                    or witness["searched_moves"] != length
                    or witness["clinch_winner"] != original["result"]["winner"]
                ):
                    raise ValueError("clinch evidence disagrees with completed history")
                prefix = max(first, length - max_tail_moves)
                entry = json.loads(json.dumps(original))
                entry["actions"] = entry["actions"][:prefix]
                entry["result"] = None
                starts.append(entry)
                expected.append(original["result"]["winner"])
                metadata.append(
                    {
                        "seat": seat,
                        "first_proven_move": first,
                        "prefix_moves": prefix,
                        "original_searched_moves": length,
                        "original_tail_moves": length - prefix,
                        "opening_seed": original["opening_seed"],
                        "opening_action": original["opening_action"],
                        "expected_winner": original["result"]["winner"],
                        "pda": original["result"]["pda"],
                    }
                )
            snapshot = json.loads(json.dumps(saved))
            snapshot.update(game_states=starts, games=[], pairs=[], progress={})
            cases.append(
                {
                    "variant": variant.label,
                    "pair": selected,
                    "ring": starts[0]["ring"],
                    "seats": metadata,
                    "expected_winners": expected,
                    "resume": snapshot,
                }
            )
    return cases


def validate_cases(cases: list[dict], native, arena_config) -> None:
    from deltreltrain.arena import ArenaGame, ArenaRunner
    from deltreltrain.selfplay import GameVariant

    for case in cases:
        saved = case["resume"]
        runner = ArenaRunner(
            native_module=native,
            candidate=cast(Any, SimpleNamespace(model_version=saved["candidate"])),
            baseline=cast(Any, SimpleNamespace(model_version=saved["baseline"])),
            config=replace(arena_config, exact_clinch_termination=False),
            stable_pair_seeds=True,
        )
        runner._initialize_resume(saved, None)
        variant = GameVariant.parse(case["variant"])
        for entry, winner in zip(
            saved["game_states"], case["expected_winners"], strict=True
        ):
            game = ArenaGame(
                ring=case["ring"],
                pair=case["pair"],
                candidate_player=entry["candidate_player"],
                opening_seed=entry["opening_seed"],
                opening_action=entry["opening_action"],
                forced_opening=entry["opening_action"] is not None,
                winner=winner,
                outcome=1 if winner == entry["candidate_player"] else -1,
                searched_moves=len(entry["actions"]),
                variant=variant.label,
                segment=variant.segment,
                swapped=runner.native.StateBatch(case["ring"], 1).node_count
                in entry["actions"],
                pda=max(runner._pda_seats(variant)),
            )
            runner._verify_resume_winner(entry, variant, game)


def promotion_runtime_config(config, candidate, baseline):
    """Use the exact immutable-match seed that the promotion worker uses.

    The profile seed is a seed namespace, not the actual frozen arena seed.
    Reusing the production derivation keeps strict snapshot/seed validation.
    No supervisor is started and no model/run files are modified.
    """
    from deltreltrain.promotion import PromotionSupervisor

    arena = PromotionSupervisor._arena_config(
        cast(Any, SimpleNamespace(experiment=config)), candidate, baseline
    )
    return replace(config, arena=arena)


def prepare(args):
    from deltreltrain.checkpoint import load_model_manifest
    from deltreltrain.config import load_config
    from deltreltrain.native import load_deltrel_native

    wrapper = json.loads(args.resume.read_text())
    saved = wrapper["arena_state"]
    proof = json.loads(args.proof.read_text())
    config = load_config(args.profile)
    native = load_deltrel_native(required=True)
    candidate_path = args.candidate_manifest or Path(wrapper["candidate_manifest"])
    baseline_path = args.baseline_manifest or Path(wrapper["baseline_manifest"])
    candidate = load_model_manifest(candidate_path)
    baseline = load_model_manifest(baseline_path)
    if (
        candidate.model_identity != saved["candidate"]
        or baseline.model_identity != saved["baseline"]
    ):
        raise ValueError("frozen model identities do not match saved arena")
    profile_arena_seed = config.arena.seed
    config = promotion_runtime_config(config, candidate, baseline)
    cases = select_cases(saved, proof, args.max_tail_moves)
    validate_cases(cases, native, config.arena)
    files = [
        args.profile,
        args.resume,
        args.proof,
        candidate_path,
        baseline_path,
        candidate.checkpoint,
        baseline.checkpoint,
        Path(__file__),
        *[
            TRAINING_ROOT / "scripts" / name
            for name in (
                "benchmark_graph_buckets.py",
                "benchmark_training_shared_geometry.py",
                "prepare_arena_occupancy_benchmark.py",
            )
        ],
        _native_extension_path(native),
        *sorted((TRAINING_ROOT / "deltreltrain").glob("*.py")),
    ]
    plan = {
        "report": REPORT,
        "schema_version": 1,
        "scope": "selected-already-proven-late-game-tails-not-whole-arena-or-Elo",
        "selection": "first-completed-pair-per-mode-with-handicap9-and-capped-tail",
        "max_tail_moves": args.max_tail_moves,
        "timeout_seconds": args.timeout_seconds,
        "candidate": candidate.model_identity,
        "baseline": baseline.model_identity,
        "candidate_step": candidate.model_step,
        "baseline_step": baseline.model_step,
        "search": saved["candidate_search"],
        "profile_arena_seed": profile_arena_seed,
        "effective_promotion_arena_seed": config.arena.seed,
        "precision": config.train.precision,
        "gpu_uuid": args.gpu_uuid,
        "source_and_input_sha256": {
            str(path.resolve()): file_digest(path) for path in files
        },
        "cases": [
            {key: value for key, value in case.items() if key != "resume"}
            for case in cases
        ],
    }
    return plan, cases, config, candidate, baseline, native


def play_case(case, config, native, candidate, baseline, *, enabled, saved=None):
    from deltreltrain.arena import ArenaRunner
    from deltreltrain.selfplay import GameVariant

    subject = ArenaRunner(
        native_module=native,
        candidate=candidate,
        baseline=baseline,
        config=replace(config.arena, exact_clinch_termination=enabled),
        stable_pair_seeds=True,
    )
    subject._initialize_resume(saved or case["resume"], lambda _snapshot: None)
    variant = GameVariant.parse(case["variant"])
    specs = subject._pair_specifications(case["ring"], [case["pair"]], variant)
    with subject._inference_owner() as executor:
        games = subject._play_ring_batch(
            case["ring"],
            specs,
            variant=variant,
            progress=None,
            inference_executor=executor,
            stop_requested=lambda: False,
        )
    if len(games) != 2 or [game.winner for game in games] != case["expected_winners"]:
        raise RuntimeError("frozen-model continuation disagrees with proven winners")
    snapshot = cast(dict[str, Any], subject._resume_snapshot())
    for entry, original in zip(
        snapshot["game_states"], case["resume"]["game_states"], strict=True
    ):
        prefix = original["actions"]
        if entry["actions"][: len(prefix)] != prefix:
            raise RuntimeError("benchmark altered an actual saved action prefix")
        if enabled and entry["actions"] != prefix:
            raise RuntimeError("proven treatment appended searched or synthetic moves")
    return games, snapshot


def execute_worker(
    args, plan, cases, config, candidate_manifest, baseline_manifest, native
):
    import torch
    from deltreltrain.device import peak_memory_stats, reset_peak_memory_stats
    from deltreltrain.promotion import load_manifest_evaluator

    preflight = _gpu_snapshot(args.gpu_uuid)
    if preflight.get("verified") is not True or preflight.get("owners"):
        raise RuntimeError("reserved GPU is not verifiably idle")
    device = "cuda:0"
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("benchmark requires exactly one visible reserved CUDA GPU")
    properties = torch.cuda.get_device_properties(device)
    if _nvml_uuid(properties.uuid) != args.gpu_uuid or "H100" not in properties.name:
        raise RuntimeError("selected device is not the reserved H100")
    torch.set_num_threads(2)
    torch.cuda.set_device(device)
    allocation = torch.empty(1, device=device)
    observations, stop = [], threading.Event()

    def monitor():
        while not stop.is_set():
            observations.append(_gpu_snapshot(args.gpu_uuid))
            stop.wait(1.0)

    sampler = threading.Thread(target=monitor, daemon=True)
    sampler.start()
    started = time.perf_counter()
    records = []
    try:
        loading = time.perf_counter()
        candidate = load_manifest_evaluator(config, candidate_manifest, device=device)
        baseline = load_manifest_evaluator(config, baseline_manifest, device=device)
        torch.cuda.synchronize()
        loading_seconds = time.perf_counter() - loading
        for index, case in enumerate(cases):
            outputs = {}
            for enabled in (False, True) if index % 2 == 0 else (True, False):
                for adapter in (candidate, baseline):
                    adapter.clear_prediction_cache()
                torch.cuda.synchronize()
                before = [
                    asdict(adapter.metrics_snapshot())
                    for adapter in (candidate, baseline)
                ]
                reset_peak_memory_stats(device)
                begin = time.perf_counter()
                games, snapshot = play_case(
                    case, config, native, candidate, baseline, enabled=enabled
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - begin
                after = [
                    asdict(adapter.metrics_snapshot())
                    for adapter in (candidate, baseline)
                ]
                counters = {
                    key: sum(after[i][key] - before[i][key] for i in range(2))
                    for key in (
                        "evaluator_calls",
                        "evaluator_rows",
                        "neural_calls",
                        "neural_rows",
                    )
                }
                if enabled and any(counters.values()):
                    raise RuntimeError(
                        "already-proven treatment performed neural inference"
                    )
                if not enabled and counters["neural_calls"] <= 0:
                    raise RuntimeError(
                        "full-board control did not exercise frozen neural inference"
                    )
                records.append(
                    {
                        "variant": case["variant"],
                        "pair": case["pair"],
                        "arm": "exact-clinch" if enabled else "full-board-control",
                        "wall_seconds": elapsed,
                        "games": [asdict(game) for game in games],
                        "counters": counters,
                        "memory": peak_memory_stats(device),
                        "searched_moves_added": sum(
                            len(entry["actions"]) for entry in snapshot["game_states"]
                        )
                        - sum(
                            len(entry["actions"])
                            for entry in case["resume"]["game_states"]
                        ),
                    }
                )
                outputs[enabled] = snapshot
            before_rows = candidate.evaluator_rows + baseline.evaluator_rows
            restored, _ = play_case(
                case,
                config,
                native,
                candidate,
                baseline,
                enabled=False,
                saved=outputs[True],
            )
            if (
                candidate.evaluator_rows + baseline.evaluator_rows != before_rows
                or len(restored) != 2
            ):
                raise RuntimeError("rollback resume replayed proven completed games")
            if outputs[True]["pairs"] != outputs[False]["pairs"]:
                raise RuntimeError("treatment changed role-reversed pair accounting")
        for path, expected in plan["source_and_input_sha256"].items():
            if file_digest(Path(path)) != expected:
                raise RuntimeError(
                    "pinned benchmark source or input changed during execution"
                )
    finally:
        stop.set()
        sampler.join(timeout=8)
        if sampler.is_alive():
            raise RuntimeError("GPU ownership monitor did not stop")
        observations.append(_gpu_snapshot(args.gpu_uuid))
        del allocation
    assessment = _load_assessment(
        observations, declared="isolated", worker_pid=os.getpid()
    )
    if not assessment["comparison_adoptable"]:
        raise RuntimeError(
            "GPU ownership was not isolated throughout observed benchmark"
        )
    return {
        **plan,
        "plan_sha256": digest(plan),
        "status": "passed",
        "model_loading_seconds": loading_seconds,
        "total_worker_wall_seconds": time.perf_counter() - started,
        "records": records,
        "isolation": assessment,
        "gpu_observations": observations,
        "gpu": properties.name,
        "torch": torch.__version__,
        "math_settings": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "limitations": [
            "Selected already-decided tails, capped at eight moves by default; not the earliest-clinch work savings experiment.",
            "One counterbalanced pair per six modes, including handicap9; not an Elo estimate or full-arena throughput forecast.",
            "Cold inference compilation is charged to the first control cases; no end-to-end speedup claim is made.",
            "Proof-overhead on unclinched positions and general batching-induced BF16 changes need separate sustained evaluation.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--max-tail-moves", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pinned-plan", help=argparse.SUPPRESS)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    if not math.isfinite(args.timeout_seconds) or not 30 <= args.timeout_seconds <= 600:
        parser.error("timeout must be 30..600 seconds")
    if args.execute and (
        not args.gpu_uuid
        or not args.gpu_uuid.startswith("GPU-")
        or args.output_dir is None
    ):
        parser.error("--execute requires reserved --gpu-uuid and a new --output-dir")
    if args.worker and (not args.execute or args.pinned_plan is None):
        parser.error("worker requires an executed pinned plan")
    plan, cases, config, candidate, baseline, native = prepare(args)
    identity = digest(plan)
    if args.pinned_plan is not None and args.pinned_plan != identity:
        raise ValueError("source or inputs changed after planning")
    if not args.execute:
        print(json.dumps({**plan, "plan_sha256": identity}, indent=2))
        return 0
    output_path = args.output_dir.resolve()
    run_root = Path(config.orchestration.directories.root).expanduser().resolve()
    if any(
        output_path.is_relative_to(path.resolve()) for path in (TRAINING_ROOT, run_root)
    ):
        raise ValueError(
            "benchmark output must be outside the source tree and production run"
        )
    if args.worker:
        report = execute_worker(args, plan, cases, config, candidate, baseline, native)
        with (args.output_dir / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(
            json.dumps(
                {"status": "passed", "report": str(args.output_dir / "report.json")}
            )
        )
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=2))
    env = os.environ | {
        "CUDA_VISIBLE_DEVICES": args.gpu_uuid,
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "RAYON_NUM_THREADS": "2",
        "TORCHINDUCTOR_COMPILE_THREADS": "2",
        "TORCHINDUCTOR_CACHE_DIR": str(args.output_dir / "inductor-cache"),
        "TRITON_CACHE_DIR": str(args.output_dir / "triton-cache"),
    }
    try:
        result = _run_process(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                *arguments,
                "--worker",
                "--pinned-plan",
                identity,
            ],
            env=env,
            timeout=args.timeout_seconds,
        )
        (args.output_dir / "stdout.log").write_text(result.stdout)
        (args.output_dir / "stderr.log").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(
                f"benchmark worker exited {result.returncode}: {result.stderr[-4000:]}"
            )
    except BaseException as error:
        (args.output_dir / "failure.json").write_text(
            json.dumps({"status": "failed", "error": str(error)})
        )
        raise
    print(result.stdout, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
