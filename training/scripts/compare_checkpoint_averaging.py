#!/usr/bin/env python3
"""Frozen, resumable raw/EMA matches against one champion; never promotes.

First freeze with --source-run-root, --checkpoint, --champion-checkpoint,
--output-dir, and --plan-only. Subsequent invocations need only --output-dir.
Each session runs one arm for at most --session-seconds plus the in-flight
search. SIGTERM/SIGINT save the current game positions. CUDA execution requires
an explicitly reserved GPU, addressed by its full CUDA_VISIBLE_DEVICES UUID.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Literal, cast

import torch

from scripts.run_lineage_arena import load_candidate
from startrain.arena import (
    ArenaPair,
    ArenaRunner,
    summarize_completed_arena_pairs,
    summarize_pairs,
)
from startrain.balanced_evaluation import balanced_cells, category
from startrain.checkpoint import (
    extract_verified_checkpoint_config,
    sha256_file,
    verify_file,
)
from startrain.config import ArenaConfig
from startrain.native import load_star_native
from startrain.runtime import SignalLatch, atomic_json
from startrain.search_options import parse_search_execution
from startrain.selfplay import GameVariant

RESULT_KIND = "checkpoint_averaging_diagnostic"
PLAN_NAME = "diagnostic-plan.json"
ARMS = ("raw", "ema")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _isolated_output(source: Path, output: Path) -> Path:
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("diagnostic output must be outside and separate from the run")
    return output


@contextmanager
def _exclusive_output(output: Path) -> Iterator[None]:
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".diagnostic.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another diagnostic session owns this output") from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def diagnostic_config(
    *,
    pairs_per_cell: int = 8,
    simulations: int = 256,
    seed: int = 17,
    max_considered: int = 16,
) -> ArenaConfig:
    if type(pairs_per_cell) is not int or pairs_per_cell < 4 or pairs_per_cell % 4:
        raise ValueError("pairs per cell must cover complete 2/4/6/9 handicap cycles")
    return ArenaConfig(
        rings=(10,),
        balanced_cells=True,
        variant_policy="pie_even",
        pairs_per_ring=pairs_per_cell,
        minimum_pairs_per_ring=pairs_per_cell,
        max_pairs_per_ring=pairs_per_cell,
        simulations=simulations,
        max_considered=min(max_considered, simulations),
        seed=seed,
    )


def _freeze_checkpoint(source: Path, destination: Path) -> dict[str, object]:
    digest = sha256_file(source)
    size = source.stat().st_size
    temporary = destination.with_suffix(".tmp")
    try:
        shutil.copyfile(source, temporary)
        verify_file(temporary, expected_sha256=digest, expected_bytes=size)
        verify_file(source, expected_sha256=digest, expected_bytes=size)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        destination.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return {"source": str(source), "sha256": digest, "bytes": size}


def _implementation_hashes() -> dict[str, str]:
    training = Path(__file__).resolve().parents[1]
    return {
        name: sha256_file(training / name)
        for name in (
            "scripts/compare_checkpoint_averaging.py",
            "scripts/run_lineage_arena.py",
            "startrain/arena.py",
            "startrain/balanced_evaluation.py",
            "startrain/checkpoint.py",
            "startrain/contracts.py",
            "startrain/native.py",
        )
    }


def freeze_plan(
    *,
    source_run_root: Path,
    checkpoint: Path,
    champion_checkpoint: Path,
    output: Path,
    config: ArenaConfig,
    precision: Literal["fp32", "bf16"] = "bf16",
) -> dict[str, Any]:
    """Copy only immutable inputs; all writes stay in the isolated output."""
    source = source_run_root.expanduser().resolve(strict=True)
    output = _isolated_output(source, output)
    if (output / PLAN_NAME).exists():
        raise ValueError(
            "diagnostic plan already exists; resume with --output-dir only"
        )
    checkpoints = {
        "checkpoint": checkpoint.expanduser().resolve(strict=True),
        "champion": champion_checkpoint.expanduser().resolve(strict=True),
    }
    if any(source not in path.parents for path in checkpoints.values()):
        raise ValueError("both checkpoint inputs must belong to the source run")
    expected = diagnostic_config(
        pairs_per_cell=config.pairs_per_ring,
        simulations=config.simulations,
        seed=config.seed,
        max_considered=config.max_considered,
    )
    if config != expected or precision not in ("fp32", "bf16"):
        raise ValueError("diagnostic requires its fixed four-cell search contract")
    output.mkdir(parents=True, exist_ok=True)
    frozen = {
        name: _freeze_checkpoint(path, output / f"{name}.pt")
        for name, path in checkpoints.items()
    }
    current = extract_verified_checkpoint_config(output / "checkpoint.pt")
    extract_verified_checkpoint_config(
        output / "champion.pt", expected_game_config=current.game_config
    )
    plan = {
        "schema_version": 1,
        "result_kind": RESULT_KIND,
        "source_run_root": str(source),
        "checkpoints": frozen,
        "arena": asdict(config),
        "precision": precision,
        "implementation_hashes": _implementation_hashes(),
        "interpretation": (
            "Diagnostic only. Both raw and EMA from the same checkpoint play the "
            "same frozen champion EMA, openings, seats, and search budget. "
            "No promotion, checkpoint publication, or learner/replay changes."
        ),
    }
    plan["plan_sha256"] = _digest(plan)
    atomic_json(output / PLAN_NAME, plan)
    (output / PLAN_NAME).chmod(0o444)
    return _read_json(output / PLAN_NAME)


def verify_plan(output: Path) -> tuple[dict[str, Any], ArenaConfig]:
    plan = _read_json(output / PLAN_NAME)
    contract = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (
        plan.get("schema_version") != 1
        or plan.get("result_kind") != RESULT_KIND
        or _digest(contract) != plan.get("plan_sha256")
        or plan.get("implementation_hashes") != _implementation_hashes()
    ):
        raise ValueError("diagnostic plan or implementation hash changed")
    _isolated_output(Path(plan["source_run_root"]), output)
    for name in ("checkpoint", "champion"):
        path = output / f"{name}.pt"
        if path.is_symlink():
            raise ValueError("frozen checkpoint must not be a symlink")
        evidence = plan["checkpoints"][name]
        verify_file(
            path, expected_sha256=evidence["sha256"], expected_bytes=evidence["bytes"]
        )
    values = dict(plan["arena"])
    for name in (
        "rings",
        "segment_handicaps",
        "segment_handicap_pda",
        "handicap_severity_cycle",
    ):
        values[name] = tuple(values[name])
    values["search_execution"] = parse_search_execution(values["search_execution"])
    return plan, ArenaConfig(**values)


def require_exclusive_device(device: torch.device, *, acknowledged: bool) -> None:
    if device.type == "cpu":
        return
    if not acknowledged:
        raise ValueError("reserve the device, then pass --exclusive-device")
    if device.type != "cuda" or device.index != 0:
        raise ValueError("GPU diagnostics require --device cuda:0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.startswith("GPU-") or "," in visible:
        raise ValueError("CUDA_VISIBLE_DEVICES must name exactly one full GPU UUID")
    query = subprocess.run(
        ["nvidia-smi", f"--id={visible}", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    if query.stdout.strip() != visible:
        raise ValueError("CUDA_VISIBLE_DEVICES must be the full reserved GPU UUID")
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={visible}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    owners = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if owners:
        raise ValueError(
            f"reserved GPU still has compute processes: {', '.join(owners)}"
        )


def _load_arm(output: Path, arm: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    path = output / f"{arm}.json"
    if not path.exists():
        return {}
    value = _read_json(path)
    if value.get("plan_sha256") != plan["plan_sha256"] or value.get("weights") != arm:
        raise ValueError("persisted diagnostic arm disagrees with the frozen plan")
    return value


def _summary(
    output: Path, plan: Mapping[str, Any], config: ArenaConfig
) -> dict[str, Any]:
    arms = {name: _load_arm(output, name, plan) for name in ARMS}
    summary: dict[str, Any] = {
        "result_kind": RESULT_KIND,
        "plan_sha256": plan["plan_sha256"],
        "terminal": all(value.get("terminal") is True for value in arms.values()),
        "interpretation": plan["interpretation"],
        "arms": {},
        "matched_comparison": {},
    }
    indexed: dict[str, dict[tuple[int, str, int], ArenaPair]] = {}
    for name, value in arms.items():
        pairs = [ArenaPair(**item) for item in value.get("pairs", [])]
        indexed[name] = {(pair.ring, pair.variant, pair.pair): pair for pair in pairs}
        if len(indexed[name]) != len(pairs):
            raise ValueError("diagnostic arm contains duplicate pairs")
        summary["arms"][name] = {
            "terminal": value.get("terminal", False),
            "completed_pairs": len(pairs),
            "per_cell": {
                cell: summarize_pairs(
                    selected,
                    confidence=config.confidence,
                    bootstrap_samples=config.bootstrap_samples,
                    seed=config.seed,
                )
                for cell in balanced_cells(config)
                if (
                    selected := [
                        pair
                        for pair in pairs
                        if f"r{pair.ring}/{category(GameVariant.parse(pair.variant))}"
                        == cell
                    ]
                )
            },
        }
    common = indexed["raw"].keys() & indexed["ema"].keys()
    for cell in balanced_cells(config):
        deltas = []
        for key in sorted(common):
            raw, ema = indexed["raw"][key], indexed["ema"][key]
            if (raw.opening_seed, raw.opening_action, raw.forced_opening) != (
                ema.opening_seed,
                ema.opening_action,
                ema.forced_opening,
            ):
                raise ValueError("diagnostic arms used different paired openings")
            if f"r{raw.ring}/{category(GameVariant.parse(raw.variant))}" == cell:
                deltas.append(raw.score_rate - ema.score_rate)
        summary["matched_comparison"][cell] = {
            "matched_pairs": len(deltas),
            "raw_minus_ema_score": sum(deltas) / len(deltas) if deltas else None,
            "interpretation": "Descriptive paired difference; positive favors raw. No significance claim.",
        }
    return summary


def run_session(
    *,
    output: Path,
    device: torch.device,
    exclusive_device: bool = False,
    session_seconds: float = 300,
    arm: str = "auto",
    stop_requested: Callable[[], bool] = lambda: False,
    native_module: object | None = None,
) -> dict[str, Any]:
    if not math.isfinite(session_seconds) or not 0 < session_seconds <= 3600:
        raise ValueError("session seconds must be in (0, 3600]")
    if arm not in (*ARMS, "auto"):
        raise ValueError("arm must be auto, raw, or ema")
    deadline = time.monotonic() + session_seconds
    plan, config = verify_plan(output)
    saved = {name: _load_arm(output, name, plan) for name in ARMS}
    pending = [name for name in ARMS if saved[name].get("terminal") is not True]
    if not pending or (arm != "auto" and arm not in pending):
        return _summary(output, plan, config)
    if arm == "auto":
        arm = min(pending, key=lambda name: saved[name].get("session_count", 0))
    require_exclusive_device(device, acknowledged=exclusive_device)
    candidate, metadata = load_candidate(
        output / "checkpoint.pt",
        device=device,
        weights=cast(Any, arm),
        precision=plan["precision"],
    )
    baseline, baseline_metadata = load_candidate(
        output / "champion.pt",
        device=device,
        weights="ema",
        precision=plan["precision"],
    )
    if (
        metadata["checkpoint_sha256"] != plan["checkpoints"]["checkpoint"]["sha256"]
        or baseline_metadata["checkpoint_sha256"]
        != plan["checkpoints"]["champion"]["sha256"]
    ):
        raise ValueError("loaded checkpoint differs from the diagnostic plan")
    state = saved[arm]
    session_count = state.get("session_count", 0) + 1
    identity = {
        "result_kind": RESULT_KIND,
        "plan_sha256": plan["plan_sha256"],
        "weights": arm,
        "candidate_metadata": metadata,
        "baseline_metadata": baseline_metadata,
        "session_count": session_count,
    }

    def persist(resume: dict[str, object]) -> None:
        atomic_json(
            output / f"{arm}.json",
            {
                **identity,
                "terminal": False,
                "resume_state": resume,
                "pairs": resume["pairs"],
                "games": resume["games"],
            },
        )

    runner = ArenaRunner(
        native_module=native_module or load_star_native(),
        candidate=candidate,
        baseline=baseline,
        config=config,
        stable_pair_seeds=True,
        baseline_metadata={"kind": "frozen_champion", **baseline_metadata},
    )
    result = runner.run(
        resume_state=state.get("resume_state"),
        checkpoint=persist,
        stop_requested=lambda: stop_requested() or time.monotonic() >= deadline,
    )
    # A short resumed wave may stop before revisiting previously finished
    # groups. Its durable state still contains every completed game/pair.
    resume = result.get("resume_state")
    if isinstance(resume, Mapping):
        pairs = [ArenaPair(**item) for item in resume["pairs"]]
        result.update(summarize_completed_arena_pairs(pairs, config))
        result["pairs"] = resume["pairs"]
        result["games"] = resume["games"]
    # Arena statistics are useful, but these matches can never be a promotion.
    result.pop("promotion", None)
    result.update(identity)
    result["terminal"] = not result["interrupted"]
    result["interpretation"] = plan["interpretation"]
    atomic_json(output / f"{arm}.json", result)
    summary = _summary(output, plan, config)
    atomic_json(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--champion-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--exclusive-device", action="store_true")
    parser.add_argument("--session-seconds", type=float, default=300)
    parser.add_argument("--arm", choices=("auto", *ARMS), default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default=None)
    parser.add_argument("--pairs-per-cell", type=int, default=None)
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        output = args.output_dir.expanduser().resolve()
        if args.source_run_root is not None:
            output = _isolated_output(args.source_run_root, output)
        elif (output / PLAN_NAME).is_file():
            _isolated_output(
                Path(_read_json(output / PLAN_NAME)["source_run_root"]), output
            )
        else:
            raise ValueError("a new diagnostic requires --source-run-root")
        with _exclusive_output(output):
            if not (output / PLAN_NAME).exists():
                if (
                    args.source_run_root is None
                    or args.checkpoint is None
                    or args.champion_checkpoint is None
                ):
                    raise ValueError("a new diagnostic requires both checkpoint paths")
                freeze_plan(
                    source_run_root=args.source_run_root,
                    checkpoint=args.checkpoint,
                    champion_checkpoint=args.champion_checkpoint,
                    output=output,
                    config=diagnostic_config(
                        pairs_per_cell=args.pairs_per_cell
                        if args.pairs_per_cell is not None
                        else 8,
                        simulations=args.simulations
                        if args.simulations is not None
                        else 256,
                        seed=args.seed if args.seed is not None else 17,
                    ),
                    precision=args.precision or "bf16",
                )
            elif any(
                value is not None
                for value in (
                    args.source_run_root,
                    args.checkpoint,
                    args.champion_checkpoint,
                    args.precision,
                    args.pairs_per_cell,
                    args.simulations,
                    args.seed,
                )
            ):
                raise ValueError("resume with --output-dir only; inputs are frozen")
            if args.plan_only:
                plan, _ = verify_plan(output)
                print(
                    json.dumps(
                        {"status": "planned", "plan_sha256": plan["plan_sha256"]}
                    )
                )
                return 0
            latch = SignalLatch()
            latch.install()
            summary = run_session(
                output=output,
                device=torch.device(args.device),
                exclusive_device=args.exclusive_device,
                session_seconds=args.session_seconds,
                arm=args.arm,
                stop_requested=latch.is_set,
            )
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "error", "error": str(error)}))
        return 2
    print(json.dumps({"output": str(output), **summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
