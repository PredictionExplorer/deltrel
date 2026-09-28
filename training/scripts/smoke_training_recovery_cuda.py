#!/usr/bin/env python3
"""Qualify the production GPU path and bounded frozen averaging diagnostics.

Every GPU stage has a separate process, so ownership and memory are released
before the next one. Diagnostic sessions save partial games; this smoke makes
no strength claim and never publishes training data, weights, or a champion.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.smoke_adaptive_promotion_cuda import require_stopped_run
from scripts.benchmark_actor_throughput import _controller_signals, _run_owned
from scripts.validate_cuda_graph_runtime import validate_output
from startrain.checkpoint import load_model_manifest
from startrain.config import load_config
from startrain.runtime import atomic_json


def run(profile: Path, checkpoint: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    config = load_config(profile)
    validate_output(output, config, profile, checkpoint)
    boundary = require_stopped_run(config, checkpoint)
    training = Path(__file__).resolve().parents[1]
    root = Path(config.orchestration.directories.root)
    native_report = output.with_name(output.stem + "-native.json")
    validate_output(native_report, config, profile, checkpoint)
    diagnostic = output.parent / "checkpoint-averaging-diagnostic"
    if diagnostic.exists() or diagnostic.is_symlink():
        raise FileExistsError("smoke diagnostic requires a new output directory")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise ValueError("smoke must own exactly one explicitly selected GPU")
    uuid = subprocess.check_output(
        ["nvidia-smi", f"--id={visible}", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
        timeout=10,
    ).strip()
    if not uuid.startswith("GPU-") or "\n" in uuid or "," in uuid:
        raise ValueError("smoke GPU selection did not resolve to one GPU UUID")
    environment = os.environ | {
        "CUDA_VISIBLE_DEVICES": uuid,
        "PYTHONPATH": str(training),
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    def stage(command: list[str], timeout: int) -> None:
        # Native smoke owns another GPU worker session. Give its signal handler
        # enough time to reap that worker before killing the parent session.
        child = _run_owned(command, env=environment, timeout=timeout, grace_seconds=10)
        if child.stdout:
            print(child.stdout, end="", flush=True)
        if child.stderr:
            print(child.stderr, end="", file=sys.stderr, flush=True)
        child.check_returncode()

    stage(
        [
            sys.executable,
            "-m",
            "scripts.smoke_adaptive_promotion_cuda",
            "--profile",
            str(profile),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(native_report),
        ],
        timeout=300,
    )
    native = json.loads(native_report.read_text())
    if native.get("status") != "passed":
        raise RuntimeError("native training recovery qualification failed")
    champion = load_model_manifest(root / "learner/champion.json")
    command = [
        sys.executable,
        "-m",
        "scripts.compare_checkpoint_averaging",
        "--output-dir",
        str(diagnostic),
    ]
    stage(
        [
            *command,
            "--source-run-root",
            str(root),
            "--checkpoint",
            str(checkpoint),
            "--champion-checkpoint",
            str(champion.checkpoint),
            "--plan-only",
            "--pairs-per-cell",
            "4",
            "--simulations",
            "256",
            "--precision",
            "bf16",
        ],
        timeout=90,
    )
    for arm in ("raw", "ema"):
        stage(
            [
                *command,
                "--arm",
                arm,
                "--device",
                "cuda:0",
                "--exclusive-device",
                "--session-seconds",
                "120",
            ],
            timeout=210,
        )
    after = require_stopped_run(config, checkpoint)
    if boundary != after:
        raise RuntimeError("training boundary changed during GPU qualification")
    report = {
        "status": "passed",
        "boundary": boundary,
        "native_report": str(native_report),
        "diagnostic_output": str(diagnostic),
        "diagnostic_scope": "two bounded resumable sessions; no strength conclusion",
        "training_artifacts_written": False,
    }
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with _controller_signals():
        print(json.dumps(run(args.profile, args.checkpoint, args.output)))


if __name__ == "__main__":
    main()
