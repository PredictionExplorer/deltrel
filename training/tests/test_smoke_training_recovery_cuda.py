from pathlib import Path
import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import smoke_training_recovery_cuda as smoke

GPU_UUID = "GPU-12345678-1234-1234-1234-123456789abc"


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    profile, checkpoint = root / "profile.yaml", root / "checkpoint.pt"
    profile.write_text("frozen profile")
    checkpoint.write_bytes(b"frozen recovery")
    output = tmp_path / "evidence" / "smoke.json"
    output.parent.mkdir()
    config = SimpleNamespace(
        orchestration=SimpleNamespace(directories=SimpleNamespace(root=str(root)))
    )
    monkeypatch.setattr(smoke, "load_config", lambda path: config)
    monkeypatch.setattr(
        smoke, "require_stopped_run", lambda *args: {"checkpoint_step": 100}
    )
    monkeypatch.setattr(
        smoke,
        "load_model_manifest",
        lambda path: SimpleNamespace(checkpoint=root / "champion.pt"),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    queries = []

    def query(command, **kwargs):
        queries.append((command, kwargs))
        return GPU_UUID + "\n"

    monkeypatch.setattr(smoke.subprocess, "check_output", query)
    stages = []

    def stage(command, **kwargs):
        stages.append((command, kwargs))
        if command[2] == "scripts.smoke_adaptive_promotion_cuda":
            Path(command[command.index("--output") + 1]).write_text(
                json.dumps({"status": "passed"})
            )
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(smoke, "_run_owned", stage)
    return SimpleNamespace(
        root=root,
        profile=profile,
        checkpoint=checkpoint,
        output=output,
        stages=stages,
        queries=queries,
        stage=stage,
    )


def test_smoke_isolates_owned_gpu_stages_and_never_changes_training_files(prepared):
    before = {path.name: path.read_bytes() for path in prepared.root.iterdir()}
    report = smoke.run(prepared.profile, prepared.checkpoint, prepared.output)
    assert report["status"] == "passed"
    assert report["training_artifacts_written"] is False
    assert before == {path.name: path.read_bytes() for path in prepared.root.iterdir()}
    assert len(prepared.stages) == 4
    assert [options["timeout"] for _, options in prepared.stages] == [300, 90, 210, 210]
    for _, options in prepared.stages:
        assert options["env"]["CUDA_VISIBLE_DEVICES"] == GPU_UUID
        assert options["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        assert options["grace_seconds"] == 10
    assert "--id=2" in prepared.queries[0][0]
    plan = prepared.stages[1][0]
    assert "--plan-only" in plan
    assert plan[plan.index("--checkpoint") + 1] == str(prepared.checkpoint)
    assert plan[plan.index("--champion-checkpoint") + 1] == str(
        prepared.root / "champion.pt"
    )
    for (command, _), arm in zip(prepared.stages[2:], ("raw", "ema"), strict=True):
        assert command[command.index("--arm") + 1] == arm
        assert command[command.index("--session-seconds") + 1] == "120"
        assert command[command.index("--device") + 1] == "cuda:0"
        assert "--exclusive-device" in command


@pytest.mark.parametrize("failure", ["exit", "status"])
def test_failed_native_stage_prevents_diagnostic_and_success_report(
    prepared, monkeypatch, failure
):
    def stage(command, **kwargs):
        result = prepared.stage(command, **kwargs)
        if failure == "exit":
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="failed")
        prepared.output.with_name("smoke-native.json").write_text('{"status":"failed"}')
        return result

    monkeypatch.setattr(smoke, "_run_owned", stage)
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        smoke.run(prepared.profile, prepared.checkpoint, prepared.output)
    assert len(prepared.stages) == 1
    assert not prepared.output.exists()


def test_stopped_boundary_mismatch_is_fatal(prepared, monkeypatch):
    boundaries = iter(({"checkpoint_step": 100}, {"checkpoint_step": 101}))
    monkeypatch.setattr(smoke, "require_stopped_run", lambda *args: next(boundaries))
    with pytest.raises(RuntimeError, match="boundary changed"):
        smoke.run(prepared.profile, prepared.checkpoint, prepared.output)
    assert not prepared.output.exists()


@pytest.mark.parametrize("visible", ["", "0,1"])
def test_single_device_selection_required_before_gpu_work(
    prepared, monkeypatch, visible
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError, match="exactly one"):
        smoke.run(prepared.profile, prepared.checkpoint, prepared.output)
    assert prepared.stages == []


def test_training_output_rejected_before_any_child_process(prepared):
    with pytest.raises(ValueError, match="outside the production run"):
        smoke.run(prepared.profile, prepared.checkpoint, prepared.root / "smoke.json")
    assert prepared.stages == []


def test_owned_arm_timeout_prevents_later_arms_and_success_report(
    prepared, monkeypatch
):
    def stage(command, **kwargs):
        if "--arm" in command:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return prepared.stage(command, **kwargs)

    monkeypatch.setattr(smoke, "_run_owned", stage)
    with pytest.raises(subprocess.TimeoutExpired):
        smoke.run(prepared.profile, prepared.checkpoint, prepared.output)
    assert len(prepared.stages) == 2
    assert not prepared.output.exists()
