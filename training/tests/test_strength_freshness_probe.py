from dataclasses import fields, replace
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest
import torch
import yaml

from test_replay import sample_for
from deltreltrain import replay
from deltreltrain.checkpoint import ExponentialMovingAverage, save_checkpoint
from deltreltrain.config import load_config
from deltreltrain.contracts import (
    FEATURE_SCHEMA_HASH,
    RULES_HASH_WIRE,
    SEARCH_ALGORITHM_ID,
)
from deltreltrain.model import GraphResTNet
from deltreltrain.optim import build_optimizer
from deltreltrain.training import build_scheduler, train_step
from scripts import strength_freshness_probe as probe


def pin(path):
    return {
        "path": str(path.resolve()),
        "sha256": probe.digest(path),
        "bytes": path.stat().st_size,
    }


def write(path, value):
    path.write_bytes(probe.encoded(value))
    return pin(path)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "live"
    root.mkdir()
    (root / "sentinel").write_text("active source must remain unchanged")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    config = load_config(Path(__file__).parents[1] / "configs/h100-8gpu-pie-even.yaml")
    config = replace(
        config,
        model=replace(
            config.model, width=8, rrt_groups=1, attention_heads=2, kv_heads=1
        ),
        train=replace(
            config.train, per_rank_batch_size=2, compile=False, precision="fp32"
        ),
        learner=replace(config.learner, device="cpu"),
        orchestration=replace(
            config.orchestration,
            run_id="probe-run",
            directories=replace(config.orchestration.directories, root=str(root)),
        ),
    )
    profile = inputs / "profile.yaml"
    profile.write_text(yaml.safe_dump(json.loads(json.dumps(config.as_dict()))))
    samples = [sample_for(), sample_for()]
    shard = replay.write_replay_shard(inputs / "shard.npz", samples)
    shard_pin = {**pin(shard), "sample_count": 2}
    batch = replay.collate_replay_samples(samples, prefer_native=False)
    assert batch.variant_labels is not None

    def tensors(value):
        return {f.name: getattr(value, f.name) for f in fields(value)}

    artifact = inputs / "batch.pt"
    torch.save(
        {
            "schema_version": 1,
            "inputs": tensors(batch.inputs),
            "targets": tensors(batch.targets),
            "variant_labels": list(batch.variant_labels),
            "feature_path": batch.feature_path,
        },
        artifact,
    )
    manifest = inputs / "batch.json"
    write(
        manifest,
        {
            "schema_version": 1,
            "format": probe.BATCH_FORMAT,
            "rows": 2,
            "rings": 4,
            "rules_hash": RULES_HASH_WIRE,
            "feature_schema_hash": FEATURE_SCHEMA_HASH,
            "artifact": pin(artifact),
            "provenance": {
                "runtime_source_commit": "1" * 40,
                "collator_module_sha256": probe.digest(Path(replay.__file__)),
                "shards": [shard_pin],
                "ordered_rows": [
                    {"shard_sha256": shard_pin["sha256"], "row_index": i}
                    for i in range(2)
                ],
            },
        },
    )
    model = GraphResTNet(config.model)
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.train.scheduler)
    ema = ExponentialMovingAverage(model, decay=config.train.resolved_ema_decay(1))
    train_step(
        model, batch, optimizer, scheduler=scheduler, ema=ema, loss_weights=config.loss
    )
    checkpoint = save_checkpoint(
        inputs / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=10,
        epoch=2,
        config=config.as_dict(),
        extra={
            "run_id": "probe-run",
            "generation_family": "family",
            "examples_consumed": 20,
            "global_batch_size": 2,
        },
    )
    package = importlib.import_module(type(config).__module__.split(".")[0])
    assert package.__file__ is not None
    package_path = Path(package.__file__).resolve()
    runtime_root = package_path.parent.parent
    source_marker = inputs / "SOURCE_COMMIT"
    source_marker.write_text("1" * 40 + "\n")
    source_manifest = inputs / "SOURCE_SHA256SUMS"
    source_manifest.write_text(
        "".join(
            f"{probe.digest(p)}  {p.relative_to(runtime_root.parent)}\n"
            for p in sorted(package_path.parent.rglob("*.py"))
        )
    )
    native_name = package.__name__.removesuffix("train") + "_native"
    native = pytest.importorskip(native_name)
    binaries = []
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if (
            name.startswith(native_name + ".")
            and isinstance(filename, str)
            and Path(filename).suffix == ".so"
        ):
            binaries.append(pin(Path(filename)))
    python = Path(sys.executable).absolute()
    pyvenv = Path(sys.prefix) / "pyvenv.cfg"
    runtime = {
        "source_commit": "1" * 40,
        "source_commit_file": pin(source_marker),
        "source_manifest": pin(source_manifest),
        "python": {
            **pin(python.resolve()),
            "path": str(python),
            "resolved_path": str(python.resolve()),
        },
        "pyvenv": pin(pyvenv),
        "training_module": pin(package_path),
        "native_wrapper": pin(Path(native.__file__)),
        "native_binaries": binaries,
        "rules_hash": RULES_HASH_WIRE,
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "search_algorithm": SEARCH_ALGORITHM_ID,
        "torch_version": importlib.metadata.version("torch"),
        "torch_cuda_build": torch.version.cuda,
    }
    now = time.monotonic()
    challenge = {
        "schema_version": 1,
        "format": probe.CHALLENGE_FORMAT,
        "attempt_id": "cpu-fixture",
        "boot_id": "test-boot",
        "nonce": "a" * 64,
        "probe_unit": "probe-fixture.service",
        "issued_monotonic": now,
        "deadline_monotonic": now + 60,
        "mode": "cpu_validation",
        "runtime_root": str(runtime_root),
        "run_root": str(root),
        "output_dir": str(tmp_path / "output"),
        "profile": pin(profile),
        "checkpoint": pin(checkpoint),
        "batch_manifest": pin(manifest),
        "recovery_pointer": None,
        "admission": None,
        "runtime": runtime,
        "run_identity": {
            "run_id": "probe-run",
            "generation_family": "family",
            "created_ns": 1,
        },
        "expected_step": 10,
        "expected_examples_consumed": 20,
        "probe_source_sha256": probe.digest(Path(probe.__file__)),
    }
    path = inputs / "challenge.json"
    write(path, challenge)
    monkeypatch.setattr(probe, "boot_id", lambda: "test-boot")
    monkeypatch.setenv("INVOCATION_ID", "b" * 32)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    return SimpleNamespace(
        path=path,
        challenge=challenge,
        root=root,
        inputs=inputs,
        config=config,
        batch=batch,
        checkpoint=checkpoint,
        output=Path(challenge["output_dir"]),
    )


def execute(item):
    write(item.path, item.challenge)
    return probe.run_probe(item.path, probe.digest(item.path))


def test_real_cpu_restore_collation_keeps_all_live_inputs_and_cannot_claim_cuda(
    prepared, monkeypatch
):
    p = prepared
    before = {
        f: f.read_bytes()
        for folder in (p.root, p.inputs)
        for f in folder.rglob("*")
        if f.is_file()
    }
    # Availability may be reported, but CPU validation must never seed/init/use it.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    for name in (
        "init",
        "manual_seed_all",
        "synchronize",
        "get_device_properties",
    ):
        monkeypatch.setattr(
            torch.cuda, name, lambda *a, **k: pytest.fail("CPU validation touched CUDA")
        )
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.mps,
        "manual_seed",
        lambda *a, **k: pytest.fail("CPU validation touched MPS"),
    )
    result = probe.run_probe(p.path, probe.digest(p.path))
    assert result["status"] == "cpu_validated", result
    assert result["checks"] == {
        "cuda_available": False,
        "learner_resume_step": True,
        "ema_preserved": True,
        "native_cuda_search": False,
    }
    assert (
        "all_workers_released" not in result["checks"] and result["admission"] is None
    )
    assert result["work"] == {
        "optimizer_steps": 0,
        "native_searches": 0,
        "native_neural_calls": 0,
    }
    assert all(f.read_bytes() == data for f, data in before.items())
    rows = [
        json.loads(line)
        for line in (p.output / "observations.jsonl").read_text().splitlines()
    ]
    restored = next(row["observed"] for row in rows if row["stage"] == "restored-state")
    assert (
        restored["step"] == 10
        and restored["epoch"] == 2
        and restored["examples_consumed"] == 20
    )
    assert all(
        row["expected"] == row["actual"] for row in restored["comparisons"].values()
    )
    assert (p.output / "scratch/replay/manifest.sqlite3").is_file()
    assert not any(p.output.rglob("*.pt"))


def test_tiny_production_step_uses_restored_optimizer_scheduler_and_ema(
    prepared, monkeypatch
):
    p = prepared
    p.output.mkdir()
    observer = probe.Observations(p.output, time.monotonic() + 30)
    try:
        state, payload = probe.restore(
            p.config,
            p.checkpoint,
            pin(p.checkpoint),
            p.challenge["run_identity"],
            "cpu",
            observer,
        )
        assert payload["optimizer"]["state"]
        monkeypatch.setattr(
            probe, "maybe_compile_model", lambda *a, **k: pytest.fail("compiled twice")
        )
        result = probe.training_observations(state, p.config, p.batch, observer)
        assert (
            result["model_changed"] and result["ema_after"] == result["ema_before"] + 1
        )
        assert (
            result["restored_step"] == 10
            and result["diagnostic_optimizer_steps"] == 1
            and result["live_updates"] == 0
        )
    finally:
        observer.close()


def test_buffer_only_change_cannot_prove_parameter_update(prepared, monkeypatch):
    p = prepared
    p.output.mkdir()
    observer = probe.Observations(p.output, time.monotonic() + 30)
    try:
        state, _ = probe.restore(
            p.config,
            p.checkpoint,
            pin(p.checkpoint),
            p.challenge["run_identity"],
            "cpu",
            observer,
        )
        buffer = torch.zeros(1)
        state.model.register_buffer("probe_buffer", buffer)
        state.ema.shadow["probe_buffer"] = torch.zeros(1)
        parameters = {
            name: value.detach().clone()
            for name, value in state.model.named_parameters()
        }
        original_step = probe.train_step

        def buffer_only_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            with torch.no_grad():
                for name, value in state.model.named_parameters():
                    value.copy_(parameters[name])
                buffer.add_(1)
            return result

        monkeypatch.setattr(probe, "train_step", buffer_only_step)
        with pytest.raises(ValueError, match="training/EMA step"):
            probe.training_observations(state, p.config, p.batch, observer)
        records = [json.loads(line) for line in observer.path.read_text().splitlines()]
        observed = records[-1]["observed"]
        assert not observed["model_changed"]
        assert observed["parameter_hash_before"] == observed["parameter_hash_after"]
        assert buffer.item() == 1
    finally:
        observer.close()


@pytest.mark.native
def test_real_native_cpu_loop_covers_all24_semantic_cases_without_cuda_claim(prepared):
    p = prepared
    p.output.mkdir()
    observer = probe.Observations(p.output, time.monotonic() + 45)
    try:
        result = probe.native_observations(
            p.config,
            p.checkpoint,
            pin(p.checkpoint),
            p.challenge["run_identity"],
            "cpu",
            observer,
        )
        assert result["searches"] == 24 and result["metrics"]["neural_calls"] > 0
        assert result["device"] == "cpu" and result["metrics"]["graph_captures"] == 0
        assert not (p.output / "result.json").exists()
    finally:
        observer.close()


@pytest.mark.parametrize(
    "mutation",
    [
        "boot",
        "expired",
        "long-budget",
        "nonce",
        "source",
        "live-output",
        "runtime-output",
        "release-output",
        "relative-run",
        "symlink-run",
        "bad-invocation",
        "cpu-admission",
    ],
)
def test_challenge_refusal_is_before_output_or_gpu(prepared, monkeypatch, mutation):
    p = prepared
    if mutation == "boot":
        p.challenge["boot_id"] = "another-boot"
    elif mutation == "expired":
        p.challenge["deadline_monotonic"] = time.monotonic() - 1
    elif mutation == "long-budget":
        p.challenge["deadline_monotonic"] = p.challenge["issued_monotonic"] + 601
    elif mutation == "nonce":
        p.challenge["nonce"] = "short"
    elif mutation == "source":
        p.challenge["probe_source_sha256"] = "0" * 64
    elif mutation == "live-output":
        p.challenge["output_dir"] = str(p.root / "probe")
    elif mutation == "runtime-output":
        p.challenge["output_dir"] = str(Path(p.challenge["runtime_root"]) / "probe")
    elif mutation == "release-output":
        p.challenge["output_dir"] = str(
            Path(p.challenge["runtime_root"]).parent / "probe"
        )
    elif mutation == "relative-run":
        monkeypatch.chdir(p.root.parent)
        p.challenge["run_root"] = "live"
        p.challenge["output_dir"] = str(p.root / "probe")
    elif mutation == "symlink-run":
        alias = p.root.parent / "live-alias"
        alias.symlink_to(p.root, target_is_directory=True)
        p.challenge["run_root"] = str(alias)
        p.challenge["output_dir"] = str(p.root / "probe")
    elif mutation == "bad-invocation":
        monkeypatch.setenv("INVOCATION_ID", "")
    elif mutation == "cpu-admission":
        p.challenge["admission"] = {}
    with pytest.raises(ValueError):
        execute(p)
    assert not p.output.exists()
    assert not Path(p.challenge["output_dir"]).exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "runtime-commit",
        "python-pin",
        "native-pin",
        "torch-version",
        "torch-cuda-build",
        "checkpoint-pin",
        "step",
        "examples",
        "batch-rows",
        "batch-tensor",
        "replay-row",
        "missing-required-target",
    ],
)
def test_durable_raw_failure_for_tamper_without_live_writes(prepared, mutation):
    p = prepared
    before = (p.root / "sentinel").read_bytes()
    if mutation == "runtime-commit":
        p.challenge["runtime"]["source_commit"] = "2" * 40
    elif mutation == "python-pin":
        p.challenge["runtime"]["python"]["sha256"] = "0" * 64
    elif mutation == "native-pin":
        p.challenge["runtime"]["native_binaries"][0]["sha256"] = "0" * 64
    elif mutation == "torch-version":
        p.challenge["runtime"]["torch_version"] = "0.0.0-unqualified"
    elif mutation == "torch-cuda-build":
        p.challenge["runtime"]["torch_cuda_build"] = "0.0-unqualified"
    elif mutation == "checkpoint-pin":
        p.challenge["checkpoint"]["sha256"] = "0" * 64
    elif mutation == "step":
        p.challenge["expected_step"] += 1
    elif mutation == "examples":
        p.challenge["expected_examples_consumed"] += 1
    else:
        manifest = json.loads((p.inputs / "batch.json").read_text())
        if mutation == "batch-rows":
            manifest["rows"] += 1
        elif mutation == "replay-row":
            manifest["provenance"]["ordered_rows"][0]["row_index"] = 999
        else:
            payload = torch.load(p.inputs / "batch.pt", weights_only=True)
            if mutation == "batch-tensor":
                payload["inputs"]["node_features"][0, 0, 0] += 0.5
            else:
                payload["targets"]["policy"] = None
            torch.save(payload, p.inputs / "batch.pt")
            manifest["artifact"] = pin(p.inputs / "batch.pt")
        p.challenge["batch_manifest"] = write(p.inputs / "batch.json", manifest)
    result = execute(p)
    assert result["status"] == "failed" and not result["checks"]["cuda_available"]
    assert (p.output / "observations.jsonl").stat().st_size > 0
    assert (p.root / "sentinel").read_bytes() == before


def test_deadline_after_observation_remains_failure_and_preserves_output(
    prepared, monkeypatch
):
    p = prepared
    original = probe.verify_collation

    def expire(*args):
        original(*args)
        monkeypatch.setattr(
            probe.time, "monotonic", lambda: p.challenge["deadline_monotonic"] + 1
        )

    monkeypatch.setattr(probe, "verify_collation", expire)
    result = execute(p)
    assert result["status"] == "failed" and result["error"]["type"] == "TimeoutError"
    assert result["completed_monotonic"] > p.challenge["deadline_monotonic"]
    assert not result["checks"]["cuda_available"]


def test_output_no_clobber_and_bad_challenge_sha(prepared):
    p = prepared
    with pytest.raises(ValueError, match="explicit hash"):
        probe.run_probe(p.path, "0" * 64)
    assert execute(p)["status"] == "cpu_validated"
    old = (p.output / "result.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        execute(p)
    assert (p.output / "result.json").read_bytes() == old


def test_backend_math_flags_restore_without_accelerator_work(monkeypatch):
    for name in ("init", "synchronize", "device_count", "get_device_properties"):
        monkeypatch.setattr(
            torch.cuda,
            name,
            lambda *a, **k: pytest.fail("math flag access initialized CUDA"),
        )
    original = probe.backend_math_flags()
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = not original["cudnn_allow_tf32"]
        assert probe.backend_math_flags() != original
        probe.restore_math_flags(original)
        assert probe.backend_math_flags() == original
    finally:
        probe.restore_math_flags(original)


def test_adopted_continuation_metadata_and_drift_are_observed(prepared):
    p = prepared
    payload = torch.load(p.checkpoint, weights_only=True)
    payload["extra"]["training_segment"] = {"baseline_examples_consumed": 4}
    payload["extra"]["utd_segment"] = {
        "schema_version": 1,
        "run_id": "probe-run",
        "generation_family": "family",
        "target_updates_per_new_sample": p.config.learner.target_updates_per_new_sample,
        "baseline_examples_consumed": 4,
        "baseline_committed_replay_samples": 8,
    }
    torch.save(payload, p.checkpoint)
    p.output.mkdir()
    observations = probe.Observations(p.output, time.monotonic() + 30)
    try:
        state, saved = probe.restore(
            p.config,
            p.checkpoint,
            pin(p.checkpoint),
            p.challenge["run_identity"],
            "cpu",
            observations,
        )
        assert state._segment_baseline_examples == 4
        assert state._resume_utd_segment_state is not None
        assert state._resume_utd_segment_state.baseline_committed_replay_samples == 8
        state._segment_baseline_examples += 1
        with pytest.raises(ValueError, match="continuation metadata"):
            probe.continuation_observations(state, saved, observations)
        records = [
            json.loads(line) for line in observations.path.read_text().splitlines()
        ]
        assert records[-1]["stage"] == "restored-continuation"
        assert records[-1]["observed"]["actual"]["segment_baseline_examples"] == 5
    finally:
        observations.close()
