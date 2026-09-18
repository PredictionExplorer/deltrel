"""CPU qualification of the bounded deployment smoke's data and state checks."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import sqlite3
import sys

import pytest
import torch

from scripts import smoke_auxiliary_predictions_cuda as smoke
from startrain.auxiliary_upgrade import AUXILIARY_LOSSES
from startrain.checkpoint import ExponentialMovingAverage, save_checkpoint, sha256_file
from startrain.config import load_config
from startrain.model import GraphResTNet
from startrain.native import score_results_from_native
from startrain.optim import build_optimizer
from startrain.replay import collate_replay_samples, write_replay_shard
from startrain.training import build_scheduler, train_step
from test_auxiliary_replay import samples_from, trajectory


def tiny_profile():
    source = load_config(Path(__file__).parents[1] / "configs/small.yaml")
    source = replace(source, model=replace(source.model, width=16, rrt_groups=1))
    target = replace(
        source,
        model=replace(source.model, auxiliary_predictions=True),
        loss=replace(source.loss, **{name: 0.1 for name in AUXILIARY_LOSSES}),
        train=replace(source.train, precision="bf16"),
    )
    return source, target


def source_checkpoint(tmp_path):
    source, target = tiny_profile()
    model = GraphResTNet(source.model)
    optimizer = build_optimizer(model, source.optimizer)
    scheduler = build_scheduler(optimizer, source.train.scheduler)
    ema = ExponentialMovingAverage(model, decay=source.train.resolved_ema_decay(1))
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.001)
    optimizer.step()
    scheduler.step()
    ema.update(model)
    path = save_checkpoint(
        tmp_path / "old.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=1234,
        epoch=9,
        config={"model": asdict(source.model), "game": asdict(source.game)},
        extra={"examples_consumed": 631808, "utd_credit": {"segment": 7}},
    )
    return target, path


def replay_fixture(root):
    replay = root / "replay"
    replay.mkdir(parents=True)
    shards = replay / "shards"
    shards.mkdir()
    (root / "run.json").write_text(
        json.dumps({"run_id": "manual", "generation_family": "manual"})
    )
    connection = sqlite3.connect(replay / "manifest.sqlite3")
    connection.execute(
        "CREATE TABLE shards(relative_path TEXT,checksum_sha256 TEXT,state TEXT,ring INTEGER,created_ns INTEGER,run_id TEXT,generation_family TEXT)"
    )
    for index, mode in enumerate(("classic", "double")):
        decisions, state = trajectory(
            mode=mode, pie=True, rings=10, actions=tuple(range(8))
        )
        remaining = list(range(8, state.node_count))
        state.apply_many([0] * len(remaining), remaining)
        final = score_results_from_native(state.score_data())[0]
        samples = [
            replace(
                s,
                opponent_reply=None,
                opponent_reply_ply=-1,
                second_stone=None,
                second_stone_ply=-1,
            )
            for s in samples_from(decisions, final)
        ]
        path = write_replay_shard(shards / f"{mode}.npz", samples)
        connection.execute(
            "INSERT INTO shards VALUES(?,?,'ready',10,?,'manual','manual')",
            (str(path.relative_to(replay)), sha256_file(path), index),
        )
    connection.commit()
    connection.close()
    return replay / "manifest.sqlite3"


@pytest.mark.native
def test_core_upgrade_parity_native_search_and_real_replay_backward(tmp_path):
    native = pytest.importorskip("star_native")
    config, checkpoint = source_checkpoint(tmp_path)
    original = sha256_file(checkpoint)
    model, reference, optimizer, scheduler, ema, clipper, proof = (
        smoke.load_training_state(config, checkpoint, "cpu")
    )
    assert proof["checkpoint_step"] == 1234 and proof["checkpoint_epoch"] == 9
    assert proof["ema_updates_preserved"] == 1
    parity = smoke.check_primary_parity(model, reference, native, "cpu", "bf16")
    assert len(parity) == 12 and all(
        row["primary_outputs_bitwise_equal"] for row in parity
    )
    checks = smoke.check_native_inference(model, native, "cpu", "fp32")
    assert {row["mode"] for row in checks} == {"classic", "double"}
    manifest = replay_fixture(tmp_path / "run")
    manifest_before = manifest.read_bytes()
    samples, evidence = smoke.load_smoke_replay(tmp_path / "run")
    assert len(samples) == 8 and len(evidence) == 2
    assert manifest.read_bytes() == manifest_before
    batch = collate_replay_samples(samples)
    assert all(
        getattr(batch.targets, name + "_mask").any() for name in AUXILIARY_LOSSES
    )
    model.train()
    result = train_step(
        model,
        batch,
        optimizer,
        loss_weights=config.loss,
        precision="bf16",
        scheduler=scheduler,
        ema=ema,
        gradient_clipper=clipper,
    )
    assert all(torch.isfinite(value).all() for value in result.loss_tensors.values())
    for name, parameter in model.named_parameters():
        if smoke.is_auxiliary_parameter(name):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.count_nonzero() > 0
    assert sha256_file(checkpoint) == original
    for artifact in evidence:
        assert sha256_file(Path(artifact["path"])) == artifact["sha256"]


@pytest.mark.native
def test_replay_checksum_and_missing_real_targets_fail_closed(tmp_path):
    manifest = replay_fixture(tmp_path)
    path = tmp_path / "replay/shards/double.npz"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        smoke.load_smoke_replay(tmp_path)
    connection = sqlite3.connect(manifest)
    connection.execute("DELETE FROM shards WHERE relative_path LIKE '%double%'")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="insufficient"):
        smoke.load_smoke_replay(tmp_path)


def test_refuses_active_run_before_cuda_or_worker_creation(tmp_path, monkeypatch):
    _, config = tiny_profile()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke",
            "--profile",
            str(tmp_path / "profile.yaml"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--output",
            str(tmp_path / "report.json"),
        ],
    )
    monkeypatch.setattr(smoke, "load_config", lambda path: config)
    monkeypatch.setattr(smoke, "validate_output", lambda *args: None)

    def reject(*args):
        raise ValueError("coordinator is running")

    monkeypatch.setattr(smoke, "require_stopped_run", reject)
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("CUDA touched before stopped-run proof"),
    )
    monkeypatch.setattr(
        smoke.tempfile,
        "TemporaryDirectory",
        lambda **kwargs: pytest.fail("worker setup before stopped-run proof"),
    )
    with pytest.raises(ValueError, match="running"):
        smoke.main()
    assert not (tmp_path / "report.json").exists()


@pytest.mark.native
def test_smoke_future_reconstruction_does_not_bridge_missing_plies():
    decisions, _ = trajectory()
    rows = samples_from(decisions)
    assert smoke.reconstruct_observed_future_targets([rows[0], rows[2]]) == []
