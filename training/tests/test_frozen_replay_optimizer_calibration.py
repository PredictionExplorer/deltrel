from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts import run_frozen_replay_optimizer_calibration as calibration
from deltreltrain.checkpoint import ExponentialMovingAverage, write_model_pointer
from deltreltrain.config import load_config
from deltreltrain.contracts import (
    FEATURE_SCHEMA_HASH,
    RULES_HASH,
    RULES_HASH_WIRE,
    TARGET_OUTCOME,
)
from deltreltrain.features import DoubleDeltrelPosition
from deltreltrain.learner import ImmutableModelPublisher
from deltreltrain.model import GraphResTNet
from deltreltrain.optim import build_optimizer
from deltreltrain.replay import ReplaySample, collate_replay_samples, write_replay_shard
from deltreltrain.runtime import RunIdentity
from deltreltrain.topology import get_topology
from deltreltrain.training import build_scheduler, isolated_compile_cache

TRAINING_ROOT = Path(__file__).parents[1]
RUN_ID = "calibration-run"
FAMILY = "calibration-family"


def _tiny_ring10_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load(
        (TRAINING_ROOT / "configs" / "h100-8gpu-ring10-only.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["model"].update(
        {
            "width": 8,
            "rrt_groups": 1,
            "attention_heads": 1,
            "kv_heads": 1,
            "ff_multiplier": 1.0,
            "local_blocks_per_group": 1,
        }
    )
    raw["optimizer"].update(
        {
            "adamw_lr": 0.001,
            "muon_lr": 0.01,
            "min_muon_elements": 1,
            "fallback_to_adamw": False,
        }
    )
    raw["train"].update(
        {
            "per_rank_batch_size": 1,
            "precision": "fp32",
            "compile": False,
            "ema_decay": 0.9,
            "gradient_clip_norm": 1.0,
            "scheduler": {
                "warmup_steps": 0,
                "total_steps": 10,
                "min_lr_ratio": 0.1,
            },
        }
    )
    raw["data"].update(
        {
            "d5_augmentation": False,
            "workers": 0,
            "pin_memory": False,
        }
    )
    raw["learner"]["device"] = "cpu"
    path = tmp_path / "control.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _sample(index: int) -> ReplaySample:
    topology = get_topology(10)
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    stones[index % 5] = index % 2
    position = DoubleDeltrelPosition(
        rings=10,
        stones=stones,
        to_move=(index + 1) % 2,
        moves_left=1,
        opening=False,
        terminal=False,
    )
    policy = (stones.numpy() == -1).astype(np.float32)
    policy /= policy.sum()
    return ReplaySample.from_position(
        position,
        policy=policy,
        final_score=None,
        search_provenance="frozen-calibration",
        policy_provenance="frozen-calibration",
        run_id=RUN_ID,
        generation_family=FAMILY,
        actor_id="calibration-actor",
        generation=0,
        game_id=f"calibration-game-{index}",
        model_identity="source-model",
    )


def _write_replay(replay_root: Path, *, samples: int = 8) -> tuple[Path, Path]:
    shard = write_replay_shard(
        replay_root / "shards" / "ring10.npz",
        [_sample(index) for index in range(samples)],
    )
    manifest = replay_root / "manifest.sqlite3"
    with sqlite3.connect(manifest) as connection:
        connection.executescript(
            """
            CREATE TABLE store_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE shards (
                id INTEGER PRIMARY KEY,
                relative_path TEXT NOT NULL,
                sample_count INTEGER NOT NULL,
                ring INTEGER NOT NULL,
                model_step INTEGER NOT NULL,
                model_identity TEXT NOT NULL,
                run_id TEXT NOT NULL,
                generation_family TEXT NOT NULL,
                state TEXT NOT NULL,
                rules_hash TEXT NOT NULL,
                feature_schema_hash TEXT NOT NULL,
                checksum_sha256 TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO store_metadata(key, value) VALUES (?, ?)",
            (
                ("manifest_schema_version", str(calibration.MANIFEST_SCHEMA_VERSION)),
                ("rules_hash", RULES_HASH_WIRE),
                ("feature_schema_hash", f"{FEATURE_SCHEMA_HASH:016x}"),
            ),
        )
        connection.execute(
            """
            INSERT INTO shards(
                id, relative_path, sample_count, ring, model_step,
                model_identity, run_id, generation_family, state,
                rules_hash, feature_schema_hash, checksum_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                7,
                str(shard.relative_to(replay_root)),
                samples,
                10,
                1,
                "source-model",
                RUN_ID,
                FAMILY,
                "ready",
                f"{RULES_HASH:016x}",
                f"{FEATURE_SCHEMA_HASH:016x}",
                calibration.sha256_file(shard),
            ),
        )
    return manifest, shard


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    config_path = _tiny_ring10_config(tmp_path)
    config = load_config(config_path)
    model = GraphResTNet(config.model)
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.train.scheduler)
    ema = ExponentialMovingAverage(model, decay=config.train.ema_decay)
    publisher = ImmutableModelPublisher(
        tmp_path / "source" / "learner",
        RunIdentity(tmp_path / "source" / "run.json", RUN_ID, FAMILY, 1),
    )
    candidate = publisher.publish(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=1,
        epoch=0,
        config=config.as_dict(),
    )
    champion = publisher.root / "champion.json"
    write_model_pointer(
        champion,
        candidate,
        role="champion",
        promotion_result="bootstrap",
    )
    replay_root = tmp_path / "source" / "replay"
    manifest, shard = _write_replay(replay_root)
    return config_path, champion, replay_root, manifest, shard


def _snapshot(paths: tuple[Path, ...]) -> dict[Path, tuple[bytes, int]]:
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}


def _settings(
    tmp_path: Path,
    config: Path,
    champion: Path,
    replay_root: Path,
    *,
    dry_run: bool,
    stop_after_steps: int | None = None,
) -> calibration.CalibrationSettings:
    return calibration.CalibrationSettings(
        config=config,
        champion=champion,
        replay_root=replay_root,
        replay_cutoff=7,
        output_dir=tmp_path / "calibration-output",
        arm=calibration.CONTROL_ARM,
        steps=2,
        batch_size=1,
        evaluation_batch_size=1,
        max_samples=8,
        holdout_fraction=0.25,
        seed=23,
        device="cpu",
        budget_h100_hours=0.01,
        checkpoint_interval=1,
        stop_after_steps=stop_after_steps,
        dry_run=dry_run,
    )


def test_frozen_replay_dry_run_is_hash_pinned_and_read_only(tmp_path: Path) -> None:
    config, champion, replay_root, manifest, shard = _fixture(tmp_path)
    before = _snapshot((champion, manifest, shard))
    settings = _settings(
        tmp_path,
        config,
        champion,
        replay_root,
        dry_run=True,
    )

    with calibration.open_replay_read_only(replay_root) as connection:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM shards")

    result = calibration.run_calibration(settings)
    repeated = calibration.run_calibration(settings)

    assert result["status"] == "dry_run"
    assert result["replay"]["cutoff"] == 7
    assert len(result["replay"]["cutoff_sha256"]) == 64
    assert result["partition"]["disjoint"] is True
    assert repeated["partition"] == result["partition"]
    assert result["training"]["budget_h100_hours"] == 0.01
    assert not (tmp_path / "calibration-output").exists()
    assert _snapshot((champion, manifest, shard)) == before
    with pytest.raises(ValueError, match=r"\(0, 2\] H100-hours"):
        calibration.run_calibration(replace(settings, budget_h100_hours=2.01))


def test_calibration_splits_whole_games_and_deduplicates_revisions(tmp_path):
    config, champion, replay_root, manifest, shard = _fixture(tmp_path)
    samples = [
        replace(_sample(index), game_id=f"shared-game-{index // 4}", ply=index % 4)
        for index in range(8)
    ]
    # A later ready publication enriches a logical position. It must not receive
    # a second sample identity or leak into the opposite partition.
    samples.append(replace(samples[0], policy_weight=0.5))
    write_replay_shard(shard, samples)
    with sqlite3.connect(manifest) as connection:
        connection.execute(
            "UPDATE shards SET sample_count=?, checksum_sha256=?",
            (len(samples), calibration.sha256_file(shard)),
        )
    settings = _settings(tmp_path, config, champion, replay_root, dry_run=True)
    pin = calibration._champion_pin(champion, load_config(config))
    replay = calibration.freeze_replay(settings, pin, decode=True)
    train_games = {row.game_identity for row in replay.train}
    holdout_games = {row.game_identity for row in replay.holdout}
    assert train_games.isdisjoint(holdout_games)
    assert len(train_games) == len(holdout_games) == 1
    assert len(replay.train) + len(replay.holdout) == 8
    selected = (*replay.train, *replay.holdout)
    assert len({row.stable_id for row in selected}) == 8
    assert any(row.sample_index == 8 for row in selected)
    assert all(row.sample_index != 0 for row in selected)
    assert replay.partition_dict()["game_disjoint"] is True
    assert calibration.freeze_replay(settings, pin, decode=False).partition_sha256 == (
        replay.partition_sha256
    )

    result = calibration.run_calibration(replace(settings, dry_run=False))
    assert result["heldout"]["games"] == 1
    assert result["heldout"]["batches"] == 4
    assert len(result["heldout"]["observations"]) == 1
    assert result["heldout"]["observation_unit"] == "immutable-game"


def test_calibration_refuses_a_single_game_holdout(tmp_path):
    config, champion, replay_root, manifest, shard = _fixture(tmp_path)
    samples = [
        replace(_sample(index), game_id="one-game", ply=index) for index in range(8)
    ]
    write_replay_shard(shard, samples)
    with sqlite3.connect(manifest) as connection:
        connection.execute(
            "UPDATE shards SET checksum_sha256=?", (calibration.sha256_file(shard),)
        )
    settings = _settings(tmp_path, config, champion, replay_root, dry_run=True)
    pin = calibration._champion_pin(champion, load_config(config))
    with pytest.raises(ValueError, match="at least two games"):
        calibration.freeze_replay(settings, pin, decode=False)


def test_logical_capacity_looks_past_repeated_newest_ready_revisions(tmp_path):
    config, champion, replay_root, manifest, shard = _fixture(tmp_path)
    samples = [
        replace(_sample(index), game_id=f"game-{index // 4}", ply=index % 4)
        for index in range(8)
    ]
    write_replay_shard(shard, samples)
    with sqlite3.connect(manifest) as connection:
        connection.execute(
            "UPDATE shards SET checksum_sha256=?", (calibration.sha256_file(shard),)
        )
        for revision in (8, 9, 10):
            path = write_replay_shard(
                replay_root / "shards" / f"revision-{revision}.npz",
                [replace(row, weight=revision / 10) for row in samples[:4]],
            )
            connection.execute(
                """INSERT INTO shards SELECT ?, ?, 4, ring, model_step,
                   model_identity, run_id, generation_family, state,
                   rules_hash, feature_schema_hash, ? FROM shards WHERE id=7""",
                (
                    revision,
                    str(path.relative_to(replay_root)),
                    calibration.sha256_file(path),
                ),
            )
    settings = replace(
        _settings(tmp_path, config, champion, replay_root, dry_run=True),
        replay_cutoff=9,
    )
    pin = calibration._champion_pin(champion, load_config(config))
    replay = calibration.freeze_replay(settings, pin, decode=True)
    references = (*replay.train, *replay.holdout)
    assert len(references) == 8
    assert {row.shard.shard_id for row in references} == {7, 9}
    assert len({row.game_identity for row in references}) == 2
    assert all(
        replay.decoded[row.shard.shard_id].sample(row.sample_index).weight
        == pytest.approx(0.9)
        for row in references
        if row.shard.shard_id == 9
    )


@pytest.mark.parametrize("mismatch", ["model", "variant"])
def test_calibration_rejects_shard_or_immutable_game_provenance_drift(
    tmp_path, mismatch
):
    config, champion, replay_root, manifest, shard = _fixture(tmp_path)
    samples = [_sample(index) for index in range(8)]
    if mismatch == "model":
        samples[0] = replace(samples[0], model_identity="other-model")
    write_replay_shard(shard, samples)
    with sqlite3.connect(manifest) as connection:
        connection.execute(
            "UPDATE shards SET checksum_sha256=?", (calibration.sha256_file(shard),)
        )
        if mismatch == "variant":
            path = write_replay_shard(
                replay_root / "shards" / "changed-context.npz",
                [replace(samples[0], mode="classic")],
            )
            connection.execute(
                """INSERT INTO shards SELECT 8, ?, 1, ring, model_step,
                   model_identity, run_id, generation_family, state,
                   rules_hash, feature_schema_hash, ? FROM shards WHERE id=7""",
                (str(path.relative_to(replay_root)), calibration.sha256_file(path)),
            )
    settings = _settings(tmp_path, config, champion, replay_root, dry_run=True)
    if mismatch == "variant":
        settings = replace(settings, replay_cutoff=8)
    pin = calibration._champion_pin(champion, load_config(config))
    with pytest.raises(ValueError, match="identity differs|context changed"):
        calibration.freeze_replay(settings, pin, decode=False)


def test_game_loss_aggregation_preserves_masked_target_weight_mass_across_chunks(
    tmp_path,
):
    config = load_config(_tiny_ring10_config(tmp_path))
    model = GraphResTNet(config.model).eval()
    with torch.no_grad():
        model.outcome_head.weight.zero_()
        model.outcome_head.bias.copy_(torch.tensor([0.0, 3.0]))
    samples = []
    for index, (weight, policy_weight) in enumerate(
        ((1.0, 0.1), (9.0, 2.0), (3.0, 1.0), (2.0, 0.2))
    ):
        sample = replace(_sample(index), weight=weight, policy_weight=policy_weight)
        if index != 2:
            outcome = 0 if index == 0 else 1
            winner = sample.to_move if outcome else 1 - sample.to_move
            sample = replace(
                sample,
                target_mask=sample.target_mask | TARGET_OUTCOME,
                outcome=outcome,
                final_scores=np.asarray([int(winner == player) for player in (0, 1)]),
                final_capes=np.zeros(2, dtype=np.int8),
            )
        samples.append(sample)

    def totals(rows):
        return calibration._component_loss_totals(
            model,
            collate_replay_samples(rows),
            config,
            device=torch.device("cpu"),
            precision="fp32",
        )

    whole = totals(samples)
    assert whole["outcome"][1] == 12.0
    for chunk_size in (1, 2, 3):
        combined = {name: [0.0, 0.0] for name in whole}
        for start in range(0, len(samples), chunk_size):
            for name, values in totals(samples[start : start + chunk_size]).items():
                for index, value in enumerate(values):
                    combined[name][index] += value
        assert calibration._components_from_totals(combined, config) == pytest.approx(
            calibration._components_from_totals(whole, config), rel=1e-5, abs=1e-6
        )


def test_v1_calibration_state_cannot_resume_under_game_disjoint_contract(tmp_path):
    config, champion, replay_root, _, _ = _fixture(tmp_path)
    settings = _settings(
        tmp_path, config, champion, replay_root, dry_run=False, stop_after_steps=1
    )
    calibration.run_calibration(settings)
    path = settings.output_dir / "state.json"
    state = json.loads(path.read_text())
    state["schema_version"] = 1
    path.write_text(json.dumps(state))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="resumable calibration state differs"):
        calibration.run_calibration(replace(settings, stop_after_steps=None))
    assert path.read_bytes() == before


def test_cpu_calibration_resumes_from_fresh_champion_ema(tmp_path: Path) -> None:
    config, champion, replay_root, manifest, shard = _fixture(tmp_path)
    before = _snapshot((champion, manifest, shard))
    paused = calibration.run_calibration(
        _settings(
            tmp_path,
            config,
            champion,
            replay_root,
            dry_run=False,
            stop_after_steps=1,
        )
    )
    assert paused["status"] == "paused"
    assert paused["progress"]["completed_steps"] == 1

    resumed_settings = replace(
        _settings(
            tmp_path,
            config,
            champion,
            replay_root,
            dry_run=False,
        ),
        stop_after_steps=None,
    )
    result = calibration.run_calibration(resumed_settings)

    assert result["status"] == "complete"
    assert result["training"]["completed_steps"] == 2
    assert result["training"]["finite"] is True
    assert result["optimizer"]["fresh_from_champion_ema"] is True
    assert result["optimizer"]["source_optimizer_loaded"] is False
    assert result["heldout"]["samples"] == 2
    assert result["heldout"]["batches"] == 2
    unsigned = dict(result)
    expected = unsigned.pop("result_sha256")
    assert calibration._digest(unsigned) == expected
    persisted = json.loads(
        (tmp_path / "calibration-output" / "result.json").read_text(encoding="utf-8")
    )
    assert persisted == result
    assert _snapshot((champion, manifest, shard)) == before


def test_resume_charges_wall_clock_downtime_to_h100_budget(
    tmp_path: Path,
) -> None:
    config, champion, replay_root, _manifest, _shard = _fixture(tmp_path)
    settings = _settings(
        tmp_path,
        config,
        champion,
        replay_root,
        dry_run=False,
        stop_after_steps=1,
    )
    paused = calibration.run_calibration(settings)
    assert paused["status"] == "paused"
    state_path = settings.output_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["measurement_started_ns"] = time.time_ns() - int(2.1 * 3600 * 1e9)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    exhausted = calibration.run_calibration(
        replace(settings, stop_after_steps=None),
    )

    assert exhausted["status"] == "budget_exhausted"
    assert exhausted["progress"]["completed_steps"] == 1


def test_compile_cache_owner_is_contract_bound_and_tamper_evident(
    tmp_path: Path,
) -> None:
    config, champion, replay_root, _manifest, _shard = _fixture(tmp_path)
    settings = _settings(
        tmp_path,
        config,
        champion,
        replay_root,
        dry_run=False,
    )
    contract_sha256 = "a" * 64

    with isolated_compile_cache(settings.output_dir) as cache:
        owner = calibration._bind_compile_cache(
            settings,
            cache,
            contract_sha256,
        )
        assert owner["run_contract_sha256"] == contract_sha256
        marker = cache.root / "cache-owner.json"
        tampered = json.loads(marker.read_text(encoding="utf-8"))
        tampered["arm"] = "different"
        marker.write_text(json.dumps(tampered), encoding="utf-8")

        with pytest.raises(ValueError, match="ownership marker"):
            calibration._bind_compile_cache(settings, cache, contract_sha256)


def test_new_calibration_rejects_unowned_cache_content(tmp_path: Path) -> None:
    config, champion, replay_root, _manifest, _shard = _fixture(tmp_path)
    settings = _settings(
        tmp_path,
        config,
        champion,
        replay_root,
        dry_run=False,
    )

    with isolated_compile_cache(settings.output_dir) as cache:
        (cache.root / "inductor" / "poisoned.bin").write_bytes(b"poison")
        with pytest.raises(ValueError, match="new calibration cache is not empty"):
            calibration._validated_state(settings, "b" * 64, cache)


def test_runner_rejects_replay_overlap_before_creating_cache(tmp_path: Path) -> None:
    config, champion, replay_root, _manifest, _shard = _fixture(tmp_path)
    output = replay_root / "calibration"
    settings = replace(
        _settings(
            tmp_path,
            config,
            champion,
            replay_root,
            dry_run=False,
        ),
        output_dir=output,
    )

    with pytest.raises(ValueError, match="must not overlap"):
        calibration.run_calibration(settings)
    assert not output.exists()


def test_owned_compile_cache_is_recursively_revalidated_on_resume(
    tmp_path: Path,
) -> None:
    config, champion, replay_root, _manifest, _shard = _fixture(tmp_path)
    settings = _settings(
        tmp_path,
        config,
        champion,
        replay_root,
        dry_run=False,
    )

    with isolated_compile_cache(settings.output_dir) as cache:
        calibration._bind_compile_cache(settings, cache, "c" * 64)
        outside = tmp_path / "outside"
        outside.mkdir()
        (cache.root / "inductor" / "unsafe").symlink_to(
            outside,
            target_is_directory=True,
        )
        with pytest.raises(ValueError, match="contains a symlink"):
            calibration._bind_compile_cache(settings, cache, "c" * 64)


def test_cli_reexecutes_with_cache_bound_before_runner_process(
    tmp_path: Path,
) -> None:
    config, champion, replay_root, _manifest, _shard = _fixture(tmp_path)
    output = tmp_path / "subprocess-output"
    read_only_home = tmp_path / "read-only-home"
    read_only_home.mkdir(mode=0o500)
    environment = os.environ.copy()
    environment["HOME"] = str(read_only_home)
    environment.pop(calibration.PREIMPORT_CACHE_BOOTSTRAP_ENV, None)

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(calibration.__file__).resolve()),
            "--config",
            str(config),
            "--champion",
            str(champion),
            "--replay-root",
            str(replay_root),
            "--replay-cutoff",
            "7",
            "--output-dir",
            str(output),
            "--arm",
            calibration.CONTROL_ARM,
            "--steps",
            "2",
            "--batch-size",
            "1",
            "--evaluation-batch-size",
            "1",
            "--max-samples",
            "8",
            "--holdout-fraction",
            "0.25",
            "--seed",
            "23",
            "--device",
            "cpu",
            "--budget-h100-hours",
            "0.01",
            "--checkpoint-interval",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert (output / "compile-cache" / "v1" / "home").is_dir()
    assert json.loads(completed.stdout)["status"] == "complete"
