from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest
import torch
import yaml

from deltreltrain.checkpoint import ExponentialMovingAverage, write_model_pointer
from deltreltrain.config import load_config
from deltreltrain.contracts import TARGET_OUTCOME
from deltreltrain.learner import ImmutableModelPublisher
from deltreltrain.model import GraphResTNet
from deltreltrain.optim import build_optimizer
from deltreltrain.replay import collate_replay_samples, write_replay_shard
from deltreltrain.runtime import RunIdentity
from deltreltrain.training import build_scheduler
from scripts import run_frozen_replay_optimizer_calibration as calibration
from scripts.compare_recovery_calibration import compare, paired_cell_comparison
from scripts.frozen_gradient_diagnostics import (
    gradient_geometry,
    per_head_gradient_conflicts,
)
from test_frozen_replay_optimizer_calibration import (
    RUN_ID,
    FAMILY,
    _sample,
    _write_replay,
)


def _fixture(tmp_path: Path):
    base = Path(__file__).parents[1] / "configs/h100-8gpu-pie-even.yaml"
    raw = yaml.safe_load(base.read_text())
    raw["model"].update(
        width=8,
        rrt_groups=1,
        attention_heads=1,
        kv_heads=1,
        ff_multiplier=1.0,
        local_blocks_per_group=1,
    )
    raw["train"].update(
        per_rank_batch_size=2, precision="fp32", compile=False, ema_decay=0.9
    )
    raw["train"]["scheduler"] = {
        "warmup_steps": 100,
        "total_steps": 1000,
        "min_lr_ratio": 0.01,
    }
    raw["optimizer"].update(min_muon_elements=1, fallback_to_adamw=False)
    raw["data"].update(d5_augmentation=False, workers=0, pin_memory=False)
    raw["learner"]["device"] = "cpu"
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    model = GraphResTNet(config.model)
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.train.scheduler)
    ema = ExponentialMovingAverage(model, decay=config.train.ema_decay)
    publisher = ImmutableModelPublisher(
        tmp_path / "source/learner",
        RunIdentity(tmp_path / "source/run.json", RUN_ID, FAMILY, 1),
    )
    manifest = publisher.publish(
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
        champion, manifest, role="champion", promotion_result="bootstrap"
    )
    replay_root = tmp_path / "source/replay"
    database, shard = _write_replay(replay_root)
    samples = []
    for cell, count in calibration._cell_quotas(80).items():
        for index in range(count):
            sample = _sample(len(samples))
            samples.append(
                replace(
                    sample,
                    mode=cell.rsplit("-", 1)[1],
                    pie=cell.startswith("pie"),
                    handicap=1 if cell.startswith("pie") else 2,
                    target_mask=sample.target_mask | TARGET_OUTCOME,
                    outcome=1 if sample.to_move == 0 else 0,
                    final_scores=np.array([1, 0], dtype=np.int16),
                )
            )
    with sqlite3.connect(database) as connection:
        template = list(connection.execute("SELECT * FROM shards").fetchone())
        connection.execute("DELETE FROM shards")
        for index, cell in enumerate(calibration.RECOVERY_CELLS):
            selected = [
                sample
                for sample in samples
                if sample.mode == cell.rsplit("-", 1)[1]
                and sample.pie == cell.startswith("pie")
            ]
            shard = write_replay_shard(replay_root / "shards" / f"{cell}.npz", selected)
            row = list(template)
            row[0], row[1], row[2], row[-1] = (
                index + 7,
                str(shard.relative_to(replay_root)),
                len(selected),
                calibration.sha256_file(shard),
            )
            connection.execute(
                "INSERT INTO shards VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", row
            )
    settings = calibration.CalibrationSettings(
        config=path,
        champion=champion,
        replay_root=replay_root,
        replay_cutoff=10,
        output_dir=tmp_path / "control",
        arm=calibration.RECOVERY_ARMS[0],
        steps=2,
        batch_size=2,
        evaluation_batch_size=8,
        max_samples=80,
        holdout_fraction=0.25,
        seed=17,
        device="cpu",
        budget_h100_hours=0.02,
        checkpoint_interval=1,
        effective_muon_lr=0.0005,
        effective_adamw_lr=0.0000075,
    )
    return settings, config, samples, shard, database


def test_recovery_explicit_rates_fresh_checkpoint_and_complete_cells(tmp_path):
    settings, config, _, shard, database = _fixture(tmp_path)
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (settings.champion, shard, database)
    }
    dry = calibration.run_calibration(replace(settings, dry_run=True))
    assert dry["partition"]["method"] == calibration.RECOVERY_PARTITION
    assert dry["config_contract"]["train"]["scheduler"] == {
        "warmup_steps": 0,
        "total_steps": 2,
        "min_lr_ratio": 1.0,
    }
    assert not settings.output_dir.exists()
    result = calibration.run_calibration(settings)
    assert result["status"] == "complete"
    assert result["recovery_diagnostics"]["actual_final_learning_rates"] == [
        0.0005,
        0.0000075,
        0.0000075,
    ]
    for heldout in (result["heldout"], result["recovery_diagnostics"]["raw_heldout"]):
        assert set(heldout["cells"]) == set(calibration.RECOVERY_CELLS)
        assert all(cell["games"] >= 1 for cell in heldout["cells"].values())
    checkpoint = torch.load(result["candidate_checkpoint"]["path"], weights_only=False)
    assert checkpoint["scheduler"]["base_lrs"] == [0.0005, 0.0000075, 0.0000075]
    assert checkpoint["ema"]["num_updates"] == 2
    assert result["optimizer"]["source_optimizer_loaded"] is False
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before
    } == before
    assert config.train.scheduler.warmup_steps == 100


def test_recovery_refuses_partial_cells_and_nonfinalized_rows(tmp_path):
    settings, config, samples, shard, database = _fixture(tmp_path)
    # Merely published policy rows cannot masquerade as finalized outcome data.
    samples[-1] = replace(
        samples[-1], target_mask=samples[-1].target_mask & ~TARGET_OUTCOME, outcome=-1
    )
    write_replay_shard(shard, samples[-4:])
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE shards SET checksum_sha256=? WHERE id=10",
            (calibration.sha256_file(shard),),
        )
    with pytest.raises(ValueError, match="coverage"):
        calibration.freeze_replay(
            settings, calibration._champion_pin(settings.champion, config), decode=False
        )


def test_recovery_requires_explicit_rates_and_keeps_legacy_scope(tmp_path):
    settings, _, _, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="explicit"):
        calibration.validate_settings(replace(settings, effective_muon_lr=None))
    with pytest.raises(ValueError, match="recovery arm"):
        calibration.validate_settings(replace(settings, arm=calibration.CONTROL_ARM))


def test_recovery_resume_keeps_rates_and_rejects_changed_rate(tmp_path):
    settings, _, _, _, _ = _fixture(tmp_path)
    paused = calibration.run_calibration(replace(settings, stop_after_steps=1))
    assert paused["status"] == "paused"
    with pytest.raises(ValueError, match="state differs"):
        calibration.run_calibration(replace(settings, effective_muon_lr=0.002))
    result = calibration.run_calibration(settings)
    assert result["status"] == "complete"
    assert result["recovery_diagnostics"]["actual_final_learning_rates"] == [
        0.0005,
        0.0000075,
        0.0000075,
    ]


def test_gradient_diagnostic_is_read_only_and_reports_shared_losses(tmp_path):
    _, config, samples, _, _ = _fixture(tmp_path)
    model = GraphResTNet(config.model)
    model.train()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    diagnostic = per_head_gradient_conflicts(
        model,
        collate_replay_samples(samples[:2]),
        config,
        device=torch.device("cpu"),
        precision="fp32",
    )
    assert diagnostic["rows"] == 2
    assert diagnostic["weighted_gradient_norms"]["policy"] > 0
    assert diagnostic["weighted_gradient_norms"]["outcome"] > 0
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(
        torch.equal(value, model.state_dict()[name]) for name, value in before.items()
    )
    geometry = gradient_geometry(
        {
            "a": torch.tensor([1.0, 0.0]),
            "b": torch.tensor([-1.0, 0.0]),
            "zero": torch.zeros(2),
        }
    )
    assert geometry["pairwise_cosines"]["a/b"] == -1
    assert geometry["pairwise_cosines"]["a/zero"] is None


def test_cell_bootstrap_does_not_let_handicap_sample_count_change_objective():
    control, treatment = {}, {}
    for cell in calibration.RECOVERY_CELLS:
        for index in range(8):
            identity = f"{cell}-{index}"
            row = {
                "cell": cell,
                "samples": 10000 if cell.startswith("handicap") else 1,
                "reference": {"composite": 2.0},
                "candidate": {"composite": 2.0},
            }
            control[identity] = row
            treatment[identity] = {
                **row,
                "candidate": {"composite": 3.0 if cell.startswith("handicap") else 1.0},
            }
    result = paired_cell_comparison(
        control, treatment, alpha=0.025, samples=100, seed=17, minimum_games=8
    )
    assert result["weighted_composite_improvement"] == pytest.approx(0.8)
    assert result["passes_diagnostic_screen"] is True
    treatment.pop(next(iter(treatment)))
    with pytest.raises(ValueError, match="different games"):
        paired_cell_comparison(
            control, treatment, alpha=0.025, samples=100, seed=17, minimum_games=8
        )


def test_recovery_comparator_checks_hashed_paired_rates(tmp_path):
    settings, _, _, _, _ = _fixture(tmp_path)
    control = calibration.run_calibration(settings)
    treatment = calibration.run_calibration(
        replace(
            settings,
            output_dir=tmp_path / "moderate",
            arm=calibration.RECOVERY_ARMS[1],
            effective_muon_lr=0.002,
            effective_adamw_lr=0.00003,
        )
    )
    paths = [settings.output_dir / "result.json", tmp_path / "moderate/result.json"]
    result = compare(paths, bootstrap_samples=100, minimum_games=2)
    assert result["production_promotion_authorized"] is False
    assert (
        result["suggested_strength_screen_arm"] is None
    )  # Too few handicap holdout games.
    assert control["partition"] == treatment["partition"]
    treatment["recovery_diagnostics"]["actual_final_learning_rates"][0] = 0.04
    treatment.pop("result_sha256")
    treatment["result_sha256"] = calibration._digest(treatment)
    paths[1].write_text(json.dumps(treatment))
    with pytest.raises(ValueError, match="actual rates"):
        compare(paths, bootstrap_samples=100)
