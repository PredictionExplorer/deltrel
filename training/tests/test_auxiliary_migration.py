"""Additive prediction rollout keeps architecture and recovery boundaries strict."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts import graceful_training_deploy as deploy
from scripts import migrate_continuous_profile as migration
from scripts import prepare_auxiliary_training_profile as auxiliary_preparation
from deltreltrain.auxiliary_upgrade import AUXILIARY_LOSSES
from deltreltrain.checkpoint import (
    ExponentialMovingAverage,
    inference_model_config,
    load_model_manifest,
    save_checkpoint,
)
from deltreltrain.config import ConfigError, load_config
from deltreltrain.contracts import FEATURE_SCHEMA_HASH, RULES_HASH_WIRE
from deltreltrain.model import GraphResTNet, MODEL_SCHEMA_VERSION, ModelConfig
from test_continuous_profile_migration import _fixture
from test_graceful_training_deploy import deployment as deployment, stopped_files, write


def upgraded(config):
    return replace(
        config,
        model=replace(config.model, auxiliary_predictions=True),
        loss=replace(config.loss, **{name: 0.05 for name in AUXILIARY_LOSSES}),
    )


def test_auxiliary_migration_dry_run_preserves_checkpoint_and_counters(tmp_path):
    fixture = _fixture(tmp_path)
    before = {
        path: path.read_bytes() for path in fixture.root.rglob("*") if path.is_file()
    }
    existing_changes = {
        change["path"]
        for change in migration.migrate_continuous_profile(fixture.request)["changes"]
    }
    old = load_config(fixture.candidate_profile)
    new = upgraded(old)
    fixture.candidate_profile.write_text(yaml.safe_dump(new.as_dict()))
    result = migration.migrate_continuous_profile(fixture.request)
    assert {change["path"] for change in result["changes"]} == existing_changes | {
        "model.auxiliary_predictions",
        *(f"loss.{name}" for name in AUXILIARY_LOSSES),
    }
    assert result["boundary"]["learner_step"] == 100
    assert {
        path: path.read_bytes() for path in fixture.root.rglob("*") if path.is_file()
    } == before


@pytest.mark.parametrize(
    "change,match",
    [
        ("depth", "model configuration is immutable"),
        ("local_operator", "model configuration is immutable"),
        ("outcome", "loss configuration is immutable"),
        (
            "teacher",
            "loss configuration is immutable|new profile fails continuous validation",
        ),
        ("optimizer", "optimizer configuration is immutable"),
        ("downgrade", "model configuration is immutable"),
    ],
)
def test_auxiliary_migration_rejects_core_changes_and_downgrade(
    tmp_path, change, match
):
    fixture = _fixture(tmp_path)
    old = load_config(fixture.candidate_profile)
    new = upgraded(old)
    if change == "depth":
        new = replace(
            new, model=replace(new.model, rrt_groups=new.model.rrt_groups + 1)
        )
    elif change == "local_operator":
        new = replace(new, model=replace(new.model, local_operator="source_gated"))
    elif change == "outcome":
        new = replace(new, loss=replace(new.loss, outcome=0.8))
    elif change == "teacher":
        new = replace(new, loss=replace(new.loss, teacher_policy=0.9))
    elif change == "optimizer":
        new = replace(new, optimizer=replace(new.optimizer, adamw_lr=0.001))
    elif change == "downgrade":
        old, new = new, old
    with pytest.raises(migration.MigrationError, match=match):
        migration._validate_profile_pair(old, new, run_root=fixture.root)


def test_auxiliary_losses_can_be_disabled_without_removing_trained_heads(tmp_path):
    fixture = _fixture(tmp_path)
    old = upgraded(load_config(fixture.candidate_profile))
    new = replace(
        old, loss=replace(old.loss, **{name: 0.0 for name in AUXILIARY_LOSSES})
    )
    changes = migration._validate_profile_pair(old, new, run_root=fixture.root)
    assert {path for path, _, _ in changes} == {
        f"loss.{name}" for name in AUXILIARY_LOSSES
    }
    assert new.model == old.model and new.model.auxiliary_predictions
    with pytest.raises(ConfigError, match="auxiliary loss weights require"):
        replace(old, model=replace(old.model, auxiliary_predictions=False))


@pytest.mark.parametrize("interrupt_once", [False, True])
@pytest.mark.parametrize("source_auxiliary", [False, True])
def test_controller_recovery_retains_new_heads_checkpoint_and_progress(
    deployment, monkeypatch, interrupt_once, source_auxiliary
):
    source = {"model": {"width": 384}, "loss": {"policy": 1.0, "outcome": 1.0}}
    if source_auxiliary:
        source["model"]["auxiliary_predictions"] = True
        source["loss"].update({name: 0.02 for name in AUXILIARY_LOSSES})
    candidate = deepcopy(source)
    candidate["model"]["auxiliary_predictions"] = True
    if not source_auxiliary:
        candidate["loss"].update({name: 0.05 for name in AUXILIARY_LOSSES})
    deployment.source.write_text(yaml.safe_dump(source))
    deployment.candidate.write_text(yaml.safe_dump(candidate))
    deployment.target.write_bytes(deployment.candidate.read_bytes())
    (deployment.root / "profile.sha256").write_text(
        f"{deploy.digest(deployment.target)}  {deployment.target}\n"
    )
    (deployment.root / "source-commit.txt").write_text(deployment.plan["target_commit"])
    checkpoint = stopped_files(deployment, learner_step=413100, checkpoint_step=413100)
    checkpoint_file = deployment.root / "learner/recovery/checkpoint.pt"
    pointer_file = deployment.root / "learner/recovery.json"
    preserved = (checkpoint_file.read_bytes(), pointer_file.read_bytes())
    write(deployment.base / "prepared.json", {"plan": deployment.plan})
    write(deployment.base / "stop-requested.json", {})
    monkeypatch.setattr(
        deployment,
        "stop",
        lambda **kw: {
            "checkpoint": checkpoint,
            "progress_available": True,
        },
    )
    migrations, installed, resumes = [], [], []
    gates = []
    monkeypatch.setattr(
        auxiliary_preparation,
        "ensure_auxiliary_gate",
        lambda source_path, target_path: gates.append((source_path, target_path)),
    )

    def migrate(current, recovery_source, name, commit, *, recovery=False):
        assert recovery and current == deployment.target
        assert commit == deployment.plan["target_commit"]
        recovery_payload = yaml.safe_load(recovery_source.read_text())
        assert recovery_payload["model"] == source["model"] | {
            "auxiliary_predictions": True
        }
        if source_auxiliary:
            assert recovery_source == deployment.source
            assert recovery_payload == source
        else:
            assert recovery_payload["loss"] == source["loss"] | {
                loss_name: 0.0 for loss_name in AUXILIARY_LOSSES
            }
        migrations.append(recovery_source)
        destination = deployment.root / name
        destination.write_bytes(recovery_source.read_bytes())
        (deployment.root / "profile.sha256").write_text(
            f"{deploy.digest(destination)}  {destination}\n"
        )

    def start(profile, resume):
        resumes.append((profile, resume))
        if interrupt_once and len(resumes) == 1:
            raise RuntimeError("interrupted after inverse migration")
        return {"ready": True}

    monkeypatch.setattr(deployment, "migrate", migrate)
    monkeypatch.setattr(
        deployment, "install_units", lambda p, **kw: installed.append((p, kw))
    )
    monkeypatch.setattr(deployment, "start", start)
    monkeypatch.setattr(deployment, "restore_support", lambda: None)
    if interrupt_once:
        with pytest.raises(RuntimeError, match="interrupted after inverse"):
            deployment.recover()
    deployment.recover()
    assert len(migrations) == 1
    if source_auxiliary:
        assert gates == []
        assert not (deployment.base / "auxiliary-compatible-recovery.yaml").exists()
    else:
        assert gates and all(
            source_path == deployment.source for source_path, _ in gates
        )
    assert all(resume == checkpoint for _, resume in resumes)
    assert all(kw == {} for _, kw in installed)  # Keep compatible new release.
    assert all(
        yaml.safe_load(profile.read_text())["model"]["auxiliary_predictions"]
        for profile, _ in installed
    )
    assert (checkpoint_file.read_bytes(), pointer_file.read_bytes()) == preserved
    recovered = json.loads((deployment.base / "recovered.json").read_text())
    assert recovered["stopped_boundary"]["checkpoint"]["step"] == 413100


def published_checkpoint(tmp_path, *, auxiliary, changes=None):
    config = ModelConfig(
        width=16,
        rrt_groups=1,
        attention_heads=4,
        kv_heads=1,
        auxiliary_predictions=auxiliary,
        **(changes or {}),
    )
    model = GraphResTNet(config)
    game = load_config(Path(__file__).parents[1] / "configs/small.yaml").as_dict()[
        "game"
    ]
    source = save_checkpoint(
        tmp_path / "staging.pt",
        model=model,
        step=17,
        ema=ExponentialMovingAverage(model),
        config={"model": asdict(config), "game": game},
        extra={"run_id": "aux-test", "generation_family": "aux-family"},
    )
    checkpoint_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    checkpoint = tmp_path / f"sha256-{checkpoint_hash}.pt"
    source.rename(checkpoint)
    payload = {
        "format": "deltreltrain.model-manifest",
        "schema_version": 3,
        "model_identity": f"sha256-{checkpoint_hash}",
        "model_version": f"sha256-{checkpoint_hash}",
        "model_step": 17,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "weights": "ema",
        "run_id": "aux-test",
        "generation_family": "aux-family",
        "rules_hash": RULES_HASH_WIRE,
        "feature_schema_hash": f"{FEATURE_SCHEMA_HASH:016x}",
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "created_ns": 123,
    }
    encoded = json.dumps(payload).encode()
    manifest = tmp_path / f"manifest-{hashlib.sha256(encoded).hexdigest()}.json"
    manifest.write_bytes(encoded)
    return load_model_manifest(manifest), config, game


@pytest.mark.parametrize("auxiliary", [False, True])
def test_inference_resolves_original_and_new_champions_from_verified_artifacts(
    tmp_path, auxiliary
):
    manifest, actual, game = published_checkpoint(tmp_path, auxiliary=auxiliary)
    expected = replace(actual, auxiliary_predictions=True)
    assert (
        inference_model_config(
            manifest, expected_model=expected, expected_game_config=game
        )
        == actual
    )
    with pytest.raises(ValueError, match="trained architecture"):
        inference_model_config(
            manifest,
            expected_model=replace(expected, rrt_groups=2),
            expected_game_config=game,
        )
    with pytest.raises(ValueError, match="game/rules configuration"):
        inference_model_config(
            manifest,
            expected_model=expected,
            expected_game_config=game | {"mode": "classic"},
        )
    # A previously parsed object cannot bypass a changed checkpoint's integrity.
    manifest.checkpoint.write_bytes(b"corrupt replacement")
    with pytest.raises(ValueError, match="byte length|SHA-256"):
        inference_model_config(
            manifest, expected_model=expected, expected_game_config=game
        )
