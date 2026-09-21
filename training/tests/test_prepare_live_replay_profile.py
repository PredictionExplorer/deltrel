from dataclasses import replace
import hashlib
import json
from pathlib import Path
import stat

import pytest
import yaml

from scripts import prepare_live_replay_profile as preparation
from deltreltrain.config import load_config


CONFIGS = Path(__file__).parents[1] / "configs"


def source_profile(tmp_path, *, shared=False, fast=True):
    config = load_config(
        CONFIGS / ("h100-8gpu-largest-board-priority.yaml" if shared else "small.yaml")
    )
    config = replace(
        config, selfplay=replace(config.selfplay, record_fast_policy_targets=fast)
    )
    if shared:
        orchestration = config.orchestration
        workers = tuple(
            replace(
                worker,
                actor_cohorts=2,
                actor_lanes=1,
                actor_pipeline=(
                    replace(worker.actor_pipeline, compatible_work=True)
                    if worker.actor_pipeline is not None
                    else None
                ),
            )
            if worker.role == "actor"
            else worker
            for worker in orchestration.gpus
        )
        config = replace(
            config,
            orchestration=replace(
                orchestration,
                gpus=workers,
                model_refresh=replace(
                    orchestration.model_refresh,
                    compatible_cohort_work=True,
                    inference=replace(
                        orchestration.model_refresh.inference, shared_batching=True
                    ),
                ),
            ),
        )
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(config.as_dict(), sort_keys=False))
    return path, config


@pytest.mark.parametrize("shared", [False, True])
def test_prepared_profile_preserves_every_unrelated_control_and_source(
    tmp_path, shared
):
    source, original = source_profile(tmp_path, shared=shared)
    before = source.read_bytes()
    before_stat = source.stat()
    output = tmp_path / "candidate.yaml"
    receipt = preparation.prepare_profile(source, output, shared_scheduling=shared)
    actual = load_config(output)
    assert source.read_bytes() == before
    assert source.stat().st_mtime_ns == before_stat.st_mtime_ns
    assert source.stat().st_mode == before_stat.st_mode
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert actual.selfplay.policy_publication.enabled
    assert actual.selfplay.policy_publication.first_decisions == 8
    assert actual.selfplay.policy_publication.interval_decisions == 32
    assert actual.learner.replay_refresh_seconds == 60.0
    assert (
        actual.learner.target_updates_per_new_sample
        == original.learner.target_updates_per_new_sample
    )
    assert (
        replace(
            actual.selfplay, policy_publication=original.selfplay.policy_publication
        )
        == original.selfplay
    )
    assert (
        replace(
            actual.learner,
            replay_refresh_seconds=original.learner.replay_refresh_seconds,
        )
        == original.learner
    )
    if shared:
        work = actual.orchestration.model_refresh.work_scheduling
        assert work.enabled and work.games_per_lease is None and work.coverage_first
    else:
        assert actual.orchestration == original.orchestration
    restored_refresh = replace(
        actual.orchestration.model_refresh,
        work_scheduling=original.orchestration.model_refresh.work_scheduling,
    )
    restored = replace(
        actual,
        selfplay=original.selfplay,
        learner=original.learner,
        orchestration=replace(actual.orchestration, model_refresh=restored_refresh),
    )
    assert restored == original
    assert receipt["activated"] is False and receipt["status"] == "prepared"
    assert receipt["source"]["profile_sha256"] == hashlib.sha256(before).hexdigest()
    assert (
        receipt["target"]["profile_sha256"]
        == hashlib.sha256(output.read_bytes()).hexdigest()
    )
    assert receipt["changes"] == preparation._changes(
        original.as_dict(), actual.as_dict()
    )
    assert {row["path"] for row in receipt["changes"]} <= preparation.ALLOWED_CHANGES
    assert receipt["unchanged_utd"] == {
        "target_updates_per_new_sample": original.learner.target_updates_per_new_sample,
        "changed": False,
    }
    compatibility = receipt["replay_compatibility"]
    assert compatibility["default_manifest_schema_version"] == 5
    assert compatibility["first_game_revision_manifest_schema_version"] == 6
    assert "schema-6-capable" in compatibility["rollback_requires"]
    assert not list(tmp_path.glob(".candidate.yaml.*.tmp"))


def test_cli_receipt_and_explicit_prefix_intervals(tmp_path, capsys):
    source, _ = source_profile(tmp_path)
    output = tmp_path / "candidate.yaml"
    assert (
        preparation.main(
            [
                "--source",
                str(source),
                "--output",
                str(output),
                "--no-shared-scheduling",
                "--refresh-seconds",
                "45",
                "--first-decisions",
                "4",
                "--interval-decisions",
                "16",
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["target"]["path"] == str(output)
    config = load_config(output)
    assert config.learner.replay_refresh_seconds == 45
    assert config.selfplay.policy_publication.first_decisions == 4
    assert config.selfplay.policy_publication.interval_decisions == 16


def test_missing_fast_targets_and_unsupported_topology_are_actionable(tmp_path, capsys):
    source, _ = source_profile(tmp_path, fast=False)
    before = source.read_bytes()
    output = tmp_path / "candidate.yaml"
    assert (
        preparation.main(
            [
                "--source",
                str(source),
                "--output",
                str(output),
                "--no-shared-scheduling",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert "record_fast_policy_targets must already be true" in error["error"]
    assert source.read_bytes() == before and not output.exists()
    source, _ = source_profile(tmp_path)
    with pytest.raises(
        preparation.ProfilePreparationError, match="--no-shared-scheduling"
    ):
        preparation.prepare_profile(source, output)
    assert not output.exists()


def test_shared_scheduling_does_not_enable_incompatible_cohorts_or_change_quota(
    tmp_path,
):
    source, config = source_profile(tmp_path, shared=True)
    refresh = config.orchestration.model_refresh
    incompatible = replace(
        config,
        orchestration=replace(
            config.orchestration,
            model_refresh=replace(refresh, compatible_cohort_work=False),
            gpus=tuple(
                replace(
                    worker,
                    actor_pipeline=replace(
                        worker.actor_pipeline, compatible_work=False
                    ),
                )
                if worker.role == "actor" and worker.actor_pipeline is not None
                else worker
                for worker in config.orchestration.gpus
            ),
        ),
    )
    source.write_text(yaml.safe_dump(incompatible.as_dict()))
    output = tmp_path / "candidate.yaml"
    with pytest.raises(
        preparation.ProfilePreparationError, match="topology is not ready"
    ):
        preparation.prepare_profile(source, output)
    assert not output.exists()
    override = replace(
        config,
        orchestration=replace(
            config.orchestration,
            model_refresh=replace(
                refresh,
                work_scheduling=replace(refresh.work_scheduling, games_per_lease=128),
            ),
        ),
    )
    source.write_text(yaml.safe_dump(override.as_dict()))
    with pytest.raises(
        preparation.ProfilePreparationError, match="games_per_lease override"
    ):
        preparation.prepare_profile(source, output)
    preparation.prepare_profile(source, output, shared_scheduling=False)
    assert load_config(output).orchestration == override.orchestration


@pytest.mark.parametrize(
    "options",
    [
        {"refresh_seconds": 0},
        {"refresh_seconds": -1},
        {"refresh_seconds": float("nan")},
        {"refresh_seconds": float("inf")},
        {"refresh_seconds": True},
        {"first_decisions": 0},
        {"first_decisions": True},
        {"interval_decisions": 1025},
        {"interval_decisions": 1.5},
        {"shared_scheduling": 1},
    ],
)
def test_invalid_options_never_publish(tmp_path, options):
    source, _ = source_profile(tmp_path)
    output = tmp_path / "candidate.yaml"
    before = source.read_bytes()
    with pytest.raises(ValueError):
        preparation.prepare_profile(
            source, output, **({"shared_scheduling": False} | options)
        )
    assert not output.exists() and source.read_bytes() == before
    assert not list(tmp_path.glob(".candidate.yaml.*.tmp"))


def test_no_overwrite_even_when_another_writer_wins_the_publication_race(
    tmp_path, monkeypatch
):
    source, _ = source_profile(tmp_path)
    output = tmp_path / "candidate.yaml"
    output.write_bytes(b"existing")
    with pytest.raises(ValueError, match="already exists"):
        preparation.prepare_profile(source, output, shared_scheduling=False)
    assert output.read_bytes() == b"existing"
    output.unlink()
    original_link = preparation.os.link

    def racing_link(left, right, **kwargs):
        Path(right).write_bytes(b"concurrent writer")
        return original_link(left, right, **kwargs)

    monkeypatch.setattr(preparation.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        preparation.prepare_profile(source, output, shared_scheduling=False)
    assert output.read_bytes() == b"concurrent writer"
    assert not list(tmp_path.glob(".candidate.yaml.*.tmp"))


@pytest.mark.parametrize("location", ["source", "output", "dangling_output", "parent"])
def test_symlinks_are_refused_without_following_or_overwriting(tmp_path, location):
    source, _ = source_profile(tmp_path)
    output = tmp_path / "candidate.yaml"
    if location == "source":
        link = tmp_path / "source-link.yaml"
        link.symlink_to(source)
        source = link
    elif location == "parent":
        link = tmp_path / "linked-parent"
        link.symlink_to(tmp_path, target_is_directory=True)
        output = link / "candidate.yaml"
    else:
        output.symlink_to(source if location == "output" else tmp_path / "missing")
    with pytest.raises(ValueError, match="symbolic links"):
        preparation.prepare_profile(source, output, shared_scheduling=False)
    assert not (tmp_path / "missing").exists()


def test_serialized_validation_failure_and_source_change_publish_nothing(
    tmp_path, monkeypatch
):
    source, _ = source_profile(tmp_path)
    output = tmp_path / "candidate.yaml"
    original_load = preparation.load_config

    def invalid_temporary(path):
        if Path(path).suffix == ".tmp":
            raise ValueError("injected serialized validation failure")
        return original_load(path)

    with monkeypatch.context() as context:
        context.setattr(preparation, "load_config", invalid_temporary)
        with pytest.raises(ValueError, match="serialized validation"):
            preparation.prepare_profile(source, output, shared_scheduling=False)
    assert not output.exists() and not list(tmp_path.glob(".candidate.yaml.*.tmp"))

    def changed_source(path):
        result = original_load(path)
        if Path(path) == source:
            source.write_bytes(source.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(preparation, "load_config", changed_source)
    with pytest.raises(ValueError, match="source changed"):
        preparation.prepare_profile(source, output, shared_scheduling=False)
    assert not output.exists() and not list(tmp_path.glob(".candidate.yaml.*.tmp"))
