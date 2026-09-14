from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from types import SimpleNamespace

import pytest

import scripts.preflight_run_state as preflight
from startrain.checkpoint import game_configs_compatible
from startrain.config import GameConfig
from startrain.config_compatibility import without_search_execution_defaults
from startrain.contracts import FEATURE_SCHEMA_HASH, RULES_HASH_WIRE


def segment(scope=None, baseline=12):
    return {
        "schema_version": 1,
        "run_id": "run-pie",
        "generation_family": "family-pie",
        "target_updates_per_new_sample": 1.5,
        "baseline_examples_consumed": 100,
        "baseline_committed_replay_samples": baseline,
        **({"training_objective": scope} if scope is not None else {}),
    }


def test_checkpoint_pie_default_is_compatible_only_inside_same_admitted_family():
    old = asdict(GameConfig())
    new = {**old, "pie_rule": True}
    assert game_configs_compatible(old, new)
    assert game_configs_compatible(new, old)
    assert not game_configs_compatible(old, {**new, "mode": "classic"})
    assert not game_configs_compatible(
        old, {**new, "variants": {**old["variants"], "handicap_max": 8}}
    )
    with pytest.raises(ValueError, match="outside its variant rules"):
        game_configs_compatible(
            old, {**new, "variants": {**old["variants"], "pie_allowed": False}}
        )
    with pytest.raises(ValueError, match="pie-rule configuration"):
        game_configs_compatible(old, {**new, "pie_rule": 1})


def test_legacy_hash_defaults_are_omitted_but_pie_training_changes_remain():
    legacy = {
        "arena": {"variant_policy": "legacy_six"},
        "selfplay": {"pie_even_training": False},
    }
    assert without_search_execution_defaults(legacy) == {"arena": {}, "selfplay": {}}
    enabled = {
        "arena": {"variant_policy": "pie_even"},
        "selfplay": {"pie_even_training": True},
    }
    assert without_search_execution_defaults(enabled) == enabled
    assert legacy["arena"]["variant_policy"] == "legacy_six"
    untyped = {"selfplay": {"pie_even_training": 0}}
    assert without_search_execution_defaults(untyped) == untyped


@pytest.mark.parametrize(
    "actual,expected", [(None, "ring10_pie"), ("ring10_pie", None)]
)
def test_preflight_rejects_wrong_scope_even_if_target_matches(actual, expected):
    with pytest.raises(preflight.StatePreflightError, match="identity or target"):
        preflight._parse_utd_segment(
            segment(actual),
            run_id="run-pie",
            generation_family="family-pie",
            target=1.5,
            maximum_examples=100,
            maximum_samples=1000,
            training_objective=expected,
        )


def test_preflight_scoped_baseline_uses_scoped_credit_limit():
    with pytest.raises(preflight.StatePreflightError, match="replay precedes"):
        preflight._parse_utd_segment(
            segment("ring10_pie", baseline=13),
            run_id="run-pie",
            generation_family="family-pie",
            target=1.5,
            maximum_examples=100,
            maximum_samples=12,
            training_objective="ring10_pie",
        )


def test_readonly_preflight_scope_counts_retained_eligible_rows_or_durable_counter(
    tmp_path, monkeypatch
):
    root = tmp_path / "run"
    (root / "replay").mkdir(parents=True)
    database = root / "replay" / "manifest.sqlite3"
    with sqlite3.connect(database) as db:
        db.executescript(
            "CREATE TABLE store_metadata(key TEXT,value TEXT);"
            "CREATE TABLE runs(run_id TEXT,generation_family TEXT,created_ns INTEGER);"
            "CREATE TABLE run_counters(run_id TEXT,generation_family TEXT,"
            "committed_samples INTEGER,history_complete INTEGER);"
            "CREATE TABLE shards(run_id TEXT,generation_family TEXT,ring INTEGER,"
            "segment TEXT,variant TEXT,sample_count INTEGER,state TEXT);"
        )
        db.executemany(
            "INSERT INTO store_metadata VALUES (?,?)",
            [
                ("manifest_schema_version", "5"),
                ("rules_hash", RULES_HASH_WIRE),
                ("feature_schema_hash", f"{FEATURE_SCHEMA_HASH:016x}"),
            ],
        )
        db.execute("INSERT INTO runs VALUES ('run-pie','family-pie',1)")
        db.execute("INSERT INTO run_counters VALUES ('run-pie','family-pie',100,1)")
        db.executemany(
            "INSERT INTO shards VALUES ('run-pie','family-pie',?,?,?,?, 'ready')",
            [
                (4, "pie", "pie-classic", 5),
                (10, "handicap", "handicap-9-double", 7),
                (4, "handicap", "handicap-9-double", 30),
                (10, "standard", "double", 58),
            ],
        )
    monkeypatch.setattr(preflight, "validate_game_publications", lambda *_, **__: None)
    monkeypatch.setattr(
        preflight,
        "prove_legacy_committed_sample_history",
        lambda *_, **__: SimpleNamespace(
            complete=True,
            shard_samples=100,
            failures=(),
            shard_count=4,
            expected_games=0,
            recorded_games=0,
            maximum_shard_id=4,
            sqlite_sequence=4,
        ),
    )
    before = database.read_bytes()
    report, _ = preflight._validate_replay(
        root,
        run_id="run-pie",
        generation_family="family-pie",
        created_ns=1,
        training_objective="ring10_pie",
    )
    assert report["committed_samples"] == 100
    assert report["training_committed_samples"] == 12
    assert report["training_ready_samples_by_ring"] == {"4": 5, "10": 7}
    assert database.read_bytes() == before
    with sqlite3.connect(database) as db:
        db.execute(
            "CREATE TABLE training_objective_counters(run_id TEXT,generation_family TEXT,training_objective TEXT,committed_samples INTEGER)"
        )
        db.execute(
            "INSERT INTO training_objective_counters VALUES ('run-pie','family-pie','ring10_pie',77)"
        )
    report, _ = preflight._validate_replay(
        root,
        run_id="run-pie",
        generation_family="family-pie",
        created_ns=1,
        training_objective="ring10_pie",
    )
    assert report["training_committed_samples"] == 77


def mocked_run(tmp_path, monkeypatch, *, checkpoint_scope=None, persisted=True):
    root = tmp_path / "run"
    (root / "learner").mkdir(parents=True)
    (root / "learner" / "recovery.json").write_text("{}")
    if persisted:
        (root / "learner" / "utd-segment.json").write_text(
            json.dumps(segment("ring10_pie"))
        )
    experiment = SimpleNamespace(
        orchestration=SimpleNamespace(training_objective="ring10_pie"),
        learner=SimpleNamespace(
            target_updates_per_new_sample=1.5, selfplay_snapshot_interval_examples=None
        ),
    )
    checkpoint_segment = segment(
        checkpoint_scope, baseline=12 if checkpoint_scope else 90
    )
    monkeypatch.setattr(
        preflight,
        "load_run_identity",
        lambda _: SimpleNamespace(
            run_id="run-pie", generation_family="family-pie", created_ns=1
        ),
    )
    monkeypatch.setattr(
        preflight, "_validate_profile", lambda *_, **__: (experiment, {})
    )
    monkeypatch.setattr(
        preflight,
        "_validate_replay",
        lambda *_, **__: (
            {
                "committed_samples": 100,
                "training_committed_samples": 12,
                "history_complete": True,
            },
            False,
        ),
    )
    monkeypatch.setattr(
        preflight,
        "_validate_recovery",
        lambda *_, **__: (
            {"step": 10, "examples_consumed": 100},
            {
                "extra": {"utd_segment": checkpoint_segment},
                "config": {
                    "learner": {"target_updates_per_new_sample": 1.5},
                    "orchestration": {"training_objective": checkpoint_scope},
                },
            },
        ),
    )
    monkeypatch.setattr(preflight, "_plan_cadence", lambda *_, **__: ({}, None, False))
    return root


def test_preflight_accepts_prepared_scope_cutover_before_first_new_checkpoint(
    tmp_path, monkeypatch
):
    root = mocked_run(tmp_path, monkeypatch)
    result = preflight.run_state_preflight(root, tmp_path / "profile.yaml")
    assert result["utd_segment"] == segment("ring10_pie")
    assert result["migrations"] == []


def test_preflight_refuses_to_invent_scope_cutover_from_old_checkpoint(
    tmp_path, monkeypatch
):
    root = mocked_run(tmp_path, monkeypatch, persisted=False)
    with pytest.raises(preflight.StatePreflightError, match="prepared scoped UTD"):
        preflight.run_state_preflight(root, tmp_path / "profile.yaml", apply=True)
    assert not (root / "learner" / "utd-segment.json").exists()


def test_preflight_restores_same_scope_checkpoint_segment(tmp_path, monkeypatch):
    root = mocked_run(
        tmp_path, monkeypatch, checkpoint_scope="ring10_pie", persisted=False
    )
    report = preflight.run_state_preflight(root, tmp_path / "profile.yaml")
    assert report["utd_segment"] == segment("ring10_pie")
    assert report["migrations"] == [
        {"name": "restore_utd_segment_from_checkpoint", "status": "planned"}
    ]


def test_preflight_still_rejects_unexplained_same_scope_divergence(
    tmp_path, monkeypatch
):
    root = mocked_run(tmp_path, monkeypatch, checkpoint_scope="ring10_pie")
    (root / "learner" / "utd-segment.json").write_text(
        json.dumps(segment("ring10_pie", baseline=11))
    )
    with pytest.raises(preflight.StatePreflightError, match="disagrees with recovery"):
        preflight.run_state_preflight(root, tmp_path / "profile.yaml")
