import json
import sqlite3

import yaml

from scripts import migrate_continuous_profile as migration
from startrain.config import load_config
from startrain.pie_promotion import pie_promotion_config

from test_continuous_profile_migration import _fixture, _write_json


def test_allocation_migration_preserves_data_training_credit_and_strength_epoch(
    tmp_path,
):
    fixture = _fixture(tmp_path, "h100-8gpu-pie-even.yaml")
    source = load_config(fixture.old_profile)
    target = pie_promotion_config(source)
    fixture.candidate_profile.write_text(
        yaml.safe_dump(json.loads(json.dumps(target.as_dict())), sort_keys=False)
    )
    with sqlite3.connect(fixture.root / "replay/manifest.sqlite3") as db:
        db.execute(
            "CREATE TABLE training_objective_counters (run_id TEXT, "
            "generation_family TEXT, training_objective TEXT, committed_samples INTEGER)"
        )
        db.execute(
            "INSERT INTO training_objective_counters VALUES (?, ?, ?, ?)",
            ("continuous-test-run", "family-continuous-test", "ring10_pie", 50_000),
        )
    retained = {
        "strength-epoch.json": {"started_ns": 12, "anchor_identity": "same-anchor"},
        "arena/promotion-status.json": {"terminal": False},
        "arena/pending.resume.json": {"arena_state": {"actions": [1, 2]}},
        "arena/old-result.json": {"pairs": [{"outcomes": [1, -1]}]},
        "learner/utd-segment.json": {
            "schema_version": 1,
            "run_id": "continuous-test-run",
            "generation_family": "family-continuous-test",
            "training_objective": "ring10_pie",
            "target_updates_per_new_sample": 1.5,
            "baseline_examples_consumed": 20_000,
            "baseline_committed_replay_samples": 10_000,
            "created_ns": 12,
        },
    }
    for name, payload in retained.items():
        _write_json(fixture.root / name, payload)
    paths = [fixture.root / name for name in retained]
    paths.extend([fixture.checkpoint, fixture.root / "replay/manifest.sqlite3"])
    original = {path: path.read_bytes() for path in paths}
    plan = migration.plan_migration(fixture.request)
    assert plan.strength_epoch_payload is None
    assert plan.utd_segment_payload is None
    transition = plan.migration_record["evaluation_contract_transition"]
    assert transition["from"]["identity"] != transition["to"]["identity"]
    assert set(transition["retained_evidence"]) == {
        name for name in retained if name.startswith("arena/")
    }
    migration.apply_migration(plan)
    assert load_config(plan.target_profile) == target
    assert {path: path.read_bytes() for path in paths} == original
