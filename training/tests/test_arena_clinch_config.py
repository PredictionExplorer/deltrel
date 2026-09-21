from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from deltreltrain.balanced_evaluation import evaluation_contract
from deltreltrain.config import ArenaConfig, ConfigError, load_config
from deltreltrain.config_compatibility import (
    compatible_config_epoch_payloads,
    without_arena_clinch_default,
)


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0, None, "false", "true", [], {}])
def test_arena_clinch_requires_boolean_in_typed_and_yaml_config(tmp_path, value):
    with pytest.raises(ConfigError, match="exact_clinch_termination must be boolean"):
        ArenaConfig(exact_clinch_termination=value)
    raw = yaml.safe_load((Path(__file__).parents[1] / "configs/small.yaml").read_text())
    raw.setdefault("arena", {})["exact_clinch_termination"] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="exact_clinch_termination must be boolean"):
        load_config(path)


@pytest.mark.parametrize("enabled", [False, True])
def test_clinch_roundtrip_preserves_search_and_statistical_contract(tmp_path, enabled):
    original = load_config(
        Path(__file__).parents[1] / "configs/h100-8gpu-largest-board-priority.yaml"
    )
    changed = replace(
        original, arena=replace(original.arena, exact_clinch_termination=enabled)
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(changed.as_dict()))
    assert load_config(path) == changed
    assert evaluation_contract(changed.arena) == evaluation_contract(original.arena)
    assert replace(changed.arena, exact_clinch_termination=False) == original.arena


@pytest.mark.parametrize("value", [False, True, 0, None, "false"])
def test_clinch_epoch_omits_only_disabled_boolean_without_mutation(value):
    payload = {"arena": {"exact_clinch_termination": value, "simulations": 256}}
    original = deepcopy(payload)
    legacy = without_arena_clinch_default(payload)
    variants = compatible_config_epoch_payloads(payload)
    assert payload == original
    assert payload in variants and legacy in variants
    if value is False:
        assert legacy == {"arena": {"simulations": 256}}
    else:
        assert all(
            row["arena"]["exact_clinch_termination"] == value for row in variants
        )
        assert all(
            type(row["arena"]["exact_clinch_termination"]) is type(value)
            for row in variants
        )


def test_stopped_clinch_migration_and_rollback_preserve_pending_work(tmp_path):
    from scripts import migrate_continuous_profile as migration
    from test_continuous_profile_migration import _fixture, _write_json

    fixture = _fixture(tmp_path, "h100-8gpu-largest-board-priority.yaml")
    original = load_config(fixture.old_profile)
    pending = fixture.root / "arena/pending.resume.json"
    _write_json(pending, {"game_states": [{"actions": [1, 2, 3]}]})
    paths = [
        pending,
        fixture.checkpoint,
        fixture.root / "learner/recovery.json",
        fixture.root / "run.json",
    ]
    before = {path: path.read_bytes() for path in paths}
    source_profile = fixture.old_profile
    source_commit = fixture.request.from_source_commit
    for index, enabled in enumerate((True, False)):
        target = yaml.safe_load(source_profile.read_text())
        target["arena"]["exact_clinch_termination"] = enabled
        fixture.candidate_profile.write_text(yaml.safe_dump(target))
        request = replace(
            fixture.request,
            old_profile=source_profile,
            from_source_commit=source_commit,
            to_source_commit=str(index + 2) * 40,
            target_profile_name=f"profile-clinch-{index}.yaml",
        )
        result = migration.migrate_continuous_profile(request, apply=True)
        target_profile = fixture.root / request.target_profile_name
        changed = load_config(target_profile)
        assert changed.arena.exact_clinch_termination is enabled
        assert evaluation_contract(changed.arena) == evaluation_contract(original.arena)
        assert {path: path.read_bytes() for path in paths} == before
        assert result["changes"] == [
            {
                "path": "arena.exact_clinch_termination",
                "from": not enabled,
                "to": enabled,
            }
        ]
        for section in ("game", "model", "train", "optimizer", "learner", "selfplay"):
            assert getattr(changed, section) == getattr(original, section)
        source_profile, source_commit = target_profile, request.to_source_commit


def test_disabled_clinch_preserves_existing_search_admission_fingerprint():
    import hashlib
    import json

    from deltreltrain.search_allocation_gate import canonical_config_sha256

    config = load_config(
        Path(__file__).parents[1] / "configs/h100-8gpu-largest-board-priority.yaml"
    )
    legacy = config.as_dict()
    del legacy["arena"]["exact_clinch_termination"]
    expected = hashlib.sha256(
        json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert canonical_config_sha256(config) == expected
    enabled = replace(
        config, arena=replace(config.arena, exact_clinch_termination=True)
    )
    assert canonical_config_sha256(enabled) != expected
