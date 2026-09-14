from dataclasses import replace
from pathlib import Path

import pytest

from scripts.prepare_pie_training_profile import prepare_profile
from scripts.strength_efficiency_report import _active_strength_config
from startrain.balanced_evaluation import evaluation_contract
from startrain.config import ArenaConfig, ConfigError, load_config
from startrain.config_compatibility import without_search_execution_defaults
from startrain.pie_promotion import (
    pie_promotion_config,
    validate_pie_promotion_transition,
)


CONFIGS = Path(__file__).parents[1] / "configs"


def test_promotion_opt_in_preserves_every_training_and_execution_setting():
    before = load_config(CONFIGS / "h100-8gpu-pie-even.yaml")
    after = pie_promotion_config(before)
    assert replace(after, arena=before.arena) == before
    assert replace(after.arena, allocation_policy="equal_cells") == before.arena
    assert after.arena.max_pairs_per_ring == 40
    assert after.arena.minimum_pairs_per_ring == 4
    assert "allocation_policy" not in before.as_dict()["arena"]
    assert after.as_dict()["arena"]["allocation_policy"] == "adaptive_pie"
    assert pie_promotion_config(after) == after
    validate_pie_promotion_transition(before, after)
    with pytest.raises(ValueError, match="non-allocation"):
        validate_pie_promotion_transition(
            before, replace(after, arena=replace(after.arena, simulations=128))
        )
    with pytest.raises(ValueError, match="equal_cells source"):
        validate_pie_promotion_transition(after, after)
    with pytest.raises(ValueError, match="existing ring10_pie"):
        pie_promotion_config(
            load_config(CONFIGS / "h100-8gpu-largest-board-priority.yaml")
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"allocation_policy": "unknown"},
        {"variant_policy": "legacy_six"},
        {"rings": (4, 10)},
        {"minimum_pairs_per_ring": 5},
        {"minimum_pairs_per_ring": True},
        {"max_pairs_per_ring": 3},
    ],
)
def test_adaptive_allocation_rejects_incompatible_or_uncovered_initial_check(changes):
    values = dict(
        balanced_cells=True,
        variant_policy="pie_even",
        allocation_policy="adaptive_pie",
        rings=(10,),
        pairs_per_ring=4,
        minimum_pairs_per_ring=4,
        max_pairs_per_ring=40,
    )
    with pytest.raises(ConfigError, match="allocation|adaptive_pie"):
        ArenaConfig(**{**values, **changes})


def test_preparer_has_explicit_adaptive_opt_in_and_atomic_no_overwrite(tmp_path):
    source = CONFIGS / "h100-8gpu-pie-even.yaml"
    old_bytes = source.read_bytes()
    output = tmp_path / "adaptive.yaml"
    report = prepare_profile(source, output, adaptive_promotion=True)
    assert report["deployment_performed"] is False
    assert report["promotion_allocation"] == "adaptive_pie"
    assert load_config(output) == pie_promotion_config(load_config(source))
    target_bytes = output.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_profile(source, output, adaptive_promotion=True)
    assert output.read_bytes() == target_bytes
    assert source.read_bytes() == old_bytes


def test_compatibility_omits_only_equal_allocation_default():
    assert without_search_execution_defaults(
        {"arena": {"allocation_policy": "equal_cells"}}
    ) == {"arena": {}}
    enabled = {"arena": {"allocation_policy": "adaptive_pie"}}
    assert without_search_execution_defaults(enabled) == enabled


def test_strength_report_keeps_original_historical_contract(tmp_path):
    source = CONFIGS / "h100-8gpu-pie-even.yaml"
    target = tmp_path / "adaptive.yaml"
    prepare_profile(source, target, adaptive_promotion=True)
    original, _ = _active_strength_config(tmp_path, source)
    adaptive, _ = _active_strength_config(tmp_path, target)
    assert original is not None and adaptive is not None
    assert adaptive.allocation_policy == "equal_cells"
    assert evaluation_contract(original) == evaluation_contract(adaptive)
