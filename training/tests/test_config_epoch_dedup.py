"""Guard exact historical config authority while pruning duplicate epoch work."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from deltreltrain import config_compatibility as epochs
from deltreltrain.config import ActorInferenceConfig, load_config
from deltreltrain.search_options import SearchExecutionConfig


# Frozen parent algorithm: every eager branch, traversal order, and final dedup
# are retained. It shares only the unchanged individual omission adapters.
def _parent_epoch_payloads(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Preserve existing guards across additive performance/scheduling releases.

    The pre-broadcast representation receives the same scheduling, pause and
    efficiency omissions as the current one. Older independent guards can
    operate on every representation without losing any previously accepted
    epoch or searching arbitrary subsets of newly added fields.
    """

    current = epochs.deepcopy(dict(payload))
    inference_representations = [current]
    pre_inference_execution = epochs.without_inference_execution_defaults(current)
    if pre_inference_execution != current:
        inference_representations.append(pre_inference_execution)
    execution_representations = []
    for representation in inference_representations:
        execution_representations.append(representation)
        pre_execution = epochs.without_search_execution_defaults(representation)
        if pre_execution != representation:
            execution_representations.append(pre_execution)
    budget_representations = []
    for representation in execution_representations:
        budget_representations.append(representation)
        pre_budget = epochs.without_cohort_search_budget_defaults(representation)
        if pre_budget != representation:
            budget_representations.append(pre_budget)
    pipeline_representations = []
    for representation in budget_representations:
        pipeline_representations.append(representation)
        pre_pipeline = epochs.without_selfplay_pipeline_defaults(representation)
        if pre_pipeline != representation:
            pipeline_representations.append(pre_pipeline)
    representations = []
    for representation in pipeline_representations:
        representations.append(representation)
        pre_clipping = epochs.without_gradient_clipping_defaults(representation)
        if pre_clipping != representation:
            representations.append(pre_clipping)
    sources: list[dict[str, Any]] = []
    for representation in representations:
        sources.append(representation)
        pre_broadcast = epochs.without_broadcast_topology_default(representation)
        if pre_broadcast != representation:
            sources.append(pre_broadcast)
    variants: list[dict[str, Any]] = []
    for source in sources:
        pre_session = epochs.without_evaluation_session_defaults(source)
        previous = (
            source,
            epochs.without_efficiency_defaults(source),
            pre_session,
            epochs.without_efficiency_defaults(pre_session),
        )
        variants.extend(previous)
        if epochs.without_pause_strategy_default(source) != source:
            variants.extend(
                epochs.without_pause_strategy_default(row) for row in previous
            )
    # These two options share one release epoch. Preserve the representation
    # before both additions without inventing arbitrary subset combinations.
    # Enabled values and untyped lookalikes remain in every hash.
    for variant in tuple(variants):
        previous_training = epochs.without_training_execution_defaults(variant)
        if previous_training != variant:
            variants.append(previous_training)
    for variant in tuple(variants):
        previous_fresh_data = epochs.without_fresh_data_defaults(variant)
        if previous_fresh_data != variant:
            variants.append(previous_fresh_data)
    for variant in tuple(variants):
        previous_clinch = epochs.without_arena_clinch_default(variant)
        if previous_clinch != variant:
            variants.append(previous_clinch)
    # Both scheduling additions belong to one release, not two independently
    # shipped epochs. Preserve old hashes without another Cartesian dimension.
    for variant in tuple(variants):
        previous_scheduling = epochs.without_efficiency_program_defaults(variant)
        if previous_scheduling != variant:
            variants.append(previous_scheduling)
    # Several historical omission paths converge on the same representation.
    # Downstream provenance consumers must not hash every duplicate again.
    unique = {
        json.dumps(item, sort_keys=True, separators=(",", ":")): item
        for item in variants
    }
    return tuple(unique.values())


def _default_payload():
    return {
        "selfplay": {
            "search_execution": asdict(SearchExecutionConfig()),
            "stream_completed_games": False,
            "rolling_game_slots": False,
            "seed_contract": "cohort-v1",
            "cohort_search_budgets": False,
            "pie_even_training": False,
            "preserve_interrupted_policy": False,
            "policy_publication": {
                "enabled": False,
                "first_decisions": 8,
                "interval_decisions": 32,
            },
        },
        "train": {
            "gradient_clipping": {
                "mode": "global",
                "beta": 0.99,
                "multiplier": 1.04,
                "warmup_steps": 100,
            },
            "gradient_diagnostics": False,
            "share_homogeneous_geometry": False,
        },
        "learner": {"replay_refresh_seconds": 300.0},
        "arena": {
            "search_execution": asdict(SearchExecutionConfig()),
            "variant_policy": "legacy_six",
            "allocation_policy": "equal_cells",
            "exact_clinch_termination": False,
        },
        "orchestration": {
            "cpu_actors": (),
            "gpus": [
                {
                    "actor_pipeline": None,
                    "actor_cohorts": 1,
                    "native_threads": None,
                    "blas_threads": None,
                }
            ],
            "promotion": {
                "pause_strategy": "terminate",
                "session_seconds": 300.0,
                "cpu_affinity": None,
            },
            "historical_evaluation": {
                "session_seconds": 300.0,
                "cooldown_seconds": 1800.0,
                "measurement_service_fraction": 0.0,
                "measurement_max_wait_seconds": 3600.0,
            },
            "model_refresh": {
                "inference": asdict(ActorInferenceConfig()),
                "compatible_cohort_work": False,
                "history_horizon_enabled": False,
                "history_horizon_initial_seconds": 3600.0,
                "work_scheduling": {
                    "enabled": False,
                    "games_per_lease": None,
                    "coverage_first": True,
                },
            },
        },
        # Unknown types/keys are authoritative; none of the adapters may erase
        # or JSON-normalize them while pruning convergent omission paths.
        "future": {
            "values": [False, 0, 0.0, 1, 1.0, "false", None, (1, 2)],
            "mutable": {"values": [1, 2]},
        },
    }


def _payload(kind):
    if kind in ("throughput", "production"):
        profile = (
            "h100-8gpu-throughput.yaml"
            if kind == "throughput"
            else "h100-8gpu-pie-even.yaml"
        )
        return load_config(Path(__file__).parents[1] / "configs" / profile).as_dict()
    payload = _default_payload()
    inference = payload["orchestration"]["model_refresh"]["inference"]
    if kind == "enabled":
        inference.update(
            cuda_graphs=True,
            preserve_broadcast_topology=True,
            compact_inference_gather=True,
            small_batch_graph_buckets=True,
            shared_batching=True,
        )
        payload["selfplay"].update(
            cohort_search_budgets=True,
            preserve_interrupted_policy=True,
            stream_completed_games=True,
        )
        payload["train"].update(
            share_homogeneous_geometry=True, gradient_diagnostics=True
        )
        payload["arena"]["exact_clinch_termination"] = True
        payload["orchestration"]["promotion"].update(
            pause_strategy="suspend", session_seconds=600.0
        )
        payload["orchestration"]["historical_evaluation"][
            "measurement_service_fraction"
        ] = 0.2
    elif kind == "unknown_and_partial":
        inference["future_enabled"] = {"policy": [1, 2]}
        payload["selfplay"]["search_execution"]["future_parameter"] = "preserve"
        del payload["arena"]["search_execution"]["full_budget"]
        payload["train"]["gradient_clipping"]["future_parameter"] = 17
        payload["selfplay"]["policy_publication"]["future_parameter"] = [False]
        del payload["orchestration"]["model_refresh"]["work_scheduling"][
            "coverage_first"
        ]
    elif kind == "type_lookalikes":
        inference.update(
            preserve_broadcast_topology=0,
            compact_inference_gather=0,
            small_batch_graph_buckets="false",
        )
        payload["train"].update(share_homogeneous_geometry=0, gradient_diagnostics=0)
        payload["selfplay"].update(
            cohort_search_budgets=0, preserve_interrupted_policy=0
        )
        payload["arena"]["exact_clinch_termination"] = 0
        payload["learner"]["replay_refresh_seconds"] = 300
        payload["orchestration"]["promotion"]["session_seconds"] = 300
        payload["orchestration"]["historical_evaluation"][
            "measurement_service_fraction"
        ] = 0
    elif kind == "list_representation":
        payload["orchestration"]["cpu_actors"] = []
        payload["future"]["values"].append([1, 2])
    elif kind == "already_legacy":
        payload = epochs.without_efficiency_defaults(
            epochs.without_selfplay_pipeline_defaults(payload)
        )
    return payload


def _encode(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _typed_tree(value):
    if isinstance(value, dict):
        return (
            type(value),
            tuple((_typed_tree(key), _typed_tree(item)) for key, item in value.items()),
        )
    if isinstance(value, (list, tuple)):
        return (type(value), tuple(_typed_tree(item) for item in value))
    return (type(value), value)


@pytest.mark.parametrize(
    "kind",
    [
        "defaults",
        "enabled",
        "unknown_and_partial",
        "type_lookalikes",
        "list_representation",
        "already_legacy",
        "throughput",
        "production",
    ],
)
def test_epoch_stage_dedup_preserves_ordered_parent_payloads_and_hashes(kind):
    payload = _payload(kind)
    original = deepcopy(payload)
    expected = _parent_epoch_payloads(payload)
    actual = epochs.compatible_config_epoch_payloads(payload)
    assert actual == expected
    expected_bytes = tuple(map(_encode, expected))
    actual_bytes = tuple(map(_encode, actual))
    assert actual_bytes == expected_bytes
    assert tuple(hashlib.sha256(row).hexdigest() for row in actual_bytes) == tuple(
        hashlib.sha256(row).hexdigest() for row in expected_bytes
    )
    assert len(set(actual_bytes)) == len(actual)
    assert tuple(map(_typed_tree, actual)) == tuple(map(_typed_tree, expected))
    assert _typed_tree(payload) == _typed_tree(original)


def _mutable_ids(value):
    result = {id(value)} if isinstance(value, (dict, list)) else set()
    if isinstance(value, dict):
        for child in value.values():
            result.update(_mutable_ids(child))
    elif isinstance(value, (tuple, list)):
        for child in value:
            result.update(_mutable_ids(child))
    return result


def test_returned_epochs_are_independent_of_input_each_other_and_later_calls():
    payload = _payload("unknown_and_partial")
    original = deepcopy(payload)
    actual = epochs.compatible_config_epoch_payloads(payload)
    before = tuple(map(_encode, actual))
    identities = _mutable_ids(payload)
    for row in actual:
        owned = _mutable_ids(row)
        assert not identities & owned
        identities.update(owned)
    actual[0]["future"]["mutable"]["values"].append(3)
    assert tuple(map(_encode, actual[1:])) == before[1:]
    assert _typed_tree(payload) == _typed_tree(original)
    again = epochs.compatible_config_epoch_payloads(payload)
    assert tuple(map(_encode, again)) == before
    assert not identities & set().union(*map(_mutable_ids, again))


def test_convergent_epochs_are_not_deepcopied_through_later_stages(monkeypatch):
    payload = _default_payload()
    original_copy = epochs.deepcopy
    calls = 0

    def counted_copy(value):
        nonlocal calls
        calls += 1
        return original_copy(value)

    monkeypatch.setattr(epochs, "deepcopy", counted_copy)
    expected = _parent_epoch_payloads(payload)
    eager_calls = calls
    calls = 0
    actual = epochs.compatible_config_epoch_payloads(payload)
    assert tuple(map(_encode, actual)) == tuple(map(_encode, expected))
    assert calls < eager_calls / 2


@pytest.mark.parametrize("value", [False, True, 0, 0.0, 1, 3600, 3600.0, "false", None])
@pytest.mark.parametrize(
    "section,name",
    [
        ("model_refresh", "history_horizon_enabled"),
        ("model_refresh", "history_horizon_initial_seconds"),
        ("historical_evaluation", "measurement_service_fraction"),
        ("historical_evaluation", "measurement_max_wait_seconds"),
    ],
)
def test_scheduling_noop_guard_matches_exact_public_adapter(value, section, name):
    payload = {"orchestration": {section: {name: value}}}
    adapted = epochs.without_efficiency_program_defaults(payload)
    assert epochs._has_efficiency_program_defaults(payload) == (adapted != payload)
    assert payload == {"orchestration": {section: {name: value}}}


def test_absent_scheduling_defaults_do_not_copy_every_final_epoch(monkeypatch):
    payload = epochs.without_efficiency_program_defaults(_default_payload())
    actual_adapter = epochs.without_efficiency_program_defaults
    expected = _parent_epoch_payloads(payload)

    def reject_noop_copy(source):
        assert actual_adapter(source) != source, "copied a provably unchanged epoch"
        return actual_adapter(source)

    monkeypatch.setattr(epochs, "without_efficiency_program_defaults", reject_noop_copy)
    actual = epochs.compatible_config_epoch_payloads(payload)
    assert tuple(map(_encode, actual)) == tuple(map(_encode, expected))


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"orchestration": None},
        {"orchestration": {"model_refresh": [], "historical_evaluation": "future"}},
        {"orchestration": {"model_refresh": {}, "historical_evaluation": {}}},
    ],
)
def test_scheduling_noop_guard_preserves_missing_and_unknown_shapes(payload):
    expected = deepcopy(payload)
    assert epochs._has_efficiency_program_defaults(payload) is False
    assert epochs.without_efficiency_program_defaults(payload) == expected
    assert payload == expected
