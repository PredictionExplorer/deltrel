from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts import migrate_continuous_profile as migration
from startrain.balanced_evaluation import evaluation_contract
from startrain.config import ConfigError, DataConfig, load_config
from startrain.config_compatibility import compatible_config_epoch_payloads
from startrain.selfplay import (
    RingSearchAllocation,
    SelfPlayActor,
    SelfPlayConfig,
    SelfPlayIdentity,
)


PROFILE = Path(__file__).parents[1] / "configs/h100-8gpu-largest-board-priority.yaml"
ALLOCATION = RingSearchAllocation(10, 8, 0.15, 0.06)


@pytest.mark.parametrize(
    "changes",
    [
        {"rings": True},
        {"rings": 5},
        {"rings": 10.0},
        {"fast_simulations": True},
        {"fast_simulations": 0},
        {"fast_simulations": 8.0},
        {"fast_simulations": 1_000_001},
        {"full_probability": False},
        {"full_probability": 0},
        {"full_probability": -0.1},
        {"full_probability": 1.1},
        {"full_probability": float("nan")},
        {"full_probability": float("inf")},
        {"full_probability": "0.15"},
        {"fast_policy_weight": True},
        {"fast_policy_weight": -0.01},
        {"fast_policy_weight": 1.1},
        {"fast_policy_weight": float("nan")},
        {"fast_policy_weight": None},
    ],
)
def test_allocation_fields_are_strict_and_bounded(changes):
    with pytest.raises(ValueError, match="allocation"):
        replace(ALLOCATION, **changes)


@pytest.mark.parametrize(
    "raw",
    [
        False,
        None,
        0,
        {},
        "",
        [False],
        [{"rings": 10}],
        [{**asdict(ALLOCATION), "full_simulations": 160}],
        [asdict(ALLOCATION)] * 2,
    ],
)
def test_yaml_rejects_malformed_and_duplicate_groups(tmp_path, raw):
    payload = yaml.safe_load(PROFILE.read_text())
    payload["selfplay"]["ring_search_allocations"] = raw
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ConfigError):
        load_config(path)


def test_direct_settings_require_immutable_typed_tuple_and_fast_cap_within_full():
    for raw in ([], False, None, (asdict(ALLOCATION),), (ALLOCATION, ALLOCATION)):
        with pytest.raises(ValueError):
            SelfPlayConfig(ring_search_allocations=raw)
    with pytest.raises(ValueError, match="cannot exceed"):
        SelfPlayConfig(full_simulations=4, ring_search_allocations=(ALLOCATION,))
    assert (
        replace(ALLOCATION, full_probability=1, fast_policy_weight=0).full_probability
        == 1.0
    )


def test_empty_group_preserves_canonical_authority_and_nonempty_never_disappears(
    tmp_path,
):
    base = load_config(PROFILE)
    old_payload = asdict(base)
    del old_payload["selfplay"]["ring_search_allocations"]
    assert base.as_dict() == old_payload
    old_hash = hashlib.sha256(
        migration._canonical_config_bytes(old_payload)
    ).hexdigest()
    assert migration.canonical_config_sha256(base) == old_hash
    raw = yaml.safe_load(PROFILE.read_text())
    raw["selfplay"]["ring_search_allocations"] = []
    path = tmp_path / "empty.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert load_config(path).as_dict() == old_payload
    enabled = replace(
        base, selfplay=replace(base.selfplay, ring_search_allocations=(ALLOCATION,))
    )
    path.write_text(yaml.safe_dump(enabled.as_dict()))
    assert load_config(path) == enabled
    assert migration.canonical_config_sha256(enabled) != old_hash
    for payload in compatible_config_epoch_payloads(enabled.as_dict()):
        assert payload["selfplay"]["ring_search_allocations"] == (asdict(ALLOCATION),)
    assert evaluation_contract(base.arena) == evaluation_contract(enabled.arena)


@pytest.mark.parametrize("ring", [4, 6, 8, 10])
def test_resolution_is_ring_local_idempotent_and_preserves_full_search_and_pda(ring):
    master = load_config(PROFILE).selfplay
    pilot = replace(master, ring_search_allocations=(ALLOCATION,))
    before = replace(master, rings=ring)
    configured = replace(pilot, rings=ring)
    resolved = configured.resolved_search_allocation()
    expected = (
        replace(
            configured,
            fast_simulations=8,
            full_probability=0.15,
            fast_probability=0.85,
            fast_policy_weight=0.06,
        )
        if ring == 10
        else configured
    )
    assert resolved == expected
    assert resolved.resolved_search_allocation() is resolved
    assert pilot.fast_simulations == master.fast_simulations
    assert pilot.full_probability == master.full_probability
    assert resolved.simulation_budget(full=True) == before.simulation_budget(full=True)
    assert resolved.considered_actions() == before.considered_actions()
    for pda in range(4):
        assert resolved.playout_budgets(
            simulations=resolved.simulation_budget(full=True), pda=pda
        ) == before.playout_budgets(
            simulations=before.simulation_budget(full=True), pda=pda
        )
    facts = resolved.search_allocation_facts()
    assert facts["ring_override"] is (ring == 10)
    assert facts["fast_simulations"] == (
        13 if ring == 10 else before.simulation_budget(full=False)
    )
    assert facts["full_probability"] == (
        0.15 if ring == 10 else before.full_probability
    )


@pytest.mark.native
@pytest.mark.parametrize("ring", [4, 10])
def test_direct_native_actor_matches_manually_resolved_caps_and_targets(ring):
    from test_search_execution_selfplay import DeterministicEvaluator
    from test_selfplay_streaming import Sink, sample_fingerprints

    native = pytest.importorskip("star_native")

    def states(rings, rows, **options):
        result = native.StateBatch(rings, rows, **options)
        for row in range(rows):
            placed = result.node_count - 5
            result.apply_many([row] * placed, list(range(placed)))
        return result

    module = SimpleNamespace(
        StateBatch=states,
        SearchBatch=native.SearchBatch,
        native_search_execution_version=native.native_search_execution_version,
    )
    base = SelfPlayConfig(
        rings=ring,
        games=2,
        batch_size=2,
        fast_simulations=4,
        full_simulations=8,
        simulation_ring_exponent=0,
        max_considered=4,
        record_fast_policy_targets=True,
        seed_contract="game-v1",
    )
    allocation = RingSearchAllocation(10, 1, 0.125, 0.053)
    override = replace(base, ring_search_allocations=(allocation,))
    manual = (
        replace(
            base,
            fast_simulations=1,
            full_probability=0.125,
            fast_probability=0.875,
            fast_policy_weight=0.053,
        )
        if ring == 10
        else base
    )
    outputs = []
    for config in (override, manual):
        sink = Sink()
        actor = SelfPlayActor(
            module,
            DeterministicEvaluator(),
            sink,
            config,
            SelfPlayIdentity("run", "family", "actor", 0),
        )
        actor.run()
        outputs.append(
            (
                sample_fingerprints(sink.samples),
                actor.metrics_snapshot().full_decisions,
                actor.metrics_snapshot().fast_decisions,
            )
        )
        assert actor.config.fast_simulations == manual.fast_simulations
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("workers", True),
        ("workers", 4.0),
        ("workers", -1),
        ("prefetch_factor", False),
        ("prefetch_factor", 2.0),
        ("prefetch_factor", 0),
    ],
)
def test_migratable_loader_values_are_strict_integers(field, value):
    with pytest.raises(ConfigError):
        DataConfig(**{field: value})


def test_pilot_and_execution_migration_is_reversible_without_training_state_changes(
    tmp_path,
):
    from test_continuous_profile_migration import _fixture, _write_json

    fixture = _fixture(tmp_path, PROFILE.name)
    original = load_config(fixture.old_profile)
    _write_json(
        fixture.root / "learner/utd-segment.json",
        {
            "schema_version": 1,
            "run_id": "continuous-test-run",
            "generation_family": "family-continuous-test",
            "target_updates_per_new_sample": original.learner.target_updates_per_new_sample,
            "baseline_examples_consumed": 1024,
            "baseline_committed_replay_samples": 2048,
            "created_ns": 5,
        },
    )
    paths = [
        fixture.root / name
        for name in (
            "run.json",
            "learner/recovery.json",
            "learner/champion.json",
            "learner/utd-segment.json",
        )
    ] + [fixture.checkpoint]
    before = {p: p.read_bytes() for p in paths}
    source, commit = fixture.old_profile, fixture.request.from_source_commit
    for index, active in enumerate((True, False)):
        raw = yaml.safe_load(source.read_text())
        raw["selfplay"]["ring_search_allocations"] = (
            [asdict(replace(ALLOCATION, full_probability=0.4))] if active else []
        )
        raw["selfplay"].setdefault("search_execution", {})["first_visit_batch_size"] = (
            4 if active else 1
        )
        raw["data"]["workers"] = 4 if active else original.data.workers
        raw["data"]["prefetch_factor"] = (
            original.data.prefetch_factor + 1
            if active
            else original.data.prefetch_factor
        )
        fixture.candidate_profile.write_text(yaml.safe_dump(raw))
        request = replace(
            fixture.request,
            old_profile=source,
            target_profile_name=f"profile-ring-allocation-{index}.yaml",
            from_source_commit=commit,
            to_source_commit=str(index + 2) * 40,
        )
        plan = migration.plan_migration(request)
        # Planning is read-only, including the immutable checkpoint and UTD segment.
        assert {p: p.read_bytes() for p in paths} == before
        migration.apply_migration(plan)
        changed = load_config(plan.target_profile)
        assert changed.model == original.model and changed.train == original.train
        assert changed.optimizer == original.optimizer and changed.loss == original.loss
        assert changed.learner == original.learner and changed.arena == original.arena
        assert changed.selfplay.full_simulations == original.selfplay.full_simulations
        assert {p: p.read_bytes() for p in paths} == before
        record = json.loads(
            (fixture.root / "continuous-migrations.jsonl").read_text().splitlines()[-1]
        )
        assert {row["path"] for row in record["changes"]} == {
            "selfplay.ring_search_allocations",
            "selfplay.search_execution.first_visit_batch_size",
            "data.workers",
            "data.prefetch_factor",
        }
        assert (
            "utd_segment" not in record
            and "evaluation_contract_transition" not in record
        )
        source, commit = plan.target_profile, request.to_source_commit


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("selfplay", "simulation_reference_rings", 4),
        ("selfplay", "simulation_ring_exponent", 2),
        ("selfplay", "full_probability", 0.4),
        ("selfplay", "max_considered", 8),
        ("data", "pin_memory", True),
    ],
)
def test_narrow_allowance_does_not_authorize_global_search_or_other_data_changes(
    tmp_path, section, field, value
):
    from test_continuous_profile_migration import _fixture

    fixture = _fixture(tmp_path, PROFILE.name)
    raw = yaml.safe_load(fixture.old_profile.read_text())
    if section == "data" and field == "pin_memory":
        value = not raw[section].get(field, False)
    raw[section][field] = value
    if field == "full_probability":
        raw[section]["fast_probability"] = 1 - value
    fixture.candidate_profile.write_text(yaml.safe_dump(raw))
    with pytest.raises(migration.MigrationError, match="immutable or unsupported"):
        migration.plan_migration(fixture.request)
