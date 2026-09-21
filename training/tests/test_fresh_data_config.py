from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import yaml

from scripts import migrate_continuous_profile as migration
from deltreltrain import actor as actors
from deltreltrain.balanced_evaluation import evaluation_contract
from deltreltrain.config import (
    ActorWorkSchedulingConfig,
    ConfigError,
    GPUWorkerConfig,
    LearnerConfig,
    ModelRefreshConfig,
    RingMixtureConfig,
    RingWeightStage,
    load_config,
)
from deltreltrain.config_compatibility import (
    compatible_config_epoch_payloads,
    without_fresh_data_defaults,
)
from deltreltrain.cohort_work import CompatibleWorkCoordinator, PersistentWorkSchedule
from deltreltrain.selfplay import PolicyPublicationConfig
from test_cohort_work import fake_actor
from test_continuous_profile_migration import _fixture, _write_json


def profile():
    base = load_config(
        Path(__file__).parents[1] / "configs/h100-8gpu-largest-board-priority.yaml"
    )
    return replace(
        base,
        orchestration=replace(
            base.orchestration,
            model_refresh=replace(
                base.orchestration.model_refresh, compatible_cohort_work=True
            ),
        ),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"enabled": 1},
        {"coverage_first": 0},
        {"games_per_lease": True},
        {"games_per_lease": 0},
        {"games_per_lease": -1},
        {"games_per_lease": 64.0},
        {"games_per_lease": 1_000_001},
    ],
)
def test_work_scheduling_fields_are_strict(changes):
    assert ActorWorkSchedulingConfig() == ActorWorkSchedulingConfig(False, None, True)
    with pytest.raises(ConfigError):
        ActorWorkSchedulingConfig(**changes)
    with pytest.raises(ConfigError, match="ActorWorkSchedulingConfig"):
        ModelRefreshConfig(work_scheduling={"enabled": False})


@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan"), None, "30"])
def test_replay_refresh_requires_positive_finite_duration(value):
    assert LearnerConfig().replay_refresh_seconds == 300.0
    with pytest.raises(ConfigError, match="replay_refresh_seconds"):
        LearnerConfig(replay_refresh_seconds=value)


@pytest.mark.parametrize(
    "fault", ["no_gpu", "non_cuda", "single_cohort", "incompatible", "small_quantum"]
)
def test_work_scheduling_rejects_incompatible_actor_topology(fault):
    base = profile()
    work = ActorWorkSchedulingConfig(enabled=True)
    orchestration = base.orchestration
    changes = {}
    if fault == "no_gpu":
        changes["gpus"] = ()
    elif fault == "non_cuda":
        changes["device"] = "mps"
    elif fault == "single_cohort":
        changes["gpus"] = tuple(replace(g, actor_cohorts=1) for g in orchestration.gpus)
    elif fault == "incompatible":
        changes["model_refresh"] = replace(
            orchestration.model_refresh,
            compatible_cohort_work=False,
            work_scheduling=work,
        )
    else:
        batch = next(
            g.actor_batch_size for g in orchestration.gpus if g.role == "actor"
        )
        work = replace(work, games_per_lease=batch - 1)
    changes.setdefault(
        "model_refresh", replace(orchestration.model_refresh, work_scheduling=work)
    )
    with pytest.raises(ConfigError):
        replace(orchestration, **changes)


def test_enabled_nested_settings_roundtrip_and_preserve_all_other_controls(tmp_path):
    base = profile()
    enabled = replace(
        base,
        selfplay=replace(
            base.selfplay,
            policy_publication=PolicyPublicationConfig(
                enabled=True, first_decisions=4, interval_decisions=16
            ),
        ),
        learner=replace(base.learner, replay_refresh_seconds=30.0),
        orchestration=replace(
            base.orchestration,
            model_refresh=replace(
                base.orchestration.model_refresh,
                work_scheduling=ActorWorkSchedulingConfig(enabled=True),
            ),
        ),
    )
    path = tmp_path / "enabled.yaml"
    path.write_text(yaml.safe_dump(enabled.as_dict()))
    assert load_config(path) == enabled
    assert enabled.model == base.model and enabled.optimizer == base.optimizer
    assert enabled.train == base.train and enabled.loss == base.loss
    assert replace(enabled.learner, replay_refresh_seconds=300.0) == base.learner
    assert enabled.selfplay.full_simulations == base.selfplay.full_simulations
    assert enabled.selfplay.fast_simulations == base.selfplay.fast_simulations
    raw = enabled.as_dict()
    raw["selfplay"]["record_fast_policy_targets"] = False
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="fast policy targets"):
        load_config(path)
    raw = enabled.as_dict()
    raw["orchestration"]["model_refresh"]["work_scheduling"]["unknown"] = True
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="unknown"):
        load_config(path)


def defaults_payload():
    return {
        "selfplay": {"policy_publication": asdict(PolicyPublicationConfig())},
        "learner": {"replay_refresh_seconds": 300.0},
        "orchestration": {
            "model_refresh": {"work_scheduling": asdict(ActorWorkSchedulingConfig())}
        },
    }


def test_all_three_defaults_share_one_legacy_epoch_without_subset_explosion():
    payload = defaults_payload()
    previous = {"selfplay": {}, "learner": {}, "orchestration": {"model_refresh": {}}}
    assert without_fresh_data_defaults(payload) == previous
    assert {
        json.dumps(row, sort_keys=True)
        for row in compatible_config_epoch_payloads(payload)
    } == {json.dumps(payload, sort_keys=True), json.dumps(previous, sort_keys=True)}


@pytest.mark.parametrize(
    "path,value",
    [
        (("selfplay", "policy_publication", "enabled"), True),
        (("selfplay", "policy_publication", "first_decisions"), 4),
        (("selfplay", "policy_publication", "first_decisions"), 8.0),
        (("selfplay", "policy_publication", "unknown"), True),
        (("orchestration", "model_refresh", "work_scheduling", "enabled"), True),
        (("orchestration", "model_refresh", "work_scheduling", "games_per_lease"), 128),
        (
            ("orchestration", "model_refresh", "work_scheduling", "coverage_first"),
            False,
        ),
        (("learner", "replay_refresh_seconds"), 30.0),
        (("learner", "replay_refresh_seconds"), 300),
    ],
)
def test_nondefaults_enabled_values_and_typed_lookalikes_remain_authoritative(
    path, value
):
    payload = defaults_payload()
    parent = payload
    for name in path[:-1]:
        parent = parent[name]
    parent[path[-1]] = value
    original = deepcopy(payload)
    for representation in compatible_config_epoch_payloads(payload):
        actual = representation
        for name in path:
            actual = actual[name]
        assert actual == value and type(actual) is type(value)
    assert payload == original


def test_fresh_data_profile_migration_is_reversible_and_preserves_state_and_budgets(
    tmp_path,
):
    fixture = _fixture(tmp_path, "h100-8gpu-largest-board-priority.yaml")
    raw = yaml.safe_load(fixture.old_profile.read_text())
    raw["orchestration"]["model_refresh"]["compatible_cohort_work"] = True
    fixture.old_profile.chmod(0o644)
    fixture.old_profile.write_text(yaml.safe_dump(raw))
    fixture.old_profile.chmod(0o444)
    digest = hashlib.sha256(fixture.old_profile.read_bytes()).hexdigest()
    (fixture.root / "profile.sha256").write_text(f"{digest}  {fixture.old_profile}\n")
    original = load_config(fixture.old_profile)
    _write_json(
        fixture.root / "arena/promotion-status.json",
        {"terminal": False, "candidate_step": 90},
    )
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
            "arena/promotion-status.json",
        )
    ] + [fixture.checkpoint]
    before = {p: p.read_bytes() for p in paths}
    source, commit = fixture.old_profile, fixture.request.from_source_commit
    for index, active in enumerate((True, False)):
        raw = yaml.safe_load(source.read_text())
        raw["selfplay"]["policy_publication"] = {"enabled": active}
        raw["learner"]["replay_refresh_seconds"] = 30.0 if active else 300.0
        raw["orchestration"]["model_refresh"]["work_scheduling"] = {"enabled": active}
        fixture.candidate_profile.write_text(yaml.safe_dump(raw))
        request = replace(
            fixture.request,
            old_profile=source,
            target_profile_name=f"profile-fresh-{index}.yaml",
            from_source_commit=commit,
            to_source_commit=str(index + 2) * 40,
        )
        plan = migration.plan_migration(request)
        migration.apply_migration(plan)
        changed = load_config(plan.target_profile)
        assert (
            replace(changed.selfplay, policy_publication=PolicyPublicationConfig())
            == original.selfplay
        )
        assert (
            replace(changed.learner, replay_refresh_seconds=300.0) == original.learner
        )
        assert changed.model == original.model and changed.train == original.train
        assert changed.optimizer == original.optimizer and changed.loss == original.loss
        assert evaluation_contract(changed.arena) == evaluation_contract(original.arena)
        assert {p: p.read_bytes() for p in paths} == before
        record = json.loads(
            (fixture.root / "continuous-migrations.jsonl").read_text().splitlines()[-1]
        )
        assert {row["path"] for row in record["changes"]} == {
            "selfplay.policy_publication.enabled",
            "learner.replay_refresh_seconds",
            "orchestration.model_refresh.work_scheduling.enabled",
        }
        assert (
            "utd_segment" not in record
            and "evaluation_contract_transition" not in record
        )
        source, commit = plan.target_profile, request.to_source_commit


def enabled_actor(tmp_path, monkeypatch, gpu_id, *, games=None):
    actor, store, manifests, loaded = fake_actor(tmp_path, monkeypatch)
    gpu = GPUWorkerConfig(
        gpu_id=gpu_id, role="actor", cpu_threads=4, actor_batch_size=64, actor_cohorts=4
    )
    actor.gpu = gpu
    actor.actor_id = f"actor-gpu-{gpu_id}"
    actor.games_per_batch = 128
    refresh = replace(
        actor.experiment.orchestration.model_refresh,
        compatible_cohort_work=True,
        work_scheduling=ActorWorkSchedulingConfig(enabled=True, games_per_lease=games),
    )
    actor.experiment = replace(
        actor.experiment,
        orchestration=replace(
            actor.experiment.orchestration,
            gpus=(gpu,),
            model_refresh=refresh,
            ring_mixture=RingMixtureConfig(
                step_weights=(RingWeightStage(0, (0.05, 0.05, 0.05, 0.85)),)
            ),
        ),
    )
    actor.work_coordinator = CompatibleWorkCoordinator(
        cohort_count=4,
        bundle_cohorts=1,
        seed=gpu_id,
        schedule=PersistentWorkSchedule(
            tmp_path / "status/work-schedule.json",
            namespace="run:family",
            seed=actor.experiment.selfplay.seed,
        ),
    )
    return actor, store, manifests, loaded


@pytest.mark.parametrize("override,expected", [(None, 128), (64, 64)])
def test_two_actor_factories_share_ring_coverage_and_preserve_effective_game_quanta(
    tmp_path, monkeypatch, override, expected
):
    a, store, _, _ = enabled_actor(tmp_path, monkeypatch, 1, games=override)
    b, _, _, _ = enabled_actor(tmp_path, monkeypatch, 2, games=override)
    leases = []
    try:
        for actor in (a, b, a, b):
            lease = actor._acquire_cohort_work(store)
            leases.append(lease)
            lease.resource[0].release()
        assert {lease.metadata["ring"] for lease in leases} == {4, 6, 8, 10}
        assert [lease.metadata["games"] for lease in leases] == [expected] * 4
        assert all(lease.metadata["model_role"] == "candidate" for lease in leases)
        assert (
            a.work_coordinator.bundle_cohorts == b.work_coordinator.bundle_cohorts == 1
        )
        state = a.work_coordinator.schedule.snapshot()
        ring = next(
            row for row in state["scopes"].values() if row["coverage_initialized"]
        )
        assert sum(ring["assigned_units"].values()) == expected * 4
    finally:
        for actor in (a, b):
            actor.work_coordinator.close()
            actor.registry.close()


def test_shared_actor_runner_uses_one_persistent_namespace_and_keeps_all_producers(
    tmp_path, monkeypatch
):
    actor, _, _, _ = enabled_actor(tmp_path, monkeypatch, 1)
    captured = []

    class Child:
        def __init__(self, **kwargs):
            captured.append(kwargs)
            self.actor_id = kwargs["actor_id"]
            self.scheduler = SimpleNamespace(random=random.Random())
            self.model_random = random.Random()
            self.heartbeat = SimpleNamespace(path=Path(kwargs["heartbeat_path"]))

        def run(self, **_kwargs):
            return 0

    monkeypatch.setattr(actors, "ActorSupervisor", Child)
    assert actor._run_cohorts(stop_requested=lambda: False) == 0
    assert len(captured) == 4
    coordinator = captured[0]["work_coordinator"]
    assert all(item["work_coordinator"] is coordinator for item in captured)
    assert coordinator.cohort_count == 4 and coordinator.bundle_cohorts == 1
    assert coordinator.schedule.path == tmp_path / "status/work-schedule.json"
    assert coordinator.schedule.namespace == "run:family"
    assert coordinator.schedule.seed == actor.experiment.selfplay.seed
    assert all(item["games_per_batch"] == 128 for item in captured)
    actor.work_coordinator.close()
    actor.registry.close()
