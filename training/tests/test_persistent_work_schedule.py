from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import fcntl
import json
import multiprocessing
import os

import pytest

from deltreltrain import cohort_work
from deltreltrain.cohort_work import (
    CompatibleWorkCoordinator,
    PersistentWorkSchedule,
    WorkBundle,
)


ROLES = {"candidate": 0.5, "champion": 0.25, "history": 0.25}
RINGS = {4: 0.05, 6: 0.05, 8: 0.05, 10: 0.85}


def schedule(path, **changes):
    return PersistentWorkSchedule(
        path, namespace="run-test:family-test", seed=17, **changes
    )


def choices(owner, count, *, rings=RINGS):
    return [owner.choose_role_and_ring(ROLES, rings, units=64) for _ in range(count)]


def process_choices(path, count):
    return choices(schedule(path), count)


def test_first_fleet_assignments_cover_every_board_and_repay_exact_target_debt(
    tmp_path,
):
    owner = schedule(tmp_path / "schedule.json")
    selected = choices(owner, 80)
    assert {ring for _, ring in selected[:4]} == set(RINGS)
    assert Counter(ring for _, ring in selected[:20]) == {4: 1, 6: 1, 8: 1, 10: 17}
    assert Counter(selected) == {
        (role, ring): round(80 * rw * bw)
        for role, rw in ROLES.items()
        for ring, bw in RINGS.items()
    }
    payload = owner.snapshot()
    assert payload["transactions"] == 80
    ring_state = next(
        row for row in payload["scopes"].values() if row["coverage_initialized"]
    )
    assert ring_state["coverage_pending"] == []
    assert sum(ring_state["assigned_units"].values()) == 80 * 64
    assert all(abs(credit) <= 3 * 64 for credit in ring_state["credits"].values())


@pytest.mark.parametrize("restart_after", [1, 2, 3, 7, 21])
def test_restart_preserves_pending_coverage_and_exact_future_sequence(
    tmp_path, restart_after
):
    reference = choices(schedule(tmp_path / "reference.json"), 80)
    path = tmp_path / "resumed.json"
    first = schedule(path)
    selected = choices(first, restart_after)
    before = first.snapshot()
    first.close()
    resumed = schedule(path)
    assert resumed.snapshot() == before
    selected.extend(choices(resumed, 80 - restart_after))
    assert selected == reference


def test_independent_processes_share_one_credit_ledger_without_lost_updates(tmp_path):
    path = tmp_path / "fleet.json"
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        groups = list(pool.map(process_choices, [path] * 4, [40] * 4))
    counts = Counter(item for group in groups for item in group)
    assert counts == {
        (role, ring): round(160 * rw * bw)
        for role, rw in ROLES.items()
        for ring, bw in RINGS.items()
    }
    assert schedule(path).snapshot()["transactions"] == 160


def test_coverage_configuration_only_initializes_new_state_and_never_reprimes_it(
    tmp_path,
):
    path = tmp_path / "coverage.json"
    first = choices(schedule(path), 2)
    continued = choices(schedule(path, coverage_first=False), 78)
    assert first + continued == choices(schedule(tmp_path / "reference.json"), 80)
    assert schedule(path, coverage_first=False).snapshot()["coverage_first"] is True
    cold = tmp_path / "no-coverage.json"
    first = choices(schedule(cold, coverage_first=False), 2)
    continued = choices(schedule(cold, coverage_first=True), 78)
    assert first + continued == choices(
        schedule(tmp_path / "reference-no-coverage.json", coverage_first=False), 80
    )


def test_tiny_same_support_weight_changes_retain_credit_instead_of_starving_small_boards(
    tmp_path,
):
    owner = schedule(tmp_path / "dynamic.json")
    selected = []
    for index in range(400):
        epsilon = (index + 1) * 1e-10
        selected.append(
            owner.choose_role_and_ring(
                ROLES,
                {4: 0.05 + epsilon, 6: 0.05, 8: 0.05, 10: 0.85 - epsilon},
                units=64,
            )
        )
    counts = Counter(ring for _, ring in selected)
    assert all(abs(counts[ring] - 400 * weight) <= 1 for ring, weight in RINGS.items())
    state = owner.snapshot()
    ring_state = next(
        row for row in state["scopes"].values() if row["coverage_initialized"]
    )
    assert ring_state["segment"] == 399
    assert ring_state["coverage_pending"] == []
    assert (tmp_path / "dynamic.json").stat().st_size < 20_000


def test_support_changes_drop_obsolete_credit_without_repeating_initial_coverage(
    tmp_path,
):
    owner = schedule(tmp_path / "support.json")
    choices(owner, 2)
    selected = choices(owner, 12, rings={4: 0.0, 10: 1.0})
    assert all(ring == 10 for _, ring in selected)
    assert Counter(role for role, _ in selected) == {
        "candidate": 6,
        "champion": 3,
        "history": 3,
    }
    state = next(
        row for row in owner.snapshot()["scopes"].values() if len(row["weights"]) == 1
    )
    assert state["coverage_pending"] == [] and not state["coverage_initialized"]


def test_ring_and_role_choices_are_one_atomic_transaction(tmp_path):
    path = tmp_path / "atomic.json"
    owner = schedule(path)
    choices(owner, 1)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="positive finite mass"):
        owner.choose_role_and_ring({"candidate": 0.0}, RINGS, units=64)
    assert path.read_bytes() == before
    reference = choices(schedule(tmp_path / "reference.json"), 2)
    assert choices(owner, 1) == reference[1:]


def test_failed_atomic_write_does_not_advance_credit_in_memory_or_on_disk(
    tmp_path, monkeypatch
):
    path = tmp_path / "write.json"
    owner = schedule(path)
    choices(owner, 1)
    before = path.read_bytes()
    with monkeypatch.context() as context:

        def fail(*_args, **_kwargs):
            raise OSError("simulated write failure")

        context.setattr(cohort_work, "atomic_json", fail)
        with pytest.raises(OSError, match="simulated write failure"):
            choices(owner, 1)
    assert path.read_bytes() == before
    assert choices(owner, 1) == choices(schedule(tmp_path / "reference.json"), 2)[1:]


@pytest.mark.parametrize(
    "mutation",
    [
        "namespace",
        "seed",
        "coverage_first",
        "schema_version",
        "nan_credit",
        "forged_credit",
        "negative_units",
        "excess_debt",
    ],
)
def test_corrupt_or_wrong_authority_state_fails_closed_without_reset(
    tmp_path, mutation
):
    path = tmp_path / "corrupt.json"
    owner = schedule(path)
    choices(owner, 1)
    state = json.loads(path.read_text())
    row = next(iter(state["scopes"].values()))
    key = next(iter(row["credits"]))
    if mutation == "namespace":
        state[mutation] = "another-run"
    elif mutation == "seed":
        state[mutation] = 18
    elif mutation == "coverage_first":
        state[mutation] = False
    elif mutation == "schema_version":
        state[mutation] = True
    elif mutation == "nan_credit":
        row["credits"][key] = float("nan")
    elif mutation == "forged_credit":
        row["credits"][key] += 1.0
    elif mutation == "negative_units":
        row["assigned_units"][key] = -1
    else:
        row["credits"][key] = 1e10
    path.write_text(json.dumps(state))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        choices(owner, 1)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "weights,units",
    [
        ({}, 64),
        ({"a": 0.0}, 64),
        ({"a": True}, 64),
        ({"a": float("inf")}, 64),
        ({"a": -1.0}, 64),
        ({"a": 1.0}, True),
        ({"a": 1.0}, 0),
        ({"a": 1.0}, 1_000_001),
    ],
)
def test_invalid_requests_never_create_or_mutate_state(tmp_path, weights, units):
    path = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        schedule(path).choose("test", weights, units=units)
    assert not path.exists()


def test_lock_wait_is_bounded_and_state_symlinks_are_rejected(tmp_path):
    path = tmp_path / "blocked.json"
    descriptor = os.open(str(path) + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        with pytest.raises(TimeoutError, match="deadline"):
            choices(schedule(path, lock_timeout_seconds=0.02), 1)
    finally:
        os.close(descriptor)
    target = tmp_path / "target.json"
    target.write_text("{}")
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        choices(schedule(path), 1)
    assert target.read_text() == "{}"


def test_modes_and_severity_are_shared_across_coordinators_and_restarts(tmp_path):
    path = tmp_path / "conditional.json"
    a = CompatibleWorkCoordinator(
        cohort_count=4, bundle_cohorts=1, seed=1, schedule=schedule(path)
    )
    b = CompatibleWorkCoordinator(
        cohort_count=4, bundle_cohorts=1, seed=2, schedule=schedule(path)
    )
    modes = [
        owner.choose_mode(
            ("candidate", 10),
            dict.fromkeys(("a", "b", "c", "d", "e", "f"), 1.0),
            units=64,
        )
        for owner in (a, b) * 6
    ]
    severities = [
        owner.choose_severity(("candidate", 10, "classic-handicap"), 2, 9, units=64)
        for owner in (a, b) * 8
    ]
    assert Counter(modes) == dict.fromkeys(("a", "b", "c", "d", "e", "f"), 2)
    assert Counter(severities) == dict.fromkeys(range(2, 10), 2)
    assert schedule(path).snapshot()["transactions"] == 28


def test_small_bundles_keep_all_producers_independent_and_release_reservations(
    tmp_path,
):
    owner = CompatibleWorkCoordinator(
        cohort_count=4,
        bundle_cohorts=1,
        seed=17,
        schedule=schedule(tmp_path / "leases.json"),
    )
    released = []

    def factory(coordinator):
        role, ring = coordinator.choose_role_and_ring(ROLES, RINGS, units=64)
        # Scheduling state is unlocked before model resources are loaded.
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert (
                pool.submit(schedule(tmp_path / "leases.json").snapshot).result(
                    timeout=2
                )["transactions"]
                > 0
            )
        metadata = {
            "requested_model_role": role,
            "model_role": role,
            "ring": ring,
            "mode_category": "classic-standard",
            "games": 64,
        }
        return WorkBundle(metadata, lambda: object(), lambda: released.append(ring))

    leases = [owner.acquire(factory) for _ in range(4)]
    assert {lease.metadata["ring"] for lease in leases} == set(RINGS)
    assert len(released) == 4 and len({lease.bundle_id for lease in leases}) == 4
    assert owner.metrics_snapshot()["outstanding_promised_games"] == 256
    for lease in leases:
        assert (
            owner.record_outcome(
                lease,
                requested=64,
                started=64,
                completed=64,
                dropped=0,
                cancelling=False,
            )
            == 0
        )
    assert owner.metrics_snapshot()["outstanding_promised_games"] == 0
    assert owner.metrics_snapshot()["producer_cohorts"] == 4
    owner.close()


@pytest.mark.parametrize("bundle_cohorts", [0, 5, True])
def test_bundle_size_is_strict_and_bounded(bundle_cohorts):
    with pytest.raises(ValueError, match="bundle cohorts"):
        CompatibleWorkCoordinator(
            cohort_count=4, seed=17, bundle_cohorts=bundle_cohorts
        )
