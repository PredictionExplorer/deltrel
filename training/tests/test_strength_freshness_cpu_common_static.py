"""Legacy capture composition with real helpers and explicit memory-only I/O."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import runpy
from typing import cast

import pytest

from scripts import strength_freshness_cpu_collector as c
from scripts import strength_freshness_cpu_collect_identity as identity
from scripts import strength_freshness_cpu_collect_support as support
from scripts import strength_freshness_cpu_readonly as readonly

fixtures = runpy.run_path(
    str(Path(__file__).with_name("test_strength_freshness_cpu_collector.py"))
)
MODEL = fixtures["MODEL"]
VERIFIED = {MODEL: "b" * 64}


def sample():
    return fixtures["sample"].__wrapped__()


def capture(pair):
    reg, io = pair
    return c._Capture(
        identity.IdentityCollector(reg, cast(readonly.ReadOnlyIO, io)),
        verified=VERIFIED,
        phase="before",
        cleanup=None,
        physical_kind="learner_metrics",
        interval=0.1,
        prior_witnesses=None,
        sleep=io.sleep,
    )


def test_static_operation_returns_exact_wrapper_facts_and_inventory():
    direct_pair, wrapped_pair = sample(), sample()
    direct, wrapped = capture(direct_pair), capture(wrapped_pair)
    pid = direct_pair[1].roles["coordinator"]
    # This operation uses an already measured PID; it does not read reporters or
    # admit a new collector, and returns no authority or qualified status.
    del direct_pair[1].data["coordinator"]
    static, units, champion, check = c._read_static_authority(
        direct.identity, coordinator_pid=pid, verified_champions=VERIFIED
    )
    assert wrapped._static({"coordinator_pid": pid}) == (static, units, champion)
    assert wrapped.source_checks == [check] and direct.source_checks == []
    assert direct.identity.audit == wrapped.identity.audit
    assert direct_pair[1].calls == wrapped_pair[1].calls
    assert static == direct.policy["static"] and champion == MODEL
    assert set(check) == {
        "source_pins",
        "cached_references",
        "static_sha256",
        "default_target",
    }
    assert fixtures["identity_fixtures"]["SECRET"] not in json.dumps(check)
    authority_reads = [
        key
        for operation, key in direct_pair[1].calls
        if operation == "read" and key in direct.reg["keys"].values()
    ]
    assert authority_reads == [
        "run",
        "continuation",
        "profile_authority",
        "profile",
        "run_source",
        "release_source",
        "source_manifest",
        "champion",
    ]


@pytest.mark.parametrize(
    "fault,reason,last_read",
    [
        ("source", "source-pin-drift", "source"),
        ("profile", "profile-checksum-authority", "run_source"),
        ("run_source", "source-authority", "run_source"),
        ("release_source", "release-source-drift", "release_source"),
        ("source_manifest", "source-manifest-drift", "source_manifest"),
        ("native", "native-policy-binding", "source_manifest"),
        ("continuation", "continuation-coordinator-owner", "source_manifest"),
        ("champion_run", "champion-run-binding", "champion"),
        ("champion_proof", "unverified-champion", "champion"),
    ],
)
def test_static_fault_retains_first_refusal_and_no_source_check(
    fault, reason, last_read
):
    pair = sample()
    _, io = pair
    engine = capture(pair)
    if fault in {
        "source",
        "profile",
        "run_source",
        "release_source",
        "source_manifest",
    }:
        io.data[fault] += b"changed"
    elif fault == "native":
        engine.policy["static"]["native_sha256"] = "c" * 64
    elif fault == "continuation":
        row = json.loads(io.data["continuation"])
        row["attempts"][-1]["pid"] += 1
        io.data["continuation"] = identity.encoded(row)
    elif fault == "champion_run":
        row = json.loads(io.data["champion"])
        row["run_id"] = "other"
        io.data["champion"] = identity.encoded(row)
    else:
        engine.verified = {}
    with pytest.raises(ValueError, match=f"^{reason}$"):
        engine._static({"coordinator_pid": io.roles["coordinator"]})
    assert engine.source_checks == []
    assert [key for op, key in io.calls if op == "read"][-1] == last_read


@pytest.mark.parametrize("earlier_fault", [None, "source", "profile"])
def test_missing_coordinator_pid_preserves_deferred_lookup(earlier_fault):
    pair = sample()
    engine = capture(pair)
    _, io = pair
    if earlier_fault is not None:
        io.data[earlier_fault] += b"changed"
    reason = {
        None: "coordinator_pid",
        "source": "source-pin-drift",
        "profile": "profile-checksum-authority",
    }[earlier_fault]
    with pytest.raises((KeyError, ValueError), match=reason) as caught:
        engine._static({})
    if earlier_fault is None:
        assert type(caught.value) is KeyError
    assert engine.source_checks == []
    assert [key for op, key in io.calls if op == "read"][-1] == {
        None: "source_manifest",
        "source": "source",
        "profile": "run_source",
    }[earlier_fault]


@pytest.mark.parametrize("fault", ["job", "failure", "invocation"])
def test_final_support_mismatch_precedes_final_heartbeat_and_owner_reads(fault):
    pair = sample()
    _, io = pair
    engine = capture(pair)
    io.final_support_fault = fault
    with pytest.raises(identity.CollectionRefusal, match="final-support-state-drift"):
        engine.run()
    # Both static scans finish, but the contradictory second scan prevents any
    # subsequent reporter read, owner scan, or final clock publication.
    assert len(engine.source_checks) == 2
    audits = engine.identity.audit
    source_positions = [
        i
        for i, row in enumerate(audits)
        if row["operation"] == "read" and row["subject"] == "source"
    ]
    assert len(source_positions) == 2
    last_scan = audits[source_positions[-1] :]
    assert last_scan[-1]["subject"] == "champion"
    assert all(row["operation"] != "cgroup-members" for row in last_scan)
    assert not any(
        row["operation"] == "read"
        and row["subject"] in engine.reg["heartbeats"].values()
        for row in last_scan
    )


@pytest.mark.parametrize(
    "fault", [None, "missing_unit", "missing_field", "extra_field", "drift"]
)
def test_final_support_unit_kind_and_exact_dynamic_inventory(fault):
    pair = sample()
    reg, io = pair
    states = {
        name: {
            key: io.props[name][key]
            for key in support.DYNAMIC
            + (support.SERVICE_DYNAMIC if reg["units"][name]["kind"] != "timer" else ())
        }
        for name in reg["policy"]["support"]
    }
    frozen = copy.deepcopy(states)
    if fault == "missing_unit":
        del states["backup.service"]
    elif fault == "missing_field":
        del states["backup.service"]["Job"]
    elif fault == "extra_field":
        states["report.timer"]["ExecMainStatus"] = "0"
    elif fault == "drift":
        io.props["backup.service"]["Job"] = "9"

    def check():
        c._check_final_support_states(
            reg["policy"]["support"],
            reg["units"],
            provenance={"unit_states": states},
            final_units=io.props,
        )

    if fault is None:
        check()
        assert states == frozen
    else:
        with pytest.raises(identity.CollectionRefusal, match="final-support-state-"):
            check()


@pytest.mark.parametrize("elapsed,added", [(0, 0), (1, 1), (10**9, 1), (10**9 + 1, 2)])
def test_age_projection_preserves_aliases_none_and_integer_ceiling(elapsed, added):
    shared_job = {"id": 7, "age_seconds": 3}
    rows = {
        "known": {"running_seconds": 10, "job": shared_job},
        "unknown": {"running_seconds": None, "job": None},
    }
    known, unknown = rows["known"], rows["unknown"]
    observed = {"boot_id": "boot", "monotonic_ns": 100, "wall_ns": 1000}
    # No new wall-clock policy is invented by this monotonic-age extraction.
    final = {**observed, "monotonic_ns": 100 + elapsed, "wall_ns": 1}
    result, projection = c._project_support_at_clock(rows, observed, final)
    assert result is rows and result["known"] is known and result["unknown"] is unknown
    assert known["job"] is shared_job
    assert (
        known["running_seconds"] == 10 + added
        and shared_job["age_seconds"] == 3 + added
    )
    assert unknown == {"running_seconds": None, "job": None}
    assert projection == {
        "observed_clock": observed,
        "capture_clock": final,
        "added_upper_bound_seconds": added,
    }
    observed["wall_ns"] = final["wall_ns"] = -1
    assert projection["observed_clock"]["wall_ns"] == 1000
    assert projection["capture_clock"]["wall_ns"] == 1


@pytest.mark.parametrize("fault", ["boot", "monotonic"])
def test_bad_projection_clock_refuses_before_any_mutation(fault):
    rows = {"unit": {"running_seconds": 2, "job": {"age_seconds": 3}}}
    original = copy.deepcopy(rows)
    observed = {"boot_id": "boot", "monotonic_ns": 10, "wall_ns": 100}
    final = (
        {**observed, "boot_id": "other"}
        if fault == "boot"
        else {**observed, "monotonic_ns": 9}
    )
    with pytest.raises(identity.CollectionRefusal, match="^support-final-clock$"):
        c._project_support_at_clock(rows, observed, final)
    assert rows == original
