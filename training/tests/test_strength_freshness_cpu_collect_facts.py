from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

from scripts import strength_freshness_cpu_collect_facts as f

SHA = "a" * 64
INV = "b" * 32


def pin(path="/qualified/file", metadata=False):
    row = {"path": path, "sha256": SHA, "bytes": 10}
    if metadata:
        row = {
            "literal_path": path,
            "resolved_path": path,
            "sha256": SHA,
            "bytes": 10,
            "mode": 0o644,
            "uid": 0,
            "gid": 0,
        }
    return row


def definition(timer=False):
    unit = "work.timer" if timer else "work.service"
    props: dict[str, str] = dict.fromkeys(
        f.TIMER_STATIC if timer else f.SERVICE_STATIC, ""
    )
    props.update(
        Id=unit,
        Names=unit,
        LoadState="loaded",
        After="sysinit.target system.slice systemd-journald.socket",
    )
    if not timer:
        props["ExecStart"] = (
            "{ path=/bin/x ; argv[]=x secret-token ; ignore_errors=no ; start_time=old ; pid=123 ; status=0 }"
        )
    return {
        "unit": unit,
        "fragment": pin("/etc/systemd/system/" + unit, True),
        "dropins": [],
        "loaded_stable_properties": props,
        "need_daemon_reload": False,
    }


def environment():
    prop = {
        "present": True,
        "raw_sha256": SHA,
        "bytes": 4,
        "effective_empty_rule": None,
    }
    env = {**pin("/etc/qualified.env", True), "ignore_missing": False, "exists": True}
    return {
        "unit": "work.service",
        "inline_environment": prop,
        "pass_environment": copy.deepcopy(prop),
        "unset_environment": copy.deepcopy(prop),
        "environment_files": [env],
    }


def origin():
    return {
        "role": "coordinator",
        "executable": {
            "literal_entrypoint": "/release/.venv/bin/python",
            "resolved_proc_exe": "/qualified/python",
            "qualified_file_reference": pin("/qualified/python"),
        },
        "cwd_resolved": "/run/training",
        "argv": {"sha256": SHA, "bytes": 30, "approved_template_id": SHA},
        "relevant_environment": [
            {
                "key": "PYTHONPATH",
                "present": True,
                "value_sha256": SHA,
                "value_bytes": 8,
            }
        ],
        "source_contract": {
            "source_commit": "d" * 40,
            "source_manifest_sha256": SHA,
            "entrypoint_contract_sha256": SHA,
        },
        "native_mappings": [
            {
                "path": "/release/native.so",
                "device": 42,
                "inode": 123,
                "qualified_native_reference": pin("/release/native.so"),
            }
        ],
        "restart_counter_scope": "systemd-unit:NRestarts",
    }


def boot():
    return {
        "unit": "work.service",
        "unit_file_state": "enabled",
        "default_target": "graphical.target",
        "owned_links": [
            {
                "path": "/etc/systemd/system/multi-user.target.wants/work.service",
                "literal_target": "../work.service",
                "resolved_target": "/etc/systemd/system/work.service",
            }
        ],
        "required_reachability_edges": [
            {"from": "graphical.target", "to": "multi-user.target", "relation": "Wants"}
        ],
    }


def calibration():
    ns = {"pid": "pid:[1]", "time": "time:[2]"}
    return {
        "boot_id": "boot",
        "expected_boot_id": "boot",
        "monotonic_before_ns": 9_000_000_000,
        "boottime_before_ns": 10_000_000_000,
        "wall_ns": 1_800_000_000_000_000_000,
        "boottime_after_ns": 10_000_001_000,
        "monotonic_after_ns": 9_000_001_000,
        "max_bracket_ns": 1000,
        "offset_lower_ns": 1_799_999_989_999_999_000,
        "offset_upper_ns": 1_799_999_990_000_000_000,
        "namespaces": {k: copy.deepcopy(ns) for k in ("self", "pid1", "target")},
    }


def clock(ns):
    return {
        "boot_id": "boot",
        "monotonic_ns": ns,
        "wall_ns": 1_800_000_000_000_000_000 + ns,
    }


def job():
    return {
        "boot_id": "boot",
        "manager_id": "manager",
        "unit": "work.service",
        "id": 4,
        "kind": "start",
        "state": "waiting",
        "read_start": clock(4_000_000_000),
        "read_end": clock(4_000_010_000),
        "creation_source_sha256": None,
    }


def absence():
    return {
        "boot_id": "boot",
        "manager_id": "manager",
        "unit": "work.service",
        "job": None,
        "read_start": clock(1),
        "read_end": clock(3_999_999_999),
        "raw_sha256": SHA,
    }


@pytest.mark.parametrize(
    "make,func",
    [
        (definition, f.definition_digest),
        (environment, f.environment_digest),
        (origin, f.origin_digest),
        (boot, f.boot_digest),
    ],
)
def test_digest_safe_immutable(make, func):
    value = make()
    original = copy.deepcopy(value)
    assert len(func(value)) == 64 and func(value) == func(original)
    assert value == original
    value["unexpected"] = "secret-token"
    with pytest.raises(f.FactViolation) as error:
        func(value)
    assert "secret-token" not in str(error.value)


def test_exec_dynamic_normalization_retains_static():
    a = definition()
    b = copy.deepcopy(a)
    b["loaded_stable_properties"]["ExecStart"] = (
        b["loaded_stable_properties"]["ExecStart"]
        .replace("old", "new")
        .replace("pid=123", "pid=456")
    )
    b["loaded_stable_properties"]["After"] = (
        "system.slice systemd-journald.socket sysinit.target"
    )
    assert f.definition_digest(a) == f.definition_digest(b)
    b["loaded_stable_properties"]["ExecStart"] = b["loaded_stable_properties"][
        "ExecStart"
    ].replace("ignore_errors=no", "ignore_errors=yes")
    assert f.definition_digest(a) != f.definition_digest(b)
    b = copy.deepcopy(a)
    b["loaded_stable_properties"]["ExecStart"] = b["loaded_stable_properties"][
        "ExecStart"
    ].replace(" ; status=0", " ; unknown_static=yes ; status=0")
    assert f.definition_digest(a) != f.definition_digest(b)


@pytest.mark.parametrize(
    "key", ["TimersMonotonic", "TimersCalendar", "NextElapseUSecMonotonic"]
)
def test_timer_dynamic_not_stable(key):
    value = definition(True)
    assert f.definition_digest(value)
    value["loaded_stable_properties"][key] = "changing"
    with pytest.raises(f.FactViolation, match="fields"):
        f.definition_digest(value)


def test_timer_environment_is_inapplicable_not_fabricated_empty():
    value = {"unit": "work.timer", "applicability": "not-applicable-timer"}
    assert f.environment_digest(value)
    bad = environment()
    bad["unit"] = "work.timer"
    with pytest.raises(f.FactViolation):
        f.environment_digest(bad)


@pytest.mark.parametrize(
    "change", ["reload", "missing", "exec", "multi", "bool_mode", "duplicate"]
)
def test_definition_refusals(change):
    value = definition()
    if change == "reload":
        value["need_daemon_reload"] = True
    if change == "missing":
        del value["loaded_stable_properties"]["Type"]
    if change == "exec":
        value["loaded_stable_properties"]["ExecStart"] = "secret-token"
    if change == "multi":
        value["loaded_stable_properties"]["ExecStart"] *= 2
    if change == "bool_mode":
        value["fragment"]["mode"] = True
    if change == "duplicate":
        value["loaded_stable_properties"]["After"] = "a.target a.target"
    with pytest.raises(f.FactViolation) as error:
        f.definition_digest(value)
    assert "secret-token" not in str(error.value)


def test_dependency_escaped_device_allowed_not_path_or_shell():
    value = definition()
    value["loaded_stable_properties"]["After"] = (
        r"dev-disk-by\x2duuid.device root.mount"
    )
    assert f.definition_digest(value)
    for invalid in ["/tmp/unit.service", "x.service;secret-token", r"dev\xx.device"]:
        value["loaded_stable_properties"]["After"] = invalid
        with pytest.raises(f.FactViolation):
            f.definition_digest(value)


def test_environment_order_mode_and_presence_bound():
    a = environment()
    b = copy.deepcopy(a)
    b["environment_files"][0]["mode"] = 0o600
    assert f.environment_digest(a) != f.environment_digest(b)
    a["environment_files"].append(
        {
            **a["environment_files"][0],
            "literal_path": "/etc/other.env",
            "resolved_path": "/etc/other.env",
        }
    )
    b = copy.deepcopy(a)
    b["environment_files"].reverse()
    assert f.environment_digest(a) != f.environment_digest(b)
    b["inline_environment"] = {
        "present": False,
        "raw_sha256": None,
        "bytes": 0,
        "effective_empty_rule": "systemd-255-empty-Environment",
    }
    assert f.environment_digest(b)
    b["inline_environment"]["effective_empty_rule"] = None
    with pytest.raises(f.FactViolation, match="unqualified-empty"):
        f.environment_digest(b)


def test_optional_missing_environment():
    value = environment()
    env = value["environment_files"][0]
    env.update(
        exists=False,
        ignore_missing=True,
        **dict.fromkeys("sha256 bytes mode uid gid".split()),
    )
    assert f.environment_digest(value)
    env["ignore_missing"] = False
    with pytest.raises(f.FactViolation, match="missing-environment"):
        f.environment_digest(value)


@pytest.mark.parametrize(
    "change",
    [
        "counter",
        "argv_raw",
        "env_raw",
        "env_order",
        "native_order",
        "source",
        "symlink_literal",
    ],
)
def test_origin_exact_scope(change):
    value = origin()
    baseline = f.origin_digest(value)
    if change == "counter":
        value["restart_counter_scope"] = "coordinator-worker:restart_count"
    if change == "argv_raw":
        value["argv"]["raw"] = "secret-token"
    if change == "env_raw":
        value["relevant_environment"][0]["value"] = "secret-token"
    if change == "env_order":
        value["relevant_environment"] *= 2
    if change == "native_order":
        value["native_mappings"] *= 2
    if change == "source":
        value["source_contract"]["source_commit"] = "e" * 40
    if change == "symlink_literal":
        value["executable"]["literal_entrypoint"] = "/other/.venv/bin/python"
    if change in {"source", "symlink_literal"}:
        assert f.origin_digest(value) != baseline
    else:
        with pytest.raises(f.FactViolation) as error:
            f.origin_digest(value)
        assert "secret-token" not in str(error.value)


@pytest.mark.parametrize(
    "change",
    ["unreachable", "foreignlink", "duplicate", "default", "target", "enabled"],
)
def test_boot_known_owned_edges(change):
    value = boot()
    if change == "unreachable":
        value["required_reachability_edges"] = []
    if change == "foreignlink":
        value["owned_links"][0]["path"] = (
            "/etc/systemd/system/multi-user.target.wants/other.service"
        )
    if change == "duplicate":
        value["owned_links"] *= 2
    if change == "default":
        value["default_target"] = "other.target"
    if change == "target":
        value["owned_links"][0]["resolved_target"] = "/etc/systemd/system/other.service"
    if change == "enabled":
        value["unit_file_state"] = "disabled"
    with pytest.raises(f.FactViolation):
        f.boot_digest(value)


def test_canonical_encoding_exact():
    value = environment()
    expected = hashlib.sha256(
        json.dumps(
            {
                "format": "strength-preservation-unit-environment-v1",
                "schema_version": 1,
                **value,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    assert f.environment_digest(value) == expected


def test_birth_boottime_integer_bound():
    c = calibration()
    assert (
        f.birth_upper_ns(start_ticks=3, hz=3, calibration=c)
        == c["wall_ns"] - c["boottime_before_ns"] + (4 * 10**9 + 2) // 3
    )
    c["monotonic_before_ns"] -= 10**9
    c["monotonic_after_ns"] -= 10**9
    assert (
        f.birth_upper_ns(start_ticks=3, hz=3, calibration=c)
        == c["wall_ns"] - c["boottime_before_ns"] + (4 * 10**9 + 2) // 3
    )


@pytest.mark.parametrize(
    "change", ["boot", "namespace", "bracket", "offset", "future", "bool", "hz"]
)
def test_birth_refuses_unknown_mapping(change):
    c = calibration()
    ticks, hz = 3, 3
    if change == "boot":
        c["expected_boot_id"] = "other"
    if change == "namespace":
        c["namespaces"]["target"]["time"] = "time:[3]"
    if change == "bracket":
        c["max_bracket_ns"] = 999
    if change == "offset":
        c["offset_upper_ns"] -= 1
    if change == "future":
        ticks = 300
    if change == "bool":
        ticks = True
    if change == "hz":
        hz = 0
    with pytest.raises(f.FactViolation):
        f.birth_upper_ns(start_ticks=ticks, hz=hz, calibration=c)


def test_running_age_rounds_up_and_joins_invocation():
    assert (
        f.running_seconds(
            now_monotonic_ns=1_000_002_001,
            activation_start_us=2,
            invocation_id=INV,
            expected_invocation_id=INV,
        )
        == 2
    )
    for start, invocation in [(0, INV), (2, "c" * 32), (2_000_000, INV)]:
        with pytest.raises(f.FactViolation):
            f.running_seconds(
                now_monotonic_ns=1_000_002_001,
                activation_start_us=start,
                invocation_id=invocation,
                expected_invocation_id=INV,
            )


def test_job_absence_start_not_end_or_first_seen():
    assert (
        f.job_age_seconds(current=job(), now=clock(5_000_000_001), absence=absence())
        == 5
    )
    with pytest.raises(f.FactViolation, match="unknown-job-age"):
        f.job_age_seconds(current=job(), now=clock(5_000_000_001))
    j = job()
    j["creation_source_sha256"] = SHA
    assert (
        f.job_age_seconds(
            current=j, now=clock(5_000_000_001), creation_lower_ns=3_000_000_000
        )
        == 3
    )


@pytest.mark.parametrize(
    "change",
    ["manager", "unit", "boot", "present", "late", "source", "both", "future", "kind"],
)
def test_job_witness_refusals(change):
    j, a = job(), absence()
    kwargs: dict[str, Any] = {"current": j, "now": clock(5_000_000_001), "absence": a}
    if change == "manager":
        a["manager_id"] = "new-manager"
    if change == "unit":
        a["unit"] = "other.service"
    if change == "boot":
        a["boot_id"] = "other"
    if change == "present":
        a["job"] = 1
    if change == "late":
        a["read_end"] = clock(4_000_000_001)
    if change == "source":
        a["raw_sha256"] = "not-a-hash"
    if change == "both":
        kwargs["creation_lower_ns"] = 0
    if change == "future":
        kwargs["now"] = clock(3)
    if change == "kind":
        j["kind"] = "stop"
    with pytest.raises(f.FactViolation):
        f.job_age_seconds(**kwargs)


@pytest.mark.parametrize(
    "make,func", [(environment, f.environment_digest), (boot, f.boot_digest)]
)
def test_controlled_roots_are_not_dependency_units(make, func):
    value = make()
    value["unit"] = "system.slice"
    with pytest.raises(f.FactViolation):
        func(value)


def test_process_reference_must_pin_observed_path():
    value = origin()
    value["executable"]["qualified_file_reference"]["path"] = "/other/python"
    with pytest.raises(f.FactViolation, match="executable-reference"):
        f.origin_digest(value)
    value = origin()
    value["native_mappings"][0]["qualified_native_reference"]["path"] = (
        "/other/native.so"
    )
    with pytest.raises(f.FactViolation, match="native-reference"):
        f.origin_digest(value)


def test_empty_omission_rule_is_property_specific():
    value = environment()
    value["inline_environment"] = {
        "present": False,
        "raw_sha256": None,
        "bytes": 0,
        "effective_empty_rule": "systemd-255-empty-PassEnvironment",
    }
    with pytest.raises(f.FactViolation, match="unqualified-empty"):
        f.environment_digest(value)


@pytest.mark.parametrize(
    "role",
    [
        "learner",
        "actor-cpu-ring4",
        "arena-promotion",
        *[f"actor-gpu-{i}" for i in range(1, 7)],
    ],
)
def test_known_worker_restart_scope_is_coordinator_owned(role):
    value = origin()
    value["role"] = role
    with pytest.raises(f.FactViolation, match="restart-scope"):
        f.origin_digest(value)
    value["restart_counter_scope"] = "coordinator-worker:restart_count"
    assert f.origin_digest(value)
