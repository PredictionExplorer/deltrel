from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import cast

import pytest

from scripts import strength_freshness_cpu_collect_processes as p
from scripts import strength_freshness_cpu_preservation as preservation
from scripts.strength_freshness_cpu_readonly import Observation, ProcessAdmission

RUNTIME = "edgeconnect-strength-recovery-20261002.service"
MONITOR = "edgeconnect-strength-recovery-20261002-monitor.service"
BOOT_OFFSET = 1_800_000_000_000_000_000
INV = "1" * 32
SHA = "a" * 64


def obs(value):
    return Observation(copy.deepcopy(value), {"operation": "fixture"})


@pytest.fixture
def sample():
    workers = [
        "learner",
        *sorted(preservation.ACTOR_ROLES),
        "actor-cpu-ring4",
        "arena-promotion",
    ]
    roles = ["controller", "coordinator", *workers]
    processes = {
        role: {
            "pid": i + 100,
            "start_ticks": i + 1000,
            "cgroup": "/system.slice/" + RUNTIME,
            "invocation_id": INV,
            "restarts": 0,
            "origin_sha256": SHA,
        }
        for i, role in enumerate(roles)
    }
    monitor = {
        "pid": 200,
        "start_ticks": 1200,
        "cgroup": "/system.slice/" + MONITOR,
        "invocation_id": "2" * 32,
        "restarts": 0,
        "origin_sha256": SHA,
    }
    gpu_roles = dict(
        zip(
            [f"GPU-{i}" for i in range(8)],
            ["learner", *sorted(preservation.ACTOR_ROLES), "arena-promotion"],
            strict=True,
        )
    )
    bounds = {
        role: BOOT_OFFSET + (row["start_ticks"] + 2) * 10_000_000
        for role, row in processes.items()
    }
    policy = {
        "static": {
            "runtime_name": RUNTIME,
            "runtime_cgroup": "/system.slice/" + RUNTIME,
        },
        "workers": workers,
        "expected_processes": processes,
        "expected_monitors": {MONITOR: monitor},
        "gpu_roles": gpu_roles,
        "gpu_uuids": list(gpu_roles),
        "learner_birth_upper_ns": bounds["learner"],
    }
    birth = {
        "boot_id": "boot",
        "qualification_sha256": SHA,
        "offset_lower_ns": BOOT_OFFSET - 1000,
        "offset_upper_ns": BOOT_OFFSET + 1000,
        "max_bracket_ns": 100,
        "bounds": bounds,
    }
    aux = {
        "python_argv0": "/venv/bin/python",
        "executable": "/usr/bin/python3",
        "cwd": "/release/training",
        "compile_worker_script": "/venv/site/torch/_inductor/compile_worker/__main__.py",
        "torch_key": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
        "compile_parent_roles": ["learner"],
    }
    raw = {}
    for role, row in {**processes, "monitor": monitor}.items():
        parent = (
            1
            if role in {"controller", "monitor"}
            else processes["controller" if role == "coordinator" else "coordinator"][
                "pid"
            ]
        )
        raw[row["pid"]] = {
            "pid": row["pid"],
            "start_ticks": row["start_ticks"],
            "ppid": parent,
            "cgroup": row["cgroup"],
            "exe": "/usr/bin/python3",
            "cwd": "/release/training",
            "cmdline": b"/venv/bin/python\0-m\0runtime\0",
            "environ": b"PYTHONPATH=/release/training\0",
            "maps": b"native",
        }
    units = {
        name: {
            "ActiveState": "active",
            "SubState": "running",
            "Result": "success",
            "ControlGroup": row["cgroup"],
            "MainPID": str(row["pid"]),
            "InvocationID": row["invocation_id"],
            "NRestarts": "0",
        }
        for name, row in [(RUNTIME, processes["controller"]), (MONITOR, monitor)]
    }
    heartbeat_ns = BOOT_OFFSET + 99_000_000_000
    coordinator = {
        "state": "running",
        "draining": False,
        "failure": None,
        "timestamp_ns": heartbeat_ns,
        "coordinator_pid": processes["coordinator"]["pid"],
        "workers": {
            role: {
                "state": "running",
                "pid": processes[role]["pid"],
                "failure_class": None,
                "failure_reason": None,
                "restart_count": 0,
            }
            for role in workers
        },
    }
    heartbeats = {
        role: {
            "worker": role,
            "pid": processes[role]["pid"],
            "heartbeat_ns": heartbeat_ns,
        }
        for role in workers
    }

    class IO:
        def __init__(self):
            self.raw = raw
            self.members_calls = 0
            self.race = False
            self.namespace_drift = False
            self.inventory = "\n".join(f"{i}, GPU-{i}" for i in range(8))
            self.owners = "\n".join(
                f"{processes[role]['pid']}, {uuid}" for uuid, role in gpu_roles.items()
            )

        def members(self, name):
            self.members_calls += 1
            value = tuple(
                ProcessAdmission(pid, row["start_ticks"], row["cgroup"])
                for pid, row in self.raw.items()
                if row["cgroup"] == "/system.slice/" + name
            )
            if self.race and self.members_calls == 3:
                value = value[:-1]
            return obs(value)

        def process(self, admission, *, maps=True):
            result = copy.deepcopy(self.raw[admission.pid])
            if not maps:
                result["maps"] = None
            return obs(result)

        def birth_bracket(self):
            return obs(
                {
                    "boot_id": "boot",
                    "monotonic_before_ns": 100_000_000_000,
                    "boottime_before_ns": 100_000_000_000,
                    "wall_ns": BOOT_OFFSET + 100_000_000_000,
                    "boottime_after_ns": 100_000_000_001,
                    "monotonic_after_ns": 100_000_000_001,
                    "clock_ticks_per_second": 100,
                }
            )

        def namespaces(self, subject):
            different = self.namespace_drift and isinstance(subject, ProcessAdmission)
            return obs(
                {
                    "pid": {"literal": "pid:[2]" if different else "pid:[1]"},
                    "time": {"literal": "time:[1]"},
                }
            )

        def query(self, kind):
            return obs(self.inventory if kind == "gpu-inventory" else self.owners)

    io = IO()
    identity = SimpleNamespace(
        reg={"policy": policy, "birth_reference": birth, "auxiliary_policy": aux},
        io=io,
        take=lambda o: o.value,
        origin=lambda role, raw: SHA,
    )
    return SimpleNamespace(
        collector=p.ProcessCollector(identity),
        identity=identity,
        io=io,
        raw=raw,
        policy=policy,
        units=units,
        coordinator=coordinator,
        heartbeats=heartbeats,
    )


def capture(s):
    return s.collector.capture(
        unit_properties=s.units, coordinator=s.coordinator, heartbeats=s.heartbeats
    )


def test_real_join_shape_no_private_values(sample):
    result = capture(sample)
    assert len(result["processes"]) == 11
    assert len(result["gpu_owners"]) == 8
    assert result["monitors"] == sample.policy["expected_monitors"]
    assert (
        result["processes"]["learner"]["heartbeat_ns"]
        == sample.heartbeats["learner"]["heartbeat_ns"]
    )
    assert (
        result["births"]["learner"]["measured_upper_ns"]
        < sample.policy["learner_birth_upper_ns"]
    )
    assert "PYTHONPATH" not in repr(result) and "cmdline" not in repr(result)
    p.same_owners(result, capture(sample))


@pytest.mark.parametrize(
    "change",
    [
        "main",
        "invocation",
        "restart",
        "heartbeat-pid",
        "worker-state",
        "parent",
        "origin",
        "start",
        "monitor",
        "namespace",
        "birth",
        "future-birth",
        "race",
        "gpu-owner",
        "gpu-duplicate",
        "gpu-index",
        "coordinator",
        "extra-worker",
    ],
)
def test_refuses_unjoined_or_changed_facts(sample, change):
    learner = sample.policy["expected_processes"]["learner"]["pid"]
    if change == "main":
        sample.units[RUNTIME]["MainPID"] = "400"
    if change == "invocation":
        sample.units[RUNTIME]["InvocationID"] = "3" * 32
    if change == "restart":
        sample.coordinator["workers"]["learner"]["restart_count"] = 1
    if change == "heartbeat-pid":
        sample.heartbeats["learner"]["pid"] = 400
    if change == "worker-state":
        sample.coordinator["workers"]["learner"]["state"] = "failed"
    if change == "parent":
        sample.raw[learner]["ppid"] = 400
    if change == "origin":
        sample.identity.origin = lambda *_: "b" * 64
    if change == "start":
        sample.raw[learner]["start_ticks"] += 1
    if change == "monitor":
        sample.raw[200]["start_ticks"] += 1
    if change == "namespace":
        sample.io.namespace_drift = True
    if change == "birth":
        sample.identity.reg["birth_reference"]["offset_upper_ns"] = 1
    if change == "future-birth":
        sample.identity.reg["birth_reference"]["bounds"]["learner"] = (
            BOOT_OFFSET + 10**12
        )
    if change == "race":
        sample.io.race = True
    if change == "gpu-owner":
        sample.io.owners = sample.io.owners.replace(str(learner), "400", 1)
    if change == "gpu-duplicate":
        sample.io.owners = sample.io.owners.replace("GPU-7", "GPU-6")
    if change == "gpu-index":
        sample.io.inventory = sample.io.inventory.replace("7, GPU-7", "6, GPU-7")
    if change == "coordinator":
        sample.coordinator["draining"] = True
    if change == "extra-worker":
        sample.coordinator["workers"]["foreign"] = {}
    with pytest.raises((p.ProcessRefusal, p.facts.FactViolation)):
        capture(sample)


def add_tracker(s):
    parent = s.policy["expected_processes"]["learner"]["pid"]
    raw = copy.deepcopy(s.raw[parent])
    raw.update(
        pid=300,
        start_ticks=1300,
        ppid=parent,
        cmdline=b"/venv/bin/python\0-B\0-c\0from multiprocessing.resource_tracker import main;main(7)\0",
    )
    s.raw[300] = raw


def test_closed_auxiliary_is_only_owner_evidence(sample):
    add_tracker(sample)
    result = capture(sample)
    assert result["auxiliaries"] == [
        {
            "pid": 300,
            "start_ticks": 1300,
            "ppid": 102,
            "parent_role": "learner",
            "kind": "multiprocessing-resource-tracker",
        }
    ]
    assert all(x["pid"] != 300 for x in result["gpu_owners"])


@pytest.mark.parametrize(
    "change",
    ["unowned", "command", "env", "duplicate-env", "new-override", "exe", "cwd"],
)
def test_foreign_auxiliary_never_certifies_collection(sample, change):
    add_tracker(sample)
    child = sample.raw[300]
    if change == "unowned":
        child["ppid"] = 500
    if change == "command":
        child["cmdline"] = b"/venv/bin/python\0-c\0print('foreign')\0"
    if change == "env":
        child["environ"] = b"PYTHONPATH=/foreign\0"
    if change == "duplicate-env":
        child["environ"] += b"PYTHONPATH=/release/training\0"
    if change == "new-override":
        child["environ"] += b"PYTHONINSPECT=1\0"
    if change == "exe":
        child["exe"] = "/foreign/python"
    if change == "cwd":
        child["cwd"] = "/foreign"
    with pytest.raises(p.ProcessRefusal):
        capture(sample)


@pytest.mark.parametrize("change", ["gpu", "monitor", "aux", "origin", "heartbeat"])
def test_final_owner_recheck(sample, change):
    before = capture(sample)
    after = copy.deepcopy(before)
    if change == "gpu":
        after["gpu_owners"][0]["pid"] += 1
    if change == "monitor":
        after["monitors"][MONITOR]["pid"] += 1
    if change == "aux":
        after["auxiliaries"].append({"pid": 500})
    if change == "origin":
        after["processes"]["learner"]["origin_sha256"] = "b" * 64
    if change == "heartbeat":
        after["processes"]["learner"]["heartbeat_ns"] -= 1
    with pytest.raises(p.ProcessRefusal):
        p.same_owners(before, after)


@pytest.mark.parametrize(
    "role",
    [
        "coordinator",
        "learner",
        *sorted(preservation.ACTOR_ROLES),
        "actor-cpu-ring4",
        "arena-promotion",
    ],
)
@pytest.mark.parametrize("delta", [0, -1])
def test_fresh_previous_lifetime_heartbeat_refuses(sample, role, delta):
    stamp = sample.identity.reg["birth_reference"]["bounds"][role] + delta
    # Within the ordinary 120-second freshness window, but not this lifetime.
    if role == "coordinator":
        sample.coordinator["timestamp_ns"] = stamp
    else:
        sample.heartbeats[role]["heartbeat_ns"] = stamp
    with pytest.raises(p.ProcessRefusal, match="producer-heartbeat-before-birth"):
        capture(sample)


def test_private_unicode_failure_never_echoes_environment(sample):
    raw = copy.deepcopy(sample.raw[102])
    raw["environ"] = b"TOKEN=secret\xff\0"
    with pytest.raises(p.ProcessRefusal, match="^process-private-encoding$"):
        p.private_process(raw)


@pytest.mark.parametrize("drift", [False, True])
def test_real_identity_collector_composes_with_kernel_joins(sample, drift):
    from scripts import strength_freshness_cpu_collect_identity as identities
    from tests.test_strength_freshness_cpu_collect_identity import identity_fixture

    reg, reference_io, example = identity_fixture()
    original_static = reg["policy"]["static"]
    reg["policy"] = sample.policy
    reg["policy"]["static"].update(original_static)
    reg["policy"]["cohorts"] = list(reg["cohorts"])
    reg["birth_reference"] = sample.identity.reg["birth_reference"]
    reg["auxiliary_policy"] = sample.identity.reg["auxiliary_policy"]
    reg["scope"]["units"][MONITOR] = reg["scope"]["units"].pop("monitor.service")
    reg["units"][MONITOR] = reg["units"].pop("monitor.service")
    fragment = reg["units"][MONITOR]["fragment"]
    reg["scope"]["files"][fragment]["path"] = "/etc/systemd/system/" + MONITOR
    for raw in sample.raw.values():
        raw.update(
            {key: example[key] for key in ("exe", "cwd", "cmdline", "environ", "maps")}
        )

    class JoinedIO:
        scope = identities.scope_from(reg["scope"])

        def __getattr__(self, name):
            if hasattr(sample.io, name):
                return getattr(sample.io, name)
            return getattr(reference_io, name)

    host = identities.IdentityCollector(
        reg, cast(identities.readonly.ReadOnlyIO, JoinedIO())
    )
    # Frozen fixture origins are prepared before the tested measurement. The
    # changed-import arm then exercises real digest/pin validation, not a stub.
    for role, row in host.reg["policy"]["expected_processes"].items():
        row["origin_sha256"] = host.origin(role, sample.raw[row["pid"]])
    monitor = host.reg["policy"]["expected_monitors"][MONITOR]
    monitor["origin_sha256"] = host.origin("monitor", sample.raw[monitor["pid"]])
    if drift:
        sample.raw[102]["environ"] += b"LD_PRELOAD=/foreign.so\0"
    collector = p.ProcessCollector(host)
    if drift:
        with pytest.raises(
            identities.CollectionRefusal, match="unregistered-import-override"
        ):
            collector.capture(
                unit_properties=sample.units,
                coordinator=sample.coordinator,
                heartbeats=sample.heartbeats,
            )
    else:
        result = collector.capture(
            unit_properties=sample.units,
            coordinator=sample.coordinator,
            heartbeats=sample.heartbeats,
        )
        assert len(result["processes"]) == 11
        assert (
            result["processes"]["learner"]["origin_sha256"]
            == host.reg["policy"]["expected_processes"]["learner"]["origin_sha256"]
        )
        assert host.audit and "private-sentinel-value" not in repr(host.audit)
