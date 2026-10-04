"""Join bounded kernel and manager observations to predeclared R3 owners.

This collector component does not grant authority or certify progress. Its
caller must repeat the joins after file/producer sampling and qualify the
registration, source closure and outer deadline independently.
"""

from __future__ import annotations

import csv
import io
from typing import Any, Mapping

from scripts import strength_freshness_auxiliary as auxiliary
from scripts import strength_freshness_cpu_collect_facts as facts
from scripts import strength_freshness_cpu_preservation as preservation


class ProcessRefusal(ValueError):
    """A fixed non-private refusal code."""


def require(value: object, code: str) -> None:
    if not value:
        raise ProcessRefusal(code)


def natural(value: object, *, minimum: int = 0) -> int:
    require(
        isinstance(value, str) and value.isascii() and value.isdecimal(),
        "manager-integer",
    )
    assert isinstance(value, str)
    result = int(value)
    require(result >= minimum, "manager-integer-range")
    return result


def private_process(raw: Mapping[str, Any]) -> dict[str, Any]:
    """For the closed auxiliary matcher only; never serialize this result."""

    def decode(value: bytes) -> str:
        try:
            return value.decode("utf-8")
        except UnicodeError:
            raise ProcessRefusal("process-private-encoding") from None

    require(
        raw["cmdline"].endswith(b"\0") and raw["environ"].endswith(b"\0"),
        "process-record-termination",
    )
    environment = {}
    for item in raw["environ"][:-1].split(b"\0"):
        require(b"=" in item, "process-environment-record")
        key, value = item.split(b"=", 1)
        name = decode(key)
        require(name and name not in environment, "process-environment-duplicate")
        environment[name] = decode(value)
    return {
        **{k: raw[k] for k in ("pid", "start_ticks", "ppid", "exe", "cwd")},
        "argv": [decode(x) for x in raw["cmdline"][:-1].split(b"\0")],
        "environment": environment,
    }


def gpu_records(
    inventory: str, owners: str, *, policy, processes
) -> list[dict[str, Any]]:
    def rows(text):
        parsed = list(csv.reader(io.StringIO(text), skipinitialspace=True))
        require(
            len(parsed) == 8 and all(len(r) == 2 for r in parsed), "gpu-query-shape"
        )
        return [[v.strip() for v in r] for r in parsed]

    devices = rows(inventory)
    require({natural(r[0]) for r in devices} == set(range(8)), "gpu-index-inventory")
    require({r[1] for r in devices} == set(policy["gpu_uuids"]), "gpu-uuid-inventory")
    compute = rows(owners)
    require(len({r[1] for r in compute}) == 8, "gpu-duplicate-owner")
    result = []
    for pid_text, uuid in compute:
        require(uuid in policy["gpu_roles"], "gpu-unregistered-device")
        role = policy["gpu_roles"][uuid]
        process = processes[role]
        require(natural(pid_text, minimum=1) == process["pid"], "gpu-foreign-owner")
        result.append(
            {
                "uuid": uuid,
                "role": role,
                "unit": policy["static"]["runtime_name"],
                **{
                    k: process[k]
                    for k in ("pid", "start_ticks", "cgroup", "invocation_id")
                },
            }
        )
    return sorted(result, key=lambda r: r["uuid"])


def _current_process_fields(identity, role, pid, raw, props):
    """Private kernel/origin facts only; no reporter restart or birth verdict."""
    require(raw["cgroup"] == props["ControlGroup"], "process-cgroup-drift")
    return {
        "pid": pid,
        "start_ticks": raw["start_ticks"],
        "cgroup": raw["cgroup"],
        "invocation_id": props["InvocationID"],
        "origin_sha256": identity.origin(role, raw),
    }


def _current_monitor(identity, name, props, expected):
    """Private one-monitor observation, shared without weakening old admission."""
    take, io = identity.take, identity.io
    require(
        props["ActiveState"] == "active" and props["SubState"] == "running",
        "monitor-not-running",
    )
    members = take(io.members(name))
    require(
        len(members) == 1
        and members[0].pid == natural(props["MainPID"], minimum=1) == expected["pid"],
        "monitor-cgroup-owner",
    )
    raw = take(io.process(members[0]))
    row = {
        "pid": raw["pid"],
        "start_ticks": raw["start_ticks"],
        "cgroup": raw["cgroup"],
        "invocation_id": props["InvocationID"],
        "restarts": natural(props["NRestarts"]),
        "origin_sha256": identity.origin("monitor", raw),
    }
    require(
        row == expected and row["cgroup"] == props["ControlGroup"],
        "predeclared-monitor-drift",
    )
    return row, members[0], raw


def _current_auxiliaries(reg, private, roles):
    pending = set(private).difference(roles.values())
    known = {pid: role for role, pid in roles.items()}
    normalized = {pid: private_process(raw) for pid, raw in private.items()}
    output = []
    while pending:
        advanced = False
        for pid in sorted(pending):
            child = normalized[pid]
            parent_pid = child["ppid"]
            if parent_pid not in known:
                continue
            parent_role = known[parent_pid]
            kind = auxiliary.classify_auxiliary(
                reg["auxiliary_policy"],
                child,
                parent=normalized[parent_pid],
                parent_role=parent_role,
            )
            require(kind is not None, "unclassified-runtime-child")
            # The matcher owns known compiler exceptions. All other import
            # overrides must inherit; an extra PYTHON*/loader flag refuses.
            prefixes = (
                "PYTHON",
                "LD_",
                "DYLD_",
                "CUDA_",
                "TORCH",
                "TRITON",
                "OMP_",
                "MKL_",
                "OPENBLAS_",
            )
            parent_env, child_env = (
                normalized[parent_pid]["environment"],
                child["environment"],
            )
            exceptions = (
                reg["auxiliary_policy"].get("compile_environment", {})
                if kind == "torch-inductor-pool"
                else {}
            )
            relevant = {k for k in {*parent_env, *child_env} if k.startswith(prefixes)}
            require(
                all(
                    child_env.get(k) == exceptions.get(k, parent_env.get(k))
                    for k in relevant
                ),
                "auxiliary-import-environment",
            )
            output.append(
                {
                    "pid": pid,
                    "start_ticks": child["start_ticks"],
                    "ppid": parent_pid,
                    "parent_role": parent_role,
                    "kind": kind,
                }
            )
            known[pid] = parent_role
            pending.remove(pid)
            advanced = True
        require(advanced, "unowned-runtime-child")
    return sorted(output, key=lambda r: r["pid"])


class ProcessCollector:
    def __init__(self, identity: Any):
        self.identity = identity
        self.reg, self.io = identity.reg, identity.io
        self.policy = self.reg["policy"]

    def _births(self, admissions, processes):
        reference = self.reg["birth_reference"]
        require(
            set(reference)
            == {
                "boot_id",
                "qualification_sha256",
                "offset_lower_ns",
                "offset_upper_ns",
                "max_bracket_ns",
                "bounds",
            },
            "birth-reference-fields",
        )
        require(
            preservation.sha(reference["qualification_sha256"]),
            "birth-unqualified-reference",
        )
        require(set(reference["bounds"]) == set(processes), "birth-role-inventory")
        take = self.identity.take
        bracket = dict(take(self.io.birth_bracket()))
        hz = bracket.pop("clock_ticks_per_second")

        def namespace(subject):
            raw = take(self.io.namespaces(subject))
            return {k: raw[k]["literal"] for k in ("pid", "time")}

        caller, init = namespace("self"), namespace(1)
        result = {}
        for role, process in processes.items():
            material = {
                **bracket,
                "expected_boot_id": reference["boot_id"],
                **{
                    k: reference[k]
                    for k in ("offset_lower_ns", "offset_upper_ns", "max_bracket_ns")
                },
                "namespaces": {
                    "self": caller,
                    "pid1": init,
                    "target": namespace(admissions[process["pid"]]),
                },
            }
            measured = facts.birth_upper_ns(
                start_ticks=process["start_ticks"], hz=hz, calibration=material
            )
            fixed = reference["bounds"][role]
            require(
                type(fixed) is int and measured <= fixed < bracket["wall_ns"],
                "birth-bound-drift",
            )
            result[role] = {
                "measured_upper_ns": measured,
                "registered_upper_ns": fixed,
                "calibration": material,
                "clock_ticks_per_second": hz,
            }
        require(
            reference["bounds"]["learner"] == self.policy["learner_birth_upper_ns"],
            "learner-birth-policy",
        )
        return result

    def capture(self, *, unit_properties, coordinator, heartbeats) -> dict[str, Any]:
        """Capture only; a final repeated call must bracket producer sampling."""
        runtime = self.policy["static"]["runtime_name"]
        props = unit_properties[runtime]
        require(
            props["ActiveState"] == "active" and props["SubState"] == "running",
            "runtime-not-running",
        )
        require(
            props["ControlGroup"] == self.policy["static"]["runtime_cgroup"],
            "runtime-cgroup",
        )
        require(props["Result"] == "success", "runtime-result")
        require(
            coordinator["state"] == "running"
            and coordinator["draining"] is False
            and coordinator["failure"] is None,
            "coordinator-unhealthy",
        )
        require(
            set(coordinator["workers"])
            == set(self.policy["workers"])
            == set(heartbeats),
            "worker-inventory",
        )
        take = self.identity.take
        runtime_members = take(self.io.members(runtime))
        admissions = {p.pid: p for p in runtime_members}
        require(len(admissions) == len(runtime_members), "duplicate-cgroup-pid")
        roles = {
            role: p["pid"] for role, p in self.policy["expected_processes"].items()
        }
        require(
            set(roles) == {"controller", "coordinator", *self.policy["workers"]},
            "process-role-inventory",
        )
        require(len(set(roles.values())) == len(roles), "duplicate-role-pid")
        require(
            natural(props["MainPID"], minimum=1) == roles["controller"]
            and coordinator["coordinator_pid"] == roles["coordinator"],
            "controller-coordinator-pid",
        )
        require(set(roles.values()) <= set(admissions), "missing-runtime-owner")
        private = {
            pid: take(self.io.process(a, maps=pid in roles.values()))
            for pid, a in admissions.items()
        }
        processes = {}
        for role, pid in roles.items():
            raw = private[pid]
            require(raw["cgroup"] == props["ControlGroup"], "process-cgroup-drift")
            restarts = natural(props["NRestarts"])
            heartbeat = None
            if role == "coordinator":
                heartbeat = coordinator["timestamp_ns"]
                require(raw["ppid"] == roles["controller"], "coordinator-parent")
            elif role != "controller":
                row, beat = coordinator["workers"][role], heartbeats[role]
                require(
                    row["state"] == "running"
                    and row["pid"] == pid
                    and row["failure_class"] is None
                    and row["failure_reason"] is None,
                    "worker-health-or-pid",
                )
                require(
                    type(row["restart_count"]) is int and row["restart_count"] >= 0,
                    "worker-restart-counter",
                )
                restarts = row["restart_count"]
                require(
                    beat["worker"] == role and beat["pid"] == pid,
                    "worker-heartbeat-pid",
                )
                require(raw["ppid"] == roles["coordinator"], "worker-parent")
                heartbeat = beat["heartbeat_ns"]
            current = _current_process_fields(self.identity, role, pid, raw, props)
            result = {
                "pid": pid,
                "start_ticks": current["start_ticks"],
                "cgroup": current["cgroup"],
                "invocation_id": current["invocation_id"],
                "restarts": restarts,
                "origin_sha256": current["origin_sha256"],
            }
            require(
                result == self.policy["expected_processes"][role],
                "predeclared-process-drift",
            )
            if heartbeat is not None:
                require(
                    type(heartbeat) is int and heartbeat > 0, "process-heartbeat-stamp"
                )
                result["heartbeat_ns"] = heartbeat
            processes[role] = result
        auxiliary_rows = self._auxiliaries(private, roles)
        monitors = {}
        for name, expected in self.policy["expected_monitors"].items():
            row, _, _ = _current_monitor(
                self.identity, name, unit_properties[name], expected
            )
            monitors[name] = row
        births = self._births(admissions, processes)
        for role, process in processes.items():
            process["birth_upper_ns"] = births[role]["registered_upper_ns"]
            if "heartbeat_ns" in process:
                require(
                    process["heartbeat_ns"] > births[role]["registered_upper_ns"],
                    "producer-heartbeat-before-birth",
                )
        gpu = gpu_records(
            take(self.io.query("gpu-inventory")),
            take(self.io.query("gpu-owners")),
            policy=self.policy,
            processes=processes,
        )
        # Re-admit after all private reads and NVML commands. Expected owners
        # cannot be replaced or disappear while the capture is being assembled.
        again = take(self.io.members(runtime))
        require(set(again) == set(runtime_members), "runtime-members-raced")
        return {
            "processes": processes,
            "monitors": monitors,
            "gpu_owners": gpu,
            "births": births,
            "auxiliaries": auxiliary_rows,
        }

    def _auxiliaries(self, private, roles):
        return _current_auxiliaries(self.reg, private, roles)


def same_owners(first, second) -> None:
    """Final join ignores heartbeat advancement, never owner/auxiliary drift."""
    for key in ("monitors", "gpu_owners", "auxiliaries"):
        require(first[key] == second[key], "final-owner-drift")
    require(set(first["processes"]) == set(second["processes"]), "final-role-drift")
    for role, before in first["processes"].items():
        after = second["processes"][role]
        require(
            {k: before[k] for k in preservation.PROCESS_FIELDS}
            == {k: after[k] for k in preservation.PROCESS_FIELDS},
            "final-process-drift",
        )
        if "heartbeat_ns" in before:
            require(
                after["heartbeat_ns"] >= before["heartbeat_ns"],
                "final-heartbeat-regressed",
            )
