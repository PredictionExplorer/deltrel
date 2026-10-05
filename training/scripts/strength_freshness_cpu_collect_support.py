"""Read-only support observations under the caller's original IO deadline.

Prior witnesses require external source/request/capture authentication. Their
hashes prove internal consistency, not provenance. No service actions, inferred
creation times, first-seen age resets or target execution certificates exist.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import re
from typing import Any, Mapping

from scripts import strength_freshness_cpu_collect_facts as facts
from scripts import strength_freshness_cpu_collect_identity as identity_module
from scripts import strength_freshness_cpu_preservation as preservation

WITNESS = "strength-support-no-job-witness-v1"
ACTIVE = {"active", "activating", "deactivating"}
DYNAMIC = (
    "Id",
    "ActiveState",
    "SubState",
    "InvocationID",
    "Job",
    "InactiveExitTimestampMonotonic",
    "ActiveEnterTimestampMonotonic",
    "ActiveExitTimestampMonotonic",
    "InactiveEnterTimestampMonotonic",
    "StateChangeTimestampMonotonic",
)
SERVICE_DYNAMIC = (
    "MainPID",
    "ExecMainPID",
    "ControlGroup",
    "NRestarts",
    "ExecMainStartTimestampMonotonic",
    "ExecMainExitTimestampMonotonic",
    "ExecMainCode",
    "ExecMainStatus",
    "Result",
)


class SupportRefusal(ValueError):
    """A fixed non-private code; always means incomplete capture."""


def require(value: object, code: str) -> None:
    if not value:
        raise SupportRefusal(code)


def digest(value: object) -> str:
    return identity_module.sha(identity_module.encoded(value))


def number(value: object, *, positive=False) -> int:
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9]+", value), "support-integer"
    )
    assert isinstance(value, str)
    result = int(value)
    require(result >= int(positive), "support-integer-range")
    return result


def parse_jobs(raw: str) -> dict[str, dict[str, Any]]:
    """Exact systemd255 --no-legend --plain --full four-column table."""
    require(isinstance(raw, str), "jobs-text")
    result = {}
    ids = set()
    lines = raw.splitlines()
    require(len(lines) <= 4096, "jobs-count")
    for line in lines:
        fields = line.split()
        require(len(fields) == 4, "jobs-columns")
        job, unit, kind, state = fields
        job_id = number(job, positive=True)
        require(
            re.fullmatch(
                r"(?:[A-Za-z0-9_.@:-]|\\x[0-9a-fA-F]{2})+\.(?:service|timer|target|slice|socket|mount|automount|swap|path|scope|device)",
                unit,
            ),
            "jobs-unit",
        )
        require(
            re.fullmatch(r"[a-z-]+", kind) and state in {"waiting", "running"},
            "jobs-type-state",
        )
        require(job_id not in ids and unit not in result, "jobs-duplicate")
        ids.add(job_id)
        result[unit] = {"id": job_id, "kind": kind, "state": state}
    return result


def audit_pin(observation) -> dict[str, Any]:
    a = observation.audit
    return copy.deepcopy(
        {k: a[k] for k in ("operation", "subject", "read_start", "read_end", "raw")}
    )


def manager_id(value: Mapping[str, Any]) -> str:
    require(
        set(value) == {"boot_id", "pid", "start_ticks", "ppid", "cgroup", "namespaces"},
        "manager-fields",
    )
    require(
        value["pid"] == 1
        and type(value["pid"]) is int
        and value["ppid"] == 0
        and type(value["start_ticks"]) is int
        and value["start_ticks"] > 0,
        "manager-process",
    )
    require(
        isinstance(value["boot_id"], str)
        and value["boot_id"]
        and isinstance(value["cgroup"], str)
        and value["cgroup"].startswith("/"),
        "manager-boot-cgroup",
    )
    require(set(value["namespaces"]) == {"pid", "time"}, "manager-namespaces")
    for kind, item in value["namespaces"].items():
        require(
            set(item) == {"literal", "stat"}
            and re.fullmatch(kind + r":\[[0-9]+\]", item["literal"]),
            "manager-namespace-shape",
        )
        require(
            isinstance(item["stat"], dict)
            and all(
                type(item["stat"].get(k)) is int and item["stat"][k] >= 0
                for k in ("device", "inode")
            ),
            "manager-namespace-stat",
        )
    return digest({"format": "strength-support-manager-v1", **value})


def _clocks(first, last):
    require(
        set(first) == set(last) == {"boot_id", "monotonic_ns", "wall_ns"},
        "support-clock-fields",
    )
    require(
        first["boot_id"] == last["boot_id"]
        and isinstance(first["boot_id"], str)
        and first["boot_id"],
        "support-clock-boot",
    )
    for key in ("monotonic_ns", "wall_ns"):
        require(
            type(first[key]) is int
            and type(last[key]) is int
            and 0 <= first[key] <= last[key],
            "support-clock-order",
        )


def _witness(absence, proof):
    body = {"format": WITNESS, "absence": absence, "proof": proof}
    return {**body, "sha256": digest(body)}


def checked_witness(value, *, name, manager, now):
    require(
        isinstance(value, dict)
        and set(value) == {"format", "absence", "proof", "sha256"},
        "witness-fields",
    )
    require(
        value["format"] == WITNESS
        and value["sha256"]
        == digest({k: v for k, v in value.items() if k != "sha256"}),
        "witness-integrity",
    )
    a = value["absence"]
    p = value["proof"]
    require(
        set(a)
        == {
            "boot_id",
            "manager_id",
            "unit",
            "job",
            "read_start",
            "read_end",
            "raw_sha256",
        },
        "absence-fields",
    )
    require(
        set(p) == {"manager", "unit_reads", "jobs_reads"} and p["manager"] == manager,
        "witness-manager",
    )
    require(
        a["unit"] == name
        and a["boot_id"] == manager["boot_id"]
        and a["manager_id"] == manager_id(manager)
        and a["job"] is None,
        "witness-unit-binding",
    )
    require(a["raw_sha256"] == digest(p), "witness-raw-binding")
    require(len(p["unit_reads"]) == len(p["jobs_reads"]) == 2, "witness-read-count")
    for kind, reads in (("unit", p["unit_reads"]), ("jobs", p["jobs_reads"])):
        for row in reads:
            require(
                set(row) == {"audit", "job"} and row["job"] is None, "witness-no-job"
            )
            audit = row["audit"]
            require(
                set(audit) == {"operation", "subject", "read_start", "read_end", "raw"}
                and set(audit["raw"]) == {"sha256", "bytes"},
                "witness-audit-fields",
            )
            require(
                preservation.sha(audit["raw"]["sha256"])
                and type(audit["raw"]["bytes"]) is int
                and audit["raw"]["bytes"] >= 0,
                "witness-audit-pin",
            )
            require(
                audit["operation"] == "query"
                and audit["subject"] == ("unit:" + name if kind == "unit" else "jobs"),
                "witness-query-binding",
            )
            _clocks(a["read_start"], audit["read_start"])
            _clocks(audit["read_start"], audit["read_end"])
            _clocks(audit["read_end"], a["read_end"])
    require(
        a["read_start"] == p["jobs_reads"][0]["audit"]["read_start"],
        "witness-lower-bound",
    )
    _clocks(a["read_start"], a["read_end"])
    _clocks(a["read_end"], now)
    return a


@dataclass(frozen=True)
class SupportCapture:
    support: dict[str, dict[str, Any]]
    witnesses: dict[str, dict[str, Any]]
    provenance: dict[str, Any]


class SupportCollector:
    def __init__(self, identity):
        self.identity = identity
        self.io = identity.io
        self.reg = identity.reg
        self.policy = self.reg["policy"]

    def _unit(self, name):
        obs = self.io.query("unit", name)
        raw = self.identity.take(obs)
        return identity_module.properties(raw), audit_pin(obs)

    def _jobs(self):
        obs = self.io.query("jobs")
        return parse_jobs(self.identity.take(obs)), audit_pin(obs)

    def _joined(self, name, p, jobs):
        require("Job" in p, "missing-unit-job")
        job = jobs.get(name)
        if p["Job"] == "":
            require(job is None, "job-unit-list-mismatch")
        else:
            require(
                job is not None and number(p["Job"], positive=True) == job["id"],
                "job-unit-list-mismatch",
            )
        return job

    def _members(self, name, p, monitor):
        members = self.identity.take(self.io.members(name))
        main = number(p["MainPID"])
        require(not main or main in {x.pid for x in members}, "support-main-not-member")
        if p["ActiveState"] == "inactive":
            require(not members and not main, "inactive-support-members")
        if monitor is not None:
            require(
                len(members) == 1
                and members[0].pid == main == monitor["pid"]
                and members[0].start_ticks == monitor["start_ticks"]
                and members[0].cgroup == monitor["cgroup"] == p["ControlGroup"],
                "monitor-lifetime-drift",
            )
            require(
                p["InvocationID"] == monitor["invocation_id"]
                and number(p["NRestarts"]) == monitor["restarts"],
                "monitor-invocation-drift",
            )
            process = self.identity.take(self.io.process(members[0]))
            require(
                self.identity.origin("monitor", process) == monitor["origin_sha256"],
                "monitor-origin-drift",
            )
        return tuple((x.pid, x.start_ticks, x.cgroup) for x in members)

    def capture(self, *, monitors, previous_witnesses=None) -> SupportCapture:
        names = sorted(self.policy["support"])
        require(
            set(names)
            == {n for n, s in self.reg["units"].items() if s["kind"] != "runtime"},
            "support-inventory",
        )
        require(
            set(monitors)
            == {n for n in names if self.reg["units"][n]["kind"] == "long_running"},
            "monitor-inventory",
        )
        prior = {} if previous_witnesses is None else previous_witnesses
        require(
            isinstance(prior, Mapping) and set(prior) <= set(names), "witness-inventory"
        )
        audit_start = len(self.identity.audit)
        first_manager = self.identity.take(self.io.manager_identity())
        mid = manager_id(first_manager)
        actual_default = self.identity.take(self.io.query("default-target"))
        require(
            actual_default.rstrip("\n") == self.reg["boot"]["default_target"],
            "default-target-drift",
        )
        targets = {}
        for target in self.io.scope.targets:
            p = identity_module.properties(
                self.identity.take(self.io.query("registered-target", target))
            )
            require(
                p.get("Id") == target and p.get("LoadState") == "loaded",
                "boot-target-observation",
            )
            targets[target] = p
        jobs_a, jobs_a_audit = self._jobs()
        props_a = {}
        unit_a_audit = {}
        static = {}
        members_a = {}
        for name in names:
            p, unit_a_audit[name] = self._unit(name)
            props_a[name] = p
            kind = self.reg["units"][name]["kind"]
            static[name] = {"kind": kind, **self.identity.unit_static(name, p, targets)}
            require(
                static[name] == self.policy["support"][name], "support-static-drift"
            )
            self._joined(name, p, jobs_a)
            if kind != "timer":
                members_a[name] = self._members(name, p, monitors.get(name))
        jobs_b, jobs_b_audit = self._jobs()
        props_b = {}
        unit_b_audit = {}
        for name in names:
            p, unit_b_audit[name] = self._unit(name)
            props_b[name] = p
            fields = DYNAMIC + (
                SERVICE_DYNAMIC if self.reg["units"][name]["kind"] != "timer" else ()
            )
            require(
                all(
                    k in p and k in props_a[name] and p[k] == props_a[name][k]
                    for k in fields
                ),
                "support-unit-raced",
            )
            require(
                self._joined(name, p, jobs_b) == jobs_a.get(name), "support-job-raced"
            )
            require(
                {
                    "kind": self.reg["units"][name]["kind"],
                    **self.identity.unit_static(name, p, targets),
                }
                == static[name],
                "support-static-raced",
            )
            if self.reg["units"][name]["kind"] != "timer":
                require(
                    self._members(name, p, monitors.get(name)) == members_a[name],
                    "support-members-raced",
                )
        require(
            self.identity.take(self.io.query("default-target")).rstrip("\n")
            == actual_default.rstrip("\n"),
            "default-target-raced",
        )
        for target in self.io.scope.targets:
            p = identity_module.properties(
                self.identity.take(self.io.query("registered-target", target))
            )
            require(
                p.get("Id") == target and p.get("LoadState") == "loaded",
                "boot-target-observation",
            )
            for edge in self.reg["boot"]["edges"]:
                if edge["from"] == target:
                    require(
                        edge["to"] in p.get(edge["relation"], "").split(),
                        "boot-edge-raced",
                    )
        last_manager = self.identity.take(self.io.manager_identity())
        require(last_manager == first_manager, "manager-raced")
        now = self.identity.clock()
        _clocks(jobs_a_audit["read_start"], now)
        require(now["boot_id"] == first_manager["boot_id"], "manager-clock-binding")
        result = {}
        witnesses = {}
        for name in names:
            p = props_b[name]
            kind = static[name]["kind"]
            job = jobs_b.get(name)
            common = {"boot_id": now["boot_id"], "manager_id": mid, "unit": name}
            if job is None:
                proof = {
                    "manager": first_manager,
                    "unit_reads": [
                        {"audit": unit_a_audit[name], "job": None},
                        {"audit": unit_b_audit[name], "job": None},
                    ],
                    "jobs_reads": [
                        {"audit": jobs_a_audit, "job": None},
                        {"audit": jobs_b_audit, "job": None},
                    ],
                }
                absence = {
                    **common,
                    "job": None,
                    "read_start": jobs_a_audit["read_start"],
                    "read_end": now,
                    "raw_sha256": digest(proof),
                }
                witnesses[name] = _witness(absence, proof)
                checked_witness(
                    witnesses[name], name=name, manager=first_manager, now=now
                )
                public_job = None
            else:
                require(
                    kind == "oneshot" and job["kind"] == "start", "support-job-kind"
                )
                require(name in prior, "unknown-job-age")
                absence = checked_witness(
                    prior[name],
                    name=name,
                    manager=first_manager,
                    now=jobs_a_audit["read_start"],
                )
                current = {
                    **common,
                    **job,
                    "read_start": jobs_a_audit["read_start"],
                    "read_end": jobs_b_audit["read_end"],
                    "creation_source_sha256": None,
                }
                age = facts.job_age_seconds(current=current, now=now, absence=absence)
                require(
                    age <= self.policy["maximum_support_seconds"], "support-job-overdue"
                )
                public_job = {"kind": "start", "age_seconds": age}
                witnesses[name] = copy.deepcopy(prior[name])
            running = None
            exit_code = None
            result_value = None
            process = None
            if kind == "timer":
                require(p["ActiveState"] == "active", "timer-inactive")
            else:
                result_value = p["Result"]
                exit_code = number(p["ExecMainStatus"])
                if kind == "long_running":
                    require(
                        p["ActiveState"] == "active"
                        and p["SubState"] == "running"
                        and result_value in {"", "success"},
                        "monitor-unhealthy",
                    )
                    process = copy.deepcopy(monitors[name])
                    require(
                        process == self.policy["expected_monitors"][name],
                        "monitor-policy-drift",
                    )
                elif p["ActiveState"] in ACTIVE:
                    running = facts.running_seconds(
                        now_monotonic_ns=now["monotonic_ns"],
                        activation_start_us=number(
                            p["InactiveExitTimestampMonotonic"], positive=True
                        ),
                        invocation_id=p["InvocationID"],
                        expected_invocation_id=props_a[name]["InvocationID"],
                    )
                    require(
                        running <= self.policy["maximum_support_seconds"],
                        "support-activation-overdue",
                    )
                else:
                    require(
                        p["ActiveState"] == "inactive"
                        and result_value in {"", "success"}
                        and exit_code == 0,
                        "support-inactive-failure",
                    )
            result[name] = {
                **static[name],
                "active": p["ActiveState"],
                "result": result_value,
                "exit_code": exit_code,
                "process": process,
                "running_seconds": running,
                "job": public_job,
            }
        return SupportCapture(
            result,
            witnesses,
            {
                "format": "strength-support-capture-v1",
                "clock": now,
                "manager": last_manager,
                "manager_id": mid,
                # Later outer reads must not silently contradict these actual
                # selected dynamic facts. No Environment or Exec data is copied.
                "unit_states": {
                    name: {
                        key: props_b[name][key]
                        for key in DYNAMIC
                        + (
                            SERVICE_DYNAMIC
                            if self.reg["units"][name]["kind"] != "timer"
                            else ()
                        )
                    }
                    for name in names
                },
                "jobs": {n: jobs_b[n] for n in names if n in jobs_b},
                "audit": copy.deepcopy(self.identity.audit[audit_start:]),
                "prior_witness_authentication": "external caller required",
                "execution_qualified": False,
            },
        )
