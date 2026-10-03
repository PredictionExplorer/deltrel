from __future__ import annotations

import copy
import json
from pathlib import Path
import runpy
from typing import cast

import pytest

from scripts import strength_freshness_cpu_collector as c
from scripts import strength_freshness_cpu_collect_identity as ident
from scripts import strength_freshness_cpu_collect_records as records
from scripts import strength_freshness_cpu_preservation as preservation
from scripts import strength_freshness_cpu_readonly as readonly
from scripts import strength_freshness_cpu_collect_support as support

# Reuse frozen local fixture definitions by explicit sibling paths, independent
# of pytest's import mode/cwd. All actual collectors remain the production code.
identity_fixtures = runpy.run_path(
    str(Path(__file__).with_name("test_strength_freshness_cpu_collect_identity.py"))
)
record_fixtures = runpy.run_path(
    str(Path(__file__).with_name("test_strength_freshness_cpu_collect_records.py"))
)
BASE = 1_800_000_000_000_000_000
MODEL = "sha256-" + "a" * 64
INV = "1" * 32


class FakeIO(identity_fixtures["FakeIO"]):
    def __init__(self, reg):
        super().__init__(reg)
        self.ns = 100_000_000_000
        self.deadline = 120.0
        self.bytes_used = 0
        self.budget = 32 * 1024**2
        self.raw_processes = {}
        self.roles = {}
        self.actor_counts = {}
        self.stalled = False
        self.reset_counters = False
        self.learner_step = 20
        self.default = "multi-user.target"
        self.changed_source = False
        self.final_support_fault = None
        self.source_reads = 0
        self.owner_capture_count = 0
        self.foreign_final = False
        self.unknown_jobs = False
        self.reg = reg

    def stamp(self):
        return {"boot_id": "boot", "monotonic_ns": self.ns, "wall_ns": BASE + self.ns}

    def observation(self, operation, subject, value, raw=None, **metadata):
        self.ns += 1_000_000
        if self.ns / 1e9 >= self.deadline:
            raise readonly.ReadRefusal("absolute-read-deadline")
        if raw is None:
            raw = ident.encoded(value)
        self.bytes_used += len(raw)
        if self.bytes_used > self.budget:
            raise readonly.ReadRefusal("capture-byte-budget")
        return readonly.Observation(
            copy.deepcopy(value),
            {
                "operation": operation,
                "subject": subject,
                "read_start": self.stamp(),
                "read_end": self.stamp(),
                "raw": {"sha256": ident.sha(raw), "bytes": len(raw)},
                **metadata,
            },
        )

    def clock(self):
        obs = self.observation("clock", "collector", self.stamp())
        return readonly.Observation(copy.deepcopy(obs.audit["read_end"]), obs.audit)

    def sleep(self, seconds):
        self.ns += int(seconds * 1e9)

    def read(self, key):
        if key == "source":
            self.source_reads += 1
            if self.source_reads == 2:
                p = self.props["backup.service"]
                if self.final_support_fault == "job":
                    self.unknown_jobs = True
                    p["Job"] = "9"
                elif self.final_support_fault == "failure":
                    p.update(
                        ActiveState="failed",
                        SubState="failed",
                        Result="exit-code",
                        ExecMainStatus="1",
                    )
                elif self.final_support_fault == "invocation":
                    p["InvocationID"] = "3" * 32
        if key in records.WORKERS:
            role = key
            row = record_fixtures["heartbeat"](role, self.roles[role])
            row.update(
                heartbeat_ns=BASE + self.ns - 1_000_000,
                progress_ns=BASE + self.ns - 2_000_000,
                step=self.learner_step,
            )
            if role in records.ACTORS:
                count = self.actor_counts.get(role, 0) + (0 if self.stalled else 1)
                if self.reset_counters and role == records.ACTORS[0] and count >= 3:
                    count = 0
                self.actor_counts[role] = count
                row.update(
                    phase="shared_cohorts",
                    inference={
                        "worker_phase": "idle",
                        "worker_phase_since_ns": BASE + self.ns - 500_000_000,
                        "failed_requests": 0,
                        "worker_failures": 0,
                        "physical_inference": {
                            "neural_calls": 100 + count,
                            "neural_rows": 10000 + count * 512,
                        },
                    },
                )
            elif role == "learner":
                row["phase"] = "update_to_data_wait"
            else:
                row["phase"] = "selfplay" if role.startswith("actor") else "arena_batch"
            self.data[key] = ident.encoded(row)
        if key == "coordinator":
            row = json.loads(self.data[key])
            row["timestamp_ns"] = BASE + self.ns - 1_000_000
            self.data[key] = ident.encoded(row)
        if key in records.COHORTS:
            row = json.loads(self.data[key])
            row.update(
                heartbeat_ns=BASE + self.ns - 1_000_000,
                progress_ns=BASE + self.ns - 2_000_000,
            )
            self.data[key] = ident.encoded(row)
        if key == "source" and self.changed_source and self.owner_capture_count >= 2:
            self.data[key] = b"changed source"
        return super().read(key)

    def query(self, kind, name=None):
        if kind == "unit":
            self.calls.append(("unit", name))
            raw = "\n".join(k + "=" + v for k, v in self.props[name].items())
        elif kind == "default-target":
            raw = self.default + "\n"
        elif kind == "registered-target":
            raw = f"Id={name}\nLoadState=loaded\nWants=\nRequires=\n"
        elif kind == "jobs":
            raw = "" if not self.unknown_jobs else "9 backup.service start waiting\n"
        elif kind == "gpu-inventory":
            raw = "\n".join(f"{i}, GPU-{i}" for i in range(8))
        elif kind == "gpu-owners":
            raw = "\n".join(
                f"{self.roles[role]}, {uuid}"
                for uuid, role in self.reg["policy"]["gpu_roles"].items()
            )
        else:
            raise AssertionError(kind)
        return self.observation(
            "query", kind + (":" + name if name else ""), raw, raw.encode()
        )

    def members(self, name):
        if name == ident.RUNTIME:
            self.owner_capture_count += 1
        rows = [
            readonly.ProcessAdmission(pid, p["start_ticks"], p["cgroup"])
            for pid, p in self.raw_processes.items()
            if p["cgroup"] == "/system.slice/" + name
        ]
        if (
            self.foreign_final
            and name == ident.RUNTIME
            and self.owner_capture_count >= 3
        ):
            rows = rows[:-1]
        return self.observation("cgroup-members", name, tuple(rows), b"member metadata")

    def process(self, admission, *, maps=True):
        result = copy.deepcopy(self.raw_processes[admission.pid])
        result["maps"] = result["maps"] if maps else None
        return self.observation(
            "process", str(admission.pid), result, b"private component hashes"
        )

    def namespaces(self, subject):
        return self.observation(
            "namespaces",
            "registered",
            {
                k: {"literal": k + ":[1]", "stat": {"device": 0, "inode": 1}}
                for k in ("pid", "time")
            },
        )

    def birth_bracket(self):
        return self.observation(
            "birth-bracket",
            "collector",
            {
                "boot_id": "boot",
                "monotonic_before_ns": self.ns,
                "boottime_before_ns": self.ns,
                "wall_ns": BASE + self.ns,
                "boottime_after_ns": self.ns + 1,
                "monotonic_after_ns": self.ns + 1,
                "clock_ticks_per_second": 100,
            },
        )

    def manager_identity(self):
        value = {
            "boot_id": "boot",
            "pid": 1,
            "start_ticks": 1,
            "ppid": 0,
            "cgroup": "/init.scope",
            "namespaces": {
                k: {"literal": k + ":[1]", "stat": {"device": 0, "inode": 1}}
                for k in ("pid", "time")
            },
        }
        return self.observation("manager-identity", "pid1", value)

    def tail(self, key):
        self.calls.append(("tail", key))
        raw = self.data[key]
        return self.observation("tail", key, raw, raw, offset=0, end_offset=len(raw))


@pytest.fixture
def sample():
    reg, original, private = identity_fixtures["identity_fixture"]()
    policy, before, _, _ = runpy.run_path(
        str(Path(__file__).with_name("test_strength_freshness_cpu_preservation.py"))
    )["observations"].__wrapped__()
    reg["scope"]["units"]["backup.service"] = "service"
    reg["scope"]["files"]["fragment-backup"] = {
        "path": "/etc/systemd/system/backup.service"
    }
    reg["units"]["backup.service"] = dict(
        kind="oneshot",
        fragment="fragment-backup",
        dropins=[],
        environment_files=[],
        owned_links=[],
    )
    original.props["backup.service"] = {
        **copy.deepcopy(original.props["monitor.service"]),
        "Id": "backup.service",
        "Names": "backup.service",
        "FragmentPath": "/etc/systemd/system/backup.service",
    }
    reg["policy"].update(
        {k: copy.deepcopy(v) for k, v in policy.items() if k != "static"}
    )
    reg["policy"]["static"] = {
        **policy["static"],
        "runtime_name": ident.RUNTIME,
        "runtime_cgroup": "/system.slice/" + ident.RUNTIME,
        "continuation_started_ns": BASE - 1000,
        "native_sha256": "a" * 64,
    }
    roles = ["controller", "coordinator", *records.WORKERS]
    reg["policy"]["workers"] = list(records.WORKERS)
    reg["policy"]["cohorts"] = list(records.COHORTS)
    reg["cohorts"] = {
        name: {
            "key": name,
            "worker": name,
            "parent_role": name.rsplit("-cohort-", 1)[0],
        }
        for name in records.COHORTS
    }
    for name in records.COHORTS:
        reg["scope"]["files"][name] = {"path": "/run/" + name}
    reg["birth_reference"].update(
        offset_lower_ns=BASE - 1000, offset_upper_ns=BASE + 1000, max_bracket_ns=100
    )
    reg["birth_reference"]["bounds"] = {role: BASE + 11_000_000_000 for role in roles}
    reg["policy"]["learner_birth_upper_ns"] = BASE + 11_000_000_000
    reg["auxiliary_policy"] = {}
    reg["policy"]["expected_processes"] = {role: {} for role in roles}
    reg["policy"]["expected_monitors"] = {"monitor.service": {}}
    reg["policy"]["support"] = {}
    io = FakeIO(reg)
    io.props = copy.deepcopy(original.props)
    io.data.update(original.data)
    for i, role in enumerate([*roles, "monitor"]):
        pid = 200 if role == "monitor" else 100 + i
        io.roles[role] = pid
        group = "/system.slice/" + (
            "monitor.service" if role == "monitor" else ident.RUNTIME
        )
        row = copy.deepcopy(private)
        row.update(
            pid=pid,
            start_ticks=100 + i,
            ppid=1
            if role in {"controller", "monitor"}
            else io.roles["controller" if role == "coordinator" else "coordinator"],
            cgroup=group,
        )
        io.raw_processes[pid] = row
    for name, props in io.props.items():
        role = "controller" if name == ident.RUNTIME else "monitor"
        props.update(
            {
                k: "0"
                for k in support.DYNAMIC + support.SERVICE_DYNAMIC
                if k not in {"Id"}
            }
        )
        props.update(
            ActiveState="active",
            SubState="waiting" if name.endswith(".timer") else "running",
            InvocationID="2" * 32 if role == "monitor" else INV,
            Job="",
            MainPID=str(io.roles[role]),
            ExecMainPID=str(io.roles[role]),
            ControlGroup="/system.slice/" + name,
            NRestarts="0",
            Result="success",
            ExecMainStatus="0",
        )
    io.props["backup.service"].update(
        ActiveState="inactive",
        SubState="dead",
        MainPID="0",
        ExecMainPID="0",
        ControlGroup="/system.slice/backup.service",
    )
    manifest = b"qualified source inventory\n"
    io.data["source_manifest"] = manifest
    reg["policy"]["static"]["source_manifest_sha256"] = ident.sha(manifest)
    # Compute the predeclared synthetic policy with the real pure identity APIs.
    temp = ident.IdentityCollector(reg, cast(readonly.ReadOnlyIO, io))
    for role, pid in io.roles.items():
        row = {
            "pid": pid,
            "start_ticks": io.raw_processes[pid]["start_ticks"],
            "cgroup": io.raw_processes[pid]["cgroup"],
            "invocation_id": "2" * 32 if role == "monitor" else INV,
            "restarts": 0,
            "origin_sha256": temp.origin(role, io.raw_processes[pid]),
        }
        if role == "monitor":
            reg["policy"]["expected_monitors"]["monitor.service"] = row
        else:
            reg["policy"]["expected_processes"][role] = row
    for name in reg["units"]:
        result = temp.unit_static(name, io.props[name], {})
        if name == ident.RUNTIME:
            for short, full in [
                ("definition_sha256", "runtime_definition_sha256"),
                ("environment_sha256", "runtime_environment_sha256"),
                ("boot_links_sha256", "runtime_boot_links_sha256"),
            ]:
                reg["policy"]["static"][full] = result[short]
        else:
            reg["policy"]["support"][name] = {
                "kind": reg["units"][name]["kind"],
                **result,
            }
    coord = {
        "schema_version": 1,
        "timestamp_ns": BASE + 99_000_000_000,
        "coordinator_pid": io.roles["coordinator"],
        "state": "running",
        "draining": False,
        "failure": None,
        "hardware_failure_reason": None,
        "hardware_failure_class": None,
        "workers": {
            role: {
                "pid": io.roles[role],
                "role": "learner"
                if role == "learner"
                else "arena"
                if role == "arena-promotion"
                else "actor",
                "state": "running",
                "restart_count": 0,
                "failure_reason": None,
                "failure_class": None,
                "failure_exit_code": None,
            }
            for role in records.WORKERS
        },
    }
    io.data["coordinator"] = ident.encoded(coord)
    for name in records.COHORTS:
        parent = name.rsplit("-cohort-", 1)[0]
        row = record_fixtures["heartbeat"](name, io.roles[parent])
        row.update(
            model_role="champion",
            requested_model_role="champion",
            model_version=MODEL,
            cumulative_games=100,
            cohort=0,
        )
        io.data[name] = ident.encoded(row)
    run = {
        "schema_version": 1,
        "created_ns": 1,
        "run_id": "run",
        "generation_family": "family",
    }
    continuation = {
        "schema_version": 1,
        "phase": "continuation",
        "continuation_started_ns": BASE - 1000,
        "plan_sha256": "a" * 64,
        "profile": io.scope.files["profile"].path,
        "provisioned_gpus": 8,
        "attempts": [
            {"pid": io.roles["coordinator"], "started_ns": BASE, "status": "running"}
        ],
    }
    io.data["run"] = ident.encoded(run)
    io.data["continuation"] = ident.encoded(continuation)
    io.data["profile"] = b"registered profile\n"
    checksum = ident.sha(io.data["profile"])
    io.data["profile_authority"] = (
        checksum + "  " + io.scope.files["profile"].path + "\n"
    ).encode()
    io.data["run_source"] = io.data["release_source"] = (
        preservation.R3_SOURCE_COMMIT + "\n"
    ).encode()
    reg["policy"]["static"]["profile_sha256"] = checksum
    pointer = record_fixtures["pointer"]()
    pointer.update(run_id="run", generation_family="family")
    io.data["champion"] = ident.encoded(pointer)
    metrics = copy.deepcopy(before["progress"]["metrics"])
    for i, row in enumerate(metrics):
        row.update(
            schema_version=1, timestamp_ns=BASE + (98 + i) * 10**9, step=10 * (i + 1)
        )
        row["ema"]["num_updates"] = row["step"]
    reg["policy"]["recipe"]["bootstrap_step"] = 0
    io.data["metrics"] = b"".join(ident.encoded(row) + b"\n" for row in metrics)
    io.scope = ident.scope_from(reg["scope"])
    io.ns = 100_000_000_000
    io.bytes_used = 0
    io.calls = []
    return reg, io


def collect(sample, **kwargs):
    reg, io = sample
    return c.collect_capture(
        reg,
        cast(readonly.ReadOnlyIO, io),
        verified_champions={MODEL: "b" * 64},
        phase=kwargs.pop("phase", "before"),
        sleep=io.sleep,
        **kwargs,
    )


def cleanup():
    return {"boot_id": "boot", "monotonic": 97.0, "wall_ns": BASE + 97_000_000_000}


def test_before_real_helper_composition_is_private_measurement(sample):
    result = collect(sample)
    assert result["execution_qualified"] is result["cuda_qualified"] is False
    assert "physical_work" not in result["capture"]
    assert (
        len(result["capture"]["processes"]) == 11
        and len(result["capture"]["cohorts"]) == 24
    )
    assert result["provenance"]["capture_sha256"] == preservation.digest(
        result["capture"]
    )
    assert identity_fixtures["SECRET"] not in json.dumps(result)
    assert result["capture"]["clock"]["wall_ns"] >= max(
        x["read_end"]["wall_ns"] for x in result["provenance"]["raw_inventory"]
    )


@pytest.mark.parametrize("kind", ["learner_metrics", "actor_broker"])
def test_after_real_producer_paths(sample, kind):
    result = collect(sample, phase="after", cleanup_clock=cleanup(), physical_kind=kind)
    assert result["capture"]["physical_work"]["kind"] == kind
    if kind == "actor_broker":
        assert len(result["capture"]["physical_work"]["actors"]) == 6
    assert not result["execution_qualified"]


@pytest.mark.parametrize(
    "fault",
    ["default", "source", "owner", "champion", "profile", "cohort", "counter", "birth"],
)
def test_true_fact_mismatch_refuses(sample, fault):
    reg, io = sample
    if fault == "default":
        io.default = "other.target"
    if fault == "source":
        io.changed_source = True
    if fault == "owner":
        io.foreign_final = True
    if fault == "champion":
        row = json.loads(io.data["champion"])
        row["model_identity"] = "sha256-" + "f" * 64
        io.data["champion"] = ident.encoded(row)
    if fault == "profile":
        io.data["profile"] += b" "
    if fault == "cohort":
        row = json.loads(io.data[records.COHORTS[0]])
        row["pid"] = 999
        io.data[records.COHORTS[0]] = ident.encoded(row)
    if fault == "counter":
        reg["policy"]["recipe"]["learning_rates"][0] = 9
    if fault == "birth":
        reg["birth_reference"]["bounds"][records.ACTORS[0]] = BASE + 200_000_000_000
    with pytest.raises(c.CaptureRefusal) as caught:
        collect(sample)
    assert identity_fixtures["SECRET"] not in json.dumps(caught.value.provenance)


def test_original_deadline_and_budget_are_not_reset(sample):
    reg, io = sample
    io.stalled = True
    io.deadline = io.ns / 1e9 + 1
    with pytest.raises(c.CaptureRefusal):
        collect(
            sample, phase="after", cleanup_clock=cleanup(), physical_kind="actor_broker"
        )
    assert io.ns / 1e9 < 102 and io.deadline == 101
    reg, io = sample
    io.deadline = 120
    io.budget = 1
    with pytest.raises(c.CaptureRefusal):
        collect(sample)


def test_cleanup_is_required_and_cannot_be_future(sample):
    with pytest.raises(c.CaptureRefusal):
        collect(sample, phase="after")
    future = {"boot_id": "boot", "monotonic": 1000, "wall_ns": BASE + 1000 * 10**9}
    with pytest.raises(c.CaptureRefusal, match="capture-before-cleanup"):
        collect(sample, phase="after", cleanup_clock=future)


def test_after_reuses_only_validated_previous_absence_packet(sample):
    before = collect(sample)
    reg, io = sample
    prior = before["provenance"]["support_witnesses"]
    mark = before["capture"]["clock"]
    clean = {
        "boot_id": "boot",
        "monotonic": mark["monotonic"] + 0.001,
        "wall_ns": mark["wall_ns"] + 1_000_000,
    }
    io.ns = clean["wall_ns"] - BASE + 1_000_000
    io.unknown_jobs = True
    io.props["backup.service"]["Job"] = "9"
    after = collect(
        sample,
        phase="after",
        cleanup_clock=clean,
        physical_kind="actor_broker",
        previous_support_witnesses=prior,
    )
    assert after["capture"]["support"]["backup.service"]["job"]["age_seconds"] >= 1
    assert (
        after["provenance"]["support_witnesses"]["backup.service"]
        == prior["backup.service"]
    )
    assert (
        after["provenance"]["support_age_projection"]["added_upper_bound_seconds"] >= 1
    )


def test_unknown_job_never_gets_first_seen_age_zero(sample):
    _, io = sample
    io.unknown_jobs = True
    io.props["backup.service"]["Job"] = "9"
    with pytest.raises(c.CaptureRefusal, match="unknown-job-age"):
        collect(sample)


def test_unchecked_prior_witness_is_not_returned_on_refusal(sample):
    _, io = sample
    io.default = "wrong.target"
    raw = {"private": identity_fixtures["SECRET"]}
    with pytest.raises(c.CaptureRefusal) as caught:
        collect(
            sample,
            phase="after",
            cleanup_clock=cleanup(),
            previous_support_witnesses=raw,
        )
    assert identity_fixtures["SECRET"] not in json.dumps(caught.value.provenance)


def test_unbounded_deadline_refuses(sample):
    _, io = sample
    io.deadline = 10**9
    with pytest.raises(c.CaptureRefusal, match="capture-window-bound"):
        collect(sample)


def test_complete_malformed_metric_line_refuses(sample):
    _, io = sample
    io.data["metrics"] += b"broken complete line\n"
    with pytest.raises(c.CaptureRefusal, match="malformed-json"):
        collect(sample)


def test_after_waits_for_actual_new_learner_rows(sample):
    _, io = sample
    original_sleep = io.sleep

    def publish(seconds):
        original_sleep(seconds)
        io.learner_step = 40
        rows = [json.loads(x) for x in io.data["metrics"].splitlines()]
        for index, row in enumerate(rows):
            row["step"] = 30 + index * 10
            row["ema"]["num_updates"] = row["step"]
            row["timestamp_ns"] = BASE + io.ns - (2 - index) * 1_000_000
        io.data["metrics"] = b"".join(ident.encoded(x) + b"\n" for x in rows)

    io.sleep = publish
    clean = {"boot_id": "boot", "monotonic": 99.999, "wall_ns": BASE + 99_999_000_000}
    result = collect(sample, phase="after", cleanup_clock=clean)
    assert result["provenance"]["polls"] >= 1
    assert [row["step"] for row in result["capture"]["progress"]["metrics"]] == [30, 40]
    assert all(
        row["timestamp_ns"] > clean["wall_ns"]
        for row in result["capture"]["progress"]["metrics"]
    )


def test_broker_counter_reset_refuses_instead_of_new_generation(sample):
    _, io = sample
    io.reset_counters = True
    with pytest.raises(c.CaptureRefusal, match="broker-counter-reset"):
        collect(
            sample, phase="after", cleanup_clock=cleanup(), physical_kind="actor_broker"
        )


def test_unexpected_exception_text_is_not_public(sample):
    _, io = sample
    io.data["source"] = ValueError("private-sensitive-token")
    with pytest.raises(c.CaptureRefusal, match="^capture-component-refusal$") as error:
        collect(sample)
    assert "private-sensitive-token" not in json.dumps(error.value.provenance)


@pytest.mark.parametrize("fault", ["job", "failure", "invocation"])
def test_later_unit_observation_cannot_contradict_retained_support(sample, fault):
    _, io = sample
    io.final_support_fault = fault
    with pytest.raises(c.CaptureRefusal, match="final-support-state-drift"):
        collect(sample)


def test_known_failed_query_audit_hashes_survive_refusal(sample):
    _, io = sample
    original = io.query
    audit = {
        "operation": "query",
        "subject": "default-target",
        "read_start": io.stamp(),
        "read_end": {**io.stamp(), "boot_id": None},
        "boot_rechecked": False,
        "stdout_sha256": "a" * 64,
        "stdout_bytes": 5,
        "stderr_sha256": "b" * 64,
        "stderr_bytes": 10,
        "returncode": -9,
    }

    def fail(kind, name=None):
        if kind == "default-target":
            raise readonly.ReadRefusal("command-timeout", audit)
        return original(kind, name)

    io.query = fail
    with pytest.raises(c.CaptureRefusal, match="command-timeout") as caught:
        collect(sample)
    assert caught.value.provenance["failed_operation"] == audit
    assert "stdout" not in caught.value.provenance["failed_operation"]
    assert identity_fixtures["SECRET"] not in json.dumps(caught.value.provenance)


def test_arbitrary_exception_audit_is_not_serialized(sample):
    _, io = sample

    class OtherFailure(ValueError):
        audit = {"stdout": "private-sensitive-token"}

    io.data["source"] = OtherFailure("private-sensitive-token")
    with pytest.raises(c.CaptureRefusal) as caught:
        collect(sample)
    assert "failed_operation" not in caught.value.provenance
    assert "private-sensitive-token" not in json.dumps(caught.value.provenance)


def test_actual_core_provenance_keeps_registered_exec_absence(sample):
    reg, io = sample
    reg["empty_property_rules"]["ExecStartPre"] = "systemd-255-empty-ExecStartPre"
    del io.props[ident.RUNTIME]["ExecStartPre"]
    result = collect(sample)
    observations = result["provenance"]["identity_derivations"][
        "omitted_execution_properties"
    ]
    assert observations and all(
        row["observed_present"] is False for row in observations
    )
    assert all(
        row["normalization_rules"] == {"ExecStartPre": "systemd-255-empty-ExecStartPre"}
        for row in observations
    )
    assert "ExecStartPre" not in io.props[ident.RUNTIME]
    assert identity_fixtures["SECRET"] not in json.dumps(result)
