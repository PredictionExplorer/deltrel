"""Closed capability tests use fake I/O; no host service/proc/GPU calls."""

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from scripts import strength_freshness_cpu_qualification as q

NONCE = "1" * 32
BOOT = "11111111-1111-1111-1111-111111111111"
PREFIX = "edgeconnect-cpuqual-" + NONCE
INPUT = Path("/opt/cpuqual-fixture")
SCRATCH = Path("/run") / PREFIX


def artifact(path, data):
    return {"path": str(path), "sha256": q.sha(data), "bytes": len(data)}


def make_plan():
    p = {
        "format": q.FORMAT,
        "schema_version": 1,
        "attempt_id": "case-one",
        "nonce": NONCE,
        "boot_id": BOOT,
        "protocol_sha256": q.PROTOCOL_SHA,
        "addenda_sha256": list(q.ADDENDA),
        "limits": q.LIMITS.copy(),
        "claims": q.CLAIMS.copy(),
        "input_root": str(INPUT),
        "scratch_root": str(SCRATCH),
        "protected_roots": ["/prod/run", "/prod/release", "/prod/r4"],
        "runtime_inputs": {
            "runtime_root": "/prod/r4",
            "source_commit": "7dca37252714bbe0c52d35a380ec175d743d1938",
            "source_marker": artifact(
                "/prod/SOURCE_COMMIT", b"7dca37252714bbe0c52d35a380ec175d743d1938\n"
            ),
            "source_manifest": artifact("/prod/SOURCE_SHA256SUMS", b"sums"),
            "pyvenv": artifact("/prod/r4/.venv/pyvenv.cfg", b"venv"),
            "native_wrapper": artifact("/prod/r4/.venv/native/__init__.py", b"wrapper"),
            "native_binary": artifact("/prod/r4/.venv/native/native.so", b"native"),
            "qualification": artifact(INPUT / "runtime-qualification.json", b"{}"),
        },
        "control_root": "/control/training",
        "source_pins": [],
        "units": {},
        "files": {},
        "python": {
            **artifact("/prod/r4/.venv/bin/python", b"python"),
            "resolved_path": "/usr/bin/python3",
        },
        "boot_topology": {
            "default_target": "multi-user.target",
            "target_paths": {"multi-user.target": ["multi-user.target"]},
        },
        "cases": list(q.CASES),
        "bindings": {
            key: PREFIX + "-" + suffix
            for key, suffix in {
                "holder": "worker.service",
                "contender": "fast.service",
                "barrier": "barrier.service",
                "dependent": "dependent.service",
                "stubborn": "stubborn.service",
                "support_guard": "supportguard.service",
                "support_service": "barrier.service",
                "timer": "timer.timer",
            }.items()
        },
        "preservation": {
            "policy": artifact(INPUT / "policy.json", b"{}"),
            "before": artifact(INPUT / "before.json", b"{}"),
            "before_request": artifact(INPUT / "before-request.json", b"{}"),
            "before_receipt": artifact(INPUT / "before-receipt.json", b"{}"),
            "before_execution": artifact(INPUT / "before-execution.json", b"{}"),
            "after_path": str(SCRATCH / "external/r3-after.json"),
            "verified_champions": {"sha256-" + "2" * 64: "3" * 64},
        },
    }
    for module in (
        "strength_freshness_cpu_qualification.py",
        "strength_freshness_cpu_lifecycle.py",
        "strength_freshness_linux.py",
        "strength_freshness_units.py",
        "strength_freshness_guard.py",
        "qualify_cloud_gpu_window.py",
        "strength_freshness_cpu_driver.py",
        "strength_freshness_cpu_support_case.py",
        "strength_freshness_cpu_preservation.py",
        "strength_freshness_cpu_fixture.py",
        "strength_freshness_cpu_completion.py",
        "strength_freshness_cpu_capture_request.py",
        "strength_freshness_cpu_collect_records.py",
        "strength_freshness_cpu_collect_support.py",
        "strength_freshness_cpu_collect_identity.py",
        "strength_freshness_cpu_collect_facts.py",
        "strength_freshness_cpu_readonly.py",
        "strength_freshness_cpu_outer.py",
    ):
        p["source_pins"].append(
            artifact(Path(p["control_root"]) / "scripts" / module, b"source")
        )
    data = {p["preservation"][key]["path"]: b"{}" for key in q.PRESERVATION_INPUTS}
    for suffix, role, kind in (
        ("dispatcher", "dispatcher", "service"),
        ("observer", "observer", "service"),
        ("publisher", "publisher", "service"),
        ("cleanup", "cleanup", "service"),
        ("watchdog", "watchdog", "timer"),
        ("worker", "workload", "service"),
        ("fast", "workload", "service"),
        ("timer", "workload", "timer"),
        ("barrier", "workload", "service"),
        ("dependent", "workload", "service"),
        ("stubborn", "workload", "service"),
        ("supportguard", "workload", "service"),
    ):
        name = PREFIX + "-" + suffix + "." + kind
        installed = "/etc/systemd/system/" + name
        mode = {
            "worker": "lock_holder",
            "fast": "lock_contender",
            "dependent": "fast_receipt",
            "stubborn": "term_tree",
            "supportguard": "support_transaction",
        }.get(suffix, "sleep")
        unit = {
            "role": role,
            "mode": mode if role == "workload" and kind == "service" else None,
            "payload": {
                "seconds": 2,
                "output_dir": str(SCRATCH / "payloads" / name),
                "lock_path": str(SCRATCH / "qualification.lock")
                if mode.startswith("lock_") or mode == "support_transaction"
                else None,
            }
            if role == "workload" and kind == "service"
            else None,
            "installed_path": installed,
            "boot_links": {
                "/etc/systemd/system/multi-user.target.wants/" + name: installed
            },
        }
        if role in {"dispatcher", "observer", "publisher"}:
            unit["boot_links"] = {}
        if mode == "support_transaction":
            unit["payload"]["scenario"] = artifact(INPUT / "scenario.json", b"{}")
        p["units"][name] = unit
        props = dict.fromkeys(set(q.linux.property_names(name)) - q.linux.VARIABLE, "")
        props.update(Id=name, FragmentPath=installed, LoadState="loaded")
        if kind == "service":
            props.update(
                User="root",
                Type="oneshot" if suffix in {"cleanup", "fast", "barrier"} else "exec",
                Restart="no",
                KillMode="control-group",
                SendSIGKILL="yes",
                TimeoutStopUSec="5s",
                Environment=q.environment_string(p),
            )
        unit["before"] = unit["after"] = {"properties": props, "environment_files": []}
    for name, unit in p["units"].items():
        role = unit["role"]
        header = (
            "[Unit]\nDescription=CPU fixture\nConditionPathExists="
            + str(SCRATCH / "attempt-armed")
            + "\n"
        )
        if name.endswith(".timer"):
            target = PREFIX + (
                "-cleanup.service" if role == "watchdog" else "-worker.service"
            )
            body = (
                (
                    "[Timer]\n"
                    + ("OnBootSec=490s" if role == "watchdog" else "OnActiveSec=1s")
                    + "\nAccuracySec=1s\nRandomizedDelaySec=0\nPersistent=false\nUnit="
                )
                + target
                + "\n"
            )
        else:
            entry = (
                "strength_freshness_cpu_support_case.py"
                if unit["mode"] == "support_transaction"
                else "strength_freshness_cpu_lifecycle.py"
                if role == "workload"
                else "strength_freshness_cpu_qualification.py"
            )
            argv = (
                p["python"]["path"]
                + " -s "
                + p["control_root"]
                + "/scripts/"
                + entry
                + " --authorization "
                + str(INPUT / (name + ".authorization.json"))
            )
            body = (
                "[Service]\nType="
                + unit["before"]["properties"]["Type"]
                + "\nUser=root\nGroup=root\nWorkingDirectory="
                + str(SCRATCH)
                + "\nExecStart="
                + argv
                + "\nRestart=no\nTimeoutStartSec=5s\nTimeoutStopSec=5s\nKillMode=control-group\nSendSIGKILL=yes\n"
                "PrivateDevices=yes\nDevicePolicy=closed\nNoNewPrivileges=yes\nProtectSystem=strict\nProtectHome=read-only\n"
                "Nice=19\nCPUQuota=100%\nMemoryMax=134217728\nTasksMax=8\nEnvironment="
                + q.environment_string(p)
                + "\nReadWritePaths="
                + str(SCRATCH)
                + (
                    " /etc/systemd/system"
                    if role in {"dispatcher", "publisher", "cleanup"}
                    or unit["mode"] == "support_transaction"
                    else ""
                )
                + "\n"
            )
            if unit["before"]["properties"]["Type"] != "oneshot":
                body += "RuntimeMaxSec=30s\n"
        raw = (
            header
            + body
            + ("[Install]\nWantedBy=multi-user.target\n" if unit["boot_links"] else "")
        ).encode()
        pin = artifact(INPUT / name, raw)
        for side in ("before", "after"):
            unit[side]["unit"] = pin
        p["files"][unit["installed_path"]] = [
            {"source": pin, "mode": 0o644, "uid": 0, "gid": 0}
        ]
        data[unit["installed_path"]] = raw
        data[pin["path"]] = raw
    return p, data


class Facts:
    def __init__(self):
        self.events = []

    def record(self, event, data):
        self.events.append((event, copy.deepcopy(data)))
        return {
            "path": str(len(self.events)),
            "sha256": q.sha(q.encode(data)),
            "bytes": len(q.encode(data)),
        }


class Backend:
    def __init__(self, data):
        self.time, self.boot, self.data = 100.0, BOOT, data
        self.actions, self.commands, self.links = [], [], {}
        self.processes = {}
        self.response = ""
        self.metadata = {"uid": 0, "gid": 0, "mode": 0o644}
        self.overrun = False

    def now(self):
        return self.time

    def boot_id(self):
        return self.boot

    def wall_ns(self):
        return 10**18 + int((self.time - 100) * 1e9)

    def read(self, path, maximum, deadline):
        assert self.time < deadline
        if str(path) not in self.data:
            raise FileNotFoundError(path)
        return self.data[str(path)]

    def file_metadata(self, path, deadline):
        return self.metadata

    def command(self, argv, deadline):
        self.commands.append(argv)
        if self.overrun:
            self.time = deadline + 1
        return self.response

    def action(self, verb, name, deadline):
        self.actions.append((verb, name))
        if self.overrun:
            self.time = deadline + 1

    def boot_links(self, name, deadline):
        return self.links

    def exists(self, path):
        return str(path) in self.data

    def members(self, path, deadline):
        return tuple(
            q.linux.core.Process(v["pid"], v["start_ticks"])
            for v in self.processes.values()
        )

    def process(self, pid, deadline):
        return self.processes.get(pid)

    def process_origin(self, pid, deadline):
        return {"exe": "/usr/bin/python3", "cwd": str(SCRATCH)}

    def sleep(self, seconds):
        self.time += seconds

    def atomic(self, path, data, *, overwrite=False, mode=0o600):
        self.data[str(path)] = data


@pytest.fixture
def fixture():
    value, data = make_plan()
    raw = q.encode(value)
    plan = q.Plan.parse(raw, q.sha(raw))
    from scripts.strength_freshness_cpu_lifecycle import Anchor

    anchor = Anchor("case-one", NONCE, plan.checksum, BOOT, 100.0, 10**18)
    backend, facts = Backend(data), Facts()
    io = q.ClosedIO(plan, cast(q.Budget, anchor), backend, facts, purpose="dispatcher")
    return SimpleNamespace(
        value=value,
        plan=plan,
        backend=backend,
        io=io,
        facts=facts,
        anchor=anchor,
        worker=PREFIX + "-worker.service",
    )


def test_original_plan_and_definition_are_closed(fixture):
    for name, unit in fixture.value["units"].items():
        q.validate_unit_text(
            fixture.plan, name, fixture.backend.data[unit["installed_path"]]
        )
    assert not any(fixture.plan.value["claims"].values())
    changed = fixture.plan.value
    changed["limits"]["work"] = 999
    assert fixture.plan.value["limits"]["work"] == 390


@pytest.mark.parametrize(
    "field,value",
    [
        ("limits", {**q.LIMITS, "work": 391}),
        ("claims", {**q.CLAIMS, "execution_qualified": True}),
        ("scratch_root", "/prod/run"),
        ("cases", list(q.CASES[:-1])),
        ("protocol_sha256", "0" * 64),
        ("nonce", "../evil"),
    ],
)
def test_bad_plan_refuses(field, value):
    p, _ = make_plan()
    p[field] = value
    with pytest.raises(q.Refusal):
        q.Plan.parse(q.encode(p), q.sha(q.encode(p)))


def test_duplicate_json_key_and_byte_drift_refuse(fixture):
    raw = fixture.plan.data
    with pytest.raises(q.Refusal, match="byte-pin"):
        q.Plan.parse(raw + b" ", fixture.plan.checksum)
    raw = raw.replace(
        b'"schema_version": 1', b'"schema_version": 1, "schema_version": 1'
    )
    with pytest.raises(q.Refusal, match="duplicate-json"):
        q.Plan.parse(raw, q.sha(raw))


@pytest.mark.parametrize(
    "replacement",
    [
        "ExecStart=/bin/sh -c evil",
        "ExecStart=/usr/bin/systemctl stop prod.service",
        "ExecStartPre=/usr/bin/true",
        "BindPaths=/prod/run",
        "DeviceAllow=/dev/nvidia0",
        "RootDirectory=/prod",
        "PermissionsStartOnly=true",
    ],
)
def test_pinned_unit_cannot_escape_systemd(fixture, replacement):
    raw = fixture.backend.data[fixture.value["units"][fixture.worker]["installed_path"]]
    if replacement.startswith("ExecStart="):
        lines = raw.decode().splitlines()
        lines = [replacement if x.startswith("ExecStart=") else x for x in lines]
        raw = ("\n".join(lines) + "\n").encode()
    else:
        raw = raw.replace(b"[Service]\n", b"[Service]\n" + replacement.encode() + b"\n")
    with pytest.raises(q.Refusal):
        q.validate_unit_text(fixture.plan, fixture.worker, raw)


@pytest.mark.parametrize(
    "argv",
    [
        ["sh", "-c", "true"],
        ["nvidia-smi"],
        ["systemctl", "stop", "prod.service"],
        ["systemctl", "stop", PREFIX + "-*"],
        ["systemctl", "set-default", "rescue.target"],
        ["systemctl", "isolate", "rescue.target"],
    ],
)
def test_arbitrary_commands_never_reach_backend(fixture, argv):
    with pytest.raises(q.Refusal, match="command-not-allowed"):
        fixture.io.command(argv, 120)
    assert fixture.backend.commands == []


def test_no_work_before_independent_cleanup_ack(fixture):
    with pytest.raises(q.Refusal, match="cleanup-not-acknowledged"):
        fixture.io.action("start", fixture.worker, 120)
    assert fixture.backend.actions == []


@pytest.mark.parametrize(
    "role", ["dispatcher", "observer", "publisher", "cleanup", "watchdog"]
)
@pytest.mark.parametrize("verb", ["start", "stop", "enable", "disable"])
def test_workload_cannot_touch_control_plane(fixture, role, verb):
    name = next(n for n, u in fixture.value["units"].items() if u["role"] == role)
    with pytest.raises(q.Refusal, match="protected-observer"):
        fixture.io.action(verb, name, 120)
    assert fixture.backend.actions == []


def test_cutoff_and_boot_change_refuse_before_action(fixture):
    fixture.backend.time = 490
    with pytest.raises(q.Refusal, match="original-boot-deadline"):
        fixture.io.action("stop", fixture.worker, 700)
    fixture.backend.time = 100
    fixture.backend.boot = "other"
    with pytest.raises(q.Refusal, match="original-boot-deadline"):
        fixture.io.action("stop", fixture.worker, 120)
    assert fixture.backend.actions == []


def test_unknown_file_mode_and_alias_refuse_before_stop(fixture):
    fixture.backend.metadata = {"uid": 1000, "gid": 0, "mode": 0o644}
    with pytest.raises(q.Refusal, match="unknown-file-variant"):
        fixture.io.action("stop", fixture.worker, 120)
    assert fixture.backend.actions == []
    assert any(name == "owned-file.raw" for name, _ in fixture.facts.events)


def test_foreign_boot_edge_is_not_silently_removed(fixture):
    fixture.backend.links = {
        "/etc/systemd/system/multi-user.target.wants/foreign.service": "/prod/unit"
    }
    with pytest.raises(q.Refusal, match="foreign-boot-edge"):
        fixture.io.action("stop", fixture.worker, 120)
    assert fixture.backend.actions == []


def test_raw_overrun_is_saved_before_refusal(fixture):
    fixture.backend.response = "multi-user.target\n"
    fixture.backend.overrun = True
    with pytest.raises(q.Refusal, match="command-output-bound"):
        fixture.io.command(["systemctl", "get-default"], 110)
    assert fixture.facts.events[-1] == ("command.raw", "multi-user.target\n")


def test_process_reuse_cannot_be_certified(fixture):
    group = "/system.slice/" + fixture.worker
    fixture.backend.processes[1] = {
        "pid": 1,
        "start_ticks": 10,
        "cgroup": "0::" + group,
        "ppid": 0,
    }
    fixture.io.members(group, 120)
    fixture.backend.processes[1]["start_ticks"] = 11
    with pytest.raises(q.Refusal, match="reuse"):
        fixture.io.process(1, 120)
    assert fixture.facts.events[-1][0] == "process.raw"


def test_unregistered_process_and_cgroup_are_not_read(fixture):
    with pytest.raises(q.Refusal, match="not-observed"):
        fixture.io.process(987654, 120)
    with pytest.raises(q.Refusal, match="cgroup-scope"):
        fixture.io.members("/system.slice/production.service", 120)


def test_candidate_never_manufactures_missing_cases(fixture):
    result = q.Driver(q.CaseRunner(fixture.io)).candidate()
    assert result["status"] == "incomplete" and result["missing_cases"] == sorted(
        q.CASES
    )
    assert result["execution_qualified"] is False


def test_finalizer_and_cleanup_have_distinct_authority(fixture):
    io = q.ClosedIO(
        fixture.plan, fixture.anchor, fixture.backend, fixture.facts, purpose="cleanup"
    )
    with pytest.raises(q.Refusal, match="purpose-verb"):
        io.action("start", fixture.worker, 120)
    io = q.ClosedIO(
        fixture.plan,
        fixture.anchor,
        fixture.backend,
        fixture.facts,
        purpose="finalizer",
    )
    with pytest.raises(q.Refusal, match="finalizer-not-admitted"):
        io.action("stop", fixture.worker, 120)


def test_cli_validation_cannot_be_execution_qualification(tmp_path, fixture):
    path = tmp_path / "plan.json"
    path.write_bytes(fixture.plan.data)
    result = q.validate_file(path, fixture.plan.checksum)
    assert result["status"] == "plan-validated-no-target-execution"
    assert result["execution_qualified"] is False


def test_runtime_inputs_cannot_be_removed_or_point_to_another_release():
    p, _ = make_plan()
    p["runtime_inputs"]["source_commit"] = "0" * 40
    with pytest.raises(q.Refusal, match="immutable-r4-source"):
        q.Plan.parse(q.encode(p), q.sha(q.encode(p)))
    p, _ = make_plan()
    p["runtime_inputs"]["native_binary"]["path"] = "/unqualified/native.so"
    with pytest.raises(q.Refusal, match="native-locations"):
        q.Plan.parse(q.encode(p), q.sha(q.encode(p)))


def test_control_sources_and_runtime_artifacts_have_distinct_size_limits():
    p, _ = make_plan()
    p["source_pins"][0]["bytes"] = 2**20 + 1
    with pytest.raises(q.Refusal, match="control-source-size"):
        q.Plan.parse(q.encode(p), q.sha(q.encode(p)))
    p, _ = make_plan()
    p["runtime_inputs"]["native_binary"]["bytes"] = 2**20 + 1
    q.Plan.parse(q.encode(p), q.sha(q.encode(p)))


@pytest.mark.parametrize(
    "phase,start,duration,allowed",
    [
        ("observer", 100, "570s", True),
        ("observer", 101, "570s", False),
        ("publisher", 100, "590s", True),
        ("publisher", 110, "590s", False),
        ("cleanup", 491, "120s", True),
        ("cleanup", 516, "120s", False),
    ],
)
def test_independent_unit_caps_do_not_renew_absolute_deadlines(
    fixture, phase, start, duration, allowed
):
    unit = q.linux.core.Unit(
        "dummy", "a" * 64, "/system.slice/dummy", entered_monotonic=start
    )
    props = {
        "Type": "oneshot" if phase == "cleanup" else "exec",
        "RuntimeMaxUSec": duration,
        "TimeoutStopUSec": "5s",
    }
    if allowed:
        q.validate_actor_bounds(phase, unit, props, duration, fixture.anchor)
    else:
        with pytest.raises(q.Refusal, match="original-lifetime"):
            q.validate_actor_bounds(phase, unit, props, duration, fixture.anchor)


def test_exact_alias_frees_only_one_unit_for_dispatcher(fixture):
    p = fixture.plan.value
    assert len(p["units"]) == 12
    assert len({v for v in p["bindings"].values()}) == 7
    assert p["bindings"]["barrier"] == p["bindings"]["support_service"]
    shared = p["units"][p["bindings"]["barrier"]]
    assert shared["role"] == "workload" and shared["mode"] == "sleep"
    assert all(
        shared[side]["properties"]["Type"] == "oneshot" for side in ("before", "after")
    )


@pytest.mark.parametrize(
    "key", ["holder", "contender", "dependent", "stubborn", "support_guard", "timer"]
)
def test_no_other_binding_alias_is_admitted(key):
    p, _ = make_plan()
    p["bindings"][key] = p["bindings"]["barrier"]
    with pytest.raises(q.Refusal, match="exact-barrier-support-alias"):
        q.Plan.parse(q.encode(p), q.sha(q.encode(p)))


@pytest.mark.parametrize("side", ["before", "after"])
def test_shared_specimen_cannot_switch_to_exec(side):
    p, _ = make_plan()
    p["units"][p["bindings"]["barrier"]][side]["properties"]["Type"] = "exec"
    with pytest.raises(q.Refusal, match="shared-barrier-fixed-oneshot"):
        q.Plan.parse(q.encode(p), q.sha(q.encode(p)))


@pytest.mark.parametrize("role", ["dispatcher", "observer", "publisher"])
def test_all_three_control_definitions_are_inert(role):
    p, _ = make_plan()
    name = next(n for n, v in p["units"].items() if v["role"] == role)
    p["units"][name]["boot_links"] = {
        "/etc/systemd/system/multi-user.target.wants/" + name: p["units"][name][
            "installed_path"
        ]
    }
    with pytest.raises(q.Refusal, match="inert-control"):
        q.Plan.parse(q.encode(p), q.sha(q.encode(p)))


@pytest.mark.parametrize(
    "start,runtime,stop,passed",
    [
        (100, "390s", "15s", True),
        (105, "385s", "15s", True),
        (105, "390s", "10s", False),
        (100, "390s", "16s", False),
        (99, "390s", "15s", False),
    ],
)
def test_dispatcher_actual_start_charges_original_work_and_terminal_bounds(
    fixture, start, runtime, stop, passed
):
    unit = q.linux.core.Unit(
        "dummy", "a" * 64, "/system.slice/dummy", entered_monotonic=start
    )
    props = {"Type": "exec", "RuntimeMaxUSec": runtime, "TimeoutStopUSec": stop}
    if passed:
        q.validate_actor_bounds("dispatcher", unit, props, "5s", fixture.anchor)
    else:
        with pytest.raises(q.Refusal):
            q.validate_actor_bounds("dispatcher", unit, props, "5s", fixture.anchor)


def test_observer_has_neither_unit_nor_file_mutation_capability(fixture):
    io = q.ClosedIO(
        fixture.plan, fixture.anchor, fixture.backend, fixture.facts, purpose="observer"
    )
    for verb in ("start", "stop", "enable", "disable"):
        with pytest.raises(q.Refusal, match="read-only-capability"):
            io.action(verb, fixture.worker, 120)
    with pytest.raises(q.Refusal, match="read-only-capability"):
        io.atomic(SCRATCH / "synthetic-support-state/state.json", b"{}")
    with pytest.raises(q.Refusal, match="case-workload-capability"):
        q.CaseRunner(io)
    assert not fixture.backend.actions


def test_cleanup_can_only_stop_dispatcher_not_start_or_disable_it(fixture):
    io = q.ClosedIO(
        fixture.plan, fixture.anchor, fixture.backend, fixture.facts, purpose="cleanup"
    )
    name = next(
        n for n, u in fixture.plan.value["units"].items() if u["role"] == "dispatcher"
    )
    io.action("stop", name, 120)
    for verb in ("start", "enable", "disable"):
        with pytest.raises(q.Refusal, match="cleanup-dispatcher-stop-only"):
            io.action(verb, name, 120)
    assert fixture.backend.actions == [("stop", name)]


@pytest.mark.parametrize("fault", [None, "shared-as-runtime", "extra-support"])
def test_shared_support_scenario_keeps_runtime_roles_disjoint(fixture, fault):
    b = fixture.plan.value["bindings"]
    scenario = {
        "guard_unit": b["support_guard"],
        "synthetic_plan": {
            "units": {
                key: {"name": name}
                for key, name in {
                    "guard": b["support_guard"],
                    "r3": b["holder"],
                    "r4": b["contender"],
                    "probe": b["dependent"],
                }.items()
            },
            "support_transition": {
                side: {b["barrier"]: {}} for side in ("before", "after")
            },
        },
    }
    if fault == "shared-as-runtime":
        scenario["synthetic_plan"]["units"]["r3"]["name"] = b["barrier"]
    elif fault == "extra-support":
        scenario["synthetic_plan"]["support_transition"]["after"][b["holder"]] = {}
    if fault is None:
        q.validate_shared_support_bindings(fixture.plan, scenario)
    else:
        with pytest.raises(q.Refusal, match="shared-support"):
            q.validate_shared_support_bindings(fixture.plan, scenario)


@pytest.mark.parametrize("key", ["before_request", "before_receipt"])
def test_plan_precommits_required_before_provenance(key):
    value, _ = make_plan()
    initial = q.Plan.parse(q.encode(value), q.sha(q.encode(value)))
    value["preservation"][key]["sha256"] = "e" * 64
    changed = q.Plan.parse(q.encode(value), q.sha(q.encode(value)))
    assert initial.checksum != changed.checksum
    del value["preservation"][key]
    with pytest.raises(q.Refusal, match="preservation-inputs"):
        q.Plan.parse(q.encode(value), q.sha(q.encode(value)))


@pytest.mark.parametrize("key", q.PRESERVATION_INPUTS)
@pytest.mark.parametrize(
    "fault", ["outside", "root", "traversal", "alias", "large", "empty", "bool"]
)
def test_preservation_input_scope_and_bounds(key, fault):
    value, _ = make_plan()
    pin = value["preservation"][key]
    if fault == "outside":
        pin["path"] = str(INPUT) + "-other/input.json"
    elif fault == "root":
        pin["path"] = str(INPUT)
    elif fault == "traversal":
        pin["path"] = str(INPUT / ".." / "input.json")
    elif fault == "alias":
        other = next(k for k in q.PRESERVATION_INPUTS if k != key)
        pin["path"] = value["preservation"][other]["path"]
    elif fault == "large":
        pin["bytes"] = 2**20 + 1
    elif fault == "empty":
        pin["bytes"] = 0
    else:
        pin["bytes"] = True
    with pytest.raises(q.Refusal):
        q.Plan.parse(q.encode(value), q.sha(q.encode(value)))


def test_before_provenance_addendum_is_required():
    value, _ = make_plan()
    value["addenda_sha256"].remove(
        "8d3ad081564569564ca8eefbb79f3565c87f55ec8759bdefa793aff7e874143c"
    )
    with pytest.raises(q.Refusal):
        q.Plan.parse(q.encode(value), q.sha(q.encode(value)))


def preservation_pin_reader(fixture):
    seen = []

    def pin(value, deadline):
        raw = fixture.io.read(Path(value["path"]), 2**20, deadline)
        q.require(
            len(raw) == value["bytes"] and q.sha(raw) == value["sha256"],
            "test-pin-mismatch",
        )
        seen.append(value["path"])

    return SimpleNamespace(pin=pin), seen


def test_before_provenance_uses_existing_bounded_input_read_path(fixture):
    reader, seen = preservation_pin_reader(fixture)
    q.verify_preservation_inputs(
        fixture.plan, cast(q.linux.LinuxIO, reader), fixture.facts, 120
    )
    assert seen == [
        fixture.value["preservation"][key]["path"] for key in q.PRESERVATION_INPUTS
    ]
    assert fixture.facts.events[-1] == (
        "preservation-input-pins",
        {key: fixture.value["preservation"][key] for key in q.PRESERVATION_INPUTS},
    )
    assert not fixture.backend.actions and not fixture.backend.commands
    with pytest.raises(q.Refusal, match="read-path-scope"):
        fixture.io.read(Path(str(INPUT) + "-other/before-receipt.json"), 2**20, 120)


@pytest.mark.parametrize("key", ["before_request", "before_receipt"])
def test_source_admission_refuses_changed_prework_artifact_before_capability(
    fixture, key
):
    reader, seen = preservation_pin_reader(fixture)
    fixture.backend.data[fixture.value["preservation"][key]["path"]] = b"changed"
    with pytest.raises(q.Refusal, match="test-pin-mismatch"):
        q.verify_sources(
            fixture.plan, cast(q.linux.LinuxIO, reader), fixture.facts, 120
        )
    assert fixture.value["preservation"][key]["path"] not in seen
    assert not any(name == "source-origins.raw" for name, _ in fixture.facts.events)
    assert not fixture.backend.actions and not fixture.backend.commands


def test_preservation_input_admission_cannot_extend_original_deadline(fixture):
    reader, seen = preservation_pin_reader(fixture)
    fixture.backend.time = fixture.anchor.deadline("work")
    with pytest.raises(q.Refusal):
        q.verify_preservation_inputs(
            fixture.plan,
            cast(q.linux.LinuxIO, reader),
            fixture.facts,
            fixture.backend.time + 10,
        )
    assert seen == [] and not fixture.backend.actions


@pytest.mark.parametrize("key", ["before_request", "before_receipt"])
def test_prework_pin_hash_shape_is_not_just_a_path(key):
    value, _ = make_plan()
    value["preservation"][key]["sha256"] = "not-a-content-hash"
    with pytest.raises(q.Refusal, match="artifact-sha"):
        q.Plan.parse(q.encode(value), q.sha(q.encode(value)))


def test_preservation_schema_never_silently_accepts_extra_provenance():
    value, _ = make_plan()
    value["preservation"]["uncommitted_witness"] = artifact(INPUT / "extra.json", b"{}")
    with pytest.raises(q.Refusal, match="preservation-inputs"):
        q.Plan.parse(q.encode(value), q.sha(q.encode(value)))


def test_dummy_jobs_use_exact_plain_table_through_closed_dispatcher(fixture):
    fixture.backend.response = "7 unrelated-backup.service start running\n"
    host = q.DummyReadHost(fixture.io)
    assert host._jobs(120) == {
        7: {
            "job": 7,
            "unit": "unrelated-backup.service",
            "type": "start",
            "state": "running",
        }
    }
    assert fixture.backend.commands == [list(q.linux.JOBS_COMMAND)]
    fixture.backend.commands.clear()
    for argv in (
        ["systemctl", "list-jobs", "--all", "--no-pager", "--output=json"],
        [*q.linux.JOBS_COMMAND, "foreign.service"],
        [*q.linux.JOBS_COMMAND, "--after"],
    ):
        with pytest.raises(q.Refusal, match="command-not-allowed"):
            fixture.io.command(argv, 120)
    assert fixture.backend.commands == []


def test_new_plan_requires_precommitted_before_execution_and_exact_name():
    value, _ = make_plan()
    del value["preservation"]["before_execution"]
    with pytest.raises(q.Refusal, match="preservation"):
        q.Plan.parse(q.encode(value), q.sha(q.encode(value)))
    value, _ = make_plan()
    value["preservation"]["before_execution"]["path"] = str(INPUT / "other-proof.json")
    with pytest.raises(q.Refusal, match="before-execution"):
        q.Plan.parse(q.encode(value), q.sha(q.encode(value)))


def test_old_addenda_cannot_silently_inherit_completion_gate():
    value, _ = make_plan()
    value["addenda_sha256"].remove(q.completion.ADDENDUM_SHA256)
    with pytest.raises(q.Refusal):
        q.Plan.parse(q.encode(value), q.sha(q.encode(value)))


@pytest.mark.parametrize("shadow", [False, True])
def test_actual_driver_origin_is_joined_before_source_byte_admission(
    monkeypatch, shadow
):
    from scripts import strength_freshness_cpu_driver as driver
    from scripts import strength_freshness_cpu_capture_request as requests
    from scripts import strength_freshness_cpu_lifecycle as lifecycle
    from scripts import qualify_cloud_gpu_window as system_host

    modules = (
        q,
        driver,
        q.linux,
        q.support,
        q.linux.core,
        lifecycle,
        system_host,
        q.completion,
        requests,
        requests.records,
        requests.supports,
        requests.supports.identity_module,
        requests.supports.facts,
        requests.supports.identity_module.readonly,
        requests.preservation,
    )
    files = {str(Path(cast(str, module.__file__)).resolve()) for module in modules}
    source_pins = [artifact(path, Path(path).read_bytes()) for path in sorted(files)]
    value, _ = make_plan()
    value.update(
        control_root=str(Path(cast(str, q.__file__)).resolve().parent.parent),
        source_pins=source_pins,
    )
    driver_path = str(Path(cast(str, driver.__file__)).resolve())
    if shadow:
        monkeypatch.setattr(
            driver, "__file__", "/shadow/strength_freshness_cpu_driver.py"
        )
    seen = []

    class EndOfOriginCheck(Exception):
        pass

    def pin(item, deadline):
        assert deadline == 120
        if item == value["python"]:
            raise EndOfOriginCheck
        seen.append(item["path"])

    facts = SimpleNamespace(record=lambda event, data: None)
    with pytest.raises(
        q.Refusal if shadow else EndOfOriginCheck,
        match="executing-module-unpinned" if shadow else None,
    ):
        q.verify_sources(
            cast(q.Plan, SimpleNamespace(value=value)),
            cast(q.linux.LinuxIO, SimpleNamespace(pin=pin)),
            cast(q.RawFacts, facts),
            120,
        )
    if shadow:
        assert not (set(seen) & files)
    else:
        assert driver_path in seen and set(files) <= set(seen)
