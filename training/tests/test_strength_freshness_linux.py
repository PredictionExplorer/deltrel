"""CPU-only Linux adapter contracts; no systemctl, NVML or model execution."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import runpy
import sqlite3
import subprocess
import sys
import time

import pytest

from deltreltrain import strength_freshness_guard as g
from scripts import strength_freshness_linux as a


class FakeIO(a.LinuxIO):
    def __init__(self):
        self.time = 100.0
        self.boot = "boot-a"
        self.calls = []
        self.files = {}
        self.raw = {}
        self.processes = {}
        self.groups = {}
        self.jobs = []
        self.nvml = ""
        self.child_result = {
            "phase": "r3",
            "plan_sha256": "a" * 64,
            "evidence_sha256": "b" * 64,
        }

    def now(self):
        return self.time

    def boot_id(self):
        return self.boot

    def sleep(self, seconds):
        self.time += seconds

    def process(self, pid, deadline=None):
        if deadline is not None:
            assert self.time < deadline
        return copy.deepcopy(self.processes.get(pid))

    def members(self, path, deadline):
        return self.groups.get(path, ())

    def exists(self, path):
        return str(path) in self.files or super().exists(path)

    def boot_links(self, name, deadline):
        if self.raw[name]["UnitFileState"] == "enabled":
            return {
                "/etc/systemd/system/multi-user.target.wants/"
                + name: "/etc/systemd/system/" + name
            }
        if self.raw[name]["UnitFileState"] == "enabled-runtime":
            return {
                "/run/systemd/system/multi-user.target.wants/"
                + name: "/etc/systemd/system/" + name
            }
        return {}

    def file_metadata(self, path, deadline):
        if str(path) in self.files:
            return {"uid": os.geteuid(), "gid": os.getegid(), "mode": 0o600}
        return super().file_metadata(path, deadline)

    def process_origin(self, pid, deadline):
        assert self.time < deadline
        value = self.processes[pid]
        return {key: value[key] for key in ("exe", "cwd")}

    def read(self, path, maximum, deadline):
        assert self.time < deadline
        if str(path) in self.files:
            data = self.files[str(path)]
            assert len(data) <= maximum
            return data
        return super().read(path, maximum, deadline)

    def atomic(self, path, data, *, overwrite=False, mode=0o600):
        if str(path).startswith("/etc/"):
            self.files[str(path)] = data
        else:
            super().atomic(path, data, overwrite=overwrite, mode=mode)

    def command(self, argv, deadline):
        assert self.time < deadline
        self.calls.append((list(argv), deadline))
        if argv == ["systemctl", "get-default"]:
            return "multi-user.target\n"
        if argv[:2] == ["systemctl", "list-jobs"]:
            return json.dumps(self.jobs)
        if argv[:2] == ["systemctl", "show"]:
            return "\n".join(f"{k}={v}" for k, v in self.raw[argv[2]].items())
        if argv[0] == "nvidia-smi":
            if "--query-compute-apps=gpu_uuid,pid" in argv:
                return self.nvml
            return "\n".join(
                f"{i}, GPU-{i}, 00:{i:02x}, fake, fake-driver, 1000" for i in range(8)
            )
        if argv[:2] in (["systemctl", "enable"], ["systemctl", "disable"]):
            self.raw[argv[-1]]["UnitFileState"] = (
                "enabled" if argv[1] == "enable" else "disabled"
            )
            return ""
        if argv[:2] == ["systemctl", "stop"]:
            row = self.raw[argv[-1]]
            row.update(MainPID="0", ActiveState="inactive", SubState="dead", Job="0 /")
            self.groups[row["ControlGroup"]] = ()
            return ""
        if argv[:2] == ["systemctl", "start"]:
            return ""
        if argv == ["systemctl", "daemon-reload"]:
            return ""
        raise AssertionError(argv)

    def child(self, argv, environment, deadline):
        self.calls.append(("child", argv, environment, deadline))
        return copy.deepcopy(self.child_result)


@pytest.fixture
def host(tmp_path):
    h = a.LinuxHost.__new__(a.LinuxHost)
    h.io = FakeIO()
    h.execute = True
    h.lease_fd = 77
    h.manifest_path = tmp_path / "linux.json"
    h.manifest_sha256 = "a" * 64
    h.state = tmp_path / "state"
    g.Journal(h.state).create()
    h.root = tmp_path / "run"
    h.root.mkdir()
    h._authority_cache = h._champion_cache = None
    h._membership_changed = False
    h.last_raw = {}
    names = [
        "r3.service",
        "r4.service",
        "probe.service",
        "guard.service",
        "monitor.service",
        "report.service",
        "backup.service",
    ]
    units = {}
    for name in names:
        path = "/etc/systemd/system/" + name
        data = f"[Unit]\nDescription={name}\n".encode()
        raw = {k: "" for k in a.PROPERTIES}
        raw.update(
            Id=name,
            LoadState="loaded",
            FragmentPath=path,
            DropInPaths="",
            ExecStart="/bin/fake",
            ActiveState="inactive",
            SubState="dead",
            MainPID="0",
            ControlGroup="/system.slice/" + name,
            ExecMainStartTimestampMonotonic="0",
            Result="success",
            ExecMainStatus="0",
            ExecMainPID="0",
            NRestarts="0",
            Job="0 /",
            UnitFileState="disabled",
            KillMode="control-group",
            SendSIGKILL="yes",
            Restart="no",
            Type="simple",
            TimeoutStopUSec="10s",
            RuntimeMaxUSec="600s",
        )
        h.io.raw[name] = raw
        h.io.files[path] = data
        stage = {
            "properties": {k: raw[k] for k in set(a.PROPERTIES) - a.VARIABLE},
            "unit": {"path": path, "sha256": a.checksum(data), "bytes": len(data)},
            "environment_files": [],
        }
        units[name] = {
            "boot_links": {"/etc/systemd/system/multi-user.target.wants/" + name: path},
            "installed_path": path,
            "before": copy.deepcopy(stage),
            "after": copy.deepcopy(stage),
        }
    h.manifest = {
        "boot_topology": {
            "default_target": "multi-user.target",
            "target_paths": {"multi-user.target": ["multi-user.target"]},
        },
        "adapter_sha256": a.checksum(Path(a.__file__).read_bytes()),
        "probe_gpu_index": 0,
        "units": units,
        "control_root": str(tmp_path / "control"),
        "helper_python": {"path": "/fake/python"},
        "backup_worker_unit": "backup.service",
    }
    h.plan = {
        "probe_gpu_uuid": "GPU-0",
        "state_root": str(h.state),
        "run_root": str(h.root),
        "freshness_plan_sha256": "a" * 64,
        "units": {
            r: {
                "name": r + ".service",
                "initial_enabled": r == "r3",
                "committed_enabled": r == "r4",
            }
            for r in g.ROLES
        },
        "support_transition": {
            s: {
                name: {
                    "definition_sha256": units[name][s]["unit"]["sha256"],
                    "environment_sha256": g.digest({"environment": "", "files": []}),
                    "enabled": name != "backup.service",
                }
                for name in names[4:]
            }
            for s in ("before", "after")
        },
        "exclusion_path": str(tmp_path / "host.lock"),
        "attempt_id": "fake-once",
        "probe_output": str(tmp_path / "probe"),
    }
    return h


def activate(host, role, pid=501):
    name = role + ".service"
    p = g.Process(pid, 5000 + pid)
    host.io.raw[name].update(
        MainPID=str(pid),
        ExecMainPID=str(pid),
        InvocationID=f"{pid:032x}",
        ActiveState="active",
        SubState="running",
        ExecMainStartTimestampMonotonic="99000000",
    )
    host.io.processes[pid] = {
        "pid": pid,
        "start_ticks": p.start_ticks,
        "cgroup": "0::/system.slice/" + name,
    }
    host.io.groups["/system.slice/" + name] = (p,)
    return p


def observation(host):
    units = {r: host._unit(r + ".service", {}, host.io.now() + 5)[0] for r in g.ROLES}
    support = {
        name: {
            k: v
            for k, v in host._unit(name, {}, host.io.now() + 5)[1].items()
            if k != "stage"
        }
        for name in ("monitor.service", "report.service", "backup.service")
    }
    return g.Observation(
        g.Clock("boot-a", 100, 10**12),
        units,
        tuple(f"GPU-{i}" for i in range(8)),
        "f" * 64,
        (),
        g.Authority("r3", "a" * 64, "b" * 64),
        lease={"owner": {"pid": os.getpid()}, "nonce": "n"},
        support=support,
    )


def details(host, obs):
    return {
        "guard_plan_sha256": g.digest(host.plan),
        "expected_boot_id": obs.clock.boot_id,
        "expected_units": {r: asdict(u) for r, u in obs.units.items()},
        "expected_support": dict(obs.support),
        "nonce": "n",
    }


@pytest.mark.parametrize(
    "value",
    ["12 /org/freedesktop/systemd1/job/12", "12 /org/freedesktop/systemd1/job/13"],
)
def test_pending_jobs_are_joined_to_exact_unit_and_id(value):
    jobs = {12: {"job": 12, "unit": "r3.service", "type": "stop", "state": "running"}}
    if value.endswith("/13"):
        with pytest.raises(g.Refusal):
            a.parse_job(value, jobs, "r3.service")
    else:
        assert a.parse_job(value, jobs, "r3.service") == {
            "id": 12,
            "kind": "stop",
            "state": "running",
        }


def test_unit_joins_live_main_to_real_cgroup_membership(host):
    p = activate(host, "r3")
    unit, _ = host._unit("r3.service", {}, 105)
    assert unit.main == p and unit.members == (p,)
    host.io.groups[unit.cgroup] = ()
    with pytest.raises(a.ObservationChanged, match="cgroup-changed"):
        host._unit("r3.service", {}, 105)


@pytest.mark.parametrize("mutation", ["fragment", "properties", "pid-reuse"])
def test_unit_refuses_unpinned_or_raced_identity(host, mutation):
    p = activate(host, "r3")
    if mutation == "fragment":
        host.io.files["/etc/systemd/system/r3.service"] = b"changed"
    elif mutation == "properties":
        host.io.raw["r3.service"]["ExecStart"] = "/bin/other"
    else:
        host.io.processes[p.pid]["start_ticks"] += 1
    with pytest.raises((g.Refusal, a.ObservationChanged)):
        host._unit("r3.service", {}, 105)


def test_nvml_foreign_and_stale_owners_never_count_as_empty(host):
    p = activate(host, "r3")
    unit = host._unit("r3.service", {}, 105)[0]
    host.io.nvml = f"GPU-0, {p.pid}"
    uuids, _, owners = host._gpus({"r3": unit}, 105)
    assert len(uuids) == 8 and owners[0].process == p
    host.io.nvml = "GPU-0, 999"
    with pytest.raises(a.ObservationChanged, match="stale-owner"):
        host._gpus({"r3": unit}, 105)
    host.io.processes[999] = {"pid": 999, "start_ticks": 1, "cgroup": "0::/foreign"}
    with pytest.raises(g.Refusal, match="foreign-gpu-owner"):
        host._gpus({"r3": unit}, 105)


def test_pause_stops_support_and_r3_and_disables_boot_edges(host, monkeypatch):
    activate(host, "r3")
    obs = observation(host)
    monkeypatch.setattr(host, "observe", lambda _: obs)
    host.perform("pause-support-and-stop-r3", details(host, obs), 150)
    verbs = [x[0][:2] for x in host.io.calls if isinstance(x[0], list)]
    assert ["systemctl", "disable"] in verbs and ["systemctl", "stop"] in verbs
    assert host.io.raw["r3.service"]["MainPID"] == "0"
    assert all(
        host.io.raw[n]["UnitFileState"] == "disabled"
        for n in host.plan["support_transition"]["before"]
    )


@pytest.mark.parametrize("fault", ["unqualified", "boot", "unit", "nonce"])
def test_action_rechecks_authority_before_first_mutation(host, monkeypatch, fault):
    obs = observation(host)
    d = details(host, obs)
    if fault == "unqualified":
        host.execute = False
    elif fault == "boot":
        d["expected_boot_id"] = "old"
    elif fault == "unit":
        d["expected_units"]["r3"]["main"] = {"pid": 999, "start_ticks": 1}
    else:
        d["nonce"] = "wrong"
    monkeypatch.setattr(host, "observe", lambda _: obs)
    before = len(host.io.calls)
    with pytest.raises(g.Refusal):
        host.perform("pause-support-and-stop-r3", d, 150)
    assert len(host.io.calls) == before


def test_closed_worker_environment_cannot_inherit_import_or_gpu_overrides(
    host, monkeypatch
):
    monkeypatch.setenv("PYTHONSTARTUP", "/bad")
    monkeypatch.setenv("LD_PRELOAD", "/bad")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    host._worker("authority", 105)
    env = host.io.calls[-1][2]
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["OMP_NUM_THREADS"] == "1"
    assert "PYTHONSTARTUP" not in env and "LD_PRELOAD" not in env
    with pytest.raises(g.Refusal):
        host._worker("unregistered-shell", 105)


def test_timeout_kills_owned_cpu_process_group(tmp_path):
    io_ = a.LinuxIO()
    began = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        io_.child(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            {"PATH": os.environ["PATH"], "CUDA_VISIBLE_DEVICES": ""},
            io_.now() + 1.5,
        )
    assert time.monotonic() - began < 3


def test_child_exit_json_does_not_accept_nonzero_or_bad_output():
    io_ = a.LinuxIO()
    env = {"PATH": os.environ["PATH"], "CUDA_VISIBLE_DEVICES": ""}
    with pytest.raises(g.Refusal, match="worker-failed"):
        io_.child([sys.executable, "-c", "raise SystemExit(2)"], env, io_.now() + 3)
    with pytest.raises(json.JSONDecodeError):
        io_.child([sys.executable, "-c", 'print("bad")'], env, io_.now() + 3)


def test_qualification_contract_has_no_final_core_plan_digest_cycle():
    value = {
        "adapter_sha256": "a" * 64,
        "units": {"fixed": "bytes"},
        "qualification": {"sha256": "b" * 64},
        "guard_plan": {"sha256": "c" * 64},
        "core_plan_sha256": "d" * 64,
    }
    expected = a.qualification_contract(value)
    value["qualification"] = {"sha256": "e" * 64}
    value["guard_plan"] = {"sha256": "f" * 64}
    value["core_plan_sha256"] = "0" * 64
    assert a.qualification_contract(value) == expected
    value["units"] = {"changed": "bytes"}
    assert a.qualification_contract(value) != expected


def test_start_gate_blocks_unrequested_and_wrong_boot_runtime(host, monkeypatch):
    state = {
        "plan_sha256": g.digest(host.plan),
        "nonce": "n",
        "active_boot_id": "boot-a",
        "starts": {},
        "last_action": {"kind": "none", "deadline": 200},
        "deadline": 300,
    }
    g.Journal(host.state).save("state.json", state)
    monkeypatch.setattr(
        host,
        "_lease",
        lambda _: {
            "nonce": "n",
            "plan_sha256": g.digest(host.plan),
            "boot_id": "boot-a",
        },
    )
    with pytest.raises(g.Refusal, match="no-current-start-intent"):
        a.admit_start(host, "r4")
    state.update(
        starts={"r4": {"prior_invocation": ""}},
        last_action={"kind": "start-r4", "deadline": 200},
        active_boot_id="old",
    )
    g.Journal(host.state).save("state.json", state)
    with pytest.raises(g.Refusal, match="no-current-start-intent"):
        a.admit_start(host, "r4")


def test_finalizer_cannot_enter_generic_recovery(host):
    g.Journal(host.state).save("state.json", {"runtime_authority_retired": False})
    with pytest.raises(g.Refusal, match="not-sealed"):
        a.finalize_retirement(host)


def test_arena_allocation_and_actions_may_only_extend(tmp_path):
    root = tmp_path / "run"
    (root / "arena").mkdir(parents=True)
    row = {
        "ring": 10,
        "variant": "pie",
        "pair": 0,
        "candidate_player": 0,
        "opening_seed": 1,
        "opening_action": 0,
        "actions": [1, 2],
        "result": None,
    }
    old = {
        "run_id": "same",
        "arena_state": {
            "config": {"seed": 1},
            "game_states": [row],
            "games": [],
            "pairs": [],
        },
    }
    now = copy.deepcopy(old)
    now["arena_state"]["game_states"][0]["actions"].append(3)
    p = root / "arena/x.resume.json"
    p.write_bytes(a.encoded(now))
    io_ = a.LinuxIO()
    assert a.arena_prefix_preserved(
        {"arena/x.resume.json": old}, root, io_, io_.now() + 5
    )
    now["arena_state"]["game_states"][0]["actions"][0] = 7
    p.write_bytes(a.encoded(now))
    with pytest.raises(g.Refusal, match="action-prefix"):
        a.arena_prefix_preserved({"arena/x.resume.json": old}, root, io_, io_.now() + 5)


def test_backup_visibility_does_not_equal_clean_worker_exit(host):
    (host.state / "linux-backup-result.json").write_text("{}")
    p = activate(host, "backup")
    unit = host._unit("backup.service", {}, 105)[0]
    assert p and host._backup_result(105, unit) is None


def test_reentry_reads_actual_self_and_old_pid_birth(host, monkeypatch):
    pid = os.getpid()
    host.io.processes[pid] = {"pid": pid, "start_ticks": 10, "cgroup": "0::/guard"}
    host.io.processes[99] = {"pid": 99, "start_ticks": 21, "cgroup": "0::/other"}
    monkeypatch.setenv("INVOCATION_ID", "f" * 32)
    proof = host.guard_reentry_evidence(g.Process(99, 20), 105)
    assert proof.executing_process == g.Process(
        pid, 10
    ) and proof.previous_pid_process == g.Process(99, 21)


def test_atomic_new_never_clobbers_prior_receipt(tmp_path):
    io_ = a.LinuxIO()
    p = tmp_path / "receipt.json"
    io_.atomic(p, b"original")
    with pytest.raises(FileExistsError):
        io_.atomic(p, b"replacement")
    assert p.read_bytes() == b"original" and not list(tmp_path.glob("*.tmp-*"))


def full_manifest(host, tmp_path):
    fixtures = runpy.run_path(
        str(Path(__file__).with_name("test_strength_freshness_guard.py"))
    )
    root = tmp_path / "full"
    root.mkdir()
    plan = fixtures["make_plan"](root)
    for role in g.ROLES:
        spec = host.manifest["units"][role + ".service"]
        plan["units"][role]["definition_sha256"] = spec["before"]["unit"]["sha256"]
        if role in {"r3", "r4", "probe"}:
            for stage in ("before", "after"):
                spec[stage]["properties"]["ExecStartPre"] = (
                    "qualified-helper admit-start --role " + role
                )
        if role == "guard":
            for stage in ("before", "after"):
                spec[stage]["properties"]["Before"] = (
                    "r3.service r4.service probe.service"
                )
    for stage in ("before", "after"):
        plan["support_transition"][stage] = host.plan["support_transition"][stage]
    plan["support_transition"]["sha256"] = g.digest(
        {k: v for k, v in plan["support_transition"].items() if k != "sha256"}
    )
    python = root / "python"
    python.write_bytes(b"CPU fake interpreter fixture")
    pin = {
        "path": str(python),
        "sha256": a.checksum(python.read_bytes()),
        "bytes": python.stat().st_size,
    }
    m = {
        **host.manifest,
        "format": a.FORMAT,
        "schema_version": 1,
        "adapter_sha256": a.checksum(Path(a.__file__).read_bytes()),
        "guard_plan": {},
        "core_plan_sha256": "",
        "helper_python": pin,
        "implementation_pins": [
            pin,
            *[
                a._file_pin(p.resolve())
                for p in (
                    Path(a.__file__),
                    Path(g.__file__),
                    *(Path(a.__file__).with_name(n) for n in a.DIRECT_SCRIPTS),
                    *(Path(g.__file__).with_name(n) for n in a.DIRECT_RUNTIME),
                )
            ],
        ],
        "prestaged_inputs": [],
        "execution": {
            "host_exclusion_contract": "all-registered-gpu-starts-fenced-under-guardian-lease-v1",
            "finalizer": "sealed-retirement-only",
            "start_fences": {
                r: "qualified-helper admit-start --role " + r
                for r in ("r3", "r4", "probe")
            },
        },
    }
    script = root / "compile_worker.py"
    script.write_bytes(b"pinned fake compile worker")
    m["implementation_pins"].append(a._file_pin(script))
    m["auxiliary_policies"] = {
        role: {
            "python_argv0": str(python),
            "executable": str(python),
            "cwd": str(root),
            "compile_worker_script": str(script),
            "torch_key": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
            "compile_parent_roles": ["learner"],
        }
        for role in ("r3", "r4")
    }
    support = root / "support-source.json"
    support.write_bytes(a.encoded({"files": m["implementation_pins"]}))
    m["support_source_manifest"] = a._file_pin(support)
    plan["proof_closure_inputs"]["support_manifest"] = [
        m["support_source_manifest"]["sha256"]
    ]
    runtime_source = Path(plan["probe_runtime"]["source_manifest"]["path"])
    runtime_source.parent.mkdir(parents=True)
    runtime_source.write_bytes(b"manifest")
    plan["proof_closure_inputs"]["runtime_source_manifest"] = [
        plan["probe_runtime"]["source_manifest"]["sha256"]
    ]
    q = {
        "format": a.QUALIFICATION,
        "schema_version": 1,
        "status": "passed",
        "execution_qualified": True,
        "adapter_sha256": m["adapter_sha256"],
        "core_source_sha256": plan["guard_source_sha256"],
        "contract_sha256": a.qualification_contract(m),
        "checks": {k: True for k in a.CHECKS},
    }
    qualification = root / "qualification.json"
    qualification.write_bytes(a.encoded(q))
    m["qualification"] = {
        "path": str(qualification),
        "sha256": a.checksum(qualification.read_bytes()),
        "bytes": qualification.stat().st_size,
    }
    plan["adapter_qualification_sha256"] = m["qualification"]["sha256"]
    guard = root / "guard.json"
    guard.write_bytes(a.encoded(plan))
    m["guard_plan"] = {
        "path": str(guard),
        "sha256": a.checksum(guard.read_bytes()),
        "bytes": guard.stat().st_size,
    }
    m["core_plan_sha256"] = g.digest(plan)
    path = root / "linux.json"
    path.write_bytes(a.encoded(m))
    return path, m, plan


def test_final_pins_admit_cpu_fixture_without_any_host_commands(host, tmp_path):
    path, m, plan = full_manifest(host, tmp_path)
    loaded = a.LinuxHost(path, a.checksum(path.read_bytes()), execute=True, io_=host.io)
    assert loaded.plan == plan and loaded.manifest == m and host.io.calls == []


@pytest.mark.parametrize(
    "fault",
    [
        "missing-qualification",
        "adapter-source",
        "core-bytes",
        "restart-policy",
        "duration",
        "dropin",
        "missing-helper",
    ],
)
def test_unqualified_or_drifted_plan_never_reaches_mutations(host, tmp_path, fault):
    path, m, _ = full_manifest(host, tmp_path)
    if fault == "missing-qualification":
        m.pop("qualification")
    elif fault == "adapter-source":
        m["adapter_sha256"] = "0" * 64
    elif fault == "core-bytes":
        Path(m["guard_plan"]["path"]).write_text("{}")
    elif fault == "restart-policy":
        m["units"]["r4.service"]["after"]["properties"]["Restart"] = "always"
    elif fault == "duration":
        m["units"]["probe.service"]["after"]["properties"]["RuntimeMaxUSec"] = (
            "infinity"
        )
    elif fault == "missing-helper":
        m["implementation_pins"] = [
            p
            for p in m["implementation_pins"]
            if not p["path"].endswith("qualify_cloud_gpu_window.py")
        ]
    else:
        m["units"]["r3.service"]["after"]["properties"]["DropInPaths"] = (
            "/unreviewed.conf"
        )
    path.write_bytes(a.encoded(m))
    with pytest.raises(g.Refusal):
        a.LinuxHost(path, a.checksum(path.read_bytes()), execute=True, io_=host.io)
    assert host.io.calls == []


def test_guardian_same_numeric_pid_reuse_dispatches_typed_reentry(host, monkeypatch):
    pid = os.getpid()
    host.io.processes[pid] = {"pid": pid, "start_ticks": 20, "cgroup": "0::/guard"}
    monkeypatch.setenv("INVOCATION_ID", "f" * 32)
    g.Journal(host.state).save(
        "state.json",
        {
            "deadline_wall_ns": time.time_ns() + 120_000_000_000,
            "deadline": 220.0,
            "active_boot_id": "boot-a",
            "owners": {
                "guard": {
                    "main": {"pid": pid, "start_ticks": 10},
                    "invocation_id": "a" * 32,
                    "boot_id": "boot-a",
                }
            },
        },
    )
    calls = []

    class Controller:
        def __init__(self, *_):
            pass

        def admit_guard_reentry(self):
            calls.append("reentry")

        def tick(self):
            return {"finished": True}

    @contextmanager
    def hold():
        yield

    monkeypatch.setattr(host, "hold_lease", hold)
    monkeypatch.setattr(g, "Controller", Controller)
    assert a.guardian(host)["finished"] and calls == ["reentry"]


def test_worker_cannot_bypass_independent_guard_lease(host, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(host, "_lease", lambda _: None)
    g.Journal(host.state).save("state.json", {"nonce": "n"})
    with pytest.raises(g.Refusal, match="no-guardian-lease"):
        a.worker(host, "apply-r4", {"nonce": "n"})
    assert host.io.calls == []


def backup_fixture(host, tmp_path):
    host.manifest.update(
        proof_archive_root=str(tmp_path / "archive"), backup_root=str(tmp_path / "dr")
    )
    host.plan["run_identity"] = {"run_id": "run", "generation_family": "family"}
    root = tmp_path / "archive" / host.plan["attempt_id"]
    (root / "objects").mkdir(parents=True)
    data = b"real tiny proof bytes"
    sha = a.checksum(data)
    obj = root / "objects" / sha
    obj.write_bytes(data)
    obj.chmod(0o444)
    pin = {"path": str(obj), "sha256": sha, "bytes": len(data)}
    request = {
        "nonce": "n",
        "required_proof_sha256": [sha],
        "proof_closure_sha256": "c" * 64,
    }
    proof = {
        "nonce": "n",
        "core_plan_sha256": g.digest(host.plan),
        "manifest_sha256": host.manifest_sha256,
        "proof_closure_sha256": "c" * 64,
        "objects": [pin],
        "required_core_sha256": [sha],
    }
    (root / "manifest.json").write_bytes(a.encoded(proof))
    snapshot = tmp_path / "dr/snapshots/run/catalog.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(a.encoded({"run_id": "run", "generation_family": "family"}))
    marker = snapshot.with_name(snapshot.name + ".commit")
    marker.write_bytes(a.encoded({"sha256": a.checksum(snapshot.read_bytes())}))
    owner = {
        "format": a.FORMAT + "-backup-started",
        "schema_version": 1,
        "manifest_sha256": host.manifest_sha256,
        "core_plan_sha256": g.digest(host.plan),
        "nonce": "n",
        "unit_name": "backup.service",
        "main": {"pid": 501, "start_ticks": 5501},
        "entered_monotonic": 99.0,
        "boot_id": "boot-a",
        "invocation_id": "a" * 32,
        "request_sha256": a.checksum(a.encoded(request)),
    }
    result = {
        "manifest_sha256": host.manifest_sha256,
        "invocation_id": "a" * 32,
        "request_sha256": owner["request_sha256"],
        "metadata_pins": [
            a._file_pin(p) for p in (root / "manifest.json", snapshot, marker)
        ],
        "retained_objects": [pin],
        "core_result": {
            "verified_artifact_sha256": [sha],
            "catalog_sha256": a.checksum(snapshot.read_bytes()),
        },
    }
    for name, value in [
        ("linux-backup-request.json", request),
        ("linux-backup-started.json", owner),
        ("linux-backup-result.json", result),
    ]:
        (host.state / name).write_bytes(a.encoded(value))
    host.io.raw["backup.service"].update(
        ExecMainPID="501", ExecMainStartTimestampMonotonic="99000000"
    )
    unit = host._unit("backup.service", {}, 105)[0]
    return unit, result, obj


def test_backup_receipt_requires_clean_exit_bound_request_and_retained_bytes(
    host, tmp_path
):
    unit, result, _ = backup_fixture(host, tmp_path)
    assert host._backup_result(105, unit) == result["core_result"]


@pytest.mark.parametrize(
    "fault", ["old-invocation", "request", "missing-retained", "writable", "catalog"]
)
def test_false_or_unretained_backup_never_completes(host, tmp_path, fault):
    unit, result, obj = backup_fixture(host, tmp_path)
    if fault == "old-invocation":
        result["invocation_id"] = "b" * 32
    elif fault == "request":
        result["request_sha256"] = "f" * 64
    elif fault == "missing-retained":
        obj.unlink()
    elif fault == "writable":
        obj.chmod(0o644)
    else:
        result["core_result"]["catalog_sha256"] = "f" * 64
    (host.state / "linux-backup-result.json").write_bytes(a.encoded(result))
    with pytest.raises((g.Refusal, OSError)):
        host._backup_result(105, unit)


def test_probe_index_must_name_exact_registered_uuid(host):
    host.manifest["probe_gpu_index"] = 1
    with pytest.raises(g.Refusal, match="device-map"):
        host._gpus({}, 105)


@pytest.mark.parametrize("role", ["r3", "r4"])
def test_retired_owned_pending_telemetry_admits_only_its_verified_lineage(
    host, monkeypatch, role
):
    g.Journal(host.state).save(
        "state.json",
        {
            "plan_sha256": g.digest(host.plan),
            "runtime_authority_retired": True,
            "outcome": "owned-" + role + "-pending-telemetry",
        },
    )
    monkeypatch.setattr(
        host, "_authority", lambda _: g.Authority(role, "a" * 64, "b" * 64)
    )
    assert a.admit_start(host, role)["scope"] == "sealed-lineage"
    with pytest.raises(g.Refusal, match="retired-lineage"):
        a.admit_start(host, "r3" if role == "r4" else "r4")


def test_new_champion_uses_official_top_level_terminal_fields(
    host, tmp_path, monkeypatch
):
    from deltreltrain import champion_migration

    learned = "sha256-" + "1" * 64
    baseline = "sha256-" + "2" * 64
    proof = tmp_path / "baseline.json"
    proof.write_bytes(b"previously reviewed baseline proof")
    host.manifest["verified_champions"] = {baseline: a._file_pin(proof)}
    host.manifest["promotion_contract_identity"] = "sha256-" + "3" * 64
    host.plan.update(
        run_identity={"run_id": "run", "generation_family": "family"},
        probe_runtime={"rules_hash": "rules", "feature_schema_hash": 123},
    )
    (host.root / "learner/manifests").mkdir(parents=True)
    (host.root / "arena").mkdir()
    model = {
        "weights": "ema",
        "model_identity": learned,
        "checkpoint_sha256": "1" * 64,
        "model_step": 572377,
        "run_id": "run",
        "generation_family": "family",
        "rules_hash": "rules",
        "feature_schema_hash": f"{123:016x}",
    }
    manifest = host.root / "learner/manifests/model.json"
    manifest.write_bytes(a.encoded(model))
    pointer = {
        k: model[k]
        for k in ("model_identity", "model_step", "run_id", "generation_family")
    }
    pointer.update(
        role="champion",
        manifest="manifests/model.json",
        manifest_sha256=a.checksum(manifest.read_bytes()),
        manifest_bytes=manifest.stat().st_size,
        promotion_result="../arena/final.json",
    )
    (host.root / "learner/champion.json").write_bytes(a.encoded(pointer))
    result = {
        "evaluation_contract": {
            "identity": host.manifest["promotion_contract_identity"]
        },
        "candidate": learned,
        "baseline": baseline,
        "terminal": True,
        "conclusive": True,
        "promotion": {"decision": "promote"},
    }
    result_path = host.root / "arena/final.json"
    result_path.write_bytes(a.encoded(result))
    for name in ("final.resume.json", "final.allocation.json"):
        (host.root / "arena" / name).write_text("{}")
    observed = []
    monkeypatch.setattr(
        champion_migration,
        "_validate_recorded_evaluation",
        lambda docs: observed.append(docs),
    )
    verified = a._champions_worker(host)
    assert learned in verified and observed[0]["result.json"] == result
    oldest = "sha256-" + "4" * 64
    host.manifest["verified_champions"] = {oldest: a._file_pin(proof)}
    intermediate = {**result, "candidate": baseline, "baseline": oldest}
    middle = (
        host.root / "arena" / ("balanced-test-" + baseline + "-vs-" + oldest + ".json")
    )
    middle.write_bytes(a.encoded(intermediate))
    for suffix in (".resume.json", ".allocation.json"):
        middle.with_name(middle.stem + suffix).write_text("{}")
    verified = a._champions_worker(host)
    assert set(verified) == {oldest, baseline, learned}
    intermediate["conclusive"] = False
    middle.write_bytes(a.encoded(intermediate))
    with pytest.raises(g.Refusal, match="proof-decision"):
        a._champions_worker(host)
    host.manifest["verified_champions"] = {baseline: a._file_pin(proof)}
    result["terminal"] = False
    result_path.write_bytes(a.encoded(result))
    with pytest.raises(g.Refusal, match="proof-decision"):
        a._champions_worker(host)


def install_progress_capture(host, monkeypatch):
    fixture = runpy.run_path(
        str(Path(__file__).with_name("test_strength_freshness_progress.py"))
    )["capture"].__wrapped__
    policy, data, args = fixture()
    paths = {
        "coordinator": "status/coordinator.json",
        "continuation_state": "continuation.json",
        "metrics": "metrics.jsonl",
        "replay_db": "replay.sqlite3",
        "heartbeats": {n: "status/" + n + ".json" for n in data["heartbeats"]},
        "cohorts": {n: "status/" + n + ".json" for n in data["cohorts"]},
        "policies": {"r4": policy.as_dict()},
    }
    host.manifest["telemetry"] = paths
    for name in ("coordinator", "continuation_state"):
        p = host.root / paths[name]
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(a.encoded(data[name]))
    for name in ("heartbeats", "cohorts"):
        for worker, path in paths[name].items():
            (host.root / path).write_bytes(a.encoded(data[name][worker]))
    (host.root / "metrics.jsonl").write_bytes(
        b"".join((json.dumps(row) + "\n").encode() for row in data["metrics"])
    )
    (host.root / "profile.yaml").write_bytes(b"profile")
    sha = a.checksum(b"profile")
    paths["policies"]["r4"]["profile_sha256"] = sha
    (host.root / "profile.sha256").write_text(
        sha + "  " + str(host.root / "profile.yaml") + "\n"
    )
    (host.root / "source-commit.txt").write_text(policy.source_commit)
    host.plan["run_identity"] = {
        "run_id": "run",
        "generation_family": "family",
        "created_ns": 1,
    }
    with sqlite3.connect(host.root / "replay.sqlite3") as db:
        db.executescript(
            "CREATE TABLE runs(run_id,generation_family,created_ns); CREATE TABLE run_counters(run_id,generation_family,committed_samples,updated_ns,history_complete);"
        )
        db.execute("INSERT INTO runs VALUES(?,?,?)", ("run", "family", 1))
        db.execute(
            "INSERT INTO run_counters VALUES(?,?,?,?,?)",
            ("run", "family", 2500, data["replay_counter"]["updated_ns"], 1),
        )
    (host.root / "learner").mkdir()
    (host.root / "learner/champion.json").write_bytes(b"{}")
    host._champion_cache = (a.checksum(b"{}"), args["verified_champions"])
    g.Journal(host.state).save("state.json", {"boundary": {"arena_evidence": {}}})
    unit = args["unit"]
    host.io.groups[unit.cgroup] = unit.members
    for p in unit.members:
        host.io.processes[p.pid] = {
            "pid": p.pid,
            "start_ticks": p.start_ticks,
            "cgroup": "0::" + unit.cgroup,
            "exe": "/qualified/python3.11",
            "cwd": "/qualified/training",
        }
        host.io.files[f"/proc/{p.pid}/environ"] = b"PYTHONPATH=/qualified\0"
    host.io.files["/proc/stat"] = (
        f"btime {args['clock'].wall_ns // 10**9 - 11}\n".encode()
    )
    host.manifest["process_runtime"] = {
        "r4": {
            "executables": ["/qualified/python3.11"],
            "working_directories": ["/qualified/training"],
            "environment": {"PYTHONPATH": "/qualified"},
        }
    }
    monkeypatch.setattr(host, "clock", lambda: args["clock"])
    return data, args, paths


def test_progress_fixture_never_resolves_real_proc_paths(host, monkeypatch):
    resolve = Path.resolve

    def forbid_proc(path, *args, **kwargs):
        if path.is_relative_to("/proc"):
            raise AssertionError("fake process fixture reached real /proc")
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", forbid_proc)
    _, args, _ = install_progress_capture(host, monkeypatch)
    units = {"r4": args["unit"], "r3": g.Unit("r3.service", "a" * 64, "/r3")}
    result = host._progress(units, args["clock"], 105)
    assert result is not None and "contract_failure" not in result


def test_missing_heartbeat_cannot_hide_later_current_owned_failure(host, monkeypatch):
    data, args, paths = install_progress_capture(host, monkeypatch)
    (host.root / paths["heartbeats"]["learner"]).unlink()
    units = {"r4": args["unit"], "r3": g.Unit("r3.service", "a" * 64, "/r3")}
    assert host._progress(units, args["clock"], 105) is None
    data["heartbeats"]["actor-gpu-6"]["phase"] = "failed"
    (host.root / paths["heartbeats"]["actor-gpu-6"]).write_bytes(
        a.encoded(data["heartbeats"]["actor-gpu-6"])
    )
    assert host._progress(units, args["clock"], 105) == {
        "role": "r4",
        "contract_failure": "current-worker-failed",
    }


def test_collection_clock_is_sampled_after_new_telemetry(host, monkeypatch):
    data, args, paths = install_progress_capture(host, monkeypatch)
    units = {"r4": args["unit"], "r3": g.Unit("r3.service", "a" * 64, "/r3")}
    # The initial observation predates these records; the capture-end clock is
    # the one against which strict future-time validation must run.
    before = g.Clock(
        args["clock"].boot_id,
        args["clock"].monotonic - 1,
        args["clock"].wall_ns - 5 * 10**9,
    )
    result = host._progress(units, before, 105)
    assert result is not None and "contract_failure" not in result


def test_prepare_then_real_core_begin_creates_fresh_journal_once(host, tmp_path):
    path, m, plan = full_manifest(host, tmp_path)
    # Rebind the genuinely hashed fixture after adding the observed fake GPU
    # inventory; all service/process operations still go through FakeIO.
    plan["gpu_identity_sha256"] = g.digest(
        [
            [str(i), f"GPU-{i}", f"00:{i:02x}", "fake", "fake-driver", "1000"]
            for i in range(8)
        ]
    )
    guard = Path(m["guard_plan"]["path"])
    guard.write_bytes(a.encoded(plan))
    m["guard_plan"] = a._file_pin(guard)
    m["core_plan_sha256"] = g.digest(plan)
    path.write_bytes(a.encoded(m))
    io_ = host.io
    for name, spec in m["units"].items():
        io_.raw[name].update(spec["before"]["properties"])
    initial = plan["initial_r3"]
    p = g.Process(**initial["main"])
    io_.raw["r3.service"].update(
        MainPID=str(p.pid),
        ExecMainPID=str(p.pid),
        InvocationID=initial["invocation_id"],
        ActiveState="active",
        SubState="running",
        UnitFileState="enabled",
        ExecMainStartTimestampMonotonic="99000000",
    )
    io_.processes[p.pid] = {
        "pid": p.pid,
        "start_ticks": p.start_ticks,
        "cgroup": "0::/system.slice/r3.service",
    }
    io_.groups["/system.slice/r3.service"] = (p,)
    for name, pin in plan["support_transition"]["before"].items():
        io_.raw[name]["UnitFileState"] = "enabled" if pin["enabled"] else "disabled"
    io_.child_result = {
        "phase": "r3",
        "plan_sha256": plan["freshness_plan_sha256"],
        "evidence_sha256": "b" * 64,
    }
    real = a.LinuxHost(path, a.checksum(path.read_bytes()), execute=True, io_=io_)
    real.verify_inputs(plan, 130)
    assert (
        not real.state.exists()
        and (real.prepared_directory() / a.AUTHORIZATION).is_file()
    )
    controller = g.Controller(plan, g.digest(plan), g.Journal(real.state), real)
    state = controller.begin()
    assert state["phase"] == "arming" and (real.state / "anchor.json").is_file()
    assert any(
        call[0] == ["systemctl", "start", "--no-block", "guard.service"]
        for call in io_.calls
    )
    old = (real.state / "anchor.json").read_bytes()
    with pytest.raises(g.Refusal):
        real.verify_inputs(plan, 130)
    assert (real.state / "anchor.json").read_bytes() == old


def test_timer_uses_unit_properties_and_known_empty_service_array_is_explicit(host):
    name = "report.timer"
    raw = {k: "" for k in a.TIMER_PROPERTIES}
    raw.update(
        Id=name,
        LoadState="loaded",
        ActiveState="active",
        SubState="waiting",
        Job="0 /",
        UnitFileState="disabled",
        FragmentPath="/etc/systemd/system/" + name,
    )
    data = b"[Timer]\nOnCalendar=daily\n"
    host.io.raw[name] = raw
    host.io.files[raw["FragmentPath"]] = data
    spec = {
        "properties": {k: raw[k] for k in set(a.TIMER_PROPERTIES) - a.VARIABLE},
        "unit": {
            "path": raw["FragmentPath"],
            "bytes": len(data),
            "sha256": a.checksum(data),
        },
        "environment_files": [],
    }
    host.manifest["units"][name] = {
        "boot_links": {},
        "installed_path": raw["FragmentPath"],
        "before": spec,
        "after": spec,
    }
    unit, _ = host._unit(name, {}, 105)
    assert unit.active == "active" and unit.main is None and not unit.members
    del host.io.raw["r3.service"]["EnvironmentFiles"]
    assert host._unit("r3.service", {}, 105)[0].dead
    del host.io.raw["r3.service"]["MainPID"]
    with pytest.raises(g.Refusal, match="unit-incomplete"):
        host._unit("r3.service", {}, 105)


def test_known_cgroup_growth_retries_with_original_deadline(host, monkeypatch):
    calls = []
    wanted = observation(host)

    def capture(deadline):
        calls.append(deadline)
        if len(calls) < 3:
            raise a.ObservationChanged("new compiler child")
        return wanted

    monkeypatch.setattr(host, "_observe_once", capture)
    assert host.observe(105) == wanted and calls == [105, 105, 105]
    monkeypatch.setattr(
        host,
        "_observe_once",
        lambda _: (_ for _ in ()).throw(a.ObservationChanged("changing")),
    )
    with pytest.raises(TimeoutError):
        host.observe(105)


def test_noncertifying_compiler_is_pending_without_hiding_owned_failure(
    host, monkeypatch
):
    data, args, paths = install_progress_capture(host, monkeypatch)
    compiler = g.Process(999, 600)
    unit = replace(args["unit"], members=(*args["unit"].members, compiler))
    host.io.groups[unit.cgroup] = unit.members
    host.io.processes[compiler.pid] = {
        "pid": compiler.pid,
        "start_ticks": compiler.start_ticks,
        "cgroup": "0::" + unit.cgroup,
        "exe": "/unqualified/compiler",
        "cwd": "/unqualified",
    }
    host.io.files[f"/proc/{compiler.pid}/environ"] = b"PYTHONPATH=/qualified\0"
    host.io.files[f"/proc/{compiler.pid}/cmdline"] = b"unknown-compiler\0"
    units = {"r4": unit, "r3": g.Unit("r3.service", "a" * 64, "/r3")}
    assert host._progress(units, args["clock"], 105) is None
    data["heartbeats"]["actor-gpu-6"]["phase"] = "failed"
    (host.root / paths["heartbeats"]["actor-gpu-6"]).write_bytes(
        a.encoded(data["heartbeats"]["actor-gpu-6"])
    )
    assert host._progress(units, args["clock"], 105) == {
        "role": "r4",
        "contract_failure": "current-worker-failed",
    }


def test_guardian_orders_partial_repair_admission_and_exact_tail(host, monkeypatch):
    from scripts import strength_freshness_units as units

    pid = os.getpid()
    host.io.processes[pid] = {"pid": pid, "start_ticks": 20, "cgroup": "0::/guard"}
    monkeypatch.setenv("INVOCATION_ID", "f" * 32)
    g.Journal(host.state).save(
        "state.json",
        {
            "deadline_wall_ns": time.time_ns() + 120_000_000_000,
            "deadline": 220.0,
            "active_boot_id": "boot-a",
            "owners": {
                "guard": {
                    "main": {"pid": pid, "start_ticks": 10},
                    "invocation_id": "a" * 32,
                    "boot_id": "boot-a",
                }
            },
        },
    )
    order = []
    action = {"kind": "restore-support-and-backup-r4"}

    class Controller:
        def __init__(self, *_):
            pass

        def admit_guard_reentry(self):
            assert order == ["pre-admission-repair"]
            order.append("admit")

        def tick(self):
            assert order == [
                "pre-admission-repair",
                "admit",
                "support-complete",
                "tail",
            ]
            order.append("tick")
            return {"finished": True}

    @contextmanager
    def hold():
        yield

    def recover(_host, deadline, *, pre_admission=False):
        order.append("pre-admission-repair" if pre_admission else "support-complete")
        return None if pre_admission else {"action": action}

    monkeypatch.setattr(host, "hold_lease", hold)
    monkeypatch.setattr(g, "Controller", Controller)
    monkeypatch.setattr(units, "recover_transition", recover)
    monkeypatch.setattr(
        host,
        "resume_support_tail",
        lambda value: (
            order.append("tail") if value == action else pytest.fail("different action")
        ),
    )
    assert a.guardian(host)["finished"] and order[-1] == "tick"


def test_backup_tail_is_once_only_and_defers_until_owned_runtime(host, monkeypatch):
    from scripts.strength_freshness_units import DIRECTORY

    action = {
        "kind": "restore-support-and-backup-r4",
        "deadline": 200.0,
        "details": {
            "support_stage": "after",
            "proof_closure_sha256": "a" * 64,
            "required_proof_sha256": ["b" * 64],
        },
    }
    state = {
        "last_action": action,
        "attempt_id": "one",
        "nonce": "n",
        "deadline": 200.0,
        "owners": {},
    }
    g.Journal(host.state).save("state.json", state)
    folder = host.state / DIRECTORY
    folder.mkdir()
    identity = g.digest({"action": action, "attempt_id": "one", "nonce": "n"})
    (folder / (identity + ".intent.json")).write_bytes(
        a.encoded({"deadline_wall_ns": time.time_ns() + 90_000_000_000})
    )
    monkeypatch.setattr(host, "observe", lambda _: observation(host))
    host.resume_support_tail(action)
    assert not (host.state / "linux-backup-request.json").exists()
    activate(host, "r4")
    unit = observation(host).units["r4"]
    assert unit.main is not None
    state["owners"]["r4"] = {
        "main": asdict(unit.main),
        "invocation_id": unit.invocation_id,
        "boot_id": "boot-a",
    }
    g.Journal(host.state).save("state.json", state)
    host.resume_support_tail(action)
    host.resume_support_tail(action)
    assert (
        len(
            [
                call
                for call in host.io.calls
                if call[0] == ["systemctl", "start", "--no-block", "backup.service"]
            ]
        )
        == 1
    )
    assert (host.state / "linux-backup-dispatch.json").is_file()


def test_retirement_guard_drift_prevents_disable(host, monkeypatch):
    obs = observation(host)
    monkeypatch.setattr(host, "observe", lambda _: obs)
    d = details(host, obs)
    d["expected_guard"] = {**asdict(obs.units["guard"]), "invocation_id": "f" * 32}
    before = len(host.io.calls)
    with pytest.raises(g.Refusal, match="retirement-guard-raced"):
        host.perform("retire-guard-only", d, 150)
    assert len(host.io.calls) == before


@pytest.mark.parametrize("where", ["link", "boundary-record"])
def test_capture_boundary_resumes_exact_publication_after_crash(
    host, tmp_path, monkeypatch, where
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixed tiny weights")
    value = {"checkpoint": a._file_pin(checkpoint)}
    monkeypatch.setattr(host, "_worker", lambda *_: copy.deepcopy(value))
    link = os.link
    atomic = host.io.atomic
    fired = False

    def interrupt_link(*args, **kwargs):
        nonlocal fired
        link(*args, **kwargs)
        if where == "link" and not fired:
            fired = True
            raise RuntimeError("crashed after link")

    def interrupt_atomic(path, *args, **kwargs):
        nonlocal fired
        atomic(path, *args, **kwargs)
        if (
            where == "boundary-record"
            and path.name == "linux-stopped-boundary.json"
            and not fired
        ):
            fired = True
            raise RuntimeError("crashed after record")

    monkeypatch.setattr(os, "link", interrupt_link)
    monkeypatch.setattr(host.io, "atomic", interrupt_atomic)
    with pytest.raises(RuntimeError, match="crashed"):
        host.capture_boundary(host.plan, 105)
    result = host.capture_boundary(host.plan, 105)
    retained = Path(result["checkpoint_retention"]["path"])
    assert retained.samefile(checkpoint) and result["checkpoint"][
        "sha256"
    ] == a.checksum(checkpoint.read_bytes())
    checkpoint.unlink()
    assert retained.read_bytes() == b"fixed tiny weights"


def test_disjoint_known_env_paths_allow_coherent_after_without_deleting_before(
    host, tmp_path
):
    spec = host.manifest["units"]["monitor.service"]
    before = {
        "path": "/etc/old.env",
        "source_path": str(tmp_path / "old.env"),
        "sha256": a.checksum(b"old"),
        "bytes": 3,
    }
    after = {
        "path": "/etc/new.env",
        "source_path": str(tmp_path / "new.env"),
        "sha256": a.checksum(b"new"),
        "bytes": 3,
    }
    host.io.files[before["path"]] = b"old"
    host.io.files[after["path"]] = b"new"
    spec["before"]["environment_files"] = [before]
    spec["after"]["environment_files"] = [after]
    spec["before"]["properties"]["EnvironmentFiles"] = "/etc/old.env"
    spec["after"]["properties"]["EnvironmentFiles"] = "/etc/new.env"
    host.io.raw["monitor.service"]["EnvironmentFiles"] = "/etc/new.env"
    _, metadata = host._unit("monitor.service", {}, 105)
    assert metadata["stage"] == "after" and metadata[
        "environment_sha256"
    ] == a.env_identity("", [after])
    assert host.io.files["/etc/old.env"] == b"old"


def test_disappearing_stopping_main_and_stale_nvml_are_retryable_not_empty(host):
    p = activate(host, "r3")
    host.io.processes.pop(p.pid)
    with pytest.raises(a.ObservationChanged, match="main-disappeared"):
        host._unit("r3.service", {}, 105)
    host.io.nvml = f"GPU-0, {p.pid}"
    with pytest.raises(a.ObservationChanged, match="stale-owner"):
        host._gpus({}, 105)


def test_qualified_persistent_auxiliary_does_not_block_or_certify_progress(
    host, monkeypatch
):
    data, args, paths = install_progress_capture(host, monkeypatch)
    parent = data["heartbeats"]["learner"]["pid"]
    aux = g.Process(999, 600)
    origin = host.io.process_origin(parent, 105)
    unit = replace(args["unit"], members=(*args["unit"].members, aux))
    host.io.groups[unit.cgroup] = unit.members
    host.io.processes[aux.pid] = {
        "pid": aux.pid,
        "start_ticks": aux.start_ticks,
        "ppid": parent,
        "cgroup": "0::" + unit.cgroup,
        **origin,
    }
    host.io.files[f"/proc/{aux.pid}/environ"] = b"PYTHONPATH=/qualified\0"
    argv = [
        "/qualified/python",
        "-B",
        "-c",
        "from multiprocessing.resource_tracker import main;main(70)",
    ]
    host.io.files[f"/proc/{aux.pid}/cmdline"] = (
        b"\0".join(s.encode() for s in argv) + b"\0"
    )
    host.manifest["auxiliary_policies"] = {
        "r4": {
            "python_argv0": "/qualified/python",
            "executable": origin["exe"],
            "cwd": origin["cwd"],
            "compile_worker_script": "/qualified/compile.py",
            "torch_key": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
            "compile_parent_roles": ["learner"],
        }
    }
    units = {"r4": unit, "r3": g.Unit("r3.service", "a" * 64, "/r3")}
    result = host._progress(units, args["clock"], 105)
    assert (
        result is not None
        and result["auxiliary_processes"]["999"]["kind"]
        == "multiprocessing-resource-tracker"
    )
    assert (
        len(result["worker_processes"]) == len(data["heartbeats"])
        and "999" not in result["worker_processes"]
    )
    host.io.files[f"/proc/{aux.pid}/environ"] = b"PYTHONPATH=/unqualified\0"
    assert host._progress(units, args["clock"], 105) is None


@pytest.mark.parametrize("key", ["ExecStart", "ExecStartPre"])
def test_exec_dynamic_fields_normalize_but_static_ignore_errors_remains(host, key):
    prefix = "{ path=/qualified/python ; argv[]=/qualified/python admit-start"
    expected = (
        prefix
        + " ; ignore_errors=no ; start_time=[old] ; stop_time=[old] ; pid=123 ; code=exited ; status=0 }"
    )
    refreshed = (
        prefix
        + " ; ignore_errors=no ; start_time=[new] ; stop_time=[new] ; pid=456 ; code=killed ; status=15 }"
    )
    ignored = (
        prefix
        + " ; ignore_errors=yes ; start_time=[new] ; stop_time=[new] ; pid=456 ; code=killed ; status=15 }"
    )
    assert a.stable_properties({key: expected}) == a.stable_properties({key: refreshed})
    assert a.stable_properties({key: ignored}) != a.stable_properties({key: expected})
    spec = host.manifest["units"]["r3.service"]
    for stage in ("before", "after"):
        spec[stage]["properties"][key] = expected
    host.io.raw["r3.service"][key] = refreshed
    assert host._unit("r3.service", {}, 105)[0].dead
    host.io.raw["r3.service"][key] = ignored
    with pytest.raises(g.Refusal, match="properties-drift"):
        host._unit("r3.service", {}, 105)
