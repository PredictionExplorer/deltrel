"""Support transactions use fake systemd/filesystem facts; never touch host units."""

from dataclasses import asdict
import json
import os
from pathlib import Path

import pytest

from deltreltrain import strength_freshness_guard as core
from scripts import strength_freshness_units as units


class Crash(BaseException):
    pass


class IO:
    def __init__(self, host):
        self.host = host
        self.files = {}
        self.modes = {}
        self.foreign_owner = False
        self.events = []
        self.crash_path = None
        self.after_write = None
        self.stubborn = False
        self.fail_start = False

    def now(self):
        return self.host.time

    def read(self, path, maximum, deadline):
        assert self.now() < deadline
        value = self.files.get(str(path))
        if value is None:
            value = Path(path).read_bytes()
        assert len(value) <= maximum
        return value

    def json(self, path, deadline):
        return json.loads(self.read(path, 2**20, deadline))

    def pin(self, pin, deadline):
        data = self.read(Path(pin["path"]), 2**20, deadline)
        core.require(
            len(data) == pin["bytes"] and units._sha(data) == pin["sha256"], "fake-pin"
        )

    def atomic(self, path, data, *, overwrite=False, mode=0o600):
        path = Path(path)
        if str(path).startswith("/etc/"):
            if not overwrite and str(path) in self.files:
                raise FileExistsError(path)
            self.files[str(path)] = data
            self.modes[str(path)] = mode
            self.events.append(("write", str(path)))
        else:
            if not overwrite and path.exists():
                raise FileExistsError(path)
            path.write_bytes(data)
            path.chmod(mode)
            self.events.append(("journal", str(path)))
        if self.after_write is not None:
            self.after_write(path)
        if str(path) == self.crash_path:
            self.crash_path = None
            raise Crash()

    def process(self, pid):
        return self.host.processes.get(pid)

    def file_metadata(self, path, deadline):
        assert self.now() < deadline
        return {
            "uid": os.geteuid() + int(self.foreign_owner),
            "gid": os.getegid(),
            "mode": self.modes[str(path)],
        }

    def boot_links(self, name, deadline):
        assert self.now() < deadline
        return {p: v for p, v in self.host.links.items() if Path(p).name == name}

    def command(self, argv, deadline):
        assert self.now() < deadline
        self.events.append(("command", tuple(argv)))
        if argv == ["systemctl", "get-default"]:
            return self.host.default_target
        if argv[:2] == ["systemctl", "show"]:
            return (
                "Wants=" + " ".join(self.host.edges.get(argv[2], [])) + "\nRequires=\n"
            )
        assert argv == ["systemctl", "daemon-reload"]
        for name, spec in self.host.manifest["units"].items():
            data = self.files[spec["installed_path"]]
            stage = next(
                s
                for s in ("before", "after")
                if units._sha(data) == spec[s]["unit"]["sha256"]
            )
            self.host.loaded[name] = spec[stage]["properties"]
        return ""

    def action(self, verb, name, deadline):
        assert self.now() < deadline
        self.events.append((verb, name))
        state = self.host.status[name]
        state["job"] = None
        if verb == "stop":
            if not self.stubborn:
                state.update(active="inactive", main=None, members=())
        else:
            assert verb == "start"
            if self.fail_start:
                state.update(
                    active="failed",
                    main=None,
                    members=(),
                    result="exit-code",
                    exit_code=1,
                )
            elif self.host.loaded[name].get("Type") == "oneshot":
                state.update(active="inactive", main=None, members=())
            else:
                p = core.Process(800, 800)
                state.update(
                    active="active",
                    main=None if name.endswith(".timer") else p,
                    members=() if name.endswith(".timer") else (p,),
                )

    def sleep(self, seconds):
        self.host.time += seconds


class Host:
    def __init__(self, tmp_path, monkeypatch):
        self.state, self.root = tmp_path / "guard", tmp_path / "run"
        self.state.mkdir(mode=0o700)
        self.root.mkdir()
        self.time, self.boot = 10.0, "boot-one"
        self.execute, self.lease_fd = True, 5
        self.io = IO(self)
        self.loaded, self.status, self.links = {}, {}, {}
        self.default_target = "multi-user.target"
        self.edges = {
            "multi-user.target": ["basic.target"],
            "basic.target": ["timers.target"],
        }
        self.manifest = {
            "units": {},
            "boot_topology": {
                "default_target": "multi-user.target",
                "target_paths": {
                    "multi-user.target": ["multi-user.target"],
                    "timers.target": [
                        "multi-user.target",
                        "basic.target",
                        "timers.target",
                    ],
                },
            },
        }
        self.plan = {
            "freshness_plan_sha256": "f" * 64,
            "units": {},
            "support_transition": {"before": {}, "after": {}},
        }
        self.current = core.Process(os.getpid(), 100)
        self.invocation = "1" * 32
        monkeypatch.setenv("INVOCATION_ID", self.invocation)
        self.processes = {self.current.pid: asdict(self.current)}
        for name in (
            "guard.service",
            "r3.service",
            "r4.service",
            "probe.service",
            "monitor.service",
            "report.timer",
            "backup.timer",
        ):
            installed = "/etc/systemd/system/" + name
            target = "timers.target" if name.endswith(".timer") else "multi-user.target"
            spec = {
                "installed_path": installed,
                "boot_links": {f"/etc/systemd/system/{target}.wants/{name}": installed},
            }
            for stage in ("before", "after"):
                source = tmp_path / (name + "-" + stage)
                source.write_bytes((name + stage).encode())
                env_source = tmp_path / (name + "-" + stage + ".env")
                env_source.write_bytes(("PATH=" + stage).encode())
                env = {
                    "path": "/etc/strength/" + name + ".env",
                    "source_path": str(env_source),
                    "sha256": units._sha(env_source.read_bytes()),
                    "bytes": env_source.stat().st_size,
                }
                spec[stage] = {
                    "unit": {
                        "path": str(source),
                        "sha256": units._sha(source.read_bytes()),
                        "bytes": source.stat().st_size,
                    },
                    "properties": {"Type": "exec", "ExecStart": name + stage},
                    "environment_files": [env],
                }
            self.manifest["units"][name] = spec
            self.io.files[installed] = Path(spec["before"]["unit"]["path"]).read_bytes()
            self.io.modes[installed] = 0o644
            self.io.files[spec["before"]["environment_files"][0]["path"]] = (
                b"PATH=before"
            )
            self.io.modes[spec["before"]["environment_files"][0]["path"]] = 0o600
            self.loaded[name] = spec["before"]["properties"]
            enabled = name in (
                "guard.service",
                "monitor.service",
                "report.timer",
                "backup.timer",
            )
            if enabled:
                self.links.update(spec["boot_links"])
            main = self.current if name == "guard.service" else core.Process(700, 700)
            self.status[name] = {
                "main": main,
                "members": (main,),
                "active": "active",
                "job": None,
                "result": "success",
                "exit_code": 0,
            }
            if name in ("monitor.service", "report.timer", "backup.timer"):
                for stage in ("before", "after"):
                    self.plan["support_transition"][stage][name] = {
                        "definition_sha256": spec[stage]["unit"]["sha256"],
                        "environment_sha256": core.digest(
                            spec[stage]["environment_files"]
                        ),
                        "enabled": True,
                    }
            else:
                self.plan["units"][name.removesuffix(".service")] = {
                    "name": name,
                    "initial_enabled": name == "r3.service",
                    "committed_enabled": name == "r4.service",
                }
        self.manifest_sha256 = core.digest(self.manifest)
        state = {
            "plan_sha256": core.digest(self.plan),
            "attempt_id": "one",
            "nonce": "a" * 64,
            "phase": "backup-r4",
            "last_wall_ns": self.clock().wall_ns,
            "deadline_wall_ns": self.clock().wall_ns + 190_000_000_000,
            "active_boot_id": self.boot,
            "deadline": 200.0,
            "owners": {
                "guard": {
                    "main": asdict(self.current),
                    "invocation_id": self.invocation,
                    "boot_id": self.boot,
                }
            },
            "last_action": {
                "kind": "restore-support-and-backup-r4",
                "deadline": 100.0,
                "details": {},
            },
        }
        anchor = {
            k: state[k]
            for k in ("plan_sha256", "attempt_id", "nonce", "deadline_wall_ns")
        }
        (self.state / "anchor.json").write_bytes(units._bytes(anchor))
        state["anchor_sha256"] = units._sha((self.state / "anchor.json").read_bytes())
        (self.state / "state.json").write_bytes(units._bytes(state))

    def clock(self):
        return core.Clock(
            self.boot, self.time, 1_000_000_000_000 + int(self.time * 1e9)
        )

    def _jobs(self, deadline):
        return {}

    def _unit(self, name, jobs, deadline, *, allow_known_partial=False):
        spec = self.manifest["units"][name]
        data = self.io.read(Path(spec["installed_path"]), 2**20, deadline)
        core.require(
            any(
                units._sha(data) == spec[s]["unit"]["sha256"]
                for s in ("before", "after")
            ),
            "unknown-unit",
        )
        core.require(
            any(
                self.loaded[name] == spec[s]["properties"] for s in ("before", "after")
            ),
            "unknown-loaded",
        )
        possible = []
        for stage in ("before", "after"):
            if (
                self.loaded[name] != spec[stage]["properties"]
                or units._sha(data) != spec[stage]["unit"]["sha256"]
            ):
                continue
            if all(
                units._sha(self.io.read(Path(e["path"]), 2**20, deadline))
                == e["sha256"]
                and self.io.modes[e["path"]] == e.get("mode", 0o600)
                for e in spec[stage]["environment_files"]
            ):
                possible.append(stage)
        core.require(possible or allow_known_partial, "mixed-support")
        stage = possible[0] if possible else "mixed"
        values = self.status[name]
        unit = core.Unit(
            name,
            units._sha(data),
            "/system.slice/" + name,
            invocation_id=self.invocation,
            enabled=bool(self.io.boot_links(name, deadline)),
            substate="running" if values["active"] == "active" else "dead",
            **values,
        )
        return unit, {
            "definition_sha256": unit.definition_sha256,
            "environment_sha256": core.digest(spec[stage]["environment_files"])
            if possible
            else "mixed",
            "stage": stage,
        }

    def _lease(self, deadline):
        return {
            "owner": asdict(self.current),
            "boot_id": self.boot,
            "invocation_id": self.invocation,
            "nonce": "a" * 64,
            "attempt_id": "one",
            "plan_sha256": core.digest(self.plan),
        }

    def _authority(self, deadline):
        return core.Authority("r4", self.plan["freshness_plan_sha256"], "b" * 64)

    def _enable(self, name, enabled, deadline):
        self.io.events.append(("enable", name, enabled))
        self.links = {p: v for p, v in self.links.items() if Path(p).name != name}
        if enabled:
            self.links.update(self.manifest["units"][name]["boot_links"])


@pytest.fixture
def host(tmp_path, monkeypatch):
    return Host(tmp_path, monkeypatch)


def test_transition_is_journal_first_exact_and_idempotent(host):
    result = units.transition(host, "after", 100)
    assert result["stage"] == "after"
    assert result["action"]["kind"] == "restore-support-and-backup-r4"
    first_write = next(i for i, e in enumerate(host.io.events) if e[0] == "write")
    assert any(
        e[0] == "journal" and e[1].endswith(".intent.json")
        for e in host.io.events[:first_write]
    )
    events = list(host.io.events)
    assert units.transition(host, "after", 100) == result
    assert all(e[0] == "command" for e in host.io.events[len(events) :])
    for name in units._names(host):
        assert host._unit(name, {}, 100)[1]["stage"] == "after"
        units.verify_boot_edges(host, name, True, 100)
    assert not any(
        e[0] == "start" and e[1] in {"r3.service", "r4.service", "probe.service"}
        for e in host.io.events
    )


@pytest.mark.parametrize("part", ["unit", "env"])
def test_partial_write_recovers_forward_and_preserves_intent(host, part):
    spec = host.manifest["units"]["monitor.service"]
    host.io.crash_path = (
        spec["installed_path"]
        if part == "unit"
        else spec["after"]["environment_files"][0]["path"]
    )
    with pytest.raises(Crash):
        units.transition(host, "after", 100)
    intent = next((host.state / units.DIRECTORY).glob("*.intent.json"))
    saved = intent.read_bytes()
    with pytest.raises(core.Refusal, match="mixed"):
        host._unit("monitor.service", {}, 100)
    units.recover_transition(host, 150)
    assert intent.read_bytes() == saved
    assert host._unit("monitor.service", {}, 150)[1]["stage"] == "after"


def test_env_only_change_is_a_real_after_stage(host):
    spec = host.manifest["units"]["monitor.service"]
    spec["after"]["unit"] = spec["before"]["unit"]
    spec["after"]["properties"] = spec["before"]["properties"]
    host.plan["support_transition"]["after"]["monitor.service"]["definition_sha256"] = (
        spec["before"]["unit"]["sha256"]
    )
    state = host.io.json(host.state / "state.json", 100)
    state["plan_sha256"] = core.digest(host.plan)
    anchor = host.io.json(host.state / "anchor.json", 100)
    anchor["plan_sha256"] = state["plan_sha256"]
    (host.state / "anchor.json").write_bytes(units._bytes(anchor))
    state["anchor_sha256"] = units._sha((host.state / "anchor.json").read_bytes())
    (host.state / "state.json").write_bytes(units._bytes(state))
    units.transition(host, "after", 100)
    assert host._unit("monitor.service", {}, 100)[1]["stage"] == "after"


@pytest.mark.parametrize(
    "fault", ["foreign-env", "expired", "no-lease", "phase", "old-alive"]
)
def test_recovery_refuses_unowned_or_expired_transaction(host, monkeypatch, fault):
    spec = host.manifest["units"]["monitor.service"]
    host.io.crash_path = spec["installed_path"]
    with pytest.raises(Crash):
        units.transition(host, "after", 100)
    if fault == "foreign-env":
        host.io.files[spec["after"]["environment_files"][0]["path"]] = b"unknown"
    elif fault == "expired":
        host.time = 101
    elif fault == "no-lease":
        host.lease_fd = None
    elif fault == "phase":
        state = host.io.json(host.state / "state.json", 100)
        state["phase"] = "terminal-cleanup"
        (host.state / "state.json").write_bytes(units._bytes(state))
    else:
        previous = host.current
        host.current = core.Process(previous.pid + 1, 200)
        host.invocation = "2" * 32
        host.processes[host.current.pid] = asdict(host.current)
        host.status["guard.service"].update(main=host.current, members=(host.current,))
        monkeypatch.setattr(units.os, "getpid", lambda: host.current.pid)
        monkeypatch.setenv("INVOCATION_ID", host.invocation)
    before = list(host.io.events)
    with pytest.raises(core.Refusal):
        units.recover_transition(host, 150, pre_admission=fault == "old-alive")
    assert host.io.events == before or all(
        e[0] == "command" for e in host.io.events[len(before) :]
    )


def test_target_race_is_not_overwritten(host):
    env = host.manifest["units"]["monitor.service"]["after"]["environment_files"][0][
        "path"
    ]

    def race(path):
        if str(path) == host.manifest["units"]["monitor.service"]["installed_path"]:
            host.io.files[env] = b"foreign-env"

    host.io.after_write = race
    with pytest.raises(core.Refusal, match="unowned|mixed"):
        units.transition(host, "after", 100)
    assert host.io.files[env] == b"foreign-env"


def test_stop_job_absence_does_not_prove_cgroup_drain(host):
    host.io.stubborn = True
    with pytest.raises(core.Refusal, match="drain"):
        units.transition(host, "after", 100)
    assert not any(e[0] == "write" for e in host.io.events)


def test_inactive_queued_start_is_cancelled_before_install(host):
    host.status["monitor.service"].update(
        active="inactive", main=None, members=(), job={"id": 8, "kind": "start"}
    )
    units.transition(host, "after", 100)
    stop = host.io.events.index(("stop", "monitor.service"))
    write = host.io.events.index(
        ("write", host.manifest["units"]["monitor.service"]["installed_path"])
    )
    assert stop < write


@pytest.mark.parametrize(
    "fault", ["runtime-link", "unknown-link", "wrong-target", "unreachable", "default"]
)
def test_persistent_boot_edges_are_observed_not_asserted(host, fault):
    name = "report.timer"
    path = next(iter(host.manifest["units"][name]["boot_links"]))
    if fault == "runtime-link":
        host.links[path.replace("/etc/", "/run/")] = host.links.pop(path)
    elif fault == "unknown-link":
        host.links[path.replace("timers.target", "unused.target")] = host.links[path]
    elif fault == "wrong-target":
        host.links[path] = "/etc/systemd/system/unknown.timer"
    elif fault == "unreachable":
        host.edges["basic.target"] = []
    else:
        host.default_target = "unused.target"
    with pytest.raises(core.Refusal):
        units.verify_boot_edges(host, name, True, 100)


def test_failed_support_restart_cannot_be_certified(host):
    host.io.fail_start = True
    with pytest.raises(core.Refusal, match="start-failed"):
        units.transition(host, "after", 100)
    assert not list((host.state / units.DIRECTORY).glob("*.complete.json"))


def rebind(host):
    state = host.io.json(host.state / "state.json", 100)
    state["plan_sha256"] = core.digest(host.plan)
    anchor = host.io.json(host.state / "anchor.json", 100)
    anchor["plan_sha256"] = state["plan_sha256"]
    (host.state / "anchor.json").write_bytes(units._bytes(anchor))
    state["anchor_sha256"] = units._sha((host.state / "anchor.json").read_bytes())
    (host.state / "state.json").write_bytes(units._bytes(state))


def test_mode_only_env_change_is_applied_and_observed(host):
    spec = host.manifest["units"]["monitor.service"]
    before, after = (spec[s]["environment_files"][0] for s in ("before", "after"))
    Path(after["source_path"]).write_bytes(Path(before["source_path"]).read_bytes())
    after.update(sha256=before["sha256"], bytes=before["bytes"], mode=0o640)
    spec["after"]["unit"], spec["after"]["properties"] = (
        spec["before"]["unit"],
        spec["before"]["properties"],
    )
    host.plan["support_transition"]["after"]["monitor.service"].update(
        definition_sha256=spec["after"]["unit"]["sha256"],
        environment_sha256=core.digest(spec["after"]["environment_files"]),
    )
    rebind(host)
    units.transition(host, "after", 100)
    assert host.io.files[after["path"]] == b"PATH=before"
    assert host.io.modes[after["path"]] == 0o640
    assert ("write", after["path"]) in host.io.events


@pytest.mark.parametrize("fault", ["mode", "owner"])
def test_unknown_mode_or_owner_refuses_before_install(host, fault):
    if fault == "owner":
        host.io.foreign_owner = True
    else:
        host.io.modes["/etc/strength/monitor.service.env"] = 0o666
    with pytest.raises(core.Refusal, match="mode"):
        units.transition(host, "after", 100)
    assert not any(e[0] == "write" for e in host.io.events)


def test_replacement_guard_metadata_repair_precedes_real_core_reentry(
    host, tmp_path, monkeypatch
):
    """Real finite-core admission sees coherent facts only after explicit repair."""
    from dataclasses import replace
    from test_strength_freshness_guard import FakeHost, make_plan, until_phase, drive

    directory = tmp_path / "core"
    directory.mkdir()
    plan = make_plan(directory)
    for stage in ("before", "after"):
        plan["support_transition"][stage] = host.plan["support_transition"][stage]
    plan["support_transition"]["sha256"] = core.digest(
        {k: v for k, v in plan["support_transition"].items() if k != "sha256"}
    )
    journal = core.Journal(Path(plan["state_root"]))
    runtime = FakeHost(plan, journal)
    controller = core.Controller(plan, core.digest(plan), journal, runtime)
    host.plan, host.state, host.root = plan, journal.directory, Path(plan["run_root"])
    host.clock, host.io.now, host.io.sleep = (
        runtime.clock,
        lambda: runtime.time,
        runtime.advance,
    )
    host._authority, host._lease = lambda _: runtime.authority, lambda _: runtime.lease
    unit_reader, enabler, observer = host._unit, host._enable, runtime.observe
    role_names = {v["name"]: k for k, v in plan["units"].items()}

    def process(pid):
        for unit in runtime.units.values():
            for member in unit.members:
                if member.pid == pid:
                    return asdict(member)
        return None

    def read_unit(name, jobs, deadline, **kwargs):
        if name in role_names:
            return runtime.units[role_names[name]], {}
        return unit_reader(name, jobs, deadline, **kwargs)

    def enable(name, enabled, deadline):
        enabler(name, enabled, deadline)
        if name in role_names:
            role = role_names[name]
            runtime.units[role] = replace(runtime.units[role], enabled=enabled)

    def observe(deadline):
        value = observer(deadline)
        support = {}
        for name in units._names(host):
            unit, metadata = read_unit(name, {}, deadline)
            support[name] = {
                k: metadata[k] for k in ("definition_sha256", "environment_sha256")
            }
            support[name].update(enabled=unit.enabled, job=unit.job)
        return replace(value, support=support)

    host._unit, host._enable, host.io.process = read_unit, enable, process
    runtime.observe = observe
    original_perform = runtime.perform
    pause_kind = "pause-support-and-stop-r3"

    def pause():
        runtime.behavior.pop(pause_kind)
        _, deadline, details = runtime.calls[-1]
        original_perform(pause_kind, details, deadline)
        for name in units._names(host):
            enable(name, False, deadline)
            host.io.action("stop", name, deadline)
        runtime.behavior[pause_kind] = pause

    runtime.behavior[pause_kind] = pause
    controller.begin()
    until_phase(controller, runtime, "canary-r4")
    real_pid = os.getpid()
    monkeypatch.setattr(
        units.os,
        "getpid",
        lambda: (
            runtime.units["guard"].main.pid if runtime.units["guard"].main else real_pid
        ),
    )
    monkeypatch.setenv("INVOCATION_ID", runtime.units["guard"].invocation_id)
    host.io.crash_path = host.manifest["units"]["monitor.service"]["installed_path"]
    action = "restore-support-and-backup-r4"
    runtime.behavior[action] = lambda: units.transition(
        host, "after", journal.read("state.json")["last_action"]["deadline"]
    )
    with pytest.raises(Crash):
        until_phase(controller, runtime, "backup-r4")
    before = journal.read("state.json")
    runtime.advance()
    runtime.activate("guard")
    guard = runtime.units["guard"]
    assert runtime.lease is not None and guard.main is not None
    runtime.lease.update(owner=asdict(guard.main), invocation_id=guard.invocation_id)
    monkeypatch.setenv("INVOCATION_ID", guard.invocation_id)
    with pytest.raises(core.Refusal, match="mixed"):
        controller.admit_guard_reentry()
    events = len(host.io.events)
    units.recover_transition(
        host, before["last_action"]["deadline"], pre_admission=True
    )
    assert not any(
        e[0] in {"start", "stop"} or (e[0] == "enable" and e[2])
        for e in host.io.events[events:]
    )
    admitted = controller.admit_guard_reentry()
    assert (
        admitted["phase"] == before["phase"]
        and admitted["deadline"] == before["deadline"]
    )
    result = units.recover_transition(host, before["last_action"]["deadline"])
    assert result is not None
    assert result["action"] == before["last_action"]
    # A replacement adapter must resume the exact authorized backup-action tail.
    runtime.behavior.pop(action)
    original_perform(
        action,
        {**before["last_action"]["details"], "guard_plan_sha256": core.digest(plan)},
        before["last_action"]["deadline"],
    )
    completed = drive(controller, runtime)
    assert completed["outcome"] == "committed-r4"
    assert sum(row[0] == "start-r4" for row in runtime.calls) == 1
    assert not any(row[0] == "start-r3" for row in runtime.calls)
