"""CPU-only fault injection. No systemctl, NVML, network or real sleeps."""

from __future__ import annotations

import copy
import io
from http.client import HTTPMessage
import os
from pathlib import Path

import pytest

from scripts import qualify_cloud_gpu_window as op


class FakeHost(op.Host):
    def __init__(self, plan, directory):
        self.plan = plan
        self.directory = directory
        self.clock = 100.0
        self.boot = "boot-a"
        self.log = []
        self.units = {}
        self.processes = {
            os.getpid(): {"pid": os.getpid(), "start_ticks": 10},
            99999: {"pid": 99999, "start_ticks": 20},
        }
        self.gpu = {10}
        self.open_ports = {8080}
        self.behavior = {}
        self.due = []
        for role in ("production", "stage", "probe"):
            spec = plan[role]
            current = {key: "" for key in op.PROPERTIES}
            current.update(spec["properties"])
            current.update(
                Id=spec["unit"],
                LoadState="loaded",
                ExecStart=spec["exec_start"],
                MainPID="0",
                ActiveState="inactive",
                SubState="dead",
                InvocationID="",
                Result="success",
                ExecMainStatus="0",
                Job="0",
            )
            if role == "production":
                current.update(
                    ActiveState="active",
                    SubState="running",
                    MainPID="10",
                    InvocationID="a" * 32,
                )
            self.units[spec["unit"]] = current

    def now(self):
        return self.clock

    def boot_id(self):
        return self.boot

    def process(self, pid):
        return self.processes.get(pid)

    def sleep(self, seconds):
        self.clock += seconds
        for when, callback in list(self.due):
            if self.clock >= when:
                self.due.remove((when, callback))
                callback()

    def unit(self, name, deadline):
        if self.clock >= deadline:
            raise op.Refusal("monotonic-deadline-expired")
        return dict(self.units[name])

    def owners(self, deadline):
        return set(self.gpu)

    def port_closed(self, port, deadline):
        return port not in self.open_ports

    def check_files(self, unit, deadline):
        self.log.append((self.clock, "pins", unit["unit"]))
        if self.behavior.get("bad_files") == unit["unit"]:
            raise op.Refusal("artifact-pin-mismatch")

    def check_link(self, plan):
        if self.behavior.get("wrong_release"):
            raise op.Refusal("production-release-link-changed")

    def health(self, check, deadline):
        role = "stage" if ":8081/" in check["url"] else "production"
        if self.behavior.get("unhealthy") == role:
            return None
        spec = self.plan[role]
        return {
            "status": "ok",
            "model": {
                "ready": True,
                "role": "champion",
                "model_step": spec["model_step"],
                "model_identity": spec["model_identity"],
            },
        }

    def action(self, verb, name, deadline):
        self.log.append((self.clock, verb, name))
        role = next(
            role
            for role in ("production", "stage", "probe")
            if self.plan[role]["unit"] == name
        )
        unit = self.units[name]
        if verb == "stop":
            unit["ActiveState"] = "deactivating"
            unit["SubState"] = "stop-sigterm"

            def dead():
                pid = int(unit["MainPID"])
                self.gpu.discard(pid)
                unit.update(
                    ActiveState="inactive", SubState="dead", MainPID="0", Job="0"
                )
                if role in ("production", "stage"):
                    self.open_ports.discard(8080 if role == "production" else 8081)
                if role == "stage" and self.behavior.get("foreign_on_cleanup"):
                    self.gpu.add(444)

            delay = self.behavior.get(role + "_drain", 0)
            if delay:
                self.due.append((self.clock + delay, dead))
            else:
                dead()
            return
        assert verb == "start"
        if role in ("production", "stage"):
            assert not self.gpu, "START WITH EXISTING GPU OWNER"
            if role == "production":
                assert op._dead(self.units[self.plan["stage"]["unit"]])
                assert 8081 not in self.open_ports
            elif not op._dead(self.units[self.plan["production"]["unit"]]):
                raise AssertionError("STAGE STARTED BEFORE PRODUCTION DEAD")
        pid = {"production": 11, "stage": 20, "probe": 30}[role]
        invocation = {"production": "b", "stage": "c", "probe": "d"}[role] * 32
        unit.update(
            ActiveState="active",
            SubState="running",
            MainPID=str(pid),
            InvocationID=invocation,
        )
        if role in ("production", "stage"):
            self.gpu.add(pid)
            self.open_ports.add(8080 if role == "production" else 8081)
        if role == "production" and self.behavior.get("failed_restore"):
            self.behavior["unhealthy"] = "production"
        if role == "probe":
            if self.behavior.get("probe_hang"):
                return
            if self.behavior.get("probe_old_invocation"):
                unit.update(
                    ActiveState="inactive",
                    SubState="dead",
                    MainPID="0",
                    InvocationID="e" * 32,
                )
                return

            challenge = op.read_json(self.directory / "probe-challenge.json")
            binding = {
                k: challenge[k]
                for k in (
                    "run_id",
                    "plan_sha256",
                    "boot_id",
                    "challenge",
                    "probe_unit",
                    "model_identity",
                    "model_step",
                )
            }
            begun = {
                "format": "deltrelserve.gpu-qualification-probe-started",
                "schema_version": 1,
                **binding,
                "invocation_id": invocation,
                "started_monotonic": self.clock,
            }
            op.write_json(self.directory / "probe-started.json", begun, replace=False)

            def complete():
                evidence_dir = self.directory / "probe-evidence"
                evidence_dir.mkdir()
                raw = evidence_dir / "raw.json"
                raw.write_bytes(
                    op.encoded({"legal": True, "simulations_verified": True})
                )
                raw_hash = op.digest(raw.read_bytes())
                result = {
                    "format": "deltrelserve.gpu-qualification-probe-result",
                    "schema_version": 1,
                    **binding,
                    "invocation_id": invocation,
                    "started_monotonic": begun["started_monotonic"],
                    "completed_monotonic": self.clock,
                    "checks": {
                        name: {
                            "status": "failed"
                            if self.behavior.get("probe_fail")
                            else "passed",
                            "evidence_sha256": raw_hash,
                        }
                        for name in op.REQUIRED_CHECKS
                    },
                    "evidence": [
                        {
                            "path": "probe-evidence/raw.json",
                            "sha256": raw_hash,
                            "bytes": raw.stat().st_size,
                        }
                    ],
                }
                fault = self.behavior.get("receipt_fault")
                if fault == "started_binding":
                    bad = dict(begun, challenge="0" * 64)
                    op.write_json(self.directory / "probe-started.json", bad)
                elif fault == "started_json":
                    (self.directory / "probe-started.json").write_text("not-json")
                elif fault == "challenge_file":
                    op.write_json(
                        self.directory / "probe-challenge.json",
                        dict(challenge, challenge="0" * 64),
                    )
                elif fault == "challenge":
                    result["challenge"] = "0" * 64
                elif fault == "future":
                    result["completed_monotonic"] = self.clock + 1000
                elif fault == "raw_hash":
                    raw.write_bytes(b"tampered")
                elif fault == "missing_evidence":
                    result["checks"]["deep_1"]["evidence_sha256"] = "f" * 64
                elif fault == "old_invocation":
                    result["invocation_id"] = "e" * 32
                op.write_json(
                    self.directory / "probe-result.json", result, replace=False
                )
                unit.update(
                    ActiveState="inactive",
                    SubState="dead",
                    MainPID="0",
                    Result="exit-code"
                    if self.behavior.get("probe_fail")
                    else "success",
                    ExecMainStatus="1" if self.behavior.get("probe_fail") else "0",
                )

                if self.behavior.get("clear_probe_invocation"):
                    unit["InvocationID"] = ""

            self.due.append((self.clock + 2, complete))

    def arm(self, state, plan, plan_sha, name, deadline):
        directory = state
        self.log.append((self.clock, "arm", name))
        if self.behavior.get("arm_fail"):
            raise op.Refusal("bounded-host-command-failed")
        self.units[name] = {
            "Id": name,
            "User": "root",
            "Type": "exec",
            "Restart": "no",
            "KillMode": "control-group",
            "SendSIGKILL": "yes",
            "RuntimeMaxUSec": "15min 10s",
            "TimeoutStopUSec": "5s",
            "MainPID": "99999",
            "ActiveState": "active",
            "InvocationID": "f" * 32,
        }
        state = op.read_json(directory / "state.json")
        op.write_json(
            directory / "watchdog-armed.json",
            {
                "plan_sha256": plan_sha,
                "boot_id": self.boot,
                "deadline": state["deadline"],
                "watchdog": self.process(99999),
                "invocation_id": "f" * 32,
            },
        )


@pytest.fixture
def setup(tmp_path):
    plan = {
        "format": op.FORMAT,
        "schema_version": 1,
        "approved_for_exclusive_window": True,
        "run_id": "test-572377",
        "current_link": str(tmp_path / "current"),
    }
    for role in ("production", "stage", "probe"):
        root = tmp_path / role
        root.mkdir()
        unit = (
            "deltrelserve.service"
            if role == "production"
            else f"deltrelserve-qualification-test-572377-{role}.service"
        )
        files = []
        paths = {}
        for kind in (
            "config",
            "profile",
            "pointer",
            "manifest",
            "checkpoint",
            "native",
            "entrypoint",
            "unit",
        ):
            path = root / kind
            path.write_text(kind)
            paths[kind] = str(path)
            files.append(
                {
                    "path": str(path),
                    "bytes": len(kind),
                    "sha256": op.digest(kind.encode()),
                }
            )
        props = {key: "" for key in op.IDENTITY_PROPERTIES}
        props.update(
            WorkingDirectory=str(root),
            User="root" if role == "probe" else "deltrelserve",
            Group="deltrelserve",
            FragmentPath=paths["unit"],
            Restart="on-failure" if role == "production" else "no",
            KillMode="control-group",
            SendSIGKILL="yes",
            Type="simple",
            RuntimeMaxUSec="infinity" if role == "production" else "6min",
            TimeoutStopUSec="5s" if role == "probe" else "3min 30s",
            PrivateDevices="yes" if role == "probe" else "no",
        )
        spec = {
            "unit": unit,
            "properties": props,
            "exec_start": "{ path=/safe/server ; argv[]=/safe/server",
            "release": str(root),
            "files": files,
            "env_files": [],
            **{k: v for k, v in paths.items() if k != "unit"},
        }
        if role != "probe":
            spec.update(
                model_identity="sha256-" + ("1" if role == "production" else "2") * 64,
                model_step=566428 if role == "production" else 572377,
                health=[
                    {
                        "url": f"http://127.0.0.1:{8080 if role == 'production' else 8081}/v2/health"
                    }
                ],
            )
        if role == "probe":
            spec["required_checks"] = list(op.REQUIRED_CHECKS)
        plan[role] = spec
    plan["production"].update(initial_main_pid=10, initial_invocation_id="a" * 32)
    plan["production"]["health"].append({"url": "https://example.test/v2/health"})
    # Both the deployed stage and probe belong to the new release.
    plan["probe"]["release"] = plan["stage"]["release"]
    proposal = {
        "phase_c_exclusive_gpu_qualification": {
            "total_outage_budget_seconds": 900,
            "production_stop_max_seconds": 210,
            "staging_cleanup_max_seconds": 210,
            "empty_gate_budget_seconds": 15,
            "production_restore_and_health_max_seconds": 90,
            "gpu_execution_max_seconds": 360,
        },
        "paths": {
            "rollback_release": plan["production"]["release"],
            "new_release": plan["stage"]["release"],
            "rollback_production_yaml_sha256": op.digest(b"config"),
        },
    }
    ref = tmp_path / "proposal.json"
    ref.write_bytes(op.encoded(proposal))
    plan["qualification_plan"] = {
        "path": str(ref),
        "sha256": op.digest(ref.read_bytes()),
    }
    bounds = tmp_path / "bounds.json"
    bounds.write_bytes(
        op.encoded({"plan_sha256": plan["qualification_plan"]["sha256"]})
    )
    plan["cpu_preparation_bounds"] = {
        "path": str(bounds),
        "sha256": op.digest(bounds.read_bytes()),
    }
    path = tmp_path / "execution.json"
    path.write_bytes(op.encoded(plan))
    sha = op.digest(path.read_bytes())
    directory = tmp_path / "receipt"
    host = FakeHost(plan, directory)
    return plan, path, sha, directory, host


def run(setup):
    plan, path, sha, directory, host = setup
    return op.run(plan, path, sha, directory, host)


def stops(host, role):
    return [
        (t, v, u)
        for t, v, u in host.log
        if v == "stop" and u == host.plan[role]["unit"]
    ]


def starts(host, role):
    return [
        (t, v, u)
        for t, v, u in host.log
        if v == "start" and u == host.plan[role]["unit"]
    ]


def test_normal_pass_always_restores_old_model_and_orders_gates(setup):
    plan, path, sha, directory, host = setup
    assert op.load_plan(path, sha) == plan
    result = run(setup)
    assert result["restored"] and result["qualification_passed"]
    assert host.gpu == {11} and host.open_ports == {8080}
    assert len(starts(host, "production")) == 1 and len(starts(host, "stage")) == 1
    actions = [v for _, v, _ in host.log if v in ("arm", "stop", "start")]
    assert actions == ["arm", "stop", "start", "start", "stop", "stop", "start"]
    assert result["deadline"] - result["budget_started"] == 900
    assert result["restore_by"] - result["budget_started"] == op.RESTORE_BY
    # Successful restore is idempotent and does not issue another start.
    assert op.restore(plan, sha, directory, host) == result
    assert len(starts(host, "production")) == 1


@pytest.mark.parametrize(
    "failure",
    ["arm_fail", "probe_fail", "probe_hang", "failed_restore", "foreign_on_cleanup"],
)
def test_failure_paths_are_bounded_and_never_overlap(setup, failure):
    plan, path, sha, directory, host = setup
    host.behavior[failure] = True
    with pytest.raises(op.Refusal):
        run(setup)
    state = op.read_json(directory / "state.json")
    assert host.clock <= state["deadline"]
    if failure == "arm_fail":
        assert (
            host.gpu == {10}
            and not stops(host, "production")
            and not starts(host, "production")
            and state["restored"]
        )
    elif failure == "foreign_on_cleanup":
        assert (
            host.gpu == {444}
            and not starts(host, "production")
            and not state["restored"]
        )
    elif failure == "failed_restore":
        assert (
            not state["restored"]
            and state["restore_failure"] == "old-production-health-timeout"
        )
    else:
        assert state["restored"] and len(starts(host, "production")) == 1


@pytest.mark.parametrize(
    "kind", ["foreign", "wrong_release", "wrong_unit", "wrong_pid"]
)
def test_preflight_rejects_without_arming_or_stopping(setup, kind):
    plan, path, sha, directory, host = setup
    if kind == "foreign":
        host.gpu.add(444)
    elif kind == "wrong_release":
        host.behavior["wrong_release"] = True
    elif kind == "wrong_unit":
        host.units[plan["stage"]["unit"]]["ExecStart"] = "different"
    else:
        host.units[plan["production"]["unit"]]["MainPID"] = "12"
    with pytest.raises(op.Refusal):
        run(setup)
    assert (
        not directory.exists()
        and not starts(host, "stage")
        and not stops(host, "production")
    )


def interrupted_state(setup):
    """Controller died after stage start; watchdog must operate without run/finally."""
    plan, path, sha, directory, host = setup
    directory.mkdir()
    state = {
        "format": op.FORMAT,
        "schema_version": 1,
        "plan_sha256": sha,
        "operator_sha256": op.digest(Path(op.__file__).read_bytes()),
        "boot_id": host.boot,
        "controller": {"pid": 12345, "start_ticks": 99},
        "budget_started": 100.0,
        "restore_by": 100.0 + op.RESTORE_BY,
        "deadline": 1000.0,
        "production_invocation": "a" * 32,
        "watchdog_unit": "watch.service",
        "restore_requested": False,
        "restored": False,
        "production_stop_intent": True,
        "production_stop_deadline": 100.0 + op.DRAIN + op.OBSERVATION_RESERVE,
        "stage_start_intent": True,
        "stage_invocation": "c" * 32,
        "events": [],
        "prior_invocations": {"stage": "", "probe": ""},
    }
    op.write_json(directory / "state.json", state)
    host.action("stop", plan["production"]["unit"], 1000)
    host.action("start", plan["stage"]["unit"], 1000)
    return state


def test_independent_watchdog_restores_after_controller_death(setup, monkeypatch):
    plan, path, sha, directory, host = setup
    interrupted_state(setup)
    host.units["watch.service"] = {
        "Id": "watch.service",
        "User": "root",
        "Type": "exec",
        "Restart": "no",
        "KillMode": "control-group",
        "SendSIGKILL": "yes",
        "RuntimeMaxUSec": "15min 10s",
        "TimeoutStopUSec": "5s",
        "MainPID": "99999",
        "ActiveState": "active",
        "InvocationID": "f" * 32,
    }
    monkeypatch.setattr(op.os, "getpid", lambda: 99999)
    result = op.watch(plan, sha, directory, host)
    assert result["restored"] and host.gpu == {11}
    assert op.read_json(directory / "watchdog-armed.json")["watchdog"]["pid"] == 99999


def test_watchdog_deadline_overrides_hung_live_controller_and_reserves_full_cleanup(
    setup, monkeypatch
):
    plan, path, sha, directory, host = setup
    state = interrupted_state(setup)
    host.processes[12345] = state["controller"]
    host.behavior["stage_drain"] = 209.5
    host.units["watch.service"] = {
        "Id": "watch.service",
        "User": "root",
        "Type": "exec",
        "Restart": "no",
        "KillMode": "control-group",
        "SendSIGKILL": "yes",
        "RuntimeMaxUSec": "15min 10s",
        "TimeoutStopUSec": "5s",
        "MainPID": "99999",
        "ActiveState": "active",
        "InvocationID": "f" * 32,
    }
    monkeypatch.setattr(op.os, "getpid", lambda: 99999)
    result = op.watch(plan, sha, directory, host)
    assert stops(host, "stage")[0][0] == 100 + op.RESTORE_BY
    assert result["restored"] and host.clock <= 1000
    assert starts(host, "production")[0][0] == 100 + op.RESTORE_BY + 209.5


@pytest.mark.parametrize(
    "fault",
    [
        "expired",
        "boot_changed",
        "new_invocation",
        "new_definition",
        "port_still_open",
        "foreign_owner",
    ],
)
def test_restore_refuses_unknown_ownership_and_expired_state(setup, fault):
    plan, path, sha, directory, host = setup
    interrupted_state(setup)
    if fault == "expired":
        host.clock = 1000
    elif fault == "boot_changed":
        host.boot = "other-boot"
    elif fault == "new_invocation":
        host.units[plan["stage"]["unit"]]["InvocationID"] = "e" * 32
    elif fault == "new_definition":
        host.units[plan["stage"]["unit"]]["ExecStart"] = "other"
    elif fault == "port_still_open":
        original = host.port_closed
        host.port_closed = lambda port, deadline: (
            False if port == 8081 else original(port, deadline)
        )
    else:
        host.gpu.add(444)
    with pytest.raises(op.Refusal):
        op.restore(plan, sha, directory, host)
    assert not starts(host, "production")
    if fault in ("new_invocation", "new_definition", "boot_changed", "expired"):
        assert not stops(host, "stage")


def test_recovery_after_start_request_does_not_start_with_present_owner(setup):
    plan, path, sha, directory, host = setup
    state = interrupted_state(setup)
    host.action("stop", plan["stage"]["unit"], 1000)
    host.action("start", plan["production"]["unit"], 1000)
    state["production_restore_intent"] = True
    op.write_json(directory / "state.json", state)
    result = op.restore(plan, sha, directory, host)
    assert result["restored"] and len(starts(host, "production")) == 1


def test_old_successful_probe_invocation_cannot_count_as_fresh_completion(setup):
    plan, path, sha, directory, host = setup
    host.units[plan["probe"]["unit"]]["InvocationID"] = "e" * 32
    host.behavior["probe_old_invocation"] = True
    with pytest.raises(op.Refusal, match="qualification-stage-deadline"):
        run(setup)
    state = op.read_json(directory / "state.json")
    assert state["restored"] and not state.get("qualification_passed")


@pytest.mark.parametrize(
    "change",
    [
        "unapproved",
        "stage_runtime",
        "probe_devices",
        "unbound_config",
        "wrong_reference",
        "credentials_in_url",
        "missing_tls",
    ],
)
def test_execution_plan_guards(setup, change):
    plan, path, sha, directory, host = setup
    bad = copy.deepcopy(plan)
    if change == "unapproved":
        bad["approved_for_exclusive_window"] = False
    elif change == "stage_runtime":
        bad["stage"]["properties"]["RuntimeMaxUSec"] = "infinity"
    elif change == "probe_devices":
        bad["probe"]["properties"]["PrivateDevices"] = "no"
    elif change == "unbound_config":
        bad["production"]["config"] = "/not/pinned"
    elif change == "wrong_reference":
        bad["cpu_preparation_bounds"]["sha256"] = "f" * 64
    elif change == "credentials_in_url":
        bad["production"]["health"][0]["url"] = (
            "https://user:secret@example.test/v2/health"
        )
    else:
        bad["production"]["health"] = bad["production"]["health"][:1]
    path.write_bytes(op.encoded(bad))
    with pytest.raises(op.Refusal):
        op.load_plan(path, op.digest(path.read_bytes()))


def test_real_subprocess_boundary_is_bounded_and_discards_stderr(monkeypatch):
    captured = {}

    def run(argv, **kwargs):
        captured.update(kwargs)
        raise op.subprocess.TimeoutExpired(
            argv, kwargs["timeout"], stderr="secret must never surface"
        )

    monkeypatch.setattr(op.subprocess, "run", run)
    host = op.Host()
    with pytest.raises(op.Refusal, match="bounded-host-command-failed") as result:
        host.command(["systemctl", "show"], host.now() + 1)
    assert 0 < captured["timeout"] <= 1
    assert captured["stderr"] == op.subprocess.DEVNULL
    assert "secret" not in str(result.value)


def test_health_redirects_are_refused_without_forwarding_credentials():
    with pytest.raises(op.urllib.error.URLError):
        op._NoRedirect().redirect_request(
            op.urllib.request.Request("https://example.test"),
            io.BytesIO(),
            302,
            "redirect",
            HTTPMessage(),
            "https://other.test",
        )


def test_restore_intent_before_start_rpc_still_starts_dead_old_service(setup):
    plan, path, sha, directory, host = setup
    state = interrupted_state(setup)
    host.action("stop", plan["stage"]["unit"], 1000)
    state["production_restore_intent"] = True
    op.write_json(directory / "state.json", state)
    result = op.restore(plan, sha, directory, host)
    assert (
        result["restored"] and len(starts(host, "production")) == 1 and host.gpu == {11}
    )


def test_stop_intent_before_rpc_preserves_original_healthy_process(setup):
    plan, path, sha, directory, host = setup
    interrupted_state(setup)
    host.action("stop", plan["stage"]["unit"], 1000)
    current = host.units[plan["production"]["unit"]]
    current.update(
        ActiveState="active",
        SubState="running",
        MainPID="10",
        InvocationID="a" * 32,
        Job="0",
    )
    host.gpu = {10}
    host.open_ports = {8080}
    count = len(stops(host, "production"))
    result = op.restore(plan, sha, directory, host)
    assert result["restored"] and host.gpu == {10}
    assert len(stops(host, "production")) == count and not starts(host, "production")


def test_watchdog_allows_remaining_production_drain_before_empty_gate(setup):
    plan, path, sha, directory, host = setup
    state = interrupted_state(setup)
    host.action("stop", plan["stage"]["unit"], 1000)
    state["stage_start_intent"] = False
    op.write_json(directory / "state.json", state)
    host.units[plan["production"]["unit"]].update(
        ActiveState="active", SubState="running", MainPID="10", InvocationID="a" * 32
    )
    host.gpu = {10}
    host.open_ports = {8080}
    host.behavior["production_drain"] = 209.5
    host.action("stop", plan["production"]["unit"], 1000)
    host.clock = 101
    result = op.restore(plan, sha, directory, host)
    assert result["restored"] and starts(host, "production")[0][0] == 309.5


def test_pending_inactive_stage_start_is_cancelled_before_production_restart(setup):
    plan, path, sha, directory, host = setup
    state = interrupted_state(setup)
    host.action("stop", plan["stage"]["unit"], 1000)
    state.pop("stage_invocation")
    op.write_json(directory / "state.json", state)
    stage = host.units[plan["stage"]["unit"]]
    stage.update(
        ActiveState="inactive", SubState="dead", MainPID="0", Job="123", InvocationID=""
    )

    def delayed_start():
        if stage["Job"] == "123":
            host.action("start", plan["stage"]["unit"], 1000)

    host.due.append((host.clock + 3, delayed_start))
    result = op.restore(plan, sha, directory, host)
    host.sleep(5)
    assert result["restored"] and host.gpu == {11} and stage["Job"] == "0"
    assert len(starts(host, "stage")) == 1  # only the setup's earlier stage


def test_changed_production_invocation_after_arm_is_not_stopped(setup):
    plan, path, sha, directory, host = setup
    original = host.arm

    def changed(*args):
        original(*args)
        host.units[plan["production"]["unit"]]["InvocationID"] = "e" * 32

    host.arm = changed
    with pytest.raises(op.Refusal):
        run(setup)
    assert (
        not stops(host, "production") and not starts(host, "stage") and host.gpu == {10}
    )


def test_full_210s_drain_plus_five_second_host_calls_fit_total_deadline(
    setup, monkeypatch
):
    plan, path, sha, directory, host = setup
    state = interrupted_state(setup)
    host.processes[12345] = state["controller"]
    host.behavior["stage_drain"] = 210
    state["probe_start_intent"] = True
    state["probe_invocation"] = "d" * 32
    op.write_json(directory / "state.json", state)
    host.units[plan["probe"]["unit"]].update(
        ActiveState="active", SubState="running", MainPID="30", InvocationID="d" * 32
    )
    host.units["watch.service"] = {
        "Id": "watch.service",
        "User": "root",
        "Type": "exec",
        "Restart": "no",
        "KillMode": "control-group",
        "SendSIGKILL": "yes",
        "RuntimeMaxUSec": "15min 10s",
        "TimeoutStopUSec": "5s",
        "MainPID": "99999",
        "ActiveState": "active",
        "InvocationID": "f" * 32,
    }
    original_unit, original_action = host.unit, host.action

    def slow_unit(name, deadline):
        host.sleep(5)
        return original_unit(name, deadline)

    def slow_action(verb, name, deadline):
        host.sleep(5)
        return original_action(verb, name, deadline)

    original_owners, original_health = host.owners, host.health

    def slow_owners(deadline):
        host.sleep(5)
        return original_owners(deadline)

    def slow_health(check, deadline):
        host.sleep(3)
        return original_health(check, deadline)

    host.unit = slow_unit
    host.action = slow_action
    host.owners = slow_owners
    host.health = slow_health
    monkeypatch.setattr(op.os, "getpid", lambda: 99999)
    result = op.watch(plan, sha, directory, host)
    assert result["restored"] and host.clock <= 1000
    assert stops(host, "stage")[-1][0] <= 685


def test_durable_probe_receipt_survives_cleared_inactive_invocation_id(setup):
    *_, host = setup
    host.behavior["clear_probe_invocation"] = True
    result = run(setup)
    assert (
        result["qualification_passed"]
        and result["restored"]
        and result["probe_result_sha256"]
    )


@pytest.mark.parametrize(
    "fault",
    [
        "challenge",
        "future",
        "raw_hash",
        "missing_evidence",
        "old_invocation",
        "started_binding",
        "started_json",
        "challenge_file",
    ],
)
def test_probe_receipt_replay_tamper_and_unbound_evidence_never_pass(setup, fault):
    plan, path, sha, directory, host = setup
    host.behavior["receipt_fault"] = fault
    with pytest.raises((op.Refusal, ValueError)):
        run(setup)
    state = op.read_json(directory / "state.json")
    assert state["restored"] and not state.get("qualification_passed")


def test_http_bound_is_total_elapsed_time_not_per_receive():
    with pytest.raises(TimeoutError, match="bounded-health-timeout"):
        with op._http_wall_bound(0.01):
            op.time.sleep(0.05)


def test_only_connection_refused_proves_closed_port(monkeypatch):
    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def settimeout(self, value):
            pass

        def connect_ex(self, address):
            return op.errno.ETIMEDOUT

    monkeypatch.setattr(op.socket, "socket", Socket)
    host = op.Host()
    assert not host.port_closed(8081, host.now() + 1)


def test_full_production_drain_after_stop_dispatch_does_not_orphan_service(setup):
    plan, path, sha, directory, host = setup
    state = interrupted_state(setup)
    host.action("stop", plan["stage"]["unit"], 1000)
    state["stage_start_intent"] = False
    state.pop("stage_invocation")
    op.write_json(directory / "state.json", state)
    host.units[plan["production"]["unit"]].update(
        ActiveState="active", SubState="running", MainPID="10", InvocationID="a" * 32
    )
    host.gpu = {10}
    host.open_ports = {8080}
    host.behavior["production_drain"] = 210
    original_action, original_unit = host.action, host.unit

    def slow_action(verb, name, deadline):
        host.sleep(5)
        return original_action(verb, name, deadline)

    def slow_unit(name, deadline):
        host.sleep(5)
        return original_unit(name, deadline)

    host.action = slow_action
    host.unit = slow_unit
    host.action("stop", plan["production"]["unit"], 1000)
    # Original controller dies while the request's full210s drain is underway.
    result = op.restore(plan, sha, directory, host)
    assert result["restored"] and starts(host, "production")[0][0] >= 315
    assert host.clock <= 1000


def test_protected_credentials_cannot_be_sent_to_external_https(setup):
    plan, path, sha, directory, host = setup
    protected = "/etc/deltrelserve/service.env"
    plan["production"]["env_files"] = [protected]
    plan["production"]["health"][1].update(
        env_file=protected, env_name="DELTRELSERVE_BEARER_TOKEN"
    )
    path.write_bytes(op.encoded(plan))
    with pytest.raises(op.Refusal, match="credentials-require-exact-role-loopback"):
        op.load_plan(path, op.digest(path.read_bytes()))
