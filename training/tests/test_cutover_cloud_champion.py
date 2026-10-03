"""CPU-only transaction tests; no real systemd, GPU or HTTP calls."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

from scripts import cutover_cloud_champion as op


class FakeHost(op.Host):
    def __init__(self, p):
        self.p = p
        self.clock = 100.0
        self.boot = "boot-one"
        self.log = []
        self.pair_value = op.desired(p, "old")
        self.gpu = {10}
        self.ports = {8080}
        self.role = "old"
        self.units = {}
        self.flags = {}
        self.fault = {}
        self.due = []
        self.starts = 0
        self.processes = {
            os.getpid(): {"pid": os.getpid(), "start_ticks": 10},
            99: {"pid": 99, "start_ticks": 11},
        }
        for role in ("old", "guard", "acceptance"):
            spec = p[role]
            unit = {k: "" for k in op.q.PROPERTIES}
            unit.update(spec["properties"])
            unit.update(
                Id=spec["unit"],
                LoadState="loaded",
                ExecStart=spec["exec_start"],
                ActiveState="inactive",
                SubState="dead",
                MainPID="0",
                InvocationID="",
                Result="success",
                ExecMainStatus="0",
                Job="",
            )
            self.flags[spec["unit"]] = "enabled" if role == "old" else "disabled"
            if role == "old":
                unit.update(
                    ActiveState="active",
                    SubState="running",
                    MainPID="10",
                    InvocationID="a" * 32,
                )
            self.units[spec["unit"]] = unit

    def now(self):
        return self.clock

    def boot_id(self):
        return self.boot

    def process(self, pid):
        return self.processes.get(pid)

    def sleep(self, n):
        self.clock += n
        for at, fn in list(self.due):
            if self.clock >= at:
                self.due.remove((at, fn))
                fn()

    def unit(self, name, deadline):
        if self.clock >= deadline:
            raise op.Refusal("deadline")
        return dict(self.units[name])

    def enablement(self, unit, deadline):
        return self.flags[unit]

    def enabled(self, verb, unit, directory, deadline):
        name = unit
        self.log.append((verb, name))
        self.flags[name] = "enabled" if verb == "enable" else "disabled"
        if verb == "disable" and name == self.p["old"]["unit"]:
            assert self.flags[self.p["guard"]["unit"]] == "enabled"
        if (
            verb == "disable"
            and name == self.p["guard"]["unit"]
            and self.fault.pop("retirement", False)
        ):
            raise OSError("retirement interruption")

    def confirm_enablement(self, unit, wanted, directory, deadline):
        assert self.flags[unit] == wanted

    def check_files(self, spec, deadline):
        if self.fault.get("files") == spec["release"]:
            raise op.Refusal("file-pin-changed")

    def pair(self, p):
        return self.pair_value

    def replace_config(self, p, data):
        assert self.flags[p["old"]["unit"]] == "disabled"
        assert op.q._dead(self.units[p["old"]["unit"]]) and not self.gpu
        self.log.append(("config", op.q.digest(data)))
        self.pair_value = (op.q.digest(data), self.pair_value[1])
        if data == op.pin_bytes(p["candidate_config"]) and self.fault.pop(
            "config_write", False
        ):
            raise OSError("after config write")

    def replace_link(self, p, target):
        assert op.q._dead(self.units[p["old"]["unit"]]) and not self.gpu
        self.log.append(("link", target))
        self.pair_value = (self.pair_value[0], target)
        if target == p["new"]["release"] and self.fault.pop("link_write", False):
            raise OSError("after link write")

    def owners(self, deadline):
        return set(self.gpu)

    def port_closed(self, port, deadline):
        return port not in self.ports

    def health(self, check, deadline):
        if self.fault.get("unhealthy") == self.role:
            return None
        return {
            "status": "ok",
            "model": {
                "ready": True,
                "role": "champion",
                "model_identity": self.p[self.role]["model_identity"],
                "model_step": self.p[self.role]["model_step"],
            },
        }

    def action(self, verb, name, deadline):
        self.log.append((verb, name))
        unit = self.units[name]
        if verb == "stop":
            if unit["ActiveState"] == "deactivating":
                return
            if name == self.p["old"]["unit"] and self.fault.get("stop_hang"):
                return

            def dead():
                self.gpu.discard(int(unit["MainPID"]))
                unit.update(
                    ActiveState="inactive", SubState="dead", MainPID="0", Job=""
                )
                if name == self.p["old"]["unit"]:
                    self.ports.discard(8080)

            delay = self.fault.get("drain", 0) if name == self.p["old"]["unit"] else 0
            if delay:
                unit.update(ActiveState="deactivating", SubState="stop-sigterm")
                self.due.append((self.clock + delay, dead))
            else:
                dead()
            return
        assert verb == "start"
        if name == self.p["guard"]["unit"]:
            unit.update(
                ActiveState="active",
                SubState="running",
                MainPID="99",
                InvocationID="f" * 32,
            )
            if self.fault.get("arm"):
                return
            state = op.q.read_json(Path(self.p["state_directory"]) / "state.json")
            op.q.write_json(
                Path(self.p["state_directory"]) / "guard-armed.json",
                {
                    "plan_sha256": state["plan_sha256"],
                    "boot_id": self.boot,
                    "invocation_id": "f" * 32,
                    "process": self.process(99),
                },
            )
        elif name == self.p["old"]["unit"]:
            assert not self.gpu, "GPU OVERLAP"
            assert self.pair_value in (
                op.desired(self.p, "old"),
                op.desired(self.p, "new"),
            ), "MIXED PAIR START"
            self.role = "old" if self.pair_value == op.desired(self.p, "old") else "new"
            self.starts += 1
            unit.update(
                ActiveState="active",
                SubState="running",
                MainPID=str(20 + self.starts),
                InvocationID=f"{self.starts:032x}",
                Job="",
            )
            self.gpu.add(20 + self.starts)
            self.ports.add(8080)
        else:
            assert name == self.p["acceptance"]["unit"]
            unit.update(
                ActiveState="active",
                SubState="running",
                MainPID="30",
                InvocationID="d" * 32,
            )
            if self.fault.get("acceptance_hang"):
                return

            def complete():
                result_receipt(
                    self.p,
                    Path(self.p["state_directory"]),
                    self.clock,
                    passed=not self.fault.get("acceptance_fail"),
                )
                unit.update(
                    ActiveState="inactive",
                    SubState="dead",
                    MainPID="0",
                    InvocationID="",
                    Job="",
                )

            self.due.append((self.clock + 2, complete))


def result_receipt(p, directory, completed, passed=True):
    c = op.q.read_json(directory / "acceptance-challenge.json")
    binding = {k: c[k] for k in op.BINDINGS}
    base = {
        "schema_version": 1,
        **binding,
        "invocation_id": "d" * 32,
        "started_monotonic": c["issued_monotonic"],
    }
    raw = directory / "acceptance-evidence/standard_1.json"
    raw.parent.mkdir(exist_ok=True)
    raw.write_bytes(b'{"verified":true}')
    evidence = {
        "path": "acceptance-evidence/standard_1.json",
        "sha256": op.q.digest(raw.read_bytes()),
        "bytes": raw.stat().st_size,
    }
    op.q.write_json(
        directory / "acceptance-started.json",
        {"format": "deltrelserve.cloud-cutover-acceptance-started", **base},
        replace=False,
    )
    op.q.write_json(
        directory / "acceptance-result.json",
        {
            "format": "deltrelserve.cloud-cutover-acceptance",
            **base,
            "completed_monotonic": completed,
            "checks": {
                "standard_1": {
                    "status": "passed" if passed else "failed",
                    "evidence_sha256": evidence["sha256"],
                }
            },
            "evidence": [evidence],
        },
        replace=False,
    )


@pytest.fixture
def setup(tmp_path):
    def pin(name, data):
        path = tmp_path / name
        path.write_bytes(data)
        return {"path": str(path), "sha256": op.q.digest(data), "bytes": len(data)}

    old = pin("old.yaml", b"old config\n")
    gpu = pin("qualified.yaml", b"schema_version: 2\nport: 8081\nmodel: new\n")
    new = pin("candidate.yaml", op.production_yaml(op.pin_bytes(gpu)))
    p = {
        "run_id": "test572377",
        "state_directory": str(tmp_path / "state"),
        "production_config": str(tmp_path / "production.yaml"),
        "current_link": str(tmp_path / "current"),
        "old_link_target": "releases/old",
        "original_enablement": "enabled",
        "initial_pid": 10,
        "initial_invocation": "a" * 32,
        "config_metadata": {"uid": os.getuid(), "gid": os.getgid(), "mode": 0o644},
        "old_config": old,
        "candidate_config": new,
        "qualified_gpu_config": gpu,
        "enablement_directory": str(tmp_path / "multi-user.target.wants"),
    }
    for role in ("old", "new", "guard", "acceptance"):
        props = {k: "" for k in op.q.IDENTITY_PROPERTIES}
        props.update(
            User="deltrelserve" if role in ("old", "new") else "root",
            Group="root",
            Type="simple" if role in ("old", "new") else "exec",
            Restart="on-failure" if role in ("old", "new") else "no",
            KillMode="control-group",
            SendSIGKILL="yes",
            TimeoutStopUSec="3min 30s" if role in ("old", "new") else "5s",
            RuntimeMaxUSec="15min 10s"
            if role == "guard"
            else "3min 20s"
            if role == "acceptance"
            else "infinity",
            PrivateDevices="yes" if role == "acceptance" else "no",
        )
        unit = (
            "deltrelserve.service"
            if role in ("old", "new")
            else f"deltrelserve-cutover-test572377-{role}.service"
        )
        spec = {
            "unit": unit,
            "properties": props,
            "exec_start": "fixed " + unit,
            "env_files": [],
            "release": str(tmp_path / "releases" / role),
            "files": [],
        }
        if role in ("old", "new"):
            spec.update(
                model_step=566428 if role == "old" else 572377,
                model_identity="sha256-" + ("1" if role == "old" else "2") * 64,
                health=[
                    {"url": "http://127.0.0.1:8080/v2/health"},
                    {"url": "https://deltrel.com/v2/health"},
                ],
            )
        p[role] = spec
    p["new"]["properties"] = copy.deepcopy(p["old"]["properties"])
    p["new"]["exec_start"] = p["old"]["exec_start"]
    host = FakeHost(p)
    return p, tmp_path / "plan.json", "a" * 64, Path(p["state_directory"]), host


def run(setup):
    return op.execute(*setup)


def state_at(setup, pair="old", role="old", boot="boot-one", commit=False):
    p, path, checksum, directory, host = setup
    directory.mkdir(mode=0o700)
    path.write_bytes(b"prepared plan")
    state = {
        "format": op.FORMAT,
        "schema_version": 1,
        "plan_sha256": checksum,
        "plan_path": str(path.resolve()),
        "operator_sha256": op.q.digest(Path(op.__file__).read_bytes()),
        "host_helper_sha256": op.q.digest(Path(op.q.__file__).read_bytes()),
        "original_boot_id": boot,
        "active_boot_id": boot,
        "controller": {"pid": 12345, "start_ticks": 1},
        "budget_started": 100.0,
        "rollback_by": 640.0,
        "original_rollback_by": 640.0,
        "deadline": 1000.0,
        "rollback_requested": False,
        "finished": False,
        "service_invocations": ["a" * 32],
        "acceptance_prior_invocation": "",
        "events": [],
        "recovery_boots": [],
    }
    host.flags[p["guard"]["unit"]] = "enabled"
    host.flags[p["old"]["unit"]] = "disabled"
    host.pair_value = op.desired(p, pair)
    host.role = role
    if pair == "new":
        host.units[p["old"]["unit"]]["InvocationID"] = "c" * 32
        state["service_invocations"].append("c" * 32)
    op.q.write_json(directory / "state.json", state)
    if commit:
        c = {
            "format": "deltrelserve.cloud-cutover-challenge",
            "schema_version": 1,
            "run_id": p["run_id"],
            "plan_sha256": checksum,
            "boot_id": boot,
            "challenge": "f" * 64,
            "issued_monotonic": 101.0,
            "deadline_monotonic": 200.0,
            "acceptance_unit": p["acceptance"]["unit"],
            "model_identity": p["new"]["model_identity"],
            "model_step": 572377,
        }
        op.q.write_json(directory / "acceptance-challenge.json", c)
        state["acceptance_challenge_sha256"] = op.q.digest(
            (directory / "acceptance-challenge.json").read_bytes()
        )
        result_receipt(p, directory, 110.0)
        op.q.write_json(
            directory / "accepted.json",
            {
                "format": op.FORMAT + ".accepted",
                "schema_version": 1,
                "plan_sha256": checksum,
                "pair": list(op.desired(p, "new")),
                "result_sha256": op.q.digest(
                    (directory / "acceptance-result.json").read_bytes()
                ),
                "authenticated_and_public_health": True,
                "service_invocation": "c" * 32,
                "boot_id": boot,
                "accepted_monotonic": 111.0,
            },
        )
        host.clock = 112.0
        op.q.write_json(directory / "state.json", state)
    return state


def test_success_switches_only_pair_and_retires_guard_after_acceptance(setup):
    p, _, _, directory, host = setup
    result = run(setup)
    assert result["finished"] and result["outcome"] == "committed-new"
    assert (
        host.pair_value == op.desired(p, "new")
        and host.flags[p["old"]["unit"]] == "enabled"
        and host.flags[p["guard"]["unit"]] == "disabled"
    )
    actions = host.log
    assert (
        actions.index(("enable", p["guard"]["unit"]))
        < actions.index(("disable", p["old"]["unit"]))
        < next(i for i, row in enumerate(actions) if row[0] == "config")
    )
    assert (directory / "accepted.json").exists()
    assert result["events"][-1]["event"] == "persistent-guard-retired"


@pytest.mark.parametrize(
    "fault", ["arm", "config_write", "link_write", "acceptance_fail", "acceptance_hang"]
)
def test_failures_restore_exact_old_pair_and_original_enablement(setup, fault):
    p, _, _, directory, host = setup
    host.fault[fault] = True
    with pytest.raises((op.Refusal, OSError)):
        run(setup)
    state = op.q.read_json(directory / "state.json")
    assert state["finished"] and state["outcome"] == "rolled-back-old"
    assert (
        host.pair_value == op.desired(p, "old")
        and host.flags[p["old"]["unit"]] == "enabled"
    )
    assert host.flags[p["guard"]["unit"]] == "disabled" and host.clock <= 1000
    if fault == "arm":
        assert not any(
            verb == "stop" and name == p["old"]["unit"] for verb, name in host.log
        )


@pytest.mark.parametrize(
    "config_new,link_new", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("reboot", [False, True])
def test_crash_recovers_all_recognized_partial_pairs_before_restart(
    setup, config_new, link_new, reboot
):
    p, _, checksum, directory, host = setup
    state_at(setup)
    host.action("stop", p["old"]["unit"], 1000)
    host.pair_value = (
        op.desired(p, "new" if config_new else "old")[0],
        op.desired(p, "new" if link_new else "old")[1],
    )
    if reboot:
        host.boot = "boot-two"
        host.clock = 1.0
    result = op.recover(p, checksum, directory, host)
    assert result["outcome"] == "rolled-back-old" and host.pair_value == op.desired(
        p, "old"
    )
    assert (
        host.flags[p["old"]["unit"]] == "enabled"
        and host.flags[p["guard"]["unit"]] == "disabled"
    )
    if reboot:
        assert result["recovery_boots"] == [
            {"boot_id": "boot-two", "started_monotonic": 1.0, "deadline": 901.0}
        ]


@pytest.mark.parametrize("reboot", [False, True])
def test_commit_before_enablement_or_retirement_finalizes_new(setup, reboot):
    p, _, checksum, directory, host = setup
    state_at(setup, pair="new", role="new", commit=True)
    if reboot:
        host.action("stop", p["old"]["unit"], 1000)
        host.boot = "boot-two"
        host.clock = 1.0
    result = op.recover(p, checksum, directory, host)
    assert result["outcome"] == "committed-new" and host.pair_value == op.desired(
        p, "new"
    )
    assert (
        host.flags[p["old"]["unit"]] == "enabled"
        and host.flags[p["guard"]["unit"]] == "disabled"
    )


def test_invalid_commit_proof_does_not_prevent_owned_rollback(setup):
    p, _, checksum, directory, host = setup
    state_at(setup, pair="new", role="new", commit=True)
    (directory / "acceptance-result.json").write_text("invalid")
    result = op.recover(p, checksum, directory, host)
    assert (
        result["outcome"] == "rolled-back-old" and result["invalid_acceptance_commit"]
    )


@pytest.mark.parametrize("kind", ["config", "link", "foreign_gpu", "changed_unit"])
def test_unknown_state_never_overwritten_or_overlapped(setup, kind):
    p, _, checksum, directory, host = setup
    state_at(setup, pair="new", role="new")
    if kind == "config":
        host.pair_value = ("unknown", host.pair_value[1])
    elif kind == "link":
        host.pair_value = (host.pair_value[0], "/someone/elses/release")
    elif kind == "foreign_gpu":
        host.gpu.add(444)
    else:
        host.units[p["old"]["unit"]]["ExecStart"] = "different"
    before = host.pair_value
    with pytest.raises(op.Refusal):
        op.recover(p, checksum, directory, host)
    assert host.pair_value == before and not any(
        verb in ("config", "link", "start") for verb, _ in host.log
    )
    if kind == "foreign_gpu":
        assert host.gpu == {444}


def test_acceptance_queued_start_cancelled_even_while_inactive(setup):
    p, _, checksum, directory, host = setup
    state = state_at(setup)
    state["acceptance_start_intent"] = True
    op.q.write_json(directory / "state.json", state)
    host.units[p["acceptance"]["unit"]]["Job"] = "123"
    result = op.recover(p, checksum, directory, host)
    assert result["finished"] and ("stop", p["acceptance"]["unit"]) in host.log
    assert host.units[p["acceptance"]["unit"]]["Job"] == ""


def test_retirement_interruption_is_idempotently_completed(setup):
    p, _, checksum, directory, host = setup
    state_at(setup, pair="new", role="new", commit=True)
    host.fault["retirement"] = True
    with pytest.raises(OSError):
        op.recover(p, checksum, directory, host)
    assert not op.q.read_json(directory / "state.json")["finished"]
    result = op.recover(p, checksum, directory, host)
    assert result["finished"] and result["outcome"] == "committed-new"


def test_completed_transaction_never_regains_authority_after_reboot(setup):
    p, _, checksum, directory, host = setup
    run(setup)
    host.boot = "later-boot"
    host.clock = 2.0
    before = list(host.log)
    result = op.recover(p, checksum, directory, host)
    assert result["finished"] and host.log == before


def test_real_pair_writes_preserve_yaml_permissions_and_relative_rollback_link(
    setup, tmp_path
):
    p, *_ = setup
    config = Path(p["production_config"])
    config.write_bytes(op.pin_bytes(p["old_config"]))
    config.chmod(0o644)
    (tmp_path / "releases/old").mkdir(parents=True)
    (tmp_path / "releases/new").mkdir()
    link = Path(p["current_link"])
    link.symlink_to("releases/old")
    host = op.Host()
    host.replace_config(p, op.pin_bytes(p["candidate_config"]))
    host.replace_link(p, p["new"]["release"])
    assert (
        host.pair(p) == op.desired(p, "new") and config.stat().st_mode & 0o777 == 0o644
    )
    host.replace_config(p, op.pin_bytes(p["old_config"]))
    host.replace_link(p, p["old_link_target"])
    assert (
        config.read_bytes() == op.pin_bytes(p["old_config"])
        and os.readlink(link) == "releases/old"
    )


def test_port_derivation_is_byte_exact_and_refuses_ambiguous_input():
    original = b"schema_version: 2\nport: 8081\nsecurity: unchanged\n"
    assert op.production_yaml(original) == original.replace(
        b"port: 8081", b"port: 8080"
    )
    for bad in (
        b"port: 8081\n",
        original + original,
        original.replace(b"8081", b"8080"),
    ):
        with pytest.raises(op.Refusal):
            op.production_yaml(bad)


def test_guard_bootstrap_uses_prior_authorized_hash_not_current_plan_bytes(setup):
    p, path, checksum, directory, host = setup
    state_at(setup)
    path.write_bytes(b"arbitrary replacement must not become authorized")
    assert op.guard_checksum(directory, path) == checksum
    with pytest.raises(op.Refusal, match="cutover-plan-hash-mismatch"):
        op.load_plan(path, op.guard_checksum(directory, path))


@pytest.mark.parametrize(
    "change", ["directory_mode", "file_mode", "symlink", "plan_path", "source_hash"]
)
def test_guard_bootstrap_refuses_unprotected_or_differently_bound_state(setup, change):
    p, path, checksum, directory, host = setup
    state = state_at(setup)
    if change == "directory_mode":
        directory.chmod(0o755)
    elif change == "file_mode":
        (directory / "state.json").chmod(0o644)
    elif change == "symlink":
        file = directory / "state.json"
        other = directory / "other.json"
        file.rename(other)
        file.symlink_to(other)
    elif change == "plan_path":
        state["plan_path"] = str(path.with_name("other-plan.json"))
        op.q.write_json(directory / "state.json", state)
    else:
        state["operator_sha256"] = "f" * 64
        op.q.write_json(directory / "state.json", state)
    with pytest.raises(op.Refusal):
        op.guard_checksum(directory, path)


def test_guard_without_controller_finally_restores_owned_new_service(
    setup, monkeypatch
):
    p, path, checksum, directory, host = setup
    state_at(setup, pair="new", role="new")
    host.units[p["guard"]["unit"]].update(
        ActiveState="active", SubState="running", MainPID="99", InvocationID="f" * 32
    )
    monkeypatch.setattr(op.os, "getpid", lambda: 99)
    result = op.guard(p, checksum, directory, host)
    assert result["outcome"] == "rolled-back-old" and result["finished"]


def test_guard_deadline_with_210_second_drain_and_control_latency_stays_bounded(
    setup, monkeypatch
):
    p, path, checksum, directory, host = setup
    state = state_at(setup, pair="new", role="new")
    host.processes[12345] = state["controller"]
    host.fault["drain"] = 210
    host.units[p["guard"]["unit"]].update(
        ActiveState="active", SubState="running", MainPID="99", InvocationID="f" * 32
    )
    original_unit, original_action, original_owners, original_health = (
        host.unit,
        host.action,
        host.owners,
        host.health,
    )

    def slow_unit(name, deadline):
        host.sleep(5)
        return original_unit(name, deadline)

    def slow_action(verb, name, deadline):
        host.sleep(5)
        return original_action(verb, name, deadline)

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
    monkeypatch.setattr(op.os, "getpid", lambda: 99)
    result = op.guard(p, checksum, directory, host)
    assert result["outcome"] == "rolled-back-old" and host.clock <= 1000


def test_boot_guard_preserves_original_healthy_autostart_when_disable_never_happened(
    setup,
):
    p, path, checksum, directory, host = setup
    state_at(setup)
    host.boot = "boot-two"
    host.clock = 1.0
    host.flags[p["old"]["unit"]] = "enabled"
    host.units[p["old"]["unit"]]["InvocationID"] = "e" * 32
    before = list(host.log)
    result = op.recover(p, checksum, directory, host)
    assert result["outcome"] == "rolled-back-old" and not any(
        row[0] == "stop" for row in host.log[len(before) :]
    )


def test_acceptance_static_unit_is_never_enabled(setup):
    p, _, _, _, host = setup
    host.flags[p["acceptance"]["unit"]] = "static"
    assert run(setup)["outcome"] == "committed-new"
    assert ("enable", p["acceptance"]["unit"]) not in host.log


def test_boot_enablement_requires_actual_multiuser_link_to_pinned_fragment(
    tmp_path, monkeypatch
):
    directory = tmp_path / "multi-user.target.wants"
    directory.mkdir()
    fragment = tmp_path / "guard.service"
    fragment.write_text("unit")
    host = op.Host()
    monkeypatch.setattr(host, "enablement", lambda unit, deadline: "enabled")
    monkeypatch.setattr(
        host, "unit", lambda unit, deadline: {"FragmentPath": str(fragment)}
    )
    with pytest.raises(op.Refusal, match="boot-enablement-link-not-confirmed"):
        host.confirm_enablement(
            "guard.service", "enabled", str(directory), host.now() + 1
        )
    link = directory / "guard.service"
    link.symlink_to(fragment)
    host.confirm_enablement("guard.service", "enabled", str(directory), host.now() + 1)
    link.unlink()
    other = tmp_path / "other.service"
    other.write_text("other")
    link.symlink_to(other)
    with pytest.raises(op.Refusal, match="boot-enablement-link-not-confirmed"):
        host.confirm_enablement(
            "guard.service", "enabled", str(directory), host.now() + 1
        )


def loader_plan(setup, tmp_path):
    p, path, _, _, _ = setup
    p = copy.deepcopy(p)

    def pin(name, value):
        file = tmp_path / name
        data = op.q.encoded(value) if not isinstance(value, bytes) else value
        file.write_bytes(data)
        return {"path": str(file), "sha256": op.q.digest(data), "bytes": len(data)}

    unit = pin("prod-unit", b"production unit")
    p["old"]["properties"]["FragmentPath"] = unit["path"]
    p["old"]["config"] = p["production_config"]
    p["old"]["files"] = [unit, {**p["old_config"], "path": p["production_config"]}]
    p["old"]["env_files"] = ["/etc/deltrelserve/service.env"]
    p["old"]["health"][0].update(
        env_file="/etc/deltrelserve/service.env", env_name="DELTRELSERVE_BEARER_TOKEN"
    )
    stage = copy.deepcopy(p["new"])
    stage["config"] = p["qualified_gpu_config"]["path"]
    stage["files"] = [p["qualified_gpu_config"]]
    execution = {
        "production": p["old"],
        "stage": stage,
        "current_link": p["current_link"],
    }
    qp = pin("qa-plan.json", execution)
    result = {
        "plan_sha256": qp["sha256"],
        "boot_id": "qualified-boot",
        "model_identity": stage["model_identity"],
        "model_step": 572377,
        "checks": {name: {"status": "passed"} for name in op.q.REQUIRED_CHECKS},
    }
    qr = pin("qa-result.json", result)
    qs = pin(
        "qa-state.json",
        {
            "qualification_passed": True,
            "restored": True,
            "plan_sha256": qp["sha256"],
            "boot_id": "qualified-boot",
            "probe_result_sha256": qr["sha256"],
        },
    )
    qv = pin(
        "qa-verification.json",
        {
            "status": "passed-exclusive-gpu-qualification-and-independent-old-production-restoration",
            "state_sha256": qs["sha256"],
            "probe_result_sha256": qr["sha256"],
            "authenticated_loopback_and_public_tls_healthy": True,
            "stage_probe_inactive_no_jobs": True,
            "port8081_connection_refused": True,
            "watchdog_inactive": True,
        },
    )
    p.update(
        format=op.FORMAT,
        schema_version=1,
        approved_for_cutover=True,
        qualification={"plan": qp, "state": qs, "result": qr, "verification": qv},
        enablement_directory="/etc/systemd/system/multi-user.target.wants",
    )
    for role in ("guard", "acceptance"):
        file = pin(role + "-unit", role.encode())
        p[role]["properties"]["FragmentPath"] = file["path"]
        p[role]["files"] = [file]
    p["dependencies"] = []
    for file in (Path(op.__file__), Path(op.q.__file__)):
        p["dependencies"].append(
            {
                "path": str(file.resolve()),
                "sha256": op.q.digest(file.read_bytes()),
                "bytes": file.stat().st_size,
            }
        )
    path.write_bytes(op.q.encoded(p))
    return p, path, op.q.digest(path.read_bytes())


def test_plan_binds_closed_qualification_exact_pair_and_executed_dependencies(
    setup, tmp_path
):
    original, path, sha = loader_plan(setup, tmp_path)
    parsed = op.load_plan(path, sha)
    assert parsed["new"]["properties"] == parsed["old"]["properties"]
    assert parsed["new"]["health"] == parsed["old"]["health"]
    assert parsed["new"]["env_files"] == parsed["old"]["env_files"]
    assert parsed["new"]["model_identity"] == original["new"]["model_identity"]


@pytest.mark.parametrize(
    "fault",
    [
        "old_link",
        "config_extra_change",
        "unapproved",
        "missing_dependency",
        "guard_oneshot",
        "missing_unit_pin",
        "unused_boot_target",
        "unclosed_qualification",
    ],
)
def test_plan_refuses_invalid_or_unqualified_cutover(setup, tmp_path, fault):
    p, path, sha = loader_plan(setup, tmp_path)
    if fault == "old_link":
        p["old_link_target"] = "releases/unknown"
    elif fault == "config_extra_change":
        file = Path(p["candidate_config"]["path"])
        file.write_bytes(file.read_bytes() + b"other: changed\n")
        p["candidate_config"].update(
            sha256=op.q.digest(file.read_bytes()), bytes=file.stat().st_size
        )
    elif fault == "unapproved":
        p["approved_for_cutover"] = False
    elif fault == "missing_dependency":
        p["dependencies"] = p["dependencies"][:1]
    elif fault == "guard_oneshot":
        p["guard"]["properties"]["Type"] = "oneshot"
    elif fault == "missing_unit_pin":
        p["guard"]["files"] = []
    elif fault == "unused_boot_target":
        p["enablement_directory"] = "/etc/systemd/system/unused.target.wants"
    else:
        row = p["qualification"]["verification"]
        file = Path(row["path"])
        data = op.q.read_json(file)
        data["watchdog_inactive"] = False
        file.write_bytes(op.q.encoded(data))
        row.update(sha256=op.q.digest(file.read_bytes()), bytes=file.stat().st_size)
    path.write_bytes(op.q.encoded(p))
    with pytest.raises(op.Refusal):
        op.load_plan(path, op.q.digest(path.read_bytes()))
