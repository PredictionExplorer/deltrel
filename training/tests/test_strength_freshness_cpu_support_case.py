"""Dummy phase engine with fake Linux facts and real tiny journal files.

These tests do not invoke the protected target context factory, acquire a host
lease, resolve /proc or run systemctl. The unchanged support helper is executed.
"""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import strength_freshness_cpu_support_case as case
from scripts import strength_freshness_units as support
from test_strength_freshness_units import Host


class PlannedCrash(BaseException):
    pass


@pytest.fixture
def engine(tmp_path, monkeypatch):
    host: Any = Host(tmp_path, monkeypatch)
    # Unit files sort before /tmp env files, just as real /etc unit files sort
    # before the plan's /run/<nonce>/env files. No target path is accessed.
    for name, spec in host.manifest["units"].items():
        for side in ("before", "after"):
            env = spec[side]["environment_files"][0]
            old = env["path"]
            env["path"] = str(tmp_path / "env" / (name + ".env"))
            data = ("CPUQUAL_PHASE=" + side + "\n").encode()
            Path(env["source_path"]).write_bytes(data)
            env.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
            if old in host.io.files:
                host.io.files.pop(old)
                host.io.files[env["path"]] = b"CPUQUAL_PHASE=before\n"
                host.io.modes[env["path"]] = host.io.modes.pop(old)
    for side in ("before", "after"):
        for name, expected in host.plan["support_transition"][side].items():
            expected["environment_sha256"] = support.core.digest(
                host.manifest["units"][name][side]["environment_files"]
            )
    atomic = host.io.atomic

    def write(path, data, *, overwrite=False, mode=0o600):
        if str(path) in host.io.files:
            if not overwrite:
                raise FileExistsError(path)
            host.io.files[str(path)] = data
            host.io.modes[str(path)] = mode
            host.io.events.append(("write", str(path)))
        else:
            if Path(path).exists() and not overwrite:
                raise FileExistsError(path)
            atomic(path, data, overwrite=overwrite, mode=mode)

    host.io.atomic = write
    host.io.anchor = SimpleNamespace(deadline=lambda _: 100.0)
    host.io.exists = lambda path: str(path) in host.io.files or Path(path).exists()
    host.io.remembered = []
    host.io.remember_prior_process = lambda name, process: host.io.remembered.append(
        (name, process)
    )
    host.io.process = lambda pid, deadline=None: host.processes.get(pid)
    host.status["guard.service"]["entered_monotonic"] = 9.0
    state = json.loads((host.state / "state.json").read_bytes())
    # The existing helper fixture's plan changed only by relocating tiny env
    # targets; bind its new hash before the case begins.
    state["plan_sha256"] = support.core.digest(host.plan)
    anchor = json.loads((host.state / "anchor.json").read_bytes())
    anchor["plan_sha256"] = state["plan_sha256"]
    (host.state / "anchor.json").write_bytes(support._bytes(anchor))
    state["anchor_sha256"] = hashlib.sha256(
        (host.state / "anchor.json").read_bytes()
    ).hexdigest()
    (host.state / "state.json").write_bytes(support._bytes(state))
    binding = {
        "plan_sha256": "e" * 64,
        "anchor": {"original": True},
        "scenario_sha256": "c" * 64,
        "nonce": "a" * 64,
        "guard_unit": "guard.service",
    }
    scenario = {
        "guard_unit": "guard.service",
        "synthetic_plan": {"state_root": str(host.state)},
        "unit_target": host.manifest["units"]["monitor.service"]["installed_path"],
        "env_target": host.manifest["units"]["monitor.service"]["after"][
            "environment_files"
        ][0]["path"],
    }
    read_host = SimpleNamespace(
        snapshot=lambda name, deadline: host._unit(name, {}, deadline),
        _exit_pids={"guard.service": host.current.pid},
    )
    context = SimpleNamespace(io=host.io, host=read_host)
    return SimpleNamespace(
        host=host,
        binding=binding,
        scenario=scenario,
        context=context,
        monkeypatch=monkeypatch,
    )


def owner(engine):
    h = engine.host
    return {
        "main": asdict(h.current),
        "invocation_id": h.invocation,
        "boot_id": h.boot,
        "entered_monotonic": h.status["guard.service"]["entered_monotonic"],
    }


def crash_exit(code):
    assert code == case.CRASH_EXIT
    raise PlannedCrash()


def run_stage(engine, stage):
    h = engine.host
    if stage < 3:
        with pytest.raises(PlannedCrash):
            case.execute_stage(
                h, engine.scenario, engine.binding, owner(engine), crash_exit=crash_exit
            )
    else:
        result = case.execute_stage(
            h, engine.scenario, engine.binding, owner(engine), crash_exit=crash_exit
        )
        assert result["status"] == "recovered"
    expected = case.CRASH_EXIT if stage < 3 else 0
    h.status["guard.service"].update(
        active="inactive",
        main=None,
        members=(),
        result="exit-code" if stage < 3 else "success",
        exit_code=expected,
    )
    h.processes.pop(h.current.pid)
    engine.context.host._exit_pids["guard.service"] = h.current.pid
    result = case._observe_terminal(
        engine.context, engine.scenario, engine.binding, stage, 100
    )
    assert result["exit_code"] == expected


def replacement(engine, number):
    h = engine.host
    # A same numeric PID with a new birth is intentional fake reuse evidence,
    # not an attempt to force the operating system to reuse a PID.
    h.current = case.q.linux.core.Process(h.current.pid, 100 + number)
    h.invocation = f"{number + 1:032x}"
    engine.monkeypatch.setenv("INVOCATION_ID", h.invocation)
    h.processes[h.current.pid] = asdict(h.current)
    h.status["guard.service"].update(
        active="active",
        main=h.current,
        members=(h.current,),
        entered_monotonic=10.0 + number,
        result="success",
        exit_code=0,
    )
    h.time = 10.0 + number


def test_four_fixed_crash_stages_recover_without_rewriting_original_contract(engine):
    h = engine.host
    original = case._state_signature(h, 100)
    anchor = (h.state / "anchor.json").read_bytes()
    intents = []
    for n in range(4):
        if n:
            replacement(engine, n)
        run_stage(engine, n)
        intents.append(case._intent_pin(h, 100))
        assert case._state_signature(h, 100) == original
        assert (h.state / "anchor.json").read_bytes() == anchor
    assert intents == [intents[0]] * 4
    records = [
        json.loads(case._path(h.state, "result", n).read_bytes()) for n in range(4)
    ]
    assert [r["label"] for r in records] == list(case.STAGES)
    assert len({r["owner"]["invocation_id"] for r in records}) == 4
    assert all(r["synthetic_state_sha256"] == original for r in records)
    assert all(records[-1][k] is False for k in case.q.CLAIMS)
    assert len(list((h.state / support.DIRECTORY).glob("*.intent.json"))) == 1
    assert len(list((h.state / support.DIRECTORY).glob("*.complete.json"))) == 1
    for name in h.plan["support_transition"]["after"]:
        assert h._unit(name, {}, 100)[1]["stage"] == "after"
    with pytest.raises(case.q.Refusal, match="already-complete"):
        case.execute_stage(
            h, engine.scenario, engine.binding, owner(engine), crash_exit=crash_exit
        )


def test_unacknowledged_old_exit_cannot_advance_to_next_stage(engine):
    h = engine.host
    with pytest.raises(PlannedCrash):
        case.execute_stage(
            h, engine.scenario, engine.binding, owner(engine), crash_exit=crash_exit
        )
    replacement(engine, 1)
    before = list(h.io.events)
    with pytest.raises(case.q.Refusal, match="unobserved-prior-exit"):
        case.execute_stage(
            h, engine.scenario, engine.binding, owner(engine), crash_exit=crash_exit
        )
    assert h.io.events == before


@pytest.mark.parametrize(
    "field,value",
    [("nonce", "wrong"), ("plan_sha256", "0" * 64), ("stage", 2), ("format", "wrong")],
)
def test_prior_record_binding_drift_refuses_without_support_mutation(
    engine, field, value
):
    run_stage(engine, 0)
    path = case._path(engine.host.state, "result", 0)
    row = json.loads(path.read_bytes())
    row[field] = value
    path.write_bytes(case.q.encode(row))
    replacement(engine, 1)
    before = list(engine.host.io.events)
    with pytest.raises(case.q.Refusal):
        case.execute_stage(
            engine.host,
            engine.scenario,
            engine.binding,
            owner(engine),
            crash_exit=crash_exit,
        )
    assert engine.host.io.events == before


@pytest.mark.parametrize(
    "field", ["phase", "last_action", "deadline", "deadline_wall_ns"]
)
def test_original_state_is_not_renewed_between_invocations(engine, field):
    run_stage(engine, 0)
    replacement(engine, 1)
    path = engine.host.state / "state.json"
    row = json.loads(path.read_bytes())
    if field == "last_action":
        row[field]["deadline"] += 1
    elif field == "phase":
        row[field] = "another-phase"
    else:
        row[field] += 1
    path.write_bytes(case.q.encode(row))
    with pytest.raises(case.q.Refusal, match="original-contract-changed"):
        case.execute_stage(
            engine.host,
            engine.scenario,
            engine.binding,
            owner(engine),
            crash_exit=crash_exit,
        )


def test_guard_death_record_requires_actual_pid_exit_not_only_dead_unit(engine):
    h = engine.host
    with pytest.raises(PlannedCrash):
        case.execute_stage(
            h, engine.scenario, engine.binding, owner(engine), crash_exit=crash_exit
        )
    h.status["guard.service"].update(
        active="inactive",
        main=None,
        members=(),
        result="exit-code",
        exit_code=case.CRASH_EXIT,
    )
    with pytest.raises(case.q.Refusal, match="still-alive"):
        case._observe_terminal(engine.context, engine.scenario, engine.binding, 0, 100)
    assert not case._path(h.state, "terminal", 0).exists()


@pytest.mark.parametrize("failure", ["exit-code", "invocation", "entered", "pid"])
def test_wrong_terminal_unit_does_not_certify_planned_crash(engine, failure):
    h = engine.host
    with pytest.raises(PlannedCrash):
        case.execute_stage(
            h, engine.scenario, engine.binding, owner(engine), crash_exit=crash_exit
        )
    h.status["guard.service"].update(
        active="inactive",
        main=None,
        members=(),
        result="exit-code",
        exit_code=case.CRASH_EXIT,
    )
    h.processes.clear()
    if failure == "exit-code":
        h.status["guard.service"]["exit_code"] = 1
    elif failure == "invocation":
        h.invocation = "f" * 32
    elif failure == "entered":
        h.status["guard.service"]["entered_monotonic"] = 1
    else:
        engine.context.host._exit_pids["guard.service"] += 1
    with pytest.raises(case.q.Refusal, match="terminal-binding"):
        case._observe_terminal(engine.context, engine.scenario, engine.binding, 0, 100)
    assert not case._path(h.state, "terminal", 0).exists()


def test_crash_callback_return_is_not_success(engine):
    with pytest.raises(case.q.Refusal, match="crash-exit-returned"):
        case.execute_stage(
            engine.host,
            engine.scenario,
            engine.binding,
            owner(engine),
            crash_exit=lambda _: None,
        )


def test_missing_original_intent_cannot_be_replaced_on_reentry(engine):
    run_stage(engine, 0)
    replacement(engine, 1)
    pin = case._intent_pin(engine.host, 100)
    Path(pin["path"]).unlink()
    with pytest.raises(FileNotFoundError):
        case.execute_stage(
            engine.host,
            engine.scenario,
            engine.binding,
            owner(engine),
            crash_exit=crash_exit,
        )


def test_residual_cgroup_member_prevents_terminal_receipt(engine):
    h = engine.host
    with pytest.raises(PlannedCrash):
        case.execute_stage(
            h, engine.scenario, engine.binding, owner(engine), crash_exit=crash_exit
        )
    h.status["guard.service"].update(
        active="inactive",
        main=None,
        members=(h.current,),
        result="exit-code",
        exit_code=case.CRASH_EXIT,
    )
    with pytest.raises(case.q.Refusal, match="did-not-exit"):
        case._observe_terminal(
            engine.context, engine.scenario, engine.binding, 0, h.time + 0.2
        )
    assert not case._path(h.state, "terminal", 0).exists()


def test_same_invocation_is_not_a_new_predeclared_stage(engine):
    h = engine.host
    original = h.invocation
    run_stage(engine, 0)
    replacement(engine, 1)
    h.invocation = original
    engine.monkeypatch.setenv("INVOCATION_ID", original)
    before = list(h.io.events)
    with pytest.raises(case.q.Refusal, match="invocation-reused"):
        case.execute_stage(
            h, engine.scenario, engine.binding, owner(engine), crash_exit=crash_exit
        )
    assert h.io.events == before


def test_future_stage_record_cannot_skip_fixed_sequence(engine):
    path = case._path(engine.host.state, "result", 2)
    path.write_bytes(b"{}\n")
    with pytest.raises(case.q.Refusal, match="record-gap"):
        case.execute_stage(
            engine.host,
            engine.scenario,
            engine.binding,
            owner(engine),
            crash_exit=crash_exit,
        )
    assert not case._path(engine.host.state, "started", 0).exists()


def test_self_birth_is_rechecked_against_authorized_context(engine):
    h = engine.host
    admitted, metadata = h._unit("guard.service", {}, 100)
    changed = replace(admitted, main=case.q.linux.core.Process(h.current.pid, 999))
    changed = replace(changed, members=(changed.main,))
    context = SimpleNamespace(
        unit=admitted,
        io=h.io,
        host=SimpleNamespace(snapshot=lambda *_: (changed, metadata)),
    )
    with pytest.raises(case.q.Refusal, match="birth-changed"):
        case._owner(context, "guard.service", 100)


def test_result_birth_cannot_differ_from_started_owner(engine):
    h = engine.host
    with pytest.raises(PlannedCrash):
        case.execute_stage(
            h, engine.scenario, engine.binding, owner(engine), crash_exit=crash_exit
        )
    path = case._path(h.state, "result", 0)
    row = json.loads(path.read_bytes())
    row["owner"]["main"]["start_ticks"] += 1
    path.write_bytes(case.q.encode(row))
    h.processes.clear()
    h.status["guard.service"].update(
        active="inactive",
        main=None,
        members=(),
        result="exit-code",
        exit_code=case.CRASH_EXIT,
    )
    with pytest.raises(case.q.Refusal, match="terminal-binding"):
        case._observe_terminal(engine.context, engine.scenario, engine.binding, 0, 100)


@pytest.fixture
def scenario_input(tmp_path):
    """Only this helper's scenario sub-schema; full Plan admission is separate."""
    nonce = "1" * 32
    scratch = Path("/run") / ("edgeconnect-cpuqual-" + nonce)
    names = {
        r: "edgeconnect-cpuqual-" + nonce + "-" + r + ".service"
        for r in ("guard", "r3", "r4", "probe", "support", "observer")
    }
    units = {}
    files = {}
    for role, name in names.items():
        spec: dict[str, Any] = {
            "role": "observer" if role == "observer" else "workload",
            "mode": "support_transaction" if role == "guard" else "sleep",
            "payload": {},
            "installed_path": "/etc/systemd/system/" + name,
        }
        for side, marker in (("before", "a"), ("after", "b")):
            env = {
                "path": str(scratch / "env" / (name + ".env")),
                "source_path": str(tmp_path / (name + side + ".env")),
                "sha256": marker * 64,
                "bytes": 1,
                "mode": 0o600,
            }
            spec[side] = {
                "unit": {
                    "path": str(tmp_path / (name + side)),
                    "sha256": marker * 64,
                    "bytes": 1,
                },
                "properties": {"Environment": "", "Type": "exec"},
                "environment_files": [env],
            }
        units[name] = spec
        for target in (
            spec["installed_path"],
            spec["after"]["environment_files"][0]["path"],
        ):
            files[target] = [
                {"source": {"sha256": m * 64}, "mode": 0o600} for m in ("a", "b")
            ]
    synthetic = {
        "run_root": str(scratch / "synthetic-run"),
        "state_root": str(scratch / "synthetic-support-state"),
        "exclusion_path": str(scratch / "qualification.lock"),
        "freshness_plan_sha256": "f" * 64,
        "units": {
            r: {"name": names[r], "initial_enabled": False, "committed_enabled": False}
            for r in ("guard", "r3", "r4", "probe")
        },
        "support_transition": {
            side: {
                names["support"]: {
                    "definition_sha256": units[names["support"]][side]["unit"][
                        "sha256"
                    ],
                    "environment_sha256": case.q.linux.env_identity(
                        "", units[names["support"]][side]["environment_files"]
                    ),
                    "enabled": True,
                }
            }
            for side in ("before", "after")
        },
    }
    scenario = {
        "format": case.FORMAT,
        "schema_version": 1,
        "nonce": nonce,
        "guard_unit": names["guard"],
        "synthetic_plan": synthetic,
        "unit_target": units[names["support"]]["installed_path"],
        "env_target": units[names["support"]]["after"]["environment_files"][0]["path"],
    }
    p = {
        "input_root": str(tmp_path),
        "scratch_root": str(scratch),
        "units": units,
        "files": files,
        "nonce": nonce,
    }

    def load():
        raw = case.q.encode(scenario)
        pin = {
            "path": str(tmp_path / "scenario.json"),
            "sha256": case.q.sha(raw),
            "bytes": len(raw),
        }
        units[names["guard"]]["payload"]["scenario"] = pin
        io: Any = SimpleNamespace(plan=SimpleNamespace(value=p), read=lambda *_: raw)
        return case.load_scenario(io, pin, 100)

    return SimpleNamespace(
        plan=p, scenario=scenario, synthetic=synthetic, names=names, load=load
    )


def test_scenario_subschema_is_bound_without_a_circular_plan_hash(scenario_input):
    x = scenario_input
    assert x.load() == x.scenario
    assert "plan_sha256" not in x.scenario


@pytest.mark.parametrize(
    "fault",
    [
        "format",
        "version",
        "nonce",
        "extra",
        "guard-role",
        "guard-mode",
        "run-root",
        "state-root",
        "lease",
        "duplicate-role",
        "observer-role",
        "support-overlap",
        "unit-target",
        "env-target",
        "unit-hash",
        "env-hash",
        "boolean",
    ],
)
def test_scenario_refuses_role_scope_or_pin_drift(scenario_input, fault):
    x = scenario_input
    if fault == "format":
        x.scenario["format"] = "production"
    elif fault == "version":
        x.scenario["schema_version"] = True
    elif fault == "nonce":
        x.scenario["nonce"] = "2" * 32
    elif fault == "extra":
        x.scenario["role_override"] = "publisher"
    elif fault == "guard-role":
        x.plan["units"][x.names["guard"]]["role"] = "observer"
    elif fault == "guard-mode":
        x.plan["units"][x.names["guard"]]["mode"] = "sleep"
    elif fault == "run-root":
        x.synthetic["run_root"] = "/production/run"
    elif fault == "state-root":
        x.synthetic["state_root"] = "/production/state"
    elif fault == "lease":
        x.synthetic["exclusion_path"] = "/production.lock"
    elif fault == "duplicate-role":
        x.synthetic["units"]["r4"]["name"] = x.names["guard"]
    elif fault == "observer-role":
        x.synthetic["units"]["r4"]["name"] = x.names["observer"]
    elif fault == "support-overlap":
        for side in ("before", "after"):
            x.synthetic["support_transition"][side][x.names["guard"]] = x.synthetic[
                "support_transition"
            ][side].pop(x.names["support"])
    elif fault == "unit-target":
        x.scenario["unit_target"] = "/etc/systemd/system/production.service"
    elif fault == "env-target":
        x.scenario["env_target"] = "/production.env"
    elif fault == "unit-hash":
        x.synthetic["support_transition"]["after"][x.names["support"]][
            "definition_sha256"
        ] = "e" * 64
    elif fault == "env-hash":
        x.synthetic["support_transition"]["after"][x.names["support"]][
            "environment_sha256"
        ] = "e" * 64
    else:
        x.synthetic["units"]["r4"]["committed_enabled"] = 1
    with pytest.raises(case.q.Refusal):
        x.load()
