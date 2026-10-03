"""Four fixed dummy support invocations; never a production guard admission.

The immutable plan authorizes one support_transaction workload unit. Its
observer starts exactly three planned crashes and one recovery invocation.
Actual PID/invocation/flock facts are separate from synthetic helper authority.
Importing this module starts no process, service, lease or GPU work.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Iterator, NoReturn

from scripts import strength_freshness_cpu_qualification as q
from scripts import strength_freshness_units as support

FORMAT = "strength-freshness-cpu-support-scenario-v1"
STATE = "strength-freshness-cpu-synthetic-support-state-v1"
CRASH_EXIT = 83
STAGES = ("unit-write", "env-write", "reload", "recovered")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _path(state: Path, kind: str, stage: int) -> Path:
    q.require(
        kind in {"started", "result", "terminal"} and 0 <= stage < 4,
        "support-record-kind",
    )
    return state / f"support-case-{kind}-{stage}.json"


def load_scenario(
    io: q.ClosedIO, pin: dict[str, Any], deadline: float
) -> dict[str, Any]:
    p = io.plan.value
    q.pin_shape(pin)
    q.require(
        Path(pin["path"]).is_relative_to(Path(p["input_root"])),
        "support-scenario-input",
    )
    raw = io.read(Path(pin["path"]), 2**20, deadline)
    q.require(
        len(raw) == pin["bytes"] and _hash(raw) == pin["sha256"], "support-scenario-pin"
    )
    value = q.exact(
        json.loads(raw),
        {
            "format",
            "schema_version",
            "nonce",
            "guard_unit",
            "synthetic_plan",
            "unit_target",
            "env_target",
        },
        "support-scenario-fields",
    )
    q.require(
        value["format"] == FORMAT
        and type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and value["nonce"] == p["nonce"],
        "support-scenario-binding",
    )
    guard = value["guard_unit"]
    q.require(
        guard in p["units"]
        and p["units"][guard]["role"] == "workload"
        and p["units"][guard]["mode"] == "support_transaction"
        and p["units"][guard]["payload"]["scenario"] == pin,
        "support-fixed-guard",
    )
    q.require(
        all(
            p["units"][guard][side]["properties"].get("Type") == "exec"
            for side in ("before", "after")
        ),
        "support-guard-must-be-exec",
    )
    synthetic = q.exact(
        value["synthetic_plan"],
        {
            "run_root",
            "state_root",
            "exclusion_path",
            "freshness_plan_sha256",
            "units",
            "support_transition",
        },
        "support-synthetic-plan-fields",
    )
    scratch = Path(p["scratch_root"])
    q.require(
        synthetic["run_root"] == str(scratch / "synthetic-run")
        and synthetic["state_root"] == str(scratch / "synthetic-support-state")
        and synthetic["exclusion_path"] == str(scratch / "qualification.lock")
        and q.HASH.fullmatch(synthetic["freshness_plan_sha256"]),
        "support-synthetic-roots",
    )
    q.require(
        set(synthetic["units"]) == {"guard", "r3", "r4", "probe"},
        "support-synthetic-roles",
    )
    names = [v["name"] for v in synthetic["units"].values()]
    q.require(
        len(set(names)) == 4
        and synthetic["units"]["guard"]["name"] == guard
        and all(n in p["units"] and p["units"][n]["role"] == "workload" for n in names),
        "support-role-inventory",
    )
    for spec in synthetic["units"].values():
        q.exact(
            spec,
            {"name", "initial_enabled", "committed_enabled"},
            "support-role-fields",
        )
        q.require(
            type(spec["initial_enabled"]) is bool
            and type(spec["committed_enabled"]) is bool,
            "support-role-enablement",
        )
    transition = q.exact(
        synthetic["support_transition"],
        {"before", "after"},
        "support-transition-fields",
    )
    supports = set(transition["before"])
    q.require(
        supports
        and supports == set(transition["after"])
        and not supports.intersection(names),
        "support-disjoint-inventory",
    )
    for name in supports:
        q.require(
            name in p["units"] and p["units"][name]["role"] == "workload",
            "support-workload-only",
        )
        for side in ("before", "after"):
            row = q.exact(
                transition[side][name],
                {"definition_sha256", "environment_sha256", "enabled"},
                "support-transition-pin",
            )
            spec = p["units"][name][side]
            q.require(
                row["definition_sha256"] == spec["unit"]["sha256"]
                and row["environment_sha256"]
                == q.linux.env_identity(
                    spec["properties"].get("Environment", ""), spec["environment_files"]
                )
                and type(row["enabled"]) is bool,
                "support-transition-identity",
            )
    unit_target, env_target = value["unit_target"], value["env_target"]
    q.require(
        unit_target in {p["units"][n]["installed_path"] for n in supports}
        and env_target
        in {
            e["path"]
            for n in supports
            for e in p["units"][n]["after"]["environment_files"]
        }
        and unit_target < env_target,
        "support-ordered-fault-targets",
    )
    for target in (unit_target, env_target):
        q.require(
            target in p["files"]
            and len({(v["source"]["sha256"], v["mode"]) for v in p["files"][target]})
            >= 2,
            "support-fault-needs-real-change",
        )
    return value


def _bindings(
    context: Any, scenario: dict[str, Any], pin: dict[str, Any]
) -> dict[str, Any]:
    return {
        "plan_sha256": context.plan.checksum,
        "anchor": context.anchor.as_dict(),
        "scenario_sha256": pin["sha256"],
        "nonce": scenario["nonce"],
        "guard_unit": scenario["guard_unit"],
    }


def _same(value: dict[str, Any], binding: dict[str, Any]) -> None:
    q.require(
        all(value.get(k) == v for k, v in binding.items()), "support-record-binding"
    )


def _next_stage(io: Any, state: Path, binding: dict[str, Any], deadline: float) -> int:
    for stage in range(4):
        started, result, terminal = (
            _path(state, k, stage) for k in ("started", "result", "terminal")
        )
        if not io.exists(started):
            q.require(
                not io.exists(result) and not io.exists(terminal),
                "support-orphan-record",
            )
            q.require(
                all(
                    not io.exists(_path(state, k, later))
                    for later in range(stage + 1, 4)
                    for k in ("started", "result", "terminal")
                ),
                "support-record-gap",
            )
            return stage
        q.require(
            io.exists(result) and io.exists(terminal), "support-unobserved-prior-exit"
        )
        s, r, t = (io.json(p, deadline) for p in (started, result, terminal))
        for value, kind in ((s, "started"), (r, "result"), (t, "terminal")):
            _same(value, binding)
            q.require(
                value.get("format") == FORMAT + "-" + kind
                and value.get("stage") == stage
                and value.get("label") == STAGES[stage],
                "support-record-stage",
            )
        q.require(
            t["started_sha256"] == _hash(io.read(started, 2**20, deadline))
            and t["result_sha256"] == _hash(io.read(result, 2**20, deadline))
            and t["invocation_id"]
            == s["owner"]["invocation_id"]
            == r["owner"]["invocation_id"]
            and t["exit_code"] == (CRASH_EXIT if stage < 3 else 0)
            and r["status"] == ("planned-crash" if stage < 3 else "recovered")
            and r["synthetic_state_sha256"] == s["synthetic_state_sha256"],
            "support-terminal-chain",
        )
    raise q.Refusal("support-case-already-complete")


def _owner(context: Any, guard: str, deadline: float) -> dict[str, Any]:
    unit, _ = context.host.snapshot(guard, deadline)
    while unit.active == "activating" and context.io.now() + 0.01 < deadline:
        context.io.sleep(0.01)
        unit, _ = context.host.snapshot(guard, deadline)
    q.require(
        unit.main is not None
        and unit.main.pid == os.getpid()
        and unit.main in unit.members
        and unit.active == "active"
        and unit.job is None
        and unit.enabled
        and unit.invocation_id == os.environ.get("INVOCATION_ID")
        and re.fullmatch(r"[0-9a-f]{32}", unit.invocation_id),
        "support-actual-self-unit",
    )
    assert unit.main is not None
    q.require(
        unit.main == context.unit.main
        and unit.invocation_id == context.unit.invocation_id,
        "support-self-birth-changed",
    )
    return {
        "main": asdict(unit.main),
        "invocation_id": unit.invocation_id,
        "boot_id": context.anchor.boot_id,
        "entered_monotonic": unit.entered_monotonic,
    }


@contextmanager
def _lease_file(path: Path) -> Iterator[int]:
    q.require(path.resolve() == path and path.parent.is_dir(), "support-lock-path")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        st = os.fstat(fd)
        q.require(
            stat.S_ISREG(st.st_mode)
            and st.st_uid == os.geteuid()
            and stat.S_IMODE(st.st_mode) == 0o600,
            "support-lock-protection",
        )
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield fd
    finally:
        os.close(fd)


def _initialize_state(host: Any, context: Any, owner: dict[str, Any]) -> None:
    anchor = context.anchor.as_dict()
    deadline = context.anchor.deadline("work")
    wall_end = anchor["started_wall_ns"] + int(q.LIMITS["work"] * 1e9)
    expected = {
        "format": STATE,
        "plan_sha256": q.linux.core.digest(host.plan),
        "attempt_id": anchor["attempt_id"],
        "nonce": anchor["nonce"],
        "phase": "dummy-support-transaction",
        "dummy_authority": "r4",
        "active_boot_id": anchor["boot_id"],
        "deadline": deadline,
        "deadline_wall_ns": wall_end,
        "last_wall_ns": anchor["started_wall_ns"],
        "last_action": {
            "kind": "restore-support-without-proof-r4",
            "deadline": deadline,
            "details": {},
        },
    }
    path = host.state / "state.json"
    if host.io.exists(path):
        current = host.io.json(path, deadline)
        q.require(
            all(current.get(k) == v for k, v in expected.items()),
            "support-synthetic-state-drift",
        )
        return
    anchor_data = {
        k: expected[k]
        for k in ("plan_sha256", "attempt_id", "nonce", "deadline_wall_ns")
    }
    raw = q.encode(anchor_data)
    host.io.atomic(host.state / "anchor.json", raw)
    expected.update(
        anchor_sha256=_hash(raw),
        owners={"guard": {k: owner[k] for k in ("main", "invocation_id", "boot_id")}},
    )
    host.io.atomic(path, q.encode(expected))


def _state_signature(host: Any, deadline: float) -> str:
    state = host.io.json(host.state / "state.json", deadline)
    state["owners"] = {k: v for k, v in state["owners"].items() if k != "guard"}
    return q.sha(q.encode(state))


def _intent_pin(host: Any, deadline: float) -> dict[str, Any]:
    state = host.io.json(host.state / "state.json", deadline)
    identity = q.linux.core.digest(
        {
            "action": state["last_action"],
            "attempt_id": state["attempt_id"],
            "nonce": state["nonce"],
        }
    )
    path = host.state / support.DIRECTORY / (identity + ".intent.json")
    raw = host.io.read(path, 2**20, deadline)
    value = json.loads(raw)
    q.require(
        value["action"] == state["last_action"] and value["phase"] == state["phase"],
        "support-exact-original-intent",
    )
    return {"path": str(path), "sha256": _hash(raw), "bytes": len(raw)}


class _FaultIO:
    """Observation/fault wrapper; every operation still uses the original capability."""

    def __init__(
        self,
        base: Any,
        stage: int,
        scenario: dict[str, Any],
        fault: Callable[[], NoReturn],
    ):
        self.base, self.stage, self.scenario, self.fault = base, stage, scenario, fault

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def atomic(self, path: Path, data: bytes, **kwargs: Any) -> None:
        self.base.atomic(path, data, **kwargs)
        if (self.stage == 0 and str(path) == self.scenario["unit_target"]) or (
            self.stage == 1 and str(path) == self.scenario["env_target"]
        ):
            self.fault()

    def command(self, argv: list[str], deadline: float) -> str:
        value = self.base.command(argv, deadline)
        if self.stage == 2 and argv == ["systemctl", "daemon-reload"]:
            self.fault()
        return value


def execute_stage(
    host: Any,
    scenario: dict[str, Any],
    binding: dict[str, Any],
    owner: dict[str, Any],
    *,
    crash_exit: Callable[[int], Any],
) -> dict[str, Any]:
    """Phase engine; tests supply fake Linux facts, never fake production authority."""
    deadline = host.io.anchor.deadline("work")
    stage = _next_stage(host.io, host.state, binding, deadline)
    before = _state_signature(host, deadline)
    previous_invocations = {
        host.io.json(_path(host.state, "started", i), deadline)["owner"][
            "invocation_id"
        ]
        for i in range(stage)
    }
    q.require(
        owner["invocation_id"] not in previous_invocations, "support-invocation-reused"
    )
    if stage:
        first = host.io.json(_path(host.state, "started", 0), deadline)
        prior = host.io.json(_path(host.state, "result", stage - 1), deadline)
        q.require(
            before == first["synthetic_state_sha256"]
            and _intent_pin(host, deadline) == prior["helper_intent"],
            "support-original-contract-changed",
        )
    record = {
        **binding,
        "stage": stage,
        "label": STAGES[stage],
        "owner": owner,
        "synthetic_state_sha256": before,
        "scope": "dummy-only; authority synthetic; unit/lease facts independently observed",
    }
    base = host.io
    base.atomic(
        _path(host.state, "started", stage),
        q.encode({"format": FORMAT + "-started", **record}),
    )

    def fault() -> NoReturn:
        q.require(
            _state_signature(host, deadline) == before, "support-original-state-changed"
        )
        base.atomic(
            _path(host.state, "result", stage),
            q.encode(
                {
                    "format": FORMAT + "-result",
                    **record,
                    "status": "planned-crash",
                    "expected_exit": CRASH_EXIT,
                    "helper_intent": _intent_pin(host, deadline),
                }
            ),
        )
        crash_exit(CRASH_EXIT)
        raise q.Refusal("support-crash-exit-returned")

    host.io = _FaultIO(base, stage, scenario, fault)
    try:
        if stage == 0:
            support.transition(host, "after", deadline)
        else:
            state = base.json(host.state / "state.json", deadline)
            old = q.linux.core.Process(**state["owners"]["guard"]["main"])
            base.remember_prior_process(scenario["guard_unit"], old)
            pre = support.recover_transition(host, deadline, pre_admission=True)
            q.require(pre is not None, "support-prior-transaction-missing")
            q.require(
                _state_signature(host, deadline) == before,
                "support-original-state-changed",
            )
            state["owners"]["guard"] = {
                k: owner[k] for k in ("main", "invocation_id", "boot_id")
            }
            base.atomic(host.state / "state.json", q.encode(state), overwrite=True)
            completed = support.recover_transition(host, deadline)
            q.require(
                completed is not None
                and completed.get("format") == support.FORMAT + "-complete",
                "support-completion-missing",
            )
        q.require(stage == 3, "support-planned-crash-point-not-reached")
        q.require(
            _state_signature(host, deadline) == before, "support-original-state-changed"
        )
        result = {
            "format": FORMAT + "-result",
            **record,
            "status": "recovered",
            "expected_exit": 0,
            "helper_intent": _intent_pin(host, deadline),
            **q.CLAIMS,
        }
        base.atomic(_path(host.state, "result", stage), q.encode(result))
        return result
    finally:
        host.io = base


def run_support_payload(
    context: Any, *, crash_exit: Callable[[int], Any] = os._exit
) -> dict[str, Any]:
    auth, io = context.authorization, context.io
    q.require(
        auth["role"] == "workload" and auth["mode"] == "support_transaction",
        "support-payload-role",
    )
    deadline = io._deadline(context.anchor.deadline("work"))
    pin = auth["payload"]["scenario"]
    scenario = load_scenario(io, pin, deadline)
    q.require(
        auth["unit"] == scenario["guard_unit"]
        and context.plan.checksum == context.anchor.plan_sha256,
        "support-payload-binding",
    )
    owner = _owner(context, scenario["guard_unit"], deadline)
    for path in (
        Path(scenario["synthetic_plan"]["state_root"]),
        Path(scenario["synthetic_plan"]["run_root"]),
    ):
        q.require(path.resolve() == path, "support-directory-alias")
        path.mkdir(mode=0o700, exist_ok=True)
        st = path.stat()
        q.require(
            st.st_uid == os.geteuid() and stat.S_IMODE(st.st_mode) == 0o700,
            "support-directory-protection",
        )
    with _lease_file(Path(scenario["synthetic_plan"]["exclusion_path"])) as fd:
        host = q.DummySupportHost(io, scenario["synthetic_plan"], fd)
        _initialize_state(host, context, owner)
        lease = {
            "path": host.plan["exclusion_path"],
            "held": True,
            "attempt_id": context.anchor.attempt_id,
            "nonce": context.anchor.nonce,
            "plan_sha256": q.linux.core.digest(host.plan),
            **{k: owner[k] for k in ("main", "invocation_id", "boot_id")},
        }
        lease["owner"] = lease.pop("main")
        io.atomic(host.state / q.linux.LEASE, q.encode(lease), overwrite=True)
        q.require(host._lease(deadline) == lease, "support-actual-kernel-lease")
        return execute_stage(
            host,
            scenario,
            _bindings(context, scenario, pin),
            owner,
            crash_exit=crash_exit,
        )


def _observe_terminal(
    context: Any,
    scenario: dict[str, Any],
    binding: dict[str, Any],
    stage: int,
    deadline: float,
) -> dict[str, Any]:
    io, host = context.io, context.host
    state = Path(scenario["synthetic_plan"]["state_root"])
    unit = None
    while io.now() < deadline:
        unit, _ = host.snapshot(scenario["guard_unit"], deadline)
        if unit.dead:
            break
        q.require(io.now() + 0.002 < deadline, "support-guard-did-not-exit")
        io.sleep(min(0.05, max(0, deadline - io.now() - 0.002)))
    q.require(unit is not None and unit.dead, "support-guard-did-not-exit")
    assert unit is not None
    started, result = (
        io.json(_path(state, k, stage), deadline) for k in ("started", "result")
    )
    for value, kind in ((started, "started"), (result, "result")):
        _same(value, binding)
        q.require(
            value["format"] == FORMAT + "-" + kind
            and value["stage"] == stage
            and value["label"] == STAGES[stage],
            "support-observed-stage",
        )
    expected = CRASH_EXIT if stage < 3 else 0
    owner = started["owner"]
    previous = q.linux.core.Process(**owner["main"])
    io.remember_prior_process(scenario["guard_unit"], previous)
    live = io.process(previous.pid, deadline)
    q.require(
        live is None or live["start_ticks"] != previous.start_ticks,
        "support-exited-process-still-alive",
    )
    q.require(
        result["owner"] == owner
        and result["expected_exit"] == expected
        and result["status"] == ("planned-crash" if stage < 3 else "recovered")
        and unit.exit_code == expected
        and unit.result == ("exit-code" if stage < 3 else "success")
        and unit.entered_monotonic == owner["entered_monotonic"]
        and unit.invocation_id in ("", owner["invocation_id"])
        and host._exit_pids[unit.name] == owner["main"]["pid"],
        "support-real-terminal-binding",
    )
    record = {
        "format": FORMAT + "-terminal",
        **binding,
        "stage": stage,
        "label": STAGES[stage],
        "invocation_id": owner["invocation_id"],
        "exit_code": expected,
        "unit": asdict(unit),
        "started_sha256": _hash(
            io.read(_path(state, "started", stage), 2**20, deadline)
        ),
        "result_sha256": _hash(io.read(_path(state, "result", stage), 2**20, deadline)),
    }
    io.atomic(_path(state, "terminal", stage), q.encode(record))
    return record


def verify_support_case(context: Any) -> dict[str, Any]:
    """Surviving observer drives four explicit stages, never an automatic retry."""
    io = context.io
    p = context.plan.value
    guards = [
        (n, u) for n, u in p["units"].items() if u["mode"] == "support_transaction"
    ]
    q.require(len(guards) == 1, "support-one-fixed-guard")
    guard, spec = guards[0]
    deadline = io._deadline(context.anchor.deadline("work"))
    scenario = load_scenario(io, spec["payload"]["scenario"], deadline)
    binding = _bindings(context, scenario, spec["payload"]["scenario"])
    state = Path(scenario["synthetic_plan"]["state_root"])
    q.require(
        all(
            not io.exists(_path(state, k, n))
            for n in range(4)
            for k in ("started", "result", "terminal")
        ),
        "support-no-case-rerun",
    )
    io.action("enable", guard, deadline)
    results = []
    for stage in range(4):
        unit, _ = context.host.snapshot(guard, deadline)
        q.require(unit.dead, "support-prior-guard-not-dead")
        io.action("start", guard, deadline)
        results.append(_observe_terminal(context, scenario, binding, stage, deadline))
    q.require(
        len({r["invocation_id"] for r in results}) == 4,
        "support-invocations-not-distinct",
    )
    return {
        "status": "dummy-support-case-observed",
        **binding,
        "stages": results,
        "authority": "synthetic-dummy-only",
        "production_entrypoints_used": False,
        **q.CLAIMS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    context = q.authorized_context(
        args.authorization,
        expected_role="workload",
        expected_mode="support_transaction",
    )
    print(json.dumps(run_support_payload(context), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
