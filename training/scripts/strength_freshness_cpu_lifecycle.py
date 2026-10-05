"""Finite dummy CPU qualification lifetimes; never controls systemd or GPUs.

The separately reviewed harness owns plan, path, unit and process authority.
These gates consume its ownership-qualified observations. Raw facts are saved
before admission checks; a candidate or published result is not a final pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import time
from typing import Any, Callable, Iterator, Mapping


ENDS = {
    "work": 390,
    "dispatcher_terminal": 405,
    "cleanup": 540,
    "observer": 570,
    "observer_terminal": 575,
    "publisher": 590,
    "publisher_terminal": 595,
    "audit": 600,
}
FALSE_CLAIMS = {
    "execution_qualified": False,
    "full_linux_controller_qualified": False,
    "real_reboot_qualified": False,
    "cuda_qualified": False,
    "r4_activated": False,
}
MAX_LOG_FILES = 4096
MAX_LOG_BYTES = 32 * 2**20
MAX_SOURCE_PINS = 256
MAX_SOURCE_BYTES = 2**20


class Refusal(ValueError):
    """Non-secret refusal suitable for a bounded raw evidence log."""


def require(condition: object, reason: str) -> None:
    if not condition:
        raise Refusal(reason)


def finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)  # type: ignore[arg-type]


def encoded(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def digest(value: object) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def strict_json(raw: bytes) -> Any:
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate-json-key")
            value[key] = item
        return value

    def invalid(_value):
        raise Refusal("nonfinite-json")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


@dataclass(frozen=True)
class Clock:
    boot_id: str
    monotonic: float
    wall_ns: int


@dataclass(frozen=True)
class Anchor:
    attempt_id: str
    nonce: str
    plan_sha256: str
    boot_id: str
    started_monotonic: float
    started_wall_ns: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Anchor:
        require(
            set(value)
            == {
                "attempt_id",
                "nonce",
                "plan_sha256",
                "boot_id",
                "started_monotonic",
                "started_wall_ns",
            },
            "anchor-fields",
        )
        require(
            isinstance(value["attempt_id"], str)
            and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", value["attempt_id"]),
            "anchor-attempt",
        )
        for field, length in (("nonce", 32), ("plan_sha256", 64)):
            require(
                isinstance(value[field], str)
                and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value[field]),
                "anchor-" + field,
            )
        require(
            isinstance(value["boot_id"], str) and 0 < len(value["boot_id"]) <= 128,
            "anchor-boot",
        )
        require(
            finite(value["started_monotonic"])
            and value["started_monotonic"] >= 0
            and type(value["started_wall_ns"]) is int
            and value["started_wall_ns"] > 0,
            "anchor-clock",
        )
        return cls(**value)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def deadline(self, phase: str) -> float:
        require(phase in ENDS, "unknown-phase")
        return self.started_monotonic + ENDS[phase]

    def remaining(self, phase: str, now: Clock, stop_reserve: float = 0) -> float:
        require(
            now.boot_id == self.boot_id
            and finite(now.monotonic)
            and now.monotonic >= self.started_monotonic
            and type(now.wall_ns) is int
            and now.wall_ns >= self.started_wall_ns,
            "clock-identity-or-regression",
        )
        require(finite(stop_reserve) and stop_reserve >= 0, "stop-reserve")
        remaining = (
            min(
                self.deadline(phase) - now.monotonic,
                ENDS[phase] - (now.wall_ns - self.started_wall_ns) / 1e9,
            )
            - stop_reserve
        )
        require(remaining > 0, "deadline-" + phase)
        return remaining

    def effective_deadline(self, phase: str, now: Clock) -> float:
        return now.monotonic + self.remaining(phase, now)


class RawLog:
    """Atomic no-clobber receipts in a caller-authorized private directory.

    The dispatcher must pass only allowlisted non-secret observations. This is
    an evidence writer, not a capability issuer or a replacement for ownership
    validation. Each record retains the original anchor and plan binding.
    """

    def __init__(self, root: Path, anchor: Anchor):
        Anchor.from_dict(anchor.as_dict())
        require(root.is_absolute() and root.resolve() == root, "log-canonical")
        if not root.exists():
            root.mkdir(mode=0o700)
        info = root.lstat()
        require(
            stat.S_ISDIR(info.st_mode)
            and info.st_uid == os.geteuid()
            and stat.S_IMODE(info.st_mode) == 0o700,
            "log-private-directory",
        )
        self.root, self.anchor = root, anchor

    def _directory(self) -> None:
        info = self.root.lstat()
        require(
            self.root.resolve() == self.root
            and stat.S_ISDIR(info.st_mode)
            and info.st_uid == os.geteuid()
            and stat.S_IMODE(info.st_mode) == 0o700,
            "log-directory-changed",
        )

    def record(self, event: str, data: Mapping[str, Any]) -> dict[str, Any]:
        return self._record(event, data, once=False)

    def record_once(self, event: str, data: Mapping[str, Any]) -> dict[str, Any]:
        return self._record(event, data, once=True)

    def _record(
        self, event: str, data: Mapping[str, Any], *, once: bool
    ) -> dict[str, Any]:
        self._directory()
        require(re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", event), "event-name")
        body = {
            "format": "strength-freshness-cpu-lifecycle-evidence-v1",
            "schema_version": 1,
            "anchor": self.anchor.as_dict(),
            "anchor_sha256": digest(self.anchor.as_dict()),
            "event": event,
            "data": dict(data),
        }
        raw = encoded(body)
        require(len(raw) <= 262144, "raw-evidence-size")
        with self._append_lock():
            if once:
                require(
                    not list(self.root.glob(event + "-*.json")), "duplicate-" + event
                )
            return self._publish(event, raw)

    @contextmanager
    def _append_lock(self) -> Iterator[None]:
        # A short bounded filesystem critical section, not a workload lease.
        # It cannot move any caller's original phase/action deadline.
        lock = self.root / ".append.lock"
        fd = os.open(
            lock,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            info = os.fstat(fd)
            require(
                stat.S_ISREG(info.st_mode)
                and info.st_uid == os.geteuid()
                and stat.S_IMODE(info.st_mode) == 0o600
                and info.st_size == 0,
                "raw-log-lock-file",
            )
            end = time.monotonic() + 0.2
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    require(time.monotonic() < end, "raw-log-lock-busy")
                    time.sleep(0.001)
            self._directory()
            yield
        finally:
            os.close(fd)

    def _publish(self, event: str, raw: bytes) -> dict[str, Any]:
        existing = [p for p in self.root.iterdir() if p.name != ".append.lock"]
        require(
            len(existing) < MAX_LOG_FILES
            and sum(p.lstat().st_size for p in existing) + len(raw) <= MAX_LOG_BYTES,
            "raw-evidence-cap",
        )
        token = secrets.token_hex(16)
        temporary = self.root / ("." + token + ".partial")
        destination = self.root / (event + "-" + token + ".json")
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
        os.link(temporary, destination)
        temporary.unlink()
        directory = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return {
            "path": str(destination),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }

    def load(self, pin: Mapping[str, Any], event: str) -> dict[str, Any]:
        self._directory()
        path = Path(pin["path"])
        require(path.parent == self.root and path.resolve() == path, "receipt-location")
        info = path.lstat()
        require(
            stat.S_ISREG(info.st_mode)
            and info.st_uid == os.geteuid()
            and stat.S_IMODE(info.st_mode) == 0o444
            and type(pin["bytes"]) is int
            and info.st_size == pin["bytes"] <= 262144,
            "receipt-file",
        )
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == pin["sha256"], "receipt-hash")
        value = strict_json(raw)
        require(
            value["event"] == event
            and value["anchor"] == self.anchor.as_dict()
            and value["anchor_sha256"] == digest(self.anchor.as_dict()),
            "receipt-binding",
        )
        return value["data"]


def _integer(value: Any) -> int:
    require(
        type(value) is int
        or (isinstance(value, str) and re.fullmatch(r"[0-9]+", value)),
        "integer-observation",
    )
    return int(value)


def _drained(unit: Mapping[str, Any]) -> bool:
    return (
        unit.get("ActiveState") == "inactive"
        and unit.get("SubState") == "dead"
        and _integer(unit.get("MainPID")) == 0
        and unit.get("Job") in ("", "0", 0)
        and unit.get("members") == []
    )


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _recorded_clock(anchor: Anchor, value: Mapping[str, Any], phase: str) -> Clock:
    """Validate an immutable recorded time, without calling it a current clock."""
    require(set(value) == {"boot_id", "monotonic", "wall_ns"}, "recorded-clock-fields")
    recorded = Clock(**value)
    require(
        recorded.boot_id == anchor.boot_id
        and finite(recorded.monotonic)
        and type(recorded.wall_ns) is int
        and 0 <= recorded.monotonic - anchor.started_monotonic < ENDS[phase]
        and 0 <= recorded.wall_ns - anchor.started_wall_ns < ENDS[phase] * 10**9,
        "recorded-clock-window",
    )
    return recorded


def _single(log: RawLog, pin: Mapping[str, Any], event: str) -> dict[str, Any]:
    data = log.load(pin, event)
    require(
        [str(p) for p in log.root.glob(event + "-*.json")] == [pin["path"]],
        "nonunique-" + event,
    )
    return data


def role_started(
    log: RawLog, now: Clock, role: str, owner: Mapping[str, Any]
) -> dict[str, Any]:
    """A role records its own qualified systemd identity before doing work."""
    raw = log.record(
        "raw-role-start", {"clock": asdict(now), "role": role, "owner": dict(owner)}
    )
    require(role in ("dispatcher", "cleanup", "observer", "publisher"), "start-role")
    log.anchor.remaining("work" if role == "dispatcher" else role, now)
    require(
        type(owner["pid"]) is int
        and owner["pid"] == os.getpid()
        and type(owner["start_monotonic_us"]) is int
        and log.anchor.started_monotonic
        <= owner["start_monotonic_us"] / 1e6
        <= now.monotonic
        and re.fullmatch(r"[0-9a-f]{32}", owner["invocation_id"])
        and owner["unit"].startswith("edgeconnect-cpuqual-" + log.anchor.nonce + "-")
        and owner["unit"].endswith(".service")
        and isinstance(owner["cgroup"], str)
        and owner["cgroup"].startswith("/")
        and owner["cgroup"] != "/",
        "start-owner",
    )
    return log.record_once(
        role + "-started",
        {
            "raw": raw,
            "owner": dict(owner),
            "clock": asdict(now),
            "source_sha256": _source_sha256(),
        },
    )


def natural_exit(
    anchor: Anchor,
    now: Clock,
    owner: Mapping[str, Any],
    unit: Mapping[str, Any],
    phase: str,
) -> None:
    """Join actual terminal fields to the previously qualified start identity."""
    require(
        phase in ("dispatcher", "cleanup", "observer", "publisher"), "terminal-role"
    )
    anchor.remaining(
        "dispatcher_terminal"
        if phase == "dispatcher"
        else "audit"
        if phase == "publisher"
        else "publisher",
        now,
    )
    _terminal_observation(anchor, now, unit)
    _terminal_fields(
        anchor,
        owner,
        unit,
        "dispatcher_terminal" if phase == "dispatcher" else phase,
        natural=True,
    )


def _terminal_observation(
    anchor: Anchor, checked: Clock, unit: Mapping[str, Any]
) -> None:
    require(
        unit["observed_boot_id"] == anchor.boot_id
        and finite(unit["observed_monotonic"])
        and 0 <= checked.monotonic - unit["observed_monotonic"] <= 5,
        "terminal-observation-time",
    )


def _terminal_fields(
    anchor: Anchor,
    owner: Mapping[str, Any],
    unit: Mapping[str, Any],
    end_phase: str,
    *,
    natural: bool,
) -> None:
    require(
        unit["Id"] == owner["unit"]
        and unit["ControlGroup"] == owner["cgroup"]
        and unit["InvocationID"] in ("", owner["invocation_id"])
        and re.fullmatch(r"[0-9a-f]{32}", owner["invocation_id"])
        and type(owner["pid"]) is int
        and type(owner["start_monotonic_us"]) is int
        and _integer(unit["ExecMainPID"]) == owner["pid"] > 0
        and _integer(unit["ExecMainStartTimestampMonotonic"])
        == owner["start_monotonic_us"],
        "terminal-owner",
    )
    started = owner["start_monotonic_us"] / 1e6
    ended = _integer(unit["ExecMainExitTimestampMonotonic"]) / 1e6
    require(
        anchor.started_monotonic <= started <= ended <= unit["observed_monotonic"]
        and ended <= anchor.deadline(end_phase),
        "terminal-lifetime",
    )
    if natural:
        require(
            _drained(unit)
            and unit["Result"] == "success"
            and _integer(unit["ExecMainCode"]) == 1
            and _integer(unit["ExecMainStatus"]) == 0,
            "terminal-not-natural-success",
        )
    else:
        require(
            unit["ActiveState"] in {"inactive", "failed"}
            and unit["SubState"] in {"dead", "failed"}
            and _integer(unit["MainPID"]) == 0
            and unit["Job"] in ("", "0", 0)
            and unit["members"] == [],
            "dispatcher-barrier-not-drained",
        )


def acknowledge_arm(
    log: RawLog, now: Clock, facts: Mapping[str, Any]
) -> dict[str, Any]:
    raw = log.record("raw-arm", {"clock": asdict(now), "facts": dict(facts)})
    anchor = log.anchor
    anchor.remaining("work", now)
    for key in (
        "next_trigger_monotonic",
        "accuracy_seconds",
        "randomized_delay_seconds",
        "manager_slack_seconds",
        "observed_dispatch_slack_seconds",
        "cleanup_timeout_start_seconds",
        "cleanup_timeout_stop_seconds",
    ):
        require(finite(facts[key]) and facts[key] >= 0, "arm-timing-" + key)
    require(
        facts["active_waiting"] is True
        and facts["independent_cleanup"] is True
        and facts["ownership_validated"] is True,
        "cleanup-not-armed",
    )
    require(
        abs(facts["next_trigger_monotonic"] - anchor.deadline("work")) <= 0.001
        and facts["accuracy_seconds"] <= 1
        and facts["randomized_delay_seconds"] == 0
        and 0 < facts["cleanup_timeout_start_seconds"] <= 120
        and 0 < facts["cleanup_timeout_stop_seconds"] <= 5,
        "cleanup-arm-bounds",
    )
    slack = max(
        facts["manager_slack_seconds"], facts["observed_dispatch_slack_seconds"]
    )
    require(
        facts["next_trigger_monotonic"]
        + facts["accuracy_seconds"]
        + slack
        + facts["cleanup_timeout_start_seconds"]
        + facts["cleanup_timeout_stop_seconds"]
        <= anchor.effective_deadline("cleanup", now),
        "cleanup-reserve",
    )
    return log.record("cleanup-armed", {"raw": raw, "clock": asdict(now)})


def require_work(log: RawLog, now: Clock, armed: Mapping[str, Any]) -> float:
    log.record("raw-work-admission", {"clock": asdict(now), "armed": dict(armed)})
    receipt = log.load(armed, "cleanup-armed")
    log.load(receipt["raw"], "raw-arm")
    return log.anchor.effective_deadline("work", now)


def _dispatcher_started(log: RawLog, pin: Mapping[str, Any]) -> dict[str, Any]:
    value = _single(log, pin, "dispatcher-started")
    raw = log.load(value["raw"], "raw-role-start")
    require(
        value["source_sha256"] == _source_sha256()
        and raw["role"] == "dispatcher"
        and raw["owner"] == value["owner"]
        and raw["clock"] == value["clock"],
        "dispatcher-start-binding",
    )
    _recorded_clock(log.anchor, value["clock"], "work")
    return value


def _dispatcher_cases(cases: Mapping[str, Any], expected_cases: set[str]) -> None:
    require(
        1 < len(expected_cases) <= 32
        and "cleanup-and-retirement" in expected_cases
        and set(cases) == expected_cases
        and all(
            isinstance(v, dict)
            and v.get("status")
            == (
                "pending-independent-cleanup"
                if name == "cleanup-and-retirement"
                else "passed"
            )
            for name, v in cases.items()
        ),
        "dispatcher-case-set-or-status",
    )


def dispatcher_result(
    log: RawLog,
    now: Clock,
    started: Mapping[str, Any],
    cases: Mapping[str, Any],
    expected_cases: set[str],
) -> dict[str, Any]:
    raw = log.record(
        "raw-dispatcher-result",
        {
            "clock": asdict(now),
            "started": dict(started),
            "cases": dict(cases),
        },
    )
    log.anchor.remaining("work", now)
    owner = _dispatcher_started(log, started)
    require(
        owner["owner"]["pid"] == os.getpid()
        and owner["clock"]["monotonic"] <= now.monotonic,
        "dispatcher-producer-owner",
    )
    _dispatcher_cases(cases, expected_cases)
    return log.record_once(
        "dispatcher-result",
        {
            "started": dict(started),
            "cases": dict(cases),
            "expected_cases": sorted(expected_cases),
            "clock": asdict(now),
            "raw": raw,
            "source_sha256": _source_sha256(),
            "claims": FALSE_CLAIMS,
        },
    )


def _dispatcher_result(
    log: RawLog,
    pin: Mapping[str, Any],
    started: Mapping[str, Any],
    expected_cases: set[str],
) -> dict[str, Any]:
    value = _single(log, pin, "dispatcher-result")
    raw = log.load(value["raw"], "raw-dispatcher-result")
    require(
        value["source_sha256"] == _source_sha256()
        and value["started"] == started == raw["started"]
        and value["cases"] == raw["cases"]
        and value["clock"] == raw["clock"]
        and value["expected_cases"] == sorted(expected_cases)
        and value["claims"] == FALSE_CLAIMS
        and all(v is False for v in value["claims"].values()),
        "dispatcher-result-binding",
    )
    _recorded_clock(log.anchor, value["clock"], "work")
    _dispatcher_cases(value["cases"], expected_cases)
    return value


def dispatcher_complete(
    log: RawLog,
    now: Clock,
    started: Mapping[str, Any],
    result: Mapping[str, Any],
    terminal: Mapping[str, Any],
    expected_cases: set[str],
) -> dict[str, Any]:
    """Observer seals a *fresh* terminal observation once, no later than405."""
    raw = log.record(
        "raw-dispatcher-terminal",
        {
            "clock": asdict(now),
            "started": dict(started),
            "result": dict(result),
            "terminal": dict(terminal),
        },
    )
    log.anchor.remaining("dispatcher_terminal", now)
    start = _dispatcher_started(log, started)
    produced = _dispatcher_result(log, result, started, expected_cases)
    natural_exit(log.anchor, now, start["owner"], terminal, "dispatcher")
    require(
        start["clock"]["monotonic"]
        <= produced["clock"]["monotonic"]
        <= _integer(terminal["ExecMainExitTimestampMonotonic"]) / 1e6,
        "dispatcher-result-after-exit",
    )
    return log.record_once(
        "dispatcher-complete",
        {
            "started": dict(started),
            "result": dict(result),
            "terminal": dict(terminal),
            "clock": asdict(now),
            "expected_cases": sorted(expected_cases),
            "raw": raw,
            "source_sha256": _source_sha256(),
            "claims": FALSE_CLAIMS,
        },
    )


def checked_dispatcher_complete(log: RawLog, pin: Mapping[str, Any]) -> dict[str, Any]:
    """Read immutable checked history; never present the old clock as current."""
    value = _single(log, pin, "dispatcher-complete")
    raw = log.load(value["raw"], "raw-dispatcher-terminal")
    require(
        value["source_sha256"] == _source_sha256()
        and all(value[k] == raw[k] for k in ("started", "result", "terminal", "clock"))
        and value["claims"] == FALSE_CLAIMS
        and all(v is False for v in value["claims"].values()),
        "dispatcher-complete-binding",
    )
    recorded = _recorded_clock(log.anchor, value["clock"], "dispatcher_terminal")
    start = _dispatcher_started(log, value["started"])
    result = _dispatcher_result(
        log, value["result"], value["started"], set(value["expected_cases"])
    )
    _terminal_observation(log.anchor, recorded, value["terminal"])
    _terminal_fields(
        log.anchor,
        start["owner"],
        value["terminal"],
        "dispatcher_terminal",
        natural=True,
    )
    require(
        start["clock"]["monotonic"]
        <= result["clock"]["monotonic"]
        <= _integer(value["terminal"]["ExecMainExitTimestampMonotonic"]) / 1e6,
        "dispatcher-result-after-exit",
    )
    return {**value, "cases": result["cases"]}


def checked_dispatcher_barrier(log: RawLog, facts: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only proof of safe death by405; forced failure is not a pass."""
    require(set(facts) == {"started", "unit", "clock"}, "dispatcher-barrier-fields")
    started = _dispatcher_started(log, facts["started"])
    recorded = _recorded_clock(log.anchor, facts["clock"], "dispatcher_terminal")
    _terminal_observation(log.anchor, recorded, facts["unit"])
    # Cleanup may establish safe death after a forced failure; such a barrier
    # never substitutes for the separate natural dispatcher-complete proof.
    _terminal_fields(
        log.anchor,
        started["owner"],
        facts["unit"],
        "dispatcher_terminal",
        natural=False,
    )
    return dict(facts)


def cleanup_complete(
    log: RawLog,
    now: Clock,
    facts: Mapping[str, Any],
    expected_workloads: set[str],
) -> dict[str, Any]:
    raw = log.record("raw-cleanup", {"clock": asdict(now), "facts": dict(facts)})
    log.anchor.remaining("cleanup", now)
    checked_dispatcher_barrier(log, facts["dispatcher_barrier"])
    require(
        facts["dispatcher_barrier"]["clock"]["monotonic"] <= now.monotonic,
        "cleanup-before-dispatcher-barrier",
    )
    require(
        expected_workloads
        and set(facts["units"]) == expected_workloads
        and all(
            u.get("Id") == name and _drained(u) for name, u in facts["units"].items()
        )
        and facts["jobs"] == []
        and facts["members"] == []
        and facts["links"] == []
        and facts["leftovers"] == [],
        "cleanup-incomplete",
    )
    return log.record_once(
        "cleanup-complete",
        {
            "raw": raw,
            "clock": asdict(now),
            "expected_workloads": sorted(expected_workloads),
            "source_sha256": _source_sha256(),
        },
    )


def _checked_cleanup(log: RawLog, pin: Mapping[str, Any]) -> dict[str, Any]:
    value = _single(log, pin, "cleanup-complete")
    raw = log.load(value["raw"], "raw-cleanup")
    require(
        value["source_sha256"] == _source_sha256() and raw["clock"] == value["clock"],
        "cleanup-binding",
    )
    recorded = _recorded_clock(log.anchor, value["clock"], "cleanup")
    facts = raw["facts"]
    checked_dispatcher_barrier(log, facts["dispatcher_barrier"])
    require(
        facts["dispatcher_barrier"]["clock"]["monotonic"] <= recorded.monotonic
        and value["expected_workloads"]
        and set(facts["units"]) == set(value["expected_workloads"])
        and all(
            u.get("Id") == name and _drained(u) for name, u in facts["units"].items()
        )
        and all(facts[k] == [] for k in ("jobs", "members", "links", "leftovers")),
        "cleanup-incomplete",
    )
    return {**value, "facts": facts}


def run_cleanup(
    log: RawLog,
    clock: Callable[[], Clock],
    expected_workloads: set[str],
    action: Callable[[str, float], Mapping[str, Any]],
) -> dict[str, Any]:
    """Callback is independently launched and restricted to dummy workloads."""
    now = clock()
    log.record("cleanup-intent", {"clock": asdict(now)})
    deadline = log.anchor.effective_deadline("cleanup", now)
    try:
        facts = action("cleanup-workloads", deadline)
    except BaseException as exc:
        log.record("cleanup-action-failed", {"error_type": type(exc).__name__})
        raise
    return cleanup_complete(log, clock(), facts, expected_workloads)


def observer_candidate(
    log: RawLog,
    now: Clock,
    cleanup: Mapping[str, Any],
    dispatcher_complete_pin: Mapping[str, Any],
    expected_cases: set[str],
    r3_gate: Mapping[str, Any],
) -> dict[str, Any]:
    raw = log.record(
        "raw-observer",
        {
            "clock": asdict(now),
            "dispatcher": dict(dispatcher_complete_pin),
            "r3_gate": dict(r3_gate),
        },
    )
    log.anchor.remaining("observer", now)
    dispatched = checked_dispatcher_complete(log, dispatcher_complete_pin)
    require(
        dispatched["expected_cases"] == sorted(expected_cases),
        "observer-dispatcher-case-set",
    )
    cleaned = _checked_cleanup(log, cleanup)
    require(
        cleaned["facts"]["dispatcher_barrier"]["started"] == dispatched["started"],
        "cleanup-dispatcher-owner",
    )
    require(
        r3_gate.get("status") == "passed"
        and r3_gate.get("owners_unchanged") is True
        and r3_gate.get("productive") is True
        and finite(r3_gate.get("observed_monotonic"))
        and cleaned["clock"]["monotonic"]
        <= r3_gate["observed_monotonic"]
        <= now.monotonic,
        "r3-gate",
    )
    return log.record(
        "observer-candidate",
        {
            "raw": raw,
            "cleanup": dict(cleanup),
            "dispatcher": dict(dispatcher_complete_pin),
            "expected_cases": sorted(expected_cases),
            "cases": {
                **dispatched["cases"],
                "cleanup-and-retirement": {
                    "status": "passed",
                    "cleanup": dict(cleanup),
                },
            },
            "claims": FALSE_CLAIMS,
        },
    )


def checked_observer_candidate(log: RawLog, pin: Mapping[str, Any]) -> dict[str, Any]:
    value = log.load(pin, "observer-candidate")
    require(
        value["claims"] == FALSE_CLAIMS
        and all(v is False for v in value["claims"].values()),
        "candidate-claims",
    )
    raw = log.load(value["raw"], "raw-observer")
    dispatched = checked_dispatcher_complete(log, value["dispatcher"])
    cleaned = _checked_cleanup(log, value["cleanup"])
    recorded = _recorded_clock(log.anchor, raw["clock"], "observer")
    require(
        value["dispatcher"] == raw["dispatcher"]
        and dispatched["expected_cases"] == value["expected_cases"]
        and cleaned["facts"]["dispatcher_barrier"]["started"] == dispatched["started"]
        and value["cases"]
        == {
            **dispatched["cases"],
            "cleanup-and-retirement": {"status": "passed", "cleanup": value["cleanup"]},
        }
        and raw["r3_gate"]["status"] == "passed"
        and raw["r3_gate"]["owners_unchanged"] is True
        and raw["r3_gate"]["productive"] is True,
        "candidate-evidence",
    )
    require(
        finite(raw["r3_gate"]["observed_monotonic"])
        and cleaned["clock"]["monotonic"]
        <= raw["r3_gate"]["observed_monotonic"]
        <= recorded.monotonic,
        "r3-gate",
    )
    return value


def publish(
    log: RawLog,
    clock: Callable[[], Clock],
    candidate: Mapping[str, Any],
    observer_started: Mapping[str, Any],
    observer_terminal: Mapping[str, Any],
    cleanup_started: Mapping[str, Any],
    cleanup_terminal: Mapping[str, Any],
    retire: Callable[[str, float], Mapping[str, Any]],
) -> dict[str, Any]:
    now = clock()
    raw = log.record(
        "raw-publisher",
        {
            "clock": asdict(now),
            "observer_started": dict(observer_started),
            "observer_terminal": dict(observer_terminal),
            "cleanup_started": dict(cleanup_started),
            "cleanup_terminal": dict(cleanup_terminal),
        },
    )
    checked_observer_candidate(log, candidate)
    observer_owner = log.load(observer_started, "observer-started")["owner"]
    cleanup_owner = log.load(cleanup_started, "cleanup-started")["owner"]
    require(
        observer_owner["unit"] != cleanup_owner["unit"]
        and observer_owner["pid"] != cleanup_owner["pid"]
        and observer_owner["invocation_id"] != cleanup_owner["invocation_id"],
        "independent-observer-and-cleanup",
    )
    natural_exit(log.anchor, now, observer_owner, observer_terminal, "observer")
    natural_exit(log.anchor, now, cleanup_owner, cleanup_terminal, "cleanup")
    deadline = log.anchor.effective_deadline("publisher", now)
    # Only the separately fenced finalizer may consume this purpose. Namespace
    # membership alone never gives the workload dispatcher retirement rights.
    log.record("retirement-intent", {"candidate": dict(candidate), "raw": raw})
    try:
        facts = retire("retire-cleanup-watchdog", deadline)
    except BaseException as exc:
        log.record("retirement-failed", {"error_type": type(exc).__name__})
        raise
    retirement = log.record("raw-retirement", dict(facts))
    log.anchor.remaining("publisher", clock())
    require(
        facts.get("jobs") == []
        and facts.get("members") == []
        and facts.get("links") == []
        and facts.get("leftovers") == [],
        "retirement-incomplete",
    )
    return log.record(
        "published-candidate",
        {
            "candidate": dict(candidate),
            "retirement": retirement,
            "status": "pending_external_terminal_audit",
            "claims": FALSE_CLAIMS,
        },
    )


def audit(
    log: RawLog,
    now: Clock,
    published: Mapping[str, Any],
    publisher_started: Mapping[str, Any],
    publisher_terminal: Mapping[str, Any],
) -> dict[str, Any]:
    raw = log.record(
        "raw-audit",
        {
            "clock": asdict(now),
            "started": dict(publisher_started),
            "terminal": dict(publisher_terminal),
        },
    )
    log.anchor.remaining("audit", now)
    value = log.load(published, "published-candidate")
    require(
        value["status"] == "pending_external_terminal_audit"
        and value["claims"] == FALSE_CLAIMS
        and all(v is False for v in value["claims"].values()),
        "published-candidate-schema",
    )
    checked_observer_candidate(log, value["candidate"])
    retirement = log.load(value["retirement"], "raw-retirement")
    require(
        all(retirement.get(k) == [] for k in ("jobs", "members", "links", "leftovers")),
        "retirement-incomplete",
    )
    publisher_owner = log.load(publisher_started, "publisher-started")["owner"]
    natural_exit(log.anchor, now, publisher_owner, publisher_terminal, "publisher")
    return log.record(
        "cpu-scope-result",
        {
            "status": "passed_cpu_scope",
            "published": dict(published),
            "raw": raw,
            "claims": FALSE_CLAIMS,
        },
    )


def _protected_file(path: Path, uid: int, *, private: bool = False) -> bytes:
    require(path.is_absolute() and path.resolve() == path, "input-canonical")
    info = path.lstat()
    require(
        stat.S_ISREG(info.st_mode)
        and info.st_uid == uid
        and info.st_size <= MAX_SOURCE_BYTES
        and (
            stat.S_IMODE(info.st_mode) == 0o600
            if private
            else not stat.S_IMODE(info.st_mode) & 0o222
        ),
        "input-protection",
    )
    raw = path.read_bytes()
    after = path.lstat()
    require(
        (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        "input-raced",
    )
    return raw


def load_payload_authorization(
    path: Path, expected_sha256: str | None = None, *, expected_uid: int = 0
) -> tuple[dict[str, Any], Anchor]:
    """Read a companion created by the explicitly pinned plan preparer.

    There is deliberately no inline authorization hash in fixed unit text:
    that would create a unit -> authorization -> plan -> unit hash cycle.
    The CLI always requires root ownership. The UID argument is a local-test
    seam, never a command-line option or a production ownership fallback.
    """
    parent = path.parent.lstat()
    require(
        path.parent.resolve() == path.parent
        and stat.S_ISDIR(parent.st_mode)
        and parent.st_uid == expected_uid
        and stat.S_IMODE(parent.st_mode) == 0o700,
        "authorization-directory",
    )
    raw = _protected_file(path, expected_uid, private=True)
    if expected_sha256 is not None:
        require(hashlib.sha256(raw).hexdigest() == expected_sha256, "authorization-sha")
    auth = strict_json(raw)
    require(
        set(auth)
        == {
            "format",
            "schema_version",
            "plan_path",
            "plan_sha256",
            "anchor_path",
            "anchor_sha256",
            "source_pins",
            "role",
            "unit",
            "mode",
            "payload",
        }
        and auth["format"] == "strength-freshness-cpu-authorization-v1"
        and type(auth["schema_version"]) is int
        and auth["schema_version"] == 1
        and auth["role"] == "workload",
        "payload-authorization-schema",
    )
    documents = {}
    for name in ("plan", "anchor"):
        data = _protected_file(Path(auth[name + "_path"]), expected_uid)
        require(
            hashlib.sha256(data).hexdigest() == auth[name + "_sha256"],
            "payload-" + name + "-sha",
        )
        documents[name] = strict_json(data)
    anchor = Anchor.from_dict(documents["anchor"])
    plan = documents["plan"]
    require(
        plan["format"] == "strength-freshness-cpu-plan-v1"
        and type(plan["schema_version"]) is int
        and plan["schema_version"] == 1
        and plan["attempt_id"] == anchor.attempt_id
        and plan["nonce"] == anchor.nonce
        and plan["boot_id"] == anchor.boot_id
        and auth["plan_sha256"] == anchor.plan_sha256,
        "payload-plan-binding",
    )
    spec = plan["units"][auth["unit"]]
    require(
        auth["unit"].startswith("edgeconnect-cpuqual-" + anchor.nonce + "-")
        and auth["unit"].endswith(".service")
        and spec["role"] == "workload"
        and spec["mode"] == auth["mode"]
        and spec["payload"] == auth["payload"]
        and auth["mode"]
        in {"sleep", "term_tree", "fast_receipt", "lock_holder", "lock_contender"},
        "payload-unit-binding",
    )
    payload = auth["payload"]
    require(
        set(payload) == {"seconds", "output_dir", "lock_path"}
        and finite(payload["seconds"])
        and 0 <= payload["seconds"] <= 30,
        "payload-bounds",
    )
    scratch = Path("/run/edgeconnect-cpuqual-" + anchor.nonce)
    require(
        Path(payload["output_dir"]) == scratch / "payloads" / auth["unit"]
        and payload["lock_path"] in (None, str(scratch / "qualification.lock")),
        "payload-paths",
    )
    require(
        (payload["lock_path"] is not None) == auth["mode"].startswith("lock_"),
        "payload-lock-mode",
    )
    own = str(Path(__file__).resolve())
    require(
        isinstance(auth["source_pins"], list)
        and 1 <= len(auth["source_pins"]) <= MAX_SOURCE_PINS
        and auth["source_pins"] == plan["source_pins"]
        and len({p["path"] for p in auth["source_pins"]}) == len(auth["source_pins"])
        and sum(p["path"] == own for p in auth["source_pins"]) == 1,
        "payload-source-inventory",
    )
    for pin in auth["source_pins"]:
        data = _protected_file(Path(pin["path"]), expected_uid)
        require(
            type(pin["bytes"]) is int
            and len(data) == pin["bytes"]
            and hashlib.sha256(data).hexdigest() == pin["sha256"],
            "payload-source-pin",
        )
    return auth, anchor


def actual_clock() -> Clock:
    return Clock(
        Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        time.monotonic(),
        time.time_ns(),
    )


def _self_identity(unit: str, invocation: str) -> dict[str, Any]:
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    cgroups = Path("/proc/self/cgroup").read_text().splitlines()
    require(len(cgroups) == 1 and cgroups[0].startswith("0::"), "payload-cgroup-v2")
    return {
        "unit": unit,
        "invocation_id": invocation,
        "pid": os.getpid(),
        "start_ticks": int(fields[19]),
        "cgroup": cgroups[0][3:],
    }


def _wait_until(end: float, anchor: Anchor, clock: Callable[[], Clock]) -> None:
    while True:
        now = clock()
        anchor.remaining("work", now)
        if now.monotonic >= end:
            return
        time.sleep(min(0.05, end - now.monotonic))


def run_payload(
    auth: Mapping[str, Any],
    anchor: Anchor,
    log: RawLog,
    clock: Callable[[], Clock] = actual_clock,
) -> dict[str, Any]:
    """Tiny sacrificial work only; no cleanup/finalizer/systemctl capability."""
    now = clock()
    invocation = os.environ.get("INVOCATION_ID", "")
    require(re.fullmatch(r"[0-9a-f]{32}", invocation), "payload-invocation")
    identity = _self_identity(auth["unit"], invocation)
    started = log.record(
        "payload-started",
        {
            "clock": asdict(now),
            "owner": identity,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "plan_sha256": anchor.plan_sha256,
            "anchor_sha256": digest(anchor.as_dict()),
            "role": "workload",
            "mode": auth["mode"],
        },
    )
    payload, mode = auth["payload"], auth["mode"]
    require(
        finite(payload["seconds"]) and 0 <= payload["seconds"] <= 30, "payload-duration"
    )
    end = now.monotonic + payload["seconds"]
    require(end < anchor.effective_deadline("work", now), "payload-work-deadline")
    child, lock_fd, old_handler = None, None, None
    stop_requested = False

    def graceful_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    try:
        if mode == "term_tree":
            require(
                signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL,
                "payload-child-handler",
            )
            old_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
            child = os.fork()
            if child == 0:
                try:
                    log.record(
                        "payload-descendant",
                        {"pid": os.getpid(), "parent_pid": os.getppid()},
                    )
                    _wait_until(end, anchor, clock)
                except BaseException:
                    os._exit(1)
                os._exit(0)
        elif mode in ("lock_holder", "lock_contender"):
            if mode == "lock_holder":
                old_handler = signal.signal(signal.SIGTERM, graceful_stop)
            path = Path(payload["lock_path"])
            require(path.is_absolute() and path.resolve() == path, "payload-lock-path")
            lock_fd = os.open(
                path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600
            )
            info = os.fstat(lock_fd)
            require(
                stat.S_ISREG(info.st_mode)
                and info.st_uid == os.geteuid()
                and stat.S_IMODE(info.st_mode) == 0o600,
                "payload-lock-file",
            )
            acquired = False
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                pass
            log.record(
                "payload-lock",
                {
                    "acquired": acquired,
                    "device": info.st_dev,
                    "inode": info.st_ino,
                    "pid": os.getpid(),
                },
            )
            require(acquired == (mode == "lock_holder"), "payload-lock-expectation")
        else:
            require(mode in ("sleep", "fast_receipt"), "payload-mode")
        if mode == "lock_holder":
            while not stop_requested:
                current = clock()
                anchor.remaining("work", current)
                if current.monotonic >= end:
                    break
                time.sleep(min(0.05, end - current.monotonic))
        elif mode != "fast_receipt":
            _wait_until(end, anchor, clock)
        if child is not None:
            while True:
                waited, status = os.waitpid(child, os.WNOHANG)
                if waited:
                    child = None
                    require(
                        os.waitstatus_to_exitcode(status) == 0, "payload-child-exit"
                    )
                    break
                anchor.remaining("work", clock())
                time.sleep(0.01)
        anchor.remaining("work", clock())
        if stop_requested:
            require(mode == "lock_holder" and lock_fd is not None, "payload-stop-mode")
            assert lock_fd is not None
            os.close(lock_fd)
            lock_fd = None
            log.record(
                "payload-stopped",
                {
                    "started": started,
                    "signal": "SIGTERM",
                    "lock_released": True,
                    "clock": asdict(clock()),
                },
            )
        return log.record(
            "payload-result",
            {
                "started": started,
                "status": "stopped" if stop_requested else "completed",
                "claims": FALSE_CLAIMS,
            },
        )
    except BaseException as exc:
        log.record("payload-failed", {"error_type": type(exc).__name__})
        raise
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGTERM, old_handler)
        if lock_fd is not None:
            os.close(lock_fd)
        if child is not None:
            # This is our unreaped direct fork child; it cannot have a reused PID.
            os.kill(child, signal.SIGKILL)
            cleanup_end = min(
                clock().monotonic + 1,
                anchor.effective_deadline("cleanup", clock()),
            )
            while os.waitpid(child, os.WNOHANG)[0] == 0:
                require(
                    clock().monotonic < cleanup_end, "payload-child-cleanup-timeout"
                )
                time.sleep(0.01)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    auth, anchor = load_payload_authorization(args.authorization, args.sha256)
    log = RawLog(Path(auth["payload"]["output_dir"]), anchor)
    run_payload(auth, anchor, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
