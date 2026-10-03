"""Unarmed external CPU lifecycle primitives; no installer or execution grant.

All syscall effects are behind an injected kernel. The real Linux adapter
deliberately refuses execution admission until an independently reviewed caller
and session-survival verifier exists. A local CPU receipt cannot unlock it.
"""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import select
import signal
import sys
import time
from typing import Callable, Protocol

SECOND = 10**9
ROLES = {"supervisor", "operator", "guardian"}
CAPTURE_LIMITS = {
    "cpu_affinity_count": 1,
    "nice": 19,
    "address_space_bytes": 512 * 2**20,
    "open_files": 128,
    "file_bytes": 32 * 2**20,
    "stdout_stderr_bytes": 2**20,
    "owned_children": 32,
}
ENV = {
    "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
    "LANG": "C",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


class OuterRefusal(RuntimeError):
    """Fixed non-private reason only."""


def require(ok: object, reason: str) -> None:
    if not ok:
        raise OuterRefusal(reason)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def checksum(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True)
class Clock:
    boot_id: str
    monotonic_ns: int
    wall_ns: int

    def validate(self) -> None:
        require(
            bool(self.boot_id)
            and type(self.monotonic_ns) is int
            and self.monotonic_ns >= 0
            and type(self.wall_ns) is int
            and self.wall_ns > 0,
            "clock-shape",
        )


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_ticks: int
    ppid: int
    pgid: int
    sid: int
    uid: int
    boot_id: str
    cgroup: str
    pid_namespace_inode: int

    def validate(self) -> None:
        require(
            all(
                type(x) is int and x > 0
                for x in (
                    self.pid,
                    self.start_ticks,
                    self.pgid,
                    self.sid,
                    self.pid_namespace_inode,
                )
            )
            and type(self.ppid) is int
            and self.ppid >= 0
            and type(self.uid) is int
            and self.uid >= 0
            and bool(self.boot_id)
            and self.cgroup.startswith("/"),
            "process-shape",
        )


@dataclass(frozen=True)
class PhaseBudget:
    phase: str
    start: Clock
    phase_start: Clock
    work_ns: int
    cleanup_ns: int
    gate_ns: int
    work_wall_ns: int
    cleanup_wall_ns: int
    gate_wall_ns: int

    @classmethod
    def before(cls, start: Clock) -> PhaseBudget:
        start.validate()
        return cls(
            "before",
            start,
            start,
            *(start.monotonic_ns + x * SECOND for x in (105, 115, 120)),
            *(start.wall_ns + x * SECOND for x in (105, 115, 120)),
        )

    @classmethod
    def after(cls, anchor: Clock, launch: Clock) -> PhaseBudget:
        anchor.validate()
        launch.validate()
        require(
            launch.boot_id == anchor.boot_id
            and launch.monotonic_ns >= anchor.monotonic_ns
            and launch.wall_ns >= anchor.wall_ns,
            "after-clock",
        )
        work = min(
            launch.monotonic_ns + 105 * SECOND, anchor.monotonic_ns + 555 * SECOND
        )
        wall = min(launch.wall_ns + 105 * SECOND, anchor.wall_ns + 555 * SECOND)
        require(
            launch.monotonic_ns < work and launch.wall_ns < wall, "after-no-work-window"
        )
        return cls(
            "after",
            anchor,
            launch,
            work,
            min(work + 5 * SECOND, anchor.monotonic_ns + 560 * SECOND),
            min(work + 10 * SECOND, anchor.monotonic_ns + 565 * SECOND),
            wall,
            min(wall + 5 * SECOND, anchor.wall_ns + 560 * SECOND),
            min(wall + 10 * SECOND, anchor.wall_ns + 565 * SECOND),
        )

    def validate(self) -> None:
        require(self.phase in {"before", "after"}, "phase")
        expected = (
            self.before(self.start)
            if self.phase == "before"
            else self.after(self.start, self.phase_start)
        )
        require(self == expected, "altered-phase-budget")

    def compact(self) -> dict:
        return {
            stage: {
                "monotonic_ns": getattr(self, stage + "_ns"),
                "wall_ns": getattr(self, stage + "_wall_ns"),
            }
            for stage in ("work", "cleanup", "gate")
        }

    def remaining_ns(self, stage: str, now: Clock) -> int:
        now.validate()
        require(stage in {"work", "cleanup", "gate"}, "budget-stage")
        require(
            now.boot_id == self.start.boot_id
            and now.monotonic_ns >= self.start.monotonic_ns
            and now.wall_ns >= self.start.wall_ns,
            "budget-clock",
        )
        return min(
            getattr(self, stage + "_ns") - now.monotonic_ns,
            getattr(self, stage + "_wall_ns") - now.wall_ns,
        )


@dataclass(frozen=True)
class CaptureSpec:
    phase: str
    python: str
    control_root: str
    launch_path: str
    launch_sha256: str
    deadline_monotonic_ns: int

    def argv(self) -> tuple[str, ...]:
        for path in (self.python, self.control_root, self.launch_path):
            require(
                isinstance(path, str)
                and Path(path).is_absolute()
                and str(Path(path)) == path
                and ".." not in Path(path).parts,
                "capture-path",
            )
        require(
            Path(self.python).name == "python"
            and self.phase in {"before", "after"}
            and checksum(self.launch_sha256),
            "capture-contract",
        )
        require(
            type(self.deadline_monotonic_ns) is int and self.deadline_monotonic_ns > 0,
            "capture-deadline",
        )
        return (
            self.python,
            "-S",
            "-E",
            "-B",
            "-m",
            "scripts.strength_freshness_cpu_capture_cli",
            "--launch",
            self.launch_path,
            "--launch-sha256",
            self.launch_sha256,
            "--deadline-monotonic-ns",
            str(self.deadline_monotonic_ns),
        )

    def contract(self) -> dict:
        return {
            "argv": self.argv(),
            "cwd": self.control_root,
            "env": dict(ENV),
            "resources": dict(CAPTURE_LIMITS),
        }


@dataclass(frozen=True)
class RoleSpec:
    """Only the two fixed composition entrypoints; phase data is not argv."""

    phase: str
    role: str
    python: str
    control_root: str
    authorization_path: str
    authorization_sha256: str
    deadline_monotonic_ns: int
    frame: bytes
    qualified_sources: tuple[str, str]

    @property
    def process_deadline_ns(self) -> int:
        if self.role == "guardian":
            doc = json.loads(self.frame)
            deadline = doc["budget"]["cleanup"]["monotonic_ns"]
            require(
                type(deadline) is int
                and self.deadline_monotonic_ns
                <= deadline
                <= self.deadline_monotonic_ns + 10 * SECOND,
                "guardian-terminal-bound",
            )
            return deadline
        return self.deadline_monotonic_ns

    def argv(self) -> tuple[str, ...]:
        require(self.role in {"operator", "guardian"}, "fixed-outer-entrypoint")
        require(
            self.phase
            in ({"session"} if self.role == "operator" else {"before", "after"}),
            "fixed-role-phase",
        )
        for path in (self.python, self.control_root, self.authorization_path):
            require(
                isinstance(path, str)
                and Path(path).is_absolute()
                and str(Path(path)) == path
                and ".." not in Path(path).parts,
                "role-path",
            )
        require(
            Path(self.python).name == "python" and checksum(self.authorization_sha256),
            "role-contract",
        )
        require(
            type(self.frame) is bytes
            and 0 < len(self.frame) <= 65536
            and type(self.deadline_monotonic_ns) is int
            and self.deadline_monotonic_ns > 0,
            "role-frame-bound",
        )
        require(
            len(self.qualified_sources) == 2
            and all(checksum(v) for v in self.qualified_sources),
            "role-qualified-sources",
        )
        return (
            self.python,
            "-S",
            "-E",
            "-B",
            "-m",
            "scripts.strength_freshness_cpu_outer_runtime",
            "--role",
            self.role,
            "--authorization",
            self.authorization_path,
            "--sha256",
            self.authorization_sha256,
        )

    def contract(self) -> dict:
        argv = list(self.argv())
        require(
            argv[-2] == "--sha256" and argv[-1] == self.authorization_sha256,
            "role-authority-argument",
        )
        argv[-1] = "@approved-outer-intent-sha256"
        return {
            "format": "strength-freshness-fixed-role-contract-v1",
            "argv_template": argv,
            "cwd": self.control_root,
            "env": dict(ENV),
            "qualified_sources": {
                "primitive": self.qualified_sources[0],
                "runtime": self.qualified_sources[1],
            },
            "resources": {
                **CAPTURE_LIMITS,
                "address_space_hard_bytes": 16 * 2**30
                if self.role == "operator"
                else CAPTURE_LIMITS["address_space_bytes"],
                "inherited_frame_fd": 127,
                "inherited_frame_max_bytes": 65536,
            },
        }


@dataclass(frozen=True)
class BudgetWindow:
    """Supervisor's one bounded session; never a capture budget or renewal."""

    start: Clock
    dummy_start: Clock | None = None

    phase = "session"

    @property
    def phase_start(self) -> Clock:
        return self.start

    def validate(self) -> None:
        self.start.validate()
        if self.dummy_start is not None:
            self.dummy_start.validate()
            require(
                self.dummy_start.boot_id == self.start.boot_id
                and 0
                <= self.dummy_start.monotonic_ns - self.start.monotonic_ns
                <= 118 * SECOND
                and 0 <= self.dummy_start.wall_ns - self.start.wall_ns <= 118 * SECOND,
                "dummy-handoff-window",
            )

    def compact(self) -> dict:
        self.validate()
        if self.dummy_start is None:
            return {
                name: {
                    axis: getattr(self.start, axis) + seconds * SECOND
                    for axis in ("monotonic_ns", "wall_ns")
                }
                for name, seconds in (("work", 118), ("cleanup", 119), ("gate", 120))
            }
        return {
            name: {
                axis: min(
                    getattr(self.dummy_start, axis) + seconds * SECOND,
                    getattr(self.start, axis) + total * SECOND,
                )
                for axis in ("monotonic_ns", "wall_ns")
            }
            for name, seconds, total in (
                ("work", 598, 718),
                ("cleanup", 599, 719),
                ("gate", 600, 720),
            )
        }

    @property
    def work_ns(self) -> int:
        return self.compact()["work"]["monotonic_ns"]

    def remaining_ns(self, stage: str, now: Clock) -> int:
        now.validate()
        require(stage in {"work", "cleanup", "gate"}, "budget-stage")
        require(
            now.boot_id == self.start.boot_id
            and now.monotonic_ns >= self.start.monotonic_ns
            and now.wall_ns >= self.start.wall_ns,
            "budget-clock",
        )
        return min(
            self.compact()[stage][axis] - getattr(now, axis)
            for axis in ("monotonic_ns", "wall_ns")
        )

    def handoff(self, dummy_start: Clock, observed: Clock) -> BudgetWindow:
        require(
            self.dummy_start is None and self.remaining_ns("work", observed) > 0,
            "late-or-repeated-handoff",
        )
        require(
            dummy_start.monotonic_ns <= observed.monotonic_ns
            and dummy_start.wall_ns <= observed.wall_ns,
            "future-dummy-start",
        )
        result = BudgetWindow(self.start, dummy_start)
        result.validate()
        return result


def guardian_contract_sha256(
    python: str,
    control_root: str,
    authorization_path: str,
    qualified_sources: tuple[str, str],
) -> str:
    spec = RoleSpec(
        "before",
        "guardian",
        python,
        control_root,
        authorization_path,
        "0" * 64,
        1,
        b"{}",
        qualified_sources,
    )
    return digest(spec.contract())


HELPER_MODES = {"prepare-install", "inspect-cleanup", "audit", "prearm-cleanup"}
ACK_FORMAT = "strength-freshness-dummy-start-ack-v1"


def dummy_start_ack(
    *,
    outer_intent_sha256: str,
    nonce: str,
    start_pin: dict,
    before_execution_pin: dict,
    enclosing: dict,
) -> dict:
    """Pure identity/content binding only; this does not attest publication time."""
    require(
        checksum(outer_intent_sha256)
        and isinstance(nonce, str)
        and re.fullmatch(r"[0-9a-f]{32}", nonce) is not None,
        "dummy-ack-authority",
    )
    pins = []
    for value in (start_pin, before_execution_pin):
        require(
            isinstance(value, dict) and set(value) == {"path", "sha256", "bytes"},
            "dummy-ack-pin-fields",
        )
        path = value["path"]
        require(
            isinstance(path, str)
            and Path(path).is_absolute()
            and str(Path(path)) == path
            and ".." not in Path(path).parts
            and checksum(value["sha256"])
            and type(value["bytes"]) is int
            and 0 < value["bytes"] <= 2**20,
            "dummy-ack-pin",
        )
        pins.append(dict(value))
    inputs = Path(pins[0]["path"]).parent
    require(
        Path(pins[0]["path"]) == inputs / "dummy-start.json"
        and Path(pins[1]["path"]) == inputs / "before-execution.json",
        "dummy-ack-paths",
    )
    require(
        isinstance(enclosing, dict) and set(enclosing) == {"operator", "supervisor"},
        "dummy-ack-enclosing",
    )
    owners = []
    for role in ("operator", "supervisor"):
        try:
            owner = ProcessIdentity(**enclosing[role])
        except (TypeError, KeyError):
            raise OuterRefusal("dummy-ack-owner-fields") from None
        owner.validate()
        owners.append(owner)
    op, sup = owners
    require(
        op.uid == sup.uid == 0
        and op.pid != sup.pid
        and op.ppid == sup.pid
        and sup.start_ticks <= op.start_ticks
        and op.boot_id == sup.boot_id
        and op.pid_namespace_inode == sup.pid_namespace_inode
        and op.cgroup == sup.cgroup,
        "dummy-ack-owners",
    )
    return {
        "format": ACK_FORMAT,
        "schema_version": 1,
        "outer_intent_sha256": outer_intent_sha256,
        "nonce": nonce,
        "start_pin": pins[0],
        "before_execution_pin": pins[1],
        "enclosing": {"operator": asdict(op), "supervisor": asdict(sup)},
    }


def validate_dummy_start_ack(value: dict, **expected) -> dict:
    canonical = dummy_start_ack(**expected)
    require(value == canonical, "dummy-ack-binding")
    return canonical


@dataclass(frozen=True)
class HelperSpec:
    mode: str
    python: str
    control_root: str
    authorization_path: str
    authorization_sha256: str
    deadline_monotonic_ns: int
    frame: bytes

    phase = "helper"

    def argv(self) -> tuple[str, ...]:
        require(self.mode in HELPER_MODES, "helper-operation")
        # Reuse the closed role path/frame validation, never a caller argv.
        RoleSpec(
            "session",
            "operator",
            self.python,
            self.control_root,
            self.authorization_path,
            self.authorization_sha256,
            self.deadline_monotonic_ns,
            self.frame,
            ("0" * 64, "0" * 64),
        ).argv()
        return (
            self.python,
            "-s",
            "-B",
            str(
                Path(self.control_root)
                / "scripts/strength_freshness_cpu_install_helper.py"
            ),
            "--operation",
            self.mode,
            "--authorization",
            self.authorization_path,
            "--sha256",
            self.authorization_sha256,
        )

    def contract(self) -> dict:
        return {
            "argv": self.argv(),
            "cwd": self.control_root,
            "env": {**ENV, "PYTHONPATH": self.control_root},
            "resources": {
                **CAPTURE_LIMITS,
                "address_space_bytes": 16 * 2**30,
                "address_space_hard_bytes": 16 * 2**30,
                "stdout_stderr_bytes": 8 * 2**20,
                "inherited_frame_fd": 127,
                "inherited_frame_max_bytes": 65536,
            },
        }


@dataclass(frozen=True)
class HelperBudget:
    mode: str
    start: Clock
    phase_start: Clock
    ceiling: Clock

    phase = "helper"

    def validate(self) -> None:
        require(self.mode in HELPER_MODES, "helper-operation")
        for value in (self.start, self.phase_start, self.ceiling):
            value.validate()
        cap = (60 if self.mode == "prepare-install" else 10) * SECOND
        require(
            self.start.boot_id == self.phase_start.boot_id == self.ceiling.boot_id
            and self.start.monotonic_ns <= self.phase_start.monotonic_ns
            and self.start.wall_ns <= self.phase_start.wall_ns
            and 2 * SECOND
            < self.ceiling.monotonic_ns - self.phase_start.monotonic_ns
            <= cap
            and 2 * SECOND < self.ceiling.wall_ns - self.phase_start.wall_ns <= cap,
            "helper-original-window",
        )

    @property
    def work_ns(self) -> int:
        return self.ceiling.monotonic_ns - 2 * SECOND

    def compact(self) -> dict:
        self.validate()
        return {
            name: {
                axis: getattr(self.ceiling, axis) - reserve * SECOND
                for axis in ("monotonic_ns", "wall_ns")
            }
            for name, reserve in (("work", 2), ("cleanup", 0), ("gate", 0))
        }

    def remaining_ns(self, stage: str, now: Clock) -> int:
        now.validate()
        require(
            stage in {"work", "cleanup", "gate"}
            and now.boot_id == self.start.boot_id
            and now.monotonic_ns >= self.phase_start.monotonic_ns
            and now.wall_ns >= self.phase_start.wall_ns,
            "helper-clock",
        )
        return min(
            self.compact()[stage][axis] - getattr(now, axis)
            for axis in ("monotonic_ns", "wall_ns")
        )


@dataclass(frozen=True)
class Binding:
    """Expected caller-provided bindings, not an execution certificate."""

    nonce: str
    intent_sha256: str
    source_sha256: str
    session_admission_sha256: str
    budget_sha256: str
    exec_sha256: str

    def validate(
        self,
        budget: PhaseBudget | BudgetWindow | HelperBudget,
        spec: CaptureSpec | RoleSpec | HelperSpec,
    ) -> None:
        require(
            re.fullmatch(r"[0-9a-f]{32}", self.nonce) is not None
            and all(
                checksum(x)
                for x in (
                    self.intent_sha256,
                    self.source_sha256,
                    self.session_admission_sha256,
                    self.budget_sha256,
                    self.exec_sha256,
                )
            ),
            "binding-shape",
        )
        require(
            self.budget_sha256 == digest(asdict(budget))
            and self.exec_sha256 == digest(spec.contract()),
            "binding-mismatch",
        )


@dataclass(frozen=True)
class Terminal:
    pid: int
    code: str
    status: int


@dataclass(frozen=True)
class Reaped:
    pid: int
    status: int


@dataclass
class Handle:
    identity: ProcessIdentity
    pidfd: int


@dataclass
class Spawn:
    pid: int
    gate_fd: int
    stdout_fd: int
    stderr_fd: int


class Kernel(Protocol):
    def clock(self) -> Clock: ...
    def self_identity(self) -> ProcessIdentity: ...
    def task_ids(self) -> tuple[int, ...]: ...
    def subreaper(self, enabled: bool | None = None) -> int: ...
    def admit_execution(
        self, binding: Binding, spec: CaptureSpec | RoleSpec | HelperSpec
    ) -> None: ...
    def children(self) -> tuple[int, ...]: ...
    def identity(self, pid: int) -> ProcessIdentity: ...
    def open_pidfd(self, pid: int) -> int: ...
    def pidfd_pid(self, fd: int) -> int: ...
    def exited(self, fd: int) -> bool: ...
    def peek(self, fd: int) -> Terminal | None: ...
    def reap(self, fd: int) -> Reaped: ...
    def signal(self, fd: int, sig: int) -> None: ...
    def close(self, fd: int) -> None: ...
    def fork_capture(self, spec: CaptureSpec | RoleSpec | HelperSpec) -> Spawn: ...
    def release(self, fd: int) -> None: ...
    def read_output(self, fd: int, maximum: int) -> bytes | None: ...
    def sleep(self, seconds: float) -> None: ...


@dataclass(frozen=True)
class RoleProof:
    role: str
    owner: ProcessIdentity
    clock: Clock
    subreaper_before: int
    subreaper_after: int
    task_ids: tuple[int, ...]

    def raw(self) -> dict:
        return {
            "role": self.role,
            "self": asdict(self.owner),
            "clock": asdict(self.clock),
            "subreaper_before": self.subreaper_before,
            "subreaper_after": self.subreaper_after,
            "task_ids": list(self.task_ids),
        }


def initialize_role(kernel: Kernel, role: str, expected: ProcessIdentity) -> RoleProof:
    require(role in ROLES, "outer-role")
    expected.validate()
    owner = kernel.self_identity()
    require(owner.uid == 0, "root-outer-role-required")
    require(
        owner == expected and kernel.task_ids() == (owner.pid,),
        "role-identity-or-threads",
    )
    before = kernel.subreaper()
    require(before in (0, 1), "subreaper-readback")
    kernel.subreaper(True)
    after = kernel.subreaper()
    require(
        after == 1
        and kernel.task_ids() == (owner.pid,)
        and kernel.self_identity() == owner,
        "subreaper-not-proven",
    )
    now = kernel.clock()
    require(now.boot_id == owner.boot_id, "role-boot")
    return RoleProof(role, owner, now, before, after, (owner.pid,))


class OwnedFamily:
    """One fixed capture and its adopted children; no process-group authority."""

    def __init__(
        self,
        kernel: Kernel,
        proof: RoleProof,
        budget: PhaseBudget | BudgetWindow | HelperBudget,
        *,
        maximum_children: int = 32,
        maximum_output: int = 2**20,
    ):
        require(
            1 <= maximum_children <= 32 and 1 <= maximum_output <= 8 * 2**20,
            "family-limits",
        )
        budget.validate()
        require(
            proof.clock.boot_id == budget.phase_start.boot_id
            and proof.clock.monotonic_ns >= budget.phase_start.monotonic_ns
            and proof.clock.wall_ns >= budget.phase_start.wall_ns,
            "role-before-phase",
        )
        self.kernel, self.proof, self.budget = kernel, proof, budget
        self.maximum_children, self.maximum_output = maximum_children, maximum_output
        self.handles: dict[int, Handle] = {}
        self.seen: set[tuple[int, int]] = set()
        self.events: list[dict] = []
        self.signals: list[dict] = []
        self.adopted: list[dict] = []
        self.last_clock = proof.clock
        self.spawn: Spawn | None = None
        self.root: ProcessIdentity | None = None
        self.binding: Binding | None = None
        self.terminals: list[dict] = []
        self.output = {"stdout": bytearray(), "stderr": bytearray()}
        self.eof = {"stdout": False, "stderr": False}
        self.parent: Handle | None = None
        self.sent: set[tuple[int, int, int]] = set()
        self.start_failure: str | None = None
        self.subject: CaptureSpec | RoleSpec | HelperSpec | None = None

    def _clock(self) -> Clock:
        now = self.kernel.clock()
        now.validate()
        require(
            now.boot_id == self.last_clock.boot_id
            and now.monotonic_ns >= self.last_clock.monotonic_ns
            and now.wall_ns >= self.last_clock.wall_ns,
            "family-clock-regressed",
        )
        # Reparenting after caller death does not revoke the subreaper's cleanup
        # ownership. Every other self identity field must remain exact.
        current = asdict(self.kernel.self_identity())
        expected = asdict(self.proof.owner)
        current.pop("ppid")
        expected.pop("ppid")
        require(
            current == expected
            and self.kernel.task_ids() == (self.proof.owner.pid,)
            and self.kernel.subreaper() == 1,
            "family-owner-or-threads",
        )
        self.last_clock = now
        return now

    def _bind(self, pid: int) -> Handle:
        require(
            pid in self.kernel.children()
            and pid not in self.handles
            and len(self.seen) < self.maximum_children,
            "unowned-or-too-many-children",
        )
        p = self.kernel.identity(pid)
        p.validate()
        require(
            p.ppid == self.proof.owner.pid
            and p.boot_id == self.proof.owner.boot_id
            and p.pid_namespace_inode == self.proof.owner.pid_namespace_inode
            and p.start_ticks >= self.proof.owner.start_ticks,
            "not-current-direct-child",
        )
        fd = self.kernel.open_pidfd(pid)
        try:
            require(
                self.kernel.pidfd_pid(fd) == pid
                and self.kernel.identity(pid) == p
                and pid in self.kernel.children(),
                "pidfd-child-raced",
            )
        except BaseException:
            self.kernel.close(fd)
            raise
        h = Handle(p, fd)
        require((p.pid, p.start_ticks) not in self.seen, "child-identity-rebound")
        self.seen.add((p.pid, p.start_ticks))
        self.handles[pid] = h
        return h

    def start_capture(
        self,
        spec: CaptureSpec | RoleSpec | HelperSpec,
        binding: Binding,
        parent: Handle,
    ) -> None:
        require(
            self.spawn is None and not self.handles and not self.kernel.children(),
            "family-not-fresh",
        )
        binding.validate(self.budget, spec)
        require(
            self.maximum_output <= spec.contract()["resources"]["stdout_stderr_bytes"],
            "subject-output-limit",
        )
        now = self._clock()
        require(
            spec.phase == self.budget.phase
            and spec.deadline_monotonic_ns == self.budget.work_ns
            and self.budget.remaining_ns("work", now) > 0,
            "capture-original-work-budget",
        )
        require(
            parent.identity.pid == self.proof.owner.ppid
            and parent.identity.boot_id == self.proof.owner.boot_id
            and self.kernel.pidfd_pid(parent.pidfd) == parent.identity.pid
            and not self.kernel.exited(parent.pidfd)
            and self.kernel.identity(parent.identity.pid) == parent.identity,
            "parent-handle-binding",
        )
        self.kernel.admit_execution(binding, spec)
        self.parent = parent
        self.binding = binding
        self.subject = spec
        spawned = self.kernel.fork_capture(spec)
        self.spawn = spawned
        try:
            h = self._bind(spawned.pid)
            self.root = h.identity
            require(spawned.gate_fd >= 0, "child-not-ready")
            self.events.append(
                {
                    "event": "child_started",
                    "owner": asdict(self.proof.owner),
                    "child": asdict(h.identity),
                    "clock": asdict(self._clock()),
                    "exec_contract_sha256": binding.exec_sha256,
                    "pidfd_target_pid": self.kernel.pidfd_pid(h.pidfd),
                    "gate_closed": True,
                }
            )
            require(
                self.budget.remaining_ns("work", self._clock()) > 0, "capture-gate-late"
            )
            require(
                not self.kernel.exited(parent.pidfd)
                and self.kernel.pidfd_pid(parent.pidfd) == parent.identity.pid
                and self.kernel.identity(parent.identity.pid) == parent.identity,
                "parent-died-before-release",
            )
            # This is gate-write initiation, not a completion timestamp. A fast
            # child may run before the parent is scheduled again after write.
            release_clock = self._clock()
            self.kernel.release(spawned.gate_fd)
            spawned.gate_fd = -1
            require(
                self.budget.remaining_ns("work", self._clock()) > 0
                and not self.kernel.exited(parent.pidfd),
                "post-release-deadline-or-parent",
            )
            self.events.append(
                {
                    "event": "child_released",
                    "child": asdict(h.identity),
                    "clock": asdict(release_clock),
                    "exec_contract_sha256": binding.exec_sha256,
                }
            )
        except BaseException:
            if spawned.gate_fd >= 0:
                self.kernel.close(spawned.gate_fd)
                spawned.gate_fd = -1
            self.start_failure = "start-not-complete"
            raise

    def _adopt(self) -> None:
        for pid in self.kernel.children():
            if pid not in self.handles:
                h = self._bind(pid)
                if (
                    self.spawn is not None
                    and pid == self.spawn.pid
                    and self.root is None
                ):
                    self.root = h.identity
                self.adopted.append(
                    {"child": asdict(h.identity), "clock": asdict(self._clock())}
                )

    def _drain_output(self) -> None:
        assert self.spawn is not None
        for name, fd in (
            ("stdout", self.spawn.stdout_fd),
            ("stderr", self.spawn.stderr_fd),
        ):
            if self.eof[name]:
                continue
            raw = self.kernel.read_output(fd, 65536)
            if raw is None:
                continue
            self.eof[name] = not raw
            require(
                sum(map(len, self.output.values())) + len(raw) <= self.maximum_output,
                "output-limit",
            )
            self.output[name].extend(raw)

    def _terminals(self) -> None:
        for pid, h in list(self.handles.items()):
            result = self.kernel.peek(h.pidfd)
            if result is None:
                continue
            require(
                result.pid == pid
                and result.code in {"CLD_EXITED", "CLD_KILLED", "CLD_DUMPED"},
                "terminal-child-binding",
            )
            require(
                self.kernel.identity(pid) == h.identity
                and pid in self.kernel.children(),
                "terminal-owner-drift",
            )
            reaped = self.kernel.reap(h.pidfd)
            require(
                reaped.pid == pid
                and (
                    (
                        result.code == "CLD_EXITED"
                        and os.WIFEXITED(reaped.status)
                        and os.WEXITSTATUS(reaped.status) == result.status
                    )
                    or (
                        result.code in {"CLD_KILLED", "CLD_DUMPED"}
                        and os.WIFSIGNALED(reaped.status)
                        and os.WTERMSIG(reaped.status) == result.status
                    )
                ),
                "wait-reap-drift",
            )
            self.terminals.append(
                {
                    "owner": asdict(self.proof.owner),
                    "child": asdict(h.identity),
                    "clock": asdict(self._clock()),
                    "waitid": asdict(result),
                    "waitpid": asdict(reaped),
                    "reaped": True,
                }
            )
            self.kernel.close(h.pidfd)
            del self.handles[pid]

    def _signal(self, sig: int) -> None:
        for pid, h in self.handles.items():
            require(
                pid in self.kernel.children()
                and self.kernel.identity(pid) == h.identity
                and self.kernel.pidfd_pid(h.pidfd) == pid,
                "signal-owner-drift",
            )
            key = (pid, h.identity.start_ticks, sig)
            if key not in self.sent and self.kernel.peek(h.pidfd) is None:
                self.kernel.signal(h.pidfd, sig)
                self.sent.add(key)
                self.signals.append(
                    {
                        "child": asdict(h.identity),
                        "signal": sig,
                        "clock": asdict(self._clock()),
                        "method": "pidfd",
                    }
                )

    def finish(
        self,
        parent: Handle,
        *,
        handoff_check: Callable[[Clock], Clock | None] | None = None,
    ) -> dict:
        require(
            self.spawn is not None and self.binding is not None, "capture-not-started"
        )
        require(parent == self.parent, "parent-handle-binding")
        assert self.spawn is not None and self.binding is not None
        failure: str | None = self.start_failure
        term_until = None
        closed = None
        while True:
            now = self._clock()
            if handoff_check is not None:
                require(isinstance(self.budget, BudgetWindow), "handoff-budget-kind")
                assert isinstance(self.budget, BudgetWindow)
                if self.budget.dummy_start is None:
                    observed_start = handoff_check(now)
                    if observed_start is not None:
                        now = self._clock()
                        self.budget = self.budget.handoff(observed_start, now)
            if self.budget.remaining_ns("cleanup", now) <= 0:
                failure = failure or "cleanup-deadline"
                break
            # Only the fixed guardian may use its cleanup tail to publish the
            # already-closed collector family. Its own code enforces capture work.
            work_stage = (
                "cleanup"
                if isinstance(self.subject, RoleSpec)
                and self.subject.role == "guardian"
                else "work"
            )
            if self.budget.remaining_ns(work_stage, now) <= 0:
                failure = failure or "work-deadline"
            if self.kernel.exited(parent.pidfd):
                failure = failure or "parent-died"
            self._adopt()
            if self.adopted:
                failure = failure or "orphan-adopted"
            if failure is None:
                try:
                    self._drain_output()
                except OuterRefusal:
                    failure = "output-limit"
            self._terminals()
            roots = [
                x
                for x in self.terminals
                if self.root is not None and x["child"] == asdict(self.root)
            ]
            if roots and roots[0]["waitid"] != {
                "pid": self.spawn.pid,
                "code": "CLD_EXITED",
                "status": 0,
            }:
                failure = failure or "child-nonnatural-or-nonzero"
            if not self.handles and not self.kernel.children():
                if failure is None:
                    try:
                        self._drain_output()
                    except OuterRefusal:
                        failure = "output-limit"
                if failure is not None or all(self.eof.values()):
                    closed = {
                        "owner": asdict(self.proof.owner),
                        "clock": asdict(self._clock()),
                        "task_ids": list(self.kernel.task_ids()),
                        "direct_children": [],
                        "retained_children": [],
                    }
                    break
            if failure:
                if term_until is None:
                    term_until = now.monotonic_ns + min(
                        2 * SECOND, max(0, self.budget.remaining_ns("cleanup", now))
                    )
                self._signal(
                    signal.SIGTERM if now.monotonic_ns < term_until else signal.SIGKILL
                )
            self.kernel.sleep(
                min(
                    0.01,
                    max(0, self.budget.remaining_ns("cleanup", self._clock())) / SECOND,
                )
            )
        for fd in (self.spawn.stdout_fd, self.spawn.stderr_fd):
            self.kernel.close(fd)
        complete = (
            failure is None
            and closed is not None
            and len(self.terminals) == 1
            and not self.signals
            and not self.adopted
        )
        child_terminal = next(
            (
                x
                for x in self.terminals
                if self.root is not None and x["child"] == asdict(self.root)
            ),
            None,
        )
        return {
            "format": "strength-freshness-owned-family-v1",
            "role_admission": self.proof.raw(),
            "binding": {
                "phase": self.budget.phase,
                "nonce": self.binding.nonce,
                "boot_id": self.budget.phase_start.boot_id,
                "outer_intent_sha256": self.binding.intent_sha256,
                "qualified_outer_source_sha256": self.binding.source_sha256,
                "phase_start": asdict(self.budget.phase_start),
            },
            "budget": self.budget.compact(),
            "child_started": next(
                (
                    {k: v for k, v in x.items() if k != "event"}
                    for x in self.events
                    if x["event"] == "child_started"
                ),
                None,
            ),
            "child_released": next(
                (
                    {k: v for k, v in x.items() if k != "event"}
                    for x in self.events
                    if x["event"] == "child_released"
                ),
                None,
            ),
            "child_terminal": child_terminal,
            "other_terminals": [x for x in self.terminals if x != child_terminal],
            "family_closed": closed,
            "signals": self.signals,
            "adopted_children": self.adopted,
            "natural_complete": complete,
            "reason": None if complete else failure or "incomplete",
            "output": {
                k: v
                for name, b in self.output.items()
                for k, v in (
                    (name + "_bytes", len(b)),
                    (name + "_sha256", hashlib.sha256(b).hexdigest()),
                )
            },
        }


class LinuxKernel:
    """Raw Linux syscalls, unarmed until caller/session admission is implemented."""

    def __init__(self):
        require(
            sys.platform == "linux"
            and all(
                hasattr(os, name)
                for name in (
                    "pidfd_open",
                    "P_PIDFD",
                    "waitid",
                    "pipe2",
                    "sched_setaffinity",
                    "sched_getaffinity",
                )
            )
            and hasattr(signal, "pidfd_send_signal"),
            "linux-pidfd-unavailable",
        )
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.prctl.restype = ctypes.c_int
        self._boot = (
            self._read(Path("/proc/sys/kernel/random/boot_id"), 128).decode().strip()
        )
        self._admitted_exec: str | None = None

    @staticmethod
    def _read(path: Path, maximum: int) -> bytes:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            data = os.read(fd, maximum + 1)
            require(len(data) <= maximum, "proc-read-bound")
            return data
        finally:
            os.close(fd)

    def clock(self) -> Clock:
        return Clock(self._boot, time.monotonic_ns(), time.time_ns())

    def identity(self, pid: int) -> ProcessIdentity:
        require(type(pid) is int and pid > 0, "process-id")
        root = Path("/proc") / str(pid)
        raw = self._read(root / "stat", 8192).decode()
        require(raw.split(" ", 1)[0] == str(pid) and ")" in raw, "proc-stat")
        parts = raw[raw.rindex(")") + 2 :].split()
        status = self._read(root / "status", 16384).decode().splitlines()
        uid = next(x.split()[1] for x in status if x.startswith("Uid:"))
        cg = self._read(root / "cgroup", 8192).decode().splitlines()
        require(len(cg) == 1 and cg[0].startswith("0::/"), "unified-cgroup-required")
        return ProcessIdentity(
            pid,
            int(parts[19]),
            int(parts[1]),
            int(parts[2]),
            int(parts[3]),
            int(uid),
            self._boot,
            cg[0][3:],
            (root / "ns/pid").stat().st_ino,
        )

    def self_identity(self) -> ProcessIdentity:
        return self.identity(os.getpid())

    def task_ids(self) -> tuple[int, ...]:
        return tuple(sorted(int(p.name) for p in Path("/proc/self/task").iterdir()))

    def subreaper(self, enabled: bool | None = None) -> int:
        require(
            self.task_ids() == (os.getpid(),)
            and signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL,
            "single-thread-default-sigchld-required",
        )
        if enabled is not None:
            require(
                self.libc.prctl(36, int(enabled), 0, 0, 0) == 0, "set-subreaper-failed"
            )
        result = ctypes.c_int()
        require(
            self.libc.prctl(37, ctypes.byref(result), 0, 0, 0) == 0,
            "get-subreaper-failed",
        )
        return result.value

    def admit_execution(
        self, binding: Binding, spec: CaptureSpec | RoleSpec | HelperSpec
    ) -> None:
        del binding, spec
        raise OuterRefusal("actual-caller-session-source-admission-unqualified")

    def children(self) -> tuple[int, ...]:
        require(self.task_ids() == (os.getpid(),), "child-enumeration-thread-drift")
        raw = self._read(Path(f"/proc/self/task/{os.getpid()}/children"), 4096).decode()
        result = tuple(int(x) for x in raw.split())
        require(
            len(result) <= 32 and len(set(result)) == len(result), "child-count-bound"
        )
        return result

    def open_pidfd(self, pid: int) -> int:
        return getattr(os, "pidfd_open")(pid)

    def pidfd_pid(self, fd: int) -> int:
        text = self._read(Path(f"/proc/self/fdinfo/{fd}"), 4096).decode()
        return int(
            next(x.split()[1] for x in text.splitlines() if x.startswith("Pid:"))
        )

    def exited(self, fd: int) -> bool:
        return bool(select.select([fd], [], [], 0)[0])

    @staticmethod
    def _terminal(result) -> Terminal | None:
        if result is None:
            return None
        codes = {
            os.CLD_EXITED: "CLD_EXITED",
            os.CLD_KILLED: "CLD_KILLED",
            os.CLD_DUMPED: "CLD_DUMPED",
        }
        require(result.si_code in codes, "unexpected-wait-status")
        return Terminal(result.si_pid, codes[result.si_code], result.si_status)

    def peek(self, fd: int) -> Terminal | None:
        return self._terminal(
            getattr(os, "waitid")(
                getattr(os, "P_PIDFD"), fd, os.WEXITED | os.WNOHANG | os.WNOWAIT
            )
        )

    def reap(self, fd: int) -> Reaped:
        # WNOWAIT above kept the actual child unreaped, so its PID cannot be
        # reused. Preserve the real waitpid return, not an inferred Boolean.
        observed = self.peek(fd)
        require(observed is not None, "child-not-terminal")
        assert observed is not None
        pid, status = os.waitpid(observed.pid, os.WNOHANG)
        require(pid == observed.pid, "waitpid-child")
        return Reaped(pid, status)

    def signal(self, fd: int, sig: int) -> None:
        require(sig in (signal.SIGTERM, signal.SIGKILL), "signal-not-registered")
        getattr(signal, "pidfd_send_signal")(fd, sig)

    def close(self, fd: int) -> None:
        os.close(fd)

    def fork_capture(self, spec: CaptureSpec | RoleSpec | HelperSpec) -> Spawn:
        expected = digest(spec.contract())
        require(
            self._admitted_exec == expected,
            "actual-caller-session-source-admission-unqualified",
        )
        self._admitted_exec = None
        return self._fork_capture_gated(spec)

    def _fork_capture_gated(self, spec: CaptureSpec | RoleSpec | HelperSpec) -> Spawn:
        """Private syscall primitive; no public admission path currently arms it."""
        require(
            self.task_ids() == (os.getpid(),) and self.subreaper() == 1,
            "fork-role-not-proven",
        )
        frame_fd = None
        if isinstance(spec, (RoleSpec, HelperSpec)):
            require(hasattr(os, "memfd_create"), "sealed-frame-unavailable")
            frame_fd = getattr(os, "memfd_create")(
                "cpu-outer-frame",
                getattr(os, "MFD_CLOEXEC") | getattr(os, "MFD_ALLOW_SEALING"),
            )
            try:
                require(
                    os.write(frame_fd, spec.frame) == len(spec.frame), "frame-write"
                )
                os.lseek(frame_fd, 0, os.SEEK_SET)
                seals = (
                    getattr(fcntl, "F_SEAL_WRITE")
                    | getattr(fcntl, "F_SEAL_GROW")
                    | getattr(fcntl, "F_SEAL_SHRINK")
                    | getattr(fcntl, "F_SEAL_SEAL")
                )
                fcntl.fcntl(frame_fd, getattr(fcntl, "F_ADD_SEALS"), seals)
                require(
                    fcntl.fcntl(frame_fd, getattr(fcntl, "F_GET_SEALS")) == seals,
                    "frame-seals",
                )
            except BaseException:
                os.close(frame_fd)
                raise
        gate_r, gate_w = getattr(os, "pipe2")(os.O_CLOEXEC)
        ready_r, ready_w = getattr(os, "pipe2")(os.O_CLOEXEC | os.O_NONBLOCK)
        out_r, out_w = getattr(os, "pipe2")(os.O_CLOEXEC)
        err_r, err_w = getattr(os, "pipe2")(os.O_CLOEXEC)
        pid = os.fork()
        if pid == 0:
            try:
                for fd in (gate_w, ready_r, out_r, err_r):
                    os.close(fd)
                os.setsid()
                # Explicit per-process limits; not claimed as aggregate cgroup
                # limits. The future admission must qualify executable needs.
                getattr(os, "sched_setaffinity")(
                    0, {min(getattr(os, "sched_getaffinity")(0))}
                )
                os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
                address_space = spec.contract()["resources"]["address_space_bytes"]
                hard_address_space = spec.contract()["resources"].get(
                    "address_space_hard_bytes", address_space
                )
                inherited_hard = resource.getrlimit(resource.RLIMIT_AS)[1]
                require(
                    inherited_hard == resource.RLIM_INFINITY
                    or inherited_hard >= hard_address_space,
                    "inherited-address-space-ceiling",
                )
                resource.setrlimit(
                    resource.RLIMIT_AS, (address_space, hard_address_space)
                )
                require(
                    resource.getrlimit(resource.RLIMIT_AS)
                    == (address_space, hard_address_space),
                    "address-space-readback",
                )
                resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
                resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 2**20, 32 * 2**20))
                os.dup2(out_w, 1)
                os.dup2(err_w, 2)
                os.close(out_w)
                os.close(err_w)
                os.chdir(spec.control_root)
                if frame_fd is not None:
                    os.dup2(frame_fd, 127, inheritable=True)
                    if frame_fd != 127:
                        os.close(frame_fd)
                os.write(ready_w, b"R")
                os.close(ready_w)
                child_deadline = (
                    spec.process_deadline_ns
                    if isinstance(spec, RoleSpec)
                    else spec.deadline_monotonic_ns
                )
                remaining = (child_deadline - time.monotonic_ns()) / SECOND
                if remaining <= 0:
                    os._exit(124)
                signal.signal(signal.SIGALRM, signal.SIG_DFL)
                signal.setitimer(signal.ITIMER_REAL, remaining)
                if os.read(gate_r, 1) != b"G":
                    os._exit(125)
                os.close(gate_r)
                os.execve(spec.python, spec.argv(), spec.contract()["env"])
            except BaseException:
                os._exit(125)
        for fd in (gate_r, ready_w, out_w, err_w):
            os.close(fd)
        if frame_fd is not None:
            os.close(frame_fd)
        # Readiness is before recorded identity: setsid and resource setup must
        # not change the child after its start proof has been captured.
        remaining = (spec.deadline_monotonic_ns - time.monotonic_ns()) / SECOND
        ready = select.select([ready_r], [], [], max(0, min(1.0, remaining)))[0]
        ok = bool(ready) and os.read(ready_r, 1) == b"R"
        os.close(ready_r)
        os.set_blocking(out_r, False)
        os.set_blocking(err_r, False)
        if not ok:
            os.close(gate_w)
            # Return the actually owned gated child for bounded adoption/reap;
            # a missing readiness proof must never release executable work.
            return Spawn(pid, -1, out_r, err_r)
        return Spawn(pid, gate_w, out_r, err_r)

    def release(self, fd: int) -> None:
        require(os.write(fd, b"G") == 1, "exec-gate-write")
        os.close(fd)

    def read_output(self, fd: int, maximum: int) -> bytes | None:
        require(0 < maximum <= 65536, "output-read-bound")
        try:
            return os.read(fd, maximum)
        except BlockingIOError:
            return None

    def sleep(self, seconds: float) -> None:
        require(0 <= seconds <= 0.01, "poll-bound")
        time.sleep(seconds)
