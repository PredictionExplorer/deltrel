"""Finite, durable freshness handoff core; deliberately no Linux/GPU backend.

The host interface is a separate qualification boundary. This module contains
the complete transition and recovery decisions, but importing it or validating
a plan cannot run systemctl, initialize CUDA, or launch a worker. A production
adapter must prove its deadline/ownership/host-wide lease semantics separately.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass, field
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
from typing import Any, Iterator, Mapping, Protocol, TypeGuard

from .runtime import atomic_json

FORMAT = "deltreltrain.strength-freshness-guard-plan"
STATE_FORMAT = "deltreltrain.strength-freshness-guard-state"
RECEIPT_FORMAT = "deltreltrain.strength-freshness-cuda-qualification"
TOTAL = 2700.0
STOP = 390.0
PROBE = 600.0
EMPTY = 90.0
APPLY = 120.0
STARTUP = 900.0
CANARY = 120.0
TAIL = 480.0
START_RESERVE = STARTUP + CANARY + TAIL
APPLY_RESERVE = APPLY + START_RESERVE
PROBE_RESERVE = PROBE + EMPTY + APPLY_RESERVE
CHECKS = {
    "cuda_available",
    "learner_resume_step",
    "ema_preserved",
    "native_cuda_search",
}
ADMISSION = {
    "plan_sha256",
    "source_commit",
    "source_manifest_sha256",
    "profile_sha256",
    "recovery_pointer_sha256",
    "execution_pins",
    "pyvenv",
    "training_module",
    "native_file",
    "native_binaries",
}
PRESERVED = {
    "run_identity",
    "champion",
    "arena",
    "utd",
    "replay_watermark",
    "cadence",
    "work_schedule",
    "continuation_clock",
}
ROLES = ("r3", "r4", "probe", "guard")
PROOF_INPUTS = {
    "guard_source",
    "probe_source",
    "batch_manifest",
    "batch_payloads",
    "runtime_qualification",
    "runtime_source_manifest",
    "support_manifest",
}


class Refusal(RuntimeError):
    """A non-secret, fail-closed reason suitable for the durable journal."""


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def finite(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and type(value) in (int, float)
        and math.isfinite(value)
    )


def sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def require(value: object, code: str) -> None:
    if not value:
        raise Refusal(code)


def validate_plan(plan: dict[str, Any]) -> None:
    require(
        plan.get("format") == FORMAT
        and type(plan.get("schema_version")) is int
        and plan["schema_version"] == 1,
        "guard-plan-schema",
    )
    require(
        re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", plan.get("attempt_id", "")),
        "attempt-id",
    )
    for key in (
        "freshness_plan_sha256",
        "guard_source_sha256",
        "adapter_qualification_sha256",
        "probe_source_sha256",
        "gpu_identity_sha256",
    ):
        require(sha(plan.get(key)), "plan-pin-" + key)
    require(
        plan.get("budget")
        == {
            "total": 2700,
            "stop": 390,
            "probe": 600,
            "cleanup": 90,
            "apply": 120,
            "startup": 900,
            "canary": 120,
            "tail": 480,
        },
        "fixed-budget-required",
    )
    require(
        all(type(value) is int for value in plan["budget"].values()),
        "integer-budget-required",
    )
    for name in (
        "run_root",
        "runtime_root",
        "probe_output",
        "state_root",
        "exclusion_path",
    ):
        value = plan.get(name)
        require(
            isinstance(value, str)
            and Path(value).is_absolute()
            and Path(value).as_posix() == value
            and ".." not in Path(value).parts,
            "canonical-plan-path",
        )
    for b in (Path(plan["run_root"]), Path(plan["runtime_root"]).parent):
        a = Path(plan["probe_output"])
        require(
            not a.is_relative_to(b) and not b.is_relative_to(a),
            "probe-output-overlaps-runtime-inputs",
        )
    for name in ("run_root", "runtime_root", "probe_output"):
        a, b = Path(plan["state_root"]), Path(plan[name])
        require(
            not a.is_relative_to(b) and not b.is_relative_to(a),
            "guard-state-overlaps-runtime-inputs",
        )
    require(
        type(plan.get("continuation_started_ns")) is int
        and plan["continuation_started_ns"] > 0,
        "continuation-clock",
    )
    uuids = plan.get("gpu_uuids")
    require(
        isinstance(uuids, list)
        and len(uuids) == len(set(uuids)) == 8
        and all(isinstance(x, str) and x.startswith("GPU-") for x in uuids),
        "eight-gpu-identities-required",
    )
    assert isinstance(uuids, list)
    require(plan.get("probe_gpu_uuid") in uuids, "probe-gpu-not-registered")
    require(set(plan.get("units", {})) == set(ROLES), "unit-inventory")
    names = []
    for role, spec in plan["units"].items():
        require(
            re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", spec.get("name", "")),
            "unit-name",
        )
        require(sha(spec.get("definition_sha256")), "unit-definition-pin")
        require(spec.get("cgroup") == "/system.slice/" + spec["name"], "unit-cgroup")
        names.append(spec["name"])
        if role in ("r4", "probe"):
            require(
                spec.get("autonomous_retries") == 0
                and type(spec["autonomous_retries"]) is int,
                "autonomous-retries-forbidden",
            )
        require(
            type(spec.get("initial_enabled")) is bool
            and type(spec.get("committed_enabled")) is bool,
            "unit-enablement-pin",
        )
    require(len(names) == len(set(names)), "distinct-units-required")
    require(
        plan["units"]["r3"]["committed_enabled"] is False
        and plan["units"]["r4"]["initial_enabled"] is False
        and plan["units"]["probe"]["initial_enabled"] is False,
        "runtime-boot-enablement",
    )
    require(
        set(plan.get("admission", {})) == ADMISSION - {"recovery_pointer_sha256"},
        "admission-template-keys",
    )
    require(
        plan["admission"]["plan_sha256"] == plan["freshness_plan_sha256"],
        "admission-plan",
    )
    require(
        sha(plan["admission"]["profile_sha256"])
        and sha(plan["admission"]["source_manifest_sha256"]),
        "admission-profile-source",
    )
    require(
        set(plan.get("probe_inputs", {})) == {"profile", "batch_manifest"},
        "probe-inputs",
    )
    require(
        all(
            not Path(pin["path"]).is_relative_to(plan["state_root"])
            for pin in plan["probe_inputs"].values()
        ),
        "guard-state-overlaps-probe-input",
    )
    require(
        plan["probe_inputs"]["profile"]["sha256"]
        == plan["admission"]["profile_sha256"],
        "probe-profile",
    )
    runtime = plan.get("probe_runtime", {})
    require(
        set(runtime)
        == {
            "source_commit",
            "source_commit_file",
            "source_manifest",
            "python",
            "pyvenv",
            "training_module",
            "native_wrapper",
            "native_binaries",
            "rules_hash",
            "feature_schema_hash",
            "search_algorithm",
            "torch_version",
            "torch_cuda_build",
        },
        "probe-runtime-pins",
    )
    require(
        all(
            isinstance(runtime[key], str) and runtime[key]
            for key in ("torch_version", "torch_cuda_build")
        ),
        "cuda-runtime-build-required",
    )
    require(
        runtime["source_commit"] == plan["admission"]["source_commit"]
        and runtime["source_manifest"]["sha256"]
        == plan["admission"]["source_manifest_sha256"]
        and runtime["pyvenv"] == plan["admission"]["pyvenv"]
        and runtime["training_module"]["path"] == plan["admission"]["training_module"]
        and runtime["native_wrapper"]["path"] == plan["admission"]["native_file"]
        and runtime["native_binaries"] == plan["admission"]["native_binaries"]
        and runtime["python"] in plan["admission"]["execution_pins"],
        "probe-runtime-admission-mismatch",
    )
    identity = plan.get("run_identity", {})
    require(
        set(identity) == {"run_id", "generation_family", "created_ns"}
        and all(
            isinstance(identity[k], str) and identity[k]
            for k in ("run_id", "generation_family")
        )
        and type(identity["created_ns"]) is int
        and identity["created_ns"] > 0,
        "run-identity",
    )
    require(
        isinstance(plan.get("expected_workers"), list)
        and plan["expected_workers"]
        and len(plan["expected_workers"]) == len(set(plan["expected_workers"])),
        "worker-inventory",
    )
    require(plan.get("expected_cohorts") == 24, "cohort-inventory")
    closure = plan.get("proof_closure_inputs", {})
    require(
        set(closure) == PROOF_INPUTS
        and all(
            isinstance(v, list) and v and all(sha(x) for x in v)
            for v in closure.values()
        ),
        "proof-closure-inputs",
    )
    require(
        plan["guard_source_sha256"] in closure["guard_source"]
        and plan["probe_source_sha256"] in closure["probe_source"]
        and plan["probe_inputs"]["batch_manifest"]["sha256"]
        in closure["batch_manifest"],
        "proof-closure-input-binding",
    )
    support = plan.get("support_transition", {})
    require(
        set(support) == {"before", "after", "bindings", "sha256"}
        and support["sha256"]
        == digest({k: v for k, v in support.items() if k != "sha256"}),
        "support-transition-pin",
    )
    require(
        set(support["before"]) == set(support["after"]) and len(support["before"]) >= 3,
        "support-unit-inventory",
    )
    for stage in ("before", "after"):
        for name, value in support[stage].items():
            require(
                re.fullmatch(r"[A-Za-z0-9_.@-]+\.(service|timer)", name)
                and set(value) == {"definition_sha256", "environment_sha256", "enabled"}
                and sha(value["definition_sha256"])
                and sha(value["environment_sha256"])
                and type(value["enabled"]) is bool,
                "support-unit-pin",
            )
    binding = support["bindings"]
    require(
        binding.get("monitor_unit_after") == plan["units"]["r4"]["name"]
        and binding.get("profile_authority")
        == str(Path(plan["run_root"]) / "profile.sha256")
        and binding.get("python_after") == plan["probe_runtime"]["python"]["path"]
        and binding.get("runtime_pythonpath") == plan["runtime_root"]
        and binding.get("guard_before_units")
        == sorted([plan["units"]["r3"]["name"], plan["units"]["r4"]["name"]])
        and binding.get("control_cuda_visible_devices") == ""
        and isinstance(binding.get("control_pythonpath"), str)
        and Path(binding["control_pythonpath"]).is_absolute(),
        "support-command-boot-binding",
    )


@dataclass(frozen=True)
class Clock:
    boot_id: str
    monotonic: float
    wall_ns: int


@dataclass(frozen=True)
class Process:
    pid: int
    start_ticks: int


@dataclass(frozen=True)
class GuardReentryEvidence:
    """Independent self identity and old PID observation from the qualified host.

    ``None`` means the previous PID is absent. A present process must retain
    that PID and report its actual start ticks; a different process is not an
    absence assertion. The adapter must read these facts, not infer them from
    a changed systemd InvocationID.
    """

    clock: Clock
    executing_process: Process
    executing_invocation_id: str
    previous_pid_process: Process | None


@dataclass(frozen=True)
class Unit:
    name: str
    definition_sha256: str
    cgroup: str
    invocation_id: str = ""
    main: Process | None = None
    members: tuple[Process, ...] = ()
    active: str = "inactive"
    substate: str = "dead"
    job: Mapping[str, Any] | None = None
    entered_monotonic: float = 0.0
    result: str = "success"
    exit_code: int = 0
    runtime_restarts: int = 0
    enabled: bool = False

    @property
    def dead(self) -> bool:
        return (
            self.active in ("inactive", "failed")
            and self.substate in ("dead", "failed")
            and self.main is None
            and not self.members
            and self.job is None
        )


@dataclass(frozen=True)
class GPUOwner:
    uuid: str
    process: Process
    unit: str
    invocation_id: str
    cgroup: str


@dataclass(frozen=True)
class Authority:
    # The adapter must use the existing registration/record/journal validators,
    # not infer this from a profile filename or the guard's last phase.
    phase: str  # r3, pending, r4, unknown
    plan_sha256: str
    evidence_sha256: str


@dataclass(frozen=True)
class Observation:
    clock: Clock
    units: Mapping[str, Unit]
    gpu_uuids: tuple[str, ...]
    gpu_identity_sha256: str
    owners: tuple[GPUOwner, ...]
    authority: Authority
    controller_alive: bool = True
    lease: Mapping[str, Any] | None = None
    guard_ack: Mapping[str, Any] | None = None
    progress: Mapping[str, Any] | None = None
    backup: Mapping[str, Any] | None = None
    support: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


class Host(Protocol):
    """Qualified syscall boundary; CPU tests provide a deterministic fake.

    Every method must be wall-bounded by deadline, recheck expected unit/PID/job
    identities under the guard-owned host-wide lease, and never kill a foreign
    owner. The independent guard, not the short-lived controller, owns that
    lease through probe, receipt, apply and restart. No concrete live adapter is
    supplied in this module. Qualification must also prove the exact support
    command bindings (monitor --unit, dynamic profile, interpreter/import path),
    persistent guard Before= edges, and old/new unit enablement. Hashes alone
    are not permission to ignore those declared semantic requirements.
    """

    adapter_qualification_sha256: str

    def clock(self) -> Clock: ...
    def guard_reentry_evidence(
        self, previous: Process, deadline: float
    ) -> GuardReentryEvidence: ...
    def guard_boot_evidence(self, deadline: float) -> GuardReentryEvidence: ...
    def observe(self, deadline: float) -> Observation: ...
    def perform(self, kind: str, details: dict[str, Any], deadline: float) -> None: ...
    def verify_inputs(self, plan: dict[str, Any], deadline: float) -> None: ...
    def capture_boundary(
        self, plan: dict[str, Any], deadline: float
    ) -> dict[str, Any]: ...
    def check_boundary(self, boundary: dict[str, Any], deadline: float) -> None: ...


class Journal:
    """Protected durable control data with short nonblocking transaction locks."""

    def __init__(self, directory: Path):
        self.directory = directory.absolute()

    def create(self) -> None:
        require(self.directory.resolve() == self.directory, "state-directory-symlink")
        self.directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        self._check()

    def _check(self) -> None:
        require(self.directory.resolve() == self.directory, "state-directory-symlink")
        info = self.directory.stat()
        require(
            info.st_uid == os.geteuid() and info.st_mode & 0o777 == 0o700,
            "state-directory-unprotected",
        )

    @contextmanager
    def locked(self) -> Iterator[None]:
        self._check()
        descriptor = os.open(
            self.directory / "lock",
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise Refusal("guard-state-busy") from None
            yield
        finally:
            os.close(descriptor)

    def read(self, name: str) -> dict[str, Any]:
        require(Path(name).name == name, "state-file-name")
        self._check()
        path = self.directory / name
        require(
            not path.is_symlink() and path.is_file() and path.stat().st_size <= 2**20,
            "unsafe-state-file",
        )
        require(
            path.stat().st_uid == os.geteuid() and path.stat().st_mode & 0o777 == 0o600,
            "state-file-unprotected",
        )
        value = json.loads(path.read_bytes())
        require(isinstance(value, dict), "state-object-required")
        digest(value)  # Also rejects NaN/Infinity before authority is consumed.
        return value

    def save(self, name: str, value: dict[str, Any], *, new: bool = False) -> None:
        require(Path(name).name == name, "state-file-name")
        path = self.directory / name
        require(not path.is_symlink(), "state-file-symlink")
        require(not new or not path.exists(), "state-file-already-exists")
        require(
            len(json.dumps(value, allow_nan=False).encode()) <= 2**20, "state-too-large"
        )
        atomic_json(path, value)
        os.chmod(path, 0o600)


def _proof_file(root: Path, relative: str, maximum: int) -> bytes:
    require(
        isinstance(relative, str)
        and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts
        and Path(relative).as_posix() == relative,
        "proof-path",
    )
    path = root / relative
    require(
        path.resolve() == path and path.is_file() and path.stat().st_size <= maximum,
        "unsafe-proof-file",
    )
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    require(len(data) <= maximum, "proof-too-large")
    return data


def validate_probe(
    plan: dict[str, Any], state: dict[str, Any], unit: Unit, now: float
) -> dict[str, Any]:
    """A child observation is never an all-workers-released qualification."""
    challenge = state["challenge"]
    current_challenge = _proof_file(Path(plan["state_root"]), "challenge.json", 2**20)
    require(
        hashlib.sha256(current_challenge).hexdigest() == state["challenge_file_sha256"]
        and json.loads(current_challenge) == challenge,
        "probe-challenge-changed",
    )
    root = Path(plan["probe_output"])
    started_raw = _proof_file(root, "started.json", 2**20)
    result_raw = _proof_file(root, "result.json", 2**20)
    started, result = json.loads(started_raw), json.loads(result_raw)
    require(
        isinstance(started, dict)
        and isinstance(result, dict)
        and "probe" in state["owners"],
        "probe-schema-or-owner",
    )
    require("all_workers_released" not in result, "probe-cannot-certify-release")
    binding = ("attempt_id", "boot_id", "nonce", "probe_unit")
    for value, kind in ((started, "started"), (result, "result")):
        require(
            value.get("format") == f"strength-freshness-probe-{kind}-v1"
            and type(value.get("schema_version")) is int
            and value["schema_version"] == 1,
            "probe-schema",
        )
        require(all(value.get(k) == challenge[k] for k in binding), "probe-binding")
        require(
            value.get("challenge_sha256") == state["challenge_file_sha256"],
            "probe-challenge-hash",
        )
        require(
            value.get("invocation_id") == state["owners"]["probe"]["invocation_id"]
            and (
                value["invocation_id"] == unit.invocation_id
                or (unit.dead and not unit.invocation_id)
            ),
            "probe-invocation",
        )
        require(value.get("mode") == "cuda", "actual-cuda-required")
    first, last = result.get("started_monotonic"), result.get("completed_monotonic")
    require(
        finite(first)
        and finite(last)
        and first == started.get("started_monotonic")
        and challenge["issued_monotonic"]
        <= first
        <= last
        <= min(now, challenge["deadline_monotonic"]),
        "probe-time-window",
    )
    require(
        result.get("mode") == "cuda" and result.get("status") == "cuda_observed",
        "actual-cuda-required",
    )
    require(result.get("admission") == challenge["admission"], "probe-admission")
    work = result.get("work")
    require(
        isinstance(work, dict)
        and set(work) == {"optimizer_steps", "native_searches", "native_neural_calls"}
        and all(type(value) is int for value in work.values())
        and work["optimizer_steps"] == 1
        and work["native_searches"] == 24
        and work["native_neural_calls"] >= 24,
        "probe-real-work-required",
    )
    checks = result.get("checks")
    require(
        isinstance(checks, dict)
        and set(checks) == CHECKS
        and all(value is True for value in checks.values()),
        "probe-checks",
    )
    evidence = result.get("evidence")
    require(isinstance(evidence, list) and 1 <= len(evidence) <= 64, "probe-evidence")
    assert isinstance(evidence, list)
    seen: set[str] = set()
    total = 0
    for item in evidence:
        require(
            isinstance(item, dict) and set(item) == {"path", "sha256", "bytes"},
            "probe-evidence-pin",
        )
        name, size = item["path"], item["bytes"]
        require(
            isinstance(name, str)
            and name not in seen
            and name not in {"started.json", "result.json"}
            and type(size) is int
            and size >= 0
            and sha(item["sha256"]),
            "probe-evidence-inventory",
        )
        seen.add(name)
        total += size
        require(total <= 16 * 2**20, "probe-evidence-budget")
        raw = _proof_file(root, name, size)
        require(
            len(raw) == size and hashlib.sha256(raw).hexdigest() == item["sha256"],
            "probe-evidence-tamper",
        )
    require(total > 0, "nonempty-execution-evidence-required")
    return {
        "result": result,
        "result_sha256": hashlib.sha256(result_raw).hexdigest(),
        "started_sha256": hashlib.sha256(started_raw).hexdigest(),
    }


class Controller:
    """One bounded transition. Each tick is a short, recoverable transaction."""

    def __init__(
        self, plan: dict[str, Any], plan_sha256: str, journal: Journal, host: Host
    ):
        validate_plan(plan)
        require(digest(plan) == plan_sha256, "guard-plan-hash")
        require(
            host.adapter_qualification_sha256 == plan["adapter_qualification_sha256"],
            "unqualified-host-adapter",
        )
        require(
            plan["guard_source_sha256"] == source_sha256(),
            "executing-guard-source-mismatch",
        )
        require(
            journal.directory == Path(plan["state_root"])
            and journal.directory.resolve() == journal.directory,
            "guard-state-location-mismatch",
        )
        self.plan, self.plan_sha, self.journal, self.host = (
            plan,
            plan_sha256,
            journal,
            host,
        )

    def admit_guard_reentry(self) -> dict[str, Any]:
        """Admit this replacement guardian on the same boot, recovery only.

        This is not a launcher or a lease transfer. The independently executing
        pinned guardian must already own the exclusion lease. Reboot handling
        and retirement-only cleanup continue through their existing paths.
        No runtime ownership, start intent, history or budget is reset here.
        """
        with self.journal.locked():
            require(
                digest(self.plan) == self.plan_sha
                and self.plan["guard_source_sha256"] == source_sha256(),
                "guard-reentry-plan-or-source-drift",
            )
            state = self.journal.read("state.json")
            require(
                state.get("format") == STATE_FORMAT
                and state.get("plan_sha256") == self.plan_sha,
                "state-plan-mismatch",
            )
            require(
                not state.get("finished")
                and not state.get("runtime_authority_retired")
                and not state.get("handoff_complete"),
                "guard-reentry-authority-retired",
            )
            anchor = self.journal.read("anchor.json")
            require(
                hashlib.sha256(
                    (self.journal.directory / "anchor.json").read_bytes()
                ).hexdigest()
                == state.get("anchor_sha256")
                and all(
                    state.get(k) == anchor[k]
                    for k in ("plan_sha256", "attempt_id", "nonce", "deadline_wall_ns")
                ),
                "guard-anchor-changed",
            )
            require(
                state["started_monotonic"] == anchor["initial_clock"]["monotonic"]
                and finite(state["deadline"])
                and finite(state["last_monotonic"])
                and 0 <= state["remaining_ns"] <= int(TOTAL * 1e9)
                and abs(
                    state["deadline"]
                    - state["last_monotonic"]
                    - state["remaining_ns"] / 1e9
                )
                <= 1e-6,
                "guard-budget-state",
            )
            if state["active_boot_id"] == anchor["initial_clock"]["boot_id"]:
                require(
                    state["deadline"] == anchor["initial_clock"]["monotonic"] + TOTAL,
                    "deadline-extended",
                )
            previous_owner = state["owners"].get("guard")
            require(previous_owner is not None, "guard-reentry-no-previous-owner")
            previous = Process(**previous_owner["main"])
            now = self.host.clock()
            require(
                now.boot_id == state["active_boot_id"] == previous_owner["boot_id"],
                "guard-reentry-requires-same-boot",
            )
            require(
                finite(now.monotonic)
                and state["last_monotonic"] <= now.monotonic < state["deadline"]
                and state["last_wall_ns"] <= now.wall_ns < state["deadline_wall_ns"],
                "guard-reentry-original-deadline",
            )
            deadline = min(now.monotonic + 10, state["deadline"])
            evidence = self.host.guard_reentry_evidence(previous, deadline)
            require(
                evidence.clock.boot_id == now.boot_id
                and finite(evidence.clock.monotonic)
                and now.monotonic <= evidence.clock.monotonic < deadline
                and now.wall_ns <= evidence.clock.wall_ns < state["deadline_wall_ns"],
                "guard-reentry-evidence-clock",
            )
            require(
                evidence.previous_pid_process is None
                or (
                    evidence.previous_pid_process.pid == previous.pid
                    and evidence.previous_pid_process.start_ticks
                    != previous.start_ticks
                    and evidence.previous_pid_process.start_ticks > 0
                ),
                "guard-reentry-previous-process-present",
            )
            require(
                evidence.executing_process != previous
                and evidence.executing_process.pid > 0
                and evidence.executing_process.start_ticks > 0
                and re.fullmatch(r"[0-9a-f]{32}", evidence.executing_invocation_id)
                and evidence.executing_invocation_id != previous_owner["invocation_id"],
                "guard-reentry-executing-identity",
            )
            new_owner = {
                "invocation_id": evidence.executing_invocation_id,
                "main": asdict(evidence.executing_process),
                "boot_id": evidence.clock.boot_id,
            }
            # Validate every current unit/GPU/support owner using a disposable
            # state. _observe may adopt newly started runtime owners or advance
            # clocks; none of those changes belong to this re-entry transaction.
            candidate = copy.deepcopy(state)
            candidate["owners"]["guard"] = new_owner
            observed = self._observe(candidate, read_deadline=deadline)
            unit = observed.units["guard"]
            require(
                observed.clock.boot_id == evidence.clock.boot_id
                and evidence.clock.monotonic <= observed.clock.monotonic < deadline
                and evidence.clock.wall_ns
                <= observed.clock.wall_ns
                < state["deadline_wall_ns"],
                "guard-reentry-observation-clock",
            )
            require(
                unit.active == "active"
                and unit.job is None
                and unit.enabled
                and unit.main == evidence.executing_process
                and unit.main in unit.members
                and unit.invocation_id == evidence.executing_invocation_id
                and state["last_monotonic"]
                <= unit.entered_monotonic
                <= observed.clock.monotonic,
                "guard-reentry-pinned-unit",
            )
            self._lease(candidate, observed)
            state["owners"]["guard"] = new_owner
            state.setdefault("guard_reentries", []).append(
                {
                    "previous_owner": previous_owner,
                    "new_owner": new_owner,
                    "evidence": asdict(evidence),
                    "observation_clock": asdict(observed.clock),
                }
            )
            state.update(recovery_requested=True, recovery_mode=True)
            self._save(state, "same-boot-guard-reentry-recovery-only")
            return state

    def admit_guard_boot(self) -> dict[str, Any]:
        """Admit a new-boot guardian without dispatching recovery or any action.

        The original wall expiry caps the rebased monotonic budget. Empty
        runtime/probe resources and actual guardian self/lease identity are
        mandatory. Phase, action, anchor and event history stay unchanged;
        phase/terminal deadlines retain their original remaining wall bounds.
        Subsequent normal ticks own all runtime recovery decisions.
        """
        with self.journal.locked():
            require(
                digest(self.plan) == self.plan_sha
                and self.plan["guard_source_sha256"] == source_sha256(),
                "guard-boot-plan-or-source-drift",
            )
            state = self.journal.read("state.json")
            require(
                state.get("format") == STATE_FORMAT
                and state.get("plan_sha256") == self.plan_sha,
                "state-plan-mismatch",
            )
            require(
                not state.get("finished")
                and not state.get("runtime_authority_retired")
                and not state.get("handoff_complete"),
                "guard-boot-authority-retired",
            )
            anchor = self.journal.read("anchor.json")
            require(
                hashlib.sha256(
                    (self.journal.directory / "anchor.json").read_bytes()
                ).hexdigest()
                == state.get("anchor_sha256")
                and all(
                    state.get(k) == anchor[k]
                    for k in ("plan_sha256", "attempt_id", "nonce", "deadline_wall_ns")
                ),
                "guard-anchor-changed",
            )
            require(
                state["started_monotonic"] == anchor["initial_clock"]["monotonic"]
                and finite(state["deadline"])
                and finite(state["last_monotonic"])
                and 0 < state["remaining_ns"] <= int(TOTAL * 1e9)
                and abs(
                    state["deadline"]
                    - state["last_monotonic"]
                    - state["remaining_ns"] / 1e9
                )
                <= 1e-6,
                "guard-budget-state",
            )
            if state["active_boot_id"] == anchor["initial_clock"]["boot_id"]:
                require(
                    state["deadline"] == anchor["initial_clock"]["monotonic"] + TOTAL,
                    "deadline-extended",
                )
            now = self.host.clock()
            require(
                now.boot_id != state["active_boot_id"], "guard-boot-requires-new-boot"
            )
            require(
                finite(now.monotonic)
                and now.monotonic >= 0
                and state["last_wall_ns"] <= now.wall_ns < state["deadline_wall_ns"],
                "guard-boot-original-deadline",
            )
            remaining = (
                min(state["deadline_wall_ns"] - now.wall_ns, state["remaining_ns"])
                / 1e9
            )
            deadline = now.monotonic + min(10, remaining)
            evidence = self.host.guard_boot_evidence(deadline)
            require(
                evidence.clock.boot_id == now.boot_id
                and finite(evidence.clock.monotonic)
                and now.monotonic <= evidence.clock.monotonic < deadline
                and now.wall_ns <= evidence.clock.wall_ns < state["deadline_wall_ns"]
                and evidence.previous_pid_process is None,
                "guard-boot-evidence-clock",
            )
            require(
                type(evidence.executing_process.pid) is int
                and evidence.executing_process.pid > 0
                and type(evidence.executing_process.start_ticks) is int
                and evidence.executing_process.start_ticks > 0
                and re.fullmatch(r"[0-9a-f]{32}", evidence.executing_invocation_id),
                "guard-boot-executing-identity",
            )
            candidate = copy.deepcopy(state)
            observed = self._observe(candidate, read_deadline=deadline)
            unit = observed.units["guard"]
            require(
                observed.clock.boot_id == evidence.clock.boot_id
                and evidence.clock.monotonic <= observed.clock.monotonic < deadline
                and evidence.clock.wall_ns
                <= observed.clock.wall_ns
                < state["deadline_wall_ns"],
                "guard-boot-observation-clock",
            )
            require(
                unit.active == "active"
                and unit.job is None
                and unit.enabled
                and unit.main == evidence.executing_process
                and unit.main in unit.members
                and unit.invocation_id == evidence.executing_invocation_id
                and 0 <= unit.entered_monotonic <= observed.clock.monotonic,
                "guard-boot-pinned-unit",
            )
            self._lease(candidate, observed)
            previous_boot = state["active_boot_id"]
            for key in (
                "active_boot_id",
                "deadline",
                "last_monotonic",
                "last_wall_ns",
                "remaining_ns",
                "owners",
                "starts",
                "recovery_requested",
                "boot_recovery",
                "phase_deadline",
                "boot_clock_rebases",
            ):
                state[key] = candidate[key]
            if "terminal_deadline" in candidate:
                state["terminal_deadline"] = candidate["terminal_deadline"]
            state["recovery_mode"] = True
            state.setdefault("guard_boot_admissions", []).append(
                {
                    "previous_boot_id": previous_boot,
                    "evidence": asdict(evidence),
                    "observation_clock": asdict(observed.clock),
                    "deadline": state["deadline"],
                    "remaining_ns": state["remaining_ns"],
                }
            )
            state["revision"] += 1
            # Preserve the original event/action history exactly. Boot admission
            # is audited separately and grants no action before the caller's
            # pending-support recovery finishes.
            self.journal.save("state.json", state)
            return state

    def _save(self, state: dict[str, Any], event: str) -> None:
        state["revision"] += 1
        jobs = {
            group: {
                name: row["job"]
                for name, row in state.get(key, {}).items()
                if row.get("job") is not None
            }
            for group, key in (("runtime", "last_units"), ("support", "last_support"))
        }
        previous = state["events"][-1] if state["events"] else None
        if (
            event == "observed"
            and previous is not None
            and all(
                previous.get(key) == value
                for key, value in {
                    "event": event,
                    "phase": state["phase"],
                    "jobs": jobs,
                }.items()
            )
        ):
            previous["last_at"] = state["last_monotonic"]
            previous["observations"] = previous.get("observations", 1) + 1
        else:
            state["events"].append(
                {
                    "event": event,
                    "phase": state["phase"],
                    "at": state["last_monotonic"],
                    "jobs": jobs,
                }
            )
        require(len(state["events"]) <= 128, "guard-event-budget")
        self.journal.save("state.json", state)

    def _action(
        self,
        state: dict[str, Any],
        observation: Observation,
        kind: str,
        deadline: float,
        **details: Any,
    ) -> None:
        require(
            self.plan["guard_source_sha256"] == source_sha256(),
            "executing-guard-source-mismatch",
        )
        require(
            observation.clock.monotonic < deadline <= state["deadline"],
            "action-deadline",
        )
        # Persist intent before any host mutation; recovery re-observes durable
        # migration authority even if perform() dies after writing its journal.
        state["last_action"] = {"kind": kind, "deadline": deadline, "details": details}
        role = (
            "r3"
            if kind == "pause-support-and-stop-r3"
            else kind.removeprefix("stop-")
            if kind.startswith("stop-")
            else None
        )
        if role in ROLES:
            state.setdefault("stops", {})[role] = observation.units[role].invocation_id
        self._save(state, "intent-" + kind)
        self.host.perform(
            kind,
            {
                **details,
                "expected_boot_id": observation.clock.boot_id,
                "expected_units": {r: asdict(u) for r, u in observation.units.items()},
                "expected_support": dict(observation.support),
                "attempt_id": state["attempt_id"],
                "nonce": state["nonce"],
                "guard_plan_sha256": self.plan_sha,
            },
            deadline,
        )

    def _observe(
        self, state: dict[str, Any], *, read_deadline: float | None = None
    ) -> Observation:
        now = self.host.clock().monotonic
        require(finite(now), "observation-clock")
        # A reboot can reset monotonic time. Read-only observation still has
        # its own short bound before the original absolute expiry is applied.
        deadline = now + 5 if read_deadline is None else min(now + 5, read_deadline)
        require(now < deadline, "observation-deadline")
        observed = self.host.observe(deadline)
        clock = observed.clock
        require(finite(clock.monotonic) and type(clock.wall_ns) is int, "invalid-clock")
        require(clock.wall_ns >= state["last_wall_ns"], "wall-clock-reversed")
        if clock.boot_id != state["active_boot_id"]:
            # Recovery-only across boot. Keep the original absolute expiry;
            # never grant a fresh 45-minute window or resume CUDA qualification.
            remaining = (
                min(state["deadline_wall_ns"] - clock.wall_ns, state["remaining_ns"])
                / 1e9
            )
            require(remaining > 0, "original-deadline-expired-after-boot")
            new_deadline = clock.monotonic + remaining
            original_deadlines = {
                key: state[key]
                for key in ("deadline", "phase_deadline", "terminal_deadline")
                if key in state
            }
            rebased = {}
            for key, value in original_deadlines.items():
                require(finite(value), "original-phase-deadline-invalid")
                wall_expiry = state["last_wall_ns"] + int(
                    (value - state["last_monotonic"]) * 1e9
                )
                rebased[key] = min(
                    new_deadline,
                    clock.monotonic + (wall_expiry - clock.wall_ns) / 1e9,
                )
            require(
                all(observed.units[r].dead for r in ("r3", "r4", "probe"))
                and not observed.owners,
                "new-boot-owners-not-registered",
            )
            owners: dict[str, Any] = {}
            guard = observed.units["guard"]
            if not guard.dead:
                require(
                    guard.main is not None
                    and guard.active == "active"
                    and guard.job is None
                    and guard.enabled
                    and re.fullmatch(r"[0-9a-f]{32}", guard.invocation_id),
                    "new-boot-guard-not-ready",
                )
                assert guard.main is not None
                expected_lease = {
                    "path": self.plan["exclusion_path"],
                    "held": True,
                    "attempt_id": state["attempt_id"],
                    "nonce": state["nonce"],
                    "plan_sha256": self.plan_sha,
                    "boot_id": clock.boot_id,
                    "invocation_id": guard.invocation_id,
                    "owner": asdict(guard.main),
                }
                require(observed.lease == expected_lease, "new-boot-guard-lease")
                owners["guard"] = {
                    "invocation_id": guard.invocation_id,
                    "main": asdict(guard.main),
                    "boot_id": clock.boot_id,
                }
            state.update(
                active_boot_id=clock.boot_id,
                deadline=new_deadline,
                recovery_requested=True,
                boot_recovery=True,
                owners=owners,
                starts={},
            )
            state.setdefault("boot_clock_rebases", []).append(
                {
                    "prior_clock": {
                        "wall_ns": state["last_wall_ns"],
                        "monotonic": state["last_monotonic"],
                    },
                    "clock": asdict(clock),
                    "original_deadlines": original_deadlines,
                    "effective_deadlines": {**rebased, "deadline": new_deadline},
                }
            )
            for key in ("phase_deadline", "terminal_deadline"):
                if key in rebased:
                    state[key] = rebased[key]
        else:
            require(
                clock.monotonic >= state["last_monotonic"], "monotonic-clock-reversed"
            )
        state.update(
            last_monotonic=clock.monotonic,
            last_wall_ns=clock.wall_ns,
            remaining_ns=max(0, int((state["deadline"] - clock.monotonic) * 1e9)),
            last_units={r: asdict(u) for r, u in observed.units.items()},
        )
        require(set(observed.units) == set(ROLES), "observed-unit-inventory")
        require(
            set(observed.gpu_uuids) == set(self.plan["gpu_uuids"])
            and len(observed.gpu_uuids) == 8
            and observed.gpu_identity_sha256 == self.plan["gpu_identity_sha256"],
            "gpu-hardware-drift",
        )
        for role, unit in observed.units.items():
            spec = self.plan["units"][role]
            require(
                (unit.name, unit.definition_sha256, unit.cgroup)
                == (spec["name"], spec["definition_sha256"], spec["cgroup"]),
                "unit-definition-drift",
            )
            if unit.dead:
                if role in state["owners"]:
                    require(
                        not unit.invocation_id
                        or unit.invocation_id == state["owners"][role]["invocation_id"],
                        "exited-unit-invocation-drift",
                    )
                continue
            owner = state["owners"].get(role)
            if owner is None and role in state["starts"] and unit.main is not None:
                beginning = state["starts"][role]
                require(
                    re.fullmatch(r"[0-9a-f]{32}", unit.invocation_id)
                    and unit.invocation_id != beginning["prior_invocation"]
                    and beginning["issued"]
                    <= unit.entered_monotonic
                    <= clock.monotonic,
                    "new-unit-invocation-not-owned",
                )
                owner = {
                    "invocation_id": unit.invocation_id,
                    "main": asdict(unit.main),
                    "boot_id": clock.boot_id,
                }
                state["owners"][role] = owner
            if unit.main is not None:
                require(
                    owner is not None
                    and owner
                    == {
                        "invocation_id": unit.invocation_id,
                        "main": asdict(unit.main),
                        "boot_id": clock.boot_id,
                    },
                    "unit-process-not-owned",
                )
            else:
                require(
                    unit.job is not None
                    and (
                        (role in state["starts"] and unit.job.get("kind") == "start")
                        or (
                            state.get("stops", {}).get(role) == unit.invocation_id
                            and unit.job.get("kind") == "stop"
                        )
                    ),
                    "unowned-pending-job",
                )
            require(
                role != "r4" or unit.runtime_restarts == 0, "unbudgeted-runtime-retry"
            )
        for owner in observed.owners:
            roles = [
                r
                for r, u in observed.units.items()
                if u.name == owner.unit and r in {"r3", "r4", "probe"}
            ]
            require(len(roles) == 1, "foreign-gpu-owner")
            role, unit = roles[0], observed.units[roles[0]]
            require(
                owner.uuid in self.plan["gpu_uuids"]
                and owner.process in unit.members
                and owner.invocation_id == unit.invocation_id
                and owner.cgroup == unit.cgroup
                and role in state["owners"],
                "foreign-gpu-owner",
            )
            require(
                role != "probe" or owner.uuid == self.plan["probe_gpu_uuid"],
                "probe-wrong-gpu",
            )
        authority = observed.authority
        require(
            authority.phase in {"r3", "pending", "r4"}
            and authority.plan_sha256 == self.plan["freshness_plan_sha256"]
            and sha(authority.evidence_sha256),
            "unknown-durable-authority",
        )
        support = self.plan["support_transition"]
        require(
            set(observed.support) == set(support["before"]),
            "support-observation-inventory",
        )
        for name, current in observed.support.items():
            pins = {
                k: current.get(k) for k in ("definition_sha256", "environment_sha256")
            }
            require(
                any(
                    pins == {k: support[stage][name][k] for k in pins}
                    for stage in ("before", "after")
                ),
                "support-definition-drift",
            )
        state["last_support"] = dict(observed.support)
        return observed

    def _support_matches(self, obs: Observation, stage: str) -> bool:
        expected = self.plan["support_transition"][stage]
        return set(obs.support) == set(expected) and all(
            all(obs.support[name].get(k) == v for k, v in pin.items())
            and obs.support[name].get("job") is None
            for name, pin in expected.items()
        )

    def _lease(self, state: dict[str, Any], obs: Observation) -> None:
        unit, lease = obs.units["guard"], obs.lease
        require(
            unit.main is not None
            and unit.active == "active"
            and unit.job is None
            and lease is not None,
            "host-exclusion-unavailable",
        )
        assert unit.main is not None and lease is not None
        expected = {
            "path": self.plan["exclusion_path"],
            "held": True,
            "attempt_id": state["attempt_id"],
            "nonce": state["nonce"],
            "plan_sha256": self.plan_sha,
            "boot_id": obs.clock.boot_id,
            "invocation_id": unit.invocation_id,
            "owner": asdict(unit.main),
        }
        require(dict(lease) == expected, "host-exclusion-identity")

    def _start(
        self, state: dict[str, Any], obs: Observation, role: str, deadline: float
    ) -> None:
        require(
            obs.units[role].dead and not obs.owners, "start-requires-empty-ownership"
        )
        require(role not in state["starts"], "start-retry-forbidden")
        prior = state["owners"].pop(role, None)
        state["starts"][role] = {
            "issued": obs.clock.monotonic,
            "prior_invocation": obs.units[role].invocation_id
            or (prior["invocation_id"] if prior else ""),
        }
        details = (
            {
                "challenge": str(self.journal.directory / "challenge.json"),
                "challenge_sha256": state["challenge_file_sha256"],
            }
            if role == "probe"
            else {}
        )
        self._action(state, obs, "start-" + role, deadline, **details)

    def begin(self) -> dict[str, Any]:
        clock = self.host.clock()
        require(finite(clock.monotonic), "initial-clock")
        obs = self.host.observe(clock.monotonic + 5)
        require(
            obs.clock.boot_id == self.plan["initial_boot_id"]
            and finite(obs.clock.monotonic),
            "initial-boot",
        )
        initial = self.plan["initial_r3"]
        r3 = obs.units["r3"]
        require(
            r3.main is not None
            and {"invocation_id": r3.invocation_id, "main": asdict(r3.main)} == initial
            and r3.active == "active"
            and r3.job is None,
            "initial-r3-owner",
        )
        require(
            all(obs.units[r].dead for r in ("r4", "probe", "guard")), "initial-overlap"
        )
        require(
            all(
                obs.units[r].enabled == self.plan["units"][r]["initial_enabled"]
                for r in ROLES
            ),
            "initial-enablement",
        )
        require(self._support_matches(obs, "before"), "support-job-or-before-state")
        self.host.verify_inputs(self.plan, obs.clock.monotonic + 30)
        self.journal.create()
        state = {
            "format": STATE_FORMAT,
            "schema_version": 1,
            "plan_sha256": self.plan_sha,
            "attempt_id": self.plan["attempt_id"],
            "nonce": secrets.token_hex(32),
            "active_boot_id": obs.clock.boot_id,
            "started_monotonic": obs.clock.monotonic,
            "deadline": obs.clock.monotonic + TOTAL,
            "deadline_wall_ns": obs.clock.wall_ns + int(TOTAL * 1e9),
            "last_wall_ns": obs.clock.wall_ns,
            "last_monotonic": obs.clock.monotonic,
            "remaining_ns": int(TOTAL * 1e9),
            "phase": "arming",
            "phase_deadline": obs.clock.monotonic + 30,
            "recovery_requested": False,
            "recovery_mode": False,
            "boot_recovery": False,
            "handoff_complete": False,
            "runtime_authority_retired": False,
            "finished": False,
            "owners": {"r3": {**initial, "boot_id": obs.clock.boot_id}},
            "starts": {},
            "events": [],
            "revision": 0,
        }
        with self.journal.locked():
            anchor = {
                "format": "deltreltrain.freshness-guard-anchor",
                "plan_sha256": self.plan_sha,
                "attempt_id": state["attempt_id"],
                "nonce": state["nonce"],
                "initial_clock": asdict(obs.clock),
                "deadline_wall_ns": state["deadline_wall_ns"],
            }
            self.journal.save("anchor.json", anchor, new=True)
            state["anchor_sha256"] = hashlib.sha256(
                (self.journal.directory / "anchor.json").read_bytes()
            ).hexdigest()
            self._save(state, "prepared-before-arm")
            obs = self._observe(state)
            state["starts"]["guard"] = {
                "issued": obs.clock.monotonic,
                "prior_invocation": obs.units["guard"].invocation_id,
            }
            self._action(
                state,
                obs,
                "arm-guard",
                state["phase_deadline"],
                state_directory=str(self.journal.directory),
                overall_deadline=state["deadline"],
            )
        return state

    def _boundary(self, state: dict[str, Any], obs: Observation) -> None:
        require(
            obs.units["r3"].dead
            and obs.units["r4"].dead
            and obs.units["probe"].dead
            and not obs.owners,
            "stopped-boundary-requires-empty-host",
        )
        require(
            obs.units["r3"].result == "success" and obs.units["r3"].exit_code == 0,
            "clean-r3-exit-required",
        )
        require(obs.units["r3"].enabled is False, "r3-must-be-disabled-before-intent")
        boundary = self.host.capture_boundary(self.plan, state["phase_deadline"])
        require(
            set(boundary["preserved"]) == PRESERVED
            and all(sha(v) for v in boundary["preserved"].values()),
            "boundary-control-pins",
        )
        require(
            boundary["continuation_started_ns"] == self.plan["continuation_started_ns"],
            "boundary-clock-drift",
        )
        require(
            boundary["run_identity"] == self.plan["run_identity"],
            "boundary-run-identity",
        )
        for key in (
            "step",
            "examples_consumed",
            "replay_committed_samples",
            "replay_counter_updated_ns",
        ):
            require(
                type(boundary[key]) is int and boundary[key] >= 0, "boundary-counter"
            )
        require(
            boundary["clean_stop"] is True
            and sha(boundary["recovery_pointer"]["sha256"]),
            "clean-checkpoint-required",
        )
        state["boundary"] = boundary

    def _challenge(self, state: dict[str, Any], obs: Observation) -> None:
        require(
            state["deadline"] - obs.clock.monotonic >= PROBE_RESERVE,
            "insufficient-probe-recovery-reserve",
        )
        boundary = state["boundary"]
        challenge = {
            "format": "strength-freshness-probe-challenge-v1",
            "schema_version": 1,
            "attempt_id": state["attempt_id"],
            "boot_id": obs.clock.boot_id,
            "nonce": state["nonce"],
            "probe_unit": self.plan["units"]["probe"]["name"],
            "issued_monotonic": obs.clock.monotonic,
            "deadline_monotonic": obs.clock.monotonic + PROBE,
            "mode": "cuda",
            "runtime_root": self.plan["runtime_root"],
            "run_root": self.plan["run_root"],
            "output_dir": self.plan["probe_output"],
            "runtime": self.plan["probe_runtime"],
            "run_identity": self.plan["run_identity"],
            "expected_step": boundary["step"],
            "expected_examples_consumed": boundary["examples_consumed"],
            **self.plan["probe_inputs"],
            "recovery_pointer": boundary["recovery_pointer"],
            "checkpoint": boundary["checkpoint"],
            "probe_source_sha256": self.plan["probe_source_sha256"],
            "admission": {
                **self.plan["admission"],
                "recovery_pointer_sha256": boundary["recovery_pointer"]["sha256"],
            },
        }
        output = Path(self.plan["probe_output"])
        require(output.resolve() == output, "probe-output-symlink")
        require(
            not output.exists() and not output.is_symlink(),
            "probe-output-replayed",
        )
        self.journal.save("challenge.json", challenge, new=True)
        state.update(
            challenge=challenge,
            challenge_file_sha256=hashlib.sha256(
                (self.journal.directory / "challenge.json").read_bytes()
            ).hexdigest(),
            phase="probing",
            phase_deadline=challenge["deadline_monotonic"],
        )
        self._start(state, obs, "probe", state["phase_deadline"])

    def _productive(self, state: dict[str, Any], obs: Observation, role: str) -> bool:
        unit = obs.units[role]
        if unit.active != "active" or unit.job is not None or unit.main is None:
            return False
        progress = obs.progress
        if progress is not None and "contract_failure" in progress:
            require(
                progress.get("role") == role
                and isinstance(progress["contract_failure"], str)
                and bool(progress["contract_failure"]),
                "invalid-progress-contract-failure",
            )
            raise Refusal("runtime-contract-failed")
        if not progress or progress.get("role") != role:
            return False
        require(
            progress.get("invocation_id") == obs.units[role].invocation_id
            and progress.get("continuation_started_ns")
            == self.plan["continuation_started_ns"],
            "progress-identity-clock",
        )
        if role == "r4":
            require(
                progress.get("profile_sha256")
                == self.plan["admission"]["profile_sha256"]
                and progress.get("source_commit")
                == self.plan["admission"]["source_commit"],
                "loaded-runtime-drift",
            )
        require(
            progress.get("learning_rates") == self.plan["learning_rates"]
            and progress.get("ema_decay") == self.plan["ema_decay"],
            "rates-or-ema-drift",
        )
        return (
            set(progress.get("workers", [])) == set(self.plan["expected_workers"])
            and progress.get("cohorts") == 24
            and progress.get("actor_sources") == ["champion"]
            and progress.get("finite_metrics") is True
            and progress.get("arena_prefix_preserved") is True
            and progress.get("replay_committed_samples", -1)
            > state["boundary"]["replay_committed_samples"]
            and progress.get("step", -1) > state["boundary"]["step"]
        )

    def _pin_canary_processes(
        self, state: dict[str, Any], obs: Observation, role: str
    ) -> None:
        progress = obs.progress
        require(progress is not None, "canary-progress-missing")
        assert progress is not None
        workers = progress.get("worker_processes")
        coordinator = progress.get("coordinator_process")
        require(
            isinstance(workers, dict)
            and set(workers) == set(self.plan["expected_workers"])
            and isinstance(coordinator, dict),
            "canary-process-proof",
        )
        assert isinstance(workers, dict) and isinstance(coordinator, dict)
        processes = []
        for value in [coordinator, *workers.values()]:
            require(
                isinstance(value, dict)
                and set(value) == {"pid", "start_ticks"}
                and all(type(x) is int and x > 0 for x in value.values()),
                "canary-process-proof",
            )
            processes.append(Process(**value))
        require(
            len(set(processes)) == len(processes)
            and all(p in obs.units[role].members for p in processes)
            and obs.units[role].main not in processes,
            "canary-process-membership",
        )
        state["canary_processes"] = {
            "runtime_owner": copy.deepcopy(state["owners"][role]),
            "required": [asdict(p) for p in processes],
            "worker_processes": copy.deepcopy(workers),
            "coordinator_process": copy.deepcopy(coordinator),
        }

    def _backup_observation(
        self, state: dict[str, Any], obs: Observation, role: str
    ) -> str:
        """Separate missing telemetry from a lost, replaced or invalid runtime."""
        require(obs.authority.phase == role, "runtime-authority-mismatch")
        unit = obs.units[role]
        pinned = state["canary_processes"]
        current_owner = {
            "invocation_id": unit.invocation_id,
            "main": asdict(unit.main) if unit.main is not None else None,
            "boot_id": obs.clock.boot_id,
        }
        if (
            unit.active != "active"
            or unit.job is not None
            or current_owner != pinned["runtime_owner"]
            or any(Process(**p) not in unit.members for p in pinned["required"])
        ):
            raise Refusal("runtime-failed-during-backup")
        if obs.progress is None:
            return "pending"
        if not self._productive(state, obs, role):
            raise Refusal("runtime-failed-during-backup")
        if (
            obs.progress.get("worker_processes") != pinned["worker_processes"]
            or obs.progress.get("coordinator_process") != pinned["coordinator_process"]
        ):
            raise Refusal("runtime-failed-during-backup")
        return "fresh"

    def _recover(self, state: dict[str, Any], obs: Observation) -> None:
        # Guardian replacement cannot turn proof-only retirement back into a
        # startup attempt, or turn terminal owned cleanup into a runtime retry.
        if state["phase"] in {"proof-incomplete", "terminal-cleanup"}:
            state["recovery_requested"] = False
            if state["phase"] == "proof-incomplete":
                self._proof_incomplete(state, obs)
            else:
                self._terminal_cleanup(state, obs)
            return
        target = "r3" if obs.authority.phase == "r3" else "r4"
        state["recovery_target"] = (
            target  # Always derive from current durable authority.
        )
        if state["boot_recovery"] and obs.units["guard"].dead:
            state["starts"]["guard"] = {
                "issued": obs.clock.monotonic,
                "prior_invocation": obs.units["guard"].invocation_id,
            }
            self._action(
                state,
                obs,
                "arm-guard",
                min(obs.clock.monotonic + 30, state["deadline"]),
                state_directory=str(self.journal.directory),
                overall_deadline=state["deadline"],
            )
            return
        self._lease(state, obs)
        if (
            target == "r3"
            and "boundary" not in state
            and obs.units["r3"].active == "active"
            and obs.units["r3"].job is None
        ):
            if (
                not self._support_matches(obs, "before")
                or obs.units["r3"].enabled
                != self.plan["units"]["r3"]["initial_enabled"]
            ):
                self._action(
                    state,
                    obs,
                    "restore-support-before-stop",
                    min(obs.clock.monotonic + TAIL, state["deadline"]),
                    support_transition=self.plan["support_transition"],
                )
                return
            state.update(
                phase="retiring",
                runtime_authority_retired=True,
                outcome="aborted-before-stop-r3-unchanged",
            )
            self._save(state, "no-outage-no-migration-authority")
            return
        probe = obs.units["probe"]
        if not probe.dead:
            require(
                "probe" in state["owners"] or "probe" in state["starts"],
                "unowned-probe-cleanup",
            )
            self._cleanup_request(state, obs, "probe", EMPTY)
            return
        other = "r4" if target == "r3" else "r3"
        if not obs.units[other].dead:
            require(other in state["owners"], "unowned-other-runtime")
            self._cleanup_request(state, obs, other, STOP)
            return
        if obs.authority.phase == "pending":
            require(
                not obs.owners and obs.units["r4"].dead, "repair-requires-empty-host"
            )
            require(
                state["deadline"] - obs.clock.monotonic >= APPLY_RESERVE,
                "insufficient-forward-repair-reserve",
            )
            self._action(
                state,
                obs,
                "repair-r4",
                min(obs.clock.monotonic + APPLY, state["deadline"] - START_RESERVE),
            )
            return
        if obs.units[target].dead:
            require(not obs.owners, "recovery-foreign-owners")
            require(
                state["deadline"] - obs.clock.monotonic >= START_RESERVE,
                "insufficient-restart-reserve",
            )
            if "boundary" not in state:
                state["phase_deadline"] = min(
                    obs.clock.monotonic + STOP, state["deadline"] - START_RESERVE
                )
                self._boundary(state, obs)
            if target == "r3":
                self.host.check_boundary(
                    state["boundary"], state["deadline"] - START_RESERVE
                )
            state.update(
                phase="starting-" + target,
                phase_deadline=obs.clock.monotonic + STARTUP,
                recovery_requested=False,
            )
            self._start(state, obs, target, state["phase_deadline"])
            return
        require(target in state["owners"], "unowned-running-recovery-target")
        if obs.units[target].job is not None or obs.units[target].active != "active":
            self._cleanup_request(state, obs, target, STOP)
            return
        if state["phase"] in {
            "starting-" + target,
            "canary-" + target,
            "backup-" + target,
        }:
            state["recovery_requested"] = False
        else:
            state.update(
                phase="starting-" + target,
                phase_deadline=min(
                    obs.clock.monotonic + STARTUP, state["deadline"] - CANARY - TAIL
                ),
                recovery_requested=False,
            )

    def _cleanup_request(
        self, state: dict[str, Any], obs: Observation, role: str, cap: float
    ) -> None:
        deadlines = state.setdefault("cleanup_deadlines", {})
        deadline = deadlines.setdefault(
            role, min(obs.clock.monotonic + cap, state["deadline"] - START_RESERVE)
        )
        require(obs.clock.monotonic < deadline, "owned-cleanup-timeout")
        requests = state.setdefault("cleanup_requested", [])
        if role not in requests:
            requests.append(role)
            self._action(state, obs, "stop-" + role, deadline)

    def tick(self, *, recover: bool = False) -> dict[str, Any]:
        with self.journal.locked():
            state = self.journal.read("state.json")
            require(
                state.get("format") == STATE_FORMAT
                and state.get("plan_sha256") == self.plan_sha,
                "state-plan-mismatch",
            )
            if state["finished"]:
                return state  # Never regain service authority after completion.
            anchor = self.journal.read("anchor.json")
            require(
                hashlib.sha256(
                    (self.journal.directory / "anchor.json").read_bytes()
                ).hexdigest()
                == state.get("anchor_sha256")
                and all(
                    state.get(k) == anchor[k]
                    for k in ("plan_sha256", "attempt_id", "nonce", "deadline_wall_ns")
                ),
                "guard-anchor-changed",
            )
            require(
                state["started_monotonic"] == anchor["initial_clock"]["monotonic"]
                and 0 <= state["remaining_ns"] <= int(TOTAL * 1e9),
                "guard-budget-state",
            )
            if state["active_boot_id"] == anchor["initial_clock"]["boot_id"]:
                require(
                    state["deadline"] == anchor["initial_clock"]["monotonic"] + TOTAL,
                    "deadline-extended",
                )
            if state.get("runtime_authority_retired"):
                return self._retire_only(state)
            obs: Observation | None = None
            try:
                obs = self._observe(state)
                require(
                    obs.clock.monotonic < state["deadline"], "original-deadline-expired"
                )
                if (recover or not obs.controller_alive) and not state["recovery_mode"]:
                    state.update(recovery_requested=True, recovery_mode=True)
                if state["recovery_requested"]:
                    self._recover(state, obs)
                elif state["phase"] == "terminal-cleanup":
                    self._terminal_cleanup(state, obs)
                elif state["phase"] == "proof-incomplete":
                    self._proof_incomplete(state, obs)
                else:
                    self._advance(state, obs)
                self._save(state, "observed")
            except Refusal as error:
                code = str(error)
                if (
                    code
                    in {
                        "backup-timeout-productive-proof-incomplete",
                        "backup-timeout-owned-runtime-pending-telemetry",
                    }
                    and obs is not None
                ):
                    state.update(
                        phase="proof-incomplete",
                        failure=code,
                        recovery_requested=False,
                        recovery_mode=True,
                        terminal_deadline=min(
                            obs.clock.monotonic + STOP, state["deadline"]
                        ),
                    )
                    self._save(
                        state, "preserve-productive-runtime-without-proof-success"
                    )
                elif obs is not None and code in {
                    "phase-timeout-no-implicit-retry",
                    "runtime-failed-during-backup",
                    "runtime-contract-failed",
                    "runtime-authority-mismatch",
                }:
                    state.update(
                        phase="terminal-cleanup",
                        terminal_failure=code,
                        recovery_requested=False,
                        recovery_mode=True,
                        terminal_deadline=min(
                            obs.clock.monotonic + STOP, state["deadline"]
                        ),
                    )
                    self._save(state, "timeout-reserved-cleanup-no-retry")
                elif code in {
                    "insufficient-probe-recovery-reserve",
                    "receipt-apply-reserve-or-authority",
                } or code.startswith(
                    (
                        "probe-",
                        "proof-",
                        "unsafe-proof",
                        "actual-cuda",
                        "nonempty-execution",
                    )
                ):
                    state.update(
                        recovery_requested=True, recovery_mode=True, failure=code
                    )
                    self._save(state, "invalid-probe-recover-without-admission")
                else:
                    state.update(phase="blocked", failure=code, finished=True)
                    self._save(state, "blocked-no-ownership-or-budget-waiver")
            except Exception as error:
                state.update(
                    recovery_requested=True,
                    recovery_mode=True,
                    failure=type(error).__name__,
                )
                self._save(state, "operation-failed-reobserve-durable-intent")
            return state

    def _proof_incomplete(self, state: dict[str, Any], obs: Observation) -> None:
        """A backup delay cannot justify stopping an already qualified canary."""
        self._lease(state, obs)
        role = state["first_progress"]["role"]
        observation = self._backup_observation(state, obs, role)
        deadline = state["terminal_deadline"]
        require(obs.clock.monotonic < deadline, "incomplete-proof-retirement-timeout")
        other = "r3" if role == "r4" else "r4"
        require(
            obs.units[other].dead and obs.units["probe"].dead,
            "incomplete-proof-transient-overlap",
        )
        stage = "after" if role == "r4" else "before"
        enabled = "committed_enabled" if role == "r4" else "initial_enabled"
        if not self._support_matches(obs, stage) or not all(
            obs.units[r].enabled == self.plan["units"][r][enabled]
            for r in ("r3", "r4", "probe")
        ):
            if not state.get("incomplete_support_requested"):
                state["incomplete_support_requested"] = True
                self._action(
                    state,
                    obs,
                    "restore-support-without-proof-" + role,
                    deadline,
                    support_transition=self.plan["support_transition"],
                    support_stage=stage,
                    proof_closure=str(self.journal.directory / "proof-closure.json"),
                    proof_closure_sha256=state.get("proof_closure_sha256")
                    or state.get("existing_proof_closure", {}).get("observed_sha256"),
                    required_proof_sha256=state.get("required_proof_sha256", []),
                )
            return
        state.update(
            phase="retiring",
            runtime_authority_retired=True,
            outcome=(
                "owned-" + role + "-pending-telemetry"
                if observation == "pending"
                else "productive-" + role + "-proof-incomplete"
            ),
            recovery_required=True,
            proof_closure_required=True,
        )
        self._save(state, "owned-runtime-preserved-backup-not-certified")

    def _terminal_cleanup(self, state: dict[str, Any], obs: Observation) -> None:
        """Drain proven owners with the reserved tail; never restart or rollback."""
        self._lease(state, obs)
        deadline = state["terminal_deadline"]
        require(obs.clock.monotonic < deadline, "terminal-owned-cleanup-timeout")
        if not state.get("terminal_restarts_fenced"):
            self._action(state, obs, "fence-terminal-restarts", deadline)
            state["terminal_restarts_fenced"] = True
            return
        for role in ("probe", "r3", "r4"):
            if not obs.units[role].dead:
                require(
                    role in state["owners"] or role in state["starts"],
                    "terminal-unowned-unit",
                )
                requested = state.setdefault("terminal_stops", [])
                if role not in requested:
                    requested.append(role)
                    self._action(state, obs, "stop-" + role, deadline)
                return
        require(not obs.owners, "terminal-foreign-gpu-owner")
        require(
            all(not obs.units[r].enabled for r in ("r3", "r4", "probe"))
            and all(
                row.get("enabled") is False and row.get("job") is None
                for row in obs.support.values()
            ),
            "terminal-restart-fence-unverified",
        )
        state.update(
            phase="retiring",
            runtime_authority_retired=True,
            outcome="failed-closed-no-runtime",
            failure=state["terminal_failure"],
        )
        self._save(state, "owned-workers-drained-no-runtime-recovery-authority")

    def _retire_only(self, state: dict[str, Any]) -> dict[str, Any]:
        """No runtime authority remains after sealed handoff completion.

        Bounded, idempotent cleanup of this guard/lease may run on a later boot;
        it does not extend the experiment deadline or permit runtime actions.
        A qualified adapter must never release a different attempt's lease.
        """
        require(
            self.plan["guard_source_sha256"] == source_sha256(),
            "executing-guard-source-mismatch",
        )
        now = self.host.clock().monotonic
        require(finite(now), "retirement-clock")
        obs = self.host.observe(now + 5)
        unit = obs.units["guard"]
        spec = self.plan["units"]["guard"]
        require(
            (unit.name, unit.definition_sha256, unit.cgroup)
            == (spec["name"], spec["definition_sha256"], spec["cgroup"]),
            "retirement-guard-definition-drift",
        )
        lease = obs.lease
        ours = (
            lease is not None
            and lease.get("attempt_id") == state["attempt_id"]
            and lease.get("nonce") == state["nonce"]
            and lease.get("plan_sha256") == self.plan_sha
        )
        require(lease is None or ours, "retirement-foreign-lease")
        if not unit.dead and unit.main is not None:
            owner = state["owners"].get("guard")
            current = {
                "invocation_id": unit.invocation_id,
                "main": asdict(unit.main),
                "boot_id": obs.clock.boot_id,
            }
            lease_owner = (
                ours
                and lease is not None
                and all(
                    lease.get(key) == value
                    for key, value in {
                        "path": self.plan["exclusion_path"],
                        "held": True,
                        "boot_id": obs.clock.boot_id,
                        "invocation_id": unit.invocation_id,
                        "owner": asdict(unit.main),
                    }.items()
                )
            )
            require(owner == current or lease_owner, "retirement-foreign-guard")
        if unit.dead and not unit.enabled and not ours:
            state.update(phase="done", finished=True)
            self._save(state, "guard-and-own-lease-retired")
            return state
        state["phase"] = "retiring"
        self._save(state, "own-guard-retirement-intent-no-runtime-authority")
        try:
            self.host.perform(
                "retire-guard-only",
                {
                    "guard_plan_sha256": self.plan_sha,
                    "attempt_id": state["attempt_id"],
                    "nonce": state["nonce"],
                    "expected_boot_id": obs.clock.boot_id,
                    "expected_guard": asdict(unit),
                    "expected_own_lease": dict(lease) if ours and lease else None,
                },
                now + 15,
            )
        except Exception as error:
            state["retirement_failure"] = type(error).__name__
            self._save(state, "retirement-incomplete-no-rollback")
        return state

    def _advance(self, state: dict[str, Any], obs: Observation) -> None:
        now, phase = obs.clock.monotonic, state["phase"]
        if now >= state["phase_deadline"]:
            if phase.startswith("backup-"):
                observation = self._backup_observation(state, obs, phase[-2:])
                raise Refusal(
                    "backup-timeout-owned-runtime-pending-telemetry"
                    if observation == "pending"
                    else "backup-timeout-productive-proof-incomplete"
                )
            if phase.startswith(("starting-", "canary-", "backup-")):
                raise Refusal("phase-timeout-no-implicit-retry")
            state.update(recovery_requested=True, recovery_mode=True)
            self._recover(state, obs)
            return
        if phase == "arming" and (
            obs.units["guard"].main is None
            or obs.guard_ack is None
            or obs.lease is None
        ):
            return
        self._lease(state, obs)
        if phase == "arming":
            expected = {
                "plan_sha256": self.plan_sha,
                "attempt_id": state["attempt_id"],
                "nonce": state["nonce"],
                "boot_id": obs.clock.boot_id,
                "invocation_id": obs.units["guard"].invocation_id,
                "deadline": state["deadline"],
            }
            require(
                obs.guard_ack is not None and dict(obs.guard_ack) == expected,
                "guard-not-acknowledged",
            )
            require(obs.units["guard"].enabled is True, "persistent-guard-not-enabled")
            require(self._support_matches(obs, "before"), "support-job-before-stop")
            state.update(
                phase="stopping-r3",
                phase_deadline=min(
                    state["started_monotonic"] + STOP, state["deadline"] - PROBE_RESERVE
                ),
            )
            self._action(
                state, obs, "pause-support-and-stop-r3", state["phase_deadline"]
            )
        elif phase == "stopping-r3":
            if (
                obs.units["r3"].dead
                and not obs.owners
                and all(row.get("job") is None for row in obs.support.values())
            ):
                require(obs.authority.phase == "r3", "authority-changed-before-probe")
                self._boundary(state, obs)
                self._challenge(state, obs)
        elif phase == "probing":
            require(obs.authority.phase == "r3", "authority-changed-during-probe")
            if (Path(self.plan["probe_output"]) / "result.json").exists():
                state["probe_proof"] = validate_probe(
                    self.plan, state, obs.units["probe"], now
                )
                state.update(
                    phase="cleaning-probe",
                    phase_deadline=min(now + EMPTY, state["deadline"] - APPLY_RESERVE),
                )
                # Visibility of result.json does not imply exit0 or CUDA release.
                # Allow natural successful exit; a cleanup timeout invalidates
                # admission and enters owned cleanup/recovery instead.
            elif obs.units["probe"].dead and "probe" in state["owners"]:
                raise RuntimeError("probe-exited-without-result")
        elif phase == "cleaning-probe":
            if obs.units["probe"].dead and not obs.owners:
                require(
                    obs.units["probe"].result == "success"
                    and obs.units["probe"].exit_code == 0,
                    "probe-unsuccessful-exit",
                )
                proof = validate_probe(self.plan, state, obs.units["probe"], now)
                require(
                    proof == state["probe_proof"], "probe-proof-changed-during-cleanup"
                )
                self.host.check_boundary(state["boundary"], state["phase_deadline"])
                require(
                    obs.authority.phase == "r3"
                    and state["deadline"] - now >= APPLY_RESERVE,
                    "receipt-apply-reserve-or-authority",
                )
                receipt = {
                    "format": RECEIPT_FORMAT,
                    "schema_version": 1,
                    "status": "passed",
                    **state["challenge"]["admission"],
                    "checks": {
                        **proof["result"]["checks"],
                        "all_workers_released": True,
                    },
                    "guard_evidence": {
                        "guard_plan_sha256": self.plan_sha,
                        "attempt_id": state["attempt_id"],
                        "boot_id": obs.clock.boot_id,
                        "nonce": state["nonce"],
                        "probe_invocation_id": state["owners"]["probe"][
                            "invocation_id"
                        ],
                        "probe_result_sha256": proof["result_sha256"],
                        "released_monotonic": now,
                        "deadline": state["deadline"],
                        "boundary_sha256": digest(state["boundary"]),
                    },
                }
                self.journal.save("cuda-qualification.json", receipt, new=True)
                checksum = hashlib.sha256(
                    (self.journal.directory / "cuda-qualification.json").read_bytes()
                ).hexdigest()
                state.update(
                    phase="applying",
                    phase_deadline=now + APPLY,
                    qualification_sha256=checksum,
                )
                self._action(
                    state,
                    obs,
                    "apply-r4",
                    state["phase_deadline"],
                    receipt=str(self.journal.directory / "cuda-qualification.json"),
                    receipt_sha256=checksum,
                )
        elif phase == "applying":
            if obs.authority.phase == "r4":
                require(
                    not obs.owners and state["deadline"] - now >= START_RESERVE,
                    "start-r4-reserve",
                )
                state.update(phase="starting-r4", phase_deadline=now + STARTUP)
                self._start(state, obs, "r4", state["phase_deadline"])
            elif obs.authority.phase == "pending":
                state["recovery_requested"] = True
        elif phase in ("starting-r3", "starting-r4"):
            role = phase[-2:]
            require(obs.authority.phase == role, "runtime-authority-mismatch")
            if obs.units[role].dead:
                state.update(recovery_requested=True, recovery_mode=True)
                return
            if self._productive(state, obs, role):
                self._pin_canary_processes(state, obs, role)
                state.update(
                    phase="canary-" + role,
                    phase_deadline=min(now + CANARY, state["deadline"] - STOP),
                    first_progress=dict(obs.progress or {}),
                )
        elif phase in ("canary-r3", "canary-r4"):
            role = phase[-2:]
            if (
                self._productive(state, obs, role)
                and obs.progress is not None
                and obs.progress["step"] > state["first_progress"]["step"]
                and obs.progress.get("neural_work", 0)
                > state["first_progress"].get("neural_work", 0)
            ):
                proof_path = self.journal.directory / "proof-closure.json"
                if (
                    proof_path.exists()
                    or proof_path.is_symlink()
                    or "proof_closure_sha256" in state
                ):
                    # A previous action or crash may have published this file.
                    # Preserve it; it cannot certify this recovered runtime.
                    # A fresh canary permits only explicit incomplete closure.
                    observed_hash = None
                    if proof_path.exists() or proof_path.is_symlink():
                        observed_hash = hashlib.sha256(
                            _proof_file(
                                self.journal.directory, "proof-closure.json", 2**20
                            )
                        ).hexdigest()
                    state.update(
                        phase="proof-incomplete",
                        failure="prior-proof-preserved-no-new-backup",
                        recovery_requested=False,
                        recovery_mode=True,
                        terminal_deadline=min(now + STOP, state["deadline"]),
                        existing_proof_closure={
                            "observed_sha256": observed_hash,
                            "journaled_sha256": state.get("proof_closure_sha256"),
                            "status": "diagnostic-only-not-completed-proof",
                        },
                    )
                    self._save(state, "prior-proof-preserved-after-fresh-canary")
                    return
                state.update(
                    phase="backup-" + role,
                    phase_deadline=min(now + TAIL, state["deadline"] - STOP),
                )
                required = {
                    value
                    for values in self.plan["proof_closure_inputs"].values()
                    for value in values
                }
                required.add(digest(state["boundary"]))
                required.update(
                    (
                        state["boundary"]["checkpoint"]["sha256"],
                        state["boundary"]["recovery_pointer"]["sha256"],
                    )
                )
                if "challenge" in state:
                    required.add(state["challenge_file_sha256"])
                    # Failed child output remains diagnostic evidence. Never
                    # follow a rejected child's arbitrary artifact references.
                    for name in ("started.json", "result.json", "observations.jsonl"):
                        output = Path(self.plan["probe_output"])
                        if (output / name).exists():
                            required.add(
                                hashlib.sha256(
                                    _proof_file(output, name, 16 * 2**20)
                                ).hexdigest()
                            )
                if "probe_proof" in state:
                    required.update(
                        (
                            state["probe_proof"]["result_sha256"],
                            state["probe_proof"]["started_sha256"],
                            state["challenge_file_sha256"],
                        )
                    )
                    required.update(
                        pin["sha256"]
                        for pin in state["probe_proof"]["result"]["evidence"]
                    )
                if "qualification_sha256" in state:
                    required.add(state["qualification_sha256"])
                closure = {
                    "guard_plan_sha256": self.plan_sha,
                    "role": role,
                    "required_sha256": sorted(required),
                    "guard_history": list(state["events"]),
                    "support_transition_sha256": self.plan["support_transition"][
                        "sha256"
                    ],
                }
                self.journal.save("proof-closure.json", closure, new=True)
                state["proof_closure_sha256"] = hashlib.sha256(
                    (self.journal.directory / "proof-closure.json").read_bytes()
                ).hexdigest()
                state["required_proof_sha256"] = sorted(
                    required | {state["proof_closure_sha256"]}
                )
                self._action(
                    state,
                    obs,
                    "restore-support-and-backup-" + role,
                    state["phase_deadline"],
                    proof_closure=str(self.journal.directory / "proof-closure.json"),
                    proof_closure_sha256=state["proof_closure_sha256"],
                    required_proof_sha256=state["required_proof_sha256"],
                    support_transition=self.plan["support_transition"],
                    support_stage="after" if role == "r4" else "before",
                )
        elif phase in ("backup-r3", "backup-r4"):
            role = phase[-2:]
            if self._backup_observation(state, obs, role) == "pending":
                return
            backup = obs.backup
            if backup is not None:
                require(
                    backup.get("status") == "committed"
                    and backup.get("guard_plan_sha256") == self.plan_sha
                    and backup.get("role") == role
                    and backup.get("closure_verified") is True
                    and sha(backup.get("catalog_sha256")),
                    "backup-closure",
                )
                require(
                    backup.get("proof_closure_sha256") == state["proof_closure_sha256"]
                    and set(state["required_proof_sha256"])
                    <= set(backup.get("verified_artifact_sha256", [])),
                    "external-proof-closure-incomplete",
                )
                if not self._support_matches(
                    obs, "after" if role == "r4" else "before"
                ):
                    return
                stage = "committed_enabled" if role == "r4" else "initial_enabled"
                require(
                    all(
                        obs.units[r].enabled == self.plan["units"][r][stage]
                        for r in ("r3", "r4", "probe")
                    ),
                    "runtime-enablement-not-finalized",
                )
                state.update(
                    phase="retiring",
                    handoff_complete=True,
                    runtime_authority_retired=True,
                    outcome="committed-r4" if role == "r4" else "restored-r3",
                    backup=dict(backup),
                )
                self._save(state, "complete-before-guard-retirement")
        else:
            raise Refusal("unknown-guard-phase")
