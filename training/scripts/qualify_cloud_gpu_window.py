"""One exclusive, finite cloud GPU qualification; always restore the old service.

This stdlib-only operator consumes an independently pinned *execution* plan, not
an inventory/proposal. Production, stage and CPU probe units must already exist;
it never edits their files, configs, models, release links or credentials. The
run mode arms a separate systemd watchdog before requesting production stop.
Restore is fenced, idempotent, and refuses overlap even when recovery is blocked.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shlex
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, TypeGuard
import urllib.error
import urllib.parse
import urllib.request

FORMAT = "deltrelserve.exclusive-gpu-qualification"
TOTAL = 900.0
DRAIN = 210.0
EMPTY = 15.0
RESTORE = 90.0
STAGE = 360.0
OBSERVATION_RESERVE = 15.0
# t600 cannot reserve both the empty-owner gate and the final 90-second recovery.
LATEST_STOP_REQUEST = TOTAL - DRAIN - EMPTY - RESTORE  # t585 is not free dispatch time.
DISPATCH_RESERVE = 45.0  # bounded owner/stop RPCs, completion observations and polling
RESTORE_BY = LATEST_STOP_REQUEST - DISPATCH_RESERVE  # watchdog begins at t540.
REQUIRED_CHECKS = ("semantic_16", "standard_1", "standard_8", "deep_1")
PROBE_BINDINGS = (
    "run_id",
    "plan_sha256",
    "boot_id",
    "challenge",
    "probe_unit",
    "model_identity",
    "model_step",
)
PROPERTIES = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "MainPID",
    "InvocationID",
    "Result",
    "ExecMainStatus",
    "ExecStart",
    "WorkingDirectory",
    "User",
    "Group",
    "EnvironmentFiles",
    "FragmentPath",
    "DropInPaths",
    "Restart",
    "KillMode",
    "TimeoutStopUSec",
    "RuntimeMaxUSec",
    "SendSIGKILL",
    "Type",
    "PrivateDevices",
    "Job",
)
IDENTITY_PROPERTIES = (
    "WorkingDirectory",
    "User",
    "Group",
    "EnvironmentFiles",
    "FragmentPath",
    "DropInPaths",
    "Restart",
    "KillMode",
    "TimeoutStopUSec",
    "RuntimeMaxUSec",
    "SendSIGKILL",
    "Type",
    "PrivateDevices",
)


class Refusal(RuntimeError):
    """A safe, non-secret failure code suitable for a durable receipt."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2**20:
        raise Refusal("unsafe-control-file")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise Refusal("invalid-control-object")
    return value


def write_json(path: Path, value: object, *, replace: bool = True) -> None:
    fd, name = tempfile.mkstemp(prefix=".qualification-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(encoded(value))
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(name, path)
        else:
            os.link(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def _absolute(value: object) -> str:
    if (
        not isinstance(value, str)
        or not Path(value).is_absolute()
        or ".." in Path(value).parts
    ):
        raise Refusal("absolute-path-required")
    return value


def _sha(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise Refusal("explicit-sha256-required")
    return value


def load_plan(path: Path, expected_sha256: str) -> dict[str, Any]:
    _sha(expected_sha256)
    if digest(path.read_bytes()) != expected_sha256:
        raise Refusal("plan-hash-mismatch")
    plan = read_json(path)
    if (
        plan.get("format") != FORMAT
        or plan.get("schema_version") != 1
        or plan.get("approved_for_exclusive_window") is not True
    ):
        raise Refusal("execution-plan-required")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,48}", plan.get("run_id", "")) is None:
        raise Refusal("invalid-run-id")
    names = {
        "production": "deltrelserve.service",
        "stage": f"deltrelserve-qualification-{plan['run_id']}-stage.service",
        "probe": f"deltrelserve-qualification-{plan['run_id']}-probe.service",
    }
    for role, name in names.items():
        unit = plan[role]
        if unit["unit"] != name or set(unit["properties"]) != set(IDENTITY_PROPERTIES):
            raise Refusal("unit-identity-incomplete")
        for prop in unit["properties"].values():
            if not isinstance(prop, str):
                raise Refusal("unit-property-must-be-string")
        if (
            unit["properties"]["DropInPaths"]
            or unit["properties"]["KillMode"] != "control-group"
            or unit["properties"]["SendSIGKILL"] != "yes"
        ):
            raise Refusal("unsupported-unit-lifecycle")
        if role != "production" and (
            unit["properties"]["Restart"] != "no"
            or unit["properties"]["Type"] not in ("simple", "exec")
        ):
            raise Refusal("unbounded-qualification-unit")
        if role == "stage" and (
            unit["properties"]["RuntimeMaxUSec"] != "6min"
            or unit["properties"]["TimeoutStopUSec"] != "3min 30s"
        ):
            raise Refusal("staging-bounds-mismatch")
        if role == "probe" and (
            unit["properties"]["PrivateDevices"] != "yes"
            or unit["properties"]["RuntimeMaxUSec"] != "6min"
            or unit["properties"]["TimeoutStopUSec"] != "5s"
        ):
            raise Refusal("probe-bounds-mismatch")
        if role == "probe" and (
            unit["properties"]["User"] != "root"
            or unit["required_checks"] != list(REQUIRED_CHECKS)
        ):
            raise Refusal("root-probe-contract-required")
        if role == "production" and unit["properties"]["TimeoutStopUSec"] != "3min 30s":
            raise Refusal("production-drain-bound-mismatch")
        _absolute(unit["release"])
        _absolute(unit["properties"]["FragmentPath"])
        if not isinstance(unit["exec_start"], str) or "\n" in unit["exec_start"]:
            raise Refusal("invalid-exec-start-pin")
        pins = unit["files"]
        if not isinstance(pins, list) or not pins or len(pins) > 256:
            raise Refusal("unit-file-pins-required")
        for pin in pins:
            _absolute(pin["path"])
            _sha(pin["sha256"])
            if type(pin["bytes"]) is not int or pin["bytes"] < 0:
                raise Refusal("invalid-file-size")
        if unit["properties"]["FragmentPath"] not in {p["path"] for p in pins}:
            raise Refusal("unit-fragment-not-pinned")
        for env in unit["env_files"]:
            _absolute(env)
        if role != "probe":
            for required in (
                "config",
                "profile",
                "pointer",
                "manifest",
                "checkpoint",
                "native",
                "entrypoint",
            ):
                if unit[required] not in {p["path"] for p in pins}:
                    raise Refusal("serving-artifact-not-pinned")
            _sha(unit["model_identity"].removeprefix("sha256-"))
            if not unit["model_identity"].startswith("sha256-"):
                raise Refusal("invalid-model-identity")
            expected_step = 566428 if role == "production" else 572377
            if unit["model_step"] != expected_step:
                raise Refusal("unexpected-model-step")
    for key in ("qualification_plan", "cpu_preparation_bounds"):
        ref = plan[key]
        _absolute(ref["path"])
        _sha(ref["sha256"])
        if digest(Path(ref["path"]).read_bytes()) != ref["sha256"]:
            raise Refusal("preparation-reference-changed")
    _absolute(plan["current_link"])
    if (
        type(plan["production"]["initial_main_pid"]) is not int
        or plan["production"]["initial_main_pid"] <= 0
        or re.fullmatch(r"[0-9a-f]{32}", plan["production"]["initial_invocation_id"])
        is None
    ):
        raise Refusal("production-invocation-pin-required")
    for role in ("production", "stage"):
        health = plan[role]["health"]
        if not isinstance(health, list) or not health:
            raise Refusal("health-checks-required")
        for check in health:
            url = urllib.parse.urlsplit(check["url"])
            if (
                url.username
                or url.password
                or url.query
                or url.fragment
                or url.scheme not in ("http", "https")
            ):
                raise Refusal("unsafe-health-url")
            if url.scheme == "http" and url.hostname != "127.0.0.1":
                raise Refusal("http-must-be-loopback")
            if check.get("env_file") is not None:
                credential_loopback = f"http://127.0.0.1:{8080 if role == 'production' else 8081}/v2/health"
                if check["url"] != credential_loopback:
                    raise Refusal("credentials-require-exact-role-loopback")
                if (
                    check["env_file"] not in plan[role]["env_files"]
                    or check.get("env_name") != "DELTRELSERVE_BEARER_TOKEN"
                ):
                    raise Refusal("credential-path-not-pinned")
        loopback = (
            f"http://127.0.0.1:{8080 if role == 'production' else 8081}/v2/health"
        )
        if not any(h["url"] == loopback for h in health):
            raise Refusal("loopback-identity-check-required")
    if not any(h["url"].startswith("https://") for h in plan["production"]["health"]):
        raise Refusal("public-tls-health-required")
    proposal = read_json(Path(plan["qualification_plan"]["path"]))
    bounds = read_json(Path(plan["cpu_preparation_bounds"]["path"]))
    phase = proposal["phase_c_exclusive_gpu_qualification"]
    required = {
        "total_outage_budget_seconds": 900,
        "production_stop_max_seconds": 210,
        "staging_cleanup_max_seconds": 210,
        "empty_gate_budget_seconds": 15,
        "production_restore_and_health_max_seconds": 90,
        "gpu_execution_max_seconds": 360,
    }
    if (
        any(phase.get(k) != v for k, v in required.items())
        or bounds["plan_sha256"] != plan["qualification_plan"]["sha256"]
    ):
        raise Refusal("frozen-budget-reference-mismatch")
    if proposal["paths"]["rollback_release"] != plan["production"]["release"] or any(
        plan[role]["release"] != proposal["paths"]["new_release"]
        for role in ("stage", "probe")
    ):
        raise Refusal("frozen-release-reference-mismatch")
    prod_config = next(
        pin
        for pin in plan["production"]["files"]
        if pin["path"] == plan["production"]["config"]
    )
    if prod_config["sha256"] != proposal["paths"]["rollback_production_yaml_sha256"]:
        raise Refusal("frozen-production-config-mismatch")
    return plan


@contextmanager
def _http_wall_bound(seconds: float):
    # This operator has one main thread. A socket timeout alone is reset on each
    # receive and would let a slow HTTP peer exceed the absolute recovery bound.
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError("bounded-health-timeout")

    signal.signal(signal.SIGALRM, expired)
    prior_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if prior_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *prior_timer)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError("health redirects are not admitted")


class Host:
    """Small syscall boundary; CPU tests replace it without any service/GPU access."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def boot_id(self) -> str:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    def process(self, pid: int) -> dict[str, Any] | None:
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z":
                return None
            return {"pid": pid, "start_ticks": int(fields[19])}
        except (FileNotFoundError, ProcessLookupError):
            return None

    def command(self, argv: list[str], deadline: float) -> str:
        remaining = deadline - self.now()
        if remaining <= 0:
            raise Refusal("monotonic-deadline-expired")
        try:
            done = subprocess.run(
                argv,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=min(5.0, remaining),
                env={
                    "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
                    "LANG": "C",
                    "SYSTEMD_COLORS": "0",
                },
            )
        except (subprocess.SubprocessError, OSError):
            raise Refusal("bounded-host-command-failed") from None
        return done.stdout

    def unit(self, name: str, deadline: float) -> dict[str, str]:
        text = self.command(
            [
                "systemctl",
                "show",
                name,
                "--no-pager",
                "--property=" + ",".join(PROPERTIES),
            ],
            deadline,
        )
        values = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
        if set(values) != set(PROPERTIES):
            raise Refusal("incomplete-systemd-evidence")
        # Runtime PID/timestamps are deliberately excluded from the ExecStart pin.
        values["ExecStart"] = values["ExecStart"].split(" ; ignore_errors=", 1)[0]
        return values

    def action(self, verb: str, name: str, deadline: float) -> None:
        self.command(["systemctl", verb, "--no-block", name], deadline)

    def owners(self, deadline: float) -> set[int]:
        text = self.command(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            deadline,
        )
        rows = [line.strip() for line in text.splitlines() if line.strip()]
        if any(not x.isdecimal() or int(x) <= 0 for x in rows):
            raise Refusal("invalid-nvml-owner-evidence")
        return {int(x) for x in rows}

    def port_closed(self, port: int, deadline: float) -> bool:
        remaining = deadline - self.now()
        if remaining <= 0:
            raise Refusal("monotonic-deadline-expired")
        with socket.socket() as sock:
            sock.settimeout(min(0.5, remaining))
            return sock.connect_ex(("127.0.0.1", port)) == errno.ECONNREFUSED

    def health(self, check: dict[str, str], deadline: float) -> dict[str, Any] | None:
        headers = {}
        if "env_file" in check:
            # Never source arbitrary shell or include credentials in argv/receipts/errors.
            entries = {}
            for line in Path(check["env_file"]).read_text().splitlines():
                if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    parsed = shlex.split(value)
                    if len(parsed) == 1:
                        entries[key.strip()] = parsed[0]
            token = entries.get(check["env_name"])
            if not token:
                raise Refusal("protected-credential-unavailable")
            headers["Authorization"] = "Bearer " + token
        remaining = deadline - self.now()
        if remaining <= 0:
            raise Refusal("monotonic-deadline-expired")
        try:
            request = urllib.request.Request(check["url"], headers=headers)
            with (
                _http_wall_bound(min(3.0, remaining)),
                urllib.request.build_opener(_NoRedirect).open(
                    request, timeout=min(3.0, remaining)
                ) as response,
            ):
                data = response.read(2**20 + 1)
            if len(data) > 2**20:
                return None
            result = json.loads(data)
            return result if isinstance(result, dict) else None
        except (OSError, ValueError, urllib.error.URLError):
            return None

    def check_files(self, unit: dict[str, Any], deadline: float) -> None:
        if Path(unit["release"]).resolve(strict=True) != Path(unit["release"]):
            raise Refusal("release-path-changed")
        for pin in unit["files"]:
            path = Path(pin["path"])
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != pin["bytes"]
            ):
                raise Refusal("artifact-pin-mismatch")
            h = hashlib.sha256()
            with path.open("rb") as stream:
                while block := stream.read(2**20):
                    if self.now() >= deadline:
                        raise Refusal("artifact-verification-deadline")
                    h.update(block)
            if h.hexdigest() != pin["sha256"]:
                raise Refusal("artifact-pin-mismatch")
        for value in unit["env_files"]:
            path = Path(value)
            info = path.stat()
            if (
                path.is_symlink()
                or not stat.S_ISREG(info.st_mode)
                or info.st_uid != 0
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_size > 2**20
            ):
                raise Refusal("credential-file-not-protected")

    def check_link(self, plan: dict[str, Any]) -> None:
        link = Path(plan["current_link"])
        if not link.is_symlink() or link.resolve(strict=True) != Path(
            plan["production"]["release"]
        ):
            raise Refusal("production-release-link-changed")

    def arm(
        self, state: Path, plan: Path, plan_sha: str, name: str, deadline: float
    ) -> None:
        self.command(
            [
                "systemd-run",
                "--unit=" + name,
                "--service-type=exec",
                "--property=User=root",
                "--property=RuntimeMaxSec=910",
                "--property=TimeoutStopSec=5",
                "--property=KillMode=control-group",
                "--property=Restart=no",
                "--property=SendSIGKILL=yes",
                "--property=Environment=CUDA_VISIBLE_DEVICES=",
                "--property=Environment=PYTHONDONTWRITEBYTECODE=1",
                str(Path(sys.executable).resolve()),
                str(Path(__file__).resolve()),
                "restore",
                "--watch",
                "--plan",
                str(plan),
                "--plan-sha256",
                plan_sha,
                "--state",
                str(state),
            ],
            deadline,
        )


def _dead(state: dict[str, str]) -> bool:
    return (
        state["ActiveState"] in ("inactive", "failed")
        and state["SubState"] in ("dead", "failed")
        and state["MainPID"] == "0"
        and state["Job"] in ("", "0")
    )


def _owned(
    plan: dict[str, Any], role: str, host: Host, deadline: float
) -> dict[str, str]:
    expected = plan[role]
    found = host.unit(expected["unit"], deadline)
    if (
        found["Id"] != expected["unit"]
        or found["LoadState"] != "loaded"
        or found["ExecStart"] != expected["exec_start"]
        or any(found[k] != v for k, v in expected["properties"].items())
    ):
        raise Refusal("unit-definition-changed-" + role)
    return found


def _healthy(plan: dict[str, Any], role: str, host: Host, deadline: float) -> bool:
    unit = _owned(plan, role, host, deadline)
    if (
        unit["ActiveState"] != "active"
        or unit["SubState"] != "running"
        or int(unit["MainPID"]) <= 0
    ):
        return False
    if host.owners(deadline) != {int(unit["MainPID"])}:
        return False
    for request in plan[role]["health"]:
        result = host.health(request, deadline)
        if not result or result.get("status") != "ok":
            return False
        model = result.get("model", {})
        if (
            model.get("ready") is not True
            or model.get("role") != "champion"
            or model.get("model_step") != plan[role]["model_step"]
            or model.get("model_identity") != plan[role]["model_identity"]
        ):
            return False
    return True


def _wait(host: Host, predicate: Any, deadline: float, failure: str) -> None:
    while host.now() < deadline:
        if predicate():
            if host.now() > deadline:
                raise Refusal(failure)
            return
        host.sleep(min(0.5, max(0.0, deadline - host.now())))
    raise Refusal(failure)


@contextmanager
def locked(directory: Path, host: Host, deadline: float):
    with (directory / "actions.lock").open("a") as stream:
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if host.now() >= deadline:
                    raise Refusal("action-lock-deadline")
                host.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _state(directory: Path, plan_sha: str, host: Host) -> dict[str, Any]:
    state = read_json(directory / "state.json")
    if (
        state["plan_sha256"] != plan_sha
        or state["boot_id"] != host.boot_id()
        or state["operator_sha256"] != digest(Path(__file__).read_bytes())
    ):
        raise Refusal("state-plan-boot-or-operator-mismatch")
    return state


def _save(directory: Path, state: dict[str, Any], host: Host, event: str) -> None:
    state["events"].append(
        {"event": event, "monotonic": host.now(), "utc_ns": time.time_ns()}
    )
    write_json(directory / "state.json", state)


def _admit_run(directory: Path, plan_sha: str, host: Host) -> dict[str, Any]:
    state = _state(directory, plan_sha, host)
    if state["restore_requested"] or host.now() >= state["restore_by"]:
        raise Refusal("qualification-fenced-for-restore")
    return state


def restore(
    plan: dict[str, Any], plan_sha: str, directory: Path, host: Host
) -> dict[str, Any]:
    state = _state(directory, plan_sha, host)
    with locked(directory, host, state["deadline"]):
        state = _state(directory, plan_sha, host)
        if state.get("restored"):
            return state
        state["restore_requested"] = True
        _save(directory, state, host, "restoration-fenced")
        try:
            if host.now() >= state["deadline"]:
                raise Refusal("total-outage-deadline-expired")
            # Never issue stop against a changed or unrelated unit.
            for role in ("probe", "stage"):
                current = _owned(plan, role, host, state["deadline"])
                recorded = state.get(role + "_invocation")
                if (
                    recorded
                    and current["InvocationID"] != recorded
                    and not _dead(current)
                ):
                    raise Refusal("unit-invocation-changed-" + role)
                if state.get(role + "_start_intent") or not _dead(current):
                    if not state.get(role + "_start_intent"):
                        raise Refusal("unowned-active-unit-" + role)
                    host.action("stop", plan[role]["unit"], state["deadline"])
            cleanup_end = min(
                host.now() + DRAIN + OBSERVATION_RESERVE,
                state["deadline"] - EMPTY - RESTORE,
            )
            _wait(
                host,
                lambda: all(
                    _dead(_owned(plan, role, host, cleanup_end))
                    for role in ("probe", "stage")
                ),
                cleanup_end,
                "owned-stage-cleanup-timeout",
            )
            gate_end = min(host.now() + EMPTY, state["deadline"] - RESTORE)
            production = _owned(plan, "production", host, gate_end)
            if state.get("production_restore_intent") and not _dead(production):
                # Crash after our start request: finish validating that same old
                # service without requesting a second start or an empty GPU.
                host.check_link(plan)
                resume_end = min(host.now() + RESTORE, state["deadline"])
                host.check_files(plan["production"], resume_end)
                _wait(
                    host,
                    lambda: _healthy(plan, "production", host, resume_end),
                    resume_end,
                    "old-production-resume-health-timeout",
                )
                state["production_restored_invocation"] = _owned(
                    plan, "production", host, resume_end
                )["InvocationID"]
                state["restored"] = True
                state["outage_elapsed_seconds"] = host.now() - state["budget_started"]
                state.pop("restore_failure", None)
                _save(
                    directory,
                    state,
                    host,
                    "old-production-restore-completed-after-interruption",
                )
                return state
            if state.get("production_stop_intent"):
                # A durable intent does not prove the stop RPC was delivered.
                # Preserve the original healthy process only with no pending job.
                if (
                    production["InvocationID"] == state["production_invocation"]
                    and production["Job"] in ("", "0")
                    and _healthy(plan, "production", host, gate_end)
                ):
                    host.check_link(plan)
                    host.check_files(plan["production"], state["deadline"])
                    state["restored"] = True
                    _save(
                        directory, state, host, "original-production-stop-never-began"
                    )
                    return state
                drain_end = min(
                    state["production_stop_deadline"],
                    state["deadline"] - EMPTY - RESTORE,
                )
                if not _dead(production):
                    _wait(
                        host,
                        lambda: _dead(_owned(plan, "production", host, drain_end)),
                        drain_end,
                        "production-drain-restoration-timeout",
                    )
                gate_end = min(host.now() + EMPTY, state["deadline"] - RESTORE)
                _wait(
                    host,
                    lambda: (
                        _dead(_owned(plan, "production", host, gate_end))
                        and host.port_closed(8081, gate_end)
                        and not host.owners(gate_end)
                    ),
                    gate_end,
                    "exclusive-empty-gpu-restore-gate-blocked",
                )
            else:
                # Failure before the outage: leave the known production process alone.
                if production["InvocationID"] != state[
                    "production_invocation"
                ] or not _healthy(plan, "production", host, gate_end):
                    raise Refusal("pre-outage-production-changed")
                state["restored"] = True
                _save(directory, state, host, "production-never-stopped")
                return state
            # Include integrity verification, startup and both readiness paths in 90s.
            restore_end = min(host.now() + RESTORE, state["deadline"])
            host.check_link(plan)
            host.check_files(plan["production"], restore_end)
            _owned(plan, "production", host, restore_end)
            if not host.port_closed(8081, restore_end) or host.owners(restore_end):
                raise Refusal("restore-gate-changed-before-start")
            state["production_restore_intent"] = True
            _save(directory, state, host, "unchanged-old-production-start-intent")
            host.action("start", plan["production"]["unit"], restore_end)
            _wait(
                host,
                lambda: _healthy(plan, "production", host, restore_end),
                restore_end,
                "old-production-health-timeout",
            )
            state["production_restored_invocation"] = _owned(
                plan, "production", host, restore_end
            )["InvocationID"]
            state["restored"] = True
            state["outage_elapsed_seconds"] = host.now() - state["budget_started"]
            state.pop("restore_failure", None)
            _save(directory, state, host, "old-production-566428-restored")
        except BaseException as exc:
            state["restore_failure"] = (
                str(exc) if isinstance(exc, Refusal) else type(exc).__name__
            )
            _save(directory, state, host, "restoration-blocked-no-overlap-waiver")
            raise
        return state


def _finite_number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _probe_started(directory: Path, state: dict[str, Any]) -> dict[str, Any] | None:
    path = directory / "probe-started.json"
    if not path.exists():
        return None
    challenge_path = directory / "probe-challenge.json"
    if digest(challenge_path.read_bytes()) != state["probe_challenge_sha256"]:
        raise Refusal("probe-challenge-changed")
    challenge = read_json(challenge_path)
    started = read_json(path)
    if (
        started.get("format") != "deltrelserve.gpu-qualification-probe-started"
        or started.get("schema_version") != 1
        or any(started.get(k) != challenge[k] for k in PROBE_BINDINGS)
    ):
        raise Refusal("probe-started-binding-mismatch")
    invocation = started.get("invocation_id", "")
    beginning = started.get("started_monotonic")
    if (
        re.fullmatch(r"[0-9a-f]{32}", invocation) is None
        or invocation == state["prior_invocations"]["probe"]
        or not _finite_number(beginning)
        or not challenge["issued_monotonic"]
        <= beginning
        <= challenge["deadline_monotonic"]
    ):
        raise Refusal("probe-started-identity-invalid")
    return started


def _probe_result(
    directory: Path,
    state: dict[str, Any],
    plan: dict[str, Any],
    current: dict[str, str],
    host: Host,
) -> dict[str, Any] | None:
    result_path = directory / "probe-result.json"
    if not result_path.exists():
        if (
            _dead(current)
            and current["InvocationID"]
            and current["InvocationID"] != state["prior_invocations"]["probe"]
            and current["Result"] != "success"
        ):
            raise Refusal("qualification-probe-failed-without-receipt")
        return None
    challenge_path = directory / "probe-challenge.json"
    if digest(challenge_path.read_bytes()) != state["probe_challenge_sha256"]:
        raise Refusal("probe-challenge-changed")
    challenge = read_json(challenge_path)
    started = _probe_started(directory, state)
    if started is None:
        raise Refusal("probe-started-receipt-missing")
    result = read_json(result_path)
    for value, kind in ((started, "started"), (result, "result")):
        if (
            value.get("format") != "deltrelserve.gpu-qualification-probe-" + kind
            or value.get("schema_version") != 1
        ):
            raise Refusal("probe-receipt-schema-mismatch")
        for key in (
            "run_id",
            "plan_sha256",
            "boot_id",
            "challenge",
            "probe_unit",
            "model_identity",
            "model_step",
        ):
            if value.get(key) != challenge[key]:
                raise Refusal("probe-receipt-binding-mismatch")
        if (
            re.fullmatch(r"[0-9a-f]{32}", value.get("invocation_id", "")) is None
            or value["invocation_id"] == state["prior_invocations"]["probe"]
        ):
            raise Refusal("probe-receipt-invocation-invalid")
    invocation = started["invocation_id"]
    if result["invocation_id"] != invocation or current["InvocationID"] not in (
        "",
        invocation,
    ):
        raise Refusal("probe-receipt-invocation-mismatch")
    first, last = result.get("started_monotonic"), result.get("completed_monotonic")
    if (
        not _finite_number(first)
        or not _finite_number(last)
        or first != started.get("started_monotonic")
        or not challenge["issued_monotonic"]
        <= first
        <= last
        <= min(challenge["deadline_monotonic"], host.now())
    ):
        raise Refusal("probe-receipt-outside-monotonic-window")
    checks = result.get("checks")
    evidence = result.get("evidence")
    if (
        not isinstance(checks, dict)
        or set(checks) != set(REQUIRED_CHECKS)
        or any(
            not isinstance(row, dict) or row.get("status") != "passed"
            for row in checks.values()
        )
    ):
        raise Refusal("qualification-probe-failed")
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 64:
        raise Refusal("probe-evidence-inventory-invalid")
    seen = set()
    hashes = set()
    total = 0
    for item in evidence:
        relative = item.get("path")
        if (
            not isinstance(relative, str)
            or not relative.startswith("probe-evidence/")
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or Path(relative).as_posix() != relative
            or relative in seen
        ):
            raise Refusal("probe-evidence-path-invalid")
        seen.add(relative)
        path = directory / relative
        size = item.get("bytes")
        if (
            type(size) is not int
            or size < 0
            or path.resolve(strict=True) != path
            or not path.is_file()
            or path.stat().st_size != size
        ):
            raise Refusal("probe-evidence-file-invalid")
        total += size
        if total > 16 * 2**20 or host.now() >= state["stage_deadline"]:
            raise Refusal("probe-evidence-budget-exceeded")
        actual = digest(path.read_bytes())
        if actual != item.get("sha256"):
            raise Refusal("probe-evidence-hash-mismatch")
        hashes.add(actual)
    if any(row.get("evidence_sha256") not in hashes for row in checks.values()):
        raise Refusal("probe-check-evidence-unbound")
    return result


def _watchdog_owned(unit: dict[str, str], name: str) -> bool:
    return (
        unit.get("Id") == name
        and unit.get("User") == "root"
        and unit.get("Type") == "exec"
        and unit.get("Restart") == "no"
        and unit.get("KillMode") == "control-group"
        and unit.get("SendSIGKILL") == "yes"
        and unit.get("RuntimeMaxUSec") == "15min 10s"
        and unit.get("TimeoutStopUSec") == "5s"
        and unit.get("MainPID", "0").isdecimal()
        and int(unit["MainPID"]) > 0
    )


def run(
    plan: dict[str, Any], plan_path: Path, plan_sha: str, directory: Path, host: Host
) -> dict[str, Any]:
    preflight_end = host.now() + 120
    host.check_link(plan)
    for role in ("production", "stage", "probe"):
        host.check_files(plan[role], preflight_end)
        found = _owned(plan, role, host, preflight_end)
        if role != "production" and not _dead(found):
            raise Refusal("qualification-unit-not-stopped")
    prior_invocations = {
        role: _owned(plan, role, host, preflight_end)["InvocationID"]
        for role in ("stage", "probe")
    }
    production = _owned(plan, "production", host, preflight_end)
    if (
        production["InvocationID"] != plan["production"]["initial_invocation_id"]
        or int(production["MainPID"]) != plan["production"]["initial_main_pid"]
        or not _healthy(plan, "production", host, preflight_end)
        or not host.port_closed(8081, preflight_end)
    ):
        raise Refusal("production-or-gpu-preflight-changed")
    directory.mkdir(mode=0o700)  # A run directory is never reused or clobbered.
    started = host.now()
    watchdog_name = f"deltrelserve-qualification-{plan['run_id']}-watchdog.service"
    controller_identity = host.process(os.getpid())
    if controller_identity is None:
        raise Refusal("controller-process-identity-unavailable")
    state = {
        "format": FORMAT,
        "schema_version": 1,
        "plan_sha256": plan_sha,
        "operator_sha256": digest(Path(__file__).read_bytes()),
        "boot_id": host.boot_id(),
        "controller": controller_identity,
        "budget_started": started,
        "restore_by": started + RESTORE_BY,
        "deadline": started + TOTAL,
        "production_invocation": production["InvocationID"],
        "watchdog_unit": watchdog_name,
        "restore_requested": False,
        "restored": False,
        "events": [],
        "prior_invocations": prior_invocations,
    }
    _save(directory, state, host, "prepared-before-outage")
    try:
        host.arm(directory, plan_path, plan_sha, watchdog_name, started + 10)

        def armed():
            path = directory / "watchdog-armed.json"
            if not path.exists():
                return False
            receipt = read_json(path)
            unit = host.unit(watchdog_name, started + 10)
            return (
                _watchdog_owned(unit, watchdog_name)
                and int(unit["MainPID"]) != state["controller"]["pid"]
                and host.process(int(unit["MainPID"])) is not None
                and receipt.get("plan_sha256") == plan_sha
                and receipt.get("boot_id") == state["boot_id"]
                and receipt.get("deadline") == state["deadline"]
                and receipt.get("watchdog") == host.process(int(unit["MainPID"]))
                and unit["ActiveState"] == "active"
                and unit["InvocationID"] == receipt.get("invocation_id")
            )

        _wait(host, armed, started + 10, "independent-watchdog-not-armed")
        with locked(directory, host, state["restore_by"]):
            state = _admit_run(directory, plan_sha, host)
            fresh = _owned(plan, "production", host, state["restore_by"])
            if (
                fresh["InvocationID"] != state["production_invocation"]
                or int(fresh["MainPID"]) != plan["production"]["initial_main_pid"]
                or fresh["Job"] not in ("", "0")
                or not _healthy(plan, "production", host, state["restore_by"])
            ):
                raise Refusal("production-invocation-changed-after-arm")
            host.check_link(plan)
            host.check_files(plan["production"], state["restore_by"])
            state["production_stop_deadline"] = min(
                host.now() + DRAIN + OBSERVATION_RESERVE, state["restore_by"]
            )
            state["production_stop_intent"] = True
            _save(directory, state, host, "production-stop-intent")
            host.action("stop", plan["production"]["unit"], state["restore_by"])
        stop_end = state["production_stop_deadline"]
        _wait(
            host,
            lambda: _dead(_owned(plan, "production", host, stop_end)),
            stop_end,
            "production-drain-timeout",
        )
        gate_end = min(host.now() + EMPTY, state["restore_by"])
        _wait(
            host,
            lambda: not host.owners(gate_end),
            gate_end,
            "production-empty-gpu-gate-blocked",
        )
        with locked(directory, host, state["restore_by"]):
            state = _admit_run(directory, plan_sha, host)
            host.check_link(plan)
            host.check_files(plan["stage"], state["restore_by"])
            host.check_files(plan["probe"], state["restore_by"])
            _owned(plan, "stage", host, state["restore_by"])
            if (
                not host.port_closed(8081, state["restore_by"])
                or not _dead(_owned(plan, "production", host, state["restore_by"]))
                or host.owners(state["restore_by"])
            ):
                raise Refusal("exclusive-stage-gate-changed")
            state["stage_start_intent"] = True
            _save(directory, state, host, "owned-stage-start-intent")
            host.action("start", plan["stage"]["unit"], state["restore_by"])
            state["stage_deadline"] = min(host.now() + STAGE, state["restore_by"])
            _save(directory, state, host, "stage-start-requested")

        def stage_ready():
            current = _owned(plan, "stage", host, state["stage_deadline"])
            if (
                current["InvocationID"]
                and current["InvocationID"] != state["prior_invocations"]["stage"]
            ):
                with locked(directory, host, state["stage_deadline"]):
                    current_state = _admit_run(directory, plan_sha, host)
                    if current_state.get("stage_invocation") not in (
                        None,
                        current["InvocationID"],
                    ):
                        raise Refusal("stage-invocation-changed")
                    if not current_state.get("stage_invocation"):
                        current_state["stage_invocation"] = current["InvocationID"]
                        _save(
                            directory, current_state, host, "stage-invocation-observed"
                        )
                return _healthy(plan, "stage", host, state["stage_deadline"])
            return False

        _wait(host, stage_ready, state["stage_deadline"], "stage-startup-deadline")
        with locked(directory, host, state["stage_deadline"]):
            state = _admit_run(directory, plan_sha, host)
            state["stage_invocation"] = _owned(
                plan, "stage", host, state["stage_deadline"]
            )["InvocationID"]
            challenge = {
                "format": "deltrelserve.gpu-qualification-challenge",
                "schema_version": 1,
                "run_id": plan["run_id"],
                "plan_sha256": plan_sha,
                "boot_id": state["boot_id"],
                "challenge": secrets.token_hex(32),
                "issued_monotonic": host.now(),
                "deadline_monotonic": state["stage_deadline"],
                "probe_unit": plan["probe"]["unit"],
                "model_identity": plan["stage"]["model_identity"],
                "model_step": plan["stage"]["model_step"],
                "required_checks": list(REQUIRED_CHECKS),
            }
            write_json(directory / "probe-challenge.json", challenge, replace=False)
            state["probe_challenge_sha256"] = digest(
                (directory / "probe-challenge.json").read_bytes()
            )
            state["probe_start_intent"] = True
            _save(directory, state, host, "owned-probe-start-intent")
            _owned(plan, "probe", host, state["stage_deadline"])
            host.action("start", plan["probe"]["unit"], state["stage_deadline"])

        def probes_complete():
            current = _owned(plan, "probe", host, state["stage_deadline"])
            # Capture manager ownership before examining any untrusted probe evidence.
            if (
                current["InvocationID"]
                and current["InvocationID"] != state["prior_invocations"]["probe"]
            ):
                with locked(directory, host, state["stage_deadline"]):
                    observed = _admit_run(directory, plan_sha, host)
                    if observed.get("probe_invocation") not in (
                        None,
                        current["InvocationID"],
                    ):
                        raise Refusal("probe-invocation-changed")
                    if not observed.get("probe_invocation"):
                        observed["probe_invocation"] = current["InvocationID"]
                        _save(directory, observed, host, "probe-invocation-observed")
            record = _probe_result(directory, state, plan, current, host)
            return record is not None and _dead(current)

        _wait(
            host,
            probes_complete,
            state["stage_deadline"],
            "qualification-stage-deadline",
        )
        with locked(directory, host, state["stage_deadline"]):
            state = _admit_run(directory, plan_sha, host)
            final_stage = _owned(plan, "stage", host, state["stage_deadline"])
            if final_stage["InvocationID"] != state["stage_invocation"] or not _healthy(
                plan, "stage", host, state["stage_deadline"]
            ):
                raise Refusal("stage-changed-before-qualification-complete")
            state["probe_result_sha256"] = digest(
                (directory / "probe-result.json").read_bytes()
            )
            state["qualification_passed"] = True
            _save(directory, state, host, "qualification-probes-passed")
    except BaseException as exc:
        with locked(directory, host, state["deadline"]):
            state = _state(directory, plan_sha, host)
            state["qualification_failure"] = (
                str(exc) if isinstance(exc, Refusal) else type(exc).__name__
            )
            _save(directory, state, host, "qualification-did-not-complete")
        raise
    finally:
        restore(plan, plan_sha, directory, host)
    return _state(directory, plan_sha, host)


def watch(
    plan: dict[str, Any], plan_sha: str, directory: Path, host: Host
) -> dict[str, Any]:
    state = _state(directory, plan_sha, host)
    current = host.unit(state["watchdog_unit"], state["deadline"])
    identity = host.process(os.getpid())
    if (
        not _watchdog_owned(current, state["watchdog_unit"])
        or int(current["MainPID"]) != os.getpid()
        or not identity
        or current["ActiveState"] not in ("active", "activating")
    ):
        raise Refusal("watchdog-not-independent-systemd-owner")
    write_json(
        directory / "watchdog-armed.json",
        {
            "plan_sha256": plan_sha,
            "boot_id": state["boot_id"],
            "deadline": state["deadline"],
            "watchdog": identity,
            "invocation_id": current["InvocationID"],
        },
    )
    while host.now() < state["restore_by"]:
        state = _state(directory, plan_sha, host)
        if state["restored"]:
            return state
        if (
            state["restore_requested"]
            or host.process(state["controller"]["pid"]) != state["controller"]
        ):
            break
        host.sleep(min(0.5, state["restore_by"] - host.now()))
    return restore(plan, plan_sha, directory, host)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "restore"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args(argv)
    try:
        if os.geteuid() != 0 or args.watch and args.mode != "restore":
            raise Refusal("root-operator-and-restore-watch-required")
        plan_path = args.plan.resolve(strict=True)
        directory = args.state.absolute()
        if (
            directory.is_symlink()
            or directory.parent.resolve(strict=True) != directory.parent
        ):
            raise Refusal("state-parent-must-be-canonical")
        plan = load_plan(plan_path, args.plan_sha256)
        host = Host()
        if args.mode == "run":
            result = run(plan, plan_path, args.plan_sha256, directory, host)
        elif args.watch:
            result = watch(plan, args.plan_sha256, directory, host)
        else:
            result = restore(plan, args.plan_sha256, directory, host)
        print(
            json.dumps(
                {
                    "state": str(directory / "state.json"),
                    "restored": result["restored"],
                    "qualification_passed": result.get("qualification_passed", False),
                }
            )
        )
    except BaseException as exc:
        # Do not emit tracebacks, HTTP headers, environment values or subprocess output.
        print(
            json.dumps(
                {
                    "failure": str(exc)
                    if isinstance(exc, Refusal)
                    else type(exc).__name__
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
