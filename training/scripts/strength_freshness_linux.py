"""Pinned Linux control-plane adapter for the finite freshness guard.

Importing this module performs no host inspection. Execution requires a separate
target-host qualification receipt bound to this exact adapter, core and plan.
The guardian owns the advisory host lease; registered GPU start paths must all
respect that lease. NVML observations alone are never a claim of exclusion.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import stat
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping

from deltreltrain import strength_freshness_guard as core
from scripts.qualify_cloud_gpu_window import Host as SystemHost

FORMAT = "deltreltrain.strength-freshness-linux"
QUALIFICATION = "deltreltrain.strength-freshness-linux-qualification"
AUTHORIZATION = "adapter-authorization.json"
JOBS_COMMAND = (
    "systemctl",
    "list-jobs",
    "--all",
    "--no-pager",
    "--no-legend",
    "--plain",
    "--full",
)
LEASE = "linux-lease.json"
CHECKS = {
    "ownership",
    "deadlines",
    "guardian_reentry",
    "boot_edges",
    "proof_closure",
    "fresh_progress",
}
DIRECT_SCRIPTS = (
    "strength_freshness_units.py",
    "strength_freshness_progress.py",
    "strength_freshness_auxiliary.py",
    "qualify_cloud_gpu_window.py",
    "activate_strength_freshness.py",
    "graceful_training_deploy.py",
    "migrate_continuous_profile.py",
    "deployment_metadata.py",
    "backup_active_profile.py",
    "training_disaster_recovery.py",
)
DIRECT_RUNTIME = (
    "runtime.py",
    "checkpoint.py",
    "champion_migration.py",
    "strength_freshness.py",
    "strength_recovery.py",
    "strength_recovery_archive.py",
)
PROPERTIES = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "MainPID",
    "ControlGroup",
    "InvocationID",
    "ExecMainStartTimestampMonotonic",
    "Result",
    "ExecMainStatus",
    "ExecMainPID",
    "NRestarts",
    "Job",
    "UnitFileState",
    "FragmentPath",
    "DropInPaths",
    "ExecStart",
    "Environment",
    "EnvironmentFiles",
    "User",
    "Type",
    "Group",
    "WorkingDirectory",
    "ExecStartPre",
    "Restart",
    "TimeoutStopUSec",
    "KillMode",
    "SendSIGKILL",
    "Before",
    "After",
    "RuntimeMaxUSec",
)
VARIABLE = {
    "ActiveState",
    "SubState",
    "MainPID",
    "ControlGroup",
    "InvocationID",
    "ExecMainStartTimestampMonotonic",
    "Result",
    "ExecMainStatus",
    "ExecMainPID",
    "NRestarts",
    "Job",
    "UnitFileState",
}
TIMER_PROPERTIES = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "Job",
    "UnitFileState",
    "FragmentPath",
    "DropInPaths",
    "Before",
    "After",
)


def property_names(name: str) -> tuple[str, ...]:
    return TIMER_PROPERTIES if name.endswith(".timer") else PROPERTIES


def stable_properties(values: Mapping[str, str]) -> dict[str, str]:
    result = dict(values)
    for key in ("Before", "After"):
        if key in result:
            result[key] = " ".join(sorted(result[key].split()))
    for key in ("ExecStart", "ExecStartPre"):
        if key in result and " ; ignore_errors=" in result[key]:
            require(result[key].count("{ path=") <= 1, "linux-multiple-exec-records")
            prefix, suffix = result[key].split(" ; ignore_errors=", 1)
            parts = suffix.removesuffix(" }").split(" ; ")
            require(parts[0] in {"yes", "no"}, "linux-ignore-errors-value")
            dynamic = {
                "start_time",
                "stop_time",
                "pid",
                "code",
                "status",
                "start_time_monotonic",
                "stop_time_monotonic",
            }
            retained = [
                part for part in parts[1:] if part.split("=", 1)[0] not in dynamic
            ]
            result[key] = (
                prefix
                + " ; ignore_errors="
                + parts[0]
                + "".join(" ; " + part for part in retained)
            )
    return result


def env_identity(environment: str, pins: list[dict[str, Any]]) -> str:
    return core.digest(
        {
            "environment": environment,
            "files": [
                {
                    **{k: p[k] for k in ("path", "sha256", "bytes")},
                    "mode": p.get("mode", 0o600),
                }
                for p in pins
            ],
        }
    )


def require(ok: object, reason: str) -> None:
    if not ok:
        raise core.Refusal(reason)


class ObservationChanged(RuntimeError):
    """Registered cgroup grew during bounded capture; rescan, never call it empty."""


def encoded(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(path: str) -> Path:
    p = Path(path)
    require(
        p.is_absolute() and p.as_posix() == path and ".." not in p.parts,
        "linux-canonical-path",
    )
    return p


def qualification_contract(manifest: Mapping[str, Any]) -> str:
    # The core plan contains the qualification SHA. Excluding that final plan
    # reference avoids a circular digest; launcher authorization pins both.
    return core.digest(
        {
            k: v
            for k, v in manifest.items()
            if k not in {"qualification", "guard_plan", "core_plan_sha256"}
        }
    )


def systemd_seconds(value: str) -> float:
    if value == "infinity":
        return float("inf")
    units = {"us": 1e-6, "ms": 1e-3, "s": 1.0, "min": 60.0, "h": 3600.0}
    compact = value.replace(" ", "")
    tokens = list(re.finditer(r"([0-9]+(?:\.[0-9]+)?)(us|ms|min|s|h)", compact))
    require(
        tokens and "".join(t.group() for t in tokens) == compact,
        "linux-systemd-duration",
    )
    return sum(float(t[1]) * units[t[2]] for t in tokens)


class LinuxIO(SystemHost):
    """Bounded command calls reuse the already reviewed systemd utility.

    All reads have strict byte/count limits. The guardian's independent systemd
    wall limit remains mandatory for kernel/filesystem stalls, which Python
    cannot interrupt reliably. NFS backup work is performed by a bounded child.
    """

    def exists(self, path: Path) -> bool:
        return path.exists() or path.is_symlink()

    def file_metadata(self, path: Path, deadline: float) -> dict[str, int]:
        require(self.now() < deadline and path.resolve() == path, "linux-metadata-path")
        value = path.lstat()
        require(
            stat.S_ISREG(value.st_mode) and self.now() < deadline,
            "linux-metadata-regular",
        )
        return {
            "uid": value.st_uid,
            "gid": value.st_gid,
            "mode": stat.S_IMODE(value.st_mode),
        }

    def process_origin(self, pid: int, deadline: float) -> dict[str, str]:
        require(self.now() < deadline, "linux-origin-deadline")
        try:
            value = {
                key: str(Path(f"/proc/{pid}/{key}").resolve(strict=True))
                for key in ("exe", "cwd")
            }
        except (FileNotFoundError, ProcessLookupError):
            raise ObservationChanged("linux-origin-process-disappeared") from None
        require(self.now() < deadline, "linux-origin-deadline")
        return value

    def read(self, path: Path, maximum: int, deadline: float) -> bytes:
        require(self.now() < deadline, "linux-read-deadline")
        require(path.resolve() == path, "linux-read-symlink")
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            require(
                stat.S_ISREG(before.st_mode) and before.st_size <= maximum,
                "linux-read-size",
            )
            data = stream.read(maximum + 1)
            after = os.fstat(stream.fileno())
        require(self.now() < deadline and len(data) <= maximum, "linux-read-deadline")
        require(
            (before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_ino, after.st_size, after.st_mtime_ns),
            "linux-read-raced",
        )
        return data

    def pin(self, pin: Mapping[str, Any], deadline: float) -> None:
        p = canonical(str(pin["path"]))
        resolved = p.resolve()
        require(
            str(resolved) == pin.get("resolved_path", str(p)), "linux-pin-resolution"
        )
        require(
            type(pin.get("bytes")) is int
            and pin["bytes"] >= 0
            and core.sha(pin.get("sha256")),
            "linux-pin-shape",
        )
        require(self.now() < deadline, "linux-pin-deadline")
        sha = hashlib.sha256()
        with resolved.open("rb") as stream:
            before = os.fstat(stream.fileno())
            require(
                stat.S_ISREG(before.st_mode) and before.st_size == pin["bytes"],
                "linux-pin-size",
            )
            for block in iter(lambda: stream.read(2**20), b""):
                require(self.now() < deadline, "linux-pin-deadline")
                sha.update(block)
            after = os.fstat(stream.fileno())
        require(
            (before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_ino, after.st_size, after.st_mtime_ns)
            and sha.hexdigest() == pin["sha256"],
            "linux-pin-content",
        )

    def json(self, path: Path, deadline: float, maximum: int = 2**20) -> dict[str, Any]:
        def pairs(rows):
            result = {}
            for key, value in rows:
                require(key not in result, "linux-duplicate-json-key")
                result[key] = value
            return result

        value = json.loads(
            self.read(path, maximum, deadline),
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                core.Refusal("linux-nonfinite-json")
            ),
        )
        require(isinstance(value, dict), "linux-json-object")
        return value

    def tail(
        self, path: Path, deadline: float, maximum: int = 2**20
    ) -> list[dict[str, Any]]:
        require(path.resolve() == path and self.now() < deadline, "linux-tail-path")
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            stream.seek(max(0, size - maximum))
            data = stream.read(maximum)
        lines = data.splitlines()
        if size > maximum:
            lines = lines[1:]
        if data and not data.endswith(b"\n"):
            lines = lines[:-1]
        require(self.now() < deadline, "linux-tail-deadline")
        return [json.loads(line) for line in lines[-64:] if line]

    def process(self, pid: int, deadline: float | None = None) -> dict[str, Any] | None:
        deadline = self.now() + 2 if deadline is None else deadline
        try:
            fields = (
                self.read(Path(f"/proc/{pid}/stat"), 65536, deadline)
                .decode()
                .rsplit(")", 1)[1]
                .split()
            )
            if fields[0] == "Z":
                return None
            return {
                "pid": pid,
                "start_ticks": int(fields[19]),
                "ppid": int(fields[1]),
                "cgroup": self.read(Path(f"/proc/{pid}/cgroup"), 65536, deadline)
                .decode()
                .strip(),
            }
        except (FileNotFoundError, ProcessLookupError):
            return None

    def members(self, path: str, deadline: float) -> tuple[core.Process, ...]:
        require(
            path.startswith("/system.slice/") and ".." not in Path(path).parts,
            "linux-cgroup-path",
        )
        root = Path("/sys/fs/cgroup") / path.lstrip("/")
        if not root.exists():
            return ()
        todo, found, count = [root], {}, 0
        while todo:
            current = todo.pop()
            require(
                current.resolve() == current and self.now() < deadline,
                "linux-cgroup-raced",
            )
            count += 1
            require(count <= 512, "linux-cgroup-count")
            for token in self.read(current / "cgroup.procs", 65536, deadline).split():
                require(token.isdigit(), "linux-cgroup-pid")
                pid = int(token)
                require(self.now() < deadline, "linux-cgroup-pid-deadline")
                process = self.process(pid, deadline)
                if process is None:
                    raise ObservationChanged("linux-cgroup-process-disappeared")
                assert process is not None
                actual = process["cgroup"]
                require(
                    actual == "0::" + path or actual.startswith("0::" + path + "/"),
                    "linux-cgroup-membership",
                )
                found[pid] = core.Process(pid, process["start_ticks"])
            require(len(found) <= 8192, "linux-process-count")
            todo.extend(p for p in current.iterdir() if p.is_dir())
        return tuple(found[p] for p in sorted(found))

    def atomic(
        self, path: Path, data: bytes, *, overwrite: bool = False, mode: int = 0o600
    ) -> None:
        require(
            path.parent.resolve() == path.parent and not path.is_symlink(),
            "linux-write-path",
        )
        temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fchmod(stream.fileno(), mode)
                os.fsync(stream.fileno())
            if overwrite:
                os.replace(temporary, path)
            else:
                os.link(temporary, path, follow_symlinks=False)
                temporary.unlink()
            parent = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        finally:
            temporary.unlink(missing_ok=True)

    def boot_links(self, name: str, deadline: float) -> dict[str, str]:
        links: dict[str, str] = {}
        count = 0
        for root in (Path("/etc/systemd/system"), Path("/run/systemd/system")):
            for directory in root.iterdir():
                require(self.now() < deadline, "linux-boot-link-deadline")
                count += 1
                require(count <= 4096, "linux-boot-link-count")
                if (
                    directory.name.endswith((".wants", ".requires"))
                    and directory.is_dir()
                ):
                    for path in directory.iterdir():
                        require(self.now() < deadline, "linux-boot-link-deadline")
                        count += 1
                        require(count <= 8192, "linux-boot-edge-count")
                        if path.is_symlink():
                            target = path.resolve()
                            if (
                                path.name == name
                                or target == Path("/etc/systemd/system") / name
                            ):
                                links[str(path)] = str(target)
                        elif path.name == name:
                            raise core.Refusal("linux-boot-edge-not-symlink")
        return links

    def child(
        self, argv: list[str], environment: dict[str, str], deadline: float
    ) -> dict[str, Any]:
        remaining = deadline - self.now()
        require(remaining > 1.1, "linux-child-cleanup-reserve")
        child = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            start_new_session=True,
            close_fds=True,
        )
        try:
            stdout, stderr = child.communicate(timeout=remaining - 1)
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.communicate(timeout=max(0.01, deadline - self.now()))
            raise
        require(child.returncode == 0, "linux-worker-failed")
        require(
            len(stdout) <= 16 * 2**20 and len(stderr) <= 2**20,
            "linux-worker-output-limit",
        )
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            pass
        else:
            os.killpg(child.pid, signal.SIGKILL)
            raise core.Refusal("linux-worker-descendant-survived")
        value = json.loads(stdout)
        require(isinstance(value, dict), "linux-worker-result")
        return value


def load_manifest(
    path: Path, expected: str, io_: LinuxIO, *, execute: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = io_.now() + 30
    raw = io_.read(path, 2**20, deadline)
    require(checksum(raw) == expected, "linux-manifest-hash")
    manifest = json.loads(raw)
    require(
        manifest.get("format") == FORMAT and manifest.get("schema_version") == 1,
        "linux-manifest-schema",
    )
    require(
        manifest.get("adapter_sha256") == checksum(Path(__file__).read_bytes()),
        "linux-adapter-source",
    )
    io_.pin(manifest["guard_plan"], deadline)
    plan = io_.json(Path(manifest["guard_plan"]["path"]), deadline)
    core.validate_plan(plan)
    require(plan["guard_source_sha256"] == core.source_sha256(), "linux-core-source")
    require(manifest.get("core_plan_sha256") == core.digest(plan), "linux-core-plan")
    require(
        set(manifest["units"])
        == {s["name"] for s in plan["units"].values()}
        | set(plan["support_transition"]["before"]),
        "linux-unit-inventory",
    )
    for name, spec in manifest["units"].items():
        require(
            re.fullmatch(r"[A-Za-z0-9_.@-]+\.(service|timer)", name), "linux-unit-name"
        )
        require(
            spec["installed_path"] == "/etc/systemd/system/" + name,
            "linux-unit-install-path",
        )
        for stage in ("before", "after"):
            item = spec[stage]
            require(
                set(item["properties"]) == set(property_names(name)) - VARIABLE,
                "linux-unit-property-scope",
            )
            require(
                item["properties"]["Id"] == name
                and item["properties"]["DropInPaths"] == "",
                "linux-unit-dropins",
            )
            require(
                item["properties"]["FragmentPath"] == spec["installed_path"],
                "linux-unit-fragment",
            )
            if name.endswith(".service"):
                require(
                    item["properties"]["KillMode"] == "control-group"
                    and item["properties"]["SendSIGKILL"] == "yes",
                    "linux-unit-cleanup-policy",
                )
    bindings = manifest.get("execution", {})
    require(
        bindings.get("host_exclusion_contract")
        == "all-registered-gpu-starts-fenced-under-guardian-lease-v1",
        "linux-exclusion-contract",
    )
    require(
        bindings.get("finalizer") == "sealed-retirement-only",
        "linux-finalizer-contract",
    )
    for role, spec in plan["units"].items():
        target = manifest["units"][spec["name"]]
        for stage in ("before", "after"):
            require(
                target[stage]["unit"]["sha256"] == spec["definition_sha256"],
                "linux-core-unit-binding",
            )
            if role in {"r3", "r4", "probe"}:
                require(
                    target[stage]["properties"]["ExecStartPre"]
                    == bindings["start_fences"][role]
                    and "admit-start" in bindings["start_fences"][role]
                    and "--role " + role in bindings["start_fences"][role],
                    "linux-start-fence-required",
                )
            props = target[stage]["properties"]
            require(
                systemd_seconds(props["TimeoutStopUSec"])
                <= (90 if role == "probe" else 15 if role == "guard" else 390),
                "linux-stop-bound",
            )
            if role in {"r4", "probe"}:
                require(props["Restart"] == "no", "linux-restart-loop-forbidden")
            if role in {"guard", "probe"}:
                require(
                    systemd_seconds(props["RuntimeMaxUSec"])
                    <= (2700 if role == "guard" else 600),
                    "linux-runtime-wall-bound",
                )
        if role == "guard":
            require(
                set(plan["units"][r]["name"] for r in ("r3", "r4", "probe"))
                <= set(target["after"]["properties"]["Before"].split()),
                "linux-guardian-boot-order",
            )
    for pin in manifest["implementation_pins"]:
        io_.pin(pin, deadline)
    required_helpers = {
        str(Path(__file__).resolve()),
        str(Path(core.__file__).resolve()),
        *(str(Path(__file__).with_name(name).resolve()) for name in DIRECT_SCRIPTS),
    }
    require(
        required_helpers <= {pin["path"] for pin in manifest["implementation_pins"]},
        "linux-helper-source-closure",
    )
    require(
        {str(Path(core.__file__).with_name(name).resolve()) for name in DIRECT_RUNTIME}
        <= {pin["path"] for pin in manifest["implementation_pins"]},
        "linux-runtime-helper-closure",
    )
    support_manifest = manifest["support_source_manifest"]
    io_.pin(support_manifest, deadline)
    io_.pin(plan["probe_runtime"]["source_manifest"], deadline)
    require(
        support_manifest["sha256"] in plan["proof_closure_inputs"]["support_manifest"]
        and plan["probe_runtime"]["source_manifest"]["sha256"]
        in plan["proof_closure_inputs"]["runtime_source_manifest"],
        "linux-source-manifest-closure",
    )
    require(
        manifest["helper_python"]["path"]
        in {p["path"] for p in manifest["implementation_pins"]},
        "linux-helper-python-pin",
    )
    from scripts.strength_freshness_auxiliary import validate_policy

    require(
        set(manifest["auxiliary_policies"]) == {"r3", "r4"},
        "linux-auxiliary-policy-inventory",
    )
    for policy in manifest["auxiliary_policies"].values():
        validate_policy(policy)
        require(
            policy["compile_worker_script"]
            in {p["path"] for p in manifest["implementation_pins"]},
            "linux-auxiliary-script-pin",
        )
    qualification = manifest.get("qualification")
    if execute:
        require(isinstance(qualification, dict), "linux-execution-unqualified")
        io_.pin(qualification, deadline)
        q = io_.json(Path(qualification["path"]), deadline)
        require(
            q.get("format") == QUALIFICATION
            and q.get("schema_version") == 1
            and q.get("status") == "passed"
            and q.get("execution_qualified") is True
            and q.get("adapter_sha256") == manifest["adapter_sha256"]
            and q.get("core_source_sha256") == plan["guard_source_sha256"]
            and q.get("contract_sha256") == qualification_contract(manifest)
            and q.get("checks") == {k: True for k in CHECKS},
            "linux-execution-qualification",
        )
        require(
            qualification["sha256"] == plan["adapter_qualification_sha256"],
            "linux-qualification-core-binding",
        )
    return manifest, plan


def parse_jobs_text(raw: str) -> dict[int, dict[str, Any]]:
    """Read the exact systemd255 four-column table, without legend or ellipses."""
    require(isinstance(raw, str) and len(raw.encode()) <= 2**20, "linux-job-output")
    lines = raw.splitlines()
    require(len(lines) <= 4096, "linux-job-inventory")
    result: dict[int, dict[str, Any]] = {}
    units = set()
    for line in lines:
        fields = line.split()
        require(len(fields) == 4, "linux-job-columns")
        identifier, unit, kind, state = fields
        require(re.fullmatch(r"[1-9][0-9]*", identifier), "linux-job-id")
        job_id = int(identifier)
        require(
            re.fullmatch(
                r"(?:[A-Za-z0-9_.@:-]|\\x[0-9a-fA-F]{2})+\.(?:service|timer|target|slice|socket|mount|automount|swap|path|scope|device)",
                unit,
            ),
            "linux-job-unit",
        )
        require(
            re.fullmatch(r"[a-z-]+", kind) and state in {"waiting", "running"},
            "linux-job-type-state",
        )
        require(job_id not in result and unit not in units, "linux-duplicate-job")
        units.add(unit)
        result[job_id] = {"job": job_id, "unit": unit, "type": kind, "state": state}
    return result


def parse_job(
    value: str, jobs: Mapping[int, dict[str, Any]], name: str
) -> Mapping[str, Any] | None:
    # systemctl show prints the ID from the D-Bus (uo) property, not its path.
    if value in ("", "0"):
        require(
            not any(row.get("unit") == name for row in jobs.values()), "linux-job-join"
        )
        return None
    require(
        isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value),
        "linux-job-property",
    )
    identifier = int(value)
    row = jobs.get(identifier)
    require(
        row is not None
        and row.get("job") == identifier
        and row.get("unit") == name
        and row.get("type") in {"start", "stop", "restart"}
        and row.get("state") in {"waiting", "running"},
        "linux-job-join",
    )
    assert row is not None
    return {"id": identifier, "kind": row["type"], "state": row["state"]}


class LinuxHost:
    """Concrete observations/actions; construction alone has no service effects."""

    def __init__(
        self,
        manifest_path: Path,
        manifest_sha256: str,
        *,
        execute: bool = False,
        io_: LinuxIO | None = None,
    ):
        self.io = io_ or LinuxIO()
        self.manifest_path, self.manifest_sha256 = manifest_path, manifest_sha256
        self.manifest, self.plan = load_manifest(
            manifest_path, manifest_sha256, self.io, execute=execute
        )
        self.execute = execute
        self.adapter_qualification_sha256 = self.plan["adapter_qualification_sha256"]
        self.state = Path(self.plan["state_root"])
        self.root = Path(self.plan["run_root"])
        self.lease_fd: int | None = None
        self.last_raw: dict[str, Any] = {}
        self._authority_cache: tuple[str, dict[str, Any]] | None = None
        self._champion_cache: tuple[str, dict[str, str]] | None = None
        self._membership_changed = False

    def prepared_directory(self) -> Path:
        path = self.state.with_name(self.state.name + ".prepared")
        for other in (
            self.state,
            self.root,
            Path(self.plan["runtime_root"]).parent,
            Path(self.plan["probe_output"]),
        ):
            require(
                not path.is_relative_to(other) and not other.is_relative_to(path),
                "linux-prepared-directory-overlap",
            )
        return path

    def clock(self) -> core.Clock:
        return core.Clock(self.io.boot_id(), self.io.now(), time.time_ns())

    def _worker(
        self, operation: str, deadline: float, request: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        allowed = {
            "authority",
            "boundary",
            "check-boundary",
            "champions",
            "apply-r4",
            "repair-r4",
            "backup",
        }
        require(operation in allowed, "linux-worker-operation")
        if operation in {"apply-r4", "repair-r4", "backup"}:
            require(
                self.execute and self.lease_fd is not None,
                "linux-worker-not-authorized",
            )
        request_path = None
        if request is not None:
            request_path = self.state / (
                "request-" + operation + "-" + checksum(encoded(request)) + ".json"
            )
            if not request_path.exists():
                self.io.atomic(request_path, encoded(request))
        argv = [
            self.manifest["helper_python"]["path"],
            "-s",
            str(Path(__file__).resolve()),
            "worker",
            "--manifest",
            str(self.manifest_path),
            "--sha256",
            self.manifest_sha256,
            "--operation",
            operation,
        ]
        if request_path:
            argv += [
                "--request",
                str(request_path),
                "--request-sha256",
                checksum(encoded(request)),
            ]
        env = {
            "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG": "C",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": self.manifest["control_root"],
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        return self.io.child(argv, env, deadline)

    def _authority(self, deadline: float) -> core.Authority:
        # Cache only an exact metadata signature; the validator itself verifies
        # the registered before/after bytes and never repairs unknown authority.
        names = (
            "strength-freshness-plan.json",
            "strength-freshness-apply.json",
            "profile.sha256",
            "source-commit.txt",
            "continuous-migrations.jsonl",
        )
        signature = core.digest(
            {
                name: checksum(self.io.read(self.root / name, 2**20, deadline))
                if (self.root / name).exists()
                else None
                for name in names
            }
        )
        self._authority_cache = signature, self._worker("authority", deadline)
        result = self._authority_cache[1]
        return core.Authority(
            result["phase"], result["plan_sha256"], result["evidence_sha256"]
        )

    def _lease(self, deadline: float) -> Mapping[str, Any] | None:
        path = self.state / LEASE
        if not path.exists():
            return None
        value = self.io.json(path, deadline)
        owner = value.get("owner", {})
        process = self.io.process(int(owner.get("pid", 0)), deadline)
        if process is None or process.get("start_ticks") != owner.get("start_ticks"):
            return None
        lock = Path(self.plan["exclusion_path"])
        st = lock.lstat()
        require(not lock.is_symlink() and stat.S_ISREG(st.st_mode), "linux-lease-file")
        device = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
        rows = (
            self.io.read(Path("/proc/locks"), 4 * 2**20, deadline).decode().splitlines()
        )
        require(
            any(
                len(parts := row.split()) >= 8
                and parts[1:4] == ["FLOCK", "ADVISORY", "WRITE"]
                and parts[4] == str(owner["pid"])
                and parts[5].lower() == device.lower()
                for row in rows
            ),
            "linux-lease-not-kernel-held",
        )
        return value

    def observe(self, deadline: float) -> core.Observation:
        for _ in range(3):
            try:
                return self._observe_once(deadline)
            except ObservationChanged:
                if self.io.now() + 0.01 >= deadline:
                    break
                self.io.sleep(0.01)
        raise TimeoutError("linux-observation-changing-within-bound")

    def _observe_once(self, deadline: float) -> core.Observation:
        clock = self.clock()
        require(clock.monotonic < deadline, "linux-observe-deadline")
        jobs = self._jobs(deadline)
        units, support, raw, auxiliary = {}, {}, {}, {}
        role_names = {s["name"]: role for role, s in self.plan["units"].items()}
        for name in self.manifest["units"]:
            unit, metadata = self._unit(name, jobs, deadline)
            raw[name] = asdict(unit)
            if name in role_names:
                units[role_names[name]] = unit
            else:
                auxiliary[name] = unit
                support[name] = {
                    k: metadata[k]
                    for k in (
                        "definition_sha256",
                        "environment_sha256",
                        "enabled",
                        "job",
                    )
                }
        uuids, hardware, owners = self._gpus(units, deadline)
        authority = self._authority(deadline)
        lease = self._lease(deadline)
        ack_path = self.state / "linux-guard-ack.json"
        ack = self.io.json(ack_path, deadline) if ack_path.exists() and lease else None
        self._membership_changed = False
        progress = self._progress(units, clock, deadline)
        if self._membership_changed:
            jobs = self._jobs(deadline)
            for role, spec in self.plan["units"].items():
                units[role] = self._unit(spec["name"], jobs, deadline)[0]
            uuids, hardware, owners = self._gpus(units, deadline)
            progress = None
        backup_unit = auxiliary[self.manifest["backup_worker_unit"]]
        backup = self._backup_result(deadline, backup_unit)
        end_clock = self.clock()
        require(
            end_clock.boot_id == clock.boot_id
            and end_clock.monotonic >= clock.monotonic,
            "linux-observation-clock-raced",
        )
        self.last_raw = {
            "clock": asdict(end_clock),
            "units": raw,
            "jobs": jobs,
            "hardware_sha256": hardware,
            "owners": [asdict(o) for o in owners],
            "authority": asdict(authority),
        }
        require(self.io.now() < deadline, "linux-observation-overrun")
        return core.Observation(
            end_clock,
            units,
            uuids,
            hardware,
            owners,
            authority,
            True,
            lease,
            ack,
            progress,
            backup,
            support,
        )

    def guard_reentry_evidence(
        self, previous: core.Process, deadline: float
    ) -> core.GuardReentryEvidence:
        current = self.io.process(os.getpid(), deadline)
        old = self.io.process(previous.pid, deadline)
        require(
            current is not None and self.io.now() < deadline, "linux-reentry-process"
        )
        assert current is not None
        invocation = os.environ.get("INVOCATION_ID", "")
        require(re.fullmatch(r"[0-9a-f]{32}", invocation), "linux-reentry-invocation")
        return core.GuardReentryEvidence(
            self.clock(),
            core.Process(current["pid"], current["start_ticks"]),
            invocation,
            None if old is None else core.Process(old["pid"], old["start_ticks"]),
        )

    def guard_boot_evidence(self, deadline: float) -> core.GuardReentryEvidence:
        current = self.io.process(os.getpid(), deadline)
        require(current is not None, "linux-boot-process")
        assert current is not None
        invocation = os.environ.get("INVOCATION_ID", "")
        require(re.fullmatch(r"[0-9a-f]{32}", invocation), "linux-boot-invocation")
        return core.GuardReentryEvidence(
            self.clock(),
            core.Process(current["pid"], current["start_ticks"]),
            invocation,
            None,
        )

    def verify_inputs(self, plan: dict[str, Any], deadline: float) -> None:
        require(plan == self.plan, "linux-input-plan-drift")
        for pin in self.manifest["prestaged_inputs"]:
            self.io.pin(pin, deadline)
        require(self._authority(deadline).phase == "r3", "linux-initial-authority")
        require(not Path(plan["probe_output"]).exists(), "linux-probe-output-reused")
        require(
            not self.state.exists() and not self.state.is_symlink(),
            "linux-core-journal-must-be-fresh",
        )
        prepared = self.prepared_directory()
        journal = core.Journal(prepared)
        if not prepared.exists():
            journal.create()
        journal._check()
        require(
            {p.name for p in prepared.iterdir()} <= {AUTHORIZATION},
            "linux-prepared-foreign-files",
        )
        authorization = {
            "format": FORMAT + "-authorization",
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "core_plan_sha256": core.digest(plan),
            "adapter_sha256": self.manifest["adapter_sha256"],
        }
        path = prepared / AUTHORIZATION
        if path.exists():
            require(
                self.io.json(path, deadline) == authorization,
                "linux-authorization-reused",
            )
        else:
            self.io.atomic(path, encoded(authorization))

    def capture_boundary(self, plan: dict[str, Any], deadline: float) -> dict[str, Any]:
        from scripts.strength_freshness_units import require_stopped_support

        require(plan == self.plan, "linux-boundary-plan")
        require_stopped_support(self, deadline)
        value = self._worker("boundary", deadline)
        checkpoint = value["checkpoint"]
        source = Path(checkpoint["path"])
        retained = self.state / ("checkpoint-" + checkpoint["sha256"] + ".pt")
        require(
            source.resolve() == source and source.stat().st_size == checkpoint["bytes"],
            "linux-boundary-checkpoint-retention",
        )
        if retained.exists() or retained.is_symlink():
            require(
                retained.resolve() == retained
                and retained.is_file()
                and retained.stat().st_size == checkpoint["bytes"]
                and os.path.samefile(source, retained),
                "linux-boundary-hardlink",
            )
        else:
            os.link(source, retained, follow_symlinks=False)
        require(os.path.samefile(source, retained), "linux-boundary-hardlink")
        fd = os.open(self.state, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        value["checkpoint_retention"] = {**checkpoint, "path": str(retained)}
        record = self.state / "linux-stopped-boundary.json"
        if record.exists() or record.is_symlink():
            require(
                self.io.read(record, 16 * 2**20, deadline) == encoded(value),
                "linux-boundary-record-changed",
            )
        else:
            self.io.atomic(record, encoded(value))
        return value

    def check_boundary(self, boundary: dict[str, Any], deadline: float) -> None:
        value = self._worker("check-boundary", deadline, boundary)
        require(value.get("unchanged") is True, "linux-stopped-boundary-drift")

    def _run_path(self, relative: str) -> Path:
        require(
            not Path(relative).is_absolute()
            and ".." not in Path(relative).parts
            and Path(relative).as_posix() == relative,
            "linux-run-relative-path",
        )
        return self.root / relative

    def _progress(
        self, units: Mapping[str, core.Unit], clock: core.Clock, deadline: float
    ) -> Mapping[str, Any] | None:
        from scripts.strength_freshness_progress import (
            ProgressPolicy,
            ProgressViolation,
            verify_progress,
        )
        from scripts.strength_freshness_auxiliary import (
            IMPORT_ENVIRONMENT,
            classify_auxiliary,
        )

        state_path = self.state / "state.json"
        if not state_path.exists():
            return None
        state = self.io.json(state_path, deadline)
        if "boundary" not in state:
            return None
        role = "r4" if units["r4"].active == "active" else "r3"
        unit = units[role]
        if unit.active != "active" or unit.main is None:
            return None
        spec = self.manifest["telemetry"]
        captures = {}
        for label in ("coordinator", "continuation_state"):
            path = self._run_path(spec[label])
            if not path.exists():
                if label == "coordinator":
                    return None
                captures[label] = {}
                continue
            captures[label] = self.io.json(path, deadline)
        for label in ("heartbeats", "cohorts"):
            captures[label] = {}
            for name, relative in spec[label].items():
                path = self._run_path(relative)
                if not path.exists():
                    continue
                captures[label][name] = self.io.json(path, deadline)
        captures["metrics"] = self.io.tail(self._run_path(spec["metrics"]), deadline)
        pointer = (
            self.io.read(self.root / "profile.sha256", 65536, deadline)
            .decode()
            .strip()
            .split(maxsplit=1)
        )
        require(len(pointer) == 2, "linux-profile-pointer")
        profile = canonical(pointer[1])
        require(profile.parent == self.root, "linux-profile-path")
        captures["profile_sha256"] = checksum(self.io.read(profile, 2**20, deadline))
        require(captures["profile_sha256"] == pointer[0], "linux-profile-hash")
        captures["source_commit"] = (
            self.io.read(self.root / "source-commit.txt", 256, deadline)
            .decode()
            .strip()
        )
        db = self._run_path(spec["replay_db"])
        require(db.resolve() == db, "linux-replay-path")
        with sqlite3.connect(
            db.as_uri() + "?mode=ro",
            uri=True,
            timeout=min(1.0, max(0, deadline - self.io.now())),
        ) as connection:
            connection.execute("PRAGMA query_only=ON")
            connection.set_progress_handler(lambda: int(self.io.now() >= deadline), 100)
            identity = self.plan["run_identity"]
            row = connection.execute(
                "SELECT committed_samples,updated_ns,history_complete FROM run_counters WHERE run_id=? AND generation_family=?",
                (identity["run_id"], identity["generation_family"]),
            ).fetchone()
            run = connection.execute(
                "SELECT generation_family,created_ns FROM runs WHERE run_id=?",
                (identity["run_id"],),
            ).fetchone()
        require(
            row is not None
            and run == (identity["generation_family"], identity["created_ns"])
            and row[2] == 1,
            "linux-replay-identity",
        )
        assert row is not None
        captures["replay_counter"] = {
            "run_id": identity["run_id"],
            "generation_family": identity["generation_family"],
            "committed_samples": row[0],
            "updated_ns": row[1],
        }
        stat_text = self.io.read(Path("/proc/stat"), 2**20, deadline).decode()
        boot = [
            line.split()[1]
            for line in stat_text.splitlines()
            if line.startswith("btime ")
        ]
        require(len(boot) == 1 and boot[0].isdigit(), "linux-boot-birth-clock")
        ticks = os.sysconf("SC_CLK_TCK")
        births = {
            p: (int(boot[0]) + 1) * 10**9 + (p.start_ticks * 10**9 + ticks - 1) // ticks
            for p in unit.members
        }
        policy = self.manifest["process_runtime"][role]
        certifying = {unit.main.pid, captures["coordinator"].get("coordinator_pid")}
        certifying.update(
            row.get("pid")
            for row in captures["coordinator"].get("workers", {}).values()
            if isinstance(row, dict)
        )
        certifying.update(
            row.get("pid")
            for row in captures["cohorts"].values()
            if isinstance(row, dict)
        )
        parent_roles = {
            unit.main.pid: "controller",
            captures["coordinator"].get("coordinator_pid"): "coordinator",
        }
        parent_roles.update(
            {
                row.get("pid"): name
                for name, row in captures["coordinator"].get("workers", {}).items()
                if isinstance(row, dict)
            }
        )
        additional_children = False
        process_evidence: dict[int, dict[str, Any]] = {}
        for member in unit.members:
            require(self.io.now() < deadline, "linux-progress-pid-deadline")
            current = self.io.process(member.pid, deadline)
            if current is None or current["start_ticks"] != member.start_ticks:
                self._membership_changed = True
                continue
            require(
                current["cgroup"] == "0::" + unit.cgroup
                or current["cgroup"].startswith("0::" + unit.cgroup + "/"),
                "linux-progress-cgroup",
            )
            raw_env = self.io.read(Path(f"/proc/{member.pid}/environ"), 2**20, deadline)
            environment = dict(
                item.split(b"=", 1) for item in raw_env.split(b"\0") if b"=" in item
            )
            item = {
                **current,
                **self.io.process_origin(member.pid, deadline),
                "environment": {
                    key: os.fsdecode(environment[key.encode()])
                    for key in IMPORT_ENVIRONMENT
                    if key.encode() in environment
                },
            }
            process_evidence[member.pid] = item
            if member.pid not in certifying:
                item["argv"] = [
                    os.fsdecode(part)
                    for part in self.io.read(
                        Path(f"/proc/{member.pid}/cmdline"), 65536, deadline
                    ).split(b"\0")
                    if part
                ]
                continue
            require(
                item["exe"] in policy["executables"]
                and item["cwd"] in policy["working_directories"],
                "linux-process-runtime-origin",
            )
            require(
                all(
                    environment.get(key.encode())
                    == (value.encode() if value is not None else None)
                    for key, value in policy["environment"].items()
                ),
                "linux-process-import-environment",
            )
            expected = {
                key: policy["environment"].get(key) for key in IMPORT_ENVIRONMENT
            }
            expected["CUDA_VISIBLE_DEVICES"] = policy.get("worker_devices", {}).get(
                parent_roles.get(member.pid), expected["CUDA_VISIBLE_DEVICES"]
            )
            require(
                all(
                    item["environment"].get(key) == value
                    for key, value in expected.items()
                ),
                "linux-process-import-or-device-drift",
            )
        auxiliaries = {}
        aux_policy = self.manifest.get("auxiliary_policies", {}).get(role)
        for pid, item in process_evidence.items():
            if pid in certifying:
                continue
            ppid = item.get("ppid")
            parent = process_evidence.get(ppid) if type(ppid) is int else None
            parent_role = parent_roles.get(item.get("ppid"))
            classification = (
                classify_auxiliary(
                    aux_policy, item, parent=parent, parent_role=parent_role
                )
                if aux_policy is not None
                and parent is not None
                and parent_role is not None
                else None
            )
            if classification is None:
                additional_children = True
            else:
                assert parent is not None
                again = self.io.process(parent["pid"], deadline)
                child_again = self.io.process(pid, deadline)
                if (
                    again is None
                    or child_again is None
                    or again["start_ticks"] != parent["start_ticks"]
                    or child_again["start_ticks"] != item["start_ticks"]
                ):
                    self._membership_changed = True
                    additional_children = True
                    continue
                raw_parent = dict(
                    part.split(b"=", 1)
                    for part in self.io.read(
                        Path(f"/proc/{parent['pid']}/environ"), 2**20, deadline
                    ).split(b"\0")
                    if b"=" in part
                )
                parent_env = {
                    key: os.fsdecode(raw_parent[key.encode()])
                    for key in IMPORT_ENVIRONMENT
                    if key.encode() in raw_parent
                }
                if (
                    parent_env != parent["environment"]
                    or self.io.process_origin(parent["pid"], deadline)
                    != {key: parent[key] for key in ("exe", "cwd")}
                    or again["cgroup"] != parent["cgroup"]
                    or child_again.get("ppid") != parent["pid"]
                ):
                    raise ObservationChanged("linux-auxiliary-parent-raced")
                auxiliaries[str(pid)] = {
                    "start_ticks": item["start_ticks"],
                    "parent_pid": item["ppid"],
                    "kind": classification,
                }
        if self.io.members(unit.cgroup, deadline) != unit.members:
            self._membership_changed = True
        champion_pin = checksum(
            self.io.read(self.root / "learner/champion.json", 2**20, deadline)
        )
        if self._champion_cache is None or self._champion_cache[0] != champion_pin:
            self._champion_cache = champion_pin, self._worker("champions", deadline)
        arena_ok = arena_prefix_preserved(
            state["boundary"]["arena_evidence"], self.root, self.io, deadline
        )
        capture_clock = self.clock()
        require(
            capture_clock.boot_id == clock.boot_id
            and capture_clock.monotonic >= clock.monotonic,
            "linux-progress-clock-raced",
        )
        try:
            result = verify_progress(
                ProgressPolicy.from_dict(spec["policies"][role]),
                captures,
                role=role,
                unit=unit,
                clock=capture_clock,
                births=births,
                verified_champions=self._champion_cache[1],
                arena_prefix_preserved=arena_ok,
            )
        except ProgressViolation as error:
            return {"role": role, "contract_failure": str(error)}
        if self._membership_changed or additional_children:
            return None
        if result is not None:
            result["auxiliary_processes"] = auxiliaries
            result["auxiliary_policy_sha256"] = (
                core.digest(aux_policy) if aux_policy is not None else None
            )
            self.last_raw["progress_capture_sha256"] = core.digest(captures)
        return result

    def _backup_result(
        self, deadline: float, unit: core.Unit
    ) -> Mapping[str, Any] | None:
        path = self.state / "linux-backup-result.json"
        if not path.exists() or not unit.dead:
            return None
        value = self.io.json(path, deadline)
        owner = self.io.json(self.state / "linux-backup-started.json", deadline)
        request = self.io.json(self.state / "linux-backup-request.json", deadline)
        require(
            owner.get("format") == FORMAT + "-backup-started"
            and owner.get("schema_version") == 1
            and owner.get("manifest_sha256") == self.manifest_sha256
            and owner.get("core_plan_sha256") == core.digest(self.plan)
            and owner.get("nonce") == request["nonce"]
            and owner.get("unit_name") == unit.name
            and owner.get("entered_monotonic") == unit.entered_monotonic
            and owner["main"]["pid"] == self._exit_pids[unit.name],
            "linux-backup-started-binding",
        )
        require(
            owner["boot_id"] == self.io.boot_id()
            and value.get("invocation_id") == owner["invocation_id"]
            and (not unit.invocation_id or unit.invocation_id == owner["invocation_id"])
            and unit.result == "success"
            and unit.exit_code == 0
            and checksum(encoded(request))
            == owner["request_sha256"]
            == value.get("request_sha256"),
            "linux-backup-exit-or-binding",
        )
        require(
            value.get("manifest_sha256") == self.manifest_sha256,
            "linux-backup-result-binding",
        )
        proof_root = Path(self.manifest["proof_archive_root"]) / self.plan["attempt_id"]
        pins = value["metadata_pins"]
        require(
            len(pins) == 3
            and all(
                type(p.get("bytes")) is int and 0 <= p["bytes"] <= 16 * 2**20
                for p in pins
            )
            and pins[0]["path"] == str(proof_root / "manifest.json")
            and Path(pins[1]["path"]).is_relative_to(
                Path(self.manifest["backup_root"]) / "snapshots"
            )
            and pins[2]["path"] == pins[1]["path"] + ".commit",
            "linux-backup-metadata-paths",
        )
        for pin in value["metadata_pins"]:
            self.io.pin(pin, deadline)
        proof = self.io.json(
            Path(value["metadata_pins"][0]["path"]), deadline, 16 * 2**20
        )
        catalog = self.io.json(Path(pins[1]["path"]), deadline, 16 * 2**20)
        commit = self.io.json(Path(pins[2]["path"]), deadline)
        required = set(request["required_proof_sha256"])
        require(
            proof["nonce"] == request["nonce"]
            and proof["core_plan_sha256"] == core.digest(self.plan)
            and proof["proof_closure_sha256"] == request["proof_closure_sha256"]
            and required <= {p["sha256"] for p in proof["objects"]}
            and set(proof["required_core_sha256"]) == required
            and value["retained_objects"] == proof["objects"]
            and set(value["core_result"]["verified_artifact_sha256"]) == required,
            "linux-backup-proof-inventory",
        )
        require(
            proof["manifest_sha256"] == self.manifest_sha256
            and catalog["run_id"] == self.plan["run_identity"]["run_id"]
            and catalog["generation_family"]
            == self.plan["run_identity"]["generation_family"]
            and pins[1]["sha256"]
            == commit["sha256"]
            == value["core_result"]["catalog_sha256"],
            "linux-backup-catalog-binding",
        )
        # Large objects were streamed by the qualified writer. Observe retained
        # immutable inode/size metadata, never rehash an old payload every tick.
        for pin in value["retained_objects"]:
            p = canonical(pin["path"])
            require(
                p == proof_root / "objects" / pin["sha256"],
                "linux-backup-retained-path",
            )
            s = p.lstat()
            require(
                not p.is_symlink()
                and stat.S_ISREG(s.st_mode)
                and s.st_size == pin["bytes"]
                and not s.st_mode & 0o222,
                "linux-backup-object-metadata",
            )
        return value["core_result"]

    def _action_admission(
        self, details: Mapping[str, Any], deadline: float, *, arm: bool = False
    ) -> core.Observation:
        require(self.execute, "linux-mutations-unqualified")
        require(
            checksum(Path(__file__).read_bytes()) == self.manifest["adapter_sha256"],
            "linux-executing-adapter-drift",
        )
        require(
            details.get("guard_plan_sha256") == core.digest(self.plan),
            "linux-action-plan",
        )
        observed = self.observe(deadline)
        require(
            observed.clock.boot_id == details["expected_boot_id"],
            "linux-action-boot-raced",
        )
        if "expected_units" in details:
            require(
                {r: asdict(u) for r, u in observed.units.items()}
                == details["expected_units"],
                "linux-action-unit-raced",
            )
        if "expected_support" in details:
            require(
                dict(observed.support) == details["expected_support"],
                "linux-action-support-raced",
            )
        if "expected_guard" in details:
            require(
                asdict(observed.units["guard"]) == details["expected_guard"],
                "linux-retirement-guard-raced",
            )
        if "expected_own_lease" in details:
            require(
                observed.lease == details["expected_own_lease"],
                "linux-retirement-lease-raced",
            )
        if not arm:
            require(
                self.lease_fd is not None
                and observed.lease is not None
                and observed.lease.get("owner", {}).get("pid") == os.getpid()
                and observed.lease.get("nonce") == details["nonce"],
                "linux-action-lease-not-owned",
            )
        return observed

    def _enable(self, unit: str, enabled: bool, deadline: float) -> None:
        self.io.command(
            ["systemctl", "enable" if enabled else "disable", unit], deadline
        )

    def _support(self, stage: str, deadline: float) -> None:
        from scripts.strength_freshness_units import transition

        transition(self, stage, deadline)

    def resume_support_tail(self, action: Mapping[str, Any]) -> None:
        """Resume only a previously journaled backup dispatch; never duplicate it."""
        if not str(action.get("kind", "")).startswith("restore-support-and-backup-"):
            return
        from scripts.strength_freshness_units import DIRECTORY

        state = core.Journal(self.state).read("state.json")
        require(
            state["last_action"] == action
            and not state.get("runtime_authority_retired"),
            "linux-backup-tail-action",
        )
        identity = core.digest(
            {
                "action": action,
                "attempt_id": state["attempt_id"],
                "nonce": state["nonce"],
            }
        )
        intent = self.io.json(
            self.state / DIRECTORY / (identity + ".intent.json"), self.io.now() + 5
        )
        now = self.clock()
        deadline = min(
            state["deadline"],
            now.monotonic + (intent["deadline_wall_ns"] - now.wall_ns) / 1e9,
        )
        require(now.monotonic < deadline, "linux-backup-tail-deadline")
        role = str(action["kind"])[-2:]
        observed = self.observe(min(deadline, now.monotonic + 5))
        unit = observed.units[role]
        owner = state["owners"].get(role)
        if (
            unit.active != "active"
            or unit.main is None
            or owner is None
            or owner
            != {
                "main": asdict(unit.main),
                "invocation_id": unit.invocation_id,
                "boot_id": observed.clock.boot_id,
            }
        ):
            return
        require(
            self.lease_fd is not None
            and observed.lease is not None
            and observed.lease.get("owner", {}).get("pid") == os.getpid(),
            "linux-backup-tail-lease",
        )
        request = {
            **action["details"],
            "guard_plan_sha256": core.digest(self.plan),
            "nonce": state["nonce"],
            "deadline_wall_ns": intent["deadline_wall_ns"],
        }
        path = self.state / "linux-backup-request.json"
        if path.exists():
            require(
                self.io.json(path, deadline) == request, "linux-backup-request-changed"
            )
        else:
            self.io.atomic(path, encoded(request))
        backup, _ = self._unit(
            self.manifest["backup_worker_unit"], self._jobs(deadline), deadline
        )
        if not backup.dead:
            return
        if (self.state / "linux-backup-started.json").exists() or (
            self.state / "linux-backup-result.json"
        ).exists():
            return
        dispatch = self.state / "linux-backup-dispatch.json"
        value = {
            "guard_plan_sha256": core.digest(self.plan),
            "nonce": state["nonce"],
            "request_sha256": checksum(encoded(request)),
            "deadline_wall_ns": intent["deadline_wall_ns"],
        }
        if dispatch.exists():
            require(
                self.io.json(dispatch, deadline) == value,
                "linux-backup-dispatch-changed",
            )
            return  # Uncertain prior dispatch is proof-incomplete, never a retry.
        self.io.atomic(dispatch, encoded(value))
        self.io.action("start", self.manifest["backup_worker_unit"], deadline)

    def perform(self, kind: str, details: dict[str, Any], deadline: float) -> None:
        obs = self._action_admission(details, deadline, arm=kind == "arm-guard")
        evidence = self.state / "linux-action-observations"
        evidence.mkdir(mode=0o700, exist_ok=True)
        require(evidence.resolve() == evidence, "linux-action-evidence-path")
        self.io.atomic(
            evidence / f"{time.monotonic_ns()}-{kind}.json",
            encoded(
                {
                    "manifest_sha256": self.manifest_sha256,
                    "kind": kind,
                    "observation": asdict(obs),
                }
            ),
        )
        if kind == "arm-guard":
            require(obs.units["guard"].dead, "linux-guardian-already-running")
            self._enable(self.plan["units"]["guard"]["name"], True, deadline)
            self.io.action("start", self.plan["units"]["guard"]["name"], deadline)
        elif kind == "pause-support-and-stop-r3":
            for name in self.plan["support_transition"]["before"]:
                self._enable(name, False, deadline)
                self.io.action("stop", name, deadline)
            self._enable(self.plan["units"]["r3"]["name"], False, deadline)
            self.io.action("stop", self.plan["units"]["r3"]["name"], deadline)
        elif kind.startswith("start-") and kind[6:] in {"r3", "r4", "probe"}:
            role = kind[6:]
            require(
                not obs.owners and obs.units[role].dead, "linux-start-resource-overlap"
            )
            if role == "probe":
                require(
                    details.get("challenge") == str(self.state / "challenge.json"),
                    "linux-probe-challenge-path",
                )
                require(
                    checksum(
                        self.io.read(self.state / "challenge.json", 2**20, deadline)
                    )
                    == details["challenge_sha256"],
                    "linux-probe-challenge-pin",
                )
            self.io.action("start", self.plan["units"][role]["name"], deadline)
        elif kind.startswith("stop-") and kind[5:] in {"r3", "r4", "probe"}:
            self.io.action("stop", self.plan["units"][kind[5:]]["name"], deadline)
        elif kind in {"apply-r4", "repair-r4"}:
            from scripts.strength_freshness_units import require_stopped_support

            require_stopped_support(self, deadline)
            require(
                not obs.owners
                and all(obs.units[r].dead for r in ("r3", "r4", "probe")),
                "linux-apply-resource-overlap",
            )
            self._worker(kind, deadline, details)
            self._authority_cache = None
        elif kind == "fence-terminal-restarts":
            for role in ("r3", "r4", "probe"):
                self._enable(self.plan["units"][role]["name"], False, deadline)
            for name in self.plan["support_transition"]["before"]:
                self._enable(name, False, deadline)
                self.io.action("stop", name, deadline)
        elif kind == "restore-support-before-stop":
            self._support("before", deadline)
        elif kind.startswith(
            ("restore-support-and-backup-", "restore-support-without-proof-")
        ):
            self._support(details["support_stage"], deadline)
            if kind.startswith("restore-support-and-backup-"):
                self.resume_support_tail(
                    core.Journal(self.state).read("state.json")["last_action"]
                )
        elif kind in {"retire-guard", "retire-guard-only"}:
            state = self.io.json(self.state / "state.json", deadline)
            require(
                state.get("runtime_authority_retired") is True,
                "linux-retirement-not-sealed",
            )
            self._enable(self.plan["units"]["guard"]["name"], False, deadline)
            self.io.atomic(
                self.state / "linux-retirement-intent.json",
                encoded(
                    {
                        "core_plan_sha256": core.digest(self.plan),
                        "at": asdict(self.clock()),
                    }
                ),
            )
            self.release_lease()
            raise GuardianRetired()
        else:
            raise core.Refusal("linux-unknown-action")

    @contextmanager
    def hold_lease(self) -> Iterator[None]:
        require(self.execute and self.lease_fd is None, "linux-lease-not-authorized")
        path = Path(self.plan["exclusion_path"])
        require(path.resolve() == path and path.parent.is_dir(), "linux-lock-path")
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            s = os.fstat(fd)
            require(
                stat.S_ISREG(s.st_mode)
                and s.st_uid == os.geteuid()
                and stat.S_IMODE(s.st_mode) == 0o600,
                "linux-lock-protection",
            )
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lease_fd = fd
            state = core.Journal(self.state).read("state.json")
            now = self.clock()
            process = self.io.process(os.getpid())
            invocation = os.environ.get("INVOCATION_ID", "")
            require(
                process is not None and re.fullmatch(r"[0-9a-f]{32}", invocation),
                "linux-guardian-self-identity",
            )
            assert process is not None
            guard_name = self.plan["units"]["guard"]["name"]
            unit, _ = self._unit(
                guard_name, self._jobs(now.monotonic + 5), now.monotonic + 5
            )
            require(
                unit.main == core.Process(process["pid"], process["start_ticks"])
                and unit.invocation_id == invocation,
                "linux-guardian-not-executing-unit",
            )
            lease = {
                "path": str(path),
                "held": True,
                "attempt_id": state["attempt_id"],
                "nonce": state["nonce"],
                "plan_sha256": core.digest(self.plan),
                "boot_id": now.boot_id,
                "invocation_id": invocation,
                "owner": {"pid": process["pid"], "start_ticks": process["start_ticks"]},
            }
            self.io.atomic(self.state / LEASE, encoded(lease), overwrite=True)
            remaining = (
                min(state["remaining_ns"], state["deadline_wall_ns"] - now.wall_ns)
                / 1e9
            )
            deadline = (
                state["deadline"]
                if now.boot_id == state["active_boot_id"]
                else now.monotonic + remaining
            )
            ack = {
                k: lease[k]
                for k in (
                    "plan_sha256",
                    "attempt_id",
                    "nonce",
                    "boot_id",
                    "invocation_id",
                )
            }
            ack["deadline"] = deadline
            self.io.atomic(
                self.state / "linux-guard-ack.json", encoded(ack), overwrite=True
            )
            yield
        finally:
            if self.lease_fd is not None:
                self.release_lease()
            else:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def release_lease(self) -> None:
        if self.lease_fd is not None:
            fcntl.flock(self.lease_fd, fcntl.LOCK_UN)
            os.close(self.lease_fd)
            self.lease_fd = None

    def _jobs(self, deadline: float) -> dict[int, dict[str, Any]]:
        return parse_jobs_text(self.io.command(list(JOBS_COMMAND), deadline))

    def _unit(
        self,
        name: str,
        jobs: Mapping[int, dict[str, Any]],
        deadline: float,
        *,
        allow_known_partial: bool = False,
    ) -> tuple[core.Unit, dict[str, Any]]:
        raw = self.io.command(
            [
                "systemctl",
                "show",
                name,
                "--all",
                "--no-pager",
                "--property=" + ",".join(property_names(name)),
            ],
            deadline,
        )
        rows = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        if name.endswith(".service") and "EnvironmentFiles" not in rows:
            rows["EnvironmentFiles"] = ""  # systemd's known empty-array omission
        require(set(rows) == set(property_names(name)), "linux-unit-incomplete")
        rows = stable_properties(rows)
        spec = self.manifest["units"][name]
        actual = {k: rows[k] for k in set(property_names(name)) - VARIABLE}
        matched = [
            stage
            for stage in ("before", "after")
            if actual == stable_properties(spec[stage]["properties"])
        ]
        require(matched, "linux-unit-properties-drift")
        source = self.io.read(Path(spec["installed_path"]), 2**20, deadline)
        file_stages = [
            stage
            for stage in ("before", "after")
            if checksum(source) == spec[stage]["unit"]["sha256"]
        ]
        require(file_stages, "linux-installed-unit-drift")
        candidates = [stage for stage in matched if stage in file_stages]
        by_stage = {
            stage: {p["path"]: p for p in spec[stage]["environment_files"]}
            for stage in ("before", "after")
        }
        env_stages = set(("before", "after"))
        observed_env = []
        for path in sorted(set(by_stage["before"]) | set(by_stage["after"])):
            known = [mapping[path] for mapping in by_stage.values() if path in mapping]
            if not self.io.exists(Path(path)):
                require(
                    any(path not in mapping for mapping in by_stage.values()),
                    "linux-env-missing",
                )
                env_stages &= {
                    stage for stage, mapping in by_stage.items() if path not in mapping
                }
                continue
            data = self.io.read(Path(path), 2**20, deadline)
            metadata = self.io.file_metadata(Path(path), deadline)
            require(
                metadata["uid"] == os.geteuid() and metadata["gid"] == os.getegid(),
                "linux-env-ownership",
            )
            matches = [
                pin
                for pin in known
                if len(data) == pin["bytes"]
                and checksum(data) == pin["sha256"]
                and pin.get("mode", 0o600) == metadata["mode"]
            ]
            require(matches, "linux-env-unknown-bytes")
            observed_env.append(matches[0])
            env_stages &= {
                stage
                for stage, mapping in by_stage.items()
                if path not in mapping
                or (
                    mapping[path]["sha256"] == checksum(data)
                    and mapping[path].get("mode", 0o600) == metadata["mode"]
                )
            }
        candidates = [stage for stage in candidates if stage in env_stages]
        require(candidates or allow_known_partial, "linux-support-known-partial-state")
        stage = candidates[0] if candidates else "mixed"
        env_pins = (
            spec[stage]["environment_files"] if stage != "mixed" else observed_env
        )
        if name.endswith(".timer"):
            require(not env_pins, "linux-timer-environment")
        env_sha = env_identity(rows.get("Environment", ""), env_pins)
        cgroup = rows.get("ControlGroup") or "/system.slice/" + name
        require(cgroup == "/system.slice/" + name, "linux-unit-cgroup-drift")
        members = self.io.members(cgroup, deadline) if name.endswith(".service") else ()
        main = None
        if name.endswith(".service") and rows["MainPID"] != "0":
            require(rows["MainPID"].isdecimal(), "linux-main-pid")
            p = self.io.process(int(rows["MainPID"]), deadline)
            if p is None:
                raise ObservationChanged("linux-main-disappeared")
            assert p is not None
            main = core.Process(p["pid"], p["start_ticks"])
            if main not in members and (
                p["cgroup"] == "0::" + cgroup
                or p["cgroup"].startswith("0::" + cgroup + "/")
            ):
                raise ObservationChanged("linux-main-cgroup-changed")
            require(main in members, "linux-main-outside-cgroup")
        enabled = rows["UnitFileState"] in {"enabled", "enabled-runtime"}
        require(
            rows["UnitFileState"]
            in {
                "enabled",
                "enabled-runtime",
                "disabled",
                "static",
                "masked",
                "masked-runtime",
            },
            "linux-enablement-unknown",
        )
        from scripts.strength_freshness_units import _known_links, verify_boot_edges

        if allow_known_partial:
            _known_links(self, name, deadline)
        else:
            verify_boot_edges(self, name, enabled, deadline)
        unit = core.Unit(
            name,
            checksum(source),
            cgroup,
            rows.get("InvocationID", ""),
            main,
            members,
            rows["ActiveState"],
            rows["SubState"],
            parse_job(rows["Job"], jobs, name),
            int(rows.get("ExecMainStartTimestampMonotonic") or 0) / 1e6,
            rows.get("Result", "success"),
            int(rows.get("ExecMainStatus") or 0),
            int(rows.get("NRestarts") or 0),
            enabled,
        )
        if not hasattr(self, "_exit_pids"):
            self._exit_pids = {}
        self._exit_pids[name] = int(rows.get("ExecMainPID") or 0)
        return unit, {
            "definition_sha256": unit.definition_sha256,
            "environment_sha256": env_sha,
            "enabled": enabled,
            "job": unit.job,
            "stage": stage,
        }

    def _gpus(
        self, units: Mapping[str, core.Unit], deadline: float
    ) -> tuple[tuple[str, ...], str, tuple[core.GPUOwner, ...]]:
        text = self.io.command(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            deadline,
        )
        hardware = [[v.strip() for v in row] for row in csv.reader(io.StringIO(text))]
        require(
            len(hardware) == 8 and all(len(row) == 6 for row in hardware),
            "linux-gpu-inventory",
        )
        require(len({row[1] for row in hardware}) == 8, "linux-gpu-duplicate")
        hardware.sort(key=lambda r: int(r[0]))
        index = self.manifest["probe_gpu_index"]
        require(
            type(index) is int
            and 0 <= index < len(hardware)
            and hardware[index][1] == self.plan["probe_gpu_uuid"],
            "linux-probe-device-map",
        )
        text = self.io.command(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            deadline,
        )
        owners = []
        for row in csv.reader(io.StringIO(text)):
            require(self.io.now() < deadline, "linux-nvml-owner-deadline")
            if not row:
                continue
            require(len(row) == 2 and row[1].strip().isdecimal(), "linux-nvml-owner")
            uuid, pid = row[0].strip(), int(row[1].strip())
            p = self.io.process(pid, deadline)
            if p is None:
                raise ObservationChanged("linux-nvml-stale-owner")
            assert p is not None
            process = core.Process(pid, p["start_ticks"])
            matches = [u for u in units.values() if process in u.members]
            if not matches and any(
                p["cgroup"] == "0::" + u.cgroup
                or p["cgroup"].startswith("0::" + u.cgroup + "/")
                for u in units.values()
            ):
                raise ObservationChanged("linux-known-cgroup-growth")
            require(len(matches) == 1, "linux-foreign-gpu-owner")
            u = matches[0]
            owners.append(
                core.GPUOwner(uuid, process, u.name, u.invocation_id, u.cgroup)
            )
        return tuple(row[1] for row in hardware), core.digest(hardware), tuple(owners)


class GuardianRetired(BaseException):
    """Normal guardian exit; a separate retirement-only reader observes death."""


def arena_prefix_preserved(
    previous: Mapping[str, Any], root: Path, io_: LinuxIO, deadline: float
) -> bool:
    """Preserve recorded games/actions/allocation history while allowing progress."""

    def rows_by_key(rows, fields):
        result = {tuple(row[k] for k in fields): row for row in rows}
        require(len(result) == len(rows), "linux-arena-duplicate-key")
        return result

    for relative, prior in previous.items():
        path = root / relative
        require(
            path.is_relative_to(root / "arena") and ".." not in Path(relative).parts,
            "linux-arena-path",
        )
        current = io_.json(path, deadline, 16 * 2**20)
        if current == prior:
            continue
        for key in (
            "run_id",
            "generation_family",
            "candidate",
            "baseline",
            "candidate_identity",
            "baseline_identity",
            "evaluation_contract",
        ):
            if key in prior:
                require(current.get(key) == prior[key], "linux-arena-contract-changed")
        old, new = prior.get("arena_state", prior), current.get("arena_state", current)
        for key in (
            "config",
            "seed",
            "schema_version",
            "candidate_model_identity",
            "baseline_model_identity",
        ):
            if key in old:
                require(new.get(key) == old[key], "linux-arena-state-contract")
        compared = False
        for key, fields in (
            ("games", ("ring", "variant", "pair", "candidate_player")),
            ("pairs", ("ring", "variant", "pair")),
        ):
            if key in old:
                indexed = rows_by_key(new.get(key, []), fields)
                require(
                    all(
                        indexed.get(tuple(row[k] for k in fields)) == row
                        for row in old[key]
                    ),
                    "linux-arena-completed-prefix",
                )
                compared = True
        if "game_states" in old:
            fields = ("ring", "variant", "pair", "candidate_player")
            indexed = rows_by_key(new.get("game_states", []), fields)
            for row in old["game_states"]:
                later = indexed.get(tuple(row[k] for k in fields))
                require(
                    later is not None
                    and all(
                        later.get(k) == v
                        for k, v in row.items()
                        if k not in {"actions", "result"}
                    )
                    and later.get("actions", [])[: len(row["actions"])]
                    == row["actions"]
                    and (
                        row.get("result") is None
                        or later.get("result") == row["result"]
                    ),
                    "linux-arena-action-prefix",
                )
            compared = True
        if "plan_history" in old:
            require(
                new.get("plan_history", [])[: len(old["plan_history"])]
                == old["plan_history"],
                "linux-arena-allocation-prefix",
            )
            compared = True
        require(compared, "linux-arena-unknown-mutation")
    return True


def _file_pin(path: Path) -> dict[str, Any]:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            sha.update(block)
    return {"path": str(path), "sha256": sha.hexdigest(), "bytes": path.stat().st_size}


def _boundary_worker(host: LinuxHost) -> dict[str, Any]:
    from scripts.graceful_training_deploy import validate_boundary
    from scripts.migrate_continuous_profile import _read_replay_boundary

    raw = validate_boundary(
        host.root, strict=True, expected_workers=host.plan["expected_workers"]
    )
    pointer = raw["checkpoint"]
    identity = host.plan["run_identity"]
    ledger = _read_replay_boundary(host.root, **identity)
    preserved, files, arena = {}, {}, {}
    require(
        set(host.manifest["boundary_paths"]) == core.PRESERVED,
        "linux-boundary-inventory",
    )
    for category, relatives in host.manifest["boundary_paths"].items():
        group = {}
        for relative in relatives:
            path = host._run_path(relative)
            paths = sorted(path.rglob("*.json")) if path.is_dir() else [path]
            require(len(paths) <= 2048, "linux-boundary-file-count")
            for file in paths:
                require(
                    file.resolve() == file and file.stat().st_size <= 16 * 2**20,
                    "linux-boundary-file-size",
                )
                data = file.read_bytes()
                name = str(file.relative_to(host.root))
                pin = {"path": str(file), "sha256": checksum(data), "bytes": len(data)}
                files[name] = pin
                group[name] = pin["sha256"]
                if category == "arena":
                    arena[name] = json.loads(data)
        require(group, "linux-empty-boundary-group")
        preserved[category] = core.digest(group)
    checkpoint = host.root / "learner" / pointer["checkpoint"]
    return {
        "clean_stop": True,
        "preserved": preserved,
        "run_identity": identity,
        "continuation_started_ns": host.plan["continuation_started_ns"],
        "step": pointer["step"],
        "examples_consumed": pointer["examples_consumed"],
        "replay_committed_samples": ledger.committed_samples,
        "replay_counter_updated_ns": ledger.updated_ns,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": pointer["checkpoint_sha256"],
            "bytes": pointer["checkpoint_bytes"],
        },
        "recovery_pointer": _file_pin(host.root / "learner/recovery.json"),
        "control_files": files,
        "arena_evidence": arena,
    }


def _champions_worker(host: LinuxHost) -> dict[str, str]:
    from deltreltrain.champion_migration import _validate_recorded_evaluation

    verified = {}
    for identity, pin in host.manifest["verified_champions"].items():
        host.io.pin(pin, host.io.now() + 10)
        verified[identity] = pin["sha256"]
    pointer_path = host.root / "learner/champion.json"
    pointer = host.io.json(pointer_path, host.io.now() + 5)
    manifest_path = (pointer_path.parent / pointer["manifest"]).resolve()
    require(
        manifest_path.parent == host.root / "learner/manifests",
        "linux-champion-manifest-path",
    )
    data = host.io.read(manifest_path, 2**20, host.io.now() + 5)
    require(
        len(data) == pointer["manifest_bytes"]
        and checksum(data) == pointer["manifest_sha256"],
        "linux-champion-manifest-pin",
    )
    model = json.loads(data)
    require(
        pointer.get("role") == "champion"
        and model.get("weights") == "ema"
        and model.get("model_identity")
        == "sha256-" + model.get("checkpoint_sha256", "")
        and core.sha(model.get("checkpoint_sha256"))
        and all(
            model.get(k) == pointer.get(k)
            for k in ("model_identity", "model_step", "run_id", "generation_family")
        )
        and model["run_id"] == host.plan["run_identity"]["run_id"]
        and model["generation_family"] == host.plan["run_identity"]["generation_family"]
        and model.get("rules_hash") == host.plan["probe_runtime"]["rules_hash"]
        and model.get("feature_schema_hash")
        == f"{host.plan['probe_runtime']['feature_schema_hash']:016x}",
        "linux-champion-lineage",
    )
    identity = model["model_identity"]
    if identity in verified:
        return verified
    path = (pointer_path.parent / pointer["promotion_result"]).resolve()
    require(path.parent == host.root / "arena", "linux-champion-proof-path")
    visiting: set[str] = set()

    def admit(teacher: str, result_path: Path, depth: int) -> None:
        require(depth <= 8 and teacher not in visiting, "linux-champion-chain-bound")
        if teacher in verified:
            return
        visiting.add(teacher)
        result = host.io.json(result_path, host.io.now() + 5, 16 * 2**20)
        require(
            result.get("candidate") == teacher
            and re.fullmatch(r"sha256-[0-9a-f]{64}", str(result.get("baseline")))
            and result.get("promotion", {}).get("decision") == "promote"
            and result.get("terminal") is True
            and result.get("conclusive") is True
            and result.get("evaluation_contract", {}).get("identity")
            == host.manifest["promotion_contract_identity"],
            "linux-champion-proof-decision",
        )
        baseline = result["baseline"]
        if baseline not in verified:
            matches = []
            paths = list((host.root / "arena").glob("*-" + baseline + "-vs-*.json"))
            require(len(paths) <= 64, "linux-champion-chain-candidate-bound")
            for candidate in paths:
                if candidate.name.endswith((".resume.json", ".allocation.json")):
                    continue
                value = host.io.json(candidate, host.io.now() + 5, 16 * 2**20)
                if (
                    value.get("candidate") == baseline
                    and value.get("promotion", {}).get("decision") == "promote"
                ):
                    matches.append(candidate)
            require(len(matches) == 1, "linux-champion-intermediate-proof")
            admit(baseline, matches[0], depth + 1)
        documents = {
            "result.json": result,
            "resume.json": host.io.json(
                result_path.with_name(result_path.stem + ".resume.json"),
                host.io.now() + 5,
                16 * 2**20,
            ),
            "allocation.json": host.io.json(
                result_path.with_name(result_path.stem + ".allocation.json"),
                host.io.now() + 5,
                16 * 2**20,
            ),
        }
        _validate_recorded_evaluation(documents)
        verified[teacher] = core.digest(documents)
        visiting.remove(teacher)

    admit(identity, path, 0)
    return verified


def _backup_worker(host: LinuxHost, request: dict[str, Any]) -> dict[str, Any]:
    from scripts.backup_active_profile import backup_active_profile
    from scripts.training_disaster_recovery import (
        _snapshot_envelope,
        _snapshot_commit_path,
    )

    state = core.Journal(host.state).read("state.json")
    require(
        request["guard_plan_sha256"] == core.digest(host.plan)
        and request["nonce"] == state["nonce"]
        and type(request.get("deadline_wall_ns")) is int
        and time.time_ns() < request["deadline_wall_ns"],
        "linux-backup-request",
    )
    finish_by = time.monotonic() + (request["deadline_wall_ns"] - time.time_ns()) / 1e9
    root = canonical(host.manifest["proof_archive_root"]) / host.plan["attempt_id"]
    require(not root.exists(), "linux-proof-archive-exists")
    root.mkdir(mode=0o700)
    (root / "objects").mkdir(mode=0o700)
    frozen = root / "frozen"
    frozen.mkdir(mode=0o700)
    candidates = {
        pin["sha256"]: Path(pin["path"]) for pin in host.manifest["proof_inputs"]
    }
    candidates.update(
        {
            pin["sha256"]: Path(pin["path"])
            for pin in host.manifest["implementation_pins"]
        }
    )
    candidates[state["boundary"]["checkpoint"]["sha256"]] = Path(
        state["boundary"]["checkpoint_retention"]["path"]
    )

    def freeze(file: Path) -> None:
        data = host.io.read(file, 16 * 2**20, host.io.now() + 5)
        sha = checksum(data)
        destination = frozen / sha
        if not destination.exists():
            host.io.atomic(destination, data, mode=0o444)
        candidates[sha] = destination

    for file in (
        host.manifest_path,
        Path(host.manifest["guard_plan"]["path"]),
        Path(host.manifest["qualification"]["path"]),
    ):
        freeze(file)
    for directory in (
        host.state,
        host.prepared_directory(),
        Path(host.plan["probe_output"]),
    ):
        if directory.exists():
            paths = sorted(directory.rglob("*"))
            require(len(paths) <= 4096, "linux-proof-metadata-count")
            for file in paths:
                if file.is_file() and file.suffix in {".json", ".jsonl", ".log"}:
                    require(
                        file.resolve() == file and file.stat().st_size <= 16 * 2**20,
                        "linux-proof-file-size",
                    )
                    freeze(file)
    # Core also names the canonical boundary digest, independently of the
    # pretty-printed containing journal. Preserve those exact canonical bytes.
    boundary_bytes = json.dumps(
        state["boundary"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    boundary_sha = checksum(boundary_bytes)
    host.io.atomic(root / "boundary.json", boundary_bytes, mode=0o444)
    candidates[boundary_sha] = root / "boundary.json"
    required = set(request["required_proof_sha256"])
    require(required <= set(candidates), "linux-proof-closure-missing-input")
    objects = []
    for expected in sorted(candidates):
        require(time.monotonic() < finish_by, "linux-backup-original-deadline")
        source = candidates[expected]
        destination = root / "objects" / expected
        require(source.resolve() == source, "linux-proof-source-symlink")
        sha = hashlib.sha256()
        size = 0
        with source.open("rb") as reader, destination.open("xb") as writer:
            for block in iter(lambda: reader.read(2**20), b""):
                require(time.monotonic() < finish_by, "linux-backup-original-deadline")
                sha.update(block)
                size += len(block)
                writer.write(block)
            writer.flush()
            os.fchmod(writer.fileno(), 0o444)
            os.fsync(writer.fileno())
        require(sha.hexdigest() == expected, "linux-proof-source-changed")
        objects.append({"path": str(destination), "sha256": expected, "bytes": size})
    proof = {
        "format": FORMAT + "-proof-closure",
        "core_plan_sha256": core.digest(host.plan),
        "manifest_sha256": host.manifest_sha256,
        "nonce": state["nonce"],
        "objects": objects,
        "proof_closure_sha256": request["proof_closure_sha256"],
        "required_core_sha256": sorted(required),
    }
    host.io.atomic(root / "manifest.json", encoded(proof), mode=0o444)
    os.chmod(root / "objects", 0o555)
    os.chmod(frozen, 0o555)
    os.chmod(root, 0o555)
    for path in (root / "objects", frozen, root, root.parent):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    snapshot = backup_active_profile(
        host.root,
        Path(host.manifest["backup_root"]),
        expected_backup_mount=Path(host.manifest["backup_mount"]),
    )
    payload, _, _, catalog_sha = _snapshot_envelope(
        snapshot, Path(host.manifest["backup_root"])
    )
    marker = _snapshot_commit_path(snapshot)
    commit = json.loads(marker.read_bytes())
    require(
        commit["sha256"] == catalog_sha
        and payload["run_id"] == host.plan["run_identity"]["run_id"],
        "linux-backup-commit",
    )
    result = {
        "manifest_sha256": host.manifest_sha256,
        "invocation_id": os.environ.get("INVOCATION_ID"),
        "request_sha256": checksum(encoded(request)),
        "metadata_pins": [
            _file_pin(root / "manifest.json"),
            _file_pin(snapshot),
            _file_pin(marker),
        ],
        "retained_objects": objects,
        "core_result": {
            "status": "committed",
            "guard_plan_sha256": core.digest(host.plan),
            "role": request["support_stage"] == "after" and "r4" or "r3",
            "closure_verified": True,
            "catalog_sha256": catalog_sha,
            "proof_closure_sha256": request["proof_closure_sha256"],
            "verified_artifact_sha256": sorted(required),
        },
    }
    require(time.monotonic() < finish_by, "linux-backup-original-deadline")
    host.io.atomic(host.state / "linux-backup-result.json", encoded(result))
    return result


def worker(
    host: LinuxHost, operation: str, request: dict[str, Any] | None
) -> dict[str, Any]:
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "linux-control-cuda-must-be-hidden",
    )
    from scripts import activate_strength_freshness as activation

    if operation == "authority":
        return activation.inspect_authority(host.root)
    if operation == "boundary":
        return _boundary_worker(host)
    if operation == "check-boundary":
        require(
            request is not None
            and _boundary_worker(host)
            == {k: v for k, v in request.items() if k != "checkpoint_retention"},
            "linux-boundary-changed",
        )
        assert request is not None
        retained = request["checkpoint_retention"]
        require(
            retained["sha256"] == request["checkpoint"]["sha256"]
            and retained["bytes"] == request["checkpoint"]["bytes"]
            and os.path.samefile(retained["path"], request["checkpoint"]["path"]),
            "linux-boundary-retention-changed",
        )
        return {"unchanged": True}
    if operation == "champions":
        return _champions_worker(host)
    require(host.execute, "linux-worker-execution-unqualified")
    lease = host._lease(host.io.now() + 5)
    state = core.Journal(host.state).read("state.json")
    require(
        lease is not None
        and lease.get("plan_sha256") == core.digest(host.plan)
        and lease.get("nonce") == state["nonce"]
        and lease.get("boot_id") == host.io.boot_id(),
        "linux-worker-no-guardian-lease",
    )
    assert lease is not None
    if operation in {"apply-r4", "repair-r4"}:
        require(
            lease["owner"]["pid"] == os.getppid(), "linux-worker-not-guardian-child"
        )
        require(
            request is not None
            and request["nonce"] == state["nonce"]
            and request["guard_plan_sha256"] == core.digest(host.plan),
            "linux-worker-request-authority",
        )
    elif operation == "backup":
        unit, _ = host._unit(
            host.manifest["backup_worker_unit"],
            host._jobs(host.io.now() + 5),
            host.io.now() + 5,
        )
        require(
            unit.main is not None
            and unit.main.pid == os.getpid()
            and unit.invocation_id == os.environ.get("INVOCATION_ID"),
            "linux-backup-worker-owner",
        )
        require(
            request is not None and request["nonce"] == state["nonce"],
            "linux-backup-start-request",
        )
        assert unit.main is not None and request is not None
        started = {
            "format": FORMAT + "-backup-started",
            "schema_version": 1,
            "manifest_sha256": host.manifest_sha256,
            "core_plan_sha256": core.digest(host.plan),
            "nonce": state["nonce"],
            "boot_id": host.io.boot_id(),
            "unit_name": unit.name,
            "main": asdict(unit.main),
            "invocation_id": unit.invocation_id,
            "entered_monotonic": unit.entered_monotonic,
            "request_sha256": checksum(encoded(request)),
        }
        host.io.atomic(host.state / "linux-backup-started.json", encoded(started))
    if operation == "apply-r4":
        require(request is not None, "linux-apply-request")
        assert request is not None
        return activation.apply(
            host.root,
            cuda_receipt=Path(request["receipt"]),
            cuda_sha256=request["receipt_sha256"],
        )
    if operation == "repair-r4":
        return activation.repair(host.root)
    if operation == "backup":
        require(request is not None, "linux-backup-request-missing")
        assert request is not None
        return _backup_worker(host, request)
    raise core.Refusal("linux-worker-operation")


def authorized_host(path: Path, state: Path, io_: LinuxIO | None = None) -> LinuxHost:
    io_ = io_ or LinuxIO()
    journal = core.Journal(state.with_name(state.name + ".prepared"))
    journal._check()
    auth = journal.read(AUTHORIZATION)
    require(
        auth.get("format") == FORMAT + "-authorization"
        and auth.get("manifest_path") == str(path),
        "linux-authorization-location",
    )
    host = LinuxHost(path, auth["manifest_sha256"], execute=True, io_=io_)
    require(
        host.state == state
        and host.prepared_directory() == journal.directory
        and auth["core_plan_sha256"] == core.digest(host.plan)
        and auth["adapter_sha256"] == host.manifest["adapter_sha256"],
        "linux-prepared-authorization-drift",
    )
    return host


def admit_start(host: LinuxHost, role: str) -> dict[str, Any]:
    """Closed ExecStartPre gate; no service start/stop or GPU initialization."""
    require(host.execute and role in {"r3", "r4", "probe"}, "linux-start-gate-role")
    path = host.state / "state.json"
    if not path.exists():
        require(
            role == "r3" and host._authority(host.io.now() + 5).phase == "r3",
            "linux-unstarted-lineage",
        )
        return {"allowed": True, "role": role, "scope": "original-before-attempt"}
    state = core.Journal(host.state).read("state.json")
    require(state["plan_sha256"] == core.digest(host.plan), "linux-start-gate-plan")
    if state.get("runtime_authority_retired"):
        outcome = state.get("outcome", "")
        selected = (
            "r4"
            if outcome
            in {
                "committed-r4",
                "productive-r4-proof-incomplete",
                "owned-r4-pending-telemetry",
            }
            else "r3"
            if outcome
            in {
                "restored-r3",
                "productive-r3-proof-incomplete",
                "owned-r3-pending-telemetry",
                "aborted-before-stop-r3-unchanged",
            }
            else None
        )
        require(
            role == selected and host._authority(host.io.now() + 5).phase == selected,
            "linux-retired-lineage-start",
        )
        return {"allowed": True, "role": role, "scope": "sealed-lineage"}
    lease = host._lease(host.io.now() + 5)
    require(
        lease is not None
        and lease["nonce"] == state["nonce"]
        and lease["plan_sha256"] == core.digest(host.plan)
        and lease["boot_id"] == host.io.boot_id(),
        "linux-start-gate-lease",
    )
    require(
        role in state["starts"]
        and state["last_action"]["kind"] == "start-" + role
        and state["active_boot_id"] == host.io.boot_id()
        and host.io.now() < min(state["deadline"], state["last_action"]["deadline"]),
        "linux-no-current-start-intent",
    )
    invocation = os.environ.get("INVOCATION_ID", "")
    require(
        re.fullmatch(r"[0-9a-f]{32}", invocation)
        and invocation != state["starts"][role]["prior_invocation"],
        "linux-start-gate-invocation",
    )
    permit = {
        "core_plan_sha256": core.digest(host.plan),
        "nonce": state["nonce"],
        "role": role,
        "invocation_id": invocation,
        "boot_id": host.io.boot_id(),
        "issued": host.io.now(),
    }
    host.io.atomic(
        host.state / f"linux-start-permit-{role}-{invocation}.json", encoded(permit)
    )
    return {"allowed": True, **permit}


def launch_probe(host: LinuxHost) -> None:
    """Exec the one pinned probe; static units never embed a dynamic digest."""
    require(host.execute, "linux-probe-launch-unqualified")
    state = core.Journal(host.state).read("state.json")
    invocation = os.environ.get("INVOCATION_ID", "")
    permit = host.io.json(
        host.state / f"linux-start-permit-probe-{invocation}.json", host.io.now() + 5
    )
    require(
        permit["nonce"] == state["nonce"]
        and permit["core_plan_sha256"] == core.digest(host.plan)
        and state["phase"] == "probing"
        and state["active_boot_id"] == host.io.boot_id(),
        "linux-probe-permit",
    )
    lease = host._lease(host.io.now() + 5)
    require(
        lease is not None and lease["nonce"] == state["nonce"],
        "linux-probe-launch-lease",
    )
    challenge = host.state / "challenge.json"
    raw = host.io.read(challenge, 2**20, host.io.now() + 5)
    require(
        checksum(raw) == state["challenge_file_sha256"], "linux-probe-launch-challenge"
    )
    pin = host.manifest["probe_program"]
    host.io.pin(pin, host.io.now() + 10)
    require(
        pin["sha256"] == host.plan["probe_source_sha256"], "linux-probe-program-pin"
    )
    index = host.manifest["probe_gpu_index"]
    require(type(index) is int and 0 <= index < 8, "linux-probe-device-index")
    runtime = host.plan["runtime_root"]
    env = {
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C",
        "HOME": host.manifest["probe_home"],
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": runtime,
        "CUDA_VISIBLE_DEVICES": str(index),
        "INVOCATION_ID": invocation,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TORCHINDUCTOR_CACHE_DIR": str(host.state / "probe-compile-cache"),
    }
    os.execve(
        host.plan["probe_runtime"]["python"]["path"],
        [
            host.plan["probe_runtime"]["python"]["path"],
            "-s",
            pin["path"],
            "--challenge",
            str(challenge),
            "--challenge-sha256",
            state["challenge_file_sha256"],
        ],
        env,
    )


def finalize_retirement(host: LinuxHost) -> dict[str, Any]:
    """Never dispatch generic recover from a finalizer, including after reboot."""
    journal = core.Journal(host.state)
    state = journal.read("state.json")
    require(
        state.get("runtime_authority_retired") is True, "linux-finalizer-not-sealed"
    )
    unit, _ = host._unit(
        host.plan["units"]["guard"]["name"],
        host._jobs(host.io.now() + 5),
        host.io.now() + 5,
    )
    require(
        unit.dead and not unit.enabled and host._lease(host.io.now() + 5) is None,
        "linux-finalizer-guard-not-retired",
    )
    return core.Controller(host.plan, core.digest(host.plan), journal, host).tick()


def guardian(host: LinuxHost) -> dict[str, Any]:
    from scripts.strength_freshness_units import recover_transition

    require(host.execute, "linux-guardian-unqualified")
    journal = core.Journal(host.state)
    controller = core.Controller(host.plan, core.digest(host.plan), journal, host)
    with host.hold_lease():
        state = journal.read("state.json")
        now = host.clock()
        deadline = now.monotonic + max(
            0, (state["deadline_wall_ns"] - now.wall_ns) / 1e9
        )
        if state["active_boot_id"] == now.boot_id:
            deadline = min(deadline, state["deadline"])
        recover_transition(host, deadline, pre_admission=True)
        old = state["owners"].get("guard")
        current = host.io.process(os.getpid())
        require(current is not None, "linux-guardian-current-process")
        assert current is not None
        if state["active_boot_id"] != host.io.boot_id() and not state.get(
            "runtime_authority_retired"
        ):
            controller.admit_guard_boot()
        elif (
            old is not None
            and old["boot_id"] == host.io.boot_id()
            and (
                old["main"]
                != {"pid": current["pid"], "start_ticks": current["start_ticks"]}
                or old["invocation_id"] != os.environ.get("INVOCATION_ID")
            )
            and not state.get("runtime_authority_retired")
        ):
            controller.admit_guard_reentry()
        recovered = recover_transition(host, deadline)
        if recovered is not None:
            host.resume_support_tail(recovered["action"])
        try:
            while True:
                state = controller.tick()
                if state["finished"]:
                    return state
                # Every tick is bounded; no long sleep holds the journal lock.
                host.io.sleep(0.25)
        except GuardianRetired:
            return journal.read("state.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "prepare", "run", "worker"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--sha256", required=True)
        if name == "worker":
            p.add_argument(
                "--operation",
                choices=(
                    "authority",
                    "boundary",
                    "check-boundary",
                    "champions",
                    "apply-r4",
                    "repair-r4",
                    "backup",
                ),
                required=True,
            )
            p.add_argument("--request", type=Path)
            p.add_argument("--request-sha256")
    for name in ("guard", "admit-start", "finalize", "probe", "backup"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--state", type=Path, required=True)
        if name == "admit-start":
            p.add_argument("--role", choices=("r3", "r4", "probe"), required=True)
    args = parser.parse_args()
    require(
        sys.platform == "linux" and os.geteuid() == 0, "linux-root-control-required"
    )
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "linux-control-cuda-must-be-hidden",
    )
    if args.command in {"guard", "admit-start", "finalize", "probe", "backup"}:
        host = authorized_host(args.manifest, args.state)
    else:
        host = LinuxHost(args.manifest, args.sha256, execute=args.command != "validate")
    if args.command == "validate":
        output = {
            "status": "pinned-structure-only",
            "execution_qualified": False,
            "core_plan_sha256": core.digest(host.plan),
        }
    elif args.command == "prepare":
        host.verify_inputs(host.plan, host.io.now() + 60)
        output = {"status": "prepared-authorization-only", "started": False}
    elif args.command == "run":
        output = core.Controller(
            host.plan, core.digest(host.plan), core.Journal(host.state), host
        ).begin()
    elif args.command == "guard":
        output = guardian(host)
    elif args.command == "admit-start":
        output = admit_start(host, args.role)
    elif args.command == "finalize":
        output = finalize_retirement(host)
    elif args.command == "probe":
        launch_probe(host)
        raise AssertionError("execve returned")
    elif args.command == "backup":
        output = worker(
            host,
            "backup",
            host.io.json(host.state / "linux-backup-request.json", host.io.now() + 5),
        )
    else:
        request = None
        if args.request:
            require(
                args.request.parent == host.state and args.request_sha256 is not None,
                "linux-worker-request-path",
            )
            raw = host.io.read(args.request, 2**20, host.io.now() + 5)
            require(checksum(raw) == args.request_sha256, "linux-worker-request-hash")
            request = json.loads(raw)
        output = worker(host, args.operation, request)
    print(json.dumps(output, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
