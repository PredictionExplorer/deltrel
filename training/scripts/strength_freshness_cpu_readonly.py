"""Closed stdlib-only metadata IO for preservation collection.

No training-package import, model/replay access, target writes, or service
mutation. Values are private in-memory inputs to the caller's normalizer;
serialize ``audit`` only. An outer bounded process is still required for kernel
or filesystem stalls that userspace deadlines cannot interrupt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import time
from types import MappingProxyType
from typing import Any, Callable, Generic, Mapping, TypeVar, cast

T = TypeVar("T")
MAX_CAPTURE_BYTES = 32 * 2**20
MAX_FILE_BYTES = 2**20
MAX_TAIL_BYTES = 4 * 2**20
MAX_PROC_BYTES = 65536
MAX_MAP_BYTES = 2 * 2**20
MAX_PIDS = 256
MAX_APPEND_PREFIX = 65536
COMMON_PROPERTIES = (
    "Id",
    "Names",
    "LoadState",
    "FragmentPath",
    "SourcePath",
    "DropInPaths",
    "UnitFileState",
    "NeedDaemonReload",
    "ActiveState",
    "SubState",
    "InvocationID",
    "Job",
    "InactiveExitTimestampMonotonic",
    "ActiveEnterTimestampMonotonic",
    "ActiveExitTimestampMonotonic",
    "InactiveEnterTimestampMonotonic",
    "StateChangeTimestampMonotonic",
    "After",
    "Before",
    "Wants",
    "Requires",
)
SERVICE_PROPERTIES = COMMON_PROPERTIES + (
    "Type",
    "User",
    "Group",
    "WorkingDirectory",
    "MainPID",
    "ExecMainPID",
    "ControlGroup",
    "NRestarts",
    "ExecStart",
    "ExecStartPre",
    "ExecStop",
    "ExecStopPost",
    "ExecMainStartTimestampMonotonic",
    "ExecMainExitTimestampMonotonic",
    "ExecMainCode",
    "ExecMainStatus",
    "Result",
    "Environment",
    "EnvironmentFiles",
    "PassEnvironment",
    "UnsetEnvironment",
    "Restart",
    "KillMode",
    "KillSignal",
    "SendSIGKILL",
    "TimeoutStartUSec",
    "TimeoutStopUSec",
    "RuntimeMaxUSec",
)
TIMER_PROPERTIES = COMMON_PROPERTIES + (
    "Result",
    "Unit",
    "NextElapseUSecMonotonic",
    "NextElapseUSecRealtime",
    "LastTriggerUSecMonotonic",
    "AccuracyUSec",
    "RandomizedDelayUSec",
    "TimersMonotonic",
    "TimersCalendar",
    "Persistent",
)


class ReadRefusal(RuntimeError):
    def __init__(self, reason: str, audit: Mapping[str, Any] | None = None):
        super().__init__(reason)
        self.audit = dict(audit or {})


def require(value: object, reason: str) -> None:
    if not value:
        raise ReadRefusal(reason)


def encoded(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def absolute(value: str) -> str:
    path = Path(value)
    require(
        isinstance(value, str)
        and path.is_absolute()
        and str(path) == value
        and ".." not in path.parts
        and "\0" not in value,
        "canonical-path-required",
    )
    return value


@dataclass(frozen=True)
class FileKey:
    path: str
    maximum_bytes: int = MAX_FILE_BYTES


@dataclass(frozen=True)
class CachedFile:
    literal: str
    resolved: str


@dataclass(frozen=True)
class ReadScope:
    units: Mapping[str, str]
    targets: tuple[str, ...]
    files: Mapping[str, FileKey]
    tails: Mapping[str, FileKey]
    cached: Mapping[str, CachedFile]
    property_sets: Mapping[str, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        require(0 < len(self.units) <= 16, "unit-inventory-limit")
        for name, kind in self.units.items():
            require(
                kind in {"service", "timer"}
                and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]{0,255}\." + kind, name),
                "registered-unit-shape",
            )
        require(
            len(self.targets) <= 16
            and len(set(self.targets)) == len(self.targets)
            and all(
                re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]{0,255}\.target", n)
                for n in self.targets
            ),
            "registered-targets",
        )
        require(
            len(self.files) <= 128
            and len(self.tails) <= 4
            and len(self.cached) <= 1024,
            "file-inventory-limit",
        )
        require(not set(self.files).intersection(self.tails), "duplicate-file-key")
        for collection, cap in (
            (self.files, MAX_FILE_BYTES),
            (self.tails, MAX_TAIL_BYTES),
        ):
            for key, entry in collection.items():
                require(re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", key), "file-key")
                absolute(entry.path)
                require(
                    type(entry.maximum_bytes) is int and 0 < entry.maximum_bytes <= cap,
                    "file-read-limit",
                )
                require(
                    Path(entry.path).suffix.lower()
                    not in {
                        ".pt",
                        ".pth",
                        ".npz",
                        ".sqlite",
                        ".sqlite3",
                        ".db",
                        ".so",
                        ".whl",
                    },
                    "model-replay-binary-not-readable",
                )
        for entry in self.cached.values():
            absolute(entry.literal)
            absolute(entry.resolved)
        selected = dict(
            self.property_sets
            or {"service": SERVICE_PROPERTIES, "timer": TIMER_PROPERTIES}
        )
        require(set(selected) == {"service", "timer"}, "property-set-kinds")
        for kind, allowed in (
            ("service", SERVICE_PROPERTIES),
            ("timer", TIMER_PROPERTIES),
        ):
            values = tuple(selected[kind])
            require(
                values
                and len(values) == len(set(values))
                and set(values) <= set(allowed),
                "property-not-allowlisted",
            )
            selected[kind] = values
        object.__setattr__(self, "property_sets", MappingProxyType(selected))
        for key in ("units", "files", "tails", "cached"):
            object.__setattr__(self, key, MappingProxyType(dict(getattr(self, key))))
        object.__setattr__(self, "targets", tuple(self.targets))


@dataclass(frozen=True)
class Observation(Generic[T]):
    value: T = field(repr=False)
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class ProcessAdmission:
    pid: int
    start_ticks: int
    cgroup: str


def _stat(value: os.stat_result) -> dict[str, int]:
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "mode": stat.S_IMODE(value.st_mode),
        "uid": value.st_uid,
        "gid": value.st_gid,
        "bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
        "links": value.st_nlink,
    }


def _append_identity(path: str, value: Mapping[str, int]) -> dict[str, Any]:
    return {
        "path": path,
        **{name: value[name] for name in ("device", "inode", "uid", "gid", "mode")},
    }


def _readonly_vector(argv: tuple[str, ...]) -> bool:
    if argv in (
        ("systemctl", "get-default"),
        (
            "systemctl",
            "list-jobs",
            "--all",
            "--no-pager",
            "--no-legend",
            "--plain",
            "--full",
        ),
        ("nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"),
        (
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits",
        ),
    ):
        return True
    if (
        len(argv) != 6
        or argv[:2] != ("systemctl", "show")
        or argv[3:5] != ("--all", "--no-pager")
    ):
        return False
    if (
        re.fullmatch(
            r"[A-Za-z0-9_][A-Za-z0-9_.@-]{0,255}\.(service|timer|target)", argv[2]
        )
        is None
    ):
        return False
    if not argv[5].startswith("--property="):
        return False
    fields = argv[5].removeprefix("--property=").split(",")
    return (
        bool(fields)
        and len(fields) == len(set(fields))
        and set(fields) <= set(SERVICE_PROPERTIES + TIMER_PROPERTIES)
    )


class System:
    """Actual read-only syscall boundary; tests supply an explicit separate fake."""

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def wall_ns(self) -> int:
        return time.time_ns()

    def boottime_ns(self) -> int:
        clock = getattr(time, "CLOCK_BOOTTIME", None)
        require(clock is not None, "boottime-clock-unavailable")
        assert clock is not None
        return time.clock_gettime_ns(clock)

    def hertz(self) -> int:
        return os.sysconf("SC_CLK_TCK")

    def _time(self, deadline: float) -> None:
        require(self.monotonic_ns() / 1e9 < deadline, "absolute-read-deadline")

    def read_file(
        self, path: str, maximum: int, deadline: float, *, tail: bool = False
    ):
        self._time(deadline)
        p = Path(path)
        require(str(p.resolve()) == path, "read-symlink-refused")
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode), "regular-file-required")
            if tail:
                offset = max(0, before.st_size - maximum)
                data = os.pread(fd, min(maximum, before.st_size), offset)
                require(
                    len(data) == min(maximum, before.st_size),
                    "tail-truncated-during-read",
                )
            else:
                offset = 0
                require(before.st_size <= maximum, "file-size-limit")
                chunks = []
                count = 0
                while True:
                    self._time(deadline)
                    block = os.read(fd, min(65536, maximum + 1 - count))
                    if not block:
                        break
                    chunks.append(block)
                    count += len(block)
                    require(count <= maximum, "file-output-limit")
                data = b"".join(chunks)
            after = os.fstat(fd)
            named = os.stat(path, follow_symlinks=False)
            require(
                (before.st_dev, before.st_ino)
                == (after.st_dev, after.st_ino)
                == (named.st_dev, named.st_ino),
                "file-identity-raced",
            )
            if tail:
                require(after.st_size >= before.st_size, "tail-truncated-during-read")
            else:
                require(
                    (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    == (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                    "file-bytes-raced",
                )
            self._time(deadline)
            return data, {
                "stat_before": _stat(before),
                "stat_after": _stat(after),
                "offset": offset,
                "end_offset": offset + len(data),
            }
        finally:
            os.close(fd)

    def append_file(
        self,
        path: str,
        maximum: int,
        deadline: float,
        *,
        span: tuple[int, int] | None,
        expected: Mapping[str, Any] | None,
        charge: Callable[[int], None],
        allowance: int,
    ) -> tuple[bytes, dict[str, Any]]:
        """Read one fixed span, or a bounded prefix ending at the observed EOF.

        This detects observed identity, size and byte drift; it cannot establish
        historical append-only behavior between observations. That premise and
        comparison of retained spans belong to the separate proof consumer.
        ``charge`` accounts each actual read even if a later check refuses.
        """
        self._time(deadline)
        require(str(Path(path).resolve()) == path, "append-path-alias")
        named_before = os.stat(path, follow_symlinks=False)
        require(stat.S_ISREG(named_before.st_mode), "append-regular-file-required")
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode), "append-regular-file-required")
            identity = _append_identity(path, _stat(before))
            require(
                identity == _append_identity(path, _stat(named_before))
                and (expected is None or identity == expected),
                "append-file-identity",
            )
            start, end = (
                (max(0, before.st_size - maximum), before.st_size)
                if span is None
                else span
            )
            require(
                0 <= start <= end <= before.st_size and end - start <= maximum,
                "append-range-bounds",
            )
            require(end - start <= allowance, "capture-byte-budget")
            chunks: list[bytes] = []
            offset = start
            while offset < end:
                self._time(deadline)
                size = min(65536, end - offset)
                block = os.pread(fd, size, offset)
                charge(len(block))
                require(0 < len(block) <= size, "append-short-read")
                chunks.append(block)
                offset += len(block)
            after = os.fstat(fd)
            named_after = os.stat(path, follow_symlinks=False)
            require(
                stat.S_ISREG(after.st_mode)
                and stat.S_ISREG(named_after.st_mode)
                and str(Path(path).resolve()) == path
                and identity == _append_identity(path, _stat(after))
                and identity == _append_identity(path, _stat(named_after)),
                "append-file-identity",
            )
            observations = (named_before, before, after, named_after)
            for earlier, later in zip(observations, observations[1:]):
                require(later.st_size >= earlier.st_size, "append-size-regression")
                require(
                    later.st_size > earlier.st_size
                    or (earlier.st_mtime_ns, earlier.st_ctime_ns)
                    == (later.st_mtime_ns, later.st_ctime_ns),
                    "append-observed-rewrite",
                )
            self._time(deadline)
            return b"".join(chunks), {
                "named_before": _stat(named_before),
                "stat_before": _stat(before),
                "stat_after": _stat(after),
                "named_after": _stat(named_after),
                "offset": start,
                "end_offset": end,
            }
        finally:
            os.close(fd)

    def stat_path(self, path: str, deadline: float, *, follow: bool = False):
        self._time(deadline)
        value = os.stat(path, follow_symlinks=follow)
        self._time(deadline)
        return _stat(value)

    def link(self, path: str, deadline: float):
        self._time(deadline)
        literal = os.readlink(path)
        resolved = str(Path(path).resolve(strict=False))
        self._time(deadline)
        return {"literal": literal, "resolved": resolved}

    def directory(self, path: str, deadline: float):
        self._time(deadline)
        require(str(Path(path).resolve()) == path, "directory-alias")
        result = []
        with os.scandir(path) as iterator:
            for row in iterator:
                self._time(deadline)
                require(len(result) < 8192, "directory-entry-limit")
                result.append(
                    {
                        "name": row.name,
                        "directory": row.is_dir(follow_symlinks=False),
                        "symlink": row.is_symlink(),
                    }
                )
        return result

    def proc(
        self,
        pid: int,
        deadline: float,
        *,
        details: bool = False,
        expected: ProcessAdmission | None = None,
        maps: bool = True,
    ):
        self._time(deadline)
        fd = os.open(
            f"/proc/{pid}", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        )

        def read(name, maximum):
            self._time(deadline)
            source = os.open(
                name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=fd
            )
            try:
                data = bytearray()
                while True:
                    self._time(deadline)
                    chunk = os.read(source, min(65536, maximum + 1 - len(data)))
                    if not chunk:
                        break
                    data.extend(chunk)
                    require(len(data) <= maximum, "proc-output-limit")
                return bytes(data)
            finally:
                os.close(source)

        try:
            value: dict[str, Any] = {
                "stat_before": read("stat", MAX_PROC_BYTES),
                "cgroup_before": read("cgroup", MAX_PROC_BYTES),
            }
            if details:
                identity = _process_identity(
                    pid, value["stat_before"], value["cgroup_before"]
                )
                require(
                    expected is not None
                    and identity["start_ticks"] == expected.start_ticks
                    and identity["cgroup"] == expected.cgroup,
                    "private-process-admission-drift",
                )
                value.update(
                    {
                        "exe": os.readlink("exe", dir_fd=fd),
                        "cwd": os.readlink("cwd", dir_fd=fd),
                        "cmdline": read("cmdline", MAX_PROC_BYTES),
                        "environ": read("environ", MAX_PROC_BYTES),
                    }
                )
                if maps:
                    value["maps"] = read("maps", MAX_MAP_BYTES)
            value.update(
                {
                    "stat_after": read("stat", MAX_PROC_BYTES),
                    "cgroup_after": read("cgroup", MAX_PROC_BYTES),
                }
            )
            self._time(deadline)
            return value
        finally:
            os.close(fd)

    def proc_credentials(
        self,
        admission: ProcessAdmission,
        deadline: float,
        *,
        charge: Callable[[int], None],
        allowance: int,
    ) -> dict[str, bytes]:
        """Only fixed credential metadata under the already admitted proc FD."""
        self._time(deadline)
        fd = os.open(
            f"/proc/{admission.pid}",
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        remaining = allowance

        def read(name: str) -> bytes:
            nonlocal remaining
            self._time(deadline)
            source = os.open(
                name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=fd,
            )
            try:
                require(stat.S_ISREG(os.fstat(source).st_mode), "credential-proc-file")
                data = bytearray()
                while True:
                    self._time(deadline)
                    require(remaining > 0, "capture-byte-budget")
                    chunk = os.read(
                        source, min(65536, MAX_PROC_BYTES + 1 - len(data), remaining)
                    )
                    charge(len(chunk))
                    remaining -= len(chunk)
                    if not chunk:
                        break
                    data.extend(chunk)
                    require(len(data) <= MAX_PROC_BYTES, "credential-proc-limit")
                return bytes(data)
            finally:
                os.close(source)

        try:
            value = {"stat_before": read("stat"), "cgroup_before": read("cgroup")}
            before = _process_identity(
                admission.pid, value["stat_before"], value["cgroup_before"]
            )
            require(
                (before["start_ticks"], before["cgroup"])
                == (admission.start_ticks, admission.cgroup),
                "credential-admission-drift",
            )
            value["status"] = read("status")
            value.update(stat_after=read("stat"), cgroup_after=read("cgroup"))
            after = _process_identity(
                admission.pid, value["stat_after"], value["cgroup_after"]
            )
            require(before == after, "credential-process-raced")
            self._time(deadline)
            return value
        finally:
            os.close(fd)

    def namespace(self, pid: str | int, deadline: float):
        self._time(deadline)
        fd = os.open(
            f"/proc/{pid}/ns",
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            result = {}
            for key in ("pid", "time"):
                self._time(deadline)
                literal = os.readlink(key, dir_fd=fd)
                metadata = _stat(os.stat(key, dir_fd=fd, follow_symlinks=True))
                result[key] = {"literal": literal, "stat": metadata}
            self._time(deadline)
            return result
        finally:
            os.close(fd)

    def command(self, argv: tuple[str, ...], deadline: float):
        """Private caller passes only closed constant query vectors.

        Byte caps apply while pipes are consumed; own process group is reaped
        on timeout/overflow. No arbitrary shell and no unbounded communicate().
        """
        require(_readonly_vector(argv), "command-vector-not-readonly")
        self._time(deadline)
        require(
            signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL, "command-child-handler"
        )
        end = min(deadline, self.monotonic_ns() / 1e9 + 5)
        require(end - self.monotonic_ns() / 1e9 > 0.25, "command-cleanup-reserve")
        child = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            start_new_session=True,
            env={
                "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
                "LANG": "C",
                "SYSTEMD_COLORS": "0",
            },
        )
        assert child.stdout is not None and child.stderr is not None
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        selector = None
        failed = None
        try:
            selector = selectors.DefaultSelector()
            for key, pipe in (("stdout", child.stdout), ("stderr", child.stderr)):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, key)
            while selector.get_map():
                remaining = end - self.monotonic_ns() / 1e9
                if remaining <= 0.2:
                    raise ReadRefusal("command-timeout")
                for key, _ in selector.select(min(0.05, remaining - 0.2)):
                    block = os.read(key.fd, 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    target = buffers[key.data]
                    target.extend(block)
                    require(
                        len(target)
                        <= (MAX_FILE_BYTES if key.data == "stdout" else MAX_PROC_BYTES),
                        "command-output-limit",
                    )
            require(
                callable(getattr(os, "waitid", None)),
                "unreaped-exit-observation-unavailable",
            )
            while True:
                require(self.monotonic_ns() / 1e9 < end - 0.2, "command-timeout")
                result = getattr(os, "waitid")(
                    os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
                )
                if result is not None:
                    break
                time.sleep(0.005)
        except BaseException as error:
            failed = error
        finally:
            if selector is not None:
                selector.close()
            child.stdout.close()
            child.stderr.close()
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=max(0.001, end - self.monotonic_ns() / 1e9))
            except subprocess.TimeoutExpired:
                raise ReadRefusal("command-cleanup-unproved") from failed
        # Read-only group observation after reap; never signal after PID reuse
        # becomes possible. A remaining group means cleanup was not proved.
        while True:
            try:
                os.killpg(child.pid, 0)
            except ProcessLookupError:
                break
            require(self.monotonic_ns() / 1e9 < end, "command-group-cleanup-unproved")
            time.sleep(0.005)
        value = {
            "stdout": bytes(buffers["stdout"]),
            "stderr": bytes(buffers["stderr"]),
            "returncode": child.returncode,
        }
        if failed is not None:
            raise ReadRefusal(
                str(failed) if isinstance(failed, ReadRefusal) else "command-failed",
                {
                    "stdout_sha256": sha(value["stdout"]),
                    "stdout_bytes": len(value["stdout"]),
                    "stderr_sha256": sha(value["stderr"]),
                    "stderr_bytes": len(value["stderr"]),
                    "returncode": child.returncode,
                },
            ) from failed
        self._time(deadline)
        return value


class ReadOnlyIO:
    def __init__(
        self,
        scope: ReadScope,
        *,
        deadline: float,
        backend: Any = None,
        maximum_bytes: int = MAX_CAPTURE_BYTES,
    ):
        require(
            type(deadline) in (int, float) and math.isfinite(deadline),
            "finite-absolute-deadline",
        )
        self.scope, self.deadline = scope, deadline
        require(
            type(maximum_bytes) is int and 1 <= maximum_bytes <= MAX_CAPTURE_BYTES,
            "capture-budget-range",
        )
        self.maximum_bytes = maximum_bytes
        self._backend = backend if backend is not None else System()
        self._consumed = 0
        self._returned = 0
        self._boot: str | None = None
        self._admitted: set[ProcessAdmission] = set()

    def _check(self) -> None:
        require(
            self._backend.monotonic_ns() / 1e9 < self.deadline, "absolute-read-deadline"
        )

    def _read(self, path: str, maximum: int, *, tail=False):
        self._check()
        data, meta = self._backend.read_file(path, maximum, self.deadline, tail=tail)
        require(len(data) <= maximum, "file-output-limit")
        self._consumed += len(data)
        require(self._consumed <= self.maximum_bytes, "capture-byte-budget")
        return data, meta

    def _clock(self) -> dict[str, Any]:
        self._check()
        raw, _ = self._read("/proc/sys/kernel/random/boot_id", 128)
        boot = raw.decode("ascii").strip()
        require(
            re.fullmatch(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}", boot),
            "boot-id-shape",
        )
        if self._boot is None:
            self._boot = boot
        require(boot == self._boot, "boot-changed")
        value = {
            "boot_id": boot,
            "monotonic_ns": self._backend.monotonic_ns(),
            "wall_ns": self._backend.wall_ns(),
        }
        self._check()
        return value

    def _observation(
        self,
        operation: str,
        subject: str,
        start: Mapping[str, Any],
        value: T,
        raw: bytes,
        **metadata,
    ) -> Observation[T]:
        end = self._clock()
        require(
            start["boot_id"] == end["boot_id"]
            and start["monotonic_ns"] <= end["monotonic_ns"]
            and start["wall_ns"] <= end["wall_ns"],
            "clock-regression",
        )
        if operation == "clock":
            value = cast(T, end)
            raw = encoded(end)
        audit = {
            "operation": operation,
            "subject": subject,
            "read_start": dict(start),
            "read_end": end,
            "raw": {"sha256": sha(raw), "bytes": len(raw)},
            **metadata,
        }
        self._returned += _value_size(value) + len(encoded(audit))
        require(self._returned <= self.maximum_bytes, "capture-output-budget")
        return Observation(value, audit)

    def clock(self) -> Observation[dict[str, Any]]:
        start = self._clock()
        return self._observation("clock", "collector", start, {}, b"")

    def birth_bracket(self) -> Observation[dict[str, Any]]:
        start = self._clock()
        m0 = self._backend.monotonic_ns()
        b0 = self._backend.boottime_ns()
        wall = self._backend.wall_ns()
        b1 = self._backend.boottime_ns()
        m1 = self._backend.monotonic_ns()
        hz = self._backend.hertz()
        require(
            type(hz) is int and hz > 0 and b0 <= b1 and m0 <= m1, "birth-clock-bracket"
        )
        value = {
            "boot_id": start["boot_id"],
            "monotonic_before_ns": m0,
            "boottime_before_ns": b0,
            "wall_ns": wall,
            "boottime_after_ns": b1,
            "monotonic_after_ns": m1,
            "clock_ticks_per_second": hz,
        }
        return self._observation(
            "birth-bracket", "collector", start, value, encoded(value)
        )

    def read(self, key: str) -> Observation[bytes]:
        require(key in self.scope.files, "unregistered-file-key")
        start = self._clock()
        entry = self.scope.files[key]
        data, meta = self._read(entry.path, entry.maximum_bytes)
        return self._observation("read", key, start, data, data, **meta)

    def tail(self, key: str) -> Observation[bytes]:
        require(key in self.scope.tails, "unregistered-tail-key")
        start = self._clock()
        entry = self.scope.tails[key]
        data, meta = self._read(entry.path, entry.maximum_bytes, tail=True)
        return self._observation("tail", key, start, data, data, **meta)

    def _metrics(self, key: str) -> FileKey:
        require(
            key == "metrics"
            and key in self.scope.tails
            and Path(self.scope.tails[key].path).name == "metrics.jsonl",
            "registered-learner-metrics-required",
        )
        return self.scope.tails[key]

    def _append_read(
        self,
        entry: FileKey,
        *,
        maximum: int,
        span: tuple[int, int] | None,
        expected: Mapping[str, Any] | None,
    ):
        self._check()

        def charge(count: int) -> None:
            self._consumed += count
            require(self._consumed <= self.maximum_bytes, "capture-byte-budget")

        data, meta = self._backend.append_file(
            entry.path,
            maximum,
            self.deadline,
            span=span,
            expected=expected,
            charge=charge,
            allowance=self.maximum_bytes - self._consumed,
        )
        require(len(data) <= maximum, "append-output-limit")
        return data, meta

    def append_fence(self, key: str) -> Observation[dict[str, Any]]:
        """Retain bounded private bytes through a fixed, observed metrics EOF.

        A prefix can begin mid-row. It is never silently line-trimmed; the pure
        consumer must refuse when the retained initial row is unrecoverable.
        This measurement alone grants no writer or preservation authority.
        """
        entry = self._metrics(key)
        start = self._clock()
        data, meta = self._append_read(
            entry,
            maximum=min(MAX_APPEND_PREFIX, entry.maximum_bytes),
            span=None,
            expected=None,
        )
        value = {
            "registered_key": key,
            "file_identity": _append_identity(entry.path, meta["stat_before"]),
            "size_at_fstat": meta["stat_before"]["bytes"],
            "prefix": {
                "start": meta["offset"],
                "end": meta["end_offset"],
                "sha256": sha(data),
                "raw": data,
            },
            "line_boundary": not data or data.endswith(b"\n"),
        }
        return self._observation("append-fence", key, start, value, data, **meta)

    def read_append_range(
        self,
        key: str,
        *,
        file_identity: Mapping[str, Any],
        start: int,
        end: int,
    ) -> Observation[dict[str, Any]]:
        """Read/reread one caller-fixed range of the same registered file.

        Ranges never chase a growing EOF. The consumer authenticates fences,
        demands contiguous coverage and compares original/reread bytes.
        """
        entry = self._metrics(key)
        require(
            isinstance(file_identity, Mapping)
            and set(file_identity) == {"path", "device", "inode", "uid", "gid", "mode"}
            and file_identity["path"] == entry.path
            and all(
                type(file_identity[k]) is int and file_identity[k] >= 0
                for k in ("device", "inode", "uid", "gid", "mode")
            )
            and file_identity["inode"] > 0
            and file_identity["mode"] <= 0o7777,
            "append-expected-identity",
        )
        require(
            type(start) is int
            and type(end) is int
            and 0 <= start <= end
            and end - start <= entry.maximum_bytes,
            "append-range-bounds",
        )
        clock = self._clock()
        data, meta = self._append_read(
            entry,
            maximum=entry.maximum_bytes,
            span=(start, end),
            expected=dict(file_identity),
        )
        value = {
            "registered_key": key,
            "file_identity": dict(file_identity),
            "start": start,
            "end": end,
            "sha256": sha(data),
            "raw": data,
            "size_before": meta["stat_before"]["bytes"],
            "size_after": meta["stat_after"]["bytes"],
        }
        return self._observation("append-range", key, clock, value, data, **meta)

    def query(self, kind: str, target: str | None = None) -> Observation[str]:
        if kind == "unit":
            require(target in self.scope.units, "unregistered-unit")
            assert target is not None and self.scope.property_sets is not None
            props = self.scope.property_sets[self.scope.units[target]]
            argv = (
                "systemctl",
                "show",
                target,
                "--all",
                "--no-pager",
                "--property=" + ",".join(props),
            )
        elif kind == "jobs":
            require(target is None, "query-target-not-permitted")
            # systemd255 list-jobs uses table_print, not the JSON output mode.
            argv = (
                "systemctl",
                "list-jobs",
                "--all",
                "--no-pager",
                "--no-legend",
                "--plain",
                "--full",
            )
        elif kind == "default-target":
            require(target is None, "query-target-not-permitted")
            argv = ("systemctl", "get-default")
        elif kind == "registered-target":
            require(target in self.scope.targets, "unregistered-target")
            assert target is not None
            argv = (
                "systemctl",
                "show",
                target,
                "--all",
                "--no-pager",
                "--property=Id,LoadState,FragmentPath,Wants,Requires,Before,After",
            )
        elif kind in {"gpu-inventory", "gpu-owners"}:
            require(target is None, "query-target-not-permitted")
            field = (
                "--query-gpu=index,uuid"
                if kind == "gpu-inventory"
                else "--query-compute-apps=pid,gpu_uuid"
            )
            argv = ("nvidia-smi", field, "--format=csv,noheader,nounits")
        else:
            raise ReadRefusal("query-kind-not-permitted")
        start = self._clock()
        try:
            result = self._backend.command(argv, self.deadline)
        except ReadRefusal as error:
            end = {
                "boot_id": None,
                "monotonic_ns": self._backend.monotonic_ns(),
                "wall_ns": self._backend.wall_ns(),
            }
            raise ReadRefusal(
                str(error),
                {
                    "operation": "query",
                    "subject": kind + (":" + target if target else ""),
                    "read_start": start,
                    "read_end": end,
                    "boot_rechecked": False,
                    **error.audit,
                },
            ) from error
        raw = result["stdout"]
        require(
            len(raw) <= MAX_FILE_BYTES and len(result["stderr"]) <= MAX_PROC_BYTES,
            "command-output-limit",
        )
        self._consumed += len(raw) + len(result["stderr"])
        require(self._consumed <= self.maximum_bytes, "capture-byte-budget")
        obs = self._observation(
            "query",
            kind + (":" + target if target else ""),
            start,
            raw.decode("utf-8"),
            raw,
            returncode=result["returncode"],
            stderr_sha256=sha(result["stderr"]),
            stderr_bytes=len(result["stderr"]),
        )
        if result["returncode"] != 0:
            raise ReadRefusal("query-returncode", obs.audit)
        return obs

    def stat_cached(self, key: str) -> Observation[dict[str, Any]]:
        require(key in self.scope.cached, "unregistered-cached-key")
        start = self._clock()
        entry = self.scope.cached[key]
        literal = self._backend.stat_path(entry.literal, self.deadline)
        link = None
        if entry.literal != entry.resolved:
            link = self._backend.link(entry.literal, self.deadline)
            require(link["resolved"] == entry.resolved, "cached-resolution-drift")
        resolved = self._backend.stat_path(entry.resolved, self.deadline)
        value = {
            "literal": entry.literal,
            "resolved": entry.resolved,
            "literal_stat": literal,
            "resolved_stat": resolved,
            "link": link,
        }
        return self._observation(
            "stat-cached", key, start, value, encoded(value), content_hashed=False
        )

    def _proc(
        self,
        pid: int,
        *,
        details: bool,
        expected: ProcessAdmission | None = None,
        maps: bool = True,
    ):
        self._check()
        raw = self._backend.proc(
            pid, self.deadline, details=details, expected=expected, maps=maps
        )
        self._consumed += sum(len(x) for x in raw.values() if isinstance(x, bytes))
        require(self._consumed <= self.maximum_bytes, "capture-byte-budget")
        before = _process_identity(pid, raw["stat_before"], raw["cgroup_before"])
        after = _process_identity(pid, raw["stat_after"], raw["cgroup_after"])
        require(before == after, "process-identity-raced")
        return raw, before

    def manager_identity(self) -> Observation[dict[str, Any]]:
        """Fixed PID1 lifetime/namespace metadata only; grants no PID admission."""
        start = self._clock()
        raw_before, before = self._proc(1, details=False, maps=False)
        self_ns = self._backend.namespace("self", self.deadline)
        manager_ns = self._backend.namespace(1, self.deadline)
        raw_after, after = self._proc(1, details=False, maps=False)
        require(
            before == after and before["ppid"] == 0 and before["start_ticks"] > 0,
            "manager-lifetime-raced",
        )
        require(
            self_ns == manager_ns and set(self_ns) == {"pid", "time"},
            "manager-namespace-mismatch",
        )
        value = {"boot_id": start["boot_id"], **before, "namespaces": manager_ns}
        components = {
            "before": _component_pins(raw_before),
            "after": _component_pins(raw_after),
            "namespaces": manager_ns,
        }
        return self._observation(
            "manager-identity",
            "pid1",
            start,
            value,
            encoded(components),
            raw_encoding="component-pins-v1",
        )

    def members(self, unit: str) -> Observation[tuple[ProcessAdmission, ...]]:
        require(self.scope.units.get(unit) == "service", "unregistered-service-cgroup")
        start = self._clock()
        group = "/system.slice/" + unit
        root = "/sys/fs/cgroup" + group
        todo = [root]
        found = {}
        components = []
        directories = 0
        while todo:
            self._check()
            path = todo.pop()
            directories += 1
            require(directories <= 64, "cgroup-directory-limit")
            try:
                data, meta = self._read(path + "/cgroup.procs", MAX_PROC_BYTES)
            except FileNotFoundError:
                require(path == root and directories == 1, "cgroup-disappeared")
                continue
            components.append(
                {
                    "path": path + "/cgroup.procs",
                    "sha256": sha(data),
                    "bytes": len(data),
                    **meta,
                }
            )
            for token in data.split():
                require(token.isdigit() and int(token) > 0, "cgroup-pid-format")
                pid = int(token)
                raw, identity = self._proc(pid, details=False)
                require(
                    identity["cgroup"] == group
                    or identity["cgroup"].startswith(group + "/"),
                    "foreign-cgroup-process",
                )
                require(pid not in found, "cgroup-duplicate-pid")
                require(len(found) < MAX_PIDS, "cgroup-process-limit")
                found[pid] = ProcessAdmission(
                    pid, identity["start_ticks"], identity["cgroup"]
                )
                components.append({"pid": pid, "components": _component_pins(raw)})
            for item in self._backend.directory(path, self.deadline):
                if item["directory"]:
                    require(
                        not item["symlink"]
                        and re.fullmatch(r"[^/\x00]+", item["name"]),
                        "cgroup-child-path",
                    )
                    todo.append(path + "/" + item["name"])
        retained = {
            a
            for a in self._admitted
            if a.cgroup != group and not a.cgroup.startswith(group + "/")
        }
        result = tuple(found[p] for p in sorted(found))
        require(len(retained | set(result)) <= MAX_PIDS, "capture-process-limit")
        self._admitted = retained | set(result)
        raw = encoded(components)
        return self._observation(
            "cgroup-members", unit, start, result, raw, raw_encoding="component-pins-v1"
        )

    def process(
        self, admission: ProcessAdmission, *, maps: bool = True
    ) -> Observation[dict[str, Any]]:
        require(type(maps) is bool, "maps-selection-type")
        require(admission in self._admitted, "process-not-admitted")
        start = self._clock()
        raw, identity = self._proc(
            admission.pid, details=True, expected=admission, maps=maps
        )
        require(
            (identity["pid"], identity["start_ticks"], identity["cgroup"])
            == (admission.pid, admission.start_ticks, admission.cgroup),
            "process-admission-drift",
        )
        value = {
            **identity,
            "exe": raw["exe"],
            "cwd": raw["cwd"],
            "cmdline": raw["cmdline"],
            "environ": raw["environ"],
            "maps": raw.get("maps"),
        }
        # Raw origin values remain private. Only component hashes enter audit.
        return self._observation(
            "process",
            str(admission.pid),
            start,
            value,
            encoded(_component_pins(raw)),
            raw_encoding="component-pins-v1",
            maps_read=maps,
        )

    def process_credentials(
        self, admission: ProcessAdmission
    ) -> Observation[dict[str, Any]]:
        """Measure a current UID vector, never historical writer authority."""
        require(
            type(admission) is ProcessAdmission and admission in self._admitted,
            "credential-process-not-admitted",
        )
        start = self._clock()

        def charge(count: int) -> None:
            self._consumed += count
            require(self._consumed <= self.maximum_bytes, "capture-byte-budget")

        raw = self._backend.proc_credentials(
            admission,
            self.deadline,
            charge=charge,
            allowance=self.maximum_bytes - self._consumed,
        )
        require(
            set(raw)
            == {"stat_before", "cgroup_before", "status", "stat_after", "cgroup_after"}
            and all(
                type(v) is bytes and len(v) <= MAX_PROC_BYTES for v in raw.values()
            ),
            "credential-observation-shape",
        )
        before = _process_identity(
            admission.pid, raw["stat_before"], raw["cgroup_before"]
        )
        after = _process_identity(admission.pid, raw["stat_after"], raw["cgroup_after"])
        require(
            before == after
            and (before["start_ticks"], before["cgroup"])
            == (admission.start_ticks, admission.cgroup),
            "credential-process-raced",
        )
        rows = [line for line in raw["status"].split(b"\n") if line.startswith(b"Uid:")]
        require(
            raw["status"].endswith(b"\n")
            and len(rows) == 1
            and re.fullmatch(rb"Uid:[ \t]+[0-9]+(?:[ \t]+[0-9]+){3}[ \t]*", rows[0]),
            "credential-uid-fields",
        )
        numbers = rows[0].split()[1:]
        require(
            all(len(n) <= 10 and int(n) <= 2**32 - 1 for n in numbers),
            "credential-uid-range",
        )
        uids = dict(
            zip(("real", "effective", "saved", "filesystem"), map(int, numbers))
        )
        components = _component_pins(raw)
        return self._observation(
            "process-credentials",
            str(admission.pid),
            start,
            {**before, "uids": uids},
            encoded(components),
            raw_encoding="component-pins-v1",
            components=components,
            uids=uids,
        )

    def namespaces(
        self, subject: str | int | ProcessAdmission
    ) -> Observation[dict[str, Any]]:
        if isinstance(subject, ProcessAdmission):
            require(subject in self._admitted, "namespace-process-not-admitted")
            pid = subject.pid
        else:
            require(
                subject == "self" or type(subject) is int and subject == 1,
                "namespace-subject",
            )
            pid = subject
        start = self._clock()
        before = None
        if isinstance(subject, ProcessAdmission):
            _, before = self._proc(subject.pid, details=False)
            require(
                (before["start_ticks"], before["cgroup"])
                == (subject.start_ticks, subject.cgroup),
                "namespace-admission-drift",
            )
        value = self._backend.namespace(pid, self.deadline)
        if isinstance(subject, ProcessAdmission):
            _, after = self._proc(subject.pid, details=False)
            require(before == after, "namespace-process-raced")
        return self._observation("namespaces", str(pid), start, value, encoded(value))

    def boot_links(self, unit: str) -> Observation[tuple[dict[str, Any], ...]]:
        require(unit in self.scope.units, "unregistered-unit")
        start = self._clock()
        links = []
        count = 0
        for root in ("/etc/systemd/system", "/run/systemd/system"):
            for row in self._backend.directory(root, self.deadline):
                count += 1
                require(count <= 8192, "boot-link-entry-limit")
                name = row["name"]
                if not name.endswith((".target.wants", ".target.requires")):
                    continue
                require(
                    row["directory"] and not row["symlink"], "boot-edge-directory-alias"
                )
                parent = root + "/" + name
                for edge in self._backend.directory(parent, self.deadline):
                    count += 1
                    require(count <= 8192, "boot-link-entry-limit")
                    path = parent + "/" + edge["name"]
                    if edge["symlink"]:
                        value = self._backend.link(path, self.deadline)
                        if edge["name"] == unit or Path(value["resolved"]).name == unit:
                            before = self._backend.stat_path(path, self.deadline)
                            again = self._backend.link(path, self.deadline)
                            after = self._backend.stat_path(path, self.deadline)
                            require(
                                value == again and before == after, "boot-link-raced"
                            )
                            links.append({"path": path, **value, "stat": before})
                    elif edge["name"] == unit:
                        raise ReadRefusal("unit-boot-edge-not-symlink")
        ordered = tuple(sorted(links, key=lambda x: x["path"]))
        return self._observation("boot-links", unit, start, ordered, encoded(ordered))


def _process_identity(pid: int, raw: bytes, cgroup: bytes) -> dict[str, Any]:
    require(raw.startswith(str(pid).encode() + b" ("), "proc-stat-pid")
    fields = raw.rsplit(b")", 1)[-1].split()
    require(len(fields) >= 20 and fields[0] not in {b"Z", b"X"}, "process-not-live")
    require(fields[1].isdigit() and fields[19].isdigit(), "proc-stat-fields")
    groups = cgroup.decode("utf-8").splitlines()
    require(len(groups) == 1 and groups[0].startswith("0::/"), "cgroup-v2-required")
    return {
        "pid": pid,
        "start_ticks": int(fields[19]),
        "ppid": int(fields[1]),
        "cgroup": groups[0][3:],
    }


def _value_size(value: Any) -> int:
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value.encode())
    if isinstance(value, Mapping):
        return sum(_value_size(k) + _value_size(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return sum(_value_size(v) for v in value)
    if isinstance(value, ProcessAdmission):
        return 32 + len(value.cgroup.encode())
    return len(encoded(value))


def _component_pins(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key, item in value.items():
        data = item if isinstance(item, bytes) else encoded(item)
        result[key] = {"sha256": sha(data), "bytes": len(data)}
    return result
