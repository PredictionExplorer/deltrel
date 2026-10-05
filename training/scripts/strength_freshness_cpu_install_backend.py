"""Closed dummy installation transaction; real admission is external and required.

No production unit name, arbitrary shell command, or JSON callback is accepted.
The private filesystem seam exists only for local fault tests. A caller must
retain the original helper and dummy deadlines; this backend cannot renew them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from typing import Any, Protocol

from scripts import strength_freshness_cpu_install as render
from scripts import strength_freshness_cpu_lifecycle as life
from scripts import strength_freshness_cpu_outer as outer

OPERATIONS = {"prepare-install", "inspect-cleanup", "audit", "prearm-cleanup"}
MAX_JOURNAL_BYTES = 2 * 2**20


class InstallRefusal(RuntimeError):
    """Fixed codes; never include private environment or rejected file contents."""


def require(value: object, code: str) -> None:
    if not value:
        raise InstallRefusal(code)


def encoded(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: str) -> Path:
    p = Path(value)
    require(
        p.is_absolute()
        and str(p) == value
        and not value.startswith("//")
        and ".." not in p.parts
        and "\0" not in value,
        "install-path",
    )
    return p


def identity(st: os.stat_result) -> dict[str, int]:
    return {
        "device": st.st_dev,
        "inode": st.st_ino,
        "uid": st.st_uid,
        "gid": st.st_gid,
        "mode": stat.S_IMODE(st.st_mode),
    }


def version(st: os.stat_result) -> tuple[int, ...]:
    return (
        st.st_dev,
        st.st_ino,
        st.st_mode,
        st.st_uid,
        st.st_gid,
        st.st_size,
        st.st_mtime_ns,
        st.st_ctime_ns,
    )


class Host(Protocol):
    def clock(self) -> life.Clock: ...
    def fresh(self, names: tuple[str, ...], deadline: float) -> dict[str, Any]: ...
    def inert(self, names: tuple[str, ...], deadline: float) -> dict[str, Any]: ...
    def reload(self, deadline: float) -> None: ...
    def verify_sources(self, prepared: render.Prepared, deadline: float) -> None: ...
    def start(self, name: str, deadline: float) -> None: ...
    def arm(self, prepared: render.Prepared, deadline: float) -> dict[str, Any]: ...
    def recheck_arm(self, pin: dict[str, Any], deadline: float) -> None: ...
    def started(self, name: str, deadline: float) -> dict[str, Any]: ...
    def inspect_cleanup(
        self, prepared: render.Prepared, deadline: float
    ) -> dict[str, Any]: ...
    def audit(self, prepared: render.Prepared, deadline: float) -> dict[str, Any]: ...


@dataclass(frozen=True)
class StartAuthorization:
    ack_pin: dict[str, Any]
    start_pin: dict[str, Any]
    enclosing: dict[str, Any]


class Files:
    """Nofollow/O_EXCL writes, with a constructor-only private test namespace."""

    def __init__(self) -> None:
        self._prefix: Path | None = None
        self.uid = self.gid = 0

    @classmethod
    def _private_test_root(cls, root: Path) -> Files:
        require(
            root.is_absolute() and root.resolve() == root and root.is_dir(), "test-root"
        )
        result = cls()
        result._prefix = root
        result.uid, result.gid = os.getuid(), os.getgid()
        return result

    def path(self, logical: str) -> Path:
        p = canonical(logical)
        return p if self._prefix is None else self._prefix / p.relative_to("/")

    def parent(self, logical: str) -> int:
        p = self.path(logical)
        require(p.parent.resolve() == p.parent, "install-parent-symlink")
        fd = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        st = os.fstat(fd)
        if st.st_uid != self.uid or st.st_gid != self.gid or st.st_mode & 0o022:
            os.close(fd)
            raise InstallRefusal("install-parent-owner-mode")
        return fd

    def stat(self, logical: str) -> os.stat_result | None:
        p = self.path(logical)
        require(p.parent.resolve() == p.parent, "install-parent-symlink")
        try:
            return p.lstat()
        except FileNotFoundError:
            return None

    def read(self, logical: str, maximum: int) -> bytes:
        parent = self.parent(logical)
        try:
            fd = os.open(
                self.path(logical).name,
                os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
                dir_fd=parent,
            )
            try:
                before = os.fstat(fd)
                require(
                    stat.S_ISREG(before.st_mode)
                    and before.st_uid == self.uid
                    and before.st_size <= maximum,
                    "install-file-kind-owner-size",
                )
                raw = b""
                while len(raw) <= maximum:
                    part = os.read(fd, min(65536, maximum + 1 - len(raw)))
                    if not part:
                        break
                    raw += part
                after = os.fstat(fd)
                linked = os.stat(
                    self.path(logical).name, dir_fd=parent, follow_symlinks=False
                )
                require(
                    len(raw) <= maximum
                    and identity(before) == identity(after) == identity(linked)
                    and before.st_size == after.st_size == len(raw)
                    and before.st_mtime_ns == after.st_mtime_ns,
                    "install-read-raced",
                )
                return raw
            finally:
                os.close(fd)
        finally:
            os.close(parent)

    def mkdir(self, logical: str) -> dict[str, int]:
        parent = self.parent(logical)
        try:
            os.mkdir(self.path(logical).name, 0o700, dir_fd=parent)
            st = os.stat(self.path(logical).name, dir_fd=parent, follow_symlinks=False)
            require(
                stat.S_ISDIR(st.st_mode)
                and st.st_uid == self.uid
                and stat.S_IMODE(st.st_mode) == 0o700,
                "install-created-directory",
            )
            os.fsync(parent)
            return identity(st)
        finally:
            os.close(parent)

    def create(self, item: render.Artifact, journal: Journal) -> None:
        journal.record("create-intent", {"pin": item.pin, "mode": item.mode})
        parent = self.parent(item.path)
        fd = -1
        try:
            fd = os.open(
                self.path(item.path).name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent,
            )
            born = identity(os.fstat(fd))
            journal.record("created-inode", {"path": item.path, "identity": born})
            offset = 0
            while offset < len(item.data):
                journal.check()
                written = os.write(fd, item.data[offset : offset + 65536])
                require(written > 0, "install-short-write")
                offset += written
            os.fsync(fd)
            journal.check()
            os.fchmod(fd, item.mode)
            os.fsync(fd)
            linked = os.stat(
                self.path(item.path).name, dir_fd=parent, follow_symlinks=False
            )
            require(
                identity(linked) == identity(os.fstat(fd))
                and linked.st_ino == born["inode"]
                and linked.st_dev == born["device"],
                "install-create-raced",
            )
            os.fsync(parent)
            require(
                self.read(item.path, len(item.data)) == item.data,
                "install-write-readback",
            )
            journal.record(
                "create-complete",
                {
                    "pin": item.pin,
                    "identity": identity(linked),
                    "version": list(version(linked)),
                },
            )
        finally:
            if fd >= 0:
                os.close(fd)
            os.close(parent)

    def unlink_owned(
        self,
        item: render.Artifact,
        born: dict[str, int],
        journal: Journal,
        completed_version: list[int],
    ) -> bool:
        current = self.stat(item.path)
        if current is None:
            return False
        require(
            stat.S_ISREG(current.st_mode)
            and current.st_uid == self.uid
            and current.st_gid == self.gid
            and current.st_dev == born["device"]
            and current.st_ino == born["inode"]
            and stat.S_IMODE(current.st_mode) == item.mode,
            "prearm-file-owner",
        )
        require(
            self.read(item.path, len(item.data)) == item.data, "prearm-file-not-exact"
        )
        require(list(version(current)) == completed_version, "prearm-file-version")
        journal.record(
            "unlink-intent", {"pin": item.pin, "identity": identity(current)}
        )
        parent = self.parent(item.path)
        try:
            again = os.stat(
                self.path(item.path).name, dir_fd=parent, follow_symlinks=False
            )
            require(version(again) == version(current), "prearm-unlink-raced")
            os.unlink(self.path(item.path).name, dir_fd=parent)
            os.fsync(parent)
        finally:
            os.close(parent)
        journal.record("unlink-complete", {"path": item.path})
        return True


class Journal:
    """Append-only durable operation evidence; never truncates prior history."""

    def __init__(
        self, files: Files, path: str, binding: dict[str, Any], check, *, create: bool
    ):
        self.files, self.path, self.binding, self.check = files, path, binding, check
        self.rows: list[dict[str, Any]] = []
        parent = files.parent(path)
        try:
            flags = os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW
            self.fd = os.open(
                files.path(path).name,
                flags | (os.O_CREAT | os.O_EXCL if create else 0),
                0o600,
                dir_fd=parent,
            )
            st = os.fstat(self.fd)
            require(
                stat.S_ISREG(st.st_mode)
                and st.st_uid == files.uid
                and stat.S_IMODE(st.st_mode) == 0o600
                and st.st_size <= MAX_JOURNAL_BYTES,
                "install-journal-protection",
            )
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.identity = identity(st)
            if create:
                os.fsync(self.fd)
                os.fsync(parent)
            else:
                raw = files.read(path, MAX_JOURNAL_BYTES)
                require(not raw or raw.endswith(b"\n"), "install-journal-partial")
                for line in raw.splitlines():
                    row = life.strict_json(line)
                    require(
                        set(row)
                        == {
                            "sequence",
                            "previous_sha256",
                            "binding",
                            "clock",
                            "event",
                            "data",
                        }
                        and row["sequence"] == len(self.rows)
                        and row["binding"] == binding
                        and row["previous_sha256"]
                        == (sha(encoded(self.rows[-1])) if self.rows else None),
                        "install-journal-chain",
                    )
                    self.rows.append(row)
                require(
                    bool(self.rows) and self.rows[0]["event"] == "installation-intent",
                    "install-journal-intent",
                )
        except BaseException:
            if hasattr(self, "fd"):
                os.close(self.fd)
            raise
        finally:
            os.close(parent)

    def record(self, event: str, data: dict[str, Any]) -> None:
        now = self.check()
        row = {
            "sequence": len(self.rows),
            "previous_sha256": sha(encoded(self.rows[-1])) if self.rows else None,
            "binding": self.binding,
            "clock": asdict(now),
            "event": event,
            "data": data,
        }
        raw = encoded(row)
        require(
            len(raw) <= 262144
            and os.fstat(self.fd).st_size + len(raw) <= MAX_JOURNAL_BYTES,
            "install-journal-bound",
        )
        linked = self.files.stat(self.path)
        require(linked is not None, "install-journal-missing")
        assert linked is not None
        require(
            identity(os.fstat(self.fd)) == self.identity
            and identity(linked) == self.identity,
            "install-journal-raced",
        )
        require(os.write(self.fd, raw) == len(raw), "install-journal-short-write")
        os.fsync(self.fd)
        self.rows.append(row)

    def close(self) -> None:
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


class Backend:
    """One original prepared transaction. No restart/adoption of unknown files."""

    def __init__(
        self,
        prepared: render.Prepared,
        host: Host,
        files: Files,
        *,
        intent_sha256: str,
        work_deadline: life.Clock,
        start_authorization: StartAuthorization,
    ):
        self.prepared, self.host, self.files = prepared, host, files
        self.plan, self.anchor = prepared.plan.value, prepared.anchor
        require(
            self.anchor.plan_sha256 == prepared.plan.checksum
            and self.anchor.nonce == self.plan["nonce"]
            and self.anchor.boot_id == self.plan["boot_id"],
            "install-original-anchor",
        )
        require(
            len(self.plan["units"]) == 12 and render.q.HASH.fullmatch(intent_sha256),
            "install-authority-shape",
        )
        self.work_deadline = work_deadline
        self.start_authorization = start_authorization
        self.names = tuple(sorted(self.plan["units"]))
        self.binding = {
            "format": "strength-freshness-install-transaction-v1",
            "intent_sha256": intent_sha256,
            "template_sha256": prepared.template_sha256,
            "plan_sha256": prepared.plan.checksum,
            "anchor": self.anchor.as_dict(),
            "start_ack_pin": start_authorization.ack_pin,
        }
        self.journal_path = str(
            Path(self.plan["input_root"]) / "installation-journal.jsonl"
        )
        self.artifacts = {a.path: a for a in (*prepared.inputs, *prepared.installed)}
        marker = render.Artifact(
            str(Path(self.plan["scratch_root"]) / "attempt-armed"),
            encoded(
                {
                    "plan_sha256": prepared.plan.checksum,
                    "anchor_sha256": life.digest(self.anchor.as_dict()),
                }
            ),
            0o444,
        )
        require(
            marker.path not in self.artifacts
            and self.journal_path not in self.artifacts
            and len(self.artifacts) == len(prepared.inputs) + len(prepared.installed),
            "install-artifact-alias",
        )
        self.artifacts[marker.path] = marker
        self.marker = marker
        expected_installed = {
            spec["installed_path"] for spec in self.plan["units"].values()
        } | {
            env["path"]
            for spec in self.plan["units"].values()
            for env in spec["before"]["environment_files"]
        }
        require(
            {a.path for a in prepared.installed} == expected_installed,
            "install-closed-targets",
        )
        for item in prepared.installed:
            require(
                any(
                    v["source"]["sha256"] == sha(item.data)
                    and v["source"]["bytes"] == len(item.data)
                    and v["mode"] == item.mode
                    for v in self.plan["files"][item.path]
                ),
                "install-closed-target-bytes",
            )
        for item in prepared.inputs:
            require(
                canonical(item.path).is_relative_to(canonical(self.plan["input_root"]))
                and item.path != self.plan["input_root"]
                and item.mode in {0o444, 0o600},
                "install-closed-inputs",
            )

    def check(self) -> life.Clock:
        now = self.host.clock()
        require(
            now.boot_id == self.anchor.boot_id == self.work_deadline.boot_id
            and self.anchor.started_monotonic
            <= now.monotonic
            < self.work_deadline.monotonic
            and self.anchor.started_wall_ns <= now.wall_ns < self.work_deadline.wall_ns,
            "install-original-deadline",
        )
        return now

    def deadline(self, *, setup: bool = False) -> float:
        now = self.check()
        end = min(
            self.work_deadline.monotonic,
            now.monotonic + (self.work_deadline.wall_ns - now.wall_ns) / 1e9,
            self.anchor.effective_deadline("audit", now),
        )
        if setup:
            end = min(
                end,
                self.anchor.started_monotonic + render.SETUP_SECONDS,
                now.monotonic
                + (
                    self.anchor.started_wall_ns
                    + render.SETUP_SECONDS * 10**9
                    - now.wall_ns
                )
                / 1e9,
            )
        require(now.monotonic < end, "install-setup-deadline")
        return end

    def _inert(self, journal: Journal, *, fresh: bool) -> None:
        raw = (self.host.fresh if fresh else self.host.inert)(
            self.names, self.deadline()
        )
        journal.record("raw-fresh" if fresh else "raw-inert", raw)
        require(set(raw) == set(self.names), "install-observed-unit-inventory")
        for name, row in raw.items():
            require(
                row["name"] == name
                and row["jobs"] == []
                and row["members"] == []
                and row["links"] == {},
                "install-unit-not-inert",
            )
            if fresh:
                require(
                    row["unit_list"] == []
                    and row["unit_files"] == []
                    and row["cgroup_exists"] is False,
                    "install-unit-already-loaded",
                )
            else:
                require(
                    row["unit_list"] == []
                    or (
                        row["active"] == "inactive"
                        and row["main_pid"] == 0
                        and row["job_id"] == 0
                    ),
                    "install-unit-not-inert",
                )

    def prepare_install(self) -> dict[str, Any]:
        self.deadline(setup=True)
        self._authorize_start()
        journal = Journal(
            self.files, self.journal_path, self.binding, self.check, create=True
        )
        try:
            journal.record(
                "installation-intent",
                {"artifact_pins": [a.pin for a in self.artifacts.values()]},
            )
            self._inert(journal, fresh=True)
            for item in self.artifacts.values():
                require(self.files.stat(item.path) is None, "install-existing-file")
            scratch = Path(self.plan["scratch_root"])
            require(self.files.stat(str(scratch)) is None, "install-existing-scratch")
            # Only registered input descendants and the one exact scratch root.
            logical_dirs = {
                scratch,
                scratch / "env",
                scratch / "evidence",
                scratch / "external",
                scratch / "payloads",
                scratch / "synthetic-run",
                scratch / "synthetic-support-state",
            }
            input_root = Path(self.plan["input_root"])
            for item in self.artifacts.values():
                p = Path(item.path).parent
                while p != input_root and p.is_relative_to(input_root):
                    logical_dirs.add(p)
                    p = p.parent
            for directory in sorted(logical_dirs, key=lambda p: (len(p.parts), str(p))):
                self.deadline(setup=True)
                journal.record("mkdir-intent", {"path": str(directory)})
                born = self.files.mkdir(str(directory))
                journal.record(
                    "mkdir-complete", {"path": str(directory), "identity": born}
                )
            for item in (*self.prepared.inputs, *self.prepared.installed):
                self.deadline(setup=True)
                self.files.create(item, journal)
            journal.record("reload-intent", {})
            self.host.reload(self.deadline(setup=True))
            journal.record("reload-complete", {})
            self.host.verify_sources(self.prepared, self.deadline(setup=True))
            self._inert(journal, fresh=False)
            self.files.create(self.marker, journal)
            watchdog, *controls = self.prepared.start_order
            journal.record("watchdog-start-intent", {"name": watchdog})
            self.host.start(watchdog, self.deadline(setup=True))
            arm = self.host.arm(self.prepared, self.deadline(setup=True))
            journal.record("cleanup-armed", {"pin": arm})
            for name in controls:
                self.host.recheck_arm(arm, self.deadline(setup=True))
                journal.record("control-start-intent", {"name": name})
                self.host.start(name, self.deadline(setup=True))
                observed = self.host.started(name, self.deadline(setup=True))
                journal.record("control-started", {"name": name, "observed": observed})
            journal.record("setup-complete", {"arm": arm})
            return {
                "status": "installed-and-armed",
                "binding": self.binding,
                "cleanup_arm": arm,
                "journal_path": self.journal_path,
                "execution_qualified": False,
            }
        finally:
            journal.close()

    def prearm_cleanup(self) -> dict[str, Any]:
        self._authorize_start()
        journal = Journal(
            self.files, self.journal_path, self.binding, self.check, create=False
        )
        try:
            require(
                not any(
                    r["event"]
                    in {"control-start-intent", "control-started", "setup-complete"}
                    for r in journal.rows
                ),
                "prearm-control-started",
            )
            self._inert(journal, fresh=False)
            # Unknown watchdog outcomes require the same fresh inert/job/link
            # proof above; no stop or early timer retirement is performed here.
            require(
                journal.rows[0]["data"]["artifact_pins"]
                == [a.pin for a in self.artifacts.values()],
                "prearm-artifact-inventory",
            )
            intents, born, completed = {}, {}, {}
            for row in journal.rows:
                if row["event"] == "create-intent":
                    data = row["data"]
                    path = data["pin"]["path"]
                    require(
                        path in self.artifacts
                        and path not in intents
                        and data
                        == {
                            "pin": self.artifacts[path].pin,
                            "mode": self.artifacts[path].mode,
                        },
                        "prearm-create-intent",
                    )
                    intents[path] = data
                elif row["event"] == "created-inode":
                    path, observed = row["data"]["path"], row["data"]["identity"]
                    require(
                        path in intents
                        and path not in born
                        and set(observed) == {"device", "inode", "uid", "gid", "mode"}
                        and all(type(v) is int and v >= 0 for v in observed.values())
                        and observed["mode"] == 0o600
                        and observed["uid"] == self.files.uid
                        and observed["gid"] == self.files.gid,
                        "prearm-creation-proof",
                    )
                    born[path] = observed
                elif row["event"] == "create-complete":
                    data = row["data"]
                    path = data["pin"]["path"]
                    require(
                        path in born
                        and path not in completed
                        and data["pin"] == self.artifacts[path].pin
                        and data["identity"]["inode"] == born[path]["inode"]
                        and data["identity"]["device"] == born[path]["device"]
                        and isinstance(data["version"], list)
                        and len(data["version"]) == 8
                        and all(type(v) is int and v >= 0 for v in data["version"]),
                        "prearm-completion-proof",
                    )
                    completed[path] = data["version"]
            removed = []
            # Remove definitions before their env/source inputs, so a crash
            # never requires accepting a still-installed definition whose own
            # dependencies were removed by this cleanup.
            order = sorted(
                self.artifacts,
                key=lambda path: (
                    0
                    if path == self.marker.path
                    else 1
                    if path.startswith("/etc/systemd/system/")
                    else 2
                    if path in {a.path for a in self.prepared.installed}
                    else 3,
                    path,
                ),
            )
            for path in order:
                item = self.artifacts[path]
                if self.files.stat(path) is None:
                    continue
                require(path in born and path in completed, "prearm-creation-unproved")
                self._inert(journal, fresh=False)
                if self.files.unlink_owned(item, born[path], journal, completed[path]):
                    removed.append(path)
            journal.record("prearm-reload-intent", {})
            self.host.reload(self.deadline())
            self._inert(journal, fresh=True)
            journal.record(
                "prearm-cleanup-complete",
                {"removed": removed, "retained_journal_and_directories": True},
            )
            return {
                "status": "prearm-owned-files-removed",
                "removed": removed,
                "binding": self.binding,
                "execution_qualified": False,
            }
        finally:
            journal.close()

    def inspect_cleanup(self) -> dict[str, Any]:
        self._authorize_start()
        return self.host.inspect_cleanup(self.prepared, self.deadline())

    def audit(self) -> dict[str, Any]:
        self._authorize_start()
        return self.host.audit(self.prepared, self.deadline())

    def _authorize_start(self) -> None:
        """Re-read the immutable supervisor ack before acquiring write authority."""
        self.check()
        a = self.start_authorization
        root = Path(self.plan["input_root"])
        require(
            a.ack_pin["path"] == str(root / "dummy-start.ack.json")
            and a.start_pin["path"] == str(root / "dummy-start.json"),
            "install-start-authority-paths",
        )
        values = []
        for pin in (a.start_pin, a.ack_pin):
            render.q.pin_shape(pin)
            require(pin["bytes"] <= 32768, "install-start-authority-bound")
            raw = self.files.read(pin["path"], 32768)
            st = self.files.stat(pin["path"])
            require(
                st is not None
                and stat.S_IMODE(st.st_mode) == 0o444
                and st.st_uid == self.files.uid
                and st.st_gid == self.files.gid
                and sha(raw) == pin["sha256"]
                and len(raw) == pin["bytes"],
                "install-start-authority-pin",
            )
            values.append(life.strict_json(raw))
        start, ack = values
        require(
            set(start)
            == {
                "format",
                "schema_version",
                "outer_intent_sha256",
                "before_execution_pin",
                "clock",
            }
            and start["format"] == "strength-freshness-dummy-start-v1"
            and type(start["schema_version"]) is int
            and start["schema_version"] == 1
            and start["outer_intent_sha256"] == self.binding["intent_sha256"]
            and start["before_execution_pin"]
            == self.plan["preservation"]["before_execution"],
            "install-start-authority-body",
        )
        clock = start["clock"]
        require(
            set(clock) == {"boot_id", "monotonic_ns", "wall_ns"}
            and type(clock["monotonic_ns"]) is int
            and type(clock["wall_ns"]) is int
            and clock["boot_id"] == self.anchor.boot_id
            and clock["monotonic_ns"] / 1e9 == self.anchor.started_monotonic
            and clock["wall_ns"] == self.anchor.started_wall_ns,
            "install-start-authority-clock",
        )
        outer.validate_dummy_start_ack(
            ack,
            outer_intent_sha256=self.binding["intent_sha256"],
            nonce=self.anchor.nonce,
            start_pin=a.start_pin,
            before_execution_pin=start["before_execution_pin"],
            enclosing=a.enclosing,
        )
        self.check()


class SiteHost:
    """Fixed systemd adapter, created only after the helper's real admission.

    It never accepts command arrays from a frame. Public operations retain the
    same twelve-unit plan and original helper deadline for all nested q calls.
    """

    def __init__(
        self, prepared: render.Prepared, work_deadline: life.Clock, *, admission_check
    ):
        from scripts import strength_freshness_cpu_driver as driver

        self.prepared, self.plan, self.anchor = (
            prepared,
            prepared.plan.value,
            prepared.anchor,
        )
        self.driver = driver
        self.stop = work_deadline
        self.log = None
        self.arm_pin = None
        require(os.geteuid() == os.getegid() == 0, "site-root-required")
        stop = self.stop

        class BoundedIO(driver.TargetIO):
            def now(self):
                admission_check()
                now = super().now()
                require(
                    now < stop.monotonic and time.time_ns() < stop.wall_ns,
                    "site-helper-deadline",
                )
                return now

            def command(self, argv, deadline):
                now = self.now()
                return super().command(
                    argv,
                    min(
                        deadline,
                        stop.monotonic,
                        now + (stop.wall_ns - time.time_ns()) / 1e9,
                    ),
                )

            def exists(self, path):
                self.now()
                result = super().exists(path)
                self.now()
                return result

            def sleep(self, seconds):
                now = self.now()
                remaining = min(
                    stop.monotonic - now, (stop.wall_ns - time.time_ns()) / 1e9
                )
                super().sleep(min(seconds, max(0, remaining)))

        self.io = BoundedIO()
        require(self.io.boot_id() == self.anchor.boot_id == stop.boot_id, "site-boot")

    def clock(self) -> life.Clock:
        return life.Clock(self.io.boot_id(), self.io.now(), self.io.wall_ns())

    def _log(self):
        if self.log is None:
            root = Path(self.plan["scratch_root"]) / "evidence"
            require(
                root.exists() and root.is_dir() and root.resolve() == root,
                "site-evidence-directory-missing",
            )
            self.log = self.driver.EvidenceLog(root, self.anchor)
        return self.log

    def _read_context(self):
        q = render.q
        io = q.ClosedIO(
            self.prepared.plan, self.anchor, self.io, self._log(), purpose="read"
        )
        return io, q.DummyReadHost(io)

    def _observe(self, names, deadline, *, fresh):
        q = render.q
        require(set(names) == set(self.plan["units"]), "site-unit-inventory")
        jobs = q.linux.parse_jobs_text(
            self.io.command(list(q.linux.JOBS_COMMAND), deadline)
        )
        result = {}
        for name in names:
            # Empty manager/unit-file lists are retained as absence observations,
            # never relabeled as invented loaded-unit properties.
            unit_list = self.io.command(
                [
                    "systemctl",
                    "list-units",
                    "--all",
                    "--plain",
                    "--no-legend",
                    "--no-pager",
                    name,
                ],
                deadline,
            ).splitlines()
            unit_files = self.io.command(
                [
                    "systemctl",
                    "list-unit-files",
                    "--plain",
                    "--no-legend",
                    "--no-pager",
                    name,
                ],
                deadline,
            ).splitlines()
            require(
                all(
                    line.split() and line.split()[0] == name
                    for line in unit_list + unit_files
                ),
                "site-list-scope",
            )
            row = {
                "name": name,
                "unit_list": unit_list,
                "unit_files": unit_files,
                "active": None,
                "main_pid": None,
                "job_id": None,
                "jobs": [v for v in jobs.values() if v["unit"] == name],
                "members": [
                    asdict(p)
                    for p in self.io.members("/system.slice/" + name, deadline)
                ],
                "links": self.io.boot_links(name, deadline),
                "cgroup_exists": self.io.exists(
                    Path("/sys/fs/cgroup/system.slice") / name
                ),
            }
            if unit_list:
                fields = [
                    "Id",
                    "LoadState",
                    "ActiveState",
                    "SubState",
                    "Job",
                    "FragmentPath",
                    "DropInPaths",
                ]
                if name.endswith(".service"):
                    fields.append("MainPID")
                text = self.io.command(
                    [
                        "systemctl",
                        "show",
                        name,
                        "--all",
                        "--no-pager",
                        "--property=" + ",".join(fields),
                    ],
                    deadline,
                )
                pairs = [line.split("=", 1) for line in text.splitlines()]
                require(
                    all(len(pair) == 2 for pair in pairs)
                    and len({pair[0] for pair in pairs}) == len(pairs),
                    "site-property-lines",
                )
                props = dict(pairs)
                require(
                    set(props) == set(fields)
                    and props["Id"] == name
                    and props["DropInPaths"] == ""
                    and props["FragmentPath"]
                    in {"", self.plan["units"][name]["installed_path"]},
                    "site-unit-definition",
                )
                job = q.linux.parse_job(props["Job"], jobs, name)
                require(props.get("MainPID", "0").isdigit(), "site-unit-owner-fields")
                row.update(
                    active=props["ActiveState"],
                    main_pid=int(props.get("MainPID", "0")),
                    job_id=0 if job is None else int(job["id"]),
                    properties=props,
                )
            if not fresh and self.io.exists(
                Path(self.plan["units"][name]["installed_path"])
            ):
                checked_io, checked_host = self._read_context()
                unit, _ = checked_host.snapshot(name, deadline)
                require(
                    unit.dead and unit.job is None and not unit.enabled,
                    "site-stopped-source-check",
                )
                row.update(
                    active=unit.active,
                    main_pid=unit.main.pid if unit.main else 0,
                    job_id=0,
                    members=[asdict(member) for member in unit.members],
                    checked_stage=checked_io.raw_units[name],
                )
            result[name] = row
        return result

    def fresh(self, names, deadline):
        return self._observe(names, deadline, fresh=True)

    def inert(self, names, deadline):
        return self._observe(names, deadline, fresh=False)

    def reload(self, deadline):
        self.io.command(["systemctl", "daemon-reload"], deadline)

    def verify_sources(self, prepared, deadline):
        render.q.verify_sources(prepared.plan, self.io, self._log(), deadline)

    def start(self, name, deadline):
        role = self.plan["units"].get(name, {}).get("role")
        require(
            role in {"watchdog", "observer", "publisher", "dispatcher"},
            "site-fixed-start-role",
        )
        io, host = self._read_context()
        unit, _ = host.snapshot(name, deadline)
        require(
            unit.dead
            and unit.job is None
            and not unit.enabled
            and self.io.boot_links(name, deadline) == {},
            "site-start-not-inert",
        )
        if role != "watchdog":
            require(self.arm_pin is not None, "site-start-before-arm")
            self.recheck_arm(self.arm_pin, deadline)
        self.io.command(["systemctl", "start", "--no-block", name], deadline)

    def arm(self, prepared, deadline):
        while self.io.now() < deadline:
            io, host = self._read_context()
            watchdog = prepared.start_order[0]
            unit, _ = host.snapshot(watchdog, deadline)
            if (
                unit.active == "active"
                and unit.substate == "waiting"
                and unit.job is None
            ):
                self.arm_pin = self.driver.arm_receipt(
                    prepared.plan, prepared.anchor, self.io, self._log()
                )
                return self.arm_pin
            require(
                unit.active not in {"failed", "deactivating"}, "site-watchdog-failed"
            )
            self.io.sleep(min(0.1, max(0, deadline - self.io.now())))
        raise InstallRefusal("site-watchdog-arm-timeout")

    def recheck_arm(self, pin, deadline):
        require(self.io.now() < deadline, "site-arm-deadline")
        io, _ = self._read_context()
        io._arm = pin
        self.driver.verify_arm_health(io)

    def started(self, name, deadline):
        q = render.q
        role = self.plan["units"][name]["role"]
        while self.io.now() < deadline:
            io, host = self._read_context()
            unit, _ = host.snapshot(name, deadline)
            if unit.active == "active" and unit.main is not None and unit.job is None:
                text = self.io.command(
                    [
                        "systemctl",
                        "show",
                        name,
                        "--all",
                        "--no-pager",
                        "--property=TimeoutStartUSec,TimeoutStopUSec",
                    ],
                    deadline,
                )
                values = dict(line.split("=", 1) for line in text.splitlines())
                require(
                    set(values) == {"TimeoutStartUSec", "TimeoutStopUSec"},
                    "site-control-timeouts",
                )
                q.validate_actor_bounds(
                    role,
                    unit,
                    io.raw_units[name],
                    values["TimeoutStartUSec"],
                    self.anchor,
                )
                return {"unit": asdict(unit), "observed_clock": asdict(self.clock())}
            require(
                unit.active not in {"failed", "deactivating"}
                and (unit.active != "inactive" or unit.job is not None),
                "site-control-exited",
            )
            self.io.sleep(min(0.1, max(0, deadline - self.io.now())))
        raise InstallRefusal("site-control-start-timeout")

    def inspect_cleanup(self, prepared, deadline):
        require(self.io.now() < deadline, "site-cleanup-deadline")
        log = self._log()
        found = self.driver.entries(log, "cleanup-complete")
        if not found:
            return {"status": "pending", "plan_sha256": prepared.plan.checksum}
        require(len(found) == 1, "site-cleanup-proof-count")
        checked = life._checked_cleanup(log, found[0][0])
        expected = {
            name
            for name, spec in self.plan["units"].items()
            if spec["role"] == "workload"
        }
        require(
            set(checked["expected_workloads"]) == expected
            and len(checked["expected_workloads"]) == len(expected),
            "site-cleanup-workloads",
        )
        require(self.io.now() < deadline, "site-cleanup-deadline")
        return {
            "status": "cleanup-observed",
            "cleanup_pin": found[0][0],
            "clock": checked["clock"],
            "plan_sha256": prepared.plan.checksum,
        }

    def audit(self, prepared, deadline):
        q = render.q
        require(self.io.now() < deadline, "site-audit-deadline")
        io, host = self._read_context()
        publisher = next(
            name
            for name, spec in self.plan["units"].items()
            if spec["role"] == "publisher"
        )
        unit, _ = host.snapshot(publisher, deadline)
        context = q.AuthorizedContext(
            prepared.plan,
            prepared.anchor,
            self._log(),
            io,
            host,
            unit,
            {"role": "external-read-only-auditor"},
        )
        return self.driver.external_audit(context)
