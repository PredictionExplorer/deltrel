"""Fixed, admission-gated external composition; never a qualification writer.

The independently approved intent must name real session/bootstrap evidence.
No test certificate, missing certificate, or self-computed manifest grants live
authority. The module imports only the standard library and pure helpers.
"""

from __future__ import annotations

import argparse
import base64
import copy
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import math
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import time
from typing import Any

from scripts import strength_freshness_cpu_completion as completion
from scripts import strength_freshness_cpu_outer as outer

FRAME = "strength-freshness-outer-role-frame-v1"
ADMISSION = "strength-freshness-outer-runtime-v1"
SESSION = "strength-freshness-real-session-qualification-v1"
SESSION_CHECKS = {
    "caller_death_survival",
    "setsid_descendant_reaping",
    "guardian_death_adoption",
    "operator_death_adoption",
    "source_bootstrap",
    "resource_limits",
    "cross_uid_metadata_access",
}
HELPER_MODES = outer.HELPER_MODES
MAX_FRAME = 65536
MAX_PAYLOAD = 32768
REQUIRED_SOURCE_NAMES = {
    "strength_freshness_cpu_outer.py",
    "strength_freshness_cpu_outer_runtime.py",
    "strength_freshness_cpu_completion.py",
    "strength_freshness_cpu_install_runtime.py",
    "strength_freshness_cpu_install_helper.py",
    "strength_freshness_cpu_install_backend.py",
    "strength_freshness_cpu_template_finalization.py",
}


class RuntimeRefusal(RuntimeError):
    """Fixed public reason; raw process/environment/input errors stay private."""


def require(ok: object, reason: str) -> None:
    if not ok:
        raise RuntimeRefusal(reason)


def encode(value: object) -> bytes:
    return completion.encoded(value)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def shape(value: Any, fields: set[str], reason: str) -> dict:
    require(isinstance(value, dict) and set(value) == fields, reason)
    return value


def clock(raw: dict) -> outer.Clock:
    completion.clock(raw)
    return outer.Clock(**raw)


def actual_clock() -> outer.Clock:
    return outer.Clock(
        Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        time.monotonic_ns(),
        time.time_ns(),
    )


def resource_policy() -> dict:
    return {
        "address_space_tuples": {
            "supervisor": [512 * 2**20, 16 * 2**30],
            "operator": [512 * 2**20, 16 * 2**30],
            "guardian": [512 * 2**20, 512 * 2**20],
            "collector": [512 * 2**20, 512 * 2**20],
            "helper": [16 * 2**30, 16 * 2**30],
        },
        "capture_contract_scalar_means_both_limits": dict(outer.CAPTURE_LIMITS),
        "helper_output_bytes": 8 * 2**20,
        "frame_fd": 127,
        "frame_bytes": MAX_FRAME,
        "helper_payload_bytes": MAX_PAYLOAD,
    }


def verify_resources(role: str, *, apply: bool = False) -> None:
    require(role in resource_policy()["address_space_tuples"], "resource-role")
    expected = tuple(resource_policy()["address_space_tuples"][role])
    if apply:
        inherited_hard = resource.getrlimit(resource.RLIMIT_AS)[1]
        require(
            inherited_hard == resource.RLIM_INFINITY or inherited_hard >= expected[1],
            "inherited-address-space-ceiling",
        )
        getattr(os, "sched_setaffinity")(0, {min(getattr(os, "sched_getaffinity")(0))})
        os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
        resource.setrlimit(resource.RLIMIT_AS, expected)
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
        resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 2**20, 32 * 2**20))
    require(
        resource.getrlimit(resource.RLIMIT_AS) == expected
        and resource.getrlimit(resource.RLIMIT_NOFILE) == (128, 128)
        and resource.getrlimit(resource.RLIMIT_FSIZE) == (32 * 2**20, 32 * 2**20)
        and len(getattr(os, "sched_getaffinity")(0)) == 1
        and os.getpriority(os.PRIO_PROCESS, 0) == 19,
        "actual-resource-policy",
    )


def guardian_contract_sha256(
    python: str,
    control_root: str,
    authorization_path: str,
    qualified_sources: tuple[str, str],
) -> str:
    """Program contract only; actual approved SHA is separately exact at launch."""
    return outer.guardian_contract_sha256(
        python, control_root, authorization_path, qualified_sources
    )


def dummy_start_ack(**expected) -> dict:
    return outer.dummy_start_ack(**expected)


def validate_dummy_start_ack(value: dict, **expected) -> dict:
    return outer.validate_dummy_start_ack(value, **expected)


def _path(value: str) -> Path:
    return Path(str(completion.path(value)))


def _pin(value: dict, maximum: int = 2**20) -> dict:
    return completion.pin(value, maximum)


INTERPRETER_STAT_FIELDS = {
    "uid",
    "gid",
    "mode",
    "device",
    "inode",
    "size",
    "mtime_ns",
    "ctime_ns",
}


def file_identity(st) -> dict:
    require(stat.S_ISREG(st.st_mode), "qualified-file-type")
    return {
        "uid": st.st_uid,
        "gid": st.st_gid,
        "mode": stat.S_IMODE(st.st_mode),
        "device": st.st_dev,
        "inode": st.st_ino,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "ctime_ns": st.st_ctime_ns,
    }


def validate_interpreter_metadata(value: dict, size: int) -> dict:
    shape(value, INTERPRETER_STAT_FIELDS, "interpreter-metadata-fields")
    require(
        all(type(v) is int and v >= 0 for v in value.values())
        and value["size"] == size
        and value["mode"] <= 0o777
        and value["mode"] & 0o111 != 0,
        "interpreter-metadata",
    )
    return value


def current_interpreter(reader, registered: dict, *, hash_image: bool = True) -> None:
    """Exact registered mutable host image; never an immutable-interpreter claim."""
    value = shape(
        registered,
        {"path", "resolved_path", "sha256", "bytes", "metadata"},
        "runtime-python-pin",
    )
    metadata = validate_interpreter_metadata(value["metadata"], value["bytes"])
    literal, resolved = _path(value["path"]), _path(value["resolved_path"])
    require(
        literal.name == "python"
        and str(Path(sys.executable).resolve()) == str(resolved)
        and literal.resolve() == resolved
        and resolved.resolve() == resolved,
        "runtime-interpreter-origin",
    )
    # Fixed proc link joins the running image, including deleted/replaced-image refusal.
    require(
        os.readlink("/proc/self/exe") == str(resolved), "runtime-executed-image-path"
    )
    fd = os.open("/proc/self/exe", os.O_RDONLY | os.O_NONBLOCK)
    try:
        before = file_identity(os.fstat(fd))
        require(
            before == metadata and file_identity(resolved.stat()) == metadata,
            "runtime-executed-image-identity",
        )
        if not hash_image:
            reader.check()
            require(
                file_identity(resolved.stat()) == metadata, "runtime-launch-image-drift"
            )
            return
        digest = hashlib.sha256()
        count = 0
        while True:
            reader.check()
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            count += len(chunk)
            require(
                count <= value["bytes"] <= 64 * 2**20, "runtime-executed-image-bound"
            )
            digest.update(chunk)
        require(
            count == value["bytes"]
            and digest.hexdigest() == value["sha256"]
            and file_identity(os.fstat(fd)) == before
            and file_identity(resolved.stat()) == before
            and os.readlink("/proc/self/exe") == str(resolved),
            "runtime-executed-image-drift",
        )
        reader.check()
    finally:
        os.close(fd)
    reader.read(
        {k: value[k] for k in ("sha256", "bytes")} | {"path": str(resolved)},
        source=True,
        maximum=64 * 2**20,
        interpreter=True,
        expected_metadata=metadata,
    )


def current_site_inventory(reader, site: dict) -> None:
    """Bounded startup hashes plus explicitly cached large-image identity joins."""
    require(
        isinstance(site.get("site_inventory_pin"), dict),
        "helper-site-inventory-missing",
    )
    inventory_pin = _pin(site["site_inventory_pin"], 2**20)
    inventory = completion.parse(reader.read(inventory_pin))
    shape(
        inventory,
        {
            "format",
            "schema_version",
            "root",
            "startup_directories",
            "startup_files",
            "cached_files",
        },
        "helper-site-inventory-fields",
    )
    require(
        inventory["format"] == "strength-freshness-qualified-site-files-v1"
        and type(inventory["schema_version"]) is int
        and inventory["schema_version"] == 1,
        "helper-site-inventory-format",
    )
    root = _path(inventory["root"])
    directories = inventory["startup_directories"]
    require(
        isinstance(directories, list) and 1 <= len(directories) <= 64,
        "helper-site-directory-count",
    )
    visited = set()
    for item in directories:
        shape(item, {"path", "entries"}, "helper-site-directory-fields")
        path = _path(item["path"])
        require(
            path.is_relative_to(root)
            and path not in visited
            and path.resolve() == path
            and isinstance(item["entries"], list)
            and len(item["entries"]) <= 4096,
            "helper-site-directory-scope",
        )
        visited.add(path)
        reader.check()
        names = []
        with os.scandir(path) as entries:
            for entry in entries:
                names.append(entry.name)
                require(len(names) <= 4096, "helper-site-directory-bound")
                reader.check()
        require(
            sorted(names) == item["entries"], "helper-site-startup-addition-or-removal"
        )
    seen = set()
    fresh_bytes = 0
    for key, limit in (("startup_files", 256), ("cached_files", 4096)):
        files = inventory[key]
        require(
            isinstance(files, list) and 1 <= len(files) <= limit,
            "helper-site-file-count",
        )
        for item in files:
            shape(
                item,
                {"path", "sha256", "metadata"}
                | ({"cache_receipt"} if key == "cached_files" else set()),
                "helper-site-file-fields",
            )
            path = _path(item["path"])
            metadata = shape(
                item["metadata"], INTERPRETER_STAT_FIELDS, "helper-site-file-metadata"
            )
            require(
                all(type(v) is int and v >= 0 for v in metadata.values())
                and path.is_relative_to(root)
                and path not in seen
                and completion.checksum(item["sha256"]),
                "helper-site-file-scope",
            )
            seen.add(path)
            reader.check()
            require(
                path.resolve() == path and file_identity(path.stat()) == metadata,
                "helper-site-current-file-drift",
            )
            if key == "startup_files":
                fresh_bytes += metadata["size"]
                require(
                    fresh_bytes <= 8 * 2**20 and path.parent in visited,
                    "helper-site-startup-budget",
                )
                reader.read(
                    {
                        "path": str(path),
                        "sha256": item["sha256"],
                        "bytes": metadata["size"],
                    },
                    source=True,
                    maximum=8 * 2**20,
                    qualified_metadata=metadata,
                )
            else:
                cache = completion.parse(reader.read(_pin(item["cache_receipt"])))
                require(
                    cache.get("format") == "strength-freshness-qualified-file-cache-v1"
                    and cache.get("status") == "independently-qualified"
                    and cache.get("evidence_kind") == "real-host-content-hash"
                    and cache.get("synthetic") is False
                    and cache.get("path") == str(path)
                    and cache.get("sha256") == item["sha256"]
                    and cache.get("metadata") == metadata,
                    "helper-site-cache-unqualified",
                )
                # No payload read: this is a current exact inode/stat join to a
                # separately approved content hash, explicitly not a fresh hash.
            require(file_identity(path.stat()) == metadata, "helper-site-file-raced")
            reader.check()


class ProtectedStore:
    """Bounded pinned reads and exact-root no-clobber publication only."""

    def __init__(
        self,
        *,
        roots: tuple[Path, ...],
        deadline_ns: int,
        owner_uid: int = 0,
        monotonic_ns=time.monotonic_ns,
    ):
        self.roots = roots
        self.deadline_ns = deadline_ns
        self.owner_uid = owner_uid
        self.monotonic_ns = monotonic_ns
        self.consumed = 0
        self.interpreter_consumed = 0
        self.original_clock: outer.Clock | None = None
        self.deadline_wall_ns: int | None = None
        self.clock_source = actual_clock
        self.last_clock: outer.Clock | None = None

    def bind_deadline(
        self, original: outer.Clock, mono: int, wall: int, clock_source
    ) -> None:
        require(
            mono > original.monotonic_ns and wall > original.wall_ns,
            "store-original-budget",
        )
        if self.original_clock is not None:
            require(
                original.boot_id == self.original_clock.boot_id, "store-budget-boot"
            )
        self.original_clock = original
        self.deadline_ns, self.deadline_wall_ns = mono, wall
        self.clock_source = clock_source
        self.check()

    def check(self) -> None:
        if self.original_clock is None:
            # Bootstrap can only read the approved intent under this short bound.
            require(self.monotonic_ns() < self.deadline_ns, "store-deadline")
            return
        now = self.clock_source()
        prior = self.last_clock or self.original_clock
        require(
            now.boot_id == self.original_clock.boot_id
            and now.monotonic_ns >= prior.monotonic_ns
            and now.wall_ns >= prior.wall_ns,
            "store-clock-drift",
        )
        require(
            now.monotonic_ns < self.deadline_ns
            and self.deadline_wall_ns is not None
            and now.wall_ns < self.deadline_wall_ns,
            "store-deadline",
        )
        self.last_clock = now

    def directory(self, path: Path) -> None:
        self.check()
        require(path in self.roots and path.resolve() == path, "store-root")
        st = path.lstat()
        require(
            stat.S_ISDIR(st.st_mode)
            and st.st_uid == self.owner_uid
            and stat.S_IMODE(st.st_mode) == 0o700,
            "store-root-protection",
        )
        self.check()

    def read(
        self,
        pin: dict,
        *,
        source: bool = False,
        maximum: int = 2**20,
        interpreter: bool = False,
        expected_metadata: dict | None = None,
        qualified_metadata: dict | None = None,
    ) -> bytes:
        p = _pin(pin, maximum)
        path = _path(p["path"])
        self.check()
        require(path.resolve() == path, "store-input-alias")
        if not source:
            self.directory(path.parent)
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        try:
            before = os.fstat(fd)
            require(
                stat.S_ISREG(before.st_mode)
                and (
                    (
                        interpreter
                        and expected_metadata is not None
                        and file_identity(before) == expected_metadata
                    )
                    or (
                        source
                        and qualified_metadata is not None
                        and file_identity(before) == qualified_metadata
                    )
                    or (
                        expected_metadata is None
                        and qualified_metadata is None
                        and before.st_uid == self.owner_uid
                        and stat.S_IMODE(before.st_mode)
                        in ((0o444, 0o555) if source else (0o444,))
                    )
                )
                and before.st_size == p["bytes"],
                "store-input-protection",
            )
            chunks = []
            count = 0
            while count <= p["bytes"]:
                self.check()
                b = os.read(fd, min(65536, p["bytes"] + 1 - count))
                if not b:
                    break
                count += len(b)
                if interpreter:
                    self.interpreter_consumed += len(b)
                    require(
                        source
                        and maximum == 64 * 2**20
                        and self.interpreter_consumed <= 64 * 2**20,
                        "interpreter-byte-budget",
                    )
                else:
                    self.consumed += len(b)
                    require(self.consumed <= 32 * 2**20, "store-byte-budget")
                chunks.append(b)
            after = os.fstat(fd)
        finally:
            os.close(fd)
        require(
            (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ),
            "store-input-raced",
        )
        data = b"".join(chunks)
        require(
            len(data) == p["bytes"] and sha(data) == p["sha256"], "store-input-hash"
        )
        self.check()
        return data

    def fixed(self, path: Path, maximum: int = 2**20) -> tuple[dict, bytes]:
        """Only fixed producer outputs; not an arbitrary receipt-supplied path."""
        self.directory(path.parent)
        st = path.lstat()
        require(
            stat.S_ISREG(st.st_mode)
            and 0 < st.st_size <= maximum
            and st.st_uid == self.owner_uid
            and stat.S_IMODE(st.st_mode) == 0o444,
            "fixed-output-protection",
        )
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        try:
            raw = os.read(fd, maximum + 1)
        finally:
            os.close(fd)
        p = {"path": str(path), "sha256": sha(raw), "bytes": len(raw)}
        return p, self.read(p, maximum=maximum)

    def ensure_directory(self, path: Path, *, parent: Path) -> None:
        self.directory(parent)
        require(path.parent == parent, "output-directory-scope")
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        self.roots = tuple(set(self.roots) | {path})
        self.directory(path)

    def exists(self, path: Path) -> bool:
        self.directory(path.parent)
        present = path.exists() or path.is_symlink()
        self.check()
        return present

    def directory_empty(self, path: Path) -> bool:
        self.directory(path)
        empty = not any(path.iterdir())
        self.check()
        return empty

    def publish(self, path: Path, raw: bytes, *, maximum: int = 2**20) -> dict:
        require(
            type(raw) is bytes
            and 0 < len(raw) <= maximum
            and path.name not in ("", ".", ".."),
            "store-output-bound",
        )
        require(self.original_clock is not None, "publication-before-admission")
        self.directory(path.parent)
        self.check()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        temporary = "." + path.name + ".publishing"
        try:
            st = os.fstat(directory)
            require(
                (st.st_dev, st.st_ino)
                == (path.parent.stat().st_dev, path.parent.stat().st_ino),
                "store-parent-raced",
            )
            fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o400,
                dir_fd=directory,
            )
            with os.fdopen(fd, "wb") as stream:
                for offset in range(0, len(raw), 65536):
                    self.check()
                    stream.write(raw[offset : offset + 65536])
                stream.flush()
                os.fchmod(stream.fileno(), 0o444)
                os.fsync(stream.fileno())
            self.check()
            os.link(
                temporary,
                path.name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
            os.fsync(directory)
            os.unlink(temporary, dir_fd=directory)
            os.fsync(directory)
        finally:
            os.close(directory)
        self.check()
        return {"path": str(path), "sha256": sha(raw), "bytes": len(raw)}


def read_sealed_frame() -> dict:
    require(sys.platform == "linux", "linux-frame-required")
    seals = sum(
        getattr(fcntl, x)
        for x in ("F_SEAL_WRITE", "F_SEAL_GROW", "F_SEAL_SHRINK", "F_SEAL_SEAL")
    )
    require(
        fcntl.fcntl(127, getattr(fcntl, "F_GET_SEALS")) == seals, "unsealed-role-frame"
    )
    st = os.fstat(127)
    require(stat.S_ISREG(st.st_mode) and 0 < st.st_size <= MAX_FRAME, "role-frame-size")
    os.lseek(127, 0, os.SEEK_SET)
    data = os.read(127, MAX_FRAME + 1)
    require(len(data) == st.st_size, "role-frame-read")
    os.close(127)
    return completion.parse(data, MAX_FRAME)


def validate_frame(
    frame: dict, *, role: str, authorization_sha256: str, nonce: str
) -> dict:
    common = {
        "format",
        "schema_version",
        "role",
        "nonce",
        "authorization_sha256",
        "parent",
        "phase_start",
    }
    extras = (
        {"supervisor", "phase", "budget", "launch", "anchor"}
        if role == "guardian"
        else {"supervisor"}
        if role == "operator"
        else {
            "operation",
            "work_deadline",
            "cleanup_deadline",
            "payload_b64",
            "payload_sha256",
            "payload_bytes",
        }
    )
    shape(frame, common | extras, "role-frame-fields")
    require(
        frame["format"] == FRAME
        and type(frame["schema_version"]) is int
        and frame["schema_version"] == 1
        and frame["role"] == role
        and frame["authorization_sha256"] == authorization_sha256
        and frame["nonce"] == nonce,
        "role-frame-binding",
    )
    completion.identity(frame["parent"])
    completion.clock(frame["phase_start"])
    if role in {"operator", "guardian"}:
        completion.identity(frame["supervisor"])
    if role == "guardian":
        require(frame["phase"] in {"before", "after"}, "guardian-phase")
        expected = (
            completion.before_budget(frame["phase_start"])
            if frame["phase"] == "before"
            else completion.after_budget(frame["phase_start"], frame["anchor"])
        )
        require(frame["budget"] == expected, "guardian-original-budget")
        _pin(frame["launch"])
    if role == "helper":
        require(
            frame["operation"] in HELPER_MODES
            and type(frame["payload_bytes"]) is int
            and 0 < frame["payload_bytes"] <= MAX_PAYLOAD
            and completion.checksum(frame["payload_sha256"]),
            "helper-payload-fields",
        )
        for name in ("work_deadline", "cleanup_deadline"):
            shape(frame[name], {"monotonic_ns", "wall_ns"}, "helper-deadline-fields")
            require(
                all(type(v) is int and v > 0 for v in frame[name].values()),
                "helper-deadline-type",
            )
        require(
            all(
                frame["cleanup_deadline"][axis] - frame["work_deadline"][axis]
                == 2 * outer.SECOND
                for axis in ("monotonic_ns", "wall_ns")
            ),
            "helper-cleanup-reserve",
        )
    return copy.deepcopy(frame)


def decode_payload(frame: dict) -> bytes:
    try:
        raw = base64.b64decode(frame["payload_b64"], validate=True)
    except (ValueError, TypeError):
        raise RuntimeRefusal("helper-payload-encoding") from None
    require(
        len(raw) == frame["payload_bytes"] <= MAX_PAYLOAD
        and sha(raw) == frame["payload_sha256"],
        "helper-payload-pin",
    )
    return raw


@dataclass
class Admission:
    raw_intent: bytes
    approved_intent_sha256: str
    intent: dict
    outer: dict
    identity: outer.ProcessIdentity
    parent: outer.Handle
    kernel: Any
    reader: ProtectedStore
    session_pin: dict
    role: str

    @property
    def preflight_start(self) -> outer.Clock:
        return clock(self.outer["preflight_start"])

    @property
    def source_sha256(self) -> str:
        return next(
            p["sha256"]
            for p in self.outer["source_pins"]
            if p["path"]
            == str(
                Path(self.outer["control_root"])
                / "scripts/strength_freshness_cpu_outer.py"
            )
        )

    @property
    def role_source_hashes(self) -> tuple[str, str]:
        runtime = str(
            Path(self.outer["control_root"])
            / "scripts/strength_freshness_cpu_outer_runtime.py"
        )
        return self.source_sha256, next(
            p["sha256"] for p in self.outer["source_pins"] if p["path"] == runtime
        )

    @property
    def intent_pin(self) -> dict:
        return {
            "path": self.outer["authorization_path"],
            "sha256": self.approved_intent_sha256,
            "bytes": len(self.raw_intent),
        }


def validate_parent_context(
    own, parent_identity, expected_parent: dict, caller: dict
) -> None:
    require(
        asdict(parent_identity) == expected_parent
        and own.ppid == parent_identity.pid
        and own.boot_id == parent_identity.boot_id
        and own.pid_namespace_inode == parent_identity.pid_namespace_inode
        and own.cgroup == parent_identity.cgroup == caller["cgroup"]
        and own.uid == parent_identity.uid == caller["uid"] == 0
        and parent_identity.start_ticks <= own.start_ticks,
        "runtime-parent-lifetime",
    )


def load_admission(
    authorization_path: str,
    approved_sha256: str,
    *,
    role: str,
    frame: dict | None = None,
    kernel=None,
    store=None,
) -> Admission:
    require(
        role in {"supervisor", "operator", "guardian", "helper"}
        and completion.checksum(approved_sha256),
        "runtime-admission-arguments",
    )
    require(
        sys.platform == "linux" and os.geteuid() == 0, "runtime-root-linux-required"
    )
    require(
        sys.flags.dont_write_bytecode
        and (
            sys.flags.no_user_site and not sys.flags.no_site
            if role == "helper"
            else sys.flags.no_site and sys.flags.ignore_environment
        ),
        "runtime-bootstrap-flags",
    )
    k = outer.LinuxKernel() if kernel is None else kernel
    now = k.clock()
    path = _path(authorization_path)
    reader = (
        ProtectedStore(
            roots=(path.parent,), deadline_ns=now.monotonic_ns + 10 * outer.SECOND
        )
        if store is None
        else store
    )
    raw = reader.read(
        {"path": str(path), "sha256": approved_sha256, "bytes": path.stat().st_size}
    )
    intent = completion.parse(raw)
    cfg = shape(
        intent.get("outer"),
        {
            "format",
            "schema_version",
            "nonce",
            "preflight_start",
            "input_root",
            "control_root",
            "python",
            "source_pins",
            "session_qualification",
            "helper_site_qualification",
            "caller",
            "before_launch",
            "before_request",
            "registration",
            "verified_champions",
            "resource_policy_sha256",
        },
        "outer-intent-fields",
    )
    require(
        cfg["format"] == ADMISSION
        and cfg["schema_version"] == 1
        and re_nonce(cfg["nonce"]),
        "outer-intent-format",
    )
    require(
        intent.get("nonce") == cfg["nonce"]
        and intent.get("boot_id") == cfg["preflight_start"]["boot_id"]
        and cfg["resource_policy_sha256"] == outer.digest(resource_policy()),
        "outer-top-level-binding",
    )
    start = clock(cfg["preflight_start"])
    require(
        now.boot_id == start.boot_id
        and now.monotonic_ns >= start.monotonic_ns
        and now.wall_ns >= start.wall_ns,
        "admission-original-clock",
    )
    limit = 120 if role == "supervisor" else 720
    require(
        now.monotonic_ns < start.monotonic_ns + limit * outer.SECOND
        and now.wall_ns < start.wall_ns + limit * outer.SECOND,
        "admission-expired",
    )
    reader.bind_deadline(
        start,
        min(reader.deadline_ns, start.monotonic_ns + limit * outer.SECOND),
        min(now.wall_ns + 10 * outer.SECOND, start.wall_ns + limit * outer.SECOND),
        k.clock,
    )
    require(_path(cfg["input_root"]) == path.parent, "runtime-input-root")
    own = k.self_identity()
    require(own.uid == 0 and k.task_ids() == (own.pid,), "runtime-current-owner")
    if role != "supervisor":
        require(frame is not None, "inherited-frame-required")
        assert frame is not None
    if role == "helper":
        assert frame is not None
        validate_frame(
            frame, role=role, authorization_sha256=approved_sha256, nonce=cfg["nonce"]
        )
        assert reader.deadline_wall_ns is not None
        reader.bind_deadline(
            start,
            min(reader.deadline_ns, frame["work_deadline"]["monotonic_ns"]),
            min(reader.deadline_wall_ns, frame["work_deadline"]["wall_ns"]),
            k.clock,
        )
    if role == "supervisor":
        expected_parent = completion.identity(cfg["caller"])
    else:
        assert frame is not None
        expected_parent = completion.identity(
            validate_frame(
                frame,
                role=role,
                authorization_sha256=approved_sha256,
                nonce=cfg["nonce"],
            )["parent"]
        )
    parent_identity = k.identity(expected_parent["pid"])
    validate_parent_context(own, parent_identity, expected_parent, cfg["caller"])
    parent_fd = k.open_pidfd(parent_identity.pid)
    try:
        require(
            k.pidfd_pid(parent_fd) == parent_identity.pid
            and not k.exited(parent_fd)
            and k.identity(parent_identity.pid) == parent_identity,
            "runtime-parent-raced",
        )
        require(
            isinstance(cfg["source_pins"], list) and len(cfg["source_pins"]) <= 128,
            "runtime-source-count",
        )
        sources = {_pin(p)["path"]: p for p in cfg["source_pins"]}
        require(
            len(sources) == len(cfg["source_pins"])
            and {
                str(_path(cfg["control_root"]) / "scripts" / name)
                for name in REQUIRED_SOURCE_NAMES
            }
            <= set(sources),
            "runtime-source-closure",
        )
        control = _path(cfg["control_root"])
        require(
            control.resolve() == control and os.getcwd() == str(control),
            "runtime-control-origin",
        )
        for module in (sys.modules[__name__], outer, completion):
            filename = module.__file__
            require(isinstance(filename, str), "runtime-module-file")
            assert isinstance(filename, str)
            actual = str(Path(filename).resolve())
            require(
                actual in sources and Path(actual).is_relative_to(control),
                "runtime-module-origin",
            )
        for item in sources.values():
            require(_path(item["path"]).is_relative_to(control), "runtime-source-root")
            reader.read(item, source=True)
        interpreter = cfg["python"]
        current_interpreter(reader, interpreter)
        require(
            cfg["session_qualification"] is not None,
            "real-session-qualification-missing",
        )
        session_pin = _pin(cfg["session_qualification"])
        session_root = _path(session_pin["path"]).parent
        reader.roots = tuple(set(reader.roots) | {session_root})
        session = completion.parse(reader.read(session_pin))
        require(
            session.get("format") == SESSION
            and session.get("schema_version") == 1
            and session.get("status") == "passed-real-linux"
            and session.get("evidence_kind") == "real-linux-process"
            and session.get("synthetic") is False,
            "real-session-qualification-required",
        )
        require(
            session.get("boot_id") == own.boot_id
            and session.get("source_closure_sha256") == outer.digest(cfg["source_pins"])
            and session.get("python_sha256") == interpreter["sha256"]
            and session.get("resource_policy_sha256") == cfg["resource_policy_sha256"]
            and session.get("caller_cgroup") == cfg["caller"]["cgroup"]
            and session.get("pid_namespace_inode") == own.pid_namespace_inode,
            "session-qualification-binding",
        )
        require(
            set(session.get("checks", {})) == SESSION_CHECKS
            and all(v is True for v in session["checks"].values()),
            "session-qualification-checks",
        )
        require(
            isinstance(session.get("raw_evidence_pins"), list)
            and 1 <= len(session["raw_evidence_pins"]) <= 16,
            "session-evidence-required",
        )
        for item in session["raw_evidence_pins"]:
            reader.read(_pin(item))
        require(
            cfg["helper_site_qualification"] is not None,
            "real-helper-site-qualification-missing",
        )
        site_pin = _pin(cfg["helper_site_qualification"])
        reader.roots = tuple(set(reader.roots) | {_path(site_pin["path"]).parent})
        site = completion.parse(reader.read(site_pin))
        site_checks = {
            "site_enabled_import_closure",
            "no_gpu_initialization",
            "resource_limits",
            "actual_helper_origin",
            "no_user_site",
            "loader_environment",
        }
        helper_environment = {**outer.ENV, "PYTHONPATH": str(control)}
        require(
            site.get("format") == "strength-freshness-real-helper-site-qualification-v1"
            and site.get("schema_version") == 1
            and site.get("status") == "passed-real-linux"
            and site.get("evidence_kind") == "real-linux-process"
            and site.get("synthetic") is False,
            "real-helper-site-qualification-required",
        )
        require(
            site.get("boot_id") == own.boot_id
            and site.get("source_closure_sha256") == outer.digest(cfg["source_pins"])
            and site.get("python_sha256") == interpreter["sha256"]
            and site.get("resource_policy_sha256") == cfg["resource_policy_sha256"]
            and site.get("helper_environment_sha256")
            == outer.digest(helper_environment),
            "helper-site-qualification-binding",
        )
        require(
            set(site.get("checks", {})) == site_checks
            and all(v is True for v in site["checks"].values())
            and isinstance(site.get("raw_evidence_pins"), list)
            and 1 <= len(site["raw_evidence_pins"]) <= 16,
            "helper-site-qualification-checks",
        )
        for item in site["raw_evidence_pins"]:
            reader.read(_pin(item))
        current_site_inventory(reader, site)
        expected_env = {
            **outer.ENV,
            **({"PYTHONPATH": str(control)} if role == "helper" else {}),
        }
        require(
            all(os.environ.get(key) == value for key, value in expected_env.items()),
            "runtime-fixed-environment",
        )
        require(
            not any(
                value
                for key, value in os.environ.items()
                if key.startswith(("LD_", "DYLD_", "PYTHON"))
                and not (role == "helper" and key == "PYTHONPATH")
            ),
            "runtime-loader-override",
        )
        if role != "helper":
            require(
                not any(
                    name == "torch"
                    or name.startswith(("torch.", "deltreltrain", "startrain"))
                    for name in sys.modules
                ),
                "runtime-heavy-import",
            )
        copied_cfg = copy.deepcopy(cfg)
        copied_cfg["authorization_path"] = authorization_path
        verify_resources(role, apply=role == "supervisor")
        admitted = Admission(
            raw,
            approved_sha256,
            copy.deepcopy(intent),
            copied_cfg,
            own,
            outer.Handle(parent_identity, parent_fd),
            k,
            reader,
            session_pin,
            role,
        )
        return admitted
    except BaseException:
        k.close(parent_fd)
        raise


def re_nonce(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(c in "0123456789abcdef" for c in value)
    )


def helper_entry_admission(
    authorization_path: str, approved_sha256: str, operation: str
):
    require(operation in HELPER_MODES, "helper-operation")
    frame = read_sealed_frame()
    admitted = load_admission(
        authorization_path, approved_sha256, role="helper", frame=frame
    )
    require(frame["operation"] == operation, "helper-operation-binding")
    store_budget(
        admitted,
        frame["work_deadline"]["monotonic_ns"],
        frame["work_deadline"]["wall_ns"],
    )
    return admitted, decode_payload(frame), frame


class AdmittedKernel(outer.LinuxKernel):
    """Only a fully checked Admission can arm one exact child contract."""

    def __init__(self, admission: Admission):
        super().__init__()
        self.admission = admission

    def admit_execution(
        self,
        binding: outer.Binding,
        spec: outer.CaptureSpec | outer.RoleSpec | outer.HelperSpec,
    ) -> None:
        a = self.admission
        require(
            self.self_identity() == a.identity and self.task_ids() == (a.identity.pid,),
            "admitted-kernel-owner",
        )
        require(
            binding.nonce == a.outer["nonce"]
            and binding.intent_sha256 == a.approved_intent_sha256
            and binding.source_sha256 == a.source_sha256
            and binding.session_admission_sha256 == a.session_pin["sha256"],
            "admitted-child-binding",
        )
        require(
            spec.python == a.outer["python"]["path"]
            and spec.control_root == a.outer["control_root"],
            "admitted-exec-origin",
        )
        if isinstance(spec, outer.CaptureSpec):
            require(a.role == "guardian", "collector-parent-role")
            root = (
                Path(a.outer["input_root"])
                if spec.phase == "before"
                else Path("/run")
                / ("edgeconnect-cpuqual-" + a.outer["nonce"])
                / "external/capture-inputs"
            )
            require(Path(spec.launch_path).parent == root, "admitted-capture-path")
            if spec.phase == "before":
                require(
                    spec.launch_sha256 == a.outer["before_launch"]["sha256"]
                    and spec.launch_path == a.outer["before_launch"]["path"],
                    "admitted-before-launch",
                )
        else:
            require(
                spec.authorization_sha256 == a.approved_intent_sha256
                and Path(spec.authorization_path).parent == Path(a.outer["input_root"]),
                "admitted-role-authorization",
            )
            role = "helper" if isinstance(spec, outer.HelperSpec) else spec.role
            require(
                (a.role, role)
                in {
                    ("supervisor", "operator"),
                    ("operator", "guardian"),
                    ("operator", "helper"),
                },
                "fixed-parent-child-role",
            )
            frame = validate_frame(
                completion.parse(spec.frame, MAX_FRAME),
                role=role,
                authorization_sha256=a.approved_intent_sha256,
                nonce=a.outer["nonce"],
            )
            require(frame["parent"] == asdict(a.identity), "admitted-child-parent")
            if isinstance(spec, outer.RoleSpec):
                require(
                    spec.qualified_sources == a.role_source_hashes,
                    "admitted-role-source-contract",
                )
            if isinstance(spec, outer.HelperSpec):
                require(frame["operation"] == spec.mode, "admitted-helper-operation")
        if isinstance(spec, outer.HelperSpec):
            # Site initialization occurs before helper code can recheck itself.
            # The parent must authenticate this separate admission first.
            require(
                a.outer["helper_site_qualification"] is not None,
                "real-helper-site-qualification-missing",
            )
            a.reader.read(_pin(a.outer["helper_site_qualification"]))
            current_site_inventory(
                a.reader,
                completion.parse(
                    a.reader.read(_pin(a.outer["helper_site_qualification"]))
                ),
            )
        current_interpreter(a.reader, a.outer["python"], hash_image=False)
        self._admitted_exec = outer.digest(spec.contract())


def prepared_run(raw: bytes) -> dict:
    value = shape(
        completion.parse(raw),
        {
            "plan_pin",
            "anchor_pin",
            "after_path",
            "authorization_root",
            "cleanup_proof_root",
        },
        "prepared-run-fields",
    )
    _pin(value["plan_pin"])
    _pin(value["anchor_pin"])
    for key in ("after_path", "authorization_root", "cleanup_proof_root"):
        _path(value[key])
    return value


def cleanup_ready(raw: bytes) -> dict:
    value = shape(
        completion.parse(raw), {"cleanup_pin", "clock"}, "cleanup-ready-fields"
    )
    _pin(value["cleanup_pin"])
    clock(value["clock"])
    return value


def final_audit(raw: bytes) -> dict:
    value = shape(
        completion.parse(raw),
        {"status", "plan_sha256", "anchor_sha256", "evidence_pins"},
        "final-audit-fields",
    )
    require(
        value["status"] == "passed_cpu_scope"
        and completion.checksum(value["plan_sha256"])
        and completion.checksum(value["anchor_sha256"])
        and isinstance(value["evidence_pins"], dict)
        and 1 <= len(value["evidence_pins"]) <= 16,
        "final-audit-status",
    )
    for item in value["evidence_pins"].values():
        _pin(item)
    return value


def family_binding(a: Admission, budget, spec) -> outer.Binding:
    return outer.Binding(
        a.outer["nonce"],
        a.approved_intent_sha256,
        a.source_sha256,
        a.session_pin["sha256"],
        outer.digest(asdict(budget)),
        outer.digest(spec.contract()),
    )


def frame_base(a: Admission, role: str, phase_start: outer.Clock) -> dict:
    return {
        "format": FRAME,
        "schema_version": 1,
        "role": role,
        "nonce": a.outer["nonce"],
        "authorization_sha256": a.approved_intent_sha256,
        "parent": asdict(a.identity),
        "phase_start": asdict(phase_start),
    }


def guarded_child(
    a: Admission, spec, budget, *, handoff_check=None
) -> tuple[dict, bytes]:
    k = a.kernel
    proof = outer.initialize_role(
        k,
        "guardian"
        if a.role == "guardian"
        else "operator"
        if a.role == "operator"
        else "supervisor",
        a.identity,
    )
    family = outer.OwnedFamily(
        k,
        proof,
        budget,
        maximum_output=spec.contract()["resources"]["stdout_stderr_bytes"],
    )
    binding = family_binding(a, budget, spec)
    try:
        family.start_capture(spec, binding, a.parent)
    except BaseException:
        if family.spawn is not None:
            try:
                family.finish(a.parent)
            except BaseException:
                pass  # Unknown ownership never grants a signal; upper tier remains.
        raise
    try:
        result = family.finish(a.parent, handoff_check=handoff_check)
    except BaseException:
        family.start_failure = "composition-observation-refused"
        try:
            family.finish(a.parent)
        except BaseException:
            pass
        raise
    require(result["natural_complete"] is True, "owned-child-incomplete")
    return result, bytes(family.output["stdout"])


class HelperExecutor:
    """Internal fixed dependency for Installer; no JSON-selected callback."""

    def __init__(self, admission: Admission, authorization_path: str):
        self.admission = admission
        self.authorization_path = authorization_path
        self.dummy_start: outer.Clock | None = None
        self.evidence: list[dict] = []

    def bind_dummy_start(self, value: outer.Clock) -> None:
        require(self.dummy_start is None, "helper-dummy-start-already-bound")
        self.dummy_start = value

    def check_alive(self) -> None:
        a = self.admission
        now = a.kernel.clock()
        require(
            not a.kernel.exited(a.parent.pidfd)
            and a.kernel.pidfd_pid(a.parent.pidfd) == a.parent.identity.pid
            and a.kernel.identity(a.parent.identity.pid) == a.parent.identity,
            "operator-supervisor-lost",
        )
        require(
            now.boot_id == a.preflight_start.boot_id
            and now.monotonic_ns >= a.preflight_start.monotonic_ns
            and now.wall_ns >= a.preflight_start.wall_ns,
            "operator-clock-drift",
        )
        if self.dummy_start is not None:
            require(
                now.monotonic_ns
                < min(
                    self.dummy_start.monotonic_ns + 598 * outer.SECOND,
                    a.preflight_start.monotonic_ns + 718 * outer.SECOND,
                )
                and now.wall_ns
                < min(
                    self.dummy_start.wall_ns + 598 * outer.SECOND,
                    a.preflight_start.wall_ns + 718 * outer.SECOND,
                ),
                "operator-original-ceiling",
            )

    def call(self, mode: str, payload: bytes, deadline_ns: int) -> bytes:
        a = self.admission
        self.check_alive()
        require(
            mode in HELPER_MODES
            and type(payload) is bytes
            and 0 < len(payload) <= MAX_PAYLOAD
            and self.dummy_start is not None,
            "helper-call-scope",
        )
        assert self.dummy_start is not None
        now = a.kernel.clock()
        cap = 60 if mode == "prepare-install" else 10
        phase_end = (
            60
            if mode in {"prepare-install", "prearm-cleanup"}
            else 540
            if mode == "inspect-cleanup"
            else 598
        )
        require(
            type(deadline_ns) is int
            and now.monotonic_ns + 2 * outer.SECOND
            < deadline_ns
            <= min(
                now.monotonic_ns + cap * outer.SECOND,
                self.dummy_start.monotonic_ns + phase_end * outer.SECOND,
            ),
            "helper-original-deadline",
        )
        ceiling = outer.Clock(
            now.boot_id,
            deadline_ns,
            min(
                now.wall_ns + deadline_ns - now.monotonic_ns,
                self.dummy_start.wall_ns + phase_end * outer.SECOND,
            ),
        )
        budget = outer.HelperBudget(mode, self.dummy_start, now, ceiling)
        budget.validate()
        frame = {
            **frame_base(a, "helper", now),
            "operation": mode,
            "work_deadline": budget.compact()["work"],
            "cleanup_deadline": budget.compact()["cleanup"],
            "payload_b64": base64.b64encode(payload).decode(),
            "payload_sha256": sha(payload),
            "payload_bytes": len(payload),
        }
        spec = outer.HelperSpec(
            mode,
            a.outer["python"]["path"],
            a.outer["control_root"],
            self.authorization_path,
            a.approved_intent_sha256,
            budget.work_ns,
            encode(frame),
        )
        report, result = guarded_child(a, spec, budget)
        self.evidence.append(
            {"operation": mode, "family": report, "payload_sha256": sha(payload)}
        )
        return result


def store_budget(a: Admission, mono: int, wall: int) -> None:
    a.reader.bind_deadline(a.preflight_start, mono, wall, a.kernel.clock)


def handoff_clock(a: Admission) -> outer.Clock:
    now = a.kernel.clock()
    require(
        now.boot_id == a.preflight_start.boot_id
        and now.monotonic_ns >= a.preflight_start.monotonic_ns
        and now.wall_ns >= a.preflight_start.wall_ns
        and now.monotonic_ns < a.preflight_start.monotonic_ns + 118 * outer.SECOND
        and now.wall_ns < a.preflight_start.wall_ns + 118 * outer.SECOND
        and not a.kernel.exited(a.parent.pidfd),
        "dummy-handoff-deadline-or-parent",
    )
    return now


def publish_failure(
    a: Admission | None, role: str, reason: str = "runtime-refused"
) -> None:
    """No late writes or private error serialization; failure never grants pass."""
    if a is None:
        return
    try:
        now = a.kernel.clock()
        if (
            now.boot_id != a.preflight_start.boot_id
            or now.monotonic_ns >= a.preflight_start.monotonic_ns + 720 * outer.SECOND
        ):
            return
        root = Path(a.outer["input_root"])
        a.reader.publish(
            root / (role + "-failure.json"),
            encode(
                {
                    "format": "strength-freshness-outer-failure-v1",
                    "schema_version": 1,
                    "role": role,
                    "nonce": a.outer["nonce"],
                    "authorization_sha256": a.approved_intent_sha256,
                    "identity": asdict(a.identity),
                    "clock": asdict(now),
                    "reason": reason,
                    "status": "incomplete",
                }
            ),
            maximum=32768,
        )
    except BaseException:
        return


def phase_budget(frame: dict) -> outer.PhaseBudget:
    start = clock(frame["phase_start"])
    if frame["phase"] == "before":
        result = outer.PhaseBudget.before(start)
    else:
        anchor = frame["anchor"]
        result = outer.PhaseBudget.after(
            outer.Clock(
                anchor["boot_id"],
                math.floor(anchor["started_monotonic"] * outer.SECOND),
                anchor["started_wall_ns"],
            ),
            start,
        )
    require(result.compact() == frame["budget"], "phase-budget-binding")
    return result


def activate_kernel(a: Admission) -> None:
    # Test kernels are injected through the Python API only, never CLI input.
    if type(a.kernel) is outer.LinuxKernel:
        a.kernel = AdmittedKernel(a)


def run_guardian(a: Admission, frame: dict) -> dict:
    validate_frame(
        frame,
        role="guardian",
        authorization_sha256=a.approved_intent_sha256,
        nonce=a.outer["nonce"],
    )
    budget = phase_budget(frame)
    store_budget(a, budget.cleanup_ns, budget.compact()["cleanup"]["wall_ns"])
    activate_kernel(a)
    launch = _pin(frame["launch"])
    a.reader.read(launch)
    spec = outer.CaptureSpec(
        frame["phase"],
        a.outer["python"]["path"],
        a.outer["control_root"],
        launch["path"],
        launch["sha256"],
        budget.work_ns,
    )
    report, _ = guarded_child(a, spec, budget)
    return report


def bind_outputs(a: Admission, phase: str, launch_pin: dict) -> dict:
    launch_raw = a.reader.read(launch_pin)
    launch = completion.parse(launch_raw)
    parent = (
        Path(a.outer["input_root"])
        if phase == "before"
        else Path("/run") / ("edgeconnect-cpuqual-" + a.outer["nonce"]) / "external"
    )
    capture, _ = a.reader.fixed(parent / ("r3-" + phase + ".json"))
    receipt_pin, receipt_raw = a.reader.fixed(
        parent / ("r3-" + phase + ".receipt.json")
    )
    receipt = completion.parse(receipt_raw)
    require(
        receipt["status"] == "complete"
        and receipt["capture_pin"] == capture
        and receipt["request_sha256"] == launch["request"]["sha256"]
        and receipt["registration_sha256"] == launch["registration"]["sha256"]
        and receipt["derivations"]["launch_sha256"] == launch_pin["sha256"],
        "produced-artifact-binding",
    )
    provenance = _pin(receipt["derivations"]["provenance_pin"], 4 * 2**20)
    require(
        Path(provenance["path"])
        == parent / ("r3-" + phase + ".provenance-" + provenance["sha256"] + ".json"),
        "produced-provenance-path",
    )
    a.reader.read(provenance, maximum=4 * 2**20)
    for key in ("request", "registration"):
        a.reader.read(_pin(launch[key]))
    return {
        "launch": launch_pin,
        "request": launch["request"],
        "registration": launch["registration"],
        "capture": capture,
        "receipt": receipt_pin,
        "provenance": provenance,
    }


def publish_completion(
    a: Admission,
    *,
    phase: str,
    budget: outer.PhaseBudget,
    launch_pin: dict,
    guardian_family: dict,
    collector_family: dict,
    supervisor_admission: dict,
    enclosing: dict,
    guardian_contract: str,
    anchor: dict | None,
    plan_pin: dict | None,
    before_execution: dict | None,
) -> bytes:
    artifacts = bind_outputs(a, phase, launch_pin)
    parent = Path(artifacts["capture"]["path"]).parent
    evidence_root = parent / "execution-evidence"
    a.reader.ensure_directory(evidence_root, parent=parent)
    documents = {
        "collector_family": collector_family,
        "guardian_family": guardian_family,
        "supervisor_admission": supervisor_admission,
    }
    evidence = {name: encode(doc) for name, doc in documents.items()}
    evidence_pins = {
        name: {
            "path": str(
                evidence_root / (phase + "-" + name.replace("_", "-") + ".json")
            ),
            "sha256": sha(raw),
            "bytes": len(raw),
        }
        for name, raw in evidence.items()
    }
    now = a.kernel.clock()
    expected = {
        "nonce": a.outer["nonce"],
        "boot_id": budget.phase_start.boot_id,
        "outer_intent_sha256": a.approved_intent_sha256,
        "qualified_outer_source_sha256": a.source_sha256,
        "phase_start": asdict(budget.phase_start),
        "budget": budget.compact(),
        "enclosing": enclosing,
        "exec_contracts": {
            "guardian": guardian_contract,
            "collector": completion.collector_contract_sha256(
                a.outer["python"]["path"],
                a.outer["control_root"],
                launch_pin,
                budget.work_ns,
            ),
        },
        "artifacts": artifacts,
        "plan_sha256": None if plan_pin is None else plan_pin["sha256"],
        "anchor_sha256": None if anchor is None else sha(encode(anchor)),
        "before_execution_pin": before_execution,
    }
    value = {
        "format": completion.FORMAT,
        "schema_version": 1,
        "status": "natural-exit-complete",
        "phase": phase,
        **expected,
        "terminal_clock": guardian_family["family_closed"]["clock"],
        "published_clock": asdict(now),
        "evidence_pins": evidence_pins,
    }
    raw = encode(value)
    if phase == "before":
        completion.validate_before(raw, expected, asdict(now), evidence=evidence)
    else:
        assert anchor is not None
        completion.validate_after(
            raw, expected, asdict(now), evidence=evidence, anchor=anchor
        )
    for name, body in evidence.items():
        a.reader.publish(
            Path(evidence_pins[name]["path"]), body, maximum=completion.MAX_EVIDENCE
        )
    now = a.kernel.clock()
    require(budget.remaining_ns("gate", now) > 0, "completion-publication-late")
    value["published_clock"] = asdict(now)
    raw = encode(value)
    if phase == "before":
        completion.validate_before(raw, expected, asdict(now), evidence=evidence)
    else:
        assert anchor is not None
        completion.validate_after(
            raw, expected, asdict(now), evidence=evidence, anchor=anchor
        )
    store_budget(a, budget.gate_ns, budget.compact()["gate"]["wall_ns"])
    output = parent / (
        "before-execution.json" if phase == "before" else "r3-after.execution.json"
    )
    completion_pin = a.reader.publish(output, raw)
    require(
        budget.remaining_ns("gate", a.kernel.clock()) > 0, "completion-publication-late"
    )
    return encode(
        {
            "phase": phase,
            "completion_pin": completion_pin,
            "artifacts": artifacts,
            "evidence_pins": evidence_pins,
        }
    )


def capture_phase(
    a: Admission,
    *,
    authorization_path: str,
    phase: str,
    phase_start: outer.Clock,
    launch_pin: dict,
    supervisor_admission: dict,
    enclosing: dict,
    anchor: dict | None = None,
    plan_pin: dict | None = None,
    before_execution: dict | None = None,
) -> bytes:
    frame = {
        **frame_base(a, "guardian", phase_start),
        "supervisor": enclosing["supervisor"],
        "phase": phase,
        "budget": completion.before_budget(asdict(phase_start))
        if phase == "before"
        else completion.after_budget(asdict(phase_start), anchor),
        "launch": launch_pin,
        "anchor": anchor,
    }
    budget = phase_budget(frame)
    store_budget(a, budget.gate_ns, budget.compact()["gate"]["wall_ns"])
    spec = outer.RoleSpec(
        phase,
        "guardian",
        a.outer["python"]["path"],
        a.outer["control_root"],
        authorization_path,
        a.approved_intent_sha256,
        budget.work_ns,
        encode(frame),
        a.role_source_hashes,
    )
    guardian_family, raw = guarded_child(a, spec, budget)
    collector_family = completion.parse(raw, completion.MAX_EVIDENCE)
    return publish_completion(
        a,
        phase=phase,
        budget=budget,
        launch_pin=launch_pin,
        guardian_family=guardian_family,
        collector_family=collector_family,
        supervisor_admission=supervisor_admission,
        enclosing=enclosing,
        guardian_contract=outer.digest(spec.contract()),
        anchor=anchor,
        plan_pin=plan_pin,
        before_execution=before_execution,
    )


def original_start_record(a: Admission, before_bundle: bytes, now: outer.Clock) -> dict:
    bundle = completion.parse(before_bundle)
    shape(
        bundle,
        {"phase", "completion_pin", "artifacts", "evidence_pins"},
        "before-bundle-fields",
    )
    require(bundle["phase"] == "before", "before-bundle-phase")
    proof = completion.parse(a.reader.read(_pin(bundle["completion_pin"])))
    evidence = {
        name: a.reader.read(pin, maximum=completion.MAX_EVIDENCE)
        for name, pin in completion.evidence_pins(encode(proof)).items()
    }
    expected = completion.expected_from(proof)
    require(
        expected["outer_intent_sha256"] == a.approved_intent_sha256
        and expected["qualified_outer_source_sha256"] == a.source_sha256
        and expected["nonce"] == a.outer["nonce"],
        "before-handoff-authority",
    )
    now = handoff_clock(a)
    completion.validate_before(encode(proof), expected, asdict(now), evidence=evidence)
    require(
        now.monotonic_ns < a.preflight_start.monotonic_ns + 118 * outer.SECOND
        and now.wall_ns < a.preflight_start.wall_ns + 118 * outer.SECOND,
        "dummy-handoff-late",
    )
    body = {
        "format": "strength-freshness-dummy-start-v1",
        "schema_version": 1,
        "outer_intent_sha256": a.approved_intent_sha256,
        "before_execution_pin": bundle["completion_pin"],
        "clock": asdict(now),
    }
    return a.reader.publish(
        Path(a.outer["input_root"]) / "dummy-start.json", encode(body)
    )


def await_start_ack(
    a: Admission, start_pin: dict, before_execution_pin: dict, enclosing: dict
) -> None:
    path = Path(a.outer["input_root"]) / "dummy-start.ack.json"
    while True:
        now = a.kernel.clock()
        require(
            now.monotonic_ns < a.preflight_start.monotonic_ns + 118 * outer.SECOND
            and now.wall_ns < a.preflight_start.wall_ns + 118 * outer.SECOND
            and not a.kernel.exited(a.parent.pidfd),
            "dummy-ack-deadline-or-parent",
        )
        if a.reader.exists(path):
            _, raw = a.reader.fixed(path)
            value = completion.parse(raw)
            validate_dummy_start_ack(
                value,
                outer_intent_sha256=a.approved_intent_sha256,
                nonce=a.outer["nonce"],
                start_pin=start_pin,
                before_execution_pin=before_execution_pin,
                enclosing=enclosing,
            )
            handoff_clock(a)
            return
        a.kernel.sleep(0.01)


def after_launch(
    a: Admission, prepared: dict, cleanup: dict, start: outer.Clock
) -> tuple[dict, dict, outer.PhaseBudget]:
    plan = completion.parse(a.reader.read(prepared["plan_pin"]))
    anchor = completion.parse(a.reader.read(prepared["anchor_pin"]))
    require(
        plan["nonce"] == a.outer["nonce"]
        and anchor["nonce"] == a.outer["nonce"]
        and anchor["plan_sha256"] == prepared["plan_pin"]["sha256"],
        "after-plan-binding",
    )
    external = Path("/run") / ("edgeconnect-cpuqual-" + a.outer["nonce"]) / "external"
    require(
        prepared["after_path"] == str(external / "r3-after.json")
        and prepared["cleanup_proof_root"] == str(external.parent / "evidence")
        and Path(prepared["authorization_root"]) == Path(a.outer["input_root"]),
        "prepared-path-scope",
    )
    require(
        Path(cleanup["cleanup_pin"]["path"]).parent
        == Path(prepared["cleanup_proof_root"]),
        "cleanup-proof-location",
    )
    completion.order(cleanup["clock"], asdict(start))
    budget = outer.PhaseBudget.after(
        outer.Clock(
            anchor["boot_id"],
            math.floor(anchor["started_monotonic"] * outer.SECOND),
            anchor["started_wall_ns"],
        ),
        start,
    )
    store_budget(a, budget.gate_ns, budget.compact()["gate"]["wall_ns"])
    a.reader.roots = tuple(set(a.reader.roots) | {external})
    a.reader.directory(external)
    inputs = external / "capture-inputs"
    a.reader.ensure_directory(inputs, parent=external)
    require(a.reader.directory_empty(inputs), "after-inputs-not-fresh")
    before_request = completion.parse(a.reader.read(a.outer["before_request"]))
    request = copy.deepcopy(before_request)
    preserved = plan["preservation"]
    request.update(
        phase="after",
        plan_sha256=prepared["plan_pin"]["sha256"],
        anchor_sha256=sha(encode(anchor)),
        cleanup_pin=cleanup["cleanup_pin"],
        **{
            key + "_pin": preserved[key]
            for key in ("before", "policy", "before_request", "before_receipt")
        },
    )
    request["limits"]["started"] = asdict(start)
    request["limits"]["deadline_monotonic_ns"] = budget.work_ns
    request["limits"]["deadline_wall_ns"] = budget.work_wall_ns
    registration = a.reader.read(a.outer["registration"])
    registration_pin = a.reader.publish(inputs / "registration.json", registration)
    require(
        request["registration_sha256"] == registration_pin["sha256"],
        "after-registration-copy",
    )
    request_pin = a.reader.publish(inputs / "request.json", encode(request))
    before_launch = completion.parse(a.reader.read(a.outer["before_launch"]))
    launch = {
        "format": "strength-preservation-capture-launch-v1",
        "schema_version": 1,
        "phase": "after",
        "request": request_pin,
        "registration": registration_pin,
        "physical_kind": before_launch["physical_kind"],
        "plan": prepared["plan_pin"],
        "anchor": prepared["anchor_pin"],
    }
    return a.reader.publish(inputs / "launch.json", encode(launch)), anchor, budget


def actual_installer_factory(a: Admission):
    facade_path = str(
        Path(a.outer["control_root"])
        / "scripts/strength_freshness_cpu_install_runtime.py"
    )
    facade_pin = next(
        (p for p in a.outer["source_pins"] if p["path"] == facade_path), None
    )
    require(facade_pin is not None, "installer-source-missing")
    assert facade_pin is not None
    a.reader.read(facade_pin, source=True)
    from scripts import strength_freshness_cpu_install_runtime as facade

    require(
        str(Path(facade.__file__).resolve()) == facade_path,
        "installer-import-origin",
    )
    a.reader.read(facade_pin, source=True)
    return facade.Installer


def run_operator(a: Admission, frame: dict, *, installer_factory=None) -> dict:
    validate_frame(
        frame,
        role="operator",
        authorization_sha256=a.approved_intent_sha256,
        nonce=a.outer["nonce"],
    )
    activate_kernel(a)
    enclosing = {"operator": asdict(a.identity), "supervisor": frame["supervisor"]}
    source = Path(a.outer["input_root"])
    store_budget(
        a,
        a.preflight_start.monotonic_ns + 120 * outer.SECOND,
        a.preflight_start.wall_ns + 120 * outer.SECOND,
    )
    for name in ("execution-evidence", "installer-receipts"):
        directory = source / name
        a.reader.ensure_directory(directory, parent=source)
        require(a.reader.directory_empty(directory), "preflight-output-not-fresh")
    supervisor_admission = completion.parse(
        a.reader.fixed(source / "supervisor-admission.json")[1]
    )
    before_bundle = capture_phase(
        a,
        authorization_path=a.outer["authorization_path"],
        phase="before",
        phase_start=a.preflight_start,
        launch_pin=a.outer["before_launch"],
        supervisor_admission=supervisor_admission,
        enclosing=enclosing,
    )
    start = a.kernel.clock()
    start_pin = original_start_record(a, before_bundle, start)
    start = clock(completion.parse(a.reader.read(start_pin))["clock"])
    await_start_ack(
        a, start_pin, completion.parse(before_bundle)["completion_pin"], enclosing
    )
    # Reset only to the derived original dummy ceiling, never now+new allowance.
    endpoint = min(
        start.monotonic_ns + 598 * outer.SECOND,
        a.preflight_start.monotonic_ns + 718 * outer.SECOND,
    )
    store_budget(
        a,
        endpoint,
        min(
            start.wall_ns + 598 * outer.SECOND,
            a.preflight_start.wall_ns + 718 * outer.SECOND,
        ),
    )
    arm_alarm(endpoint)
    helper = HelperExecutor(a, a.outer["authorization_path"])
    helper.bind_dummy_start(start)
    if installer_factory is None:
        installer_factory = actual_installer_factory(a)
    installer = installer_factory(
        a.raw_intent,
        a.approved_intent_sha256,
        enclosing,
        intent_pin=a.intent_pin,
        helper_executor=helper,
    )
    try:
        prepared = prepared_run(
            installer.finalize_template_and_install(before_bundle, asdict(start))
        )
    except BaseException:
        installer.prearm_cleanup()
        raise
    ready = cleanup_ready(installer.await_checked_cleanup())
    launch, anchor, budget = after_launch(a, prepared, ready, a.kernel.clock())
    before = completion.parse(before_bundle)
    after_bundle = capture_phase(
        a,
        authorization_path=a.outer["authorization_path"],
        phase="after",
        phase_start=budget.phase_start,
        launch_pin=launch,
        supervisor_admission=supervisor_admission,
        enclosing=enclosing,
        anchor=anchor,
        plan_pin=prepared["plan_pin"],
        before_execution=before["completion_pin"],
    )
    store_budget(
        a,
        endpoint,
        min(
            start.wall_ns + 598 * outer.SECOND,
            a.preflight_start.wall_ns + 718 * outer.SECOND,
        ),
    )
    audit = final_audit(installer.final_audit_and_retire())
    require(
        audit["plan_sha256"] == prepared["plan_pin"]["sha256"]
        and audit["anchor_sha256"] == sha(encode(anchor)),
        "final-audit-authority",
    )
    helper.check_alive()
    a.reader.check()
    return {
        "format": "strength-freshness-operator-result-v1",
        "status": "passed_cpu_scope",
        "before_bundle": completion.parse(before_bundle),
        "after_bundle": completion.parse(after_bundle),
        "start_pin": start_pin,
        "audit": audit,
    }


def arm_alarm(deadline_ns: int) -> None:
    require(sys.platform == "linux", "linux-alarm-required")
    remaining = (deadline_ns - time.monotonic_ns()) / outer.SECOND
    require(remaining > 0, "alarm-deadline")
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, remaining)


def admit_dummy_handoff(
    a: Admission, window: outer.BudgetWindow, enclosing: dict
) -> outer.Clock | None:
    source = Path(a.outer["input_root"])
    path = source / "dummy-start.json"
    if not a.reader.exists(path):
        return None
    start_pin, raw = a.reader.fixed(path)
    value = shape(
        completion.parse(raw),
        {
            "format",
            "schema_version",
            "outer_intent_sha256",
            "before_execution_pin",
            "clock",
        },
        "dummy-start-fields",
    )
    require(
        value["format"] == "strength-freshness-dummy-start-v1"
        and value["schema_version"] == 1
        and value["outer_intent_sha256"] == a.approved_intent_sha256,
        "dummy-start-binding",
    )
    before_raw = a.reader.read(_pin(value["before_execution_pin"]))
    before = completion.parse(before_raw)
    expected = completion.expected_from(before)
    require(
        expected["enclosing"] == enclosing
        and expected["outer_intent_sha256"] == a.approved_intent_sha256
        and expected["qualified_outer_source_sha256"] == a.source_sha256
        and expected["nonce"] == a.outer["nonce"],
        "supervisor-before-authority",
    )
    a.reader.roots = tuple(set(a.reader.roots) | {source / "execution-evidence"})
    evidence = {
        name: a.reader.read(p, maximum=completion.MAX_EVIDENCE)
        for name, p in completion.evidence_pins(before_raw).items()
    }
    recorded = clock(value["clock"])
    completion.validate_before(
        before_raw, expected, asdict(recorded), evidence=evidence
    )
    fresh = handoff_clock(a)
    admitted_window = window.handoff(recorded, fresh)
    store_budget(
        a,
        a.preflight_start.monotonic_ns + 118 * outer.SECOND,
        a.preflight_start.wall_ns + 118 * outer.SECOND,
    )
    ack = dummy_start_ack(
        outer_intent_sha256=a.approved_intent_sha256,
        nonce=a.outer["nonce"],
        start_pin=start_pin,
        before_execution_pin=value["before_execution_pin"],
        enclosing=enclosing,
    )
    a.reader.publish(source / "dummy-start.ack.json", encode(ack))
    handoff_clock(a)
    store_budget(
        a,
        admitted_window.compact()["gate"]["monotonic_ns"],
        admitted_window.compact()["gate"]["wall_ns"],
    )
    return recorded


def run_supervisor(a: Admission, *, authorization_path: str) -> dict:
    activate_kernel(a)
    proof = outer.initialize_role(a.kernel, "supervisor", a.identity)
    source = Path(a.outer["input_root"])
    store_budget(
        a,
        a.preflight_start.monotonic_ns + 120 * outer.SECOND,
        a.preflight_start.wall_ns + 120 * outer.SECOND,
    )
    a.reader.publish(source / "supervisor-admission.json", encode(proof.raw()))
    frame = {
        **frame_base(a, "operator", a.preflight_start),
        "supervisor": asdict(a.identity),
    }
    window = outer.BudgetWindow(a.preflight_start)
    spec = outer.RoleSpec(
        "session",
        "operator",
        a.outer["python"]["path"],
        a.outer["control_root"],
        authorization_path,
        a.approved_intent_sha256,
        window.work_ns,
        encode(frame),
        a.role_source_hashes,
    )
    family = outer.OwnedFamily(a.kernel, proof, window)
    try:
        family.start_capture(spec, family_binding(a, window, spec), a.parent)
    except BaseException:
        if family.spawn is not None:
            try:
                family.finish(a.parent)
            except BaseException:
                pass
        raise
    require(family.root is not None, "operator-root-not-bound")
    assert family.root is not None
    enclosing = {"operator": asdict(family.root), "supervisor": asdict(a.identity)}

    def handoff(_previous_loop_clock):
        return admit_dummy_handoff(a, window, enclosing)

    try:
        result = family.finish(a.parent, handoff_check=handoff)
    except BaseException:
        family.start_failure = "supervisor-handoff-refused"
        try:
            family.finish(a.parent)
        except BaseException:
            pass  # Never turn uncertain ownership into foreign signal authority.
        raise
    require(
        result["natural_complete"] is True
        and isinstance(family.budget, outer.BudgetWindow)
        and family.budget.dummy_start is not None,
        "operator-session-incomplete",
    )
    operator_result = completion.parse(bytes(family.output["stdout"]))
    require(
        operator_result["format"] == "strength-freshness-operator-result-v1"
        and operator_result["status"] == "passed_cpu_scope",
        "operator-result-binding",
    )
    require(
        family.budget.remaining_ns("gate", a.kernel.clock()) > 0,
        "supervisor-final-late",
    )
    report = {
        "format": "strength-freshness-supervisor-result-v1",
        "status": "candidate_cpu_scope",
        "operator_result": operator_result,
        "operator_family": result,
        "supervisor_admission": proof.raw(),
        "outside_caller_terminal_audit_required": True,
        "execution_qualified": False,
    }
    a.reader.publish(source / "outer-result.json", encode(report))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", choices=("supervisor", "operator", "guardian"), required=True
    )
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    admitted = None
    try:
        frame = None if args.role == "supervisor" else read_sealed_frame()
        admitted = load_admission(
            args.authorization, args.sha256, role=args.role, frame=frame
        )
        if args.role == "supervisor":
            result = run_supervisor(admitted, authorization_path=args.authorization)
        elif args.role == "operator":
            assert frame is not None
            result = run_operator(admitted, frame)
        else:
            assert frame is not None
            result = run_guardian(admitted, frame)
        raw = encode(result)
        require(
            len(raw) <= outer.CAPTURE_LIMITS["stdout_stderr_bytes"],
            "runtime-result-bound",
        )
        sys.stdout.buffer.write(raw)
        sys.stdout.buffer.flush()
        return 0
    except Exception:
        publish_failure(admitted, args.role)
        sys.stderr.write('{"status":"incomplete","reason":"runtime-refused"}\n')
        return 1
    finally:
        if admitted is not None:
            admitted.kernel.close(admitted.parent.pidfd)


if __name__ == "__main__":
    raise SystemExit(main())
