"""Read-only request/proof admission for the preservation collector.

Inputs must come from an independently qualified prospective plan and launcher.
These checks bind measurements to that authority; they do not create authority,
install units, repair production, or turn CPU evidence into CUDA qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import math
import re
import stat
import time
from typing import Any, Mapping, cast

from scripts import strength_freshness_cpu_collect_records as records
from scripts import strength_freshness_cpu_collect_support as supports
from scripts import strength_freshness_cpu_lifecycle as lifecycle
from scripts import strength_freshness_cpu_preservation as preservation

FORMAT = "strength-preservation-collect-request-v2"
RECEIPT = "strength-preservation-collector-receipt-v2"
ADDENDUM = "8d3ad081564569564ca8eefbb79f3565c87f55ec8759bdefa793aff7e874143c"
METADATA_BUDGET = 8 * 2**20
RUNTIME_BUDGET = 24 * 2**20
FILE_LIMIT = 2**20
PROOF_LIMIT = 262144
PROVENANCE_LIMIT = 4 * 2**20
COMMON = {
    "format",
    "schema_version",
    "phase",
    "registration_sha256",
    "nonce",
    "source_pins",
    "limits",
}
AFTER = {
    "plan_sha256",
    "anchor_sha256",
    "before_pin",
    "policy_pin",
    "cleanup_pin",
    "before_request_pin",
    "before_receipt_pin",
}
RECEIPT_FIELDS = {
    "format",
    "schema_version",
    "status",
    "request_sha256",
    "registration_sha256",
    "encoding_contract_sha256",
    "source_pins",
    "read_start",
    "read_end",
    "capture_pin",
    "raw_inventory_sha256",
    "derivations",
    "restart_counter_scopes",
    "refusals",
}
LOG_EVENTS = {"cleanup-complete", "raw-cleanup", "dispatcher-started", "raw-role-start"}
DUMMY_STATIC = set(
    "Id LoadState FragmentPath DropInPaths ExecStart Environment EnvironmentFiles User Type Group WorkingDirectory ExecStartPre Restart TimeoutStopUSec KillMode SendSIGKILL Before After RuntimeMaxUSec".split()
)


class RequestRefusal(ValueError):
    """Only fixed codes leave the private observation boundary."""


def require(value: object, code: str) -> None:
    if not value:
        raise RequestRefusal(code)


def integer(value: object, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def checksum(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def canonical(value: object) -> Path:
    require(isinstance(value, str), "path-type")
    assert isinstance(value, str)
    p = Path(value)
    require(
        p.is_absolute()
        and p.as_posix() == value
        and not value.startswith("//")
        and ".." not in p.parts
        and "\0" not in value,
        "path-canonical",
    )
    return p


def pin(value: Mapping[str, Any], *, limit=FILE_LIMIT) -> dict[str, Any]:
    require(set(value) == {"path", "sha256", "bytes"}, "pin-fields")
    canonical(value["path"])
    require(
        checksum(value["sha256"])
        and integer(value["bytes"], 1)
        and value["bytes"] <= limit,
        "pin-value",
    )
    return dict(value)


def clock(value: Mapping[str, Any]) -> dict[str, Any]:
    require(
        set(value) == {"boot_id", "monotonic_ns", "wall_ns"}, "request-clock-fields"
    )
    require(
        isinstance(value["boot_id"], str)
        and value["boot_id"]
        and all(integer(value[k], 1) for k in ("monotonic_ns", "wall_ns")),
        "request-clock",
    )
    return dict(value)


def provenance_json(raw: bytes, *, check=lambda: None) -> dict[str, Any]:
    """Larger *safe provenance* only; ordinary R3 records retain their1MiB cap."""
    require(
        type(raw) is bytes and 0 < len(raw) <= PROVENANCE_LIMIT, "provenance-byte-bound"
    )
    check()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=records._pairs,
            parse_constant=records._constant,
        )
    except (ValueError, UnicodeError, RecursionError):
        raise RequestRefusal("provenance-json") from None
    require(isinstance(value, dict), "provenance-object")
    stack = [(value, 0)]
    count = 0
    while stack:
        item, depth = stack.pop()
        count += 1
        require(count <= 250000 and depth <= 64, "provenance-complexity")
        if count % 1024 == 0:
            check()
        if isinstance(item, dict):
            stack.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            stack.extend((v, depth + 1) for v in item)
        elif isinstance(item, float):
            require(math.isfinite(item), "provenance-nonfinite")
    check()
    return value


def request(raw: bytes) -> dict[str, Any]:
    value = records.parse_json(raw)
    require(value.get("phase") in {"before", "after"}, "request-phase")
    require(
        set(value) == COMMON | (AFTER if value["phase"] == "after" else set()),
        "request-fields",
    )
    require(
        value["format"] == FORMAT
        and type(value["schema_version"]) is int
        and value["schema_version"] == 2,
        "request-version",
    )
    require(
        checksum(value["registration_sha256"])
        and isinstance(value["nonce"], str)
        and re.fullmatch(r"[0-9a-f]{32}", value["nonce"]),
        "request-identity",
    )
    sources = value["source_pins"]
    require(
        isinstance(sources, list) and 1 <= len(sources) <= 128, "request-source-count"
    )
    checked = [pin(x) for x in sources]
    require(
        len({x["path"] for x in checked}) == len(checked)
        and checked == sorted(checked, key=lambda p: p["path"]),
        "request-source-order",
    )
    limits = value["limits"]
    require(
        set(limits)
        == {
            "started",
            "deadline_monotonic_ns",
            "deadline_wall_ns",
            "metadata_bytes",
            "runtime_bytes",
        },
        "request-limit-fields",
    )
    started = clock(limits["started"])
    require(
        limits["metadata_bytes"] == METADATA_BUDGET
        and limits["runtime_bytes"] == RUNTIME_BUDGET
        and type(limits["metadata_bytes"]) is int
        and type(limits["runtime_bytes"]) is int,
        "request-byte-allocation",
    )
    for end, start in (
        ("deadline_monotonic_ns", "monotonic_ns"),
        ("deadline_wall_ns", "wall_ns"),
    ):
        require(
            integer(limits[end], 1) and 0 < limits[end] - started[start] <= 120 * 10**9,
            "request-time-allocation",
        )
    if value["phase"] == "after":
        require(
            checksum(value["plan_sha256"]) and checksum(value["anchor_sha256"]),
            "request-plan-pin",
        )
        for name in (
            "before_pin",
            "policy_pin",
            "cleanup_pin",
            "before_request_pin",
            "before_receipt_pin",
        ):
            pin(value[name], limit=PROOF_LIMIT if name == "cleanup_pin" else FILE_LIMIT)
    return value


def effective_deadline(value, now) -> float:
    current = clock(now)
    limits = value["limits"]
    start = limits["started"]
    require(current["boot_id"] == start["boot_id"], "request-boot-changed")
    for key, end in (
        ("monotonic_ns", "deadline_monotonic_ns"),
        ("wall_ns", "deadline_wall_ns"),
    ):
        require(
            start[key] <= current[key] < limits[end], "request-expired-or-regressed"
        )
    return (
        min(
            limits["deadline_monotonic_ns"],
            current["monotonic_ns"] + limits["deadline_wall_ns"] - current["wall_ns"],
        )
        / 1e9
    )


def _identity(st):
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


class PinnedReader:
    """One absolute deadline and one fixed metadata budget for the whole phase.

    No globbing arbitrary roots, writes, or file type based fallbacks. A caller
    must enforce the original outer process deadline for uninterruptible I/O.
    """

    def __init__(self, *, deadline: float, owner_uid: int, monotonic=time.monotonic):
        require(
            type(deadline) in (float, int)
            and math.isfinite(deadline)
            and deadline > monotonic(),
            "reader-deadline",
        )
        require(integer(owner_uid), "reader-owner")
        self.deadline, self.owner_uid, self.monotonic = deadline, owner_uid, monotonic
        self.consumed = 0
        self.audit: list[dict[str, Any]] = []

    def check(self):
        require(self.monotonic() < self.deadline, "metadata-deadline")

    def charge(self, size):
        require(integer(size), "metadata-size")
        self.consumed += size
        require(self.consumed <= METADATA_BUDGET, "metadata-budget")

    def read(self, expected, *, root: Path | None = None, limit=FILE_LIMIT):
        expected = pin(expected, limit=limit)
        path = canonical(expected["path"])
        if root is not None:
            require(path.parent == root, "proof-parent")
        self.check()
        require(path.resolve() == path, "input-symlink")
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        try:
            first = os.fstat(fd)
            require(
                stat.S_ISREG(first.st_mode)
                and first.st_uid == self.owner_uid
                and stat.S_IMODE(first.st_mode) == 0o444
                and first.st_size == expected["bytes"],
                "input-protection-or-size",
            )
            chunks = []
            read_bytes = 0
            while read_bytes <= expected["bytes"]:
                self.check()
                data = os.read(fd, min(65536, expected["bytes"] + 1 - read_bytes))
                if not data:
                    break
                read_bytes += len(data)
                self.charge(len(data))
                chunks.append(data)
            last = os.fstat(fd)
            raw = b"".join(chunks)
            require(
                _identity(first) == _identity(last) == _identity(path.lstat()),
                "input-raced",
            )
            require(
                len(raw) == expected["bytes"] and digest(raw) == expected["sha256"],
                "input-hash",
            )
        finally:
            os.close(fd)
        self.check()
        self.audit.append(
            {
                "pin": expected,
                "device": first.st_dev,
                "inode": first.st_ino,
                "mode": 0o444,
                "uid": first.st_uid,
            }
        )
        return raw

    def directory(self, root):
        self.check()
        st = root.lstat()
        require(
            stat.S_ISDIR(st.st_mode)
            and st.st_uid == self.owner_uid
            and stat.S_IMODE(st.st_mode) == 0o700
            and root.resolve() == root,
            "proof-directory-protection",
        )
        self.check()
        return (st.st_dev, st.st_ino, st.st_uid, stat.S_IMODE(st.st_mode))

    def inventory(self, root):
        paths = []
        count = 0
        with os.scandir(root) as entries:
            for entry in entries:
                self.check()
                count += 1
                require(count <= lifecycle.MAX_LOG_FILES + 1, "proof-inventory-limit")
                self.charge(len(entry.name.encode()))
                require(
                    len(entry.name) <= 128
                    and entry.is_file(follow_symlinks=False)
                    and not entry.is_symlink(),
                    "proof-entry",
                )
                if entry.name == ".append.lock":
                    continue
                require(
                    re.fullmatch(
                        r"[a-z][a-z0-9_-]{0,47}-[0-9a-f]{32}\.json", entry.name
                    ),
                    "proof-inventory-name",
                )
                paths.append(root / entry.name)
        self.check()
        return sorted(paths)


class _IndexedRoot:
    def __init__(self, root, paths):
        self.path, self.paths = root, tuple(paths)

    def glob(self, pattern):
        event = pattern.removesuffix("-*.json")
        require(
            pattern == event + "-*.json" and event in LOG_EVENTS, "proof-event-query"
        )
        return tuple(
            p
            for p in self.paths
            if re.fullmatch(re.escape(event) + r"-[0-9a-f]{32}\.json", p.name)
        )


class ReadOnlyLog:
    """Minimal read facade for lifecycle semantic checks; never calls RawLog()."""

    def __init__(self, reader: PinnedReader, root: Path, anchor: lifecycle.Anchor):
        require(root == canonical(str(root)) and root.resolve() == root, "proof-root")
        self.reader, self.directory, self.anchor = reader, root, anchor
        self.directory_identity = self._directory()
        self.root = _IndexedRoot(root, reader.inventory(root))
        self.loaded: dict[str, dict[str, Any]] = {}
        self._directory()

    def _directory(self):
        self.reader.check()
        identity = self.reader.directory(self.directory)
        if hasattr(self, "directory_identity"):
            require(identity == self.directory_identity, "proof-directory-replaced")
        return identity

    def load(self, expected, event):
        require(event in LOG_EVENTS, "proof-event")
        self._directory()
        path = canonical(expected["path"])
        require(
            path in self.root.paths
            and re.fullmatch(re.escape(event) + r"-[0-9a-f]{32}\.json", path.name),
            "proof-file-binding",
        )
        raw = self.reader.read(expected, root=self.directory, limit=PROOF_LIMIT)
        value = records.parse_json(raw, maximum_bytes=PROOF_LIMIT)
        require(
            set(value)
            == {"format", "schema_version", "anchor", "anchor_sha256", "event", "data"}
            and value["format"] == "strength-freshness-cpu-lifecycle-evidence-v1"
            and type(value["schema_version"]) is int
            and value["schema_version"] == 1,
            "proof-format",
        )
        require(
            value["event"] == event
            and value["anchor"] == self.anchor.as_dict()
            and value["anchor_sha256"] == lifecycle.digest(self.anchor.as_dict()),
            "proof-anchor",
        )
        require(
            path.as_posix() not in self.loaded
            or self.loaded[path.as_posix()] == dict(expected),
            "proof-pin-substitution",
        )
        self.loaded[path.as_posix()] = dict(expected)
        self._directory()
        return value["data"]

    def recheck(self):
        self._directory()
        current = _IndexedRoot(self.directory, self.reader.inventory(self.directory))
        for event in ("cleanup-complete", "dispatcher-started"):
            require(
                current.glob(event + "-*.json") == self.root.glob(event + "-*.json"),
                "proof-singleton-raced",
            )
        for expected in self.loaded.values():
            self.reader.read(expected, root=self.directory, limit=PROOF_LIMIT)
        self._directory()


@dataclass(frozen=True)
class AfterAdmission:
    anchor: lifecycle.Anchor
    cleanup_clock: dict[str, Any]
    policy: dict[str, Any]
    before: dict[str, Any]
    support_witnesses: dict[str, Any]
    verified_champions: dict[str, str]
    after_path: str
    proof: ReadOnlyLog


def _same_sources(value, registration):
    expected = sorted(
        [
            {"path": registration["scope"]["files"][key]["path"], **item}
            for key, item in registration["source_pins"].items()
        ],
        key=lambda p: p["path"],
    )
    require(value["source_pins"] == expected, "registered-control-sources")


def validate_registration(value, raw):
    require(digest(raw) == value["registration_sha256"], "registration-byte-binding")
    registration = records.parse_json(raw)
    _same_sources(value, registration)
    require(
        value["limits"]["started"]["boot_id"]
        == registration["birth_reference"]["boot_id"],
        "registration-boot",
    )
    return registration


def _sources(reader, value):
    for item in value["source_pins"]:
        reader.read(item)


def executed_sources(value, *, modules=()):
    """Bind actual custom module origins, not merely same-named disk copies."""
    expected = {str(Path(__file__).resolve())}
    for module in (records, supports, lifecycle, preservation, *modules):
        origin = getattr(module, "__file__", None)
        require(isinstance(origin, str), "executed-module-file")
        assert isinstance(origin, str)
        expected.add(str(Path(origin).resolve()))
    pins = {p["path"]: p for p in value["source_pins"]}
    require(expected <= set(pins), "executed-module-origin")
    return {name: pins[name] for name in sorted(expected)}


def dummy_static(
    value, *, allow_empty_environment_files=False, allow_empty_exec_start_pre=False
):
    """The pinned LinuxHost projection, without importing its training runtime.

    Keep unknown Exec fields and ignore_errors. Only volatile execution suffixes
    and dependency ordering are normalized; no other missing field is invented.
    """
    rows = dict(value)
    if "EnvironmentFiles" not in rows:
        require(allow_empty_environment_files, "unqualified-dummy-empty-property")
        rows["EnvironmentFiles"] = ""
    if "ExecStartPre" not in rows:
        require(allow_empty_exec_start_pre, "unqualified-dummy-empty-property")
        rows["ExecStartPre"] = ""
    require(
        DUMMY_STATIC <= set(rows)
        and all(isinstance(rows[k], str) for k in DUMMY_STATIC),
        "dummy-static-fields",
    )
    result = {k: rows[k] for k in DUMMY_STATIC}
    for key in ("Before", "After"):
        result[key] = " ".join(sorted(result[key].split()))
    for key in ("ExecStart", "ExecStartPre"):
        if " ; ignore_errors=" not in result[key]:
            continue
        require(result[key].count("{ path=") <= 1, "multiple-dummy-exec-records")
        prefix, suffix = result[key].split(" ; ignore_errors=", 1)
        parts = suffix.removesuffix(" }").split(" ; ")
        require(parts[0] in {"yes", "no"}, "dummy-ignore-errors")
        dynamic = {
            "start_time",
            "stop_time",
            "pid",
            "code",
            "status",
            "start_time_monotonic",
            "stop_time_monotonic",
        }
        retained = [part for part in parts[1:] if part.split("=", 1)[0] not in dynamic]
        result[key] = (
            prefix
            + " ; ignore_errors="
            + parts[0]
            + "".join(" ; " + part for part in retained)
        )
    return result


def validate_before_receipt(
    value,
    *,
    before_request,
    before_request_pin,
    before_pin,
    registration,
    before,
    anchor,
):
    require(set(value) == RECEIPT_FIELDS, "before-receipt-fields")
    require(
        value["format"] == RECEIPT
        and type(value["schema_version"]) is int
        and value["schema_version"] == 2
        and value["status"] == "complete"
        and value["refusals"] == [],
        "before-receipt-status",
    )
    require(
        value["capture_pin"] == before_pin
        and value["request_sha256"] == before_request_pin["sha256"]
        and value["registration_sha256"] == before_request["registration_sha256"]
        and value["source_pins"] == before_request["source_pins"]
        and value["encoding_contract_sha256"]
        == registration["encoding_contract_sha256"]
        and value["restart_counter_scopes"] == registration["counter_scopes"],
        "before-receipt-binding",
    )
    require(checksum(value["raw_inventory_sha256"]), "before-inventory-hash")
    start, end = clock(value["read_start"]), clock(value["read_end"])
    require(
        start["boot_id"]
        == end["boot_id"]
        == anchor.boot_id
        == before_request["limits"]["started"]["boot_id"]
        == registration["birth_reference"]["boot_id"],
        "before-receipt-boot",
    )
    limits = before_request["limits"]
    for key, limit in (
        ("monotonic_ns", "deadline_monotonic_ns"),
        ("wall_ns", "deadline_wall_ns"),
    ):
        require(
            limits["started"][key] <= start[key] <= end[key] < limits[limit],
            "before-receipt-time",
        )
    require(
        before["clock"]
        == {
            "boot_id": end["boot_id"],
            "monotonic": end["monotonic_ns"] / 1e9,
            "wall_ns": end["wall_ns"],
        },
        "before-clock-derivation",
    )
    require(
        0
        <= anchor.started_wall_ns - end["wall_ns"]
        <= registration["policy"]["maximum_age_ns"]
        and 0
        <= anchor.started_monotonic - end["monotonic_ns"] / 1e9
        <= registration["policy"]["maximum_age_ns"] / 1e9,
        "before-not-prework",
    )
    derived = value["derivations"]
    require(
        isinstance(derived, dict)
        and isinstance(derived.get("support_witnesses"), dict),
        "before-support-provenance",
    )
    return derived["support_witnesses"]


def read_before_provenance(
    reader, *, receipt, receipt_pin, before_request_pin, before_pin, registration
):
    reference = pin(receipt["derivations"]["provenance_pin"], limit=PROVENANCE_LIMIT)
    capture_path = canonical(before_pin["path"])
    expected_name = capture_path.stem + ".provenance-" + reference["sha256"] + ".json"
    path = canonical(reference["path"])
    require(
        path.parent == capture_path.parent == canonical(receipt_pin["path"]).parent
        and path.name == expected_name,
        "before-provenance-location",
    )
    value = provenance_json(
        reader.read(reference, root=capture_path.parent, limit=PROVENANCE_LIMIT),
        check=reader.check,
    )
    require(
        set(value)
        == {
            "format",
            "schema_version",
            "binding",
            "raw_inventory",
            "raw_inventory_sha256",
            "support_witnesses",
            "core_provenance",
            "metadata_audit",
            "final_clock_projection",
            "preservation_verification",
        }
        and value["format"] == "strength-preservation-capture-provenance-bundle-v1"
        and type(value["schema_version"]) is int
        and value["schema_version"] == 1,
        "before-provenance-format",
    )
    expected_binding = {
        "phase": "before",
        "request_sha256": before_request_pin["sha256"],
        "registration_file_sha256": receipt["registration_sha256"],
        "encoding_contract_sha256": receipt["encoding_contract_sha256"],
        "source_pins": receipt["source_pins"],
        "capture_pin": before_pin,
        "read_start": receipt["read_start"],
        "read_end": receipt["read_end"],
    }
    require(value["binding"] == expected_binding, "before-provenance-binding")
    inventory = value["raw_inventory"]
    require(
        isinstance(inventory, list)
        and preservation.digest(inventory)
        == value["raw_inventory_sha256"]
        == receipt["raw_inventory_sha256"],
        "before-inventory-content",
    )
    require(
        value["support_witnesses"] == receipt["derivations"]["support_witnesses"]
        and value["preservation_verification"] is None,
        "before-witness-content",
    )
    require(
        set(value["support_witnesses"]) <= set(registration["policy"]["support"]),
        "before-witness-inventory",
    )
    for name, witness in value["support_witnesses"].items():
        start = clock(witness["absence"]["read_start"])
        end = clock(witness["absence"]["read_end"])
        require(
            start["boot_id"] == end["boot_id"] == receipt["read_end"]["boot_id"],
            "before-witness-boot",
        )
        require(
            all(
                receipt["read_start"][key]
                <= start[key]
                <= end[key]
                <= receipt["read_end"][key]
                for key in ("monotonic_ns", "wall_ns")
            ),
            "before-witness-outside-capture",
        )
        supports.checked_witness(
            witness,
            name=name,
            manager=witness["proof"]["manager"],
            now=receipt["read_end"],
        )
    core = value["core_provenance"]
    require(
        core["format"] == "strength-preservation-capture-provenance-v1"
        and type(core["schema_version"]) is int
        and core["schema_version"] == 1
        and core["phase"] == "before",
        "before-core-format",
    )
    require(
        core["registration_sha256"] == preservation.digest(registration)
        and core["policy_sha256"] == preservation.digest(registration["policy"])
        and core["encoding_contract_sha256"] == registration["encoding_contract_sha256"]
        and core["support_witnesses"] == value["support_witnesses"],
        "before-core-binding",
    )
    count = core["raw_inventory_count"]
    require(
        integer(count)
        and count <= len(inventory)
        and preservation.digest(inventory[:count]) == core["raw_inventory_sha256"],
        "before-core-inventory-prefix",
    )
    require(isinstance(value["metadata_audit"], list), "before-metadata-audit")
    audit_fields = {"operation", "subject", "read_start", "read_end", "raw"}

    def projections(rows):
        require(isinstance(rows, list), "before-audit-inventory")
        return Counter(
            preservation.digest({k: row[k] for k in audit_fields})
            for row in rows
            if isinstance(row, dict) and audit_fields <= set(row)
        )

    core_inventory = projections(inventory[:count])
    retained_support = core["support_provenance"]
    require(isinstance(retained_support, list), "before-support-observations")
    for name, witness in value["support_witnesses"].items():
        evidence = witness["proof"]
        needed = projections(
            [row["audit"] for row in evidence["unit_reads"] + evidence["jobs_reads"]]
        )
        require(not (needed - core_inventory), "before-witness-unlogged")
        matches = []
        for observed in retained_support:
            if (
                observed.get("format") != "strength-support-capture-v1"
                or observed.get("manager") != evidence["manager"]
                or observed.get("manager_id") != witness["absence"]["manager_id"]
                or observed.get("clock") != witness["absence"]["read_end"]
            ):
                continue
            if (
                name not in observed.get("unit_states", {})
                or observed["unit_states"][name].get("Job") not in ("", "0")
                or name in observed.get("jobs", {})
            ):
                continue
            if not (needed - projections(observed["audit"])):
                matches.append(observed)
        require(len(matches) == 1, "before-witness-support-observation")
    return value


def admit_after(
    *, value, registration, reader: PinnedReader, plan_pin, anchor_pin, now
) -> AfterAdmission:
    """Bind an already qualified plan to measured pre-work inputs and cleanup.

    This deliberately does not import the broader plan/controller runtime. The
    outer launcher must qualify that exact plan and source/import closure first.
    """
    value = request(lifecycle.encoded(value))
    require(value["phase"] == "after", "after-admission-phase")
    _same_sources(value, registration)
    require(
        value["limits"]["started"]["boot_id"]
        == registration["birth_reference"]["boot_id"],
        "registration-boot",
    )
    executed_sources(value)
    require(pin(plan_pin)["sha256"] == value["plan_sha256"], "plan-pin")
    plan = records.parse_json(reader.read(plan_pin))
    require(
        plan["format"] == "strength-freshness-cpu-plan-v1"
        and type(plan["schema_version"]) is int
        and plan["schema_version"] == 1
        and plan["nonce"] == value["nonce"]
        and ADDENDUM in plan["addenda_sha256"],
        "prospective-plan-binding",
    )
    raw_anchor = records.parse_json(reader.read(anchor_pin))
    anchor = lifecycle.Anchor.from_dict(raw_anchor)
    require(
        lifecycle.digest(raw_anchor) == value["anchor_sha256"]
        and anchor.nonce == value["nonce"]
        and anchor.plan_sha256 == plan_pin["sha256"]
        and anchor.attempt_id == plan["attempt_id"]
        and anchor.boot_id == plan["boot_id"] == value["limits"]["started"]["boot_id"],
        "anchor-plan-binding",
    )
    current = clock(now)
    deadline = effective_deadline(value, current)
    remaining = anchor.effective_deadline(
        "observer",
        lifecycle.Clock(
            current["boot_id"], current["monotonic_ns"] / 1e9, current["wall_ns"]
        ),
    )
    require(
        deadline <= remaining and reader.deadline <= deadline,
        "original-observer-deadline",
    )
    require(
        value["limits"]["deadline_monotonic_ns"]
        <= math.floor(anchor.deadline("observer") * 1e9)
        and value["limits"]["deadline_wall_ns"] <= anchor.started_wall_ns + 570 * 10**9,
        "request-cannot-renew-clock",
    )
    inputs = canonical(plan["input_root"])
    scratch = canonical(plan["scratch_root"])
    require(
        scratch == Path("/run") / ("edgeconnect-cpuqual-" + anchor.nonce),
        "attempt-scratch-root",
    )
    p = plan["preservation"]
    require(
        all(item in plan["source_pins"] for item in value["source_pins"]),
        "precommitted-control-sources",
    )
    require(
        set(p)
        == {
            "policy",
            "before",
            "before_request",
            "before_receipt",
            "after_path",
            "verified_champions",
        },
        "preservation-plan-fields",
    )
    pairs = {
        "policy": "policy_pin",
        "before": "before_pin",
        "before_request": "before_request_pin",
        "before_receipt": "before_receipt_pin",
    }
    require(len({p[k]["path"] for k in pairs}) == 4, "prework-input-alias")
    for key, name in pairs.items():
        require(
            pin(p[key]) == value[name]
            and canonical(p[key]["path"]).is_relative_to(inputs),
            "precommitted-input-pin",
        )
    require(
        canonical(p["after_path"]) == scratch / "external" / "r3-after.json",
        "after-output-path",
    )
    before_request = request(reader.read(p["before_request"]))
    require(
        before_request["phase"] == "before"
        and all(
            before_request[k] == value[k]
            for k in ("nonce", "registration_sha256", "source_pins")
        ),
        "prework-request-binding",
    )
    before = records.parse_json(reader.read(p["before"]))
    policy = records.parse_json(reader.read(p["policy"]))
    require(policy == registration["policy"], "registered-policy-binding")
    receipt = records.parse_json(reader.read(p["before_receipt"]))
    witnesses = validate_before_receipt(
        receipt,
        before_request=before_request,
        before_request_pin=p["before_request"],
        before_pin=p["before"],
        registration=registration,
        before=before,
        anchor=anchor,
    )
    provenance = read_before_provenance(
        reader,
        receipt=receipt,
        receipt_pin=p["before_receipt"],
        before_request_pin=p["before_request"],
        before_pin=p["before"],
        registration=registration,
    )
    require(
        provenance["core_provenance"]["verified_champions_sha256"]
        == preservation.digest(p["verified_champions"]),
        "prework-champion-proof-map",
    )
    units = plan["units"]
    require(isinstance(units, dict) and len(units) == 12, "plan-unit-count")
    prefix = "edgeconnect-cpuqual-" + anchor.nonce + "-"
    require(
        all(
            re.fullmatch(
                re.escape(prefix) + r"[a-z][a-z0-9-]{0,40}\.(service|timer)", name
            )
            for name in units
        ),
        "plan-unit-namespace",
    )
    workloads = {name for name, spec in units.items() if spec["role"] == "workload"}
    require(
        len(workloads) == 7
        and all(
            sum(spec["role"] == role for spec in units.values()) == 1
            for role in ("dispatcher", "cleanup", "watchdog", "observer", "publisher")
        ),
        "plan-role-count",
    )
    dispatcher = next(
        name for name, spec in units.items() if spec["role"] == "dispatcher"
    )
    life_source = str(Path(lifecycle.__file__).resolve())
    life_pins = [x for x in value["source_pins"] if x["path"] == life_source]
    require(
        len(life_pins) == 1 and life_pins[0] in plan["source_pins"],
        "executed-lifecycle-source",
    )
    # Two semantic checks re-read this small source file internally. Charge the
    # same fixed budget in addition to independently pinned file admission.
    _sources(reader, value)
    reader.charge(2 * life_pins[0]["bytes"])
    log = ReadOnlyLog(reader, scratch / "evidence", anchor)
    checked = lifecycle._checked_cleanup(
        cast(lifecycle.RawLog, log), value["cleanup_pin"]
    )
    require(
        set(checked["expected_workloads"]) == workloads
        and len(checked["expected_workloads"]) == len(workloads),
        "cleanup-plan-workloads",
    )
    barrier = checked["facts"]["dispatcher_barrier"]
    started = log.load(barrier["started"], "dispatcher-started")
    owner = started["owner"]
    require(
        owner["unit"] == dispatcher
        and owner["cgroup"] == "/system.slice/" + dispatcher,
        "dispatcher-plan-owner",
    )
    definition = units[dispatcher]
    expected_props = dummy_static(definition["before"]["properties"])
    require(
        expected_props == dummy_static(definition["after"]["properties"])
        and definition["installed_path"] == "/etc/systemd/system/" + dispatcher,
        "dispatcher-static-plan",
    )
    before_unit, after_unit = (
        pin(definition["before"]["unit"]),
        pin(definition["after"]["unit"]),
    )
    require(
        all(before_unit[k] == after_unit[k] for k in ("sha256", "bytes")),
        "dispatcher-unit-byte-drift",
    )
    for unit_pin in (before_unit, after_unit):
        require(
            canonical(unit_pin["path"]).is_relative_to(inputs), "dispatcher-unit-input"
        )
        reader.read(unit_pin)
    empty_rule = (
        registration["empty_property_rules"].get("EnvironmentFiles")
        == "systemd-255-empty-EnvironmentFiles"
    )
    require(
        dummy_static(
            barrier["unit"],
            allow_empty_environment_files=empty_rule,
            allow_empty_exec_start_pre=(
                registration["empty_property_rules"].get("ExecStartPre")
                == "systemd-255-empty-ExecStartPre"
            ),
        )
        == expected_props,
        "dispatcher-definition-proof",
    )
    cleanup = checked["clock"]
    require(
        cleanup["wall_ns"] <= current["wall_ns"]
        and cleanup["monotonic"] <= current["monotonic_ns"] / 1e9,
        "cleanup-after-current-capture",
    )
    # The same semantic verifier checks all baseline fields; this call cannot
    # assert later progress or replace the final before/after preservation gate.
    preservation._snapshot(policy, before, p["verified_champions"])
    log.recheck()
    reader.check()
    return AfterAdmission(
        anchor,
        cleanup,
        policy,
        before,
        witnesses,
        dict(p["verified_champions"]),
        p["after_path"],
        log,
    )
