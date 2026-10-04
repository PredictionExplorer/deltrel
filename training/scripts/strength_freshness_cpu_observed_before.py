"""Retrospective observed-BEFORE artifact checks, never launch authority.

The caller independently admits the intent, enclosing processes and qualification
evidence. This reader checks their complete pinned byte/semantic joins; it does
not repeat live bootstrap or qualify opaque writer/access evidence. No old
runtime selects this route, and no AFTER or installer permission is returned.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from types import FunctionType, ModuleType
from typing import Any

from scripts import strength_freshness_cpu_capture_request as requests
from scripts import strength_freshness_cpu_collect_identity as identity
from scripts import strength_freshness_cpu_completion as completion
from scripts import strength_freshness_cpu_observed_capture_cli as producer
from scripts import strength_freshness_cpu_observed_kernel as kernel
from scripts import strength_freshness_cpu_observed_registration as registration_module
from scripts import strength_freshness_cpu_outer as outer
from scripts import strength_freshness_cpu_outer_runtime as runtime
from scripts import strength_freshness_cpu_window_contract as bodies

REQUEST = "strength-preservation-observed-window-collect-request-v1"
LAUNCH = "strength-preservation-observed-window-capture-launch-v1"
CONTRACT = registration_module.CONTRACT
OUTER_FIELDS = frozenset(
    "format schema_version nonce preflight_start input_root control_root python "
    "source_pins session_qualification helper_site_qualification caller "
    "before_launch before_request registration verified_champions "
    "resource_policy_sha256".split()
)
LIMIT_FIELDS = frozenset(
    "started deadline_monotonic_ns deadline_wall_ns metadata_bytes runtime_bytes".split()
)
SITE_CHECKS = frozenset(
    "site_enabled_import_closure no_gpu_initialization resource_limits "
    "actual_helper_origin no_user_site loader_environment".split()
)
_READER_TYPE = requests.PinnedReader
_BASE_READ = requests.PinnedReader.read
_BASE_CHECK = requests.PinnedReader.check
_BASE_DIRECTORY = requests.PinnedReader.directory


class BeforeRefusal(ValueError):
    """Fixed codes; no private input or arbitrary exception text."""


def require(ok: object, reason: str) -> None:
    if not ok:
        raise BeforeRefusal(reason)


def fields(value: Any, names, reason="before-fields") -> dict[str, Any]:
    require(type(value) is dict and set(value) == set(names), reason)
    return value


def same(a, b):
    return bodies.encoded(a) == bodies.encoded(b)


def _pin(value: Any, maximum=requests.FILE_LIMIT) -> dict[str, Any]:
    fields(value, {"path", "sha256", "bytes"}, "before-pin-fields")
    return requests.pin(value, limit=maximum)


def _sources(value: Any) -> list[dict[str, Any]]:
    require(type(value) is list and 1 <= len(value) <= 128, "before-source-count")
    result = [_pin(item) for item in value]
    require(
        len({item["path"] for item in result}) == len(result)
        and same(result, sorted(result, key=lambda item: item["path"])),
        "before-source-order",
    )
    return result


def _limits(value: Any) -> dict[str, Any]:
    fields(value, LIMIT_FIELDS, "before-limit-fields")
    completion.clock(value["started"])
    require(
        type(value["metadata_bytes"]) is int
        and value["metadata_bytes"] == requests.METADATA_BUDGET
        and type(value["runtime_bytes"]) is int
        and value["runtime_bytes"] == requests.RUNTIME_BUDGET,
        "before-byte-limits",
    )
    for axis in ("monotonic", "wall"):
        end = value["deadline_" + axis + "_ns"]
        require(
            type(end) is int
            and 0 < end - value["started"][axis + "_ns"] <= 105 * 10**9,
            "before-original-window",
        )
    return value


def parse_request(raw: bytes) -> dict[str, Any]:
    value = bodies.parse(raw)
    fields(
        value,
        {
            "format",
            "schema_version",
            "contract",
            "phase",
            "nonce",
            "registration_sha256",
            "source_pins",
            "limits",
        },
        "before-request-fields",
    )
    require(
        value["format"] == REQUEST
        and type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and value["contract"] == CONTRACT
        and value["phase"] == "before"
        and runtime.re_nonce(value["nonce"])
        and completion.checksum(value["registration_sha256"]),
        "before-request-format",
    )
    _limits(value["limits"])
    _sources(value["source_pins"])
    return value


def executed_sources(sources, control_root):
    """Join the actual imported custom dependency graph, not filename presence."""
    pins = {item["path"]: item for item in _sources(sources)}
    root = requests.canonical(control_root)
    pending = [sys.modules[__name__]]
    seen = set()
    actual = set()
    while pending:
        module = pending.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        require(len(seen) <= 128, "before-import-closure-bound")
        filename = getattr(module, "__file__", None)
        require(type(filename) is str, "before-import-origin")
        assert isinstance(filename, str)
        path = str(Path(filename).resolve())
        require(
            path in pins and Path(path).is_relative_to(root), "before-import-origin"
        )
        actual.add(path)
        for item in vars(module).values():
            if isinstance(item, ModuleType):
                candidate = item
                name = item.__name__
            elif isinstance(item, (FunctionType, type)):
                name = item.__module__
                candidate = sys.modules.get(name)
            else:
                continue
            if name.startswith("scripts."):
                require(isinstance(candidate, ModuleType), "before-import-module")
                assert isinstance(candidate, ModuleType)
                pending.append(candidate)
    return sorted(actual)


class _Read:
    def __init__(self, reader, current):
        fields(current, {"limits", "admission_clock"}, "before-current-window-fields")
        require(type(reader) is _READER_TYPE, "before-reader-type")
        self.reader, self._reader = reader, reader
        self._reader_type = _READER_TYPE
        self._methods = {
            "read": _BASE_READ,
            "check": _BASE_CHECK,
            "directory": _BASE_DIRECTORY,
        }
        self._deadline, self._owner = reader.deadline, reader.owner_uid
        self._consumed = reader.consumed
        self.limits = deepcopy(_limits(current["limits"]))
        self.last = deepcopy(completion.clock(current["admission_clock"]))
        completion.order(self.limits["started"], self.last)
        require(reader.owner_uid == 0, "before-protected-root-owner")
        wall_left = self.limits["deadline_wall_ns"] - self.last["wall_ns"]
        require(
            wall_left > 0
            and reader.deadline <= self.limits["deadline_monotonic_ns"] / 1e9
            and reader.deadline <= self.last["monotonic_ns"] / 1e9 + wall_left / 1e9,
            "before-reader-original-deadline",
        )
        self.cache: dict[tuple, bytes] = {}
        self.pins: dict[tuple, tuple[dict, Any, int]] = {}
        self._path_pins: dict[str, tuple[str, int]] = {}
        self.check()

    def check(self):
        require(
            self.reader is self._reader
            and type(self.reader) is self._reader_type
            and type(self.reader.deadline) is type(self._deadline)
            and self.reader.deadline == self._deadline
            and type(self.reader.owner_uid) is int
            and self.reader.owner_uid == self._owner == 0,
            "before-reader-context-changed",
        )
        for name, expected in self._methods.items():
            method = getattr(self.reader, name, None)
            require(
                getattr(method, "__self__", None) is self._reader
                and getattr(method, "__func__", None) is expected,
                "before-reader-method-changed",
            )
        require(
            type(self.reader.consumed) is int
            and type(self._consumed) is int
            and 0 <= self._consumed <= self.reader.consumed <= requests.METADATA_BUDGET,
            "before-reader-ledger-changed",
        )
        self._consumed = self.reader.consumed
        self._methods["check"](self.reader)
        now = {
            "boot_id": self.last["boot_id"],
            "monotonic_ns": time.monotonic_ns(),
            "wall_ns": time.time_ns(),
        }
        # Boot is an already admitted caller premise, not measured by this reader.
        completion.order(self.last, now)
        require(
            now["monotonic_ns"] < self.limits["deadline_monotonic_ns"]
            and now["wall_ns"] < self.limits["deadline_wall_ns"],
            "before-current-deadline",
        )
        self.last = now

    def read(self, item, *, root=None, maximum=requests.FILE_LIMIT, fresh=False):
        item = _pin(item, maximum)
        if root is not None:
            require(
                requests.canonical(item["path"]).parent == root, "before-input-parent"
            )
        version = (item["sha256"], item["bytes"])
        require(
            item["path"] not in self._path_pins
            or self._path_pins[item["path"]] == version,
            "before-pin-version-conflict",
        )
        self._path_pins[item["path"]] = version
        self.check()
        key = (item["path"], item["sha256"], item["bytes"])
        self.pins[key] = (deepcopy(item), root, maximum)
        if fresh or key not in self.cache:
            consumed = self.reader.consumed
            try:
                raw = self._methods["read"](self.reader, item, root=root, limit=maximum)
            finally:
                self.check()
            require(
                type(raw) is bytes
                and len(raw) == item["bytes"]
                and completion.sha(raw) == item["sha256"],
                "before-read-pin",
            )
            require(self.reader.consumed == consumed + len(raw), "before-read-charge")
            if not fresh:
                self.cache[key] = raw
        else:
            raw = self.cache[key]
        self.check()
        return raw

    def directory(self, path):
        self.check()
        value = self._methods["directory"](self.reader, path)
        self.check()
        return value


def _qualification_evidence(read, cfg, *, boot_id, caller, sources):
    """Retrospective existing document joins; no live bootstrap is repeated."""
    common = {
        "schema_version": 1,
        "status": "passed-real-linux",
        "evidence_kind": "real-linux-process",
        "synthetic": False,
        "boot_id": boot_id,
        "source_closure_sha256": outer.digest(sources),
        "python_sha256": cfg["python"]["sha256"],
        "resource_policy_sha256": cfg["resource_policy_sha256"],
    }
    result = {}
    for key, fmt, checks in (
        ("session_qualification", runtime.SESSION, runtime.SESSION_CHECKS),
        (
            "helper_site_qualification",
            "strength-freshness-real-helper-site-qualification-v1",
            SITE_CHECKS,
        ),
    ):
        item = _pin(cfg[key])  # Missing/null is a refusal, never a test grant.
        doc = bodies.parse(read.read(item))
        require(doc.get("format") == fmt, "before-qualification-format")
        require(
            all(same(doc.get(k), v) for k, v in common.items()),
            "before-qualification-binding",
        )
        fields(doc.get("checks"), checks, "before-qualification-checks")
        require(
            all(v is True for v in doc["checks"].values()),
            "before-qualification-checks",
        )
        refs = doc.get("raw_evidence_pins")
        require(
            type(refs) is list and 1 <= len(refs) <= 16,
            "before-raw-qualification-evidence",
        )
        assert isinstance(refs, list)
        for ref in refs:
            read.read(_pin(ref))
        if key == "session_qualification":
            require(
                doc.get("caller_cgroup") == caller["cgroup"]
                and same(doc.get("pid_namespace_inode"), caller["pid_namespace_inode"]),
                "before-session-context",
            )
        else:
            require(
                doc.get("helper_environment_sha256")
                == outer.digest({**outer.ENV, "PYTHONPATH": cfg["control_root"]}),
                "before-site-environment",
            )
            inv = bodies.parse(read.read(_pin(doc.get("site_inventory_pin"))))
            fields(
                inv,
                {
                    "format",
                    "schema_version",
                    "root",
                    "startup_directories",
                    "startup_files",
                    "cached_files",
                },
                "before-site-inventory-fields",
            )
            require(
                inv["format"] == "strength-freshness-qualified-site-files-v1"
                and same(inv["schema_version"], 1),
                "before-site-inventory-format",
            )
            root = requests.canonical(inv["root"])
            require(
                type(inv["startup_directories"]) is list
                and 1 <= len(inv["startup_directories"]) <= 64,
                "before-site-directory-count",
            )
            directories = set()
            for directory in inv["startup_directories"]:
                directory = fields(
                    directory, {"path", "entries"}, "before-site-directory-fields"
                )
                path = requests.canonical(directory["path"])
                require(
                    path.is_relative_to(root)
                    and path not in directories
                    and type(directory["entries"]) is list
                    and len(directory["entries"]) <= 4096
                    and all(type(name) is str for name in directory["entries"])
                    and directory["entries"] == sorted(set(directory["entries"])),
                    "before-site-directory-scope",
                )
                directories.add(path)
            seen = set()
            startup_bytes = 0
            for name, count in (("startup_files", 256), ("cached_files", 4096)):
                require(
                    type(inv[name]) is list and 1 <= len(inv[name]) <= count,
                    "before-site-file-count",
                )
                for row in inv[name]:
                    fields(
                        row,
                        {"path", "sha256", "metadata"}
                        | ({"cache_receipt"} if name == "cached_files" else set()),
                        "before-site-file-fields",
                    )
                    path = requests.canonical(row["path"])
                    require(
                        path.is_relative_to(root)
                        and path not in seen
                        and completion.checksum(row["sha256"]),
                        "before-site-file-scope",
                    )
                    seen.add(path)
                    fields(
                        row["metadata"],
                        runtime.INTERPRETER_STAT_FIELDS,
                        "before-site-metadata",
                    )
                    if name == "startup_files":
                        startup_bytes += row["metadata"]["size"]
                        require(
                            startup_bytes <= 8 * 2**20 and path.parent in directories,
                            "before-site-startup-bound",
                        )
                    require(
                        all(
                            type(v) is int and v >= 0 for v in row["metadata"].values()
                        ),
                        "before-site-metadata",
                    )
                    if name == "cached_files":
                        cache = bodies.parse(read.read(_pin(row["cache_receipt"])))
                        required = {
                            "format": "strength-freshness-qualified-file-cache-v1",
                            "status": "independently-qualified",
                            "evidence_kind": "real-host-content-hash",
                            "synthetic": False,
                            "path": str(path),
                            "sha256": row["sha256"],
                            "metadata": row["metadata"],
                        }
                        require(
                            all(same(cache.get(k), v) for k, v in required.items()),
                            "before-site-cache-binding",
                        )
        result[key] = item
    return result


@dataclass(frozen=True)
class BeforeReadFacts:
    """Detached conditional data, not an authenticated-B or launch capability."""

    _private: dict[str, Any] = field(repr=False)

    def private_copy(self):
        return deepcopy(self._private)

    def safe_summary(self):
        return {
            "format": "strength-observed-before-read-facts-v1",
            "contract": CONTRACT,
            "phase": "before",
            "execution_qualified": False,
            "historical_birth_qualified": False,
            "after_authority": False,
            "launcher_authority": False,
        }


def read_observed_before_artifacts(
    committed,
    *,
    approved_outer,
    enclosing,
    producer_sources,
    control_root,
    python,
    reader,
    historical_consume_clock,
    current_request_limits,
):
    try:
        return _read_before(
            committed,
            approved_outer=approved_outer,
            enclosing=enclosing,
            producer_sources=producer_sources,
            control_root=control_root,
            python=python,
            reader=reader,
            historical_consume_clock=historical_consume_clock,
            current_request_limits=current_request_limits,
        )
    except BeforeRefusal:
        raise
    except (
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        OSError,
        OverflowError,
        RecursionError,
        RuntimeError,
    ):
        raise BeforeRefusal("before-artifact-refused") from None


def _read_before(
    committed,
    *,
    approved_outer,
    enclosing,
    producer_sources,
    control_root,
    python,
    reader,
    historical_consume_clock,
    current_request_limits,
):
    fields(
        committed,
        {
            "before_execution",
            "before",
            "before_request",
            "before_receipt",
            "policy",
            "verified_champions",
        },
        "before-committed-fields",
    )
    fields(
        approved_outer,
        {"intent_pin", "caller_identity", "premise_evidence_pins"},
        "before-approved-fields",
    )
    read = _Read(reader, current_request_limits)
    intent_pin = _pin(approved_outer["intent_pin"])
    inputs = requests.canonical(intent_pin["path"]).parent
    initial_directory = read.directory(inputs)
    intent = bodies.parse(read.read(intent_pin, root=inputs))
    selection = fields(
        intent.get("observed_before"),
        {"contract", "premise_evidence_pins"},
        "before-intent-selection",
    )
    require(
        selection["contract"] == CONTRACT
        and same(
            selection["premise_evidence_pins"], approved_outer["premise_evidence_pins"]
        ),
        "before-intent-premises",
    )
    cfg = fields(intent.get("outer"), OUTER_FIELDS, "before-outer-fields")
    require(
        cfg["format"] == runtime.ADMISSION and same(cfg["schema_version"], 1),
        "before-outer-format",
    )
    caller = completion.identity(approved_outer["caller_identity"])
    completion.clock(cfg["preflight_start"])
    require(
        same(cfg["caller"], caller)
        and caller["uid"] == 0
        and cfg["input_root"] == str(inputs)
        and cfg["control_root"] == control_root
        and cfg["python"]["path"] == python
        and type(python) is str
        and intent.get("nonce") == cfg["nonce"]
        and runtime.re_nonce(cfg["nonce"])
        and intent.get("boot_id")
        == cfg["preflight_start"]["boot_id"]
        == caller["boot_id"]
        == read.last["boot_id"]
        and cfg["resource_policy_sha256"] == outer.digest(runtime.resource_policy()),
        "before-outer-binding",
    )
    fields(
        cfg["python"],
        {"path", "resolved_path", "sha256", "bytes", "metadata"},
        "before-python-fields",
    )
    runtime.validate_interpreter_metadata(
        cfg["python"]["metadata"], cfg["python"]["bytes"]
    )
    require(
        type(cfg["python"]["bytes"]) is int
        and 0 < cfg["python"]["bytes"] <= 64 * 2**20
        and completion.checksum(cfg["python"]["sha256"]),
        "before-python-hash",
    )
    requests.canonical(cfg["python"]["resolved_path"])
    sources = _sources(producer_sources)
    require(same(cfg["source_pins"], sources), "before-control-source-binding")
    control = requests.canonical(control_root)
    require(
        all(requests.canonical(p["path"]).is_relative_to(control) for p in sources),
        "before-control-source-root",
    )
    actual_origins = executed_sources(sources, control_root)
    source_map = {p["path"]: p for p in sources}
    mandatory = {
        str(control / "scripts" / name)
        for name in runtime.REQUIRED_SOURCE_NAMES
        | {
            "strength_freshness_cpu_before_measurement.py",
            "strength_freshness_cpu_observed_capture_cli.py",
            "strength_freshness_cpu_window_contract.py",
            "strength_freshness_cpu_observed_before.py",
        }
    }
    require(mandatory <= set(source_map), "before-control-source-closure")
    for source in sources:
        read.read(source, fresh=True)
    qualification_pins = _qualification_evidence(
        read, cfg, boot_id=caller["boot_id"], caller=caller, sources=sources
    )
    launch_pin, request_pin, reg_pin = (
        _pin(cfg[k]) for k in ("before_launch", "before_request", "registration")
    )
    require(same(committed["before_request"], request_pin), "before-request-commitment")
    launch_raw = read.read(launch_pin, root=inputs)
    request_raw = read.read(request_pin, root=inputs)
    reg_raw = read.read(reg_pin, root=inputs)
    launch, request = bodies.parse(launch_raw), parse_request(request_raw)
    fields(
        launch,
        {
            "format",
            "schema_version",
            "contract",
            "phase",
            "request",
            "registration",
            "input_root",
            "verified_champions",
            "external_premisesPin",
            "writer_evidencePin",
        },
        "before-launch-fields",
    )
    require(
        launch["format"] == LAUNCH
        and same(launch["schema_version"], 1)
        and launch["contract"] == CONTRACT
        and launch["phase"] == "before"
        and same(launch["request"], request_pin)
        and same(launch["registration"], reg_pin)
        and launch["input_root"] == str(inputs)
        and same(launch["verified_champions"], cfg["verified_champions"])
        and same(committed["verified_champions"], cfg["verified_champions"])
        and request["nonce"] == cfg["nonce"]
        and request["registration_sha256"] == reg_pin["sha256"]
        and same(request["source_pins"], sources),
        "before-launch-request-binding",
    )
    budget = completion.before_budget(cfg["preflight_start"])
    completion.order(cfg["preflight_start"], request["limits"]["started"])
    require(
        all(
            request["limits"]["deadline_" + axis + "_ns"]
            <= budget["work"][axis + "_ns"]
            for axis in ("monotonic", "wall")
        ),
        "before-preflight-work-limit",
    )
    reg_header = identity.strict_json(reg_raw)
    scope = identity.scope_from(reg_header["scope"])
    writer_pin = _pin(launch["writer_evidencePin"])
    premises_pin = _pin(launch["external_premisesPin"])
    require(
        same(writer_pin, reg_header["learner_writer"]["evidence_pin"]),
        "before-writer-pin",
    )
    require(
        all(
            any(
                entry.path == pin["path"] and pin["bytes"] <= entry.maximum_bytes
                for entry in scope.files.values()
            )
            for pin in (writer_pin, premises_pin)
        ),
        "before-private-dependency-scope",
    )
    writer_raw = read.read(writer_pin)
    premises = bodies.parse(read.read(premises_pin))
    reg = registration_module.parse_observed_registration(
        reg_raw,
        approved_sha256=request["registration_sha256"],
        scope=scope,
        writer_evidence=writer_raw,
    )
    kernel.validate_external_premises(reg, premises)
    require(
        caller["pid_namespace_inode"]
        == reg.private_copy()["kernel_context"]["namespace_expectations"]["self"][
            "pid"
        ],
        "before-producer-namespace",
    )
    require(
        same(
            bodies.parse(read.read(_pin(committed["policy"]))),
            reg.private_copy()["policy"],
        ),
        "before-policy-commitment",
    )
    writer = reg.private_copy()["learner_writer"]["binding"]
    premise_pins = fields(
        selection["premise_evidence_pins"],
        {"source", "writer", "writer_access"},
        "before-premise-pin-map",
    )
    for name, expected_hash in (
        ("source", premises["source_context"]["qualification_sha256"]),
        ("writer", writer["qualification_sha256"]),
        ("writer_access", writer["access_sha256"]),
    ):
        item = _pin(premise_pins[name])
        require(item["sha256"] == expected_hash, "before-premise-hash")
        read.read(item)
    for publication in reg.private_copy()["publication_writers"].values():
        read.read(_pin(publication["access_qualification_pin"]))
    proof_pin = _pin(committed["before_execution"])
    require(
        requests.canonical(proof_pin["path"]) == inputs / "before-execution.json",
        "before-completion-location",
    )
    proof_raw = read.read(proof_pin, root=inputs)
    proof = completion.parse(proof_raw)
    artifact_pins = fields(
        proof["artifacts"], completion.ARTIFACTS, "before-artifact-inventory"
    )
    exact = {
        "launch": launch_pin,
        "request": request_pin,
        "registration": reg_pin,
        "capture": _pin(committed["before"]),
        "receipt": _pin(committed["before_receipt"]),
    }
    require(
        all(same(artifact_pins[k], v) for k, v in exact.items()),
        "before-artifact-commitment",
    )
    prov = _pin(artifact_pins["provenance"], requests.PROVENANCE_LIMIT)
    require(
        requests.canonical(exact["capture"]["path"]) == inputs / "r3-before.json"
        and requests.canonical(exact["receipt"]["path"])
        == inputs / "r3-before.receipt.json"
        and requests.canonical(prov["path"])
        == inputs / ("r3-before.provenance-" + prov["sha256"] + ".json")
        and len({p["path"] for p in artifact_pins.values()}) == 6,
        "before-artifact-location",
    )
    raw_artifacts = {
        name: read.read(
            pin,
            root=inputs,
            maximum=requests.PROVENANCE_LIMIT
            if name == "provenance"
            else requests.FILE_LIMIT,
        )
        for name, pin in artifact_pins.items()
    }
    evidence_root = inputs / "execution-evidence"
    evidence_directory = read.directory(evidence_root)
    evidence = {}
    for name, item in completion.evidence_pins(proof_raw).items():
        require(
            requests.canonical(item["path"])
            == evidence_root / ("before-" + name.replace("_", "-") + ".json"),
            "before-family-location",
        )
        evidence[name] = read.read(
            item, root=evidence_root, maximum=completion.MAX_EVIDENCE
        )
    fields(enclosing, {"operator", "supervisor"}, "before-enclosing-fields")
    sup = completion.identity(enclosing["supervisor"])
    require(
        sup["ppid"] == caller["pid"]
        and sup["start_ticks"] >= caller["start_ticks"]
        and all(
            sup[k] == caller[k]
            for k in ("boot_id", "cgroup", "uid", "pid_namespace_inode")
        ),
        "before-supervisor-caller",
    )
    outer_pin = source_map[str(control / "scripts/strength_freshness_cpu_outer.py")]
    runtime_pin = source_map[
        str(control / "scripts/strength_freshness_cpu_outer_runtime.py")
    ]
    expected_completion = {
        "nonce": cfg["nonce"],
        "boot_id": caller["boot_id"],
        "outer_intent_sha256": intent_pin["sha256"],
        "qualified_outer_source_sha256": outer_pin["sha256"],
        "phase_start": cfg["preflight_start"],
        "budget": budget,
        "enclosing": deepcopy(enclosing),
        "exec_contracts": {
            "guardian": outer.guardian_contract_sha256(
                python,
                control_root,
                intent_pin["path"],
                (outer_pin["sha256"], runtime_pin["sha256"]),
            ),
            "collector": producer.observed_collector_contract_sha256(
                python,
                control_root,
                launch_pin,
                request["limits"]["deadline_monotonic_ns"],
            ),
        },
        "artifacts": artifact_pins,
        "plan_sha256": None,
        "anchor_sha256": None,
        "before_execution_pin": None,
    }
    verified = completion.validate_before(
        proof_raw,
        expected_completion,
        completion.clock(historical_consume_clock),
        evidence=evidence,
    )
    completion.order(historical_consume_clock, read.last)
    family = completion.parse(evidence["collector_family"], completion.MAX_EVIDENCE)
    window = {
        "phase": "before",
        "phase_start": request["limits"]["started"],
        "deadline": {
            "boot_id": caller["boot_id"],
            "monotonic_ns": request["limits"]["deadline_monotonic_ns"],
            "wall_ns": request["limits"]["deadline_wall_ns"],
        },
        "cleanup_clock": None,
    }
    reg_value = reg.private_copy()
    bindings = {
        "nonce": cfg["nonce"],
        "boot_id": caller["boot_id"],
        "request_sha256": request_pin["sha256"],
        "launch_sha256": launch_pin["sha256"],
        "registration_sha256": reg_pin["sha256"],
        "policy_sha256": bodies.digest(reg_value["policy"]),
        "source_pins_sha256": bodies.digest(sources),
        "external_premises_sha256": bodies.digest(premises),
        "writer_evidence_sha256": completion.sha(writer_raw),
        "recipe_sha256": bodies.digest(reg_value["policy"]["recipe"]),
        "verified_champions_sha256": bodies.digest(cfg["verified_champions"]),
        "expected_window_sha256": bodies.digest(window),
        "source_contract_sha256": writer["source_contract_sha256"],
        "access_sha256": writer["access_sha256"],
        "writer_qualification_sha256": writer["qualification_sha256"],
        "source_qualification_sha256": premises["source_context"][
            "qualification_sha256"
        ],
    }
    expected_body = {
        "contract": CONTRACT,
        "bindings": bindings,
        "expected_window": window,
        "writer_binding": writer,
        "expected_policy": reg_value["policy"],
        "verified_champions": cfg["verified_champions"],
        "expected_publication_writers": reg_value["publication_writers"],
    }
    body_facts = bodies.validate_before_bodies(
        raw_artifacts["capture"],
        raw_artifacts["receipt"],
        raw_artifacts["provenance"],
        expected=expected_body,
        registration=reg,
        producer_interval={
            "released": family["child_released"]["clock"],
            "terminal": family["child_terminal"]["clock"],
        },
        artifact_pins=artifact_pins,
    )
    # Every admitted body/dependency is checked again under the original ledger.
    # This can honestly exhaust8MiB; no cache or larger allowance rescues it.
    executed_sources(sources, control_root)
    for item, root, maximum in list(read.pins.values()):
        read.read(item, root=root, maximum=maximum, fresh=True)
    require(
        read.directory(inputs) == initial_directory
        and read.directory(evidence_root) == evidence_directory,
        "before-directory-raced",
    )
    read.check()
    return BeforeReadFacts(
        deepcopy(
            {
                "format": "strength-observed-before-read-facts-v1",
                "contract": CONTRACT,
                "bindings": bindings,
                "body_facts": body_facts,
                "completion_sha256": verified.sha256,
                "artifact_pins": artifact_pins,
                "qualification_document_pins": qualification_pins,
                "actual_import_origins": actual_origins,
                "current_read_end": read.last,
                "historical_consume_clock": historical_consume_clock,
                "metadata_bytes_consumed": reader.consumed,
                "execution_qualified": False,
                "historical_birth_qualified": False,
                "after_authority": False,
                "launcher_authority": False,
                "opaque_premise_semantics_independently_verified": False,
            }
        )
    )
