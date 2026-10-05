"""Pure producer-exit evidence checks; never observes or controls a process.

Expected bindings and raw-file authenticity belong to the independently approved
outer caller. Hashes and these semantic checks do not qualify Linux execution.
Only collector and phase guardian completion is certified, never the still-live
operator, supervisor or whole experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
from typing import Any, Mapping

ADDENDUM_SHA256 = "9d7408304d614690ec7abe47575ef5b18e8209870e7d3912d33af462840a9481"

FORMAT = "strength-freshness-external-capture-complete-v1"
FAMILY = "strength-freshness-owned-family-v1"
BINDINGS = {
    "nonce",
    "boot_id",
    "outer_intent_sha256",
    "qualified_outer_source_sha256",
    "phase_start",
    "budget",
    "enclosing",
    "exec_contracts",
    "artifacts",
    "plan_sha256",
    "anchor_sha256",
    "before_execution_pin",
}
ARTIFACTS = {"launch", "request", "registration", "capture", "receipt", "provenance"}
EVIDENCE = ("collector_family", "guardian_family", "supervisor_admission")
IDENTITY = {
    "pid",
    "start_ticks",
    "ppid",
    "pgid",
    "sid",
    "uid",
    "boot_id",
    "cgroup",
    "pid_namespace_inode",
}
MAX_RECEIPT = 2**20
MAX_EVIDENCE = 2**18
MAX_OUTPUT = 2**20


class CompletionRefusal(ValueError):
    """Fixed reason codes only; never rejected private values."""


def require(value: object, code: str):
    if not value:
        raise CompletionRefusal(code)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encoded(value: object) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise CompletionRefusal("completion-json") from None


def parse(raw: bytes, maximum=MAX_RECEIPT) -> dict[str, Any]:
    require(type(raw) is bytes and 0 < len(raw) <= maximum, "completion-byte-bound")

    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "completion-duplicate-key")
            result[key] = value
        return result

    def invalid(_):
        raise CompletionRefusal("completion-nonfinite")

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid
        )
    except CompletionRefusal:
        raise
    except (ValueError, UnicodeError, RecursionError):
        raise CompletionRefusal("completion-json") from None
    pending = [(value, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        require(depth <= 64 and count <= 250000, "completion-structure-bound")
        if isinstance(item, dict):
            pending.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            pending.extend((v, depth + 1) for v in item)
        elif isinstance(item, float):
            require(math.isfinite(item), "completion-nonfinite")
    require(isinstance(value, dict), "completion-object")
    return value


def shape(value, keys, reason) -> dict[str, Any]:
    require(isinstance(value, Mapping) and set(value) == set(keys), reason)
    assert isinstance(value, Mapping)
    return dict(value)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def checksum(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def clock(value):
    shape(value, {"boot_id", "monotonic_ns", "wall_ns"}, "completion-clock-fields")
    require(
        isinstance(value["boot_id"], str)
        and 0 < len(value["boot_id"]) <= 128
        and all(integer(value[k], 1) for k in ("monotonic_ns", "wall_ns")),
        "completion-clock",
    )
    return dict(value)


def order(first, last):
    clock(first)
    clock(last)
    require(
        first["boot_id"] == last["boot_id"]
        and all(first[k] <= last[k] for k in ("monotonic_ns", "wall_ns")),
        "completion-clock-order",
    )


def limit(observed, deadline, *, strict=False):
    require(
        all(
            observed[k] < deadline[k] if strict else observed[k] <= deadline[k]
            for k in ("monotonic_ns", "wall_ns")
        ),
        "completion-late",
    )


def identity(value):
    shape(value, IDENTITY, "completion-process-fields")
    require(
        all(
            integer(value[k], 1)
            for k in ("pid", "start_ticks", "pgid", "sid", "pid_namespace_inode")
        )
        and integer(value["ppid"])
        and integer(value["uid"]),
        "completion-process",
    )
    require(
        isinstance(value["boot_id"], str)
        and value["boot_id"]
        and isinstance(value["cgroup"], str),
        "completion-process-text",
    )
    path(value["cgroup"])
    return dict(value)


def path(value):
    require(
        isinstance(value, str)
        and value.startswith("/")
        and not value.startswith("//")
        and str(PurePosixPath(value)) == value
        and ".." not in PurePosixPath(value).parts
        and "\0" not in value,
        "completion-path",
    )
    return PurePosixPath(value)


def pin(value, maximum=MAX_RECEIPT):
    shape(value, {"path", "sha256", "bytes"}, "completion-pin-fields")
    path(value["path"])
    require(
        checksum(value["sha256"])
        and integer(value["bytes"], 1)
        and value["bytes"] <= maximum,
        "completion-pin",
    )
    return dict(value)


def collector_contract_sha256(python, control_root, launch_pin, deadline_monotonic_ns):
    """Exact compact CaptureSpec contract; no outer/kernel module import."""
    require(path(python).name == "python", "completion-python")
    path(control_root)
    pin(launch_pin)
    require(integer(deadline_monotonic_ns, 1), "completion-exec-deadline")
    contract = {
        "argv": [
            python,
            "-S",
            "-E",
            "-B",
            "-m",
            "scripts.strength_freshness_cpu_capture_cli",
            "--launch",
            launch_pin["path"],
            "--launch-sha256",
            launch_pin["sha256"],
            "--deadline-monotonic-ns",
            str(deadline_monotonic_ns),
        ],
        "cwd": control_root,
        "env": {
            "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG": "C",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "resources": {
            "cpu_affinity_count": 1,
            "nice": 19,
            "address_space_bytes": 512 * 2**20,
            "open_files": 128,
            "file_bytes": 32 * 2**20,
            "stdout_stderr_bytes": 2**20,
            "owned_children": 32,
        },
    }
    return sha(
        json.dumps(
            contract, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def before_budget(start):
    clock(start)
    return {
        name: {
            axis: start[axis] + seconds * 10**9 for axis in ("monotonic_ns", "wall_ns")
        }
        for name, seconds in [("work", 105), ("cleanup", 115), ("gate", 120)]
    }


def after_budget(start, anchor):
    clock(start)
    shape(
        anchor,
        {
            "attempt_id",
            "nonce",
            "plan_sha256",
            "boot_id",
            "started_monotonic",
            "started_wall_ns",
        },
        "completion-anchor-fields",
    )
    require(
        type(anchor["started_monotonic"]) in (int, float)
        and math.isfinite(anchor["started_monotonic"])
        and anchor["started_monotonic"] >= 0
        and integer(anchor["started_wall_ns"], 1),
        "completion-anchor-clock",
    )
    anchor_start = {
        "monotonic_ns": math.floor(anchor["started_monotonic"] * 10**9),
        "wall_ns": anchor["started_wall_ns"],
    }
    require(
        start["boot_id"] == anchor["boot_id"]
        and all(anchor_start[k] <= start[k] for k in anchor_start),
        "completion-anchor-start",
    )
    work = {
        k: min(start[k] + 105 * 10**9, anchor_start[k] + 555 * 10**9)
        for k in anchor_start
    }
    cleanup = {
        k: min(work[k] + 5 * 10**9, anchor_start[k] + 560 * 10**9) for k in anchor_start
    }
    gate = {
        k: min(cleanup[k] + 5 * 10**9, anchor_start[k] + 565 * 10**9)
        for k in anchor_start
    }
    return {"work": work, "cleanup": cleanup, "gate": gate}


def evidence_pins(raw):
    value = parse(raw)
    pins = shape(value.get("evidence_pins"), EVIDENCE, "completion-evidence-fields")
    return {name: pin(pins[name], MAX_EVIDENCE) for name in EVIDENCE}


def expected_from(value):
    """Extract claims, NOT authority. Consumers must overwrite/join trusted facts."""
    require(
        isinstance(value, Mapping) and BINDINGS <= set(value), "completion-bindings"
    )
    return {k: value[k] for k in BINDINGS}


def _admission(row, role, owner, start=None):
    shape(
        row,
        {"role", "self", "clock", "subreaper_before", "subreaper_after", "task_ids"},
        "completion-admission-fields",
    )
    require(row["role"] == role and row["self"] == owner, "completion-admission-owner")
    require(
        type(row["subreaper_before"]) is int
        and row["subreaper_before"] in (0, 1)
        and type(row["subreaper_after"]) is int
        and row["subreaper_after"] == 1
        and row["task_ids"] == [owner["pid"]],
        "completion-subreaper-threads",
    )
    clock(row["clock"])
    require(row["clock"]["boot_id"] == owner["boot_id"], "completion-admission-boot")
    if start is not None:
        order(start, row["clock"])


def _family(doc, *, role, owner, child, exec_hash, binding, budget):
    shape(
        doc,
        {
            "format",
            "role_admission",
            "child_started",
            "child_released",
            "child_terminal",
            "family_closed",
            "signals",
            "adopted_children",
            "other_terminals",
            "binding",
            "budget",
            "output",
            "natural_complete",
            "reason",
        },
        "completion-family-fields",
    )
    require(
        doc["format"] == FAMILY
        and doc["binding"] == binding
        and doc["budget"] == budget,
        "completion-family-binding",
    )
    require(
        doc["natural_complete"] is True
        and doc["reason"] is None
        and doc["signals"] == []
        and doc["adopted_children"] == []
        and doc["other_terminals"] == [],
        "completion-forced-or-orphan",
    )
    _admission(doc["role_admission"], role, owner, binding["phase_start"])
    born = shape(
        doc["child_started"],
        {
            "owner",
            "child",
            "clock",
            "exec_contract_sha256",
            "pidfd_target_pid",
            "gate_closed",
        },
        "completion-start-fields",
    )
    require(
        born["owner"] == owner
        and born["child"] == child
        and child["ppid"] == owner["pid"]
        and born["exec_contract_sha256"] == exec_hash
        and type(born["pidfd_target_pid"]) is int
        and born["pidfd_target_pid"] == child["pid"]
        and born["gate_closed"] is True,
        "completion-gated-start",
    )
    order(doc["role_admission"]["clock"], born["clock"])
    limit(born["clock"], budget["work"], strict=True)
    release = shape(
        doc["child_released"],
        {"child", "clock", "exec_contract_sha256"},
        "completion-release-fields",
    )
    require(
        release["child"] == child and release["exec_contract_sha256"] == exec_hash,
        "completion-release-binding",
    )
    order(born["clock"], release["clock"])
    limit(release["clock"], budget["work"], strict=True)
    terminal = shape(
        doc["child_terminal"],
        {"owner", "child", "clock", "waitid", "waitpid", "reaped"},
        "completion-terminal-fields",
    )
    require(
        terminal["owner"] == owner
        and terminal["child"] == child
        and terminal["reaped"] is True,
        "completion-terminal-owner",
    )
    wi = shape(
        terminal["waitid"], {"pid", "code", "status"}, "completion-waitid-fields"
    )
    wp = shape(terminal["waitpid"], {"pid", "status"}, "completion-waitpid-fields")
    require(
        type(wi["pid"]) is int
        and type(wp["pid"]) is int
        and wi["pid"] == wp["pid"] == child["pid"]
        and wi["code"] == "CLD_EXITED"
        and type(wi["status"]) is int
        and type(wp["status"]) is int
        and wi["status"] == wp["status"] == 0,
        "completion-not-natural-exit0",
    )
    order(release["clock"], terminal["clock"])
    limit(terminal["clock"], budget["cleanup"])
    closed = shape(
        doc["family_closed"],
        {"owner", "clock", "task_ids", "direct_children", "retained_children"},
        "completion-family-closed-fields",
    )
    require(
        closed["owner"] == owner
        and closed["task_ids"] == [owner["pid"]]
        and closed["direct_children"] == closed["retained_children"] == [],
        "completion-family-not-empty",
    )
    order(terminal["clock"], closed["clock"])
    limit(closed["clock"], budget["cleanup"])
    output = shape(
        doc["output"],
        {"stdout_sha256", "stdout_bytes", "stderr_sha256", "stderr_bytes"},
        "completion-output-fields",
    )
    require(
        checksum(output["stdout_sha256"])
        and checksum(output["stderr_sha256"])
        and integer(output["stdout_bytes"])
        and integer(output["stderr_bytes"])
        and output["stdout_bytes"] + output["stderr_bytes"] <= MAX_OUTPUT
        and all(
            output[k + "_bytes"] != 0 or output[k + "_sha256"] == sha(b"")
            for k in ("stdout", "stderr")
        ),
        "completion-output-pin",
    )
    return born, release, terminal, closed


@dataclass(frozen=True)
class VerifiedCompletion:
    sha256: str
    phase: str
    boot_id: str
    _json: bytes

    @property
    def value(self):
        return parse(self._json)

    @property
    def phase_start(self):
        return self.value["phase_start"]

    @property
    def terminal_clock(self):
        return self.value["terminal_clock"]

    @property
    def original_deadline(self):
        return self.value["budget"]["gate"]

    @property
    def artifacts(self):
        return self.value["artifacts"]


def _validate_impl(raw, expected, now, *, phase, evidence, anchor=None):
    value = parse(raw)
    shape(expected, BINDINGS, "completion-expected-fields")
    shape(
        value,
        BINDINGS
        | {
            "format",
            "schema_version",
            "status",
            "phase",
            "terminal_clock",
            "published_clock",
            "evidence_pins",
        },
        "completion-fields",
    )
    require(
        value["format"] == FORMAT
        and type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and value["status"] == "natural-exit-complete"
        and value["phase"] == phase,
        "completion-format",
    )
    require(
        all(value[k] == expected[k] for k in BINDINGS),
        "completion-expectation-mismatch",
    )
    require(
        isinstance(value["nonce"], str)
        and re.fullmatch("[0-9a-f]{32}", value["nonce"])
        and checksum(value["outer_intent_sha256"])
        and checksum(value["qualified_outer_source_sha256"]),
        "completion-intent-source",
    )
    start = clock(value["phase_start"])
    clock(now)
    require(start["boot_id"] == now["boot_id"] == value["boot_id"], "completion-boot")
    enclosing = shape(
        value["enclosing"], {"operator", "supervisor"}, "completion-enclosing-fields"
    )
    operator = identity(enclosing["operator"])
    supervisor = identity(enclosing["supervisor"])
    require(operator["ppid"] == supervisor["pid"], "completion-operator-parent")
    contracts = shape(
        value["exec_contracts"], {"guardian", "collector"}, "completion-exec-fields"
    )
    require(all(checksum(x) for x in contracts.values()), "completion-exec-hash")
    artifacts = shape(value["artifacts"], ARTIFACTS, "completion-artifact-fields")
    for name, item in artifacts.items():
        pin(item, 4 * 2**20 if name == "provenance" else MAX_RECEIPT)
    require(
        len({p["path"] for p in artifacts.values()}) == len(ARTIFACTS),
        "completion-artifact-alias",
    )
    capture = path(artifacts["capture"]["path"])
    stem = "r3-" + phase
    require(
        capture.name == stem + ".json"
        and path(artifacts["receipt"]["path"])
        == capture.parent / (stem + ".receipt.json")
        and path(artifacts["provenance"]["path"])
        == capture.parent
        / (stem + ".provenance-" + artifacts["provenance"]["sha256"] + ".json"),
        "completion-output-location",
    )
    if phase == "before":
        require(
            all(
                value[k] is None
                for k in ("plan_sha256", "anchor_sha256", "before_execution_pin")
            ),
            "completion-before-plan-cycle",
        )
        want = before_budget(start)
        consume = want["gate"]
    else:
        require(
            anchor is not None
            and checksum(value["plan_sha256"])
            and checksum(value["anchor_sha256"])
            and sha(encoded(anchor)) == value["anchor_sha256"],
            "completion-after-anchor",
        )
        assert anchor is not None
        require(
            anchor["nonce"] == value["nonce"]
            and anchor["plan_sha256"] == value["plan_sha256"],
            "completion-after-plan",
        )
        pin(value["before_execution_pin"])
        want = after_budget(start, anchor)
        consume = {
            "monotonic_ns": math.floor(anchor["started_monotonic"] * 10**9)
            + 570 * 10**9,
            "wall_ns": anchor["started_wall_ns"] + 570 * 10**9,
        }
    require(value["budget"] == want, "completion-original-budget")
    require(
        all(start[k] < want["work"][k] for k in ("monotonic_ns", "wall_ns")),
        "completion-no-work-window",
    )
    shape(evidence, EVIDENCE, "completion-evidence-inputs")
    pins = evidence_pins(raw)
    docs = {}
    for name in EVIDENCE:
        raw_doc = evidence[name]
        p = pins[name]
        require(
            type(raw_doc) is bytes
            and len(raw_doc) == p["bytes"]
            and sha(raw_doc) == p["sha256"],
            "completion-evidence-hash",
        )
        require(
            path(p["path"])
            == capture.parent
            / "execution-evidence"
            / (phase + "-" + name.replace("_", "-") + ".json"),
            "completion-evidence-location",
        )
        docs[name] = parse(raw_doc, MAX_EVIDENCE)
    guardian = identity(docs["collector_family"]["role_admission"]["self"])
    child = identity(docs["collector_family"]["child_started"]["child"])
    require(
        len({x["pid"] for x in (supervisor, operator, guardian, child)}) == 4,
        "completion-process-alias",
    )
    require(
        all(
            x["boot_id"] == value["boot_id"]
            and x["uid"] == 0
            and x["pid_namespace_inode"] == operator["pid_namespace_inode"]
            and x["cgroup"] == operator["cgroup"]
            for x in (supervisor, operator, guardian, child)
        ),
        "completion-process-context",
    )
    require(
        supervisor["start_ticks"]
        <= operator["start_ticks"]
        <= guardian["start_ticks"]
        <= child["start_ticks"],
        "completion-process-birth-order",
    )
    binding = {
        k: value[k]
        for k in (
            "phase",
            "nonce",
            "boot_id",
            "outer_intent_sha256",
            "qualified_outer_source_sha256",
            "phase_start",
        )
    }
    c_events = _family(
        docs["collector_family"],
        role="guardian",
        owner=guardian,
        child=child,
        exec_hash=contracts["collector"],
        binding=binding,
        budget=want,
    )
    g_events = _family(
        docs["guardian_family"],
        role="operator",
        owner=operator,
        child=guardian,
        exec_hash=contracts["guardian"],
        binding=binding,
        budget=want,
    )
    _admission(docs["supervisor_admission"], "supervisor", supervisor)
    order(
        docs["supervisor_admission"]["clock"],
        docs["guardian_family"]["role_admission"]["clock"],
    )
    order(g_events[1]["clock"], docs["collector_family"]["role_admission"]["clock"])
    order(c_events[3]["clock"], g_events[2]["clock"])
    require(
        value["terminal_clock"] == g_events[3]["clock"], "completion-terminal-clock"
    )
    order(value["terminal_clock"], value["published_clock"])
    limit(value["published_clock"], want["gate"], strict=True)
    order(value["published_clock"], now)
    limit(now, consume, strict=True)
    return VerifiedCompletion(sha(raw), phase, value["boot_id"], encoded(value))


def _validate(*args, **kwargs):
    try:
        return _validate_impl(*args, **kwargs)
    except CompletionRefusal:
        raise
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
        raise CompletionRefusal("completion-malformed") from None


def validate_before(
    raw: bytes,
    expected_bindings: Mapping[str, Any],
    now_clock: Mapping[str, Any],
    *,
    evidence: Mapping[str, bytes],
) -> VerifiedCompletion:
    return _validate(
        raw, expected_bindings, now_clock, phase="before", evidence=evidence
    )


def validate_after(
    raw: bytes,
    expected_bindings: Mapping[str, Any],
    now_clock: Mapping[str, Any],
    *,
    evidence: Mapping[str, bytes],
    anchor: Mapping[str, Any],
) -> VerifiedCompletion:
    return _validate(
        raw,
        expected_bindings,
        now_clock,
        phase="after",
        evidence=evidence,
        anchor=anchor,
    )
