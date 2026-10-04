"""Pure observed BEFORE safe-body codec; no IO, publication or authority grant.

The consumer must independently authenticate source, input pins and actual
producer exit. These functions inspect complete safe data, not discarded raw
history. Old birth-bound formats and verification entrypoints are untouched.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import re
from pathlib import PurePosixPath
from typing import Any

from scripts import strength_freshness_cpu_completion as completion
from scripts import strength_freshness_cpu_collect_identity as identity
from scripts import strength_freshness_cpu_collect_records as records
from scripts import strength_freshness_cpu_collect_support as support
from scripts import strength_freshness_cpu_learner_window as append
from scripts import strength_freshness_cpu_observed_registration as observed
from scripts import strength_freshness_cpu_preservation as preservation

CONTRACT = observed.CONTRACT
CAPTURE = "strength-preservation-observed-window-capture-v1"
PROVENANCE = "strength-preservation-observed-window-provenance-bundle-v1"
RECEIPT = "strength-preservation-observed-window-collector-receipt-v1"
FENCE = "strength-observed-before-fence-commitment-v1"
KERNEL = "strength-observed-kernel-safe-projection-v1"
PRIVACY = "raw-private; source-process-attested; sampled-fixed-E; no-independent-raw-replay; no-historical-birth-or-execution-qualification"
MAX_JSON = 2**20
MAX_PROVENANCE = 4 * MAX_JSON
BINDINGS = frozenset(
    "nonce boot_id request_sha256 launch_sha256 registration_sha256 policy_sha256 source_pins_sha256 external_premises_sha256 writer_evidence_sha256 recipe_sha256 verified_champions_sha256 expected_window_sha256 source_contract_sha256 access_sha256 writer_qualification_sha256 source_qualification_sha256".split()
)
EXPECTED = frozenset(
    "contract bindings expected_window writer_binding expected_policy verified_champions expected_publication_writers".split()
)
CAPTURE_FIELDS = frozenset(
    "format schema_version contract phase bindings read_start read_end common metric_window".split()
)
PROVENANCE_FIELDS = frozenset(
    "format schema_version contract phase bindings capture_pin read_start read_end audit_inventory kernel_projections source_checks renewal owner_observations metric_spans support_witnesses support_provenance support_age_projection privacy_scope".split()
)
RECEIPT_FIELDS = frozenset(
    "format schema_version contract phase bindings status capture_pin provenance_pin read_start read_end refusals privacy_scope".split()
)
OWNER_FIELDS = frozenset(
    "pid start_ticks ppid cgroup invocation_id origin_sha256 uids namespaces clock_ticks_per_second".split()
)
KERNEL_FIELDS = frozenset(
    "format measurement_sha256 registration_sha256 external_premises_sha256 window_sha256 read_start read_end owners monitor gpu_owners auxiliaries unit_facts source_facts audit_inventory".split()
)
UNIT_COMMON = tuple(
    dict.fromkeys(
        (*support.DYNAMIC, "LoadState", "NeedDaemonReload", "UnitFileState", "Result")
    )
)
UNIT_SERVICE = tuple(dict.fromkeys((*UNIT_COMMON, *support.SERVICE_DYNAMIC)))
UNIT_STATIC = frozenset(
    "definition_sha256 environment_sha256 boot_links_sha256 enabled".split()
)
SOURCE_FIELDS = frozenset(
    "source_pins cached_references unit_static default_target origins".split()
)
SOURCE_PIN_FIELDS = frozenset(
    "literal_path_sha256 resolved_path_sha256 sha256 bytes mode uid gid".split()
)
CACHE_FIELDS = frozenset(
    "literal_path_sha256 resolved_path_sha256 sha256 qualification_sha256 literal_stat resolved_stat content_hashed".split()
)
BASE_AUDIT = frozenset("operation subject read_start read_end raw".split())
FOUR = ("named_before", "stat_before", "stat_after", "named_after")
STAT = append.STAT
SAFE_FILE = frozenset("path_sha256 device inode uid gid mode".split())
SAFE_OWNER = (append.OWNER - {"cgroup"}) | {"cgroup_sha256"}
STREAMS = ("coordinator", *records.WORKERS, *records.COHORTS)


class BodyRefusal(ValueError):
    """Fixed reason only; never include captured private text."""


def require(ok: object, reason: str) -> None:
    if not ok:
        raise BodyRefusal(reason)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _bounded(value: Any) -> None:
    pending = [(value, 0)]
    nodes = 0
    string_bytes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        require(depth <= 64 and nodes <= 250000, "body-structure-bound")
        if type(item) is str:
            require(len(item) <= MAX_PROVENANCE, "body-string-bound")
            string_bytes += len(item.encode("utf-8"))
            require(string_bytes <= MAX_PROVENANCE, "body-string-budget")
        elif type(item) is dict:
            require(all(type(k) is str for k in item), "body-json-key")
            pending.extend((x, depth + 1) for pair in item.items() for x in pair)
        elif type(item) is list:
            pending.extend((x, depth + 1) for x in item)
        elif item is None or type(item) in (bool, int):
            pass
        elif type(item) is float:
            require(preservation.finite(item), "body-nonfinite")
        else:
            raise BodyRefusal("body-json-value")


def encoded(value: Any) -> bytes:
    try:
        _bounded(value)
        raw = append.encoded(value)
        require(len(raw) <= MAX_PROVENANCE, "body-encoded-budget")
        return raw
    except (TypeError, ValueError, RecursionError):
        raise BodyRefusal("body-canonical-json") from None


def digest(value: Any) -> str:
    return sha(encoded(value))


def same(a: Any, b: Any) -> bool:
    return encoded(a) == encoded(b)


def fields(value: Any, keys: Any, reason: Any = "body-fields") -> dict[str, Any]:
    require(is_dict(value) and set(value) == set(keys), reason)
    return value


def is_dict(value: Any) -> bool:
    return type(value) is dict


def is_list(value: Any) -> bool:
    return type(value) is list


def integer(value: Any, minimum: Any = 0) -> bool:
    return type(value) is int and minimum <= value <= 2**64 - 1


def checksum(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def parse(raw: bytes, maximum: Any = MAX_JSON) -> dict[str, Any]:
    try:
        return completion.parse(raw, maximum)
    except (ValueError, TypeError, RecursionError):
        raise BodyRefusal("body-json") from None


def clock(value: Any) -> dict[str, Any]:
    fields(value, append.CLOCK, "body-clock-fields")
    require(
        type(value["boot_id"]) is str
        and 0 < len(value["boot_id"]) <= 64
        and integer(value["monotonic_ns"])
        and integer(value["wall_ns"], 1),
        "body-clock",
    )
    return value


def order(a: Any, b: Any) -> None:
    clock(a)
    clock(b)
    require(
        a["boot_id"] == b["boot_id"]
        and all(a[k] <= b[k] for k in ("monotonic_ns", "wall_ns")),
        "body-clock-order",
    )


def within(value: Any, window: Any) -> None:
    order(window["phase_start"], value)
    require(
        all(value[k] < window["deadline"][k] for k in ("monotonic_ns", "wall_ns")),
        "body-original-deadline",
    )


def pin(path: Any, raw: Any) -> Any:
    p = PurePosixPath(path)
    require(
        p.is_absolute() and str(p) == path and ".." not in p.parts and "\0" not in path,
        "body-path",
    )
    return {"path": path, "sha256": sha(raw), "bytes": len(raw)}


def _header(value: Any, fmt: Any, exact: Any) -> Any:
    fields(value, exact)
    require(
        value["format"] == fmt
        and type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and value["contract"] == CONTRACT
        and value["phase"] == "before",
        "body-format",
    )


def _expected(expected: Any, registration: Any) -> Any:
    fields(expected, EXPECTED, "body-expected-fields")
    b = fields(expected["bindings"], BINDINGS, "body-binding-fields")
    require(
        expected["contract"] == CONTRACT
        and type(b["nonce"]) is str
        and re.fullmatch(r"[0-9a-f]{32}", b["nonce"]) is not None
        and all(checksum(v) for k, v in b.items() if k not in {"nonce", "boot_id"}),
        "body-bindings",
    )
    require(
        type(registration) is observed.ObservedRegistration, "body-registration-type"
    )
    reg = registration.private_copy()
    checked = observed.parse_observed_registration(
        registration._raw,
        approved_sha256=b["registration_sha256"],
        scope=identity.scope_from(reg["scope"]),
        writer_evidence=registration.private_writer_evidence(),
    )
    require(same(checked.private_copy(), reg), "body-registration-canonical")
    w = fields(
        expected["expected_window"],
        {"phase", "phase_start", "deadline", "cleanup_clock"},
    )
    require(w["phase"] == "before" and w["cleanup_clock"] is None, "body-before-window")
    order(w["phase_start"], w["deadline"])
    require(
        b["boot_id"] == w["phase_start"]["boot_id"] == reg["kernel_context"]["boot_id"]
        and all(
            0 < w["deadline"][k] - w["phase_start"][k] <= 105 * 10**9
            for k in ("monotonic_ns", "wall_ns")
        ),
        "body-window",
    )
    writer = reg["learner_writer"]["binding"]
    require(
        same(expected["writer_binding"], writer)
        and same(expected["expected_policy"], reg["policy"])
        and same(expected["expected_publication_writers"], reg["publication_writers"]),
        "body-external-expectation",
    )
    champions = expected["verified_champions"]
    require(
        is_dict(champions)
        and champions
        and all(records._identity(k) and checksum(v) for k, v in champions.items()),
        "body-champions",
    )
    joins = {
        "policy_sha256": digest(reg["policy"]),
        "recipe_sha256": digest(reg["policy"]["recipe"]),
        "writer_evidence_sha256": sha(registration.private_writer_evidence()),
        "expected_window_sha256": digest(w),
        "verified_champions_sha256": digest(champions),
        "source_contract_sha256": writer["source_contract_sha256"],
        "access_sha256": writer["access_sha256"],
        "writer_qualification_sha256": writer["qualification_sha256"],
    }
    require(all(b[k] == v for k, v in joins.items()), "body-external-digest-join")
    return reg


def safe_file(raw: Any) -> Any:
    return {
        "path_sha256": sha(raw["path"].encode()),
        **{k: raw[k] for k in SAFE_FILE - {"path_sha256"}},
    }


def safe_owner(raw: Any) -> Any:
    return {
        "cgroup_sha256": sha(raw["cgroup"].encode()),
        **{k: raw[k] for k in SAFE_OWNER - {"cgroup_sha256"}},
    }


def project_append(envelope: Any, writer: Any) -> Any:
    """Projection of actual private append bytes; not writer admission."""
    try:
        kind = envelope["audit"]["operation"]
        require(kind in {"append-fence", "append-range"}, "body-append-kind")
        value, audit, span = append._observation(envelope, kind, writer, [0])
        file = safe_file(value["file_identity"])
        if kind == "append-fence":
            return {
                "file": file,
                "fence": {
                    "registered_key": value["registered_key"],
                    "size_at_fstat": value["size_at_fstat"],
                    "line_boundary": value["line_boundary"],
                    "prefix": {k: span[k] for k in ("start", "end", "sha256")}
                    | {"bytes": len(span["raw"])},
                    "audit": deepcopy(audit),
                },
            }
        return {
            "file": file,
            "span": {
                k: value[k]
                for k in (
                    "registered_key",
                    "start",
                    "end",
                    "sha256",
                    "size_before",
                    "size_after",
                )
            }
            | {"bytes": len(value["raw"]), "audit": deepcopy(audit)},
        }
    except (KeyError, TypeError, ValueError, AttributeError):
        raise BodyRefusal("body-append-projection") from None


def project_owner(raw: Any, kernel_projection: Any) -> Any:
    fields(raw, {"owner_key", "writer_evidence_sha256", "read_start", "read_end"})
    return {
        **deepcopy(raw),
        "owner_key": safe_owner(raw["owner_key"]),
        "kernel_projection_sha256": digest(kernel_projection),
    }


def project_metric(raw_line: bytes, *, record_ref: Any, recipe: Any) -> Any:
    fields(record_ref, {"start", "end", "sha256"})
    require(
        type(raw_line) is bytes
        and raw_line.endswith(b"\n")
        and record_ref["end"] - record_ref["start"] == len(raw_line)
        and record_ref["sha256"] == sha(raw_line),
        "body-metric-raw-ref",
    )
    row = records.parse_json(raw_line)
    is_loss, _ = append._loss(row, recipe)
    require(
        is_loss
        and row["worker"] == "learner"
        and type(row["schema_version"]) is int
        and row["schema_version"] == 1,
        "body-loss-required",
    )
    keys = records.METRIC_FIELDS - {"schema_version"}
    result = {k: deepcopy(row[k]) for k in keys}
    result["losses"] = {sha(k.encode()): v for k, v in row["losses"].items()}
    require(len(result["losses"]) == len(row["losses"]) <= 64, "body-loss-key-count")
    d = row["gradient_diagnostics"]
    batch = d["batch"]
    result["gradient_diagnostics"] = {
        "global_norm_finite": d["global_norm_finite"],
        "nonfinite_gradient_tensors": d["nonfinite_gradient_tensors"],
        "batch": {
            "rows": batch["rows"],
            "six_mode_unknown": {},
            "label_availability": {
                k: batch["label_availability"][k] for k in ("policy", "outcome")
            },
        },
    }
    result["policy_batch_metrics"] = {
        "unknown_provenance_rows": row["policy_batch_metrics"][
            "unknown_provenance_rows"
        ]
    }
    result["ema"] = {k: row["ema"][k] for k in ("decay", "num_updates")}
    result["record_ref"] = deepcopy(record_ref)
    return result


def checked_expected(expected: Any, registration: Any) -> Any:
    """Validate independent expected-data consistency; never issue authority."""
    _expected(expected, registration)
    return deepcopy(expected)


def _stats(st: Any, *, complete: Any = True) -> Any:
    fields(st, STAT if complete else identity.FINGERPRINT, "body-stat-fields")
    require(
        all(integer(v) for v in st.values())
        and st["inode"] > 0
        and st["mode"] <= 0o7777
        and st["uid"] < 2**32
        and st["gid"] < 2**32
        and (not complete or st["links"] > 0),
        "body-stat",
    )


def _audit(a: Any, reg: Any, window: Any) -> Any:
    op = a.get("operation")
    extras = {
        "clock": set(),
        "birth-bracket": set(),
        "namespaces": set(),
        "boot-links": set(),
        "read": {"stat_before", "stat_after", "offset", "end_offset"},
        "tail": {"stat_before", "stat_after", "offset", "end_offset"},
        "read-publication": set(FOUR) | {"offset", "end_offset"},
        "append-fence": set(FOUR) | {"offset", "end_offset"},
        "append-range": set(FOUR) | {"offset", "end_offset"},
        "query": {"returncode", "stderr_sha256", "stderr_bytes"},
        "stat-cached": {"content_hashed"},
        "process": {"raw_encoding", "maps_read"},
        "process-credentials": {"raw_encoding", "components", "uids"},
        "manager-identity": {"raw_encoding"},
        "cgroup-members": {"raw_encoding"},
    }
    require(op in extras, "body-audit-operation")
    fields(a, BASE_AUDIT | extras[op], "body-audit-fields")
    subject = a["subject"]
    slots = (
        set(reg["scope"]["files"])
        | set(reg["scope"]["tails"])
        | set(reg["scope"]["cached"])
    )
    units = set(reg["units"])
    pids = {str(v["pid"]) for v in reg["policy"]["expected_processes"].values()} | {
        str(v["pid"]) for v in reg["policy"]["expected_monitors"].values()
    }
    if op in {
        "read",
        "tail",
        "stat-cached",
        "read-publication",
        "append-fence",
        "append-range",
    }:
        require(subject in slots, "body-audit-subject")
    elif op == "query":
        require(
            subject
            in {"jobs", "default-target", "gpu-inventory", "gpu-owners"}
            | {"unit:" + x for x in units}
            | {"registered-target:" + x for x in reg["scope"]["targets"]},
            "body-audit-subject",
        )
    elif op in {"boot-links", "cgroup-members"}:
        require(subject in units, "body-audit-subject")
    elif op in {"process", "process-credentials"}:
        require(
            type(subject) is str
            and subject.isascii()
            and subject.isdecimal()
            and int(subject) > 0,
            "body-audit-pid",
        )
    elif op == "namespaces":
        require(
            subject in pids | {"self", "1"}
            or type(subject) is str
            and subject.isascii()
            and subject.isdecimal(),
            "body-audit-subject",
        )
    elif op == "manager-identity":
        require(subject == "pid1", "body-audit-subject")
    else:
        require(subject == "collector", "body-audit-subject")
    order(a["read_start"], a["read_end"])
    within(a["read_start"], window)
    within(a["read_end"], window)
    fields(a["raw"], {"sha256", "bytes"})
    require(
        checksum(a["raw"]["sha256"]) and integer(a["raw"]["bytes"]), "body-audit-raw"
    )
    for k in ("stat_before", "stat_after", "named_before", "named_after"):
        if k in a:
            _stats(a[k])
    if "offset" in a:
        require(
            integer(a["offset"])
            and integer(a["end_offset"])
            and a["offset"] <= a["end_offset"]
            and a["end_offset"] - a["offset"] == a["raw"]["bytes"],
            "body-audit-offset",
        )
    if op == "query":
        require(
            type(a["returncode"]) is int
            and a["returncode"] == 0
            and checksum(a["stderr_sha256"])
            and integer(a["stderr_bytes"]),
            "body-query-result",
        )
    if "raw_encoding" in a:
        require(a["raw_encoding"] == "component-pins-v1", "body-raw-encoding")
    if op == "process":
        require(type(a["maps_read"]) is bool, "body-maps-selection")
    if op == "stat-cached":
        require(a["content_hashed"] is False, "body-cached-not-fresh")
    if op == "process-credentials":
        fields(a["uids"], observed.UID_FIELDS)
        require(
            all(integer(v) and v < 2**32 for v in a["uids"].values()),
            "body-credentials",
        )
        fields(
            a["components"],
            {"stat_before", "cgroup_before", "status", "stat_after", "cgroup_after"},
        )
        for row in a["components"].values():
            fields(row, {"sha256", "bytes"})
            require(
                checksum(row["sha256"]) and integer(row["bytes"]), "body-component-pin"
            )
    if op == "clock":
        require(
            a["raw"]
            == {"sha256": digest(a["read_end"]), "bytes": len(encoded(a["read_end"]))},
            "body-clock-raw",
        )


def _audits(rows: Any, reg: Any, window: Any) -> Any:
    require(is_list(rows) and 0 < len(rows) <= 16384, "body-audit-count")
    previous = None
    for a in rows:
        _audit(a, reg, window)
        if previous is not None:
            order(previous, a["read_start"])
        previous = a["read_end"]
    return Counter(digest(a) for a in rows)


def _membership(subset: Any, inventory: Any, reason: Any) -> Any:
    counts = Counter(digest(a) for a in subset)
    require(all(n <= inventory[k] for k, n in counts.items()), reason)


def _source_facts(sf: Any, reg: Any, owners: Any, audits: Any) -> Any:
    fields(sf, SOURCE_FIELDS, "body-source-fields")
    require(
        set(sf["source_pins"]) == set(reg["source_pins"])
        and set(sf["cached_references"]) == set(reg["cached_references"])
        and set(sf["unit_static"]) == set(reg["units"])
        and sf["default_target"] == reg["boot"]["default_target"]
        and same(
            sf["origins"], {name: o["origin_sha256"] for name, o in owners.items()}
        ),
        "body-source-inventory",
    )
    for key, row in sf["source_pins"].items():
        fields(row, SOURCE_PIN_FIELDS)
        path = reg["scope"]["files"][key]["path"]
        require(
            row["literal_path_sha256"]
            == row["resolved_path_sha256"]
            == sha(path.encode())
            and {k: row[k] for k in ("sha256", "bytes")} == reg["source_pins"][key]
            and all(integer(row[k]) for k in ("mode", "uid", "gid")),
            "body-source-pin",
        )
        require(
            any(
                a["operation"] == "read"
                and a["subject"] == key
                and a["raw"] == {k: row[k] for k in ("sha256", "bytes")}
                and all(a["stat_before"][k] == row[k] for k in ("mode", "uid", "gid"))
                for a in audits
            ),
            "body-source-read-membership",
        )
    for key, row in sf["cached_references"].items():
        fields(row, CACHE_FIELDS)
        ref = reg["cached_references"][key]
        paths = reg["scope"]["cached"][key]
        require(
            row["literal_path_sha256"] == sha(paths["literal"].encode())
            and row["resolved_path_sha256"] == sha(paths["resolved"].encode())
            and row["sha256"] == ref["sha256"]
            and row["qualification_sha256"] == ref["qualification_sha256"]
            and row["content_hashed"] is False,
            "body-cache-reference",
        )
        for key2 in ("literal_stat", "resolved_stat"):
            _stats(row[key2], complete=False)
            require(same(row[key2], ref[key2]), "body-cache-stat")
        require(
            any(
                a["operation"] == "stat-cached"
                and a["subject"] == key
                and a["content_hashed"] is False
                for a in audits
            ),
            "body-cache-membership",
        )
    native_keys = {
        key for origin in reg["origins"].values() for key in origin["native_keys"]
    }
    require(
        native_keys
        and all(
            sf["cached_references"][key]["sha256"]
            == reg["policy"]["static"]["native_sha256"]
            for key in native_keys
        ),
        "body-native-policy",
    )
    for name, row in sf["unit_static"].items():
        fields(row, UNIT_STATIC)
        expected = (
            (
                {
                    k: reg["policy"]["static"]["runtime_" + k]
                    for k in UNIT_STATIC - {"enabled"}
                }
                | {"enabled": row["enabled"]}
            )
            if reg["units"][name]["kind"] == "runtime"
            else {k: reg["policy"]["support"][name][k] for k in UNIT_STATIC}
        )
        require(
            type(row["enabled"]) is bool and same(row, expected), "body-unit-static"
        )


def _safe_unit_properties(name: Any, props: Any) -> None:
    require(
        all(type(v) is str and len(v) <= 512 for v in props.values())
        and props["Id"] == name,
        "body-unit-properties",
    )
    require(
        "Result" not in props
        or props["Result"]
        in {
            "",
            "success",
            "resources",
            "protocol",
            "timeout",
            "exit-code",
            "signal",
            "core-dump",
            "watchdog",
            "start-limit-hit",
            "oom-kill",
            "exec-condition",
        },
        "body-unit-result-enum",
    )
    require(
        "UnitFileState" not in props
        or props["UnitFileState"]
        in {
            "enabled",
            "enabled-runtime",
            "disabled",
            "static",
            "indirect",
            "generated",
            "transient",
            "linked",
            "linked-runtime",
            "alias",
            "masked",
            "masked-runtime",
        },
        "body-unit-enable-enum",
    )
    require(
        props["ActiveState"]
        in {
            "active",
            "inactive",
            "activating",
            "deactivating",
            "reloading",
            "failed",
            "maintenance",
            "refreshing",
        },
        "body-unit-active-enum",
    )
    require(
        props["SubState"]
        in {
            "running",
            "dead",
            "exited",
            "waiting",
            "elapsed",
            "failed",
            "start-pre",
            "start",
            "start-post",
            "stop",
            "stop-sigterm",
            "stop-sigkill",
            "stop-post",
            "final-sigterm",
            "final-sigkill",
            "auto-restart",
            "auto-restart-queued",
            "condition",
            "cleaning",
            "reload",
            "reload-signal",
            "reload-notify",
        },
        "body-unit-substate-enum",
    )
    require(
        props["InvocationID"] == ""
        or re.fullmatch(r"[0-9a-f]{32}", props["InvocationID"]) is not None,
        "body-unit-invocation",
    )
    require(
        props["Job"] == "" or props["Job"].isascii() and props["Job"].isdecimal(),
        "body-unit-job",
    )
    if "ControlGroup" in props:
        require(
            props["ControlGroup"] in {"", "/system.slice/" + name},
            "body-unit-cgroup",
        )
    for key in props:
        if key.endswith("Monotonic") or key in {
            "MainPID",
            "ExecMainPID",
            "NRestarts",
            "ExecMainCode",
            "ExecMainStatus",
        }:
            require(props[key].isascii() and props[key].isdecimal(), "body-unit-number")


def _unit_facts(table: Any, reg: Any, owners: Any, sf: Any) -> Any:
    require(set(table) == set(reg["units"]), "body-unit-inventory")
    for name, row in table.items():
        fields(row, {"kind", "properties", "static"})
        kind = reg["units"][name]["kind"]
        require(
            row["kind"] == kind and same(row["static"], sf["unit_static"][name]),
            "body-unit-kind-static",
        )
        props = fields(
            row["properties"], UNIT_COMMON if kind == "timer" else UNIT_SERVICE
        )
        require(
            all(type(v) is str and len(v) <= 512 for v in props.values())
            and props["Id"] == name
            and props["LoadState"] == "loaded"
            and props["NeedDaemonReload"] == "no",
            "body-unit-properties",
        )
        require(
            row["static"]["enabled"]
            == (props["UnitFileState"] in {"enabled", "enabled-runtime"}),
            "body-unit-enablement",
        )
        _safe_unit_properties(name, props)
        if kind in {"runtime", "long_running"}:
            role = "controller" if kind == "runtime" else "monitor"
            o = owners[role]
            require(
                props["ActiveState"] == "active"
                and props["SubState"] == "running"
                and props["Result"] == "success"
                and int(props["MainPID"]) == int(props["ExecMainPID"]) == o["pid"]
                and props["InvocationID"] == o["invocation_id"]
                and props["ControlGroup"] == o["cgroup"],
                "body-unit-owner",
            )
            expected = (
                reg["policy"]["expected_processes"][role]
                if kind == "runtime"
                else reg["policy"]["expected_monitors"][name]
            )
            require(
                int(props["NRestarts"]) == expected["restarts"], "body-manager-restarts"
            )


def _kernel(value: Any, reg: Any, expected: Any, inventory: Any) -> Any:
    fields(value, KERNEL_FIELDS, "body-kernel-fields")
    b = expected["bindings"]
    w = expected["expected_window"]
    require(
        value["format"] == KERNEL
        and checksum(value["measurement_sha256"])
        and value["registration_sha256"] == b["registration_sha256"]
        and value["external_premises_sha256"] == b["external_premises_sha256"]
        and value["window_sha256"] == b["expected_window_sha256"],
        "body-kernel-binding",
    )
    order(value["read_start"], value["read_end"])
    within(value["read_start"], w)
    within(value["read_end"], w)
    owners = value["owners"]
    require(set(owners) == set(observed.ROLES), "body-owner-roster")
    declared = {
        **reg["policy"]["expected_processes"],
        "monitor": next(iter(reg["policy"]["expected_monitors"].values())),
    }
    for role, o in owners.items():
        fields(o, OWNER_FIELDS)
        e = declared[role]
        require(
            all(
                type(o[k]) is int and o[k] > 0
                for k in ("pid", "start_ticks", "ppid", "clock_ticks_per_second")
            )
            and all(
                same(o[k], e[k])
                for k in (
                    "pid",
                    "start_ticks",
                    "cgroup",
                    "invocation_id",
                    "origin_sha256",
                )
            ),
            "body-owner",
        )
        fields(o["uids"], observed.UID_FIELDS)
        fields(o["namespaces"], {"pid", "time"})
        require(
            same(o["uids"], reg["kernel_context"]["credentials"][role])
            and same(
                o["namespaces"],
                reg["kernel_context"]["namespace_expectations"]["owners"][role],
            )
            and o["clock_ticks_per_second"]
            == reg["kernel_context"]["clock_ticks_per_second"],
            "body-owner-context",
        )
        if role not in {"controller", "monitor"}:
            require(
                o["ppid"]
                == owners["controller" if role == "coordinator" else "coordinator"][
                    "pid"
                ],
                "body-owner-parent",
            )
    require(len({o["pid"] for o in owners.values()}) == 12, "body-owner-unique")
    require(same(value["monitor"], reg["policy"]["expected_monitors"]), "body-monitor")
    _gpu(value["gpu_owners"], reg, owners)
    require(
        is_list(value["auxiliaries"]) and len(value["auxiliaries"]) <= 244,
        "body-auxiliary-count",
    )
    known = {o["pid"]: role for role, o in owners.items()}
    pending = deepcopy(value["auxiliaries"])
    while pending:
        advance = False
        for row in pending[:]:
            fields(row, {"pid", "start_ticks", "ppid", "parent_role", "kind"})
            require(
                all(integer(row[k], 1) for k in ("pid", "start_ticks", "ppid"))
                and row["pid"] not in known
                and row["kind"]
                in {
                    "torch-inductor-pool",
                    "multiprocessing-resource-tracker",
                    "multiprocessing-spawn",
                },
                "body-auxiliary",
            )
            if row["ppid"] not in known:
                continue
            require(row["parent_role"] == known[row["ppid"]], "body-auxiliary-parent")
            known[row["pid"]] = row["parent_role"]
            pending.remove(row)
            advance = True
        require(advance, "body-auxiliary-orphan")
    _audits(value["audit_inventory"], reg, w)
    _membership(value["audit_inventory"], inventory, "body-kernel-audit-membership")
    for index, a in enumerate(value["audit_inventory"]):
        if (
            index == 0
            and a["operation"] == "clock"
            and same(a["read_end"], value["read_start"])
        ):
            pass  # Original snapshot start is this actual leading clock's end.
        else:
            order(value["read_start"], a["read_start"])
        order(a["read_end"], value["read_end"])
    _source_facts(value["source_facts"], reg, owners, value["audit_inventory"])
    _unit_facts(value["unit_facts"], reg, owners, value["source_facts"])


def _gpu(rows: Any, reg: Any, owners: Any) -> Any:
    require(
        is_list(rows)
        and len(rows) == 8
        and {r["uuid"] for r in rows} == set(reg["policy"]["gpu_uuids"]),
        "body-gpu-roster",
    )
    for row in rows:
        fields(
            row,
            {"uuid", "role", "unit", "pid", "start_ticks", "cgroup", "invocation_id"},
        )
        require(
            row["role"] == reg["policy"]["gpu_roles"][row["uuid"]]
            and row["unit"] == reg["policy"]["static"]["runtime_name"]
            and all(
                same(row[k], owners[row["role"]][k])
                for k in ("pid", "start_ticks", "cgroup", "invocation_id")
            ),
            "body-gpu-owner",
        )


def _append_observation(
    value: Any, reg: Any, writer: Any, window: Any, *, fence: Any
) -> Any:
    fields(value, {"file", "fence" if fence else "span"})
    require(
        same(fields(value["file"], SAFE_FILE), safe_file(writer["file_identity"])),
        "body-append-file",
    )
    v = fields(
        value["fence" if fence else "span"],
        {"registered_key", "size_at_fstat", "line_boundary", "prefix", "audit"}
        if fence
        else {
            "registered_key",
            "start",
            "end",
            "bytes",
            "sha256",
            "size_before",
            "size_after",
            "audit",
        },
    )
    a = v["audit"]
    _audit(a, reg, window)
    require(
        v["registered_key"] == a["subject"] == "metrics"
        and a["operation"] == ("append-fence" if fence else "append-range"),
        "body-append-operation",
    )
    for name in FOUR:
        require(
            all(a[name][k] == value["file"][k] for k in SAFE_FILE - {"path_sha256"}),
            "body-append-stat-identity",
        )
    for first, last in zip(FOUR, FOUR[1:]):
        append.stat_continuity(a[first], a[last])
    if fence:
        span = fields(v["prefix"], {"start", "end", "bytes", "sha256"})
        cap = min(
            65536, identity.scope_from(reg["scope"]).tails["metrics"].maximum_bytes
        )
        require(
            integer(v["size_at_fstat"])
            and v["size_at_fstat"] == a["stat_before"]["bytes"] == span["end"]
            and span["start"] == max(0, span["end"] - cap)
            and type(v["line_boundary"]) is bool,
            "body-exact-fence-prefix",
        )
        if not v["size_at_fstat"]:
            require(v["line_boundary"] is True, "body-empty-boundary")
    else:
        span = v
        require(
            v["size_before"] == a["stat_before"]["bytes"]
            and v["size_after"] == a["stat_after"]["bytes"],
            "body-span-stat-size",
        )
    require(
        all(integer(span[k]) for k in ("start", "end", "bytes"))
        and span["start"] <= span["end"] <= a["stat_before"]["bytes"]
        and span["bytes"] == span["end"] - span["start"]
        and checksum(span["sha256"])
        and a["offset"] == span["start"]
        and a["end_offset"] == span["end"]
        and same(a["raw"], {"sha256": span["sha256"], "bytes": span["bytes"]}),
        "body-span-pin",
    )
    require(
        span["bytes"]
        <= identity.scope_from(reg["scope"]).tails["metrics"].maximum_bytes,
        "body-span-read-bound",
    )
    if span["bytes"] == 0:
        require(span["sha256"] == sha(b""), "body-empty-sha")
    return v


def _owner_observation(value: Any, expected: Any, projections: Any) -> Any:
    fields(
        value,
        {
            "owner_key",
            "writer_evidence_sha256",
            "read_start",
            "read_end",
            "kernel_projection_sha256",
        },
    )
    fields(value["owner_key"], SAFE_OWNER)
    require(
        same(value["owner_key"], safe_owner(expected["writer_binding"]["owner_key"]))
        and value["writer_evidence_sha256"]
        == expected["bindings"]["writer_evidence_sha256"],
        "body-writer-owner",
    )
    key = value["kernel_projection_sha256"]
    require(key in projections, "body-owner-kernel-body")
    p = projections[key]
    o = p["owners"]["learner"]
    raw = expected["writer_binding"]["owner_key"]
    require(
        all(
            same(o[k], raw[k])
            for k in (
                "pid",
                "start_ticks",
                "ppid",
                "cgroup",
                "invocation_id",
                "origin_sha256",
                "clock_ticks_per_second",
            )
        )
        and o["uids"]["real"] == raw["uid"]
        and o["namespaces"]
        == {"pid": raw["pid_namespace_inode"], "time": raw["time_namespace_inode"]}
        and same(value["read_start"], p["read_start"])
        and same(value["read_end"], p["read_end"]),
        "body-owner-kernel-join",
    )


def _b_and_spans(
    c: Any, prov: Any, reg: Any, expected: Any, projections: Any, inventory: Any
) -> Any:
    mw = fields(
        c["metric_window"],
        {"before_fence", "causal_fence_sha256", "end_fence", "append_proof"},
    )
    b = mw["before_fence"]
    _header(
        b,
        FENCE,
        {
            "format",
            "schema_version",
            "contract",
            "phase",
            "bindings",
            "file",
            "fence",
            "owner_before",
            "owner_after",
        },
    )
    require(
        same(b["bindings"], expected["bindings"])
        and mw["causal_fence_sha256"] == digest(b),
        "body-B-original-binding",
    )
    window = expected["expected_window"]
    writer = expected["writer_binding"]
    bf = _append_observation(
        {k: b[k] for k in ("file", "fence")}, reg, writer, window, fence=True
    )
    ef = _append_observation(mw["end_fence"], reg, writer, window, fence=True)
    require(
        bf["size_at_fstat"] <= ef["size_at_fstat"] and ef["line_boundary"] is True,
        "body-fixed-E",
    )
    order(bf["audit"]["read_end"], ef["audit"]["read_start"])
    append.stat_continuity(bf["audit"]["named_after"], ef["audit"]["named_before"])
    for name in ("owner_before", "owner_after"):
        _owner_observation(b[name], expected, projections)
    order(b["owner_before"]["read_end"], bf["audit"]["read_start"])
    order(bf["audit"]["read_end"], b["owner_after"]["read_start"])
    owners = prov["owner_observations"]
    require(is_list(owners) and 2 <= len(owners) <= 16, "body-owner-observation-count")
    require(
        any(same(x, b["owner_before"]) for x in owners)
        and any(same(x, b["owner_after"]) for x in owners),
        "body-B-owner-membership",
    )
    for n, row in enumerate(owners):
        _owner_observation(row, expected, projections)
        if n:
            order(owners[n - 1]["read_end"], row["read_start"])
    spans = fields(prov["metric_spans"], {"originals", "rereads", "end_guard"})
    cursor_start = bf["prefix"]["start"]
    end = ef["size_at_fstat"]
    last = ef["audit"]["read_end"]
    previous = ef["audit"]["named_after"]
    all_audits = [bf["audit"], ef["audit"]]
    raw_total = 2 * bf["prefix"]["bytes"] + ef["prefix"]["bytes"]
    original = []
    require(all(ef["audit"][k]["bytes"] == end for k in FOUR), "body-E-growth")
    for kind in ("originals", "rereads"):
        rows = spans[kind]
        require(is_list(rows) and 1 <= len(rows) <= 128, "body-span-count")
        cursor = cursor_start
        seen = []
        for row in rows:
            v = _append_observation(row, reg, writer, window, fence=False)
            a = v["audit"]
            require(
                v["start"] == cursor
                and (v["end"] > cursor or cursor == end and len(rows) == 1)
                and v["end"] <= end,
                "body-contiguous-spans",
            )
            require(all(a[k]["bytes"] == end for k in FOUR), "body-growth-after-E")
            order(last, a["read_start"])
            last = a["read_end"]
            append.stat_continuity(previous, a["named_before"])
            previous = a["named_after"]
            cursor = v["end"]
            raw_total += v["bytes"]
            all_audits.append(a)
            seen.append({k: v[k] for k in ("start", "end", "sha256", "bytes")})
        require(cursor == end, "body-span-coverage")
        if kind == "originals":
            original = seen
        else:
            require(same(original, seen), "body-span-reread")
    guard = _append_observation(spans["end_guard"], reg, writer, window, fence=False)
    require(
        guard["start"] == guard["end"] == end
        and guard["bytes"] == 0
        and all(guard["audit"][k]["bytes"] == end for k in FOUR),
        "body-final-E-guard",
    )
    order(last, guard["audit"]["read_start"])
    append.stat_continuity(previous, guard["audit"]["named_before"])
    order(guard["audit"]["read_end"], owners[-1]["read_start"])
    order(owners[-1]["read_end"], c["read_end"])
    all_audits.append(guard["audit"])
    _membership(all_audits, inventory, "body-append-audit-membership")
    proof = mw["append_proof"]
    fields(
        proof,
        {
            "format",
            "schema_version",
            "status",
            "full_preservation",
            "execution_qualified",
            "historical_birth_qualified",
            "source_contract_sha256",
            "writer_evidence_sha256",
            "owner_sha256",
            "file_identity_sha256",
            "phase",
            "before_offset",
            "causal_offset",
            "final_offset",
            "coverage_end_monotonic_ns",
            "coverage_end_wall_ns",
            "final_observation_monotonic_ns",
            "final_observation_wall_ns",
            "coverage_start",
            "unparsed_pre_window_prefix_bytes",
            "raw_bytes_checked_including_rereads",
            "complete_records_checked",
            "loss_records_checked",
            "qualifying_records",
            "minimum_serial_intervals",
            "window_digest",
            "history_scope",
            "writer_authority_granted",
        },
        "body-append-proof-fields",
    )
    require(
        proof["format"] == append.PROOF
        and type(proof["schema_version"]) is int
        and proof["schema_version"] == 1
        and proof["status"] == "conditional-serial-append-observed"
        and proof["phase"] == "before"
        and all(
            proof[k] is False
            for k in (
                "full_preservation",
                "execution_qualified",
                "historical_birth_qualified",
                "writer_authority_granted",
            )
        )
        and proof["history_scope"] == "sampled-since-before-fence-only",
        "body-proof-scope",
    )
    joins = {
        "source_contract_sha256": append.SOURCE_CONTRACT,
        "writer_evidence_sha256": expected["bindings"]["writer_evidence_sha256"],
        "owner_sha256": digest(writer["owner_key"]),
        "file_identity_sha256": digest(writer["file_identity"]),
        "before_offset": bf["size_at_fstat"],
        "causal_offset": bf["size_at_fstat"],
        "final_offset": end,
        "coverage_start": cursor_start,
        "coverage_end_monotonic_ns": ef["audit"]["read_end"]["monotonic_ns"],
        "coverage_end_wall_ns": ef["audit"]["read_end"]["wall_ns"],
        "final_observation_monotonic_ns": c["read_end"]["monotonic_ns"],
        "final_observation_wall_ns": c["read_end"]["wall_ns"],
        "minimum_serial_intervals": 1,
        "window_digest": digest(
            {
                "start": window["phase_start"],
                "deadline": window["deadline"],
                "final": c["read_end"],
            }
        ),
    }
    require(
        all(same(proof[k], v) for k, v in joins.items()), "body-append-proof-binding"
    )
    require(
        integer(proof["unparsed_pre_window_prefix_bytes"])
        and proof["unparsed_pre_window_prefix_bytes"] <= bf["prefix"]["bytes"]
        and (
            (cursor_start > 0 and proof["unparsed_pre_window_prefix_bytes"] > 0)
            or (cursor_start == 0 and proof["unparsed_pre_window_prefix_bytes"] == 0)
        ),
        "body-prefix-fragment",
    )
    require(
        all(
            integer(proof[k])
            for k in (
                "raw_bytes_checked_including_rereads",
                "complete_records_checked",
                "loss_records_checked",
            )
        )
        and proof["raw_bytes_checked_including_rereads"]
        == raw_total + reg["learner_writer"]["evidence_pin"]["bytes"]
        <= 24 * 2**20
        and 2
        <= proof["loss_records_checked"]
        <= proof["complete_records_checked"]
        <= 4096,
        "body-append-counts",
    )
    selected = proof["qualifying_records"]
    require(
        is_list(selected)
        and 2 <= len(selected) <= 100
        and len(selected) <= proof["loss_records_checked"],
        "body-selected-count",
    )
    prev_step = -1
    prev_stamp = -1
    prev_end = bf["size_at_fstat"]
    for row in selected:
        fields(
            row, {"start", "end", "step", "timestamp_ns", "sha256", "outcome_labels"}
        )
        require(
            all(
                integer(row[k])
                for k in ("start", "end", "step", "timestamp_ns", "outcome_labels")
            )
            and checksum(row["sha256"])
            and prev_end <= row["start"] < row["end"] <= end
            and row["end"] - row["start"] <= append.MAX_JSON
            and row["step"] > prev_step
            and prev_stamp < row["timestamp_ns"] <= c["read_end"]["wall_ns"]
            and c["read_end"]["wall_ns"] - row["timestamp_ns"] <= 120 * 10**9,
            "body-selected-record",
        )
        require(
            any(
                s["end"] >= row["end"]
                and s["audit"]["read_end"]["wall_ns"] >= row["timestamp_ns"]
                for s in [x["span"] for x in spans["originals"]]
            ),
            "body-selected-observation",
        )
        prev_step = row["step"]
        prev_stamp = row["timestamp_ns"]
        prev_end = row["end"]
    require(sum(x["outcome_labels"] for x in selected) > 0, "body-outcome-labels")
    return proof


def _original_source(sf: Any, reg: Any) -> Any:
    return {
        "source_pins": {
            key: {
                "literal_path": reg["scope"]["files"][key]["path"],
                "resolved_path": reg["scope"]["files"][key]["path"],
                **{k: row[k] for k in ("sha256", "bytes", "mode", "uid", "gid")},
            }
            for key, row in sf["source_pins"].items()
        },
        "cached_references": {
            key: {
                "path": reg["scope"]["cached"][key]["resolved"],
                "sha256": row["sha256"],
                "bytes": row["resolved_stat"]["bytes"],
            }
            for key, row in sf["cached_references"].items()
        },
        "unit_static": sf["unit_static"],
        "default_target": sf["default_target"],
        "coverage": [
            "selected-source-bytes",
            "qualified-cache-stat-references",
            "unit-definition-environment-boot",
            "actual-process-origins",
        ],
        "absent": [
            "profile-run-continuation-semantics",
            "champion-teacher-semantic-proof",
            "support-job-health-and-age",
            "reporter-renewal",
            "recipe-credit-and-metric-work",
            "historical-birth",
        ],
    }


def _renewal(proof: Any, reg: Any, expected: Any, kernels: Any, inventory: Any) -> Any:
    fields(
        proof,
        {
            "format",
            "registration_sha256",
            "window_sha256",
            "external_premises_sha256",
            "verified_champions_sha256",
            "fence_sha256",
            "observations_sha256",
            "observations",
            "kernel_brackets",
            "fence",
            "rounds",
            "streams",
            "read_end",
            "publication_renewed",
            "physical_work_proven",
            "historical_lifetime_proven",
            "writer_qualified",
            "preservation_passed",
            "execution_authorized",
            "latest",
        },
        "body-renewal-fields",
    )
    b = expected["bindings"]
    window = expected["expected_window"]
    require(
        proof["format"] == "strength-publication-renewal-measurement-v1"
        and proof["streams"] == 34
        and type(proof["streams"]) is int
        and integer(proof["rounds"], 1)
        and proof["rounds"] <= 256
        and proof["publication_renewed"] is True
        and all(
            proof[k] is False
            for k in (
                "physical_work_proven",
                "historical_lifetime_proven",
                "writer_qualified",
                "preservation_passed",
                "execution_authorized",
            )
        ),
        "body-renewal-scope",
    )
    for key, expectkey in [
        ("registration_sha256", "registration_sha256"),
        ("window_sha256", "expected_window_sha256"),
        ("external_premises_sha256", "external_premises_sha256"),
        ("verified_champions_sha256", "verified_champions_sha256"),
    ]:
        require(proof[key] == b[expectkey], "body-renewal-binding")
    within(proof["read_end"], window)
    rows = proof["observations"]
    require(
        is_list(rows)
        and len(rows) == 34 * (proof["rounds"] + 1)
        and proof["observations_sha256"] == digest(rows),
        "body-renewal-rounds",
    )
    f = fields(
        proof["fence"],
        {
            "registration_sha256",
            "window_sha256",
            "read_end",
            "observations",
            "kernel_admission",
        },
    )
    require(
        f["registration_sha256"] == b["registration_sha256"]
        and f["window_sha256"] == b["expected_window_sha256"]
        and same(f["observations"], rows[:34])
        and proof["fence_sha256"] == digest(f),
        "body-renewal-fence",
    )
    summaries = proof["kernel_brackets"]
    require(
        is_list(summaries) and len(summaries) == proof["rounds"] + 2,
        "body-renewal-kernel-count",
    )
    by_measurement = {x["measurement_sha256"]: x for x in kernels}
    require(len(by_measurement) == len(kernels), "body-kernel-measurement-alias")
    for i, s in enumerate(summaries):
        fields(
            s,
            {
                "format",
                "measurement_sha256",
                "registration_sha256",
                "external_premises_sha256",
                "window_sha256",
                "read_start",
                "read_end",
                "owners",
                "gpu_owners",
                "owner_sha256",
                "source_checks_sha256",
                "audit_inventory_sha256",
                "worker_reporters_read",
                "historical_birth_qualified",
                "writer_qualified",
                "runtime_qualified",
                "preservation_passed",
                "execution_authorized",
            },
        )
        require(
            s["measurement_sha256"] in by_measurement, "body-renewal-kernel-embedded"
        )
        k = by_measurement[s["measurement_sha256"]]
        require(
            s["format"] == "strength-observed-kernel-summary-v1"
            and all(
                same(s[key], k[key])
                for key in (
                    "registration_sha256",
                    "external_premises_sha256",
                    "window_sha256",
                    "read_start",
                    "read_end",
                )
            )
            and type(s["owners"]) is int
            and s["owners"] == 12
            and type(s["gpu_owners"]) is int
            and s["gpu_owners"] == 8
            and s["owner_sha256"] == digest(k["owners"])
            and s["source_checks_sha256"]
            == digest(_original_source(k["source_facts"], reg))
            and s["audit_inventory_sha256"] == digest(k["audit_inventory"])
            and all(
                s[key] is False
                for key in (
                    "worker_reporters_read",
                    "historical_birth_qualified",
                    "writer_qualified",
                    "runtime_qualified",
                    "preservation_passed",
                    "execution_authorized",
                )
            ),
            "body-renewal-kernel-summary",
        )
        if i:
            order(summaries[i - 1]["read_end"], s["read_start"])
    require(same(f["kernel_admission"], summaries[0]), "body-renewal-initial-kernel")
    within(f["read_end"], window)
    order(rows[33]["read_end"], f["read_end"])
    order(f["read_end"], summaries[1]["read_start"])
    latest = {}
    baseline = {}
    renewed = set()
    last = None
    safe_keys = {
        "operation",
        "read_start",
        "read_end",
        "raw",
        *FOUR,
        "offset",
        "end_offset",
        "stream",
        "round",
        "audit_sha256",
        "producer_stamp_ns",
        "counters",
        "teacher",
    }
    for i, row in enumerate(rows):
        fields(row, safe_keys, "body-renewal-observation-fields")
        name = STREAMS[i % 34]
        round_ = i // 34
        require(
            row["stream"] == name
            and type(row["round"]) is int
            and row["round"] == round_,
            "body-renewal-stream-order",
        )
        a = (
            {key: deepcopy(row[key]) for key in BASE_AUDIT - {"subject"}}
            | {"subject": reg["publication_writers"][name]["key"]}
            | {key: deepcopy(row[key]) for key in (*FOUR, "offset", "end_offset")}
        )
        _audit(a, reg, window)
        require(
            a["operation"] == "read-publication" and row["audit_sha256"] == digest(a),
            "body-renewal-audit",
        )
        _membership([a], inventory, "body-renewal-audit-membership")
        if last is not None:
            order(last, a["read_start"])
        last = a["read_end"]
        order(summaries[round_]["read_end"], a["read_start"])
        if i % 34 == 33:
            order(a["read_end"], summaries[round_ + 1]["read_start"])
        st = a["stat_before"]
        spec = reg["publication_writers"][name]
        require(
            all(same(st, a[key]) for key in FOUR)
            and st["bytes"] == a["raw"]["bytes"]
            and all(st[k] == spec[k] for k in ("uid", "gid", "mode")),
            "body-renewal-file",
        )
        stamp = row["producer_stamp_ns"]
        counters = row["counters"]
        require(
            integer(stamp, 1)
            and stamp <= a["read_end"]["wall_ns"]
            and is_dict(counters)
            and all(integer(v) for v in counters.values()),
            "body-renewal-stamp-counter",
        )
        if name == "coordinator":
            require(
                same(
                    counters,
                    {
                        r: reg["policy"]["expected_processes"][r]["restarts"]
                        for r in records.WORKERS
                    },
                ),
                "body-renewal-restarts",
            )
        else:
            require(
                {"progress", "progress_ns"}
                <= set(counters)
                <= set(("progress", "progress_ns", "step", "cumulative_games"))
                and 0 < counters["progress_ns"] <= stamp,
                "body-renewal-progress",
            )
        teacher = row["teacher"]
        if name in records.COHORTS or name == "actor-cpu-ring4":
            fields(teacher, {"model_identity", "proof_sha256"})
            require(
                teacher["model_identity"] in expected["verified_champions"]
                and teacher["proof_sha256"]
                == expected["verified_champions"][teacher["model_identity"]],
                "body-renewal-teacher",
            )
            if name in records.COHORTS:
                require("cumulative_games" in counters, "body-renewal-cohort-games")
        else:
            require(teacher is None, "body-renewal-unexpected-teacher")
        if round_ == 0:
            baseline[name] = row
        else:
            prev = latest[name]
            first = baseline[name]
            require(
                st["device"] == first["stat_before"]["device"]
                and stamp >= prev["producer_stamp_ns"]
                and set(prev["counters"]) <= set(counters)
                and all(counters[k] >= v for k, v in prev["counters"].items()),
                "body-renewal-regression",
            )
            if (
                stamp > first["producer_stamp_ns"]
                and stamp > summaries[0]["read_end"]["wall_ns"]
                and row["raw"]["sha256"] != first["raw"]["sha256"]
                and not same(st, first["stat_before"])
            ):
                renewed.add(name)
        latest[name] = row
    require(
        renewed == set(STREAMS) and set(proof["latest"]) == set(STREAMS),
        "body-renewal-incomplete",
    )
    for name, row in latest.items():
        require(
            same(
                proof["latest"][name],
                {
                    "stamp": row["producer_stamp_ns"],
                    "sha256": row["raw"]["sha256"],
                    "version": row["stat_before"],
                    "counters": row["counters"],
                },
            ),
            "body-renewal-latest",
        )
        require(
            0
            <= proof["read_end"]["wall_ns"] - row["producer_stamp_ns"]
            <= reg["policy"]["maximum_age_ns"],
            "body-renewal-final-age",
        )
    order(summaries[-1]["read_end"], proof["read_end"])
    return latest


def _support_unit_health(kind: Any, props: Any, now: Any, maximum_seconds: Any) -> None:
    if kind == "long_running":
        require(
            props["ActiveState"] == "active"
            and props["SubState"] == "running"
            and props["Result"] in {"", "success"},
            "body-observed-monitor-unhealthy",
        )
    elif kind == "timer":
        require(props["ActiveState"] == "active", "body-observed-timer-inactive")
    elif props["ActiveState"] in support.ACTIVE:
        # Result belongs to the previous invocation while this one runs.
        age = support.facts.running_seconds(
            now_monotonic_ns=now["monotonic_ns"],
            activation_start_us=int(props["InactiveExitTimestampMonotonic"]),
            invocation_id=props["InvocationID"],
            expected_invocation_id=props["InvocationID"],
        )
        require(
            age <= maximum_seconds,
            "body-observed-support-overdue",
        )
    else:
        require(
            props["ActiveState"] == "inactive"
            and props["Result"] in {"", "success"}
            and int(props["ExecMainStatus"]) == 0,
            "body-observed-support-failed",
        )


def _support_proof(
    c: Any, prov: Any, reg: Any, expected: Any, kernels: Any, inventory: Any
) -> Any:
    rows = prov["support_provenance"]
    require(is_list(rows) and 1 <= len(rows) <= 258, "body-support-captures")
    w = expected["expected_window"]
    names = set(reg["policy"]["support"])
    for sp in rows:
        fields(
            sp,
            {
                "format",
                "clock",
                "manager",
                "manager_id",
                "unit_states",
                "jobs",
                "audit",
                "prior_witness_authentication",
                "execution_qualified",
            },
        )
        require(
            sp["format"] == "strength-support-capture-v1"
            and sp["manager_id"] == support.manager_id(sp["manager"])
            and sp["prior_witness_authentication"] == "external caller required"
            and sp["execution_qualified"] is False
            and set(sp["unit_states"]) == names
            and set(sp["jobs"]) <= names,
            "body-support-provenance",
        )
        within(sp["clock"], w)
        _audits(sp["audit"], reg, w)
        _membership(sp["audit"], inventory, "body-support-audit-membership")
        for a in sp["audit"]:
            order(a["read_end"], sp["clock"])
        require(
            sp["manager"]["boot_id"] == w["phase_start"]["boot_id"],
            "body-support-manager-boot",
        )
        require(
            type(sp["manager"]["pid"]) is int
            and type(sp["manager"]["ppid"]) is int
            and type(sp["manager"]["start_ticks"]) is int,
            "body-manager-scalar-types",
        )
        for kind, ns in sp["manager"]["namespaces"].items():
            fields(ns, {"literal", "stat"}, "body-manager-namespace-fields")
            fields(ns["stat"], {"device", "inode"}, "body-manager-namespace-stat")
            require(
                ns["stat"]["inode"]
                == reg["kernel_context"]["namespace_expectations"]["pid1"][kind],
                "body-support-manager-namespace",
            )
        for name, props in sp["unit_states"].items():
            fields(
                props,
                support.DYNAMIC
                + (
                    support.SERVICE_DYNAMIC
                    if reg["units"][name]["kind"] != "timer"
                    else ()
                ),
            )
            _safe_unit_properties(name, props)
            kind = reg["units"][name]["kind"]
            _support_unit_health(
                kind, props, sp["clock"], reg["policy"]["maximum_support_seconds"]
            )
            job = sp["jobs"].get(name)
            if job is None:
                require(props["Job"] in {"", "0"}, "body-support-job-join")
            else:
                fields(job, {"id", "unit", "kind", "state"})
                require(
                    integer(job["id"], 1)
                    and job["unit"] == name
                    and kind == "oneshot"
                    and job["kind"] == "start"
                    and job["state"] in {"waiting", "running"}
                    and props["Job"] == str(job["id"]),
                    "body-support-job-join",
                )
    witnesses = prov["support_witnesses"]
    require(is_dict(witnesses) and set(witnesses) <= names, "body-support-witnesses")
    coarse = Counter(
        digest({k: a[k] for k in BASE_AUDIT}) for a in prov["audit_inventory"]
    )
    for name, record in witnesses.items():
        absence = support.checked_witness(
            record, name=name, manager=record["proof"]["manager"], now=c["read_end"]
        )
        within(absence["read_start"], w)
        within(absence["read_end"], w)
        wanted = Counter(
            digest(x["audit"])
            for x in record["proof"]["unit_reads"] + record["proof"]["jobs_reads"]
        )
        require(not (wanted - coarse), "body-support-witness-unlogged")
        matches = [
            sp
            for sp in rows
            if same(sp["manager"], record["proof"]["manager"])
            and sp["manager_id"] == absence["manager_id"]
            and same(sp["clock"], absence["read_end"])
            and name not in sp["jobs"]
            and sp["unit_states"][name]["Job"] in {"", "0"}
            and not (
                wanted
                - Counter(digest({k: a[k] for k in BASE_AUDIT}) for a in sp["audit"])
            )
        ]
        require(len(matches) == 1, "body-support-witness-source")
    last = rows[-1]
    projection = fields(
        prov["support_age_projection"],
        {"observed_clock", "capture_clock", "added_upper_bound_seconds"},
    )
    require(
        same(projection["observed_clock"], last["clock"])
        and same(projection["capture_clock"], c["read_end"]),
        "body-support-age-clock",
    )
    order(last["clock"], c["read_end"])
    extension = (
        c["read_end"]["monotonic_ns"] - last["clock"]["monotonic_ns"] + 999999999
    ) // 10**9
    require(
        type(projection["added_upper_bound_seconds"]) is int
        and projection["added_upper_bound_seconds"] == extension,
        "body-support-age-extension",
    )
    final = kernels[-1]
    for name, actual in last["unit_states"].items():
        require(
            all(
                same(value, final["unit_facts"][name]["properties"][key])
                for key, value in actual.items()
            ),
            "body-final-support-state",
        )
        row = c["common"]["support"][name]
        kind = reg["units"][name]["kind"]
        require(row["active"] == actual["ActiveState"], "body-support-active")
        if kind != "timer":
            require(
                row["result"] == actual["Result"]
                and type(row["exit_code"]) is int
                and row["exit_code"] == int(actual["ExecMainStatus"]),
                "body-support-result",
            )
        if kind == "oneshot" and row["active"] in support.ACTIVE:
            age = support.facts.running_seconds(
                now_monotonic_ns=last["clock"]["monotonic_ns"],
                activation_start_us=int(actual["InactiveExitTimestampMonotonic"]),
                invocation_id=actual["InvocationID"],
                expected_invocation_id=actual["InvocationID"],
            )
            require(
                type(row["running_seconds"]) is int
                and row["running_seconds"] == age + extension,
                "body-support-running-age",
            )
        else:
            require(
                row["running_seconds"] is None, "body-support-running-not-applicable"
            )
        job = last["jobs"].get(name)
        if job is None:
            require(row["job"] is None, "body-support-job-absence")
        else:
            require(name in witnesses, "body-support-job-age-witness")
            a = witnesses[name]["absence"]
            require(a["manager_id"] == last["manager_id"], "body-support-job-manager")
            queries = [
                x
                for x in last["audit"]
                if x["operation"] == "query" and x["subject"] == "jobs"
            ]
            require(len(queries) == 2, "body-support-job-reads")
            age = support.facts.job_age_seconds(
                current={
                    "boot_id": last["clock"]["boot_id"],
                    "manager_id": last["manager_id"],
                    "unit": name,
                    **job,
                    "read_start": queries[0]["read_start"],
                    "read_end": queries[1]["read_end"],
                    "creation_source_sha256": None,
                },
                now=last["clock"],
                absence=a,
            )
            require(
                same(row["job"], {"kind": "start", "age_seconds": age + extension}),
                "body-support-job-age",
            )


def _common(
    c: Any, prov: Any, reg: Any, expected: Any, kernels: Any, latest: Any, proof: Any
) -> Any:
    common = fields(
        c["common"],
        {
            "static",
            "owners",
            "processes",
            "gpu_owners",
            "support",
            "champion_identity",
            "cohorts",
            "progress",
        },
    )
    policy = reg["policy"]
    owners = kernels[-1]["owners"]
    now = c["read_end"]["wall_ns"]
    age = policy["maximum_age_ns"]
    require(
        same(common["static"], policy["static"])
        and same(common["owners"], owners)
        and same(common["gpu_owners"], kernels[-1]["gpu_owners"]),
        "body-common-kernel-static",
    )
    preservation._check_recipe_values(policy["recipe"])
    require(
        set(common["processes"]) == set(policy["expected_processes"]),
        "body-process-roster",
    )
    for role, p in common["processes"].items():
        fields(
            p,
            preservation.PROCESS_FIELDS
            | ({"heartbeat_ns"} if role != "controller" else set()),
        )
        require(
            same(
                {k: p[k] for k in preservation.PROCESS_FIELDS},
                policy["expected_processes"][role],
            )
            and all(
                same(p[k], owners[role][k])
                for k in preservation.PROCESS_FIELDS - {"restarts"}
            ),
            "body-common-process",
        )
        if role != "controller":
            require(
                type(p["heartbeat_ns"]) is int
                and p["heartbeat_ns"] == latest[role]["producer_stamp_ns"]
                and 0 <= now - p["heartbeat_ns"] <= age,
                "body-process-renewal",
            )
        if role in records.WORKERS:
            require(
                p["restarts"] == latest["coordinator"]["counters"][role],
                "body-process-worker-restart",
            )
    for name, row in common["support"].items():
        fields(
            row,
            preservation.SUPPORT_STATIC
            | {"active", "result", "exit_code", "process", "running_seconds", "job"},
        )
        if row["process"] is not None:
            fields(row["process"], preservation.PROCESS_FIELDS)
        if row["job"] is not None:
            fields(row["job"], {"kind", "age_seconds"})
        require(
            same(
                {key: row[key] for key in preservation.SUPPORT_STATIC},
                policy["support"][name],
            ),
            "body-common-support-static",
        )
    preservation._check_support(policy, common["support"])
    require(
        common["champion_identity"] in expected["verified_champions"],
        "body-common-champion",
    )
    require(set(common["cohorts"]) == set(records.COHORTS), "body-common-cohorts")
    for name, row in common["cohorts"].items():
        fields(row, {"model_identity", "games", "heartbeat_ns", "progress_ns"})
        renewal = latest[name]
        require(
            same(
                row,
                {
                    "model_identity": renewal["teacher"]["model_identity"],
                    "games": renewal["counters"]["cumulative_games"],
                    "heartbeat_ns": renewal["producer_stamp_ns"],
                    "progress_ns": renewal["counters"]["progress_ns"],
                },
            )
            and 0 <= now - row["heartbeat_ns"] <= age,
            "body-cohort-renewal",
        )
    p = fields(common["progress"], {"step", "phase", "heartbeat_ns", "metrics"})
    require(
        integer(p["step"])
        and p["phase"] in {"training", "update_to_data_wait"}
        and "step" in latest["learner"]["counters"]
        and p["step"] == latest["learner"]["counters"]["step"]
        and p["heartbeat_ns"] == latest["learner"]["producer_stamp_ns"]
        and 0 <= now - p["heartbeat_ns"] <= age,
        "body-learner-renewal",
    )
    metrics = p["metrics"]
    selected = proof["qualifying_records"]
    require(
        is_list(metrics) and len(metrics) == len(selected),
        "body-all-selected-metrics",
    )
    fields_metric = (records.METRIC_FIELDS - {"schema_version"}) | {"record_ref"}
    labels = 0
    for row, selected_row in zip(metrics, selected, strict=True):
        fields(row, fields_metric, "body-safe-metric-fields")
        require(
            same(
                row["record_ref"],
                {k: selected_row[k] for k in ("start", "end", "sha256")},
            )
            and row["worker"] == "learner"
            and same(row["step"], selected_row["step"])
            and same(row["timestamp_ns"], selected_row["timestamp_ns"])
            and row["step"] <= p["step"]
            and row["timestamp_ns"] <= p["heartbeat_ns"]
            and 0 <= now - row["timestamp_ns"] <= age,
            "body-metric-ref-time",
        )
        require(
            is_dict(row["losses"])
            and 0 < len(row["losses"]) <= 64
            and all(checksum(k) for k in row["losses"]),
            "body-safe-loss-keys",
        )
        fields(row["ema"], {"decay", "num_updates"})
        fields(row["policy_batch_metrics"], {"unknown_provenance_rows"})
        d = fields(
            row["gradient_diagnostics"],
            {"global_norm_finite", "nonfinite_gradient_tensors", "batch"},
        )
        batch = fields(d["batch"], {"rows", "six_mode_unknown", "label_availability"})
        fields(batch["label_availability"], {"policy", "outcome"})
        require(
            is_dict(batch["six_mode_unknown"]) and not batch["six_mode_unknown"],
            "body-safe-unknown-modes",
        )
        require(
            type(d["nonfinite_gradient_tensors"]) is int
            and type(row["policy_batch_metrics"]["unknown_provenance_rows"]) is int,
            "body-metric-scalar-alias",
        )
        count = preservation._check_metric_payload(row, policy["recipe"])
        require(count == selected_row["outcome_labels"], "body-metric-label-join")
        labels += count
    require(labels > 0, "body-common-outcomes")


def _static_authority_reads(s: Any, reg: Any) -> None:
    required = (
        "run",
        "continuation",
        "profile_authority",
        "profile",
        "run_source",
        "release_source",
        "source_manifest",
        "champion",
    )
    reads = {}
    for name in required:
        rows = [
            a
            for a in s["audit_inventory"]
            if a["operation"] == "read" and a["subject"] == reg["keys"][name]
        ]
        require(len(rows) == 1, "body-static-authority-read")
        reads[name] = rows[0]["raw"]
    static = reg["policy"]["static"]
    for name, key in (
        ("profile", "profile_sha256"),
        ("source_manifest", "source_manifest_sha256"),
    ):
        require(reads[name]["sha256"] == static[key], "body-static-authority-sha")
    permitted = [
        static["source_commit"].encode(),
        (static["source_commit"] + "\n").encode(),
    ]
    for name in ("run_source", "release_source"):
        require(
            any(
                same(reads[name], {"sha256": sha(raw), "bytes": len(raw)})
                for raw in permitted
            ),
            "body-static-source-commit",
        )


def _finish_order(prov: Any) -> None:
    # These are the producer's actual final stages; a previous healthy sample
    # cannot stand in for a later required check or extend fixed-E coverage.
    final_round = prov["renewal"]["observations"][-len(STREAMS) :]
    final_source = prov["source_checks"][-1]
    final_support = prov["support_provenance"][-1]
    order(
        prov["metric_spans"]["rereads"][-1]["span"]["audit"]["read_end"],
        final_round[0]["read_start"],
    )
    order(final_round[-1]["read_end"], final_source["read_start"])
    order(final_source["read_end"], final_support["audit"][0]["read_start"])
    order(
        final_support["clock"],
        prov["metric_spans"]["end_guard"]["span"]["audit"]["read_start"],
    )


def _semantic(c: Any, prov: Any, expected: Any, registration: Any) -> Any:
    reg = _expected(expected, registration)
    window = expected["expected_window"]
    _header(c, CAPTURE, CAPTURE_FIELDS)
    _header(prov, PROVENANCE, PROVENANCE_FIELDS)
    require(
        same(c["bindings"], expected["bindings"])
        and same(prov["bindings"], expected["bindings"])
        and same(c["read_start"], prov["read_start"])
        and same(c["read_end"], prov["read_end"])
        and prov["privacy_scope"] == PRIVACY,
        "body-common-bindings",
    )
    order(c["read_start"], c["read_end"])
    within(c["read_start"], window)
    within(c["read_end"], window)
    inventory = _audits(prov["audit_inventory"], reg, window)
    require(
        same(c["read_start"], prov["audit_inventory"][0]["read_start"])
        and same(c["read_end"], prov["audit_inventory"][-1]["read_end"]),
        "body-actual-observation-interval",
    )
    for a in prov["audit_inventory"]:
        order(c["read_start"], a["read_start"])
        order(a["read_end"], c["read_end"])
    kernels = prov["kernel_projections"]
    require(is_list(kernels) and 2 <= len(kernels) <= 258, "body-kernel-count")
    for i, k in enumerate(kernels):
        _kernel(k, reg, expected, inventory)
        if i:
            order(kernels[i - 1]["read_end"], k["read_start"])
            for field in (
                "owners",
                "monitor",
                "gpu_owners",
                "auxiliaries",
                "source_facts",
            ):
                require(same(kernels[0][field], k[field]), "body-kernel-continuity")
    projections = {digest(k): k for k in kernels}
    require(len(projections) == len(kernels), "body-kernel-alias")
    source_checks = prov["source_checks"]
    require(
        is_list(source_checks) and 2 <= len(source_checks) <= 258,
        "body-static-recheck-count",
    )
    for index, s in enumerate(source_checks):
        fields(
            s,
            {
                "static",
                "static_sha256",
                "read_start",
                "read_end",
                "source_facts",
                "unit_facts",
                "audit_inventory",
            },
        )
        require(
            same(s["static"], reg["policy"]["static"])
            and s["static_sha256"] == preservation.digest(s["static"]),
            "body-full-static",
        )
        order(s["read_start"], s["read_end"])
        within(s["read_start"], window)
        within(s["read_end"], window)
        _audits(s["audit_inventory"], reg, window)
        for a in s["audit_inventory"]:
            order(s["read_start"], a["read_start"])
            order(a["read_end"], s["read_end"])
        if index:
            order(source_checks[index - 1]["read_end"], s["read_start"])
        _membership(s["audit_inventory"], inventory, "body-static-audit-membership")
        _static_authority_reads(s, reg)
        _source_facts(
            s["source_facts"], reg, kernels[-1]["owners"], s["audit_inventory"]
        )
        _unit_facts(s["unit_facts"], reg, kernels[-1]["owners"], s["source_facts"])
    order(
        source_checks[0]["read_end"],
        c["metric_window"]["before_fence"]["fence"]["audit"]["read_start"],
    )
    order(source_checks[-1]["read_end"], kernels[-1]["read_start"])
    latest = _renewal(prov["renewal"], reg, expected, kernels, inventory)
    require(
        same(prov["renewal"]["read_end"], c["read_end"]), "body-final-renewal-clock"
    )
    proof = _b_and_spans(c, prov, reg, expected, projections, inventory)
    _common(c, prov, reg, expected, kernels, latest, proof)
    _support_proof(c, prov, reg, expected, kernels, inventory)
    _finish_order(prov)
    return reg


def _receipt(
    value: Any,
    capture_raw: Any,
    provenance_raw: Any,
    c: Any,
    prov: Any,
    artifact_pins: Any,
) -> Any:
    _header(value, RECEIPT, RECEIPT_FIELDS)
    require(
        value["status"] == "complete"
        and value["refusals"] == []
        and value["privacy_scope"] == PRIVACY
        and same(value["bindings"], c["bindings"])
        and same(value["read_start"], c["read_start"])
        and same(value["read_end"], c["read_end"]),
        "body-receipt",
    )
    for role, raw in (("capture", capture_raw), ("provenance", provenance_raw)):
        require(
            same(value[role + "_pin"], artifact_pins[role])
            and same(artifact_pins[role], pin(artifact_pins[role]["path"], raw)),
            "body-receipt-pin",
        )
    require(
        same(prov["capture_pin"], artifact_pins["capture"]), "body-provenance-capture"
    )
    root = PurePosixPath(artifact_pins["capture"]["path"]).parent
    require(
        artifact_pins["capture"]["path"] == str(root / "r3-before.json")
        and artifact_pins["receipt"]["path"] == str(root / "r3-before.receipt.json")
        and artifact_pins["provenance"]["path"]
        == str(root / ("r3-before.provenance-" + sha(provenance_raw) + ".json")),
        "body-fixed-output-paths",
    )


def encode_before_bodies(
    capture_value: Any,
    provenance_value_without_capture_pin: Any,
    *,
    expected: Any,
    registration: Any,
    output_root: Any,
) -> Any:
    """Return candidate safe bytes/pins only, not published or terminal evidence."""
    try:
        c = deepcopy(capture_value)
        p = deepcopy(provenance_value_without_capture_pin)
        require("capture_pin" not in p, "body-future-capture-pin")
        cr = encoded(c)
        require(len(cr) <= MAX_JSON, "body-capture-size")
        root = PurePosixPath(output_root)
        capture_pin = pin(str(root / "r3-before.json"), cr)
        p["capture_pin"] = capture_pin
        _semantic(c, p, expected, registration)
        pr = encoded(p)
        require(len(pr) <= MAX_PROVENANCE, "body-provenance-size")
        provenance_pin = pin(
            str(root / ("r3-before.provenance-" + sha(pr) + ".json")), pr
        )
        r = {
            "format": RECEIPT,
            "schema_version": 1,
            "contract": CONTRACT,
            "phase": "before",
            "bindings": deepcopy(c["bindings"]),
            "status": "complete",
            "capture_pin": capture_pin,
            "provenance_pin": provenance_pin,
            "read_start": deepcopy(c["read_start"]),
            "read_end": deepcopy(c["read_end"]),
            "refusals": [],
            "privacy_scope": PRIVACY,
        }
        rr = encoded(r)
        require(len(rr) <= MAX_JSON, "body-receipt-size")
        return {
            "capture": cr,
            "provenance": pr,
            "receipt": rr,
            "pins": {
                "capture": capture_pin,
                "provenance": provenance_pin,
                "receipt": pin(str(root / "r3-before.receipt.json"), rr),
            },
        }
    except BodyRefusal:
        raise
    except (
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        OverflowError,
        RecursionError,
    ):
        raise BodyRefusal("body-encoding-refused") from None


def validate_before_bodies(
    capture_raw: Any,
    receipt_raw: Any,
    provenance_raw: Any,
    *,
    expected: Any,
    registration: Any,
    producer_interval: Any,
    artifact_pins: Any,
) -> Any:
    """Complete safe consistency checks after caller's actual source/exit gate.

    A successful result is detached data, not an executable AFTER/B authority.
    The actual family bytes and external approval are the reader's responsibility.
    """
    try:
        c = parse(capture_raw)
        p = parse(provenance_raw, MAX_PROVENANCE)
        r = parse(receipt_raw)
        require(set(artifact_pins) == completion.ARTIFACTS, "body-six-artifact-pins")
        for item in artifact_pins.values():
            completion.pin(
                item,
                MAX_PROVENANCE if item == artifact_pins["provenance"] else MAX_JSON,
            )
        require(
            same(
                artifact_pins["receipt"],
                pin(artifact_pins["receipt"]["path"], receipt_raw),
            ),
            "body-receipt-bytes",
        )
        _receipt(r, capture_raw, provenance_raw, c, p, artifact_pins)
        for role, key in (
            ("request", "request_sha256"),
            ("launch", "launch_sha256"),
            ("registration", "registration_sha256"),
        ):
            require(
                artifact_pins[role]["sha256"] == expected["bindings"][key],
                "body-input-artifact-pin",
            )
        _semantic(c, p, expected, registration)
        fields(producer_interval, {"released", "terminal"}, "body-producer-interval")
        order(producer_interval["released"], c["read_start"])
        order(c["read_end"], producer_interval["terminal"])
        return {
            "capture": deepcopy(c),
            "provenance": deepcopy(p),
            "receipt": deepcopy(r),
            "execution_qualified": False,
            "writer_authority_granted": False,
            "full_preservation": False,
        }
    except BodyRefusal:
        raise
    except (
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        OverflowError,
        RecursionError,
    ):
        raise BodyRefusal("body-validation-refused") from None
