"""Pure, private-input normalization for the reviewed preservation collector.

No I/O, clock reads, process queries or actions. Callers must qualify the original
raw observations, allowed paths and owner joins. Public results are digests,
integer bounds or fixed error codes; never return secret-bearing input values.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any, TypeGuard

CONTRACT_SHA256 = "fefd383c4d7dfdb52530673956eea4efd28d2ae906b125468cc619fc6c5dce62"
TIMER_ADDENDUM_SHA256 = (
    "eb10efecb7adee04a11150b8e208adaaac0c7b7e1f01edb99512935a0d6fd930"
)
TIMER_ENVIRONMENT_ADDENDUM_SHA256 = (
    "601ea95f721dc8d4b666dbd1bbd7909750e9dd53155a41b837fbc2c664b85ccb"
)
SERVICE_STATIC = tuple(
    "Id Names LoadState SourcePath Type User Group WorkingDirectory ExecStart "
    "ExecStartPre ExecStop ExecStopPost Restart KillMode KillSignal SendSIGKILL "
    "TimeoutStartUSec TimeoutStopUSec RuntimeMaxUSec After Before Wants Requires".split()
)
TIMER_STATIC = tuple(
    "Id Names LoadState SourcePath Unit AccuracyUSec RandomizedDelayUSec "
    "Persistent After Before Wants Requires".split()
)
_EXEC_DYNAMIC = {
    "start_time",
    "stop_time",
    "pid",
    "code",
    "status",
    "start_time_monotonic",
    "stop_time_monotonic",
}


class FactViolation(ValueError):
    """Contains a fixed reason code, never a rejected input value."""


def _require(ok: object, code: str) -> None:
    if not ok:
        raise FactViolation(code)


def _integer(value: object, minimum: int = 0) -> TypeGuard[int]:
    return type(value) is int and value >= minimum


def _sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _shape(value: object, keys: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping) and set(value) == set(keys.split()), "fields")
    assert isinstance(value, Mapping)
    return value


def _path(value: object) -> None:
    _require(
        isinstance(value, str)
        and value.startswith("/")
        and not value.startswith("//")
        and str(PurePosixPath(value)) == value
        and ".." not in PurePosixPath(value).parts
        and "\x00" not in value,
        "path",
    )


def _unit(value: object) -> None:
    _require(
        isinstance(value, str)
        and re.fullmatch(
            r"(?:[A-Za-z0-9_.@:-]|\\x[0-9a-fA-F]{2})+\.(service|timer|target|slice|socket|mount|automount|swap|path|scope|device)",
            value,
        ),
        "unit",
    )


def _pin(value: object, *, metadata: bool = False) -> None:
    keys = (
        "literal_path resolved_path sha256 bytes mode uid gid"
        if metadata
        else "path sha256 bytes"
    )
    row = _shape(value, keys)
    for field in ("literal_path", "resolved_path") if metadata else ("path",):
        _path(row[field])
    _require(_sha(row["sha256"]) and _integer(row["bytes"]), "file-pin")
    if metadata:
        _require(
            _integer(row["mode"])
            and row["mode"] <= 0o7777
            and _integer(row["uid"])
            and _integer(row["gid"]),
            "file-metadata",
        )


def _digest(domain: str, material: Mapping[str, Any]) -> str:
    try:
        raw = json.dumps(
            {"format": domain, "schema_version": 1, **material},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise FactViolation("canonical-encoding") from None
    return hashlib.sha256(raw).hexdigest()


def _exec_static(value: str) -> str:
    if not value:
        return value
    _require(
        value.count("{ path=") == 1
        and value.count(" ; ignore_errors=") == 1
        and value.endswith(" }"),
        "exec-shape",
    )
    prefix, suffix = value.split(" ; ignore_errors=", 1)
    parts = suffix.removesuffix(" }").split(" ; ")
    _require(parts[0] in {"yes", "no"}, "exec-ignore-errors")
    retained = [
        part for part in parts[1:] if part.split("=", 1)[0] not in _EXEC_DYNAMIC
    ]
    return (
        prefix
        + " ; ignore_errors="
        + parts[0]
        + "".join(" ; " + part for part in retained)
    )


def definition_digest(material: Mapping[str, Any]) -> str:
    row = _shape(
        material, "unit fragment dropins loaded_stable_properties need_daemon_reload"
    )
    _unit(row["unit"])
    _require(row["unit"].endswith((".service", ".timer")), "definition-kind")
    _require(row["need_daemon_reload"] is False, "daemon-reload")
    _pin(row["fragment"], metadata=True)
    _require(isinstance(row["dropins"], list), "dropins")
    for item in row["dropins"]:
        _pin(item, metadata=True)
    _require(
        len({x["literal_path"] for x in row["dropins"]}) == len(row["dropins"]),
        "duplicate-dropin",
    )
    fields = TIMER_STATIC if row["unit"].endswith(".timer") else SERVICE_STATIC
    props = _shape(row["loaded_stable_properties"], " ".join(fields))
    _require(all(isinstance(v, str) for v in props.values()), "property-type")
    _require(
        props["Id"] == row["unit"] and props["LoadState"] == "loaded", "loaded-unit"
    )
    normalized = dict(props)
    for key in ("Names", "After", "Before", "Wants", "Requires"):
        names = props[key].split()
        _require(len(set(names)) == len(names), "duplicate-unit-name")
        for name in names:
            _unit(name)
        normalized[key] = " ".join(sorted(names))
    for key in ("ExecStart", "ExecStartPre", "ExecStop", "ExecStopPost"):
        if key in normalized:
            normalized[key] = _exec_static(normalized[key])
    return _digest(
        "strength-preservation-unit-definition-v1",
        {**row, "loaded_stable_properties": normalized},
    )


def _property_hash(value: object, property_name: str) -> None:
    row = _shape(value, "present raw_sha256 bytes effective_empty_rule")
    _require(type(row["present"]) is bool and _integer(row["bytes"]), "property-hash")
    if row["present"]:
        _require(
            _sha(row["raw_sha256"]) and row["effective_empty_rule"] is None,
            "property-hash",
        )
    else:
        _require(row["raw_sha256"] is None and row["bytes"] == 0, "absent-property")
        _require(
            isinstance(row["effective_empty_rule"], str)
            and row["effective_empty_rule"] == "systemd-255-empty-" + property_name,
            "unqualified-empty-property",
        )


def environment_digest(material: Mapping[str, Any]) -> str:
    if isinstance(material.get("unit"), str) and material["unit"].endswith(".timer"):
        timer = _shape(material, "unit applicability")
        _unit(timer["unit"])
        _require(timer["applicability"] == "not-applicable-timer", "timer-environment")
        return _digest("strength-preservation-unit-environment-v1", timer)
    row = _shape(
        material,
        "unit inline_environment pass_environment unset_environment environment_files",
    )
    _unit(row["unit"])
    _require(row["unit"].endswith(".service"), "environment-kind")
    for key, property_name in (
        ("inline_environment", "Environment"),
        ("pass_environment", "PassEnvironment"),
        ("unset_environment", "UnsetEnvironment"),
    ):
        _property_hash(row[key], property_name)
    _require(isinstance(row["environment_files"], list), "environment-files")
    for value in row["environment_files"]:
        item = _shape(
            value,
            "literal_path resolved_path ignore_missing exists sha256 bytes mode uid gid",
        )
        _path(item["literal_path"])
        _path(item["resolved_path"])
        _require(
            type(item["ignore_missing"]) is bool and type(item["exists"]) is bool,
            "environment-file-flags",
        )
        if item["exists"]:
            _pin(
                {
                    k: item[k]
                    for k in "literal_path resolved_path sha256 bytes mode uid gid".split()
                },
                metadata=True,
            )
        else:
            _require(
                item["ignore_missing"]
                and all(item[k] is None for k in "sha256 bytes mode uid gid".split()),
                "missing-environment-file",
            )
    return _digest("strength-preservation-unit-environment-v1", row)


def origin_digest(material: Mapping[str, Any]) -> str:
    row = _shape(
        material,
        "role executable cwd_resolved argv relevant_environment source_contract native_mappings restart_counter_scope",
    )
    _require(
        isinstance(row["role"], str)
        and re.fullmatch(r"[a-z0-9][a-z0-9-]*", row["role"]),
        "role",
    )
    exe = _shape(
        row["executable"],
        "literal_entrypoint resolved_proc_exe qualified_file_reference",
    )
    _path(exe["literal_entrypoint"])
    _path(exe["resolved_proc_exe"])
    _pin(exe["qualified_file_reference"])
    _require(
        exe["qualified_file_reference"]["path"] == exe["resolved_proc_exe"],
        "executable-reference",
    )
    _path(row["cwd_resolved"])
    argv = _shape(row["argv"], "sha256 bytes approved_template_id")
    _require(
        _sha(argv["sha256"])
        and _integer(argv["bytes"], 1)
        and _sha(argv["approved_template_id"]),
        "argv-pin",
    )
    _require(isinstance(row["relevant_environment"], list), "process-environment")
    keys = []
    for item in row["relevant_environment"]:
        item = _shape(item, "key present value_sha256 value_bytes")
        _require(
            isinstance(item["key"], str)
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item["key"]),
            "environment-key",
        )
        keys.append(item["key"])
        _require(
            type(item["present"]) is bool and _integer(item["value_bytes"]),
            "environment-value",
        )
        _require(
            _sha(item["value_sha256"])
            if item["present"]
            else item["value_sha256"] is None and item["value_bytes"] == 0,
            "environment-value",
        )
    _require(keys == sorted(set(keys)), "environment-key-order")
    source = _shape(
        row["source_contract"],
        "source_commit source_manifest_sha256 entrypoint_contract_sha256",
    )
    _require(
        isinstance(source["source_commit"], str)
        and re.fullmatch(r"[0-9a-f]{40}", source["source_commit"])
        and _sha(source["source_manifest_sha256"])
        and _sha(source["entrypoint_contract_sha256"]),
        "source-contract",
    )
    _require(isinstance(row["native_mappings"], list), "native-mappings")
    native = []
    for item in row["native_mappings"]:
        item = _shape(item, "path device inode qualified_native_reference")
        _path(item["path"])
        _require(
            _integer(item["device"]) and _integer(item["inode"], 1), "native-mapping"
        )
        _pin(item["qualified_native_reference"])
        _require(
            item["qualified_native_reference"]["path"] == item["path"],
            "native-reference",
        )
        native.append((item["path"], item["device"], item["inode"]))
    _require(native == sorted(set(native)), "native-mapping-order")
    _require(
        row["restart_counter_scope"]
        in {"systemd-unit:NRestarts", "coordinator-worker:restart_count"},
        "restart-scope",
    )
    if row["role"] in {"controller", "coordinator", "monitor"}:
        _require(
            row["restart_counter_scope"] == "systemd-unit:NRestarts", "restart-scope"
        )
    if row["role"] in {
        "learner",
        "actor-cpu-ring4",
        "arena-promotion",
        *(f"actor-gpu-{i}" for i in range(1, 7)),
    }:
        _require(
            row["restart_counter_scope"] == "coordinator-worker:restart_count",
            "restart-scope",
        )
    return _digest("strength-preservation-process-origin-v1", row)


def boot_digest(material: Mapping[str, Any]) -> str:
    row = _shape(
        material,
        "unit unit_file_state default_target owned_links required_reachability_edges",
    )
    _unit(row["unit"])
    _require(row["unit"].endswith((".service", ".timer")), "boot-kind")
    _unit(row["default_target"])
    _require(row["default_target"].endswith(".target"), "default-target")
    _require(
        row["unit_file_state"] in {"enabled", "enabled-runtime", "disabled", "static"},
        "enablement",
    )
    _require(
        isinstance(row["owned_links"], list)
        and isinstance(row["required_reachability_edges"], list),
        "boot-inventory",
    )
    parents = []
    links = []
    for value in row["owned_links"]:
        item = _shape(value, "path literal_target resolved_target")
        _path(item["path"])
        _path(item["resolved_target"])
        _require(
            isinstance(item["literal_target"], str)
            and item["literal_target"]
            and "\x00" not in item["literal_target"],
            "link-target",
        )
        path = PurePosixPath(item["path"])
        _require(
            path.name == row["unit"]
            and str(path.parent.parent)
            in {"/etc/systemd/system", "/run/systemd/system"},
            "owned-link",
        )
        parent = path.parent.name
        _require(parent.endswith((".target.wants", ".target.requires")), "boot-parent")
        _require(
            PurePosixPath(item["resolved_target"]).name == row["unit"], "link-unit"
        )
        parents.append(parent.rsplit(".", 1)[0])
        links.append(item["path"])
    _require(links == sorted(set(links)), "boot-link-order")
    _require(
        bool(links) == row["unit_file_state"].startswith("enabled"), "enablement-links"
    )
    edges = []
    reachable = {row["default_target"]}
    for item in row["required_reachability_edges"]:
        item = _shape(item, "from to relation")
        _unit(item["from"])
        _unit(item["to"])
        _require(item["relation"] in {"Wants", "Requires"}, "boot-relation")
        edges.append((item["from"], item["to"], item["relation"]))
    _require(edges == sorted(set(edges)), "boot-edge-order")
    for _ in edges:
        reachable.update(dst for src, dst, _rel in edges if src in reachable)
    _require(all(parent in reachable for parent in parents), "boot-unreachable")
    return _digest("strength-preservation-unit-boot-v1", row)


def birth_upper_ns(*, start_ticks: int, hz: int, calibration: Mapping[str, Any]) -> int:
    """Conditional current-wall-domain bound; caller owns historical mapping proof."""
    row = _shape(
        calibration,
        "boot_id expected_boot_id monotonic_before_ns boottime_before_ns wall_ns boottime_after_ns monotonic_after_ns max_bracket_ns offset_lower_ns offset_upper_ns namespaces",
    )
    _require(_integer(start_ticks) and _integer(hz, 1), "birth-input")
    _require(
        isinstance(row["boot_id"], str)
        and bool(row["boot_id"])
        and row["boot_id"] == row["expected_boot_id"],
        "birth-boot",
    )
    for key in "monotonic_before_ns boottime_before_ns wall_ns boottime_after_ns monotonic_after_ns max_bracket_ns offset_lower_ns offset_upper_ns".split():
        _require(_integer(row[key]), "birth-clock")
    m0, m1 = row["monotonic_before_ns"], row["monotonic_after_ns"]
    b0, b1 = row["boottime_before_ns"], row["boottime_after_ns"]
    _require(
        0 <= m1 - m0 <= row["max_bracket_ns"] and 0 <= b1 - b0 <= row["max_bracket_ns"],
        "birth-bracket",
    )
    _require(
        row["offset_lower_ns"]
        <= row["wall_ns"] - b1
        <= row["wall_ns"] - b0
        <= row["offset_upper_ns"],
        "birth-offset",
    )
    namespaces = _shape(row["namespaces"], "self pid1 target")
    for namespace in namespaces.values():
        ns = _shape(namespace, "pid time")
        for key in ("pid", "time"):
            _require(
                isinstance(ns[key], str)
                and re.fullmatch(key + r":\[[0-9]+\]", ns[key]),
                "birth-namespace",
            )
    _require(
        namespaces["self"] == namespaces["pid1"] == namespaces["target"],
        "birth-namespace",
    )
    _require(start_ticks * 1_000_000_000 // hz <= b1, "birth-future")
    return row["wall_ns"] - b0 + ((start_ticks + 1) * 1_000_000_000 + hz - 1) // hz


def running_seconds(
    *,
    now_monotonic_ns: int,
    activation_start_us: int,
    invocation_id: str,
    expected_invocation_id: str,
) -> int:
    _require(
        _integer(now_monotonic_ns) and _integer(activation_start_us, 1),
        "activation-clock",
    )
    _require(
        isinstance(invocation_id, str)
        and re.fullmatch(r"[0-9a-f]{32}", invocation_id)
        and invocation_id == expected_invocation_id,
        "activation-invocation",
    )
    age = now_monotonic_ns - activation_start_us * 1000
    _require(age >= 0, "activation-future")
    return (age + 999_999_999) // 1_000_000_000


def _clock(value: object) -> Mapping[str, Any]:
    row = _shape(value, "boot_id monotonic_ns wall_ns")
    _require(
        isinstance(row["boot_id"], str)
        and bool(row["boot_id"])
        and _integer(row["monotonic_ns"])
        and _integer(row["wall_ns"], 1),
        "clock",
    )
    return row


def job_age_seconds(
    *,
    current: Mapping[str, Any],
    now: Mapping[str, Any],
    absence: Mapping[str, Any] | None = None,
    creation_lower_ns: int | None = None,
) -> int:
    row = _shape(
        current,
        "boot_id manager_id unit id kind state read_start read_end creation_source_sha256",
    )
    _unit(row["unit"])
    _require(
        _integer(row["id"], 1)
        and row["kind"] == "start"
        and row["state"] in {"waiting", "running"},
        "job",
    )
    _require(isinstance(row["manager_id"], str) and bool(row["manager_id"]), "manager")
    end = _clock(now)
    first, last = _clock(row["read_start"]), _clock(row["read_end"])
    _require(
        row["boot_id"] == first["boot_id"] == last["boot_id"] == end["boot_id"]
        and first["monotonic_ns"] <= last["monotonic_ns"] <= end["monotonic_ns"],
        "job-clock",
    )
    _require((absence is None) != (creation_lower_ns is None), "unknown-job-age")
    if absence is not None:
        witness = _shape(
            absence, "boot_id manager_id unit job read_start read_end raw_sha256"
        )
        a0, a1 = _clock(witness["read_start"]), _clock(witness["read_end"])
        _require(
            witness["job"] is None
            and _sha(witness["raw_sha256"])
            and row["creation_source_sha256"] is None,
            "absence-proof",
        )
        _require(
            all(witness[k] == row[k] for k in ("boot_id", "manager_id", "unit")),
            "absence-binding",
        )
        _require(
            a0["boot_id"] == a1["boot_id"] == row["boot_id"]
            and a0["monotonic_ns"] <= a1["monotonic_ns"] <= first["monotonic_ns"],
            "absence-clock",
        )
        lower = a0["monotonic_ns"]
    else:
        _require(
            _integer(creation_lower_ns) and _sha(row["creation_source_sha256"]),
            "creation-proof",
        )
        assert creation_lower_ns is not None
        lower = creation_lower_ns
    _require(lower <= first["monotonic_ns"], "creation-future")
    return (end["monotonic_ns"] - lower + 999_999_999) // 1_000_000_000
