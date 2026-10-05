"""Read-only R3 preservation collection from predeclared, reviewed metadata.

This module never imports the training package or loads a model. A complete
capture is a measurement, not execution qualification. Registration and the
outer process deadline must be independently qualified before target use.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import PurePosixPath
from typing import Any, Callable, Mapping

from scripts import strength_freshness_cpu_collect_facts as facts
from scripts import strength_freshness_cpu_preservation as preservation
from scripts import strength_freshness_cpu_readonly as readonly

FORMAT = "strength-preservation-collector-registration-v1"
CONTRACT = facts.CONTRACT_SHA256
RUNTIME = "edgeconnect-strength-recovery-20261002.service"
WORKERS = {"learner", "arena-promotion", "actor-cpu-ring4", *preservation.ACTOR_ROLES}
ORIGIN_ENV_PREFIXES = (
    "PYTHON",
    "LD_",
    "DYLD_",
    "CUDA_",
    "TORCH",
    "TRITON",
    "OMP_",
    "MKL_",
    "OPENBLAS_",
)
FINGERPRINT = ("device", "inode", "mode", "uid", "gid", "bytes", "mtime_ns", "ctime_ns")
OPTIONAL_EMPTY_EXEC = ("ExecStartPre", "ExecStop", "ExecStopPost")


class CollectionRefusal(ValueError):
    """Fixed code only; private values are never interpolated into messages."""


def require(ok: object, reason: str) -> None:
    if not ok:
        raise CollectionRefusal(reason)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encoded(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError, UnicodeError):
        raise CollectionRefusal("canonical-json") from None


def strict_json(raw: bytes) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate-json-key")
            result[key] = value
        return result

    def invalid(_):
        raise CollectionRefusal("nonfinite-json")

    def finite(text):
        number = float(text)
        require(math.isfinite(number), "nonfinite-json")
        return number

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=invalid,
            parse_float=finite,
        )
    except CollectionRefusal:
        raise
    except (UnicodeError, ValueError, RecursionError):
        raise CollectionRefusal("invalid-json") from None
    require(isinstance(value, dict), "json-object")
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        require(depth <= 64, "json-depth")
        if isinstance(item, dict):
            pending.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            pending.extend((v, depth + 1) for v in item)
    return value


def properties(raw: str) -> dict[str, str]:
    result = {}
    for line in raw.splitlines():
        require("=" in line, "property-record")
        key, value = line.split("=", 1)
        require(key and key not in result, "property-key")
        result[key] = value
    return result


def natural(value: object, *, positive=False) -> int:
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9]+", value), "integer-property"
    )
    assert isinstance(value, str)
    number = int(value)
    require(number >= int(positive), "positive-property")
    return number


def capture_clock(raw: Mapping[str, Any]) -> dict[str, Any]:
    require(set(raw) == {"boot_id", "monotonic_ns", "wall_ns"}, "clock-fields")
    require(isinstance(raw["boot_id"], str) and bool(raw["boot_id"]), "clock-boot")
    require(
        all(type(raw[k]) is int and raw[k] > 0 for k in ("monotonic_ns", "wall_ns")),
        "clock-integers",
    )
    return {
        "boot_id": raw["boot_id"],
        "monotonic": raw["monotonic_ns"] / 1e9,
        "wall_ns": raw["wall_ns"],
    }


def scope_from(value: Mapping[str, Any]) -> readonly.ReadScope:
    require(
        isinstance(value, Mapping)
        and set(value)
        in (
            {"units", "targets", "files", "tails", "cached"},
            {"units", "targets", "files", "tails", "cached", "property_sets"},
        ),
        "scope-fields",
    )
    try:
        return readonly.ReadScope(
            units=value["units"],
            targets=tuple(value["targets"]),
            files={k: readonly.FileKey(**v) for k, v in value["files"].items()},
            tails={k: readonly.FileKey(**v) for k, v in value["tails"].items()},
            cached={k: readonly.CachedFile(**v) for k, v in value["cached"].items()},
            property_sets=None
            if value.get("property_sets") is None
            else {k: tuple(v) for k, v in value["property_sets"].items()},
        )
    except (TypeError, KeyError, AttributeError):
        raise CollectionRefusal("scope-shape") from None


def shape(value: object, keys: str, code: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping) and set(value) == set(keys.split()), code)
    assert isinstance(value, Mapping)
    return value


def integer(value: object, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def path(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("/")
        and not value.startswith("//")
        and str(PurePosixPath(value)) == value
        and ".." not in PurePosixPath(value).parts
        and "\0" not in value
    )


def decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeError:
        raise CollectionRefusal("private-text-encoding") from None


COMMON_FIELDS = frozenset(
    {
        "auxiliary_policy",
        "boot",
        "cached_references",
        "cohorts",
        "counter_scopes",
        "empty_property_rules",
        "encoding_contract_sha256",
        "heartbeats",
        "keys",
        "origins",
        "policy",
        "scope",
        "source_pins",
        "timer_addendum_sha256",
        "timer_environment_addendum_sha256",
        "units",
    }
)


def validate_common_registration(
    common: Mapping[str, Any], scope: readonly.ReadScope
) -> None:
    """Pure shared checks; not a complete registration or birth admission.

    Both strict format-specific constructors must validate their own header and
    authority first. This function performs no reads and grants no collector.
    """
    require(
        isinstance(common, Mapping) and set(common) == COMMON_FIELDS,
        "common-registration-fields",
    )
    r = common
    try:
        require(
            r["encoding_contract_sha256"] == CONTRACT
            and r["timer_addendum_sha256"] == facts.TIMER_ADDENDUM_SHA256
            and r["timer_environment_addendum_sha256"]
            == facts.TIMER_ENVIRONMENT_ADDENDUM_SHA256,
            "encoding-contract",
        )
        require(scope_from(r["scope"]) == scope, "io-scope-registration")
        p = r["policy"]
        require(
            p["static"]["runtime_name"] == RUNTIME
            and p["static"]["source_commit"] == preservation.R3_SOURCE_COMMIT,
            "immutable-r3-registration",
        )
        require(
            set(p["workers"]) == WORKERS
            and set(r["heartbeats"]) == WORKERS
            and len(r["cohorts"]) == 24,
            "registered-roles",
        )
        require(
            set(r["origins"]) == {"controller", "coordinator", *WORKERS, "monitor"},
            "origin-inventory",
        )
        require(
            set(r["units"]) == set(scope.units) and RUNTIME in r["units"],
            "unit-inventory",
        )
        require(
            set(r["keys"])
            == {
                "run",
                "continuation",
                "profile_authority",
                "profile",
                "run_source",
                "release_source",
                "source_manifest",
                "coordinator",
                "champion",
                "metrics",
            },
            "metadata-keys",
        )
        require(set(r["cached_references"]) == set(scope.cached), "cached-inventory")
        require(
            set(r["counter_scopes"])
            == {"controller", "coordinator", "workers", "monitor"},
            "counter-scope-fields",
        )
        require(
            r["counter_scopes"]
            == {
                "controller": "owning-unit-NRestarts",
                "coordinator": "owning-unit-NRestarts",
                "workers": "coordinator-worker-restart_count",
                "monitor": "owning-unit-NRestarts",
            },
            "restart-counter-scope",
        )
        require(set(r["cohorts"]) == set(p["cohorts"]), "cohort-policy-binding")
        require(set(r["boot"]) == {"default_target", "edges"}, "boot-registration")
        for name, spec in r["units"].items():
            require(
                set(spec)
                == {"kind", "fragment", "dropins", "environment_files", "owned_links"},
                "unit-registration-fields",
            )
            require(
                spec["kind"] in {"runtime", "long_running", "oneshot", "timer"},
                "unit-kind",
            )
            require(
                (name == RUNTIME) == (spec["kind"] == "runtime"), "runtime-unit-scope"
            )
        require(
            sum(v["kind"] == "long_running" for v in r["units"].values()) == 1,
            "monitor-count",
        )
        for row in r["cohorts"].values():
            require(
                set(row) == {"key", "worker", "parent_role"}
                and row["parent_role"] in preservation.ACTOR_ROLES,
                "cohort-registration",
            )
            require(
                row["key"] in scope.files
                and isinstance(row["worker"], str)
                and row["worker"],
                "cohort-file-binding",
            )
        for key, value in r["keys"].items():
            require(
                value in (scope.tails if key == "metrics" else scope.files),
                "metadata-file-binding",
            )
        require(
            all(key in scope.files for key in r["heartbeats"].values()),
            "heartbeat-file-binding",
        )
        require(
            isinstance(r["source_pins"], dict) and 0 < len(r["source_pins"]) <= 256,
            "source-pin-inventory",
        )
        for key, pin in r["source_pins"].items():
            shape(pin, "sha256 bytes", "source-pin-fields")
            require(
                key in scope.files
                and preservation.sha(pin["sha256"])
                and integer(pin["bytes"], 1)
                and pin["bytes"] <= 1024 * 1024,
                "source-pin",
            )
        for ref in r["cached_references"].values():
            shape(
                ref,
                "sha256 qualification_sha256 literal_stat resolved_stat",
                "cache-reference-shape",
            )
            require(
                preservation.sha(ref["sha256"])
                and preservation.sha(ref["qualification_sha256"]),
                "qualified-cache-reference",
            )
            for field in ("literal_stat", "resolved_stat"):
                st = shape(ref[field], " ".join(FINGERPRINT), "cache-stat-fields")
                require(
                    all(integer(v) for v in st.values()) and st["mode"] <= 0o7777,
                    "cache-stat-values",
                )
        roles = {"controller", "coordinator", *WORKERS}
        require(set(p["expected_processes"]) == roles, "expected-process-inventory")
        for role, spec in r["origins"].items():
            shape(
                spec,
                "interpreter cwd argv_sha256 argv_bytes approved_template_id environment native_keys entrypoint_contract_sha256 restart_counter_scope",
                "origin-spec-fields",
            )
            require(
                spec["interpreter"] in scope.cached and path(spec["cwd"]),
                "origin-interpreter-cwd",
            )
            require(
                all(
                    preservation.sha(spec[k])
                    for k in (
                        "argv_sha256",
                        "approved_template_id",
                        "entrypoint_contract_sha256",
                    )
                )
                and integer(spec["argv_bytes"], 1),
                "origin-pins",
            )
            require(
                isinstance(spec["native_keys"], list)
                and len(set(spec["native_keys"])) == len(spec["native_keys"])
                and all(k in scope.cached for k in spec["native_keys"]),
                "origin-native-inventory",
            )
            require(
                spec["restart_counter_scope"]
                == (
                    "coordinator-worker:restart_count"
                    if role in WORKERS
                    else "systemd-unit:NRestarts"
                ),
                "origin-restart-counter-scope",
            )
            require(isinstance(spec["environment"], dict), "origin-environment-fields")
            for name, row in spec["environment"].items():
                require(
                    isinstance(name, str)
                    and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name),
                    "origin-environment-key",
                )
                shape(
                    row, "present value_sha256 value_bytes", "origin-environment-fields"
                )
                require(
                    type(row["present"]) is bool
                    and integer(row["value_bytes"])
                    and (
                        preservation.sha(row["value_sha256"])
                        if row["present"]
                        else row["value_sha256"] is None and row["value_bytes"] == 0
                    ),
                    "origin-environment-pin",
                )
        rules = r["empty_property_rules"]
        require(
            isinstance(rules, dict)
            and set(rules)
            <= {
                "Environment",
                "PassEnvironment",
                "UnsetEnvironment",
                "EnvironmentFiles",
                *OPTIONAL_EMPTY_EXEC,
            }
            and all(v == "systemd-255-empty-" + k for k, v in rules.items()),
            "empty-property-rules",
        )
        boot = r["boot"]
        require(
            boot["default_target"] in scope.targets and isinstance(boot["edges"], list),
            "boot-target-binding",
        )
        edge_tuples = []
        for edge in boot["edges"]:
            shape(edge, "from to relation", "boot-edge-fields")
            require(
                edge["from"] in scope.targets
                and edge["to"] in scope.targets
                and edge["relation"] in {"Wants", "Requires"},
                "boot-edge-scope",
            )
            edge_tuples.append((edge["from"], edge["to"], edge["relation"]))
        require(edge_tuples == sorted(set(edge_tuples)), "boot-edge-order")
        for name, spec in r["units"].items():
            require(
                scope.units[name]
                == ("timer" if spec["kind"] == "timer" else "service"),
                "unit-kind-binding",
            )
            require(
                spec["fragment"] in scope.files
                and isinstance(spec["dropins"], list)
                and len(spec["dropins"]) == len(set(spec["dropins"]))
                and all(k in scope.files for k in spec["dropins"]),
                "unit-file-scope",
            )
            require(
                isinstance(spec["environment_files"], list), "environment-file-fields"
            )
            env_keys = []
            for item in spec["environment_files"]:
                shape(item, "key ignore_missing", "environment-file-fields")
                require(
                    item["key"] in scope.files and type(item["ignore_missing"]) is bool,
                    "environment-file-scope",
                )
                env_keys.append(item["key"])
            require(
                len(env_keys) == len(set(env_keys))
                and (not env_keys or spec["kind"] != "timer"),
                "environment-file-inventory",
            )
            require(isinstance(spec["owned_links"], list), "boot-links-registration")
            for link in spec["owned_links"]:
                shape(link, "path literal_target resolved_target", "boot-link-fields")
                require(
                    path(link["path"])
                    and path(link["resolved_target"])
                    and isinstance(link["literal_target"], str),
                    "boot-link-path",
                )
    except (KeyError, TypeError, AttributeError):
        raise CollectionRefusal("registration-shape") from None


class _IdentityMeasurements:
    """Private reusable operations; constructors must provide separate admission."""

    def __init__(
        self,
        registration: Mapping[str, Any],
        io: readonly.ReadOnlyIO,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.reg = json.loads(encoded(registration))
        self.io, self.sleep = io, sleep
        self.audit: list[dict[str, Any]] = []
        self.cached_measurements: dict[str, dict[str, Any]] = {}
        self.derivations: dict[str, Any] = {}
        self.absences: dict[str, Any] = {}

    def take(self, observation):
        self.audit.append(dict(observation.audit))
        return observation.value

    def clock(self):
        return self.take(self.io.clock())

    def read_json(self, key: str):
        return strict_json(self.take(self.io.read(key)))

    def unit(self, name: str):
        return properties(self.take(self.io.query("unit", name)))

    def file_pin(self, key: str):
        require(key in self.io.scope.files, "unregistered-file-key")
        obs = self.io.read(key)
        data = self.take(obs)
        st = obs.audit["stat_before"]
        require(
            obs.audit["raw"] == {"sha256": sha(data), "bytes": len(data)},
            "file-raw-binding",
        )
        require(
            all(st[k] == obs.audit["stat_after"][k] for k in FINGERPRINT)
            and st["bytes"] == len(data),
            "file-read-raced",
        )
        path = self.io.scope.files[key].path
        return {
            "literal_path": path,
            "resolved_path": path,
            "sha256": sha(data),
            "bytes": len(data),
            **{k: st[k] for k in ("mode", "uid", "gid")},
        }

    def cache(self, key: str):
        require(key in self.reg["cached_references"], "unregistered-cache-key")
        measured = self.take(self.io.stat_cached(key))
        expected = self.reg["cached_references"][key]
        require(
            (measured["literal"], measured["resolved"])
            == (self.io.scope.cached[key].literal, self.io.scope.cached[key].resolved),
            "qualified-cache-path-drift",
        )
        require(
            set(expected)
            == {"sha256", "qualification_sha256", "literal_stat", "resolved_stat"},
            "cache-reference-shape",
        )
        require(
            preservation.sha(expected["sha256"])
            and preservation.sha(expected["qualification_sha256"]),
            "qualified-cache-reference",
        )
        for field in ("literal_stat", "resolved_stat"):
            require(
                {k: measured[field][k] for k in FINGERPRINT} == expected[field],
                "qualified-cache-stat-drift",
            )
        self.cached_measurements[key] = measured
        return {
            "path": measured["resolved"],
            "sha256": expected["sha256"],
            "bytes": measured["resolved_stat"]["bytes"],
        }

    def verify_source_pins(self):
        """Read and hash only registered small control-source files."""
        result = {}
        for key, expected in self.reg["source_pins"].items():
            pin = self.file_pin(key)
            require(
                {k: pin[k] for k in ("sha256", "bytes")} == expected, "source-pin-drift"
            )
            result[key] = pin
        return result

    def verify_cached_references(self):
        """Check qualified cached stat fingerprints; never read payload bytes."""
        return {key: self.cache(key) for key in self.reg["cached_references"]}

    def environment(self, unit: str, props):
        if unit.endswith(".timer"):
            require(not self.reg["units"][unit]["environment_files"], "timer-env-files")
            return facts.environment_digest(
                {"unit": unit, "applicability": "not-applicable-timer"}
            )

        def field(name):
            if name in props:
                raw = props[name].encode()
                return {
                    "present": True,
                    "raw_sha256": sha(raw),
                    "bytes": len(raw),
                    "effective_empty_rule": None,
                }
            rule = self.reg["empty_property_rules"].get(name)
            require(rule == "systemd-255-empty-" + name, "unqualified-property-absence")
            return {
                "present": False,
                "raw_sha256": None,
                "bytes": 0,
                "effective_empty_rule": rule,
            }

        envs = []
        declared = self.reg["units"][unit]["environment_files"]
        expected_paths = [self.io.scope.files[e["key"]].path for e in declared]
        # Qualified simple absolute paths only; order and ignore_errors are real.
        raw_files = props.get("EnvironmentFiles")
        if raw_files is None:
            require(
                self.reg["empty_property_rules"].get("EnvironmentFiles")
                == "systemd-255-empty-EnvironmentFiles"
                and not declared,
                "environment-files-absent",
            )
        else:
            parsed = re.findall(r"(/[^\s()]+) \(ignore_errors=(yes|no)\)", raw_files)
            rebuilt = " ".join(
                f"{path} (ignore_errors={flag})" for path, flag in parsed
            )
            require(
                rebuilt == raw_files and [p for p, _ in parsed] == expected_paths,
                "environment-file-inventory",
            )
            require(
                [v == "yes" for _, v in parsed]
                == [e["ignore_missing"] for e in declared],
                "environment-file-options",
            )
        for item in declared:
            try:
                pin = self.file_pin(item["key"])
                envs.append(
                    {**pin, "ignore_missing": item["ignore_missing"], "exists": True}
                )
            except FileNotFoundError:
                require(item["ignore_missing"] is True, "required-env-file-missing")
                path = self.io.scope.files[item["key"]].path
                envs.append(
                    {
                        "literal_path": path,
                        "resolved_path": path,
                        "ignore_missing": True,
                        "exists": False,
                        **dict.fromkeys(("sha256", "bytes", "mode", "uid", "gid")),
                    }
                )
        return facts.environment_digest(
            {
                "unit": unit,
                "inline_environment": field("Environment"),
                "pass_environment": field("PassEnvironment"),
                "unset_environment": field("UnsetEnvironment"),
                "environment_files": envs,
            }
        )

    def unit_static(self, name: str, props, target_props) -> dict[str, Any]:
        require(name in self.reg["units"], "unregistered-unit")
        # systemd255's execution-array printer emits no property line for an
        # empty array, even with --all. Only these observed optional properties
        # may use an explicitly registered rule; never infer missing ExecStart.
        observed = props
        props = dict(props)
        omitted = {}
        if name.endswith(".service"):
            for key in OPTIONAL_EMPTY_EXEC:
                if key not in props:
                    rule = self.reg["empty_property_rules"].get(key)
                    require(
                        rule == "systemd-255-empty-" + key,
                        "unqualified-execution-property-absence",
                    )
                    props[key] = ""
                    omitted[key] = rule
        if omitted:
            self.derivations.setdefault("omitted_execution_properties", []).append(
                {
                    "unit": name,
                    "parsed_properties_sha256": sha(encoded(observed)),
                    "observed_present": False,
                    "normalization_rules": omitted,
                }
            )
        fields = facts.TIMER_STATIC if name.endswith(".timer") else facts.SERVICE_STATIC
        require(
            set(fields).union(
                {
                    "Id",
                    "FragmentPath",
                    "DropInPaths",
                    "NeedDaemonReload",
                    "UnitFileState",
                }
            )
            <= props.keys()
            and all(isinstance(v, str) for v in props.values()),
            "missing-unit-property",
        )
        spec = self.reg["units"][name]
        fragment = self.file_pin(spec["fragment"])
        drops = [self.file_pin(k) for k in spec["dropins"]]
        require(
            props["Id"] == name
            and props["FragmentPath"] == fragment["literal_path"]
            and props["DropInPaths"].split() == [x["literal_path"] for x in drops],
            "unit-file-binding",
        )
        definition = facts.definition_digest(
            {
                "unit": name,
                "fragment": fragment,
                "dropins": drops,
                "loaded_stable_properties": {k: props[k] for k in fields},
                "need_daemon_reload": props["NeedDaemonReload"] != "no",
            }
        )
        links = self.take(self.io.boot_links(name))
        normalized_links = [
            {
                "path": x["path"],
                "literal_target": x["literal"],
                "resolved_target": x["resolved"],
            }
            for x in links
        ]
        require(normalized_links == spec["owned_links"], "owned-boot-links-drift")
        for edge in self.reg["boot"]["edges"]:
            require(
                edge["from"] in target_props
                and edge["relation"] in target_props[edge["from"]]
                and edge["to"] in target_props[edge["from"]][edge["relation"]].split(),
                "missing-boot-proof-edge",
            )
        boot = facts.boot_digest(
            {
                "unit": name,
                "unit_file_state": props["UnitFileState"],
                "default_target": self.reg["boot"]["default_target"],
                "owned_links": normalized_links,
                "required_reachability_edges": self.reg["boot"]["edges"],
            }
        )
        return {
            "definition_sha256": definition,
            "environment_sha256": self.environment(name, props),
            "boot_links_sha256": boot,
            "enabled": props["UnitFileState"] in {"enabled", "enabled-runtime"},
        }

    def origin(self, role: str, process):
        require(role in self.reg["origins"], "unregistered-origin-role")
        spec = self.reg["origins"][role]
        require(
            set(spec)
            == {
                "interpreter",
                "cwd",
                "argv_sha256",
                "argv_bytes",
                "approved_template_id",
                "environment",
                "native_keys",
                "entrypoint_contract_sha256",
                "restart_counter_scope",
            },
            "origin-spec-fields",
        )
        executable = self.cache(spec["interpreter"])
        argv = process["cmdline"]
        require(
            isinstance(argv, bytes)
            and argv.endswith(b"\0")
            and sha(argv) == spec["argv_sha256"]
            and len(argv) == spec["argv_bytes"],
            "unapproved-command",
        )
        literal = decode(argv.split(b"\0", 1)[0])
        require(
            literal == self.io.scope.cached[spec["interpreter"]].literal
            and process["exe"] == executable["path"]
            and process["cwd"] == spec["cwd"],
            "process-origin-drift",
        )
        env = {}
        raw_env = process["environ"]
        require(
            isinstance(raw_env, bytes) and (not raw_env or raw_env.endswith(b"\0")),
            "environment-record-termination",
        )
        for item in raw_env[:-1].split(b"\0") if raw_env else ():
            require(b"=" in item, "environment-record")
            k, value = item.split(b"=", 1)
            key = decode(k)
            require(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key), "environment-key")
            require(key not in env, "duplicate-environment")
            env[key] = value
        expected = spec["environment"]
        require(
            not {k for k in env if k.startswith(ORIGIN_ENV_PREFIXES)}.difference(
                expected
            ),
            "unregistered-import-override",
        )
        entries = []
        for key in sorted(expected):
            value = env.get(key)
            row = {
                "key": key,
                "present": value is not None,
                "value_sha256": None if value is None else sha(value),
                "value_bytes": 0 if value is None else len(value),
            }
            require(row == {"key": key, **expected[key]}, "environment-origin-drift")
            entries.append(row)
        native = []
        require(isinstance(process["maps"], bytes), "native-maps-unavailable")
        maps = decode(process["maps"])
        approved = {self.io.scope.cached[k].resolved: k for k in spec["native_keys"]}
        found = {}
        for line in maps.splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) < 6:
                continue
            path = fields[5]
            if "star_native" not in path and "deltrel_native" not in path:
                continue
            require(
                path in approved and not path.endswith(" (deleted)"),
                "unexpected-native-mapping",
            )
            require(
                re.fullmatch(r"[0-9a-fA-F]+:[0-9a-fA-F]+", fields[3])
                and re.fullmatch(r"[0-9]+", fields[4]),
                "native-map-identity-shape",
            )
            major, minor = fields[3].split(":")
            try:
                dev = os.makedev(int(major, 16), int(minor, 16))
            except (ValueError, OverflowError):
                raise CollectionRefusal("native-map-device") from None
            found[(path, dev, int(fields[4]))] = approved[path]
        require({x[0] for x in found} == set(approved), "required-native-not-mapped")
        for (path, device, inode), key in sorted(found.items()):
            reference = self.cache(key)
            st = self.cached_measurements[key]["resolved_stat"]
            require(
                (device, inode) == (st["device"], st["inode"]),
                "native-map-file-identity",
            )
            native.append(
                {
                    "path": path,
                    "device": device,
                    "inode": inode,
                    "qualified_native_reference": reference,
                }
            )
        return facts.origin_digest(
            {
                "role": role,
                "executable": {
                    "literal_entrypoint": literal,
                    "resolved_proc_exe": process["exe"],
                    "qualified_file_reference": executable,
                },
                "cwd_resolved": process["cwd"],
                "argv": {
                    "sha256": sha(argv),
                    "bytes": len(argv),
                    "approved_template_id": spec["approved_template_id"],
                },
                "relevant_environment": entries,
                "source_contract": {
                    "source_commit": self.reg["policy"]["static"]["source_commit"],
                    "source_manifest_sha256": self.reg["policy"]["static"][
                        "source_manifest_sha256"
                    ],
                    "entrypoint_contract_sha256": spec["entrypoint_contract_sha256"],
                },
                "native_mappings": native,
                "restart_counter_scope": spec["restart_counter_scope"],
            }
        )


class IdentityCollector(_IdentityMeasurements):
    """Registered static/origin checks; no capture orchestration or CLI."""

    def __init__(
        self,
        registration: Mapping[str, Any],
        io: readonly.ReadOnlyIO,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ):
        super().__init__(registration, io, sleep=sleep)
        try:
            self._validate_registration()
        except (KeyError, TypeError, AttributeError):
            raise CollectionRefusal("registration-shape") from None

    def _validate_registration(self):
        r = self.reg
        require(
            set(r)
            == {
                "format",
                "schema_version",
                "encoding_contract_sha256",
                "timer_addendum_sha256",
                "timer_environment_addendum_sha256",
                "scope",
                "policy",
                "keys",
                "units",
                "heartbeats",
                "cohorts",
                "origins",
                "auxiliary_policy",
                "cached_references",
                "source_pins",
                "birth_reference",
                "boot",
                "empty_property_rules",
                "counter_scopes",
            },
            "registration-fields",
        )
        require(
            r["format"] == FORMAT
            and type(r["schema_version"]) is int
            and r["schema_version"] == 1,
            "registration-format",
        )
        p = r["policy"]
        roles = {"controller", "coordinator", *WORKERS}
        birth = shape(
            r["birth_reference"],
            "boot_id qualification_sha256 offset_lower_ns offset_upper_ns max_bracket_ns bounds",
            "birth-reference-fields",
        )
        require(
            isinstance(birth["boot_id"], str)
            and birth["boot_id"]
            and preservation.sha(birth["qualification_sha256"]),
            "birth-reference-binding",
        )
        require(
            all(
                integer(birth[k], 1)
                for k in ("offset_lower_ns", "offset_upper_ns", "max_bracket_ns")
            )
            and birth["offset_lower_ns"] <= birth["offset_upper_ns"],
            "birth-reference-bounds",
        )
        require(
            set(birth["bounds"]) == roles
            and all(integer(v, 1) for v in birth["bounds"].values())
            and birth["bounds"]["learner"] == p["learner_birth_upper_ns"],
            "birth-role-bounds",
        )
        validate_common_registration(
            {key: r[key] for key in COMMON_FIELDS}, self.io.scope
        )
