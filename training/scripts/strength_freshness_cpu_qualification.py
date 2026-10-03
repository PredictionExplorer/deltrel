"""Dummy-only Linux applicability harness; never a production qualification.

The CLI validates plans and exposes explicitly authorized dummy roles plus
read-only arming/audit stages. There is no installer or production entrypoint.
A later reviewed launcher must supply immutable inputs and protected authority;
target execution remains unqualified. No production LinuxHost execute path or
GPU inspection/reservation is used.
"""

from __future__ import annotations

import argparse
import configparser
import shlex
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import time
import sys
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, cast

from scripts import strength_freshness_linux as linux
from scripts import strength_freshness_units as support

FORMAT = "strength-freshness-cpu-plan-v1"
RESULT = "strength-freshness-linux-target-cpu-evidence-v1"
PROTOCOL_SHA = "62a7712654b2c08fb8d04f74cb9da8cad1a538d3d3df5913566f99c3d99c58a0"
ADDENDA = (
    "3dc0d39e1b55522fdd3615456ffa5a766f7b383a3acf324b703b6a6f031f2093",
    "713339a63dd10af56a5cf45f345ba387b077ac508d1203c01b3623460407d423",
    "bd8bb22a630a4a4e2054d3ecfcfcc5b2bd828850d074b4f297b9d1533ebc5ff4",
    "51eeeecf1b1c892c56482cba33263eb746e258a0d338e09074cf999e576ff657",
    "d44a31609e2c0645e0810a6e0da83a8b01d9c69752a778a53f9f171615de2e60",
    "c0dd13d6df6a7639550975c6c0e22139e9256ed3d29341a8a40318e83392acda",
)
CASES = (
    "typed-properties",
    "pid-cgroup-lease",
    "pending-start-job",
    "bounded-drain",
    "support-partial-transaction",
    "persistent-boot-edges",
    "fast-natural-child-exit",
    "proof-freeze-and-retention",
    "cleanup-and-retirement",
    "execution-gate-refusal",
)
CLAIMS = dict.fromkeys(
    (
        "execution_qualified",
        "full_linux_controller_qualified",
        "real_reboot_qualified",
        "cuda_qualified",
        "r4_activated",
    ),
    False,
)
LIMITS = {
    "work": 390,
    "cleanup": 540,
    "observer": 570,
    "observer_terminal": 575,
    "publisher": 590,
    "publisher_terminal": 595,
    "audit": 600,
}
ROLES = {"observer", "publisher", "cleanup", "watchdog", "workload"}
HASH = re.compile(r"[0-9a-f]{64}\Z")
HEX32 = re.compile(r"[0-9a-f]{32}\Z")
MAX_OUTPUT = 32 * 2**20


class Refusal(RuntimeError):
    pass


def require(value: object, reason: str) -> None:
    if not value:
        raise Refusal(reason)


def encode(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> Path:
    require(isinstance(value, str), "path-string")
    assert isinstance(value, str)
    path = Path(value)
    require(
        path.is_absolute()
        and str(path) == value
        and ".." not in path.parts
        and re.fullmatch(r"/[A-Za-z0-9_./-]+", value),
        "canonical-absolute-path",
    )
    return path


def exact(value: object, keys: set[str], reason: str) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == keys, reason)
    assert isinstance(value, dict)
    return value


def pin_shape(value: object) -> dict[str, Any]:
    pin = exact(value, {"path", "sha256", "bytes"}, "artifact-fields")
    canonical(pin["path"])
    require(
        isinstance(pin["sha256"], str) and HASH.fullmatch(pin["sha256"]), "artifact-sha"
    )
    require(
        type(pin["bytes"]) is int and 0 <= pin["bytes"] <= 64 * 2**20, "artifact-size"
    )
    return pin


@dataclass(frozen=True)
class Plan:
    """Validated JSON snapshot; methods never retain the caller's mutable mapping."""

    data: bytes
    checksum: str

    @property
    def value(self) -> dict[str, Any]:
        return json.loads(self.data)

    @classmethod
    def parse(cls, data: bytes, expected: str) -> Plan:
        require(
            len(data) <= 2**20 and HASH.fullmatch(expected) and sha(data) == expected,
            "plan-byte-pin",
        )

        def pairs(items):
            result = {}
            for key, value in items:
                require(key not in result, "duplicate-json-key")
                result[key] = value
            return result

        value = json.loads(
            data,
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(Refusal("nonfinite-json")),
        )
        cls.validate(value)
        return cls(bytes(data), expected)

    @staticmethod
    def validate(value: object) -> None:
        p = exact(
            value,
            {
                "format",
                "schema_version",
                "attempt_id",
                "nonce",
                "boot_id",
                "protocol_sha256",
                "addenda_sha256",
                "limits",
                "claims",
                "input_root",
                "scratch_root",
                "protected_roots",
                "units",
                "files",
                "source_pins",
                "python",
                "control_root",
                "boot_topology",
                "cases",
                "bindings",
                "preservation",
                "runtime_inputs",
            },
            "plan-fields",
        )
        require(
            p["format"] == FORMAT
            and type(p["schema_version"]) is int
            and p["schema_version"] == 1,
            "plan-schema",
        )
        require(
            isinstance(p["attempt_id"], str)
            and re.fullmatch(r"[a-z0-9-]{1,48}", p["attempt_id"]),
            "attempt-id",
        )
        require(isinstance(p["nonce"], str) and HEX32.fullmatch(p["nonce"]), "nonce")
        require(
            isinstance(p["boot_id"], str)
            and re.fullmatch(r"[0-9a-f-]{36}", p["boot_id"]),
            "boot-id",
        )
        require(
            p["protocol_sha256"] == PROTOCOL_SHA
            and p["addenda_sha256"] == list(ADDENDA),
            "protocol-binding",
        )
        require(
            p["limits"] == LIMITS
            and p["claims"] == CLAIMS
            and all(type(x) is int for x in p["limits"].values())
            and all(type(x) is bool for x in p["claims"].values()),
            "scope-or-budget",
        )
        require(p["cases"] == list(CASES), "all-predeclared-cases-required")
        prefix = "edgeconnect-cpuqual-" + p["nonce"]
        scratch, inputs = canonical(p["scratch_root"]), canonical(p["input_root"])
        require(
            scratch == Path("/run") / prefix and inputs != scratch, "scratch-namespace"
        )
        control = canonical(p["control_root"])
        require(
            isinstance(p["protected_roots"], list)
            and 2 <= len(p["protected_roots"]) <= 16,
            "protected-roots",
        )
        protected = [canonical(x) for x in p["protected_roots"]] + [control, inputs]
        for root in protected:
            require(
                not scratch.is_relative_to(root) and not root.is_relative_to(scratch),
                "scratch-overlap",
            )
        require(
            isinstance(p["source_pins"], list) and 6 <= len(p["source_pins"]) <= 256,
            "source-closure",
        )
        pins = [pin_shape(x) for x in p["source_pins"]]
        require(all(x["bytes"] <= 2**20 for x in pins), "control-source-size")
        require(len({x["path"] for x in pins}) == len(pins), "duplicate-source-pin")
        python = exact(
            p["python"], {"path", "resolved_path", "sha256", "bytes"}, "interpreter-pin"
        )
        pin_shape({k: python[k] for k in ("path", "sha256", "bytes")})
        canonical(python["resolved_path"])
        require(Path(python["path"]).name == "python", "literal-python-entry")
        runtime = exact(
            p["runtime_inputs"],
            {
                "runtime_root",
                "source_commit",
                "source_marker",
                "source_manifest",
                "pyvenv",
                "native_wrapper",
                "native_binary",
                "qualification",
            },
            "runtime-input-fields",
        )
        runtime_root = canonical(runtime["runtime_root"])
        require(
            runtime["source_commit"] == "7dca37252714bbe0c52d35a380ec175d743d1938",
            "immutable-r4-source",
        )
        require(
            canonical(p["python"]["path"]) == runtime_root / ".venv/bin/python",
            "runtime-interpreter-binding",
        )
        require(
            any(runtime_root.is_relative_to(root) for root in protected),
            "runtime-not-protected",
        )
        for key in (
            "source_marker",
            "source_manifest",
            "pyvenv",
            "native_wrapper",
            "native_binary",
            "qualification",
        ):
            pin_shape(runtime[key])
        require(
            runtime["source_marker"]["path"]
            == str(runtime_root.parent / "SOURCE_COMMIT")
            and runtime["source_manifest"]["path"]
            == str(runtime_root.parent / "SOURCE_SHA256SUMS")
            and runtime["pyvenv"]["path"] == str(runtime_root / ".venv/pyvenv.cfg"),
            "runtime-metadata-locations",
        )
        require(
            all(
                canonical(runtime[key]["path"]).is_relative_to(runtime_root / ".venv")
                for key in ("native_wrapper", "native_binary")
            ),
            "runtime-native-locations",
        )
        require(
            runtime["native_wrapper"]["bytes"] <= 2**20
            and runtime["source_marker"]["bytes"] <= 256
            and runtime["source_manifest"]["bytes"] <= 2**20
            and runtime["pyvenv"]["bytes"] <= 4096
            and runtime["qualification"]["bytes"] <= 2**20,
            "runtime-metadata-bounds",
        )
        required = {
            "strength_freshness_cpu_qualification.py",
            "strength_freshness_cpu_lifecycle.py",
            "strength_freshness_linux.py",
            "strength_freshness_units.py",
            "strength_freshness_guard.py",
            "qualify_cloud_gpu_window.py",
            "strength_freshness_cpu_driver.py",
            "strength_freshness_cpu_support_case.py",
            "strength_freshness_cpu_preservation.py",
            "strength_freshness_cpu_fixture.py",
        }
        require(
            required <= {Path(x["path"]).name for x in pins},
            "missing-control-dependency",
        )
        require(
            isinstance(p["units"], dict) and 5 <= len(p["units"]) <= 12, "unit-count"
        )
        roles = []
        for name, unit in p["units"].items():
            require(
                re.fullmatch(
                    re.escape(prefix) + r"-[a-z][a-z0-9-]{0,40}\.(service|timer)", name
                ),
                "unit-namespace",
            )
            exact(
                unit,
                {
                    "role",
                    "mode",
                    "payload",
                    "installed_path",
                    "before",
                    "after",
                    "boot_links",
                },
                "unit-fields",
            )
            require(unit["role"] in ROLES, "unit-role")
            roles.append(unit["role"])
            if unit["role"] == "workload" and name.endswith(".service"):
                require(
                    unit["mode"]
                    in {
                        "sleep",
                        "term_tree",
                        "fast_receipt",
                        "lock_holder",
                        "lock_contender",
                        "support_transaction",
                    },
                    "payload-mode",
                )
                payload = exact(
                    unit["payload"],
                    {"seconds", "output_dir", "lock_path"}
                    | (
                        {"scenario"} if unit["mode"] == "support_transaction" else set()
                    ),
                    "payload-fields",
                )
                require(
                    type(payload["seconds"]) in (int, float)
                    and math.isfinite(payload["seconds"])
                    and 0 <= payload["seconds"] <= 30,
                    "payload-duration",
                )
                require(
                    canonical(payload["output_dir"]) == scratch / "payloads" / name,
                    "payload-output",
                )
                if unit["mode"] == "support_transaction":
                    scenario = pin_shape(payload["scenario"])
                    require(
                        canonical(scenario["path"]).is_relative_to(inputs),
                        "support-scenario-location",
                    )
                require(
                    payload["lock_path"] in (None, str(scratch / "qualification.lock")),
                    "dummy-lock-path",
                )
            else:
                require(
                    unit["mode"] is None and unit["payload"] is None,
                    "control-not-payload",
                )
            require(
                unit["installed_path"] == "/etc/systemd/system/" + name,
                "unit-installed-path",
            )
            require(
                isinstance(unit["boot_links"], dict) and len(unit["boot_links"]) <= 2,
                "boot-links",
            )
            if unit["role"] in {"observer", "publisher"}:
                require(unit["boot_links"] == {}, "inert-control-has-no-boot-edge")
            for link, target in unit["boot_links"].items():
                path = canonical(link)
                require(
                    path.parent.parent == Path("/etc/systemd/system")
                    and path.name == name
                    and path.parent.name.endswith((".target.wants", ".target.requires"))
                    and target == unit["installed_path"],
                    "persistent-dummy-link",
                )
            for stage in ("before", "after"):
                s = exact(
                    unit[stage],
                    {"unit", "properties", "environment_files"},
                    "unit-stage",
                )
                pin_shape(s["unit"])
                require(
                    Path(s["unit"]["path"]).is_relative_to(inputs),
                    "unit-source-location",
                )
                props = s["properties"]
                require(
                    isinstance(props, dict)
                    and set(props) == set(linux.property_names(name)) - linux.VARIABLE
                    and all(isinstance(x, str) for x in props.values()),
                    "expected-static-properties",
                )
                require(
                    props["Id"] == name
                    and props["FragmentPath"] == unit["installed_path"]
                    and props["DropInPaths"] == ""
                    and props["LoadState"] == "loaded",
                    "unit-static-identity",
                )
                require(
                    isinstance(s["environment_files"], list)
                    and len(s["environment_files"]) <= 2,
                    "env-count",
                )
                if name.endswith(".timer"):
                    require(not s["environment_files"], "timer-env")
                else:
                    require(
                        props["Restart"] == "no"
                        and props["KillMode"] == "control-group"
                        and props["SendSIGKILL"] == "yes"
                        and props["User"] == "root",
                        "unit-lifecycle",
                    )
                    require(props["Type"] in {"exec", "oneshot"}, "unit-type")
                    cap = (
                        5
                        if unit["role"]
                        in {"workload", "cleanup", "observer", "publisher"}
                        else 15
                    )
                    require(
                        linux.systemd_seconds(props["TimeoutStopUSec"]) <= cap,
                        "stop-limit",
                    )
                    require(
                        props["Environment"] == environment_string(p),
                        "closed-unit-environment",
                    )
                for env in s["environment_files"]:
                    exact(
                        env,
                        {"path", "source_path", "sha256", "bytes", "mode"},
                        "env-fields",
                    )
                    require(
                        canonical(env["path"]).parent == scratch / "env", "env-target"
                    )
                    pin_shape(
                        {
                            "path": env["source_path"],
                            "sha256": env["sha256"],
                            "bytes": env["bytes"],
                        }
                    )
                    require(
                        canonical(env["source_path"]).is_relative_to(inputs)
                        and env["mode"] in {0o600, 0o640},
                        "env-source-mode",
                    )
        require(
            all(
                roles.count(x) == 1
                for x in ("observer", "publisher", "cleanup", "watchdog")
            ),
            "control-roles",
        )
        files = p["files"]
        require(isinstance(files, dict) and 1 <= len(files) <= 64, "file-inventory")
        for name, variants in files.items():
            path = canonical(name)
            require(
                path in {Path(x["installed_path"]) for x in p["units"].values()}
                or path.parent == scratch / "env",
                "file-target-scope",
            )
            require(
                isinstance(variants, list) and 1 <= len(variants) <= 8, "file-variants"
            )
            for variant in variants:
                exact(variant, {"source", "mode", "uid", "gid"}, "file-variant")
                source = pin_shape(variant["source"])
                require(
                    canonical(source["path"]).is_relative_to(inputs), "variant-source"
                )
                require(
                    variant["uid"] == variant["gid"] == 0
                    and type(variant["mode"]) is int
                    and variant["mode"] in {0o600, 0o640, 0o644},
                    "file-ownership",
                )
        for unit in p["units"].values():
            require(unit["installed_path"] in files, "unit-file-unregistered")
            for stage in ("before", "after"):
                expected = {
                    "source": unit[stage]["unit"],
                    "mode": 0o644,
                    "uid": 0,
                    "gid": 0,
                }
                require(
                    expected in files[unit["installed_path"]],
                    "unit-variant-unregistered",
                )
        topology = exact(
            p["boot_topology"], {"default_target", "target_paths"}, "boot-topology"
        )
        require(
            re.fullmatch(r"[a-zA-Z0-9_.-]+\.target", topology["default_target"]),
            "default-target",
        )
        require(
            isinstance(topology["target_paths"], dict)
            and len(topology["target_paths"]) <= 8,
            "target-paths",
        )
        for target, chain in topology["target_paths"].items():
            require(
                isinstance(chain, list)
                and 1 <= len(chain) <= 16
                and chain[0] == topology["default_target"]
                and chain[-1] == target
                and len(set(chain)) == len(chain)
                and all(
                    isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9_.-]+\.target", x)
                    for x in chain
                ),
                "target-chain",
            )

        bindings = exact(
            p["bindings"],
            {
                "holder",
                "contender",
                "barrier",
                "dependent",
                "stubborn",
                "support_guard",
                "support_service",
                "timer",
            },
            "case-unit-bindings",
        )
        require(
            len(set(bindings.values())) == 8
            and set(bindings.values()) <= set(p["units"]),
            "distinct-case-units",
        )
        modes = {
            "holder": "lock_holder",
            "contender": "lock_contender",
            "barrier": "sleep",
            "dependent": "fast_receipt",
            "stubborn": "term_tree",
            "support_guard": "support_transaction",
            "support_service": "sleep",
            "timer": None,
        }
        for binding, name in bindings.items():
            require(
                p["units"][name]["role"] == "workload"
                and p["units"][name]["mode"] == modes[binding],
                "case-payload-binding",
            )
        preservation = exact(
            p["preservation"],
            {"policy", "before", "after_path", "verified_champions"},
            "preservation-inputs",
        )
        for key in ("policy", "before"):
            pin = pin_shape(preservation[key])
            require(
                canonical(pin["path"]).is_relative_to(inputs), "preservation-source"
            )
        require(
            canonical(preservation["after_path"])
            == scratch / "external" / "r3-after.json",
            "preservation-after-path",
        )
        require(
            isinstance(preservation["verified_champions"], dict)
            and preservation["verified_champions"]
            and all(
                re.fullmatch(r"sha256-[0-9a-f]{64}", k)
                and isinstance(v, str)
                and HASH.fullmatch(v)
                for k, v in preservation["verified_champions"].items()
            ),
            "verified-champion-proof-map",
        )


def environment_string(p: Mapping[str, Any]) -> str:
    return " ".join(f"{k}={v}" for k, v in environment(p).items())


def environment(p: Mapping[str, Any]) -> dict[str, str]:
    return {
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": p["control_root"],
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }


def validate_unit_text(plan: Plan, name: str, data: bytes) -> None:
    """Reject a pinned dummy definition that could escape through systemd itself."""
    p = plan.value
    require(name in p["units"] and len(data) <= 32768, "definition-scope")
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = lambda optionstr: optionstr
    try:
        parser.read_string(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, configparser.Error) as error:
        raise Refusal("definition-syntax") from error
    require(not parser.defaults(), "definition-defaults")
    timer = name.endswith(".timer")
    require(
        set(parser.sections()) <= {"Unit", "Timer" if timer else "Service", "Install"}
        and "Unit" in parser,
        "definition-sections",
    )
    unit = dict(parser["Unit"])
    require(
        set(unit)
        <= {"Description", "After", "Before", "Requires", "ConditionPathExists"}
        and unit.get("ConditionPathExists") == p["scratch_root"] + "/attempt-armed",
        "volatile-boot-fence",
    )
    role = p["units"][name]["role"]
    for field in ("After", "Before", "Requires"):
        for dependency in unit.get(field, "").split():
            require(
                dependency in {"basic.target", "sysinit.target", "timers.target"}
                or (
                    role == "workload"
                    and dependency in p["units"]
                    and p["units"][dependency]["role"] == "workload"
                ),
                "unit-dependency-escape",
            )
    if "Install" in parser:
        install = dict(parser["Install"])
        require(set(install) <= {"WantedBy"}, "install-directive")
        allowed = {
            Path(x).parent.name.rsplit(".", 1)[0]
            for x in p["units"][name]["boot_links"]
        }
        require(set(install.get("WantedBy", "").split()) == allowed, "install-targets")
    if timer:
        values = dict(parser["Timer"])
        require(
            set(values)
            == {
                "OnBootSec"
                if p["units"][name]["role"] == "watchdog"
                else "OnActiveSec",
                "AccuracySec",
                "RandomizedDelaySec",
                "Unit",
                "Persistent",
            },
            "timer-directives",
        )
        require(
            values["AccuracySec"] == "1s"
            and values["RandomizedDelaySec"] == "0"
            and values["Persistent"] == "false"
            and values["Unit"] in p["units"],
            "timer-contract",
        )
        require(
            0
            < linux.systemd_seconds(
                values["OnBootSec" if role == "watchdog" else "OnActiveSec"]
            )
            < (2**40 if role == "watchdog" else 390),
            "timer-bound",
        )
        target_role = p["units"][values["Unit"]]["role"]
        require(
            target_role
            == ("cleanup" if p["units"][name]["role"] == "watchdog" else "workload"),
            "timer-target-role",
        )
        return
    values = dict(parser["Service"])
    allowed = {
        "Type",
        "User",
        "Group",
        "WorkingDirectory",
        "ExecStart",
        "Restart",
        "RuntimeMaxSec",
        "TimeoutStartSec",
        "TimeoutStopSec",
        "KillMode",
        "SendSIGKILL",
        "PrivateDevices",
        "DevicePolicy",
        "NoNewPrivileges",
        "ProtectSystem",
        "ProtectHome",
        "ReadWritePaths",
        "Environment",
        "EnvironmentFile",
        "CPUQuota",
        "MemoryMax",
        "TasksMax",
        "Nice",
    }
    require(set(values) <= allowed, "service-directive-escape")
    required = {
        "User": "root",
        "Group": "root",
        "Restart": "no",
        "KillMode": "control-group",
        "SendSIGKILL": "yes",
        "PrivateDevices": "yes",
        "DevicePolicy": "closed",
        "NoNewPrivileges": "yes",
        "ProtectSystem": "strict",
        "ProtectHome": "read-only",
        "Nice": "19",
        "CPUQuota": "100%",
    }
    require(all(values.get(k) == v for k, v in required.items()), "service-sandbox")
    require(values.get("Type") in {"exec", "oneshot"}, "service-type")
    require(
        values.get("Environment") == environment_string(p), "definition-environment"
    )
    role = p["units"][name]["role"]
    entry = (
        "strength_freshness_cpu_support_case.py"
        if p["units"][name]["mode"] == "support_transaction"
        else "strength_freshness_cpu_lifecycle.py"
        if role == "workload"
        else "strength_freshness_cpu_qualification.py"
    )
    script = str(Path(p["control_root"]) / "scripts" / entry)
    expected = [
        p["python"]["path"],
        "-s",
        script,
        "--authorization",
        str(Path(p["input_root"]) / (name + ".authorization.json")),
    ]
    require(
        shlex.split(values.get("ExecStart", "")) == expected, "fixed-entrypoint-only"
    )
    require(values.get("WorkingDirectory") == p["scratch_root"], "working-directory")
    rw = shlex.split(values.get("ReadWritePaths", ""))
    writable = {p["scratch_root"]}
    if (
        role in {"observer", "publisher", "cleanup"}
        or name == p["bindings"]["support_guard"]
    ):
        writable.add("/etc/systemd/system")
    require(len(rw) == len(set(rw)) and set(rw) == writable, "write-sandbox")
    require(
        values.get("EnvironmentFile", "")
        in {""}
        | {
            e["path"]
            for side in ("before", "after")
            for e in p["units"][name][side]["environment_files"]
        },
        "definition-env-file",
    )
    require(
        int(values.get("MemoryMax", "0"))
        in range(1, (128 * 2**20 if role == "workload" else 2**31) + 1)
        and int(values.get("TasksMax", "0"))
        in range(1, (8 if role == "workload" else 64) + 1),
        "resource-bounds",
    )
    require(
        linux.systemd_seconds(values.get("TimeoutStopSec", "infinity")) <= 5,
        "definition-stop-bound",
    )
    if values["Type"] == "oneshot":
        require(
            "RuntimeMaxSec" not in values
            and linux.systemd_seconds(values.get("TimeoutStartSec", "infinity"))
            <= (120 if role == "cleanup" else 5),
            "oneshot-start-bound",
        )
    else:
        cap = (
            30
            if role == "workload"
            else {"observer": 570, "publisher": 590, "cleanup": 120}.get(role, 390)
        )
        require(
            0 < linux.systemd_seconds(values.get("RuntimeMaxSec", "infinity")) <= cap
            and linux.systemd_seconds(values.get("TimeoutStartSec", "infinity")) <= 5,
            "service-runtime-bound",
        )


class RawFacts(Protocol):
    def record(self, event: str, data: Any) -> Mapping[str, Any]: ...


class Budget(Protocol):
    def deadline(self, phase: str) -> float: ...
    def as_dict(self) -> dict[str, Any]: ...


class ClosedIO:
    """Capability-scoped wrapper. No caller-supplied shell or wildcard is forwarded."""

    def __init__(
        self,
        plan: Plan,
        anchor: Budget,
        backend: Any,
        facts: RawFacts,
        *,
        purpose: str = "workload",
    ):
        require(
            purpose in {"workload", "observer", "cleanup", "finalizer", "read"},
            "capability-purpose",
        )
        self.plan, self.anchor, self._backend, self.facts = plan, anchor, backend, facts
        a = anchor.as_dict()
        require(
            all(a.get(k) == plan.value[k] for k in ("attempt_id", "nonce", "boot_id"))
            and a.get("plan_sha256") == plan.checksum,
            "anchor-plan-binding",
        )
        self._purpose = purpose
        self._arm: Mapping[str, Any] | None = None
        self._retirement: Mapping[str, Any] | None = None
        self._owned: dict[str, tuple[int, str]] = {}
        self.raw_units: dict[str, dict[str, str]] = {}
        self._prior: dict[int, str] = {}

    def now(self) -> float:
        return self._backend.now()

    def clock(self):
        from scripts.strength_freshness_cpu_lifecycle import Clock

        wall = (
            self._backend.wall_ns()
            if hasattr(self._backend, "wall_ns")
            else time.time_ns()
        )
        return Clock(self._backend.boot_id(), self.now(), wall)

    def _deadline(self, deadline: float, *, mutation: bool = False) -> float:
        phase = {
            "workload": "work",
            "observer": "work" if mutation else "observer",
            "cleanup": "cleanup",
            "finalizer": "publisher",
            "read": "audit",
        }[self._purpose]
        require(
            type(deadline) in (int, float) and math.isfinite(deadline),
            "finite-deadline",
        )
        from scripts.strength_freshness_cpu_lifecycle import Anchor

        original = Anchor.from_dict(self.anchor.as_dict())
        try:
            end = min(deadline, original.effective_deadline(phase, self.clock()))
        except ValueError as error:
            raise Refusal("original-boot-deadline") from error
        require(
            self.now() < end and self._backend.boot_id() == self.plan.value["boot_id"],
            "original-boot-deadline",
        )
        if mutation:
            require(self._purpose != "read", "read-only-capability")
        return end

    def _call(self, event: str, inputs: Any, callback):
        self.facts.record(event + ".intent", inputs)
        try:
            value = callback()
        except BaseException as error:
            self.facts.record(
                event + ".failed",
                {"type": type(error).__name__, "reason": str(error)[:512]},
            )
            raise
        self.facts.record(event + ".raw", value)
        return value

    def acknowledge_arm(self, receipt: Mapping[str, Any]) -> None:
        from scripts import strength_freshness_cpu_lifecycle as life

        require(isinstance(self.facts, life.RawLog), "arm-requires-bound-raw-log")
        assert isinstance(self.facts, life.RawLog)
        life.require_work(self.facts, self.clock(), receipt)
        require(
            self.facts.anchor.as_dict() == self.anchor.as_dict(), "arm-anchor-drift"
        )
        self._arm = dict(receipt)

    def acknowledge_retirement(
        self,
        candidate: Mapping[str, Any],
        observer_started: Mapping[str, Any],
        observer_terminal: Mapping[str, Any],
        cleanup_started: Mapping[str, Any],
        cleanup_terminal: Mapping[str, Any],
    ) -> None:
        from scripts import strength_freshness_cpu_lifecycle as life

        require(
            self._purpose == "finalizer" and isinstance(self.facts, life.RawLog),
            "finalizer-bound-log",
        )
        assert isinstance(self.facts, life.RawLog)
        self.facts.load(candidate, "observer-candidate")
        now = self.clock()
        original = life.Anchor.from_dict(self.anchor.as_dict())
        for role, started, terminal in (
            ("observer", observer_started, observer_terminal),
            ("cleanup", cleanup_started, cleanup_terminal),
        ):
            owner = self.facts.load(started, role + "-started")["owner"]
            life.natural_exit(original, now, owner, terminal, role)
        self._retirement = dict(candidate)

    def _unit_scope(self, name: str, verb: str) -> None:
        p = self.plan.value
        require(name in p["units"], "unregistered-unit")
        role = p["units"][name]["role"]
        if self._purpose in {"workload", "observer", "cleanup"}:
            require(role == "workload", "protected-observer-cleanup-role")
            require(
                verb
                in (
                    {"start", "stop", "enable", "disable"}
                    if self._purpose in {"workload", "observer"}
                    else {"stop", "disable"}
                ),
                "purpose-verb",
            )
            if verb in {"start", "enable"}:
                require(self._arm is not None, "cleanup-not-acknowledged")
                from scripts.strength_freshness_cpu_driver import verify_arm_health

                verify_arm_health(self)
        elif self._purpose == "finalizer":
            require(
                role in {"cleanup", "watchdog"}
                and verb in {"stop", "disable"}
                and self._retirement is not None,
                "finalizer-not-admitted",
            )
        else:
            raise Refusal("read-only-capability")

    def _known_file(self, path: Path, deadline: float, *, absent: bool = False) -> None:
        variants = self.plan.value["files"].get(str(path))
        require(variants is not None, "unregistered-file")
        require(canonical(str(path)) == path, "file-alias")
        try:
            data = self._backend.read(path, 2**20, deadline)
            metadata = self._backend.file_metadata(path, deadline)
        except FileNotFoundError:
            require(absent, "missing-owned-file")
            return
        if str(path) in {
            v["installed_path"] for v in self.plan.value["units"].values()
        }:
            validate_unit_text(self.plan, path.name, data)
        self.facts.record(
            "owned-file.raw",
            {
                "path": str(path),
                "sha256": sha(data),
                "bytes": len(data),
                "metadata": metadata,
            },
        )
        require(
            any(
                sha(data) == v["source"]["sha256"]
                and len(data) == v["source"]["bytes"]
                and metadata == {k: v[k] for k in ("uid", "gid", "mode")}
                for v in variants
            ),
            "unknown-file-variant",
        )

    def command(self, argv: list[str], deadline: float) -> str:
        end = self._deadline(deadline)
        require(
            isinstance(argv, list) and all(isinstance(x, str) for x in argv),
            "command-vector",
        )
        p = self.plan.value
        targets = {
            x for chain in p["boot_topology"]["target_paths"].values() for x in chain
        }
        allowed = argv in (
            ["systemctl", "get-default"],
            ["systemctl", "list-jobs", "--all", "--no-pager", "--output=json"],
        )
        if (
            len(argv) == 6
            and argv[:2] == ["systemctl", "show"]
            and argv[3:5] == ["--all", "--no-pager"]
        ):
            name = argv[2]
            allowed = (
                name in p["units"]
                and argv[5] == "--property=" + ",".join(linux.property_names(name))
            ) or (name in targets and argv[5] == "--property=Wants,Requires")
        if (
            len(argv) == 6
            and argv[:2] == ["systemctl", "show"]
            and argv[2] in p["units"]
            and argv[3:5] == ["--all", "--no-pager"]
        ):
            allowed |= argv[5] in {
                "--property=ExecMainCode,ExecMainExitTimestampMonotonic",
                "--property=NextElapseUSecMonotonic,AccuracyUSec,RandomizedDelayUSec,Unit,LastTriggerUSecMonotonic",
                "--property=TimeoutStartUSec,TimeoutStopUSec",
            }
        if argv == ["systemctl", "daemon-reload"]:
            end = self._deadline(end, mutation=True)
            for path in p["files"]:
                self._known_file(Path(path), end, absent=True)
            allowed = True
        require(allowed, "command-not-allowed")
        raw = self._call(
            "command",
            {"argv": argv, "deadline": end},
            lambda: self._backend.command(argv, end),
        )
        require(
            isinstance(raw, str) and len(raw.encode()) <= 2**20 and self.now() < end,
            "command-output-bound",
        )
        if (
            len(argv) == 6
            and argv[:2] == ["systemctl", "show"]
            and argv[2] in p["units"]
            and argv[5] == "--property=" + ",".join(linux.property_names(argv[2]))
        ):
            self.raw_units[argv[2]] = dict(
                line.split("=", 1) for line in raw.splitlines() if "=" in line
            )
        return raw

    def action(self, verb: str, name: str, deadline: float) -> None:
        end = self._deadline(deadline, mutation=True)
        self._unit_scope(name, verb)
        spec = self.plan.value["units"][name]
        self._known_file(Path(spec["installed_path"]), end)
        self.boot_links(name, end)
        self._call(
            "unit-action",
            {"verb": verb, "unit": name, "deadline": end},
            lambda: (
                self._backend.command(["systemctl", verb, name], end)
                if verb in {"enable", "disable"}
                else self._backend.action(verb, name, end)
            ),
        )
        require(self.now() < end, "action-overrun")

    def reset_failed(self, name: str, deadline: float) -> None:
        end = self._deadline(deadline, mutation=True)
        self._unit_scope(name, "stop")
        unit, _ = DummyReadHost(self).snapshot(name, end, partial=True)
        require(unit.dead, "reset-only-drained-dummy")
        self._known_file(Path(self.plan.value["units"][name]["installed_path"]), end)
        self._call(
            "reset-failed",
            {"unit": name, "terminal": asdict(unit)},
            lambda: self._backend.command(["systemctl", "reset-failed", name], end),
        )

    def kill_sacrificial(
        self, name: str, expected: linux.core.Unit, deadline: float
    ) -> None:
        end = self._deadline(deadline, mutation=True)
        self._unit_scope(name, "stop")
        require(name == self.plan.value["bindings"]["holder"], "sacrificial-fixed-unit")
        unit, _ = DummyReadHost(self).snapshot(name, end)
        require(
            unit == expected and unit.main is not None and unit.invocation_id,
            "sacrificial-owner-drift",
        )
        self._call(
            "sacrificial-kill",
            {"unit": asdict(unit)},
            lambda: self._backend.command(
                ["systemctl", "kill", "--kill-whom=main", "--signal=SIGKILL", name], end
            ),
        )

    def remove_owned_file(self, path: Path, names: list[str], deadline: float) -> None:
        end = self._deadline(deadline, mutation=True)
        require(self._purpose in {"cleanup", "finalizer"}, "remove-purpose")
        p = self.plan.value
        owners = [
            name
            for name, spec in p["units"].items()
            if spec["installed_path"] == str(path)
            or any(
                e["path"] == str(path)
                for side in ("before", "after")
                for e in spec[side]["environment_files"]
            )
        ]
        require(owners and set(owners) <= set(names), "remove-unowned-file")
        for name in owners:
            self._unit_scope(name, "disable")
            require(self.boot_links(name, end) == {}, "remove-linked-unit")
            require(
                not self.members("/system.slice/" + name, end), "remove-live-cgroup"
            )
        jobs = DummyReadHost(self)._jobs(end)
        require(
            not any(v["unit"] in owners for v in jobs.values()), "remove-pending-job"
        )
        self._known_file(path, end, absent=True)
        if not self._backend.exists(path):
            self.facts.record("remove-already-absent", {"path": str(path)})
            return
        data = self._backend.read(path, 2**20, end)
        expected = {
            "sha256": sha(data),
            "bytes": len(data),
            "metadata": self._backend.file_metadata(path, end),
        }
        self._call(
            "remove-file",
            {"path": str(path), **expected},
            lambda: self._backend.unlink_file(path, expected, end),
        )

    def read(self, path: Path, maximum: int, deadline: float) -> bytes:
        end = self._deadline(deadline)
        p = self.plan.value
        roots = [Path(p["input_root"]), Path(p["scratch_root"])]
        allowed = str(path) in p["files"] or str(path) in {
            x["path"] for x in p["source_pins"]
        }
        allowed |= any(path.is_relative_to(root) for root in roots)
        allowed |= str(path) in {"/proc/locks", "/proc/sys/kernel/random/boot_id"}
        require(
            allowed and canonical(str(path)) == path and 0 < maximum <= 4 * 2**20,
            "read-path-scope",
        )
        data = self._backend.read(path, maximum, end)
        self.facts.record(
            "file-read.raw",
            {"path": str(path), "sha256": sha(data), "bytes": len(data)},
        )
        return data

    def exists(self, path: Path) -> bool:
        self.read_scope(path)
        return self._backend.exists(path)

    def read_scope(self, path: Path) -> None:
        p = self.plan.value
        require(
            str(path) in p["files"]
            or path.is_relative_to(Path(p["scratch_root"]))
            or path.is_relative_to(Path(p["input_root"])),
            "path-outside-dummy",
        )

    def file_metadata(self, path: Path, deadline: float) -> dict[str, int]:
        self.read_scope(path)
        value = self._backend.file_metadata(path, self._deadline(deadline))
        self.facts.record("file-metadata.raw", {"path": str(path), **value})
        return value

    def pin(self, pin: Mapping[str, Any], deadline: float) -> None:
        pin_shape(dict(pin))
        self.read_scope(Path(pin["path"]))
        self._backend.pin(pin, self._deadline(deadline))

    def json(self, path: Path, deadline: float, maximum: int = 2**20) -> dict[str, Any]:
        value = json.loads(self.read(path, maximum, deadline))
        require(isinstance(value, dict), "json-object")
        return value

    def members(self, path: str, deadline: float) -> tuple[linux.core.Process, ...]:
        require(
            path in {"/system.slice/" + n for n in self.plan.value["units"]},
            "cgroup-scope",
        )
        end = self._deadline(deadline)
        values = self._backend.members(path, end)
        self.facts.record(
            "cgroup.raw", {"path": path, "members": [asdict(x) for x in values]}
        )
        for value in values:
            self._owned[str(value.pid)] = value.start_ticks, path
        return values

    def process(self, pid: int, deadline: float | None = None) -> dict[str, Any] | None:
        end = self._deadline(deadline if deadline is not None else self.now() + 2)
        require(
            str(pid) in self._owned or pid in self._prior,
            "process-not-observed-in-dummy-cgroup",
        )
        value = self._backend.process(pid, end)
        self.facts.record("process.raw", value)
        if value is not None and str(pid) in self._owned:
            ticks, group = self._owned[str(pid)]
            require(
                value["start_ticks"] == ticks
                and (
                    value["cgroup"] == "0::" + group
                    or value["cgroup"].startswith("0::" + group + "/")
                ),
                "process-reuse-or-cgroup",
            )
        return value

    def remember_prior_process(
        self, unit_name: str, process: linux.core.Process
    ) -> None:
        """Read-only old-owner check, after the caller validates its pinned receipt.

        Reused PIDs are returned truthfully; they never grant signal authority.
        """
        p = self.plan.value
        require(
            unit_name in p["units"]
            and p["units"][unit_name]["role"] == "workload"
            and type(process.pid) is int
            and process.pid > 0
            and process.start_ticks > 0,
            "prior-workload-owner",
        )
        self._prior[process.pid] = unit_name

    def process_origin(self, pid: int, deadline: float) -> dict[str, str]:
        end = self._deadline(deadline)
        before = self.process(pid, end)
        require(before is not None, "origin-process-gone")
        value = self._backend.process_origin(pid, end)
        self.facts.record("process-origin.raw", {"pid": pid, **value})
        require(self.process(pid, end) == before, "origin-process-race")
        return value

    def boot_links(self, name: str, deadline: float) -> dict[str, str]:
        require(name in self.plan.value["units"], "boot-unit-scope")
        value = self._backend.boot_links(name, self._deadline(deadline))
        self.facts.record("boot-links.raw", {"unit": name, "links": value})
        expected = self.plan.value["units"][name]["boot_links"]
        require(
            set(value) <= set(expected) and all(value[k] == expected[k] for k in value),
            "foreign-boot-edge",
        )
        return value

    def sleep(self, seconds: float) -> None:
        end = self._deadline(self.now() + seconds + 0.001)
        require(0 <= seconds <= 0.2 and self.now() + seconds < end, "bounded-poll")
        self._backend.sleep(seconds)

    def atomic(
        self, path: Path, data: bytes, *, overwrite: bool = False, mode: int = 0o600
    ) -> None:
        end = self._deadline(self.anchor.deadline("work"), mutation=True)
        p = self.plan.value
        if str(path) in p["files"]:
            owners = [
                name
                for name, spec in p["units"].items()
                if spec["installed_path"] == str(path)
                or any(
                    e["path"] == str(path)
                    for side in ("before", "after")
                    for e in spec[side]["environment_files"]
                )
            ]
            require(
                owners and all(p["units"][n]["role"] == "workload" for n in owners),
                "file-role-protected",
            )
            self._known_file(path, end, absent=not overwrite)
            if str(path) in {v["installed_path"] for v in p["units"].values()}:
                validate_unit_text(self.plan, path.name, data)
            require(
                any(
                    v["source"]["sha256"] == sha(data)
                    and v["source"]["bytes"] == len(data)
                    and v["mode"] == mode
                    for v in p["files"][str(path)]
                ),
                "write-unregistered-variant",
            )
        else:
            require(
                path.is_relative_to(Path(p["scratch_root"]) / "synthetic-support-state")
                and not path.is_symlink()
                and mode == 0o600
                and len(data) <= 2**20,
                "journal-path-scope",
            )
        self._call(
            "file-write",
            {"path": str(path), "sha256": sha(data), "bytes": len(data), "mode": mode},
            lambda: self._backend.atomic(path, data, overwrite=overwrite, mode=mode),
        )


class DummyReadHost:
    """Uses unchanged strict _unit/_jobs parsers, without LinuxHost construction."""

    def __init__(self, io: ClosedIO):
        self.io = io
        p = io.plan.value
        self.manifest = {"units": p["units"], "boot_topology": p["boot_topology"]}
        self._exit_pids: dict[str, int] = {}

    def _jobs(self, deadline: float):
        return linux.LinuxHost._jobs(self, deadline)  # type: ignore[arg-type]

    def _unit(self, name, jobs, deadline, *, allow_known_partial=False):
        require(name in self.manifest["units"], "dummy-unit-only")
        return linux.LinuxHost._unit(
            cast(Any, self),
            name,
            jobs,
            deadline,
            allow_known_partial=allow_known_partial,
        )  # type: ignore[arg-type]

    def snapshot(self, name: str, deadline: float, *, partial: bool = False):
        for _ in range(3):
            try:
                unit, metadata = self._unit(
                    name, self._jobs(deadline), deadline, allow_known_partial=partial
                )
                self.io.facts.record(
                    "unit.parsed", {"unit": asdict(unit), "metadata": metadata}
                )
                return unit, metadata
            except linux.ObservationChanged:
                self.io.sleep(0.01)
        raise Refusal("dummy-observation-changing")


def verify_sources(
    plan: Plan, io: linux.LinuxIO, facts: RawFacts, deadline: float
) -> None:
    """Verify actual loaded control origins before any dummy capability is used."""
    p = plan.value
    pins = {x["path"]: x for x in p["source_pins"]}
    from scripts import strength_freshness_cpu_lifecycle as lifecycle
    from scripts import qualify_cloud_gpu_window as system_host

    modules = (linux, support, linux.core, lifecycle, system_host)
    actual = {
        str(Path(__file__).resolve()),
        *(str(Path(cast(str, m.__file__)).resolve()) for m in modules),
    }
    require(actual <= set(pins), "executing-module-unpinned")
    require(
        all(Path(x).is_relative_to(Path(p["control_root"])) for x in actual),
        "actual-control-origin",
    )
    facts.record(
        "source-origins.raw",
        {"origins": sorted(actual), "control_root": p["control_root"]},
    )
    for path in sorted(actual):
        io.pin(pins[path], deadline)
    io.pin(p["python"], deadline)
    original_anchor = getattr(facts, "anchor", None)
    require(
        original_anchor is not None
        and original_anchor.as_dict()["plan_sha256"] == plan.checksum,
        "source-anchor-binding",
    )
    assert original_anchor is not None
    runtime = p["runtime_inputs"]
    for key in (
        "source_marker",
        "source_manifest",
        "pyvenv",
        "native_wrapper",
        "native_binary",
        "qualification",
    ):
        io.pin(runtime[key], deadline)
    require(
        io.read(Path(runtime["source_marker"]["path"]), 256, deadline).decode().strip()
        == runtime["source_commit"],
        "runtime-marker-identity",
    )
    facts.record(
        "runtime-input-pins",
        {
            "runtime": runtime,
            "scope": "byte admission only; no model load, inference, CUDA or target execution qualification",
        },
    )
    for name, spec in p["units"].items():
        for side in ("before", "after"):
            unit_pin = spec[side]["unit"]
            io.pin(unit_pin, deadline)
            unit_bytes = io.read(Path(unit_pin["path"]), 32768, deadline)
            validate_unit_text(plan, name, unit_bytes)
            if spec["role"] == "watchdog":
                timer = configparser.ConfigParser(interpolation=None)
                timer.read_string(unit_bytes.decode())
                require(
                    abs(
                        linux.systemd_seconds(timer["Timer"]["OnBootSec"])
                        - original_anchor.deadline("work")
                    )
                    <= 0.000001,
                    "watchdog-definition-original-deadline",
                )
            for env in spec[side]["environment_files"]:
                pin = {
                    "path": env["source_path"],
                    "sha256": env["sha256"],
                    "bytes": env["bytes"],
                }
                io.pin(pin, deadline)
                require(
                    io.read(Path(pin["path"]), 4096, deadline)
                    in (b"CPUQUAL_PHASE=before\n", b"CPUQUAL_PHASE=after\n"),
                    "inert-environment-file-only",
                )
    require(
        io.boot_id() == p["boot_id"] and io.now() < deadline, "source-boot-deadline"
    )


class CaseRunner:
    """Actual selected Linux primitives; synthetic support authority is separate.

    Every method records raw facts before enforcing its semantic gate. The
    aggregate driver cannot label an unexecuted method as a passed case.
    """

    def __init__(self, io: ClosedIO):
        require(io._purpose in {"workload", "observer"}, "case-workload-capability")
        self.io, self.host = io, DummyReadHost(io)

    def _end(self, deadline: float) -> float:
        return self.io._deadline(deadline)

    def typed_properties(self, names: list[str], deadline: float) -> dict[str, Any]:
        end = self._end(deadline)
        p = self.io.plan.value
        require(len(names) == 3 and len(set(names)) == 3, "typed-case-inventory")
        types = set()
        result = {}
        for name in names:
            unit, metadata = self.host.snapshot(name, end)
            stage = metadata["stage"]
            types.add(
                "timer"
                if name.endswith(".timer")
                else p["units"][name][stage]["properties"]["Type"]
            )
            result[name] = {"unit": asdict(unit), "metadata": metadata}
        require(types == {"exec", "oneshot", "timer"}, "typed-service-timer-coverage")
        return result

    def pid_cgroup(self, name: str, deadline: float) -> dict[str, Any]:
        end = self._end(deadline)
        unit, _ = self.host.snapshot(name, end)
        require(
            unit.main is not None and unit.main in unit.members and unit.invocation_id,
            "live-dummy-owner",
        )
        observed = []
        for member in unit.members:
            value = self.io.process(member.pid, end)
            origin = self.io.process_origin(member.pid, end)
            observed.append({"process": value, "origin": origin})
        return {"unit": asdict(unit), "processes": observed}

    def wait_drained(self, name: str, deadline: float) -> dict[str, Any]:
        end = self._end(deadline)
        while self.io.now() < end:
            unit, metadata = self.host.snapshot(name, end)
            if unit.dead:
                return {"unit": asdict(unit), "metadata": metadata}
            self.io.sleep(min(0.05, max(0, end - self.io.now() - 0.002)))
        raise Refusal("dummy-drain-deadline")

    def cancel_pending(
        self, dependent: str, barrier: str, marker: Path, deadline: float
    ) -> dict[str, Any]:
        end = self._end(deadline)
        p = self.io.plan.value
        require(
            dependent != barrier
            and all(p["units"][n]["role"] == "workload" for n in (dependent, barrier)),
            "pending-case-role",
        )
        before, _ = self.host.snapshot(dependent, end)
        require(
            before.main is None
            and before.job is not None
            and before.job["kind"] == "start",
            "real-queued-start-required",
        )
        self.io.action("stop", dependent, end)
        cancelled = self.wait_drained(dependent, end)
        self.io.action("stop", barrier, end)
        self.wait_drained(barrier, end)
        # Recheck after the dependency is gone; no manufactured grace delay.
        after, _ = self.host.snapshot(dependent, end)
        marker_exists = self.io.exists(marker)
        self.io.facts.record(
            "pending-start.marker.raw", {"path": str(marker), "exists": marker_exists}
        )
        require(after.dead and not marker_exists, "late-pending-payload")
        return {
            "before": asdict(before),
            "cancelled": cancelled,
            "after": asdict(after),
            "marker_absent": True,
        }

    def bounded_drain(
        self, name: str, deadline: float, *, expected_failure: bool
    ) -> dict[str, Any]:
        end = self._end(deadline)
        before, _ = self.host.snapshot(name, end)
        require(before.main is not None and before.members, "drain-requires-live-tree")
        self.io.action("stop", name, end)
        result = self.wait_drained(name, end)
        failed = (
            result["unit"]["result"] != "success" or result["unit"]["exit_code"] != 0
        )
        require(failed == expected_failure, "drain-fault-expectation")
        return {
            "before": asdict(before),
            "after": result,
            "expected_fault": expected_failure,
        }

    def persistent_boot_edges(self, name: str, deadline: float) -> dict[str, Any]:
        end = self._end(deadline)
        self.io.action("enable", name, end)
        support.verify_boot_edges(self.host, name, True, end)
        enabled = self.io.boot_links(name, end)
        self.io.action("disable", name, end)
        support.verify_boot_edges(self.host, name, False, end)
        return {
            "enabled_links": enabled,
            "disabled_links": self.io.boot_links(name, end),
            "real_reboot": False,
        }

    def support_transition(
        self, host: Any, stage: str, deadline: float, *, recovery: bool = False
    ) -> dict[str, Any] | None:
        require(
            isinstance(host, DummySupportHost) and host.io is self.io,
            "dummy-support-facade-required",
        )
        end = self._end(deadline)
        self.io.facts.record(
            "support.scope",
            {
                "authority": "synthetic-dummy-only",
                "linux_facts": "actual-bounded-io",
                "stage": stage,
            },
        )
        if recovery:
            return support.recover_transition(host, end, pre_admission=True)
        return support.transition(host, stage, end)


class DummySupportHost(DummyReadHost):
    """Narrow helper facade; no production methods or production root fields.

    Synthetic state/anchor must be prospectively prepared by the case fixture.
    Kernel lease proof still comes from the unchanged LinuxHost reader and is
    confined to this attempt's /run lock. This is never a production admission.
    """

    def __init__(self, io: ClosedIO, synthetic_plan: dict[str, Any], lease_fd: int):
        super().__init__(io)
        p = io.plan.value
        scratch = Path(p["scratch_root"])
        require(
            synthetic_plan["run_root"] == str(scratch / "synthetic-run")
            and synthetic_plan["state_root"] == str(scratch / "synthetic-support-state")
            and synthetic_plan["exclusion_path"] == str(scratch / "qualification.lock"),
            "synthetic-roots-only",
        )
        names = {v["name"] for v in synthetic_plan["units"].values()}
        require(
            names <= set(p["units"])
            and set(synthetic_plan["support_transition"]["before"])
            == set(synthetic_plan["support_transition"]["after"]),
            "synthetic-inventory",
        )
        supports = set(synthetic_plan["support_transition"]["before"])
        require(
            supports and all(p["units"][n]["role"] == "workload" for n in supports),
            "synthetic-support-role",
        )
        require(type(lease_fd) is int and lease_fd >= 0, "dummy-lease-descriptor")
        self.plan = json.loads(encode(synthetic_plan))
        self.state, self.root = (
            scratch / "synthetic-support-state",
            scratch / "synthetic-run",
        )
        self.execute, self.lease_fd = True, lease_fd
        self.manifest_sha256 = io.plan.checksum

    def _enable(self, name: str, enabled: bool, deadline: float) -> None:
        self.io.action("enable" if enabled else "disable", name, deadline)

    def _authority(self, deadline: float):
        self.io._deadline(deadline)
        state = self.io.json(self.state / "state.json", deadline)
        phase = state.get("dummy_authority")
        require(phase in {"r3", "r4"}, "synthetic-authority-phase")
        assert isinstance(phase, str)
        return linux.core.Authority(
            phase, self.plan["freshness_plan_sha256"], sha(encode(state))
        )

    def clock(self):
        import time

        return linux.core.Clock(
            self.io._backend.boot_id(), self.io.now(), time.time_ns()
        )

    def _lease(self, deadline: float):
        return linux.LinuxHost._lease(self, deadline)  # type: ignore[arg-type]


class Driver:
    """Bounded case ledger; publishes a candidate only, never an aggregate pass."""

    def __init__(self, runner: CaseRunner):
        self.runner, self.results = runner, {}

    def case(
        self, name: str, operation: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        allowed = {
            "typed-properties": "typed_properties",
            "pid-cgroup-lease": "pid_cgroup",
            "pending-start-job": "cancel_pending",
            "bounded-drain": "bounded_drain",
            "support-partial-transaction": "support_transition",
            "persistent-boot-edges": "persistent_boot_edges",
        }
        require(
            name not in self.results and allowed.get(name) == operation, "case-dispatch"
        )
        io = self.runner.io
        end = io._deadline(io.anchor.deadline("work"))
        require("deadline" not in arguments, "caller-deadline-override")
        io.facts.record(
            "case.started",
            {"case": name, "operation": operation, "monotonic": io.now()},
        )
        try:
            result = getattr(self.runner, operation)(**arguments, deadline=end)
        except BaseException as error:
            record = {
                "case": name,
                "status": "failed",
                "error": type(error).__name__,
                "reason": str(error)[:512],
            }
            io.facts.record("case.result", record)
            self.results[name] = record
            raise
        record = {"case": name, "status": "observed", "result": result}
        io.facts.record("case.result", record)
        self.results[name] = record
        return record

    def candidate(self) -> dict[str, Any]:
        missing = sorted(set(CASES) - set(self.results))
        return {
            "format": RESULT,
            "status": "incomplete" if missing else "candidate-only",
            "plan_sha256": self.runner.io.plan.checksum,
            "cases": self.results,
            "missing_cases": missing,
            **CLAIMS,
        }


def validate_actor_bounds(
    role: str,
    unit: linux.core.Unit,
    properties: Mapping[str, str],
    start_timeout: str,
    anchor: Budget,
) -> None:
    """Systemd's independent lifetime must fit the *original* role endpoint."""
    require(role in {"observer", "publisher", "cleanup", "workload"}, "actor-role")
    stop = linux.systemd_seconds(properties["TimeoutStopUSec"])
    if properties["Type"] == "oneshot":
        duration = linux.systemd_seconds(start_timeout)
    else:
        duration = linux.systemd_seconds(properties["RuntimeMaxUSec"])
    terminal_phase = {
        "observer": "observer_terminal",
        "publisher": "publisher_terminal",
        "cleanup": "cleanup",
        "workload": "cleanup",
    }[role]
    require(
        0 <= stop <= 5
        and 0 < duration < float("inf")
        and unit.entered_monotonic >= anchor.as_dict()["started_monotonic"]
        and unit.entered_monotonic + duration + stop <= anchor.deadline(terminal_phase),
        "actor-original-lifetime-cap",
    )


@dataclass
class AuthorizedContext:
    plan: Plan
    anchor: Any
    log: Any
    io: ClosedIO
    host: DummyReadHost
    unit: linux.core.Unit
    authorization: dict[str, Any]

    def clock(self):
        return self.io.clock()


def authorized_context(
    path: Path,
    *,
    expected_role: str | None = None,
    expected_mode: str | None = None,
    child: bool = False,
) -> AuthorizedContext:
    """Explicit protected dummy authorization only; no installer or production gate."""
    from scripts import strength_freshness_cpu_lifecycle as life
    from scripts.strength_freshness_cpu_driver import EvidenceLog, TargetIO, find_one

    require(sys.platform == "linux" and os.geteuid() == 0, "root-linux-dummy-only")
    parent = path.parent.lstat()
    require(
        path.parent.resolve() == path.parent
        and parent.st_uid == 0
        and (parent.st_mode & 0o777) == 0o700,
        "protected-authorization-parent",
    )
    data = life._protected_file(path, 0, private=True)
    auth = exact(
        json.loads(data),
        {
            "format",
            "schema_version",
            "plan_path",
            "plan_sha256",
            "anchor_path",
            "anchor_sha256",
            "source_pins",
            "role",
            "unit",
            "mode",
            "payload",
        },
        "authorization-fields",
    )
    require(
        auth["format"] == "strength-freshness-cpu-authorization-v1"
        and auth["schema_version"] == 1,
        "authorization-format",
    )
    plan_data = life._protected_file(Path(auth["plan_path"]), 0)
    plan = Plan.parse(plan_data, auth["plan_sha256"])
    anchor_data = life._protected_file(Path(auth["anchor_path"]), 0)
    require(sha(anchor_data) == auth["anchor_sha256"], "anchor-source-pin")
    anchor = life.Anchor.from_dict(json.loads(anchor_data))
    p = plan.value
    require(
        str(path.parent) == p["input_root"]
        and path.name == auth["unit"] + ".authorization.json"
        and Path(auth["plan_path"]).parent == path.parent
        and Path(auth["anchor_path"]).parent == path.parent,
        "authorization-locations",
    )
    require(auth["unit"] in p["units"], "authorization-unit")
    spec = p["units"][auth["unit"]]
    require(
        auth["role"] == spec["role"]
        and auth["mode"] == spec["mode"]
        and auth["payload"] == spec["payload"]
        and auth["source_pins"] == p["source_pins"],
        "authorization-plan-binding",
    )
    require(
        auth["role"] in {"observer", "publisher", "cleanup", "workload"}
        and (expected_role is None or auth["role"] == expected_role)
        and (expected_mode is None or auth["mode"] == expected_mode),
        "authorization-role",
    )
    for key, value in environment(p).items():
        require(os.environ.get(key) == value, "actual-closed-environment-" + key)
    require(
        all(
            not os.environ.get(key)
            for key in (
                "PYTHONHOME",
                "LD_PRELOAD",
                "LD_LIBRARY_PATH",
                "DYLD_INSERT_LIBRARIES",
            )
        ),
        "actual-import-overrides",
    )
    require(
        str(Path(sys.executable).resolve()) == p["python"]["resolved_path"],
        "actual-interpreter-origin",
    )
    backend = TargetIO()
    log = EvidenceLog(Path(p["scratch_root"]) / "evidence", anchor)
    purpose = {
        "observer": "observer",
        "workload": "workload",
        "cleanup": "cleanup",
        "publisher": "finalizer",
    }[auth["role"]]
    io = ClosedIO(plan, anchor, backend, log, purpose=purpose)
    end = io._deadline(io.now() + 10)
    verify_sources(plan, backend, log, end)
    for pin in p["source_pins"]:
        backend.pin(pin, end)
    host = DummyReadHost(io)
    unit, _ = host.snapshot(auth["unit"], end)
    process = io.process(os.getpid(), end)
    require(
        process is not None
        and unit.main is not None
        and (
            (
                not child
                and unit.main
                == linux.core.Process(process["pid"], process["start_ticks"])
            )
            or (
                child
                and auth["role"] == "observer"
                and unit.main.pid == os.getppid()
                and process["ppid"] == unit.main.pid
                and process["start_ticks"] >= unit.main.start_ticks
            )
        )
        and unit.invocation_id == os.environ.get("INVOCATION_ID")
        and unit.invocation_id
        and unit.active in {"activating", "active"},
        "actual-executing-dummy-unit",
    )
    raw_timeout = io.command(
        [
            "systemctl",
            "show",
            auth["unit"],
            "--all",
            "--no-pager",
            "--property=TimeoutStartUSec,TimeoutStopUSec",
        ],
        end,
    )
    timeout_fields = dict(
        line.split("=", 1) for line in raw_timeout.splitlines() if "=" in line
    )
    require(
        set(timeout_fields) == {"TimeoutStartUSec", "TimeoutStopUSec"},
        "actual-actor-timeouts",
    )
    validate_actor_bounds(
        auth["role"],
        unit,
        io.raw_units[auth["unit"]],
        timeout_fields["TimeoutStartUSec"],
        anchor,
    )
    if auth["role"] in {"observer", "workload"}:
        armed = find_one(log, "cleanup-armed")
        io.acknowledge_arm(armed)
    return AuthorizedContext(plan, anchor, log, io, host, unit, auth)


def readonly_context(
    plan_path: Path, expected: str, anchor_path: Path, anchor_sha256: str
) -> AuthorizedContext:
    from scripts import strength_freshness_cpu_lifecycle as life
    from scripts.strength_freshness_cpu_driver import EvidenceLog, TargetIO

    require(sys.platform == "linux" and os.geteuid() == 0, "root-linux-read-audit")
    data = life._protected_file(plan_path, 0)
    plan = Plan.parse(data, expected)
    anchor_data = life._protected_file(anchor_path, 0)
    require(sha(anchor_data) == anchor_sha256, "explicit-anchor-pin")
    anchor = life.Anchor.from_dict(json.loads(anchor_data))
    require(
        plan_path.parent == anchor_path.parent == Path(plan.value["input_root"]),
        "read-audit-input-location",
    )
    backend = TargetIO()
    log = EvidenceLog(Path(plan.value["scratch_root"]) / "evidence", anchor)
    io = ClosedIO(plan, anchor, backend, log, purpose="read")
    verify_sources(
        plan,
        backend,
        log,
        min(io.now() + 10, anchor.effective_deadline("audit", io.clock())),
    )
    host = DummyReadHost(io)
    name = next(n for n, v in plan.value["units"].items() if v["role"] == "publisher")
    unit, _ = host.snapshot(
        name, min(io.now() + 5, anchor.effective_deadline("audit", io.clock()))
    )
    return AuthorizedContext(
        plan, anchor, log, io, host, unit, {"role": "external-read-only-auditor"}
    )


def validate_file(path: Path, expected: str) -> dict[str, Any]:
    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= 2**20,
        "plan-file",
    )
    plan = Plan.parse(path.read_bytes(), expected)
    return {
        "format": RESULT,
        "status": "plan-validated-no-target-execution",
        "plan_sha256": plan.checksum,
        "attempt_id": plan.value["attempt_id"],
        **CLAIMS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", type=Path)
    mode.add_argument("--authorization", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--fixture-child", action="store_true")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--arm-read-only", action="store_true")
    stage.add_argument("--audit-read-only", action="store_true")
    parser.add_argument("--anchor", type=Path)
    parser.add_argument("--anchor-sha256")
    args = parser.parse_args()
    if args.plan is not None:
        require(not args.fixture_child, "fixture-needs-protected-authorization")
        require(args.sha256 is not None, "explicit-validation-sha")
        if args.arm_read_only or args.audit_read_only:
            require(
                args.anchor is not None and args.anchor_sha256 is not None,
                "read-stage-anchor-pin",
            )
            assert args.anchor is not None and args.anchor_sha256 is not None
            context = readonly_context(
                args.plan, args.sha256, args.anchor, args.anchor_sha256
            )
            from scripts import strength_freshness_cpu_driver as driver

            if args.arm_read_only:
                print(
                    json.dumps(
                        driver.arm_receipt(
                            context.plan,
                            context.anchor,
                            context.io._backend,
                            context.log,
                        ),
                        sort_keys=True,
                    )
                )
            else:
                print(json.dumps(driver.external_audit(context), sort_keys=True))
        else:
            require(
                args.anchor is None and args.anchor_sha256 is None, "unexpected-anchor"
            )
            print(json.dumps(validate_file(args.plan, args.sha256), sort_keys=True))
    else:
        require(
            args.sha256 is None
            and not args.arm_read_only
            and not args.audit_read_only
            and args.anchor is None
            and args.anchor_sha256 is None,
            "authorization-cli-scope",
        )
        if args.fixture_child:
            from scripts.strength_freshness_cpu_fixture import run_worker

            print(
                json.dumps(
                    run_worker(
                        authorized_context(
                            args.authorization, expected_role="observer", child=True
                        )
                    ),
                    sort_keys=True,
                )
            )
        else:
            from scripts.strength_freshness_cpu_driver import run_role

            run_role(authorized_context(args.authorization))


if __name__ == "__main__":
    main()
