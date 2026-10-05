"""Strict private observed-window registration, with no IO or execution grant.

External approval authenticates these bytes only. Writer/source/access/runtime
qualification and the future producer-to-consumer contract remain independent.
No historical birth constructor, collector, CLI or preservation result is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import PurePosixPath
import re
from typing import Any

from scripts import strength_freshness_cpu_collect_identity as identity
from scripts import strength_freshness_cpu_learner_window as append
from scripts import strength_freshness_cpu_preservation as preservation
from scripts import strength_freshness_cpu_readonly as readonly

FORMAT = "strength-preservation-observed-window-registration-v1"
POLICY = "strength-preservation-observed-window-policy-v1"
CONTRACT = "learner-append-observed-window-v1"
PUBLICATION_CONTRACT = (
    "618c4aadd29f642b703cbe4aac188362c21833c3673f85c11f40d5a619f6dc4c"
)
RUNTIME_SOURCE = "fac6f7f3de76d3e2779baf1d2669177f139b559b9b871b015812522a5f37eebc"
ACTOR_SOURCE = "8cce16d9d5b233ba209a95b275d1a9ccec6e3b0a1614324281400f435bc9bb5f"
COORDINATOR_SOURCE = "93e7d7f2495bd8c1fa023981fa316caef985576cd54e27d13903f3f40f6e29cc"
MAX_BYTES = 2**20
ROLES = frozenset({"controller", "coordinator", "monitor", *identity.WORKERS})
COHORTS = frozenset(f"actor-gpu-{i}-cohort-{j}" for i in range(1, 7) for j in range(4))
UID_FIELDS = frozenset({"real", "effective", "saved", "filesystem"})
EXTRA_FIELDS = frozenset(
    {
        "format",
        "schema_version",
        "contract",
        "kernel_context",
        "learner_writer",
        "publication_writers",
        "legacy_policy_reference_pin",
    }
)
POLICY_FIELDS = frozenset(
    {
        "format",
        "contract",
        "static",
        "workers",
        "cohorts",
        "gpu_uuids",
        "gpu_roles",
        "support",
        "expected_processes",
        "expected_monitors",
        "recipe",
        "maximum_age_ns",
        "maximum_seconds",
        "maximum_support_seconds",
    }
)


class ObservedRegistrationRefusal(ValueError):
    """Fixed reason code only; no private registration or writer content."""


def require(ok: object, code: str) -> None:
    if not ok:
        raise ObservedRegistrationRefusal(code)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encoded(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def fields(value, keys, reason):
    require(isinstance(value, dict) and set(value) == set(keys), reason)
    return value


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def same(a, b):
    return encoded(a) == encoded(b)


def path(value):
    require(
        isinstance(value, str)
        and value.startswith("/")
        and str(PurePosixPath(value)) == value
        and ".." not in PurePosixPath(value).parts,
        "observed-path",
    )


def bounded_pin(value, scope):
    fields(value, {"path", "sha256", "bytes"}, "observed-pin-fields")
    path(value["path"])
    require(
        append.digest(value["sha256"])
        and integer(value["bytes"], 1)
        and value["bytes"] <= MAX_BYTES,
        "observed-pin-values",
    )
    matches = [entry for entry in scope.files.values() if entry.path == value["path"]]
    require(
        len(matches) == 1 and value["bytes"] <= matches[0].maximum_bytes,
        "observed-pin-outside-registered-scope",
    )


def process(value, cgroup):
    fields(value, preservation.PROCESS_FIELDS, "observed-process-fields")
    require(
        integer(value["pid"], 1)
        and integer(value["start_ticks"], 1)
        and integer(value["restarts"])
        and value["cgroup"] == cgroup
        and isinstance(value["invocation_id"], str)
        and re.fullmatch("[0-9a-f]{32}", value["invocation_id"]) is not None
        and append.digest(value["origin_sha256"]),
        "observed-process-values",
    )


def policy(value, registration):
    fields(value, POLICY_FIELDS, "observed-policy-fields")
    require(
        value["format"] == POLICY and value["contract"] == CONTRACT,
        "observed-policy-format",
    )
    static = fields(
        value["static"], preservation.STATIC_FIELDS, "observed-static-fields"
    )
    require(
        static["source_commit"] == append.R3_COMMIT
        and integer(static["continuation_started_ns"], 1)
        and all(
            isinstance(static[k], str) and 0 < len(static[k]) <= 256
            for k in ("run_id", "generation_family", "runtime_name")
        )
        and static["runtime_cgroup"] == "/system.slice/" + static["runtime_name"]
        and all(
            append.digest(static[k])
            for k in preservation.STATIC_FIELDS
            if k.endswith("sha256")
        ),
        "observed-static-values",
    )
    require(
        isinstance(value["workers"], list)
        and len(value["workers"]) == 9
        and set(value["workers"]) == identity.WORKERS
        and isinstance(value["cohorts"], list)
        and len(value["cohorts"]) == 24
        and set(value["cohorts"]) == COHORTS,
        "observed-population",
    )
    roles = {"controller", "coordinator", *identity.WORKERS}
    require(set(value["expected_processes"]) == roles, "observed-process-roster")
    for row in value["expected_processes"].values():
        process(row, static["runtime_cgroup"])
    support = value["support"]
    require(
        isinstance(support, dict)
        and set(support) == set(registration["units"]) - {static["runtime_name"]},
        "observed-support-roster",
    )
    for name, row in support.items():
        fields(row, preservation.SUPPORT_STATIC, "observed-support-fields")
        require(
            row["kind"] == registration["units"][name]["kind"]
            and type(row["enabled"]) is bool
            and all(
                append.digest(row[k])
                for k in (
                    "definition_sha256",
                    "environment_sha256",
                    "boot_links_sha256",
                )
            ),
            "observed-support-values",
        )
    monitors = value["expected_monitors"]
    require(
        set(monitors)
        == {name for name, row in support.items() if row["kind"] == "long_running"},
        "observed-monitor-roster",
    )
    for name, row in monitors.items():
        process(row, "/system.slice/" + name)
    require(
        len(
            {
                p["pid"]
                for p in [*value["expected_processes"].values(), *monitors.values()]
            }
        )
        == 12,
        "observed-owner-pid-alias",
    )
    gpu = value["gpu_uuids"]
    gpu_roles = value["gpu_roles"]
    require(
        isinstance(gpu, list)
        and len(gpu) == 8
        and len(set(gpu)) == 8
        and all(isinstance(x, str) and 0 < len(x) <= 128 for x in gpu)
        and set(gpu_roles) == set(gpu)
        and set(gpu_roles.values())
        == {"learner", "arena-promotion", *[f"actor-gpu-{i}" for i in range(1, 7)]},
        "observed-gpu-placement",
    )
    require(
        integer(value["maximum_age_ns"], 1)
        and value["maximum_age_ns"] <= 120 * 10**9
        and append.finite(value["maximum_seconds"])
        and 0 < value["maximum_seconds"] <= 600
        and append.finite(value["maximum_support_seconds"])
        and 0 < value["maximum_support_seconds"] <= 1800,
        "observed-policy-limits",
    )
    append._recipe(value["recipe"])


def kernel_context(value):
    fields(
        value,
        {"boot_id", "clock_ticks_per_second", "namespace_expectations", "credentials"},
        "observed-kernel-context-fields",
    )
    require(
        isinstance(value["boot_id"], str)
        and 0 < len(value["boot_id"]) <= 64
        and integer(value["clock_ticks_per_second"], 1),
        "observed-kernel-context-values",
    )
    ns = fields(
        value["namespace_expectations"],
        {"self", "pid1", "owners"},
        "observed-namespace-fields",
    )
    require(set(ns["owners"]) == ROLES, "observed-namespace-owner-roster")
    for pair in (ns["self"], ns["pid1"], *ns["owners"].values()):
        fields(pair, {"pid", "time"}, "observed-namespace-pair")
        require(
            all(integer(v, 1) for v in pair.values()) and same(pair, ns["self"]),
            "observed-unsupported-namespace",
        )
    credentials = value["credentials"]
    require(set(credentials) == ROLES, "observed-credential-roster")
    for vector in credentials.values():
        fields(vector, UID_FIELDS, "observed-uid-fields")
        require(
            all(integer(v) and v <= 2**32 - 1 for v in vector.values())
            and len(set(vector.values())) == 1,
            "observed-r3-uid-consistency",
        )


def writer(value, registration, scope, evidence):
    fields(value, {"binding", "evidence_pin"}, "observed-writer-fields")
    bounded_pin(value["evidence_pin"], scope)
    expected = {
        "binding": value["binding"],
        "evidence_sha256": value["evidence_pin"]["sha256"],
        "evidence_bytes": value["evidence_pin"]["bytes"],
    }
    b = append._writer(expected, evidence, [250000])
    p = registration["policy"]
    k = registration["kernel_context"]
    owner = b["owner_key"]
    actual = p["expected_processes"]["learner"]
    require(
        registration["keys"]["metrics"] == b["metrics_key"] == "metrics"
        and scope.tails["metrics"].path == b["metrics_path"]
        and b["run_id"] == p["static"]["run_id"]
        and b["recipe_sha256"] == sha(append.encoded(p["recipe"]))
        and b["native_sha256"] == p["static"]["native_sha256"]
        and b["command_sha256"] == registration["origins"]["learner"]["argv_sha256"],
        "observed-writer-policy-binding",
    )
    require(
        all(
            same(owner[x], actual[x])
            for x in ("pid", "start_ticks", "cgroup", "invocation_id", "origin_sha256")
        )
        and owner["ppid"] == p["expected_processes"]["coordinator"]["pid"]
        and owner["boot_id"] == k["boot_id"]
        and owner["clock_ticks_per_second"] == k["clock_ticks_per_second"]
        and owner["pid_namespace_inode"]
        == k["namespace_expectations"]["owners"]["learner"]["pid"]
        and owner["time_namespace_inode"]
        == k["namespace_expectations"]["owners"]["learner"]["time"]
        and owner["uid"] == k["credentials"]["learner"]["real"],
        "observed-writer-kernel-binding",
    )
    # command_sha256 is the qualified raw cmdline/argv byte digest; imports,
    # access and qualification hashes are separate external proof commitments.
    # Contract maps AppendProof.uid to actual real UID. All four Uid values are
    # separately bound above, including effective and filesystem credentials.
    sources = {p["sha256"] for p in registration["source_pins"].values()}
    require(
        {
            append.LEARNER_SOURCE,
            append.TRAINING_SOURCE,
            RUNTIME_SOURCE,
            ACTOR_SOURCE,
            COORDINATOR_SOURCE,
        }
        <= sources,
        "observed-r3-source-closure",
    )


def publications(rows, registration, scope):
    require(
        isinstance(rows, dict)
        and set(rows) == {"coordinator", *identity.WORKERS, *COHORTS},
        "observed-publication-roster",
    )
    used = []
    for name, spec in rows.items():
        fields(
            spec,
            {
                "key",
                "owner_role",
                "record_kind",
                "worker_literal",
                "source_contract_sha256",
                "access_qualification_pin",
                "uid",
                "gid",
                "mode",
            },
            "observed-publication-fields",
        )
        if name == "coordinator":
            expected = (
                registration["keys"]["coordinator"],
                "coordinator",
                "coordinator-status",
                None,
            )
        elif name in identity.WORKERS:
            expected = (
                registration["heartbeats"][name],
                name,
                "worker-heartbeat",
                name,
            )
        else:
            entry = registration["cohorts"][name]
            require(
                entry["worker"] == name
                and entry["parent_role"] == name.rsplit("-cohort-", 1)[0],
                "observed-cohort-owner",
            )
            expected = (entry["key"], entry["parent_role"], "cohort-heartbeat", name)
        require(
            same(
                [
                    spec[k]
                    for k in ("key", "owner_role", "record_kind", "worker_literal")
                ],
                list(expected),
            )
            and spec["key"] in scope.files
            and spec["source_contract_sha256"] == PUBLICATION_CONTRACT,
            "observed-publication-binding",
        )
        require(
            all(integer(spec[k]) for k in ("uid", "gid", "mode"))
            and spec["mode"] == 0o600
            and spec["uid"]
            == registration["kernel_context"]["credentials"][spec["owner_role"]][
                "filesystem"
            ],
            "observed-publication-owner-mode",
        )
        bounded_pin(spec["access_qualification_pin"], scope)
        used.append(scope.files[spec["key"]].path)
    require(len(used) == len(set(used)) == 34, "observed-publication-path-alias")
    require(
        scope.tails["metrics"].path not in used, "observed-metrics-publication-alias"
    )


@dataclass(frozen=True)
class ObservedRegistration:
    """Private schema result, not a permission/capability or collector admission."""

    _raw: bytes = field(repr=False)
    _canonical: bytes = field(repr=False)
    _writer_evidence: bytes = field(repr=False)

    @property
    def sha256(self) -> str:
        return sha(self._raw)

    def private_copy(self) -> dict[str, Any]:
        return json.loads(self._canonical)

    def private_writer_evidence(self) -> bytes:
        return bytes(self._writer_evidence)

    def safe_summary(self) -> dict:
        r = self.private_copy()
        return {
            "format": "strength-preservation-observed-registration-schema-summary-v1",
            "contract": CONTRACT,
            "registration_sha256": self.sha256,
            "writer_evidence_sha256": sha(self._writer_evidence),
            "scope_sha256": sha(encoded(r["scope"])),
            "recipe_sha256": r["learner_writer"]["binding"]["recipe_sha256"],
            "owners": 12,
            "workers": 9,
            "cohorts": 24,
            "publications": 34,
            "schema_validated": True,
            "writer_qualified": False,
            "runtime_qualified": False,
            "preservation_passed": False,
            "execution_authorized": False,
        }


def parse_observed_registration(
    raw: bytes,
    *,
    approved_sha256: str,
    scope: readonly.ReadScope,
    writer_evidence: bytes,
) -> ObservedRegistration:
    """No IO. approved_sha256 must come from external authority, not this JSON."""
    try:
        require(
            type(raw) is bytes
            and 0 < len(raw) <= MAX_BYTES
            and append.digest(approved_sha256)
            and sha(raw) == approved_sha256,
            "observed-registration-approval",
        )
        require(isinstance(scope, readonly.ReadScope), "observed-read-scope-type")
        reg = identity.strict_json(raw)
        fields(
            reg,
            set(identity.COMMON_FIELDS) | set(EXTRA_FIELDS),
            "observed-registration-fields",
        )
        require(
            reg["format"] == FORMAT
            and type(reg["schema_version"]) is int
            and reg["schema_version"] == 1
            and reg["contract"] == CONTRACT,
            "observed-registration-format",
        )
        identity.validate_common_registration(
            {k: reg[k] for k in identity.COMMON_FIELDS}, scope
        )
        policy(reg["policy"], reg)
        kernel_context(reg["kernel_context"])
        writer(reg["learner_writer"], reg, scope, writer_evidence)
        publications(reg["publication_writers"], reg, scope)
        bounded_pin(reg["legacy_policy_reference_pin"], scope)
        return ObservedRegistration(bytes(raw), encoded(reg), bytes(writer_evidence))
    except ObservedRegistrationRefusal:
        raise
    except (identity.CollectionRefusal, append.AppendRefusal):
        raise ObservedRegistrationRefusal(
            "observed-shared-validation-refused"
        ) from None
    except (
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        OverflowError,
        RecursionError,
    ):
        raise ObservedRegistrationRefusal("observed-registration-invalid") from None
