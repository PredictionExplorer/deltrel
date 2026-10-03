"""Deterministic preparation for the closed dummy CPU installer.

This module has no filesystem/systemd backend or executable entry point. Its
artifacts are proposals, not target admission. The enclosing qualified operator
must validate BEFORE producer completion, retain the actual dummy start before
calling this renderer, and admit the exact host/source/runtime inputs before any
write. Original clocks and the twelve-unit namespace are never renewed here.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
import math
from pathlib import Path
import re
from typing import Any, Mapping

from scripts import strength_freshness_cpu_completion as completion
from scripts import strength_freshness_cpu_lifecycle as life
from scripts import strength_freshness_cpu_qualification as q
from scripts import strength_freshness_cpu_support_case as support_case

MAX_ARTIFACT_BYTES = 4 * 2**20
SETUP_SECONDS = 60


@dataclass(frozen=True)
class Artifact:
    path: str
    data: bytes
    mode: int

    @property
    def pin(self) -> dict[str, object]:
        return {"path": self.path, "sha256": q.sha(self.data), "bytes": len(self.data)}


@dataclass(frozen=True)
class Prepared:
    plan: q.Plan
    anchor: life.Anchor
    inputs: tuple[Artifact, ...]
    installed: tuple[Artifact, ...]
    template_sha256: str

    @property
    def start_order(self) -> tuple[str, ...]:
        units = self.plan.value["units"]
        return tuple(
            next(name for name, spec in units.items() if spec["role"] == role)
            for role in ("watchdog", "observer", "publisher", "dispatcher")
        )


def _read(pin: Mapping[str, object], sources: Mapping[str, bytes]) -> bytes:
    checked = q.pin_shape(dict(pin))
    data = sources.get(checked["path"])
    q.require(
        isinstance(data, bytes)
        and len(data) == checked["bytes"]
        and q.sha(data) == checked["sha256"],
        "setup-input-byte-pin",
    )
    assert isinstance(data, bytes)
    return data


def _watchdog(data: bytes, start: life.Clock) -> bytes:
    # systemd's duration representation is microseconds. Rounding toward the
    # earlier instant is conservative; the recorded Anchor itself is unchanged.
    seconds = (Decimal(str(start.monotonic)) + 390).quantize(
        Decimal("0.000001"), rounding=ROUND_FLOOR
    )
    changed, count = re.subn(
        rb"(?m)^OnBootSec=[^\r\n]+$",
        b"OnBootSec=" + format(seconds, "f").encode("ascii") + b"s",
        data,
    )
    q.require(count == 1, "single-watchdog-deadline")
    return changed


def _startup_bound(plan: q.Plan, name: str, data: bytes, anchor: life.Anchor) -> None:
    spec = plan.value["units"][name]
    if spec["role"] not in {"dispatcher", "observer", "publisher"}:
        return
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = lambda optionstr: optionstr
    parser.read_string(data.decode("utf-8"))
    fields = parser["Service"]
    duration = q.linux.systemd_seconds(fields["RuntimeMaxSec"])
    stop = q.linux.systemd_seconds(fields["TimeoutStopSec"])
    start_timeout = q.linux.systemd_seconds(fields["TimeoutStartSec"])
    # Include both a slow service start and its lifetime/stop. Actual entered
    # monotonic timestamps must still pass q.validate_actor_bounds on the host.
    latest = anchor.started_monotonic + SETUP_SECONDS + start_timeout
    phase = spec["role"] + "_terminal"
    q.require(latest + duration + stop <= anchor.deadline(phase), "setup-role-lifetime")
    if spec["role"] == "dispatcher":
        q.require(latest + duration <= anchor.deadline("work"), "setup-dispatcher-work")


def prepare(
    template: bytes,
    approved_template_sha256: str,
    start: life.Clock,
    sources: Mapping[str, bytes],
    *,
    before_execution: bytes,
    expected_before: Mapping[str, Any],
    before_evidence: Mapping[str, bytes],
    approved_outer_intent_sha256: str,
) -> Prepared:
    """Render only the admitted watchdog deadline and hash-bound companions.

    ``approved_template_sha256`` is a separate post-BEFORE admission supplied by
    an independently qualified outer preparer. It is not embedded in the earlier
    outer intent: this template pins BEFORE completion, which already binds that
    intent, so doing so would create a hash cycle. The real outer preparer's
    prospective template/finalization contract remains a separate implementation
    and qualification gate. Computing this digest alone does not authorize it.

    BEFORE pins already belong to this final template. This pure function checks
    their producer-completion semantics but cannot attest an actual observed
    start, source/bootstrap admission, or install/start anything. Supplied source
    bytes are copied into immutable proposal artifacts.
    """
    original = q.Plan.parse(template, approved_template_sha256)
    p = original.value
    input_root = Path(p["input_root"])
    for protected in [p["control_root"], *p["protected_roots"]]:
        root = Path(protected)
        q.require(
            not input_root.is_relative_to(root) and not root.is_relative_to(input_root),
            "setup-input-protected-overlap",
        )
    q.require(
        start.boot_id == p["boot_id"]
        and life.finite(start.monotonic)
        and start.monotonic >= 0
        and type(start.wall_ns) is int
        and start.wall_ns > 0,
        "setup-recorded-start",
    )
    q.require(len(p["units"]) == 12, "setup-exact-twelve-units")
    q.require(
        "before_execution" in p["preservation"], "setup-before-execution-required"
    )
    q.require(
        q.HASH.fullmatch(approved_outer_intent_sha256)
        and expected_before.get("outer_intent_sha256") == approved_outer_intent_sha256
        and expected_before.get("nonce") == p["nonce"]
        and expected_before.get("boot_id") == p["boot_id"],
        "setup-before-expected-authority",
    )
    before_pin = p["preservation"]["before_execution"]
    q.require(
        q.sha(before_execution) == before_pin["sha256"]
        and len(before_execution) == before_pin["bytes"],
        "setup-before-execution-pin",
    )
    verified = completion.validate_before(
        before_execution,
        expected_before,
        {
            "boot_id": start.boot_id,
            # Match the stored Anchor projection used by the proof consumers.
            "monotonic_ns": math.floor(start.monotonic * 10**9),
            "wall_ns": start.wall_ns,
        },
        evidence=before_evidence,
    )
    for artifact, field in (
        ("capture", "before"),
        ("request", "before_request"),
        ("receipt", "before_receipt"),
    ):
        q.require(
            verified.artifacts[artifact] == p["preservation"][field],
            "setup-before-artifact-binding",
        )
    q.require(
        any(
            Path(pin["path"]).name == "strength_freshness_cpu_completion.py"
            for pin in p["source_pins"]
        ),
        "setup-completion-source-required",
    )
    inputs: dict[str, Artifact] = {}
    readonly_paths = {
        Path(pin["path"])
        for pin in [
            *p["source_pins"],
            *(p["preservation"][key] for key in q.PRESERVATION_INPUTS),
            *(
                pin
                for pin in p["runtime_inputs"].values()
                if isinstance(pin, dict) and "path" in pin
            ),
        ]
    }

    def add(path: str, data: bytes, mode: int) -> None:
        root = Path(p["input_root"])
        q.require(
            q.canonical(path) != root and q.canonical(path).is_relative_to(root),
            "setup-input-scope",
        )
        item = Artifact(path, bytes(data), mode)
        candidate = Path(path)
        q.require(
            not any(
                candidate.is_relative_to(other) or other.is_relative_to(candidate)
                for other in readonly_paths
            ),
            "setup-readonly-path-alias",
        )
        if path in inputs:
            q.require(inputs[path] == item, "setup-conflicting-artifact")
        q.require(
            not any(
                path != old
                and (
                    candidate.is_relative_to(Path(old))
                    or Path(old).is_relative_to(candidate)
                )
                for old in inputs
            ),
            "setup-artifact-path-prefix",
        )
        inputs[path] = item
        q.require(
            sum(len(x.data) for x in inputs.values()) <= MAX_ARTIFACT_BYTES,
            "setup-artifact-budget",
        )

    # Validate every registered variant; unused/extra variants are still part of
    # the closed later support transition and cannot smuggle unrelated bytes.
    for name, spec in p["units"].items():
        for stage in ("before", "after"):
            q.validate_unit_text(original, name, _read(spec[stage]["unit"], sources))
            for env in spec[stage]["environment_files"]:
                env_pin = {
                    "path": env["source_path"],
                    "sha256": env["sha256"],
                    "bytes": env["bytes"],
                }
                q.require(
                    {
                        "source": env_pin,
                        "mode": env["mode"],
                        "uid": 0,
                        "gid": 0,
                    }
                    in p["files"].get(env["path"], []),
                    "setup-environment-variant-unregistered",
                )
                env_bytes = _read(env_pin, sources)
                q.require(
                    env_bytes in (b"CPUQUAL_PHASE=before\n", b"CPUQUAL_PHASE=after\n"),
                    "setup-inert-environment",
                )
                add(env_pin["path"], env_bytes, 0o444)
        if spec["role"] == "watchdog":
            q.require(spec["before"] == spec["after"], "setup-fixed-watchdog")
            old = spec["before"]["unit"]
            q.require(
                len(p["files"][spec["installed_path"]]) == 1
                and p["files"][spec["installed_path"]][0]["source"] == old,
                "setup-watchdog-variant",
            )
            raw = _watchdog(_read(old, sources), start)
            new = Artifact(old["path"], raw, 0o444).pin
            spec["before"]["unit"] = new
            spec["after"]["unit"] = new
            p["files"][spec["installed_path"]][0]["source"] = new
            add(old["path"], raw, 0o444)
    # Other source files retain their exact bytes. The only permitted content
    # change to the approved template is its absolute watchdog deadline.
    for target, variants in p["files"].items():
        for variant in variants:
            pin = variant["source"]
            if pin["path"] not in inputs:
                add(pin["path"], _read(pin, sources), 0o444)
            else:
                q.require(inputs[pin["path"]].pin == pin, "setup-source-alias")
            if target.endswith(".service") or target.endswith(".timer"):
                q.validate_unit_text(
                    original, Path(target).name, inputs[pin["path"]].data
                )
            else:
                q.require(
                    inputs[pin["path"]].data
                    in (b"CPUQUAL_PHASE=before\n", b"CPUQUAL_PHASE=after\n"),
                    "setup-inert-environment",
                )
    guard = p["units"][p["bindings"]["support_guard"]]
    scenario_pin = guard["payload"]["scenario"]
    scenario_data = _read(scenario_pin, sources)
    support_case.validate_scenario(original, scenario_pin, scenario_data)
    q.validate_shared_support_bindings(original, life.strict_json(scenario_data))
    add(scenario_pin["path"], scenario_data, 0o444)
    data = q.encode(p)
    plan = q.Plan.parse(data, q.sha(data))
    anchor = life.Anchor.from_dict(
        {
            "attempt_id": p["attempt_id"],
            "nonce": p["nonce"],
            "plan_sha256": plan.checksum,
            "boot_id": start.boot_id,
            "started_monotonic": start.monotonic,
            "started_wall_ns": start.wall_ns,
        }
    )
    root = Path(p["input_root"])
    reserved = {str(root / "plan.json"), str(root / "anchor.json")}
    reserved.update(
        str(root / (name + ".authorization.json"))
        for name in p["units"]
        if name.endswith(".service")
    )
    q.require(not reserved.intersection(inputs), "setup-generated-path-alias")
    q.require(
        not reserved.intersection(
            pin["path"]
            for pin in p["preservation"].values()
            if isinstance(pin, dict) and "path" in pin
        ),
        "setup-preservation-path-alias",
    )
    anchor_data = q.encode(anchor.as_dict())
    add(str(root / "plan.json"), plan.data, 0o444)
    add(str(root / "anchor.json"), anchor_data, 0o444)
    installed: dict[str, Artifact] = {}
    for name, spec in p["units"].items():
        raw = inputs[spec["before"]["unit"]["path"]].data
        _startup_bound(plan, name, raw, anchor)
        installed[spec["installed_path"]] = Artifact(spec["installed_path"], raw, 0o644)
        for env in spec["before"]["environment_files"]:
            item = Artifact(env["path"], inputs[env["source_path"]].data, env["mode"])
            q.require(
                item.path not in installed or installed[item.path] == item,
                "setup-environment-alias",
            )
            installed[item.path] = item
        if name.endswith(".service"):
            auth = {
                "format": "strength-freshness-cpu-authorization-v1",
                "schema_version": 1,
                "plan_path": str(root / "plan.json"),
                "plan_sha256": plan.checksum,
                "anchor_path": str(root / "anchor.json"),
                "anchor_sha256": q.sha(anchor_data),
                "source_pins": p["source_pins"],
                "role": spec["role"],
                "unit": name,
                "mode": spec["mode"],
                "payload": spec["payload"],
            }
            add(str(root / (name + ".authorization.json")), q.encode(auth), 0o600)
    return Prepared(
        plan,
        anchor,
        tuple(inputs[k] for k in sorted(inputs)),
        tuple(installed[k] for k in sorted(installed)),
        original.checksum,
    )
