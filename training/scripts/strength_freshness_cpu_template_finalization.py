"""Pure, non-circular finalization of an independently approved dummy blueprint.

The BEFORE launch/request/registration and blueprint are frozen by the external
preparer inside the original preflight allowance. Future producer outputs are
three fixed-path placeholders, not an intent hash cycle. This module performs no
writes or launches and confers no target/source/session qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from scripts import strength_freshness_cpu_completion as completion
from scripts import strength_freshness_cpu_capture_request as requests

FORMAT = "strength-freshness-install-blueprint-v1"
DYNAMIC = {"before": "capture", "before_receipt": "receipt", "before_execution": None}
EMPTY_SHA = "0" * 64


class Reader(Protocol):
    def read(
        self, pin: Mapping[str, Any], *, root=None, limit: int = 2**20
    ) -> bytes: ...


@dataclass(frozen=True)
class Finalized:
    template: bytes
    template_sha256: str
    before_execution: bytes
    _expected: bytes
    _evidence: tuple[tuple[str, bytes], ...]
    blueprint_sha256: str

    @property
    def expected_before(self) -> dict[str, Any]:
        return completion.parse(self._expected)

    @property
    def before_evidence(self) -> dict[str, bytes]:
        return dict(self._evidence)


def finalize(
    blueprint_raw: bytes,
    expected_blueprint_sha256: str,
    before_bundle_raw: bytes,
    *,
    approved_outer_intent_sha256: str,
    qualified_outer_source_sha256: str,
    enclosing: Mapping[str, Any],
    reader: Reader,
    original_start: Mapping[str, Any],
) -> Finalized:
    """Replace exactly three declared placeholders using completed BEFORE proof.

    The caller supplies independently admitted intent/source/enclosing identities
    and the measured dummy start. The reference clock is historical; the reader
    must enforce its *current* deadline, protected roots, byte pins and budget.
    It must not use that historical reference to renew I/O authority.
    """
    c = completion
    c.require(
        c.checksum(expected_blueprint_sha256)
        and c.sha(blueprint_raw) == expected_blueprint_sha256,
        "blueprint-byte-pin",
    )
    bp = c.parse(blueprint_raw, 2**20)
    c.shape(
        bp,
        {
            "format",
            "schema_version",
            "plan",
            "before_launch",
            "before_registration",
            "primitive_source_sha256",
            "guardian_contract_sha256",
        },
        "blueprint-fields",
    )
    c.require(
        bp["format"] == FORMAT
        and type(bp["schema_version"]) is int
        and bp["schema_version"] == 1,
        "blueprint-format",
    )
    c.require(
        c.checksum(approved_outer_intent_sha256)
        and c.checksum(qualified_outer_source_sha256)
        and bp["primitive_source_sha256"] == qualified_outer_source_sha256
        and c.checksum(bp["guardian_contract_sha256"]),
        "blueprint-outer-authority",
    )
    plan = bp["plan"]
    c.require(
        isinstance(plan, dict)
        and plan.get("format") == "strength-freshness-cpu-plan-v1"
        and type(plan.get("schema_version")) is int
        and plan["schema_version"] == 1,
        "blueprint-plan-format",
    )
    root = c.path(plan["input_root"])
    preserved = plan["preservation"]
    c.require(isinstance(preserved, dict), "blueprint-preservation")
    fixed = {
        "before": root / "r3-before.json",
        "before_receipt": root / "r3-before.receipt.json",
        "before_execution": root / "before-execution.json",
    }
    for name in DYNAMIC:
        p = c.pin(preserved[name])
        c.require(
            c.path(p["path"]) == fixed[name]
            and p["sha256"] == EMPTY_SHA
            and p["bytes"] == 1,
            "blueprint-dynamic-placeholder",
        )
    c.require(
        c.pin(preserved["before_request"])["sha256"] != EMPTY_SHA
        and c.pin(preserved["policy"])["sha256"] != EMPTY_SHA,
        "blueprint-fixed-before-inputs",
    )
    for name in ("before_launch", "before_registration"):
        p = c.pin(bp[name])
        c.require(
            p["sha256"] != EMPTY_SHA and c.path(p["path"]).parent == root,
            "blueprint-fixed-before-location",
        )
    bundle = c.parse(before_bundle_raw, 32768)
    c.shape(
        bundle,
        {"phase", "completion_pin", "artifacts", "evidence_pins"},
        "before-bundle-fields",
    )
    c.require(bundle["phase"] == "before", "before-bundle-phase")
    proof_pin = c.pin(bundle["completion_pin"])
    c.require(
        c.path(proof_pin["path"]) == fixed["before_execution"],
        "before-bundle-proof-location",
    )
    artifacts = c.shape(bundle["artifacts"], c.ARTIFACTS, "before-bundle-artifacts")
    c.require(
        artifacts["launch"] == bp["before_launch"]
        and artifacts["registration"] == bp["before_registration"]
        and artifacts["request"] == preserved["before_request"],
        "before-bundle-fixed-inputs",
    )
    raw = reader.read(proof_pin, root=root)
    c.require(
        c.sha(raw) == proof_pin["sha256"] and len(raw) == proof_pin["bytes"],
        "before-proof-byte-pin",
    )
    value = c.parse(raw)
    c.require(
        value.get("artifacts") == artifacts
        and value.get("evidence_pins") == bundle["evidence_pins"],
        "before-bundle-proof-join",
    )
    c.require(
        value.get("outer_intent_sha256") == approved_outer_intent_sha256
        and value.get("qualified_outer_source_sha256") == qualified_outer_source_sha256
        and value.get("enclosing") == enclosing
        and value.get("exec_contracts", {}).get("guardian")
        == bp["guardian_contract_sha256"],
        "before-proof-outer-join",
    )
    for name, artifact in DYNAMIC.items():
        replacement = proof_pin if artifact is None else c.pin(artifacts[artifact])
        c.require(c.path(replacement["path"]) == fixed[name], "before-output-location")
        preserved[name] = dict(replacement)
    verified = requests.validate_committed_before(
        preserved,
        nonce=plan["nonce"],
        boot_id=plan["boot_id"],
        source_pins=plan["source_pins"],
        control_root=plan["control_root"],
        reader=reader,
        now_clock=original_start,
        python=plan["python"]["path"],
    )
    c.require(verified.sha256 == proof_pin["sha256"], "before-verified-pin")
    evidence = {}
    evidence_root = root / "execution-evidence"
    for name, p in c.evidence_pins(raw).items():
        c.require(
            c.path(p["path"])
            == evidence_root / ("before-" + name.replace("_", "-") + ".json"),
            "before-evidence-location",
        )
        evidence[name] = reader.read(p, root=evidence_root, limit=c.MAX_EVIDENCE)
    expected = c.expected_from(verified.value)
    c.require(
        expected["outer_intent_sha256"] == approved_outer_intent_sha256
        and expected["qualified_outer_source_sha256"] == qualified_outer_source_sha256
        and expected["enclosing"] == enclosing,
        "before-final-authority",
    )
    # Prove the complete plan differs from the approved blueprint only at the
    # three output byte pins. Requests, policy, unit/source/native bindings and
    # original protocol/limits are never learned from the returned proof.
    restored = c.parse(c.encoded(plan))
    for name in DYNAMIC:
        restored["preservation"][name] = {
            "path": str(fixed[name]),
            "sha256": EMPTY_SHA,
            "bytes": 1,
        }
    c.require(restored == c.parse(blueprint_raw)["plan"], "blueprint-unapproved-delta")
    template = c.encoded(plan)
    c.require(len(template) <= 2**20, "final-template-bound")
    return Finalized(
        template,
        c.sha(template),
        bytes(raw),
        c.encoded(expected),
        tuple(sorted(evidence.items())),
        expected_blueprint_sha256,
    )
