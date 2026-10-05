"""Fixed site-enabled child entry point; admission precedes heavy imports.

Only four operations exist. The inherited sealed FD127 carries pins, never a
command, callback or test-root override. Missing real site/source/session
qualification is a refusal in the outer admission loader, before this dispatcher.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import math
import os
from pathlib import Path
import sys
from typing import Any

from scripts import strength_freshness_cpu_completion as c
from scripts import strength_freshness_cpu_outer as outer

REQUEST = "strength-freshness-install-request-v1"
RESULT = "strength-freshness-install-result-v1"
START = "strength-freshness-dummy-start-v1"
MODES = {"prepare-install", "inspect-cleanup", "audit", "prearm-cleanup"}


class HelperRefusal(RuntimeError):
    pass


def require(ok: object, reason: str) -> None:
    if not ok:
        raise HelperRefusal(reason)


def validate_request(
    raw: bytes, admission, frame: dict[str, Any], authorization_path: str
) -> dict[str, Any]:
    value = c.parse(raw, 32768)
    c.shape(
        value,
        {
            "format",
            "schema_version",
            "operation",
            "intent_pin",
            "enclosing",
            "blueprint_pin",
            "sources_pin",
            "before_bundle",
            "original_start",
            "start_pin",
            "ack_pin",
            "prepared_run",
        },
        "install-request-fields",
    )
    require(
        value["format"] == REQUEST
        and type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and value["operation"] in MODES
        and value["operation"] == frame["operation"]
        and admission.role == "helper",
        "install-request-binding",
    )
    expected = admission.intent["installer"]
    c.shape(
        expected,
        {"input_root", "template_root", "blueprint", "sources"},
        "installer-authority-fields",
    )
    inputs, templates = (
        c.path(expected["input_root"]),
        c.path(expected["template_root"]),
    )
    require(
        not inputs.is_relative_to(templates) and not templates.is_relative_to(inputs),
        "install-input-template-overlap",
    )
    require(
        value["intent_pin"]
        == {
            "path": authorization_path,
            "sha256": admission.approved_intent_sha256,
            "bytes": len(admission.raw_intent),
        }
        and c.sha(admission.raw_intent) == admission.approved_intent_sha256,
        "install-intent-pin",
    )
    require(
        value["blueprint_pin"] == expected["blueprint"]
        and value["sources_pin"] == expected["sources"],
        "install-fixed-input-pins",
    )
    for item in (value["blueprint_pin"], value["sources_pin"]):
        require(
            c.path(c.pin(item)["path"]).is_relative_to(templates),
            "install-template-pin-scope",
        )
    c.shape(value["enclosing"], {"operator", "supervisor"}, "install-enclosing")
    parent = admission.parent.identity
    require(
        value["enclosing"]["operator"] == asdict(parent)
        and frame["parent"] == asdict(parent),
        "install-operator-parent",
    )
    supervisor = admission.kernel.identity(parent.ppid)
    require(
        value["enclosing"]["supervisor"] == asdict(supervisor),
        "install-supervisor-owner",
    )
    start = c.clock(value["original_start"])
    require(
        start["boot_id"]
        == admission.intent["boot_id"]
        == frame["phase_start"]["boot_id"],
        "install-start-boot",
    )
    c.order(start, frame["phase_start"])
    c.shape(
        value["before_bundle"],
        {"phase", "completion_pin", "artifacts", "evidence_pins"},
        "install-before-bundle",
    )
    require(value["before_bundle"]["phase"] == "before", "install-before-phase")
    start_raw = c.encoded(
        {
            "format": START,
            "schema_version": 1,
            "outer_intent_sha256": admission.approved_intent_sha256,
            "before_execution_pin": value["before_bundle"]["completion_pin"],
            "clock": start,
        }
    )
    require(
        value["start_pin"]
        == {
            "path": str(inputs / "dummy-start.json"),
            "sha256": c.sha(start_raw),
            "bytes": len(start_raw),
        },
        "install-original-start-pin",
    )
    ack = outer.dummy_start_ack(
        outer_intent_sha256=admission.approved_intent_sha256,
        nonce=admission.intent["nonce"],
        start_pin=value["start_pin"],
        before_execution_pin=value["before_bundle"]["completion_pin"],
        enclosing=value["enclosing"],
    )
    ack_raw = c.encoded(ack)
    require(
        value["ack_pin"]
        == {
            "path": str(inputs / "dummy-start.ack.json"),
            "sha256": c.sha(ack_raw),
            "bytes": len(ack_raw),
        },
        "install-start-ack-pin",
    )
    if value["operation"] == "prepare-install":
        require(value["prepared_run"] is None, "install-prepare-must-be-new")
    elif value["prepared_run"] is not None:
        c.shape(
            value["prepared_run"],
            {
                "plan_pin",
                "anchor_pin",
                "after_path",
                "authorization_root",
                "cleanup_proof_root",
            },
            "install-prepared-fields",
        )
    else:
        require(value["operation"] == "prearm-cleanup", "install-prepared-required")
    return value


class Reader:
    """Adapt only approved pin locations to the shared protected store."""

    def __init__(self, admission, frame, blueprint):
        self.admission, self.frame = admission, frame
        self.inputs = c.path(admission.intent["installer"]["input_root"])
        self.templates = c.path(admission.intent["installer"]["template_root"])
        self.sources = {p["path"]: p for p in admission.outer["source_pins"]}
        for p in blueprint["plan"]["source_pins"]:
            checked = c.pin(p)
            if checked["path"] in self.sources:
                require(
                    self.sources[checked["path"]] == checked, "install-source-conflict"
                )
            self.sources[checked["path"]] = checked

    def check(self):
        now = asdict(self.admission.kernel.clock())
        c.order(self.frame["phase_start"], now)
        require(
            all(
                now[k] < self.frame["work_deadline"][k]
                for k in ("monotonic_ns", "wall_ns")
            ),
            "install-helper-original-deadline",
        )
        self.admission.reader.check()
        require(
            not self.admission.kernel.exited(self.admission.parent.pidfd),
            "install-parent-died",
        )

    def read(self, pin, *, root=None, limit=2**20):
        p = c.pin(pin, limit)
        path = c.path(p["path"])
        require(root is None or path.parent == root, "install-proof-parent")
        source = self.sources.get(p["path"]) == p
        require(
            source
            or path.is_relative_to(self.inputs)
            or path.is_relative_to(self.templates),
            "install-read-scope",
        )
        self.check()
        if not source:
            self.admission.reader.roots = tuple(
                set(self.admission.reader.roots) | {path.parent}
            )
        raw = self.admission.reader.read(p, source=source, maximum=limit)
        self.check()
        return raw


def _prepared_result(prepared) -> dict[str, Any]:
    inputs = {item.path: item.pin for item in prepared.inputs}
    p = prepared.plan.value
    root = Path(p["input_root"])
    return {
        "plan_pin": inputs[str(root / "plan.json")],
        "anchor_pin": inputs[str(root / "anchor.json")],
        "after_path": p["preservation"]["after_path"],
        "authorization_root": str(root),
        "cleanup_proof_root": str(Path(p["scratch_root"]) / "evidence"),
    }


def _actual_sources(admission, blueprint, reader, modules) -> None:
    """Observed imports must match both approved outer and final plan closure."""
    outer = {p["path"]: p for p in admission.outer["source_pins"]}
    planned = {p["path"]: p for p in blueprint["plan"]["source_pins"]}
    control = c.path(admission.outer["control_root"])
    for module in modules:
        origin = getattr(module, "__file__", None)
        require(isinstance(origin, str), "install-import-origin")
        assert isinstance(origin, str)
        path = str(Path(origin).resolve())
        require(
            Path(path).is_relative_to(control)
            and path in outer
            and outer[path] == planned.get(path),
            "install-import-source-pin",
        )
        reader.read(outer[path])


def audit_result(raw_pin, *, plan_sha256, anchor_sha256, proof_root) -> dict[str, Any]:
    """Fixed FinalAudit serializer shared with the real facade composition test."""
    checked = c.pin(raw_pin)
    require(
        c.path(checked["path"]).parent == c.path(proof_root), "audit-evidence-location"
    )
    require(
        c.checksum(plan_sha256) and c.checksum(anchor_sha256), "audit-result-bindings"
    )
    return {
        "status": "passed_cpu_scope",
        "plan_sha256": plan_sha256,
        "anchor_sha256": anchor_sha256,
        "evidence_pins": {"external_audit": checked},
    }


def dispatch(
    raw: bytes,
    admission,
    frame: dict[str, Any],
    authorization_path: str,
    *,
    _test_factory=None,
) -> bytes:
    """Internal typed seam only; the CLI never accepts a backend/test callback."""
    value = validate_request(raw, admission, frame, authorization_path)
    ack = c.parse(admission.reader.read(value["ack_pin"], maximum=32768))
    outer.validate_dummy_start_ack(
        ack,
        outer_intent_sha256=admission.approved_intent_sha256,
        nonce=admission.intent["nonce"],
        start_pin=value["start_pin"],
        before_execution_pin=value["before_bundle"]["completion_pin"],
        enclosing=value["enclosing"],
    )
    # All site-dependent imports occur only after caller/session/source/site
    # admission. The real main always obtains that Admission from the loader.
    from scripts import strength_freshness_cpu_template_finalization as finalizer
    from scripts import strength_freshness_cpu_install as render
    from scripts import strength_freshness_cpu_install_backend as backend
    from scripts import strength_freshness_cpu_driver as driver
    from scripts import qualify_cloud_gpu_window as system_host

    authority = admission.intent["installer"]
    templates, inputs = (
        c.path(authority["template_root"]),
        c.path(authority["input_root"]),
    )
    require(
        inputs == c.path(admission.outer["input_root"]), "install-authorized-input-root"
    )
    admission.reader.roots = tuple(
        set(admission.reader.roots)
        | {templates, inputs, c.path(admission.outer["control_root"])}
    )
    admission.reader.roots = tuple(
        set(admission.reader.roots) | {c.path(value["blueprint_pin"]["path"]).parent}
    )
    blueprint_raw = admission.reader.read(value["blueprint_pin"])
    blueprint = c.parse(blueprint_raw)
    reader = Reader(admission, frame, blueprint)
    q, requests = render.q, driver.requests
    _actual_sources(
        admission,
        blueprint,
        reader,
        (
            sys.modules[__name__],
            outer,
            c,
            finalizer,
            render,
            backend,
            q,
            driver,
            q.linux,
            q.support,
            q.linux.core,
            system_host,
            requests,
            requests.records,
            requests.supports,
            requests.supports.identity_module,
            requests.supports.facts,
            requests.supports.identity_module.readonly,
            requests.preservation,
            requests.lifecycle,
        ),
    )
    require(
        blueprint["plan"]["control_root"] == admission.outer["control_root"]
        and blueprint["plan"]["nonce"] == admission.intent["nonce"],
        "install-blueprint-origin",
    )
    start_raw = reader.read(value["start_pin"], root=inputs)
    start_record = c.parse(start_raw)
    require(
        start_record
        == {
            "format": START,
            "schema_version": 1,
            "outer_intent_sha256": admission.approved_intent_sha256,
            "before_execution_pin": value["before_bundle"]["completion_pin"],
            "clock": value["original_start"],
        },
        "install-retained-start",
    )
    finalized = finalizer.finalize(
        blueprint_raw,
        value["blueprint_pin"]["sha256"],
        c.encoded(value["before_bundle"]),
        approved_outer_intent_sha256=admission.approved_intent_sha256,
        qualified_outer_source_sha256=admission.source_sha256,
        enclosing=value["enclosing"],
        reader=reader,
        original_start=value["original_start"],
    )
    source_manifest = c.parse(reader.read(value["sources_pin"]))
    require(len(source_manifest) <= 128, "install-template-source-count")
    sources = {}
    for logical, pin in source_manifest.items():
        require(
            c.path(logical).is_relative_to(inputs)
            and c.path(logical) != inputs
            and c.path(c.pin(pin)["path"]).is_relative_to(templates),
            "install-template-source-scope",
        )
        sources[logical] = reader.read(pin)
    start = value["original_start"]
    prepared = render.prepare(
        finalized.template,
        finalized.template_sha256,
        render.life.Clock(
            start["boot_id"], start["monotonic_ns"] / 1e9, start["wall_ns"]
        ),
        sources,
        before_execution=finalized.before_execution,
        expected_before=finalized.expected_before,
        before_evidence=finalized.before_evidence,
        approved_outer_intent_sha256=admission.approved_intent_sha256,
    )
    result_pins = _prepared_result(prepared)
    if value["prepared_run"] is not None:
        require(value["prepared_run"] == result_pins, "install-retained-preparation")
        for key in ("plan_pin", "anchor_pin"):
            reader.read(result_pins[key], root=inputs)
    reader.check()
    end = frame["work_deadline"]
    stop = render.life.Clock(
        start["boot_id"], end["monotonic_ns"] / 1e9, end["wall_ns"]
    )
    if _test_factory is None:
        require(
            admission.role == "helper"
            and sys.platform == "linux"
            and os.geteuid() == os.getegid() == 0,
            "install-real-admission",
        )
        host, files = (
            backend.SiteHost(prepared, stop, admission_check=reader.check),
            backend.Files(),
        )
        operator = backend.Backend(
            prepared,
            host,
            files,
            intent_sha256=admission.approved_intent_sha256,
            work_deadline=stop,
            start_authorization=backend.StartAuthorization(
                value["ack_pin"], value["start_pin"], value["enclosing"]
            ),
        )
    else:
        operator = _test_factory(prepared, stop)
    op = value["operation"]
    raw_result = getattr(
        operator,
        {
            "prepare-install": "prepare_install",
            "inspect-cleanup": "inspect_cleanup",
            "audit": "audit",
            "prearm-cleanup": "prearm_cleanup",
        }[op],
    )()
    status = "complete"
    if op == "prepare-install":
        require(raw_result["status"] == "installed-and-armed", "install-not-armed")
        result = result_pins
    elif op == "inspect-cleanup":
        if raw_result.get("status") == "pending":
            status, result = "pending", {"reason": "cleanup-proof-not-yet-present"}
        else:
            require(
                raw_result["status"] == "cleanup-observed", "cleanup-proof-incomplete"
            )
            clock = raw_result["clock"]
            result = {
                "cleanup_pin": raw_result["cleanup_pin"],
                "clock": {
                    "boot_id": clock["boot_id"],
                    "monotonic_ns": math.floor(clock["monotonic"] * 1e9),
                    "wall_ns": clock["wall_ns"],
                },
            }
    elif op == "audit":
        result = audit_result(
            raw_result,
            plan_sha256=prepared.plan.checksum,
            anchor_sha256=result_pins["anchor_pin"]["sha256"],
            proof_root=result_pins["cleanup_proof_root"],
        )
    else:
        require(
            raw_result["status"] == "prearm-owned-files-removed",
            "prearm-cleanup-incomplete",
        )
        result = raw_result
    reader.check()
    receipt_root = inputs / "installer-receipts"
    # The approved external preparer creates this one protected evidence parent.
    admission.reader.roots = tuple(set(admission.reader.roots) | {receipt_root})
    admission.reader.directory(receipt_root)
    observed = asdict(admission.kernel.clock())
    receipt = {
        "format": "strength-freshness-install-operation-receipt-v1",
        "schema_version": 1,
        "operation": op,
        "nonce": prepared.anchor.nonce,
        "outer_intent_sha256": admission.approved_intent_sha256,
        "plan_sha256": prepared.plan.checksum,
        "anchor_sha256": result_pins["anchor_pin"]["sha256"],
        "original_start": start,
        "observed_clock": observed,
        "work_deadline": end,
        "status": status,
        "result": result,
        "execution_qualified": False,
    }
    receipt_pin = admission.reader.publish(
        receipt_root / (op + ".json"), c.encoded(receipt)
    )
    reader.check()
    return c.encoded(
        {
            "format": RESULT,
            "schema_version": 1,
            "operation": op,
            "status": status,
            "result": result,
            "receipt_pin": receipt_pin,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", required=True, choices=sorted(MODES))
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    admitted = None
    try:
        from scripts import strength_freshness_cpu_outer_runtime as runtime

        admitted, payload, frame = runtime.helper_entry_admission(
            args.authorization, args.sha256, args.operation
        )
        raw = dispatch(payload, admitted, frame, args.authorization)
        require(len(raw) <= 8 * 2**20, "install-response-bound")
        sys.stdout.buffer.write(raw)
    except Exception:
        # Unadmitted or expired helpers must not acquire filesystem authority in
        # order to publish a failure. Parent records actual nonzero exit/output.
        sys.stdout.write(
            '{"format":"strength-freshness-install-refused-v1","reason":"helper-refused"}\n'
        )
        raise SystemExit(1) from None
    finally:
        if admitted is not None:
            admitted.kernel.close(admitted.parent.pidfd)


if __name__ == "__main__":
    main()
