"""Stdlib facade for four fixed, independently supervised installation helpers.

This module never imports the site-enabled training package. Helper execution is
an internal typed capability supplied by the admitted outer runtime, never a
module name or callback from JSON. Directory filename checks are wake hints only;
the fixed helper validates actual lifecycle/source/ownership evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Any, Mapping, Protocol

from scripts import strength_freshness_cpu_completion as c
from scripts import strength_freshness_cpu_outer as outer

REQUEST = "strength-freshness-install-request-v1"
RESULT = "strength-freshness-install-result-v1"
START = "strength-freshness-dummy-start-v1"
MODES = {"prepare-install", "inspect-cleanup", "audit", "prearm-cleanup"}


class HelperExecutor(Protocol):
    def check_alive(self) -> None: ...
    def call(self, mode: str, payload: bytes, deadline_ns: int) -> bytes: ...


class Observation(Protocol):
    def clock(self) -> dict[str, Any]: ...
    def hint(self, root: Path, event: str) -> bool: ...
    def sleep(self, seconds: float) -> None: ...


class SystemObservation:
    def clock(self) -> dict[str, Any]:
        return {
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "monotonic_ns": time.monotonic_ns(),
            "wall_ns": time.time_ns(),
        }

    def hint(self, root: Path, event: str) -> bool:
        c.require(
            event in {"cleanup-complete", "published-candidate"}, "installer-hint-event"
        )
        c.require(root.is_absolute() and root.resolve() == root, "installer-hint-path")
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            c.require(
                stat.S_ISDIR(info.st_mode)
                and info.st_uid == 0
                and stat.S_IMODE(info.st_mode) == 0o700,
                "installer-hint-parent",
            )
            count, found = 0, False
            pattern = re.compile(re.escape(event) + r"-[0-9a-f]{32}\.json\Z")
            with os.scandir(fd) as entries:
                for entry in entries:
                    count += 1
                    # RawLog may briefly expose both the temporary and final
                    # hardlink plus its append lock during atomic publication.
                    c.require(count <= 4098, "installer-hint-file-bound")
                    if pattern.fullmatch(entry.name):
                        c.require(
                            entry.is_file(follow_symlinks=False), "installer-hint-kind"
                        )
                        found = True
            current = root.lstat()
            c.require(
                (current.st_dev, current.st_ino) == (info.st_dev, info.st_ino)
                and root.resolve() == root,
                "installer-hint-parent-replaced",
            )
            return found
        finally:
            os.close(fd)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass(frozen=True)
class Spec:
    input_root: str
    template_root: str
    blueprint: Mapping[str, Any]
    sources: Mapping[str, Any]


class Installer:
    def __init__(
        self,
        admitted_intent: bytes,
        approved_intent_sha256: str,
        enclosing: Mapping[str, Any],
        *,
        intent_pin: Mapping[str, Any],
        helper_executor: HelperExecutor,
        _observation: Observation | None = None,
    ):
        c.require(
            c.checksum(approved_intent_sha256)
            and c.sha(admitted_intent) == approved_intent_sha256,
            "installer-intent-pin",
        )
        self.intent = c.parse(admitted_intent)
        self.intent_pin = c.pin(intent_pin)
        c.require(
            self.intent_pin["sha256"] == approved_intent_sha256
            and self.intent_pin["bytes"] == len(admitted_intent),
            "installer-intent-bytes",
        )
        c.require(
            isinstance(self.intent.get("nonce"), str)
            and re.fullmatch(r"[0-9a-f]{32}", self.intent["nonce"])
            and isinstance(self.intent.get("boot_id"), str),
            "installer-attempt",
        )
        value = c.shape(
            self.intent["installer"],
            {"input_root", "template_root", "blueprint", "sources"},
            "installer-fields",
        )
        inputs, templates = c.path(value["input_root"]), c.path(value["template_root"])
        c.require(
            not inputs.is_relative_to(templates)
            and not templates.is_relative_to(inputs),
            "installer-root-overlap",
        )
        for key in ("blueprint", "sources"):
            pin = c.pin(value[key])
            c.require(
                c.path(pin["path"]).is_relative_to(templates)
                and c.path(pin["path"]) != templates,
                "installer-template-location",
            )
        c.shape(enclosing, {"operator", "supervisor"}, "installer-enclosing")
        operator, supervisor = (
            c.identity(enclosing[k]) for k in ("operator", "supervisor")
        )
        c.require(
            operator["ppid"] == supervisor["pid"]
            and all(
                x["uid"] == 0 and x["boot_id"] == self.intent["boot_id"]
                for x in (operator, supervisor)
            ),
            "installer-enclosing-identity",
        )
        self.enclosing = c.parse(c.encoded(enclosing))
        self.spec = Spec(
            str(inputs),
            str(templates),
            dict(value["blueprint"]),
            dict(value["sources"]),
        )
        self.helper = helper_executor
        self.observation = (
            _observation if _observation is not None else SystemObservation()
        )
        self.stage = "before"
        self.start: dict[str, Any] | None = None
        self.before: dict[str, Any] | None = None
        self.prepared: dict[str, Any] | None = None
        self.calls: set[str] = set()
        self.receipts: dict[str, dict[str, Any]] = {}

    def _deadline(self, seconds: int) -> int:
        assert self.start is not None
        self.helper.check_alive()
        now = c.clock(self.observation.clock())
        c.order(self.start, now)
        mono = math.floor((self.start["monotonic_ns"] / 1e9) * 1e9)
        end = min(
            mono + seconds * 10**9,
            now["monotonic_ns"]
            + self.start["wall_ns"]
            + seconds * 10**9
            - now["wall_ns"],
        )
        c.require(now["monotonic_ns"] < end, "installer-original-deadline")
        return end

    def _call(self, mode: str, cutoff: int) -> dict[str, Any]:
        c.require(
            mode in MODES and mode not in self.calls, "installer-single-helper-call"
        )
        assert self.start is not None and self.before is not None
        end = self._deadline(cutoff)
        now = c.clock(self.observation.clock())
        c.order(self.start, now)
        end = min(
            end, now["monotonic_ns"] + (60 if mode == "prepare-install" else 10) * 10**9
        )
        c.require(
            end - now["monotonic_ns"] > 2 * 10**9, "installer-helper-cleanup-reserve"
        )
        start_record = {
            "format": START,
            "schema_version": 1,
            "outer_intent_sha256": self.intent_pin["sha256"],
            "before_execution_pin": self.before["completion_pin"],
            "clock": self.start,
        }
        raw_start = c.encoded(start_record)
        start_pin = {
            "path": str(Path(self.spec.input_root) / "dummy-start.json"),
            "sha256": c.sha(raw_start),
            "bytes": len(raw_start),
        }
        ack = outer.dummy_start_ack(
            outer_intent_sha256=self.intent_pin["sha256"],
            nonce=self.intent["nonce"],
            start_pin=start_pin,
            before_execution_pin=self.before["completion_pin"],
            enclosing=self.enclosing,
        )
        raw_ack = c.encoded(ack)
        payload = {
            "format": REQUEST,
            "schema_version": 1,
            "operation": mode,
            "intent_pin": self.intent_pin,
            "enclosing": self.enclosing,
            "blueprint_pin": self.spec.blueprint,
            "sources_pin": self.spec.sources,
            "before_bundle": self.before,
            "original_start": self.start,
            "start_pin": start_pin,
            "ack_pin": {
                "path": str(Path(self.spec.input_root) / "dummy-start.ack.json"),
                "sha256": c.sha(raw_ack),
                "bytes": len(raw_ack),
            },
            "prepared_run": self.prepared,
        }
        raw = c.encoded(payload)
        c.require(len(raw) <= 32768, "installer-request-bound")
        self.calls.add(mode)  # An unknown outcome is never retried as a fresh call.
        result = c.parse(self.helper.call(mode, raw, end), 8 * 2**20)
        c.shape(
            result,
            {
                "format",
                "schema_version",
                "operation",
                "status",
                "result",
                "receipt_pin",
            },
            "installer-result-fields",
        )
        c.require(
            result["format"] == RESULT
            and type(result["schema_version"]) is int
            and result["schema_version"] == 1
            and result["operation"] == mode,
            "installer-result-binding",
        )
        receipt = c.pin(result["receipt_pin"])
        c.require(
            c.path(receipt["path"]).parent
            == c.path(self.spec.input_root) / "installer-receipts",
            "installer-receipt-location",
        )
        current = c.clock(self.observation.clock())
        c.order(now, current)
        c.require(current["monotonic_ns"] <= end, "installer-helper-late")
        self._deadline(cutoff)
        self.receipts[mode] = receipt
        c.require(result["status"] == "complete", "installer-helper-incomplete")
        c.require(isinstance(result["result"], dict), "installer-result-object")
        return result["result"]

    def finalize_template_and_install(
        self, before_bundle: bytes, original_start: Mapping[str, Any]
    ) -> bytes:
        c.require(self.stage == "before", "installer-prepare-once")
        self.stage = "preparing"
        try:
            self.start = c.clock(original_start)
            c.require(
                self.start["boot_id"] == self.intent["boot_id"], "installer-start-boot"
            )
            self.before = c.parse(before_bundle, 32768)
            c.shape(
                self.before,
                {"phase", "completion_pin", "artifacts", "evidence_pins"},
                "installer-before-bundle",
            )
            c.require(
                self.before["phase"] == "before"
                and c.path(c.pin(self.before["completion_pin"])["path"])
                == c.path(self.spec.input_root) / "before-execution.json",
                "installer-before-path",
            )
            result = self._call("prepare-install", 60)
            c.shape(
                result,
                {
                    "plan_pin",
                    "anchor_pin",
                    "after_path",
                    "authorization_root",
                    "cleanup_proof_root",
                },
                "installer-prepared-fields",
            )
            for name, filename in (
                ("plan_pin", "plan.json"),
                ("anchor_pin", "anchor.json"),
            ):
                c.require(
                    c.path(c.pin(result[name])["path"])
                    == c.path(self.spec.input_root) / filename,
                    "installer-prepared-pin-location",
                )
            scratch = c.path("/run/edgeconnect-cpuqual-" + self.intent["nonce"])
            c.require(
                c.path(result["after_path"]) == scratch / "external/r3-after.json"
                and c.path(result["authorization_root"]) == c.path(self.spec.input_root)
                and c.path(result["cleanup_proof_root"]) == scratch / "evidence",
                "installer-prepared-locations",
            )
            self.prepared, self.stage = result, "installed"
            return c.encoded(result)
        except BaseException:
            self.stage = "failed"
            raise

    def _wait_hint(self, event: str, cutoff: int) -> None:
        assert self.prepared is not None
        root = Path(self.prepared["cleanup_proof_root"])
        while True:
            end = self._deadline(cutoff)
            if self.observation.hint(root, event):
                self._deadline(cutoff)
                return
            now = c.clock(self.observation.clock())
            c.require(now["monotonic_ns"] < end, "installer-hint-deadline")
            self.observation.sleep(min(0.2, (end - now["monotonic_ns"]) / 1e9))

    def await_checked_cleanup(self) -> bytes:
        c.require(self.stage == "installed", "installer-cleanup-stage")
        self._wait_hint("cleanup-complete", 540)
        result = self._call("inspect-cleanup", 540)
        c.shape(result, {"cleanup_pin", "clock"}, "installer-cleanup-result")
        assert self.prepared is not None
        c.require(
            c.path(c.pin(result["cleanup_pin"])["path"]).parent
            == c.path(self.prepared["cleanup_proof_root"]),
            "installer-cleanup-pin",
        )
        assert self.start is not None
        c.order(self.start, result["clock"])
        c.order(result["clock"], self.observation.clock())
        self.stage = "cleaned"
        return c.encoded(result)

    def final_audit_and_retire(self) -> bytes:
        c.require(self.stage == "cleaned", "installer-audit-stage")
        self._wait_hint("published-candidate", 598)
        result = self._call("audit", 598)
        c.shape(
            result,
            {"status", "plan_sha256", "anchor_sha256", "evidence_pins"},
            "installer-audit-result",
        )
        assert self.prepared is not None
        c.require(
            result["status"] == "passed_cpu_scope"
            and result["plan_sha256"] == self.prepared["plan_pin"]["sha256"]
            and result["anchor_sha256"] == self.prepared["anchor_pin"]["sha256"],
            "installer-audit-bindings",
        )
        pins = result["evidence_pins"]
        c.require(
            isinstance(pins, dict) and 0 < len(pins) <= 16, "installer-audit-evidence"
        )
        for item in pins.values():
            c.require(
                c.path(c.pin(item)["path"]).parent
                == c.path(self.prepared["cleanup_proof_root"]),
                "installer-audit-evidence-location",
            )
        self.stage = "audited"
        return c.encoded(result)

    def prearm_cleanup(self) -> bytes:
        c.require(
            self.stage == "failed"
            and self.start is not None
            and self.before is not None,
            "installer-prearm-stage",
        )
        # The helper independently refuses if any work-role start was requested,
        # even when the prepare result was lost. This is never rollback authority.
        return c.encoded(self._call("prearm-cleanup", 60))
