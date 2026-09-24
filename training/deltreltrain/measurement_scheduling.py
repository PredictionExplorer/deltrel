"""Durable service debt driven by coordinator-confirmed GPU lease intervals.

Requested time, cooldown and nominal session limits never earn service credit.
The coordinator journal is the authority even when the evaluator dies mid-lease.
Only the single promotion supervisor writes this ledger, under its worker lock.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from .runtime import RunIdentity, atomic_json

MEASUREMENT_RESTORE_POLICY = "preserve-settled-debt-unclosed-interval-uncredited-v1"


def _integer(value: object, name: str, *, signed: bool = False) -> int:
    if type(value) is not int or (not signed and value < 0):
        raise ValueError(f"invalid measurement service {name}")
    return value


def _fraction(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not 0 < value < 1
    ):
        raise ValueError("invalid measurement service fraction")
    return float(value)


def _witness(stream: Any, offset: int) -> str:
    stream.seek(0)
    digest = hashlib.sha256()
    remaining = offset
    while remaining:
        chunk = stream.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise ValueError(
                "coordinator journal is shorter than its measurement cursor"
            )
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


class MeasurementServiceLedger:
    """Persist exact service totals, prospective debt and the pinned matchup.

    Scheduling may exceed the fractional floor to prevent an overdue job from
    starving. The deadline is checked at lease boundaries; it cannot preempt
    kernels or promise completion by that deadline.
    """

    def __init__(
        self,
        *,
        path: Path,
        coordinator_events: Path,
        request_path: Path,
        run_identity: RunIdentity,
    ) -> None:
        self.path = path
        self.coordinator_events = coordinator_events
        self.request_path = request_path
        if path.is_symlink() or coordinator_events.is_symlink():
            raise ValueError("measurement service files must not be symlinks")
        if path.exists():
            loaded = path.read_bytes()
            self.state = json.loads(loaded)
            self.validate_state(self.state, run_identity)
            loaded_sha256 = hashlib.sha256(loaded).hexdigest()
        else:
            loaded_sha256 = None
            # Activation is prospective: old promotion work grants no credit.
            with coordinator_events.open("rb") as stream:
                offset = os.fstat(stream.fileno()).st_size
                if offset:
                    stream.seek(offset - 1)
                    if stream.read(1) != b"\n":
                        raise ValueError(
                            "coordinator journal has an incomplete activation tail"
                        )
                witness = _witness(stream, offset)
            self.state: dict[str, Any] = {
                "schema_version": 1,
                "run_id": run_identity.run_id,
                "generation_family": run_identity.generation_family,
                "journal_offset": offset,
                "journal_witness": witness,
                "promotion_gpu_ns": 0,
                "measurement_gpu_ns": 0,
                "debt_ns": 0,
                "pending_since_ns": None,
                "last_measurement_ready_ns": None,
                "leases": {},
                "pinned_job": None,
                "accounting_complete": True,
                "applied_restore_receipts": [],
            }
            self._save()
        receipt_path = path.with_name("measurement-service-restore.json")
        if receipt_path.exists():
            self._consume_restore_receipt(receipt_path, loaded_sha256, run_identity)
        else:
            self.refresh()

    @staticmethod
    def validate_state(state: object, identity: RunIdentity) -> None:
        if (
            not isinstance(state, dict)
            or type(state.get("schema_version")) is not int
            or state.get("schema_version") != 1
            or state.get("run_id") != identity.run_id
            or state.get("generation_family") != identity.generation_family
        ):
            raise ValueError("measurement service ledger identity is invalid")
        for key in ("journal_offset", "promotion_gpu_ns", "measurement_gpu_ns"):
            _integer(state.get(key), key)
        _integer(state.get("debt_ns"), "debt", signed=True)
        for key in ("pending_since_ns", "last_measurement_ready_ns"):
            if state.get(key) is not None:
                _integer(state[key], key)
        witness = state.get("journal_witness")
        if not isinstance(witness, str) or len(witness) != 64:
            raise ValueError("invalid measurement journal witness")
        leases = state.get("leases")
        if not isinstance(leases, dict) or len(leases) > 1:
            raise ValueError("invalid measurement service outstanding leases")
        for token, lease in leases.items():
            if not isinstance(token, str) or not token or not isinstance(lease, dict):
                raise ValueError("invalid measurement service lease")
            if lease.get("kind") not in ("promotion", "measurement"):
                raise ValueError("invalid measurement service lease kind")
            _integer(lease.get("owner_pid"), "owner_pid")
            _fraction(lease.get("fraction"))
            if lease.get("ready_ns") is not None:
                _integer(lease["ready_ns"], "ready_ns")
        job = state.get("pinned_job")
        if job is not None and (
            not isinstance(job, dict)
            or set(job) != {"candidate", "baseline"}
            or any(not isinstance(value, str) or not value for value in job.values())
            or job["candidate"] == job["baseline"]
        ):
            raise ValueError("invalid measurement service pinned job")
        if type(state.get("accounting_complete", True)) is not bool:
            raise ValueError("invalid measurement accounting completeness")
        receipts = state.get("applied_restore_receipts", [])
        if not isinstance(receipts, list) or any(
            not isinstance(item, str) or len(item) != 64 for item in receipts
        ):
            raise ValueError("invalid measurement restore receipt history")

    def _consume_restore_receipt(
        self, path: Path, loaded_sha256: str | None, identity: RunIdentity
    ) -> None:
        """A verified disaster restore may leave one interval unknowable.

        Never synthesize GPU time: retain settled debt, explicitly mark the gap,
        and keep the admitted matchup. The receipt is bound to captured bytes.
        """
        if path.is_symlink():
            raise ValueError("measurement restore receipt must not be a symlink")
        encoded = path.read_bytes()
        digest = hashlib.sha256(encoded).hexdigest()
        receipt = json.loads(encoded)
        if (
            not isinstance(receipt, dict)
            or type(receipt.get("schema_version")) is not int
            or receipt.get("schema_version") != 1
            or receipt.get("run_id") != identity.run_id
            or receipt.get("generation_family") != identity.generation_family
            or receipt.get("policy") != MEASUREMENT_RESTORE_POLICY
        ):
            raise ValueError("measurement restore receipt identity/policy is invalid")
        if digest in self.state.get("applied_restore_receipts", []):
            self.refresh()
            return
        if loaded_sha256 is None or receipt.get("ledger_sha256") != loaded_sha256:
            raise ValueError("measurement restore receipt ledger checksum differs")
        journal_bytes = _integer(receipt.get("journal_bytes"), "restored journal size")
        restored_ns = _integer(receipt.get("restored_ns"), "restore time")
        tokens = receipt.get("unsettled_tokens")
        if not isinstance(tokens, list) or tokens != sorted(self.state["leases"]):
            raise ValueError("measurement restore receipt unsettled tokens differ")
        if journal_bytes < self.state["journal_offset"]:
            raise ValueError("measurement restore journal omits accounted prefix")
        with self.coordinator_events.open("rb") as stream:
            if _witness(stream, journal_bytes) != receipt.get("journal_sha256"):
                raise ValueError("measurement restore journal checksum differs")
        # Journal replay and receipt consumption are one atomic ledger change;
        # a crash cannot update the ledger hash before marking this receipt.
        self.refresh(persist=False)
        unknown = self.state["leases"]
        if any(token not in tokens for token in unknown):
            raise ValueError("measurement restore encountered an unbound lease")
        if unknown:
            self.state["accounting_complete"] = False
            self.state.setdefault("uncredited_restored_leases", []).append(
                {
                    "restored_ns": restored_ns,
                    "receipt_sha256": digest,
                    "leases": unknown,
                }
            )
            self.state["leases"] = {}
        self.state.setdefault("applied_restore_receipts", []).append(digest)
        self._save()

    def _save(self) -> None:
        atomic_json(self.path, self.state)

    def refresh(self, *, persist: bool = True) -> None:
        """Atomically consume complete journal records, including crash cleanup."""
        # Work on a copy: a malformed suffix cannot partially alter credit.
        state = json.loads(json.dumps(self.state))
        with self.coordinator_events.open("rb") as stream:
            offset = state["journal_offset"]
            if (
                os.fstat(stream.fileno()).st_size < offset
                or _witness(stream, offset) != state["journal_witness"]
            ):
                raise ValueError(
                    "coordinator journal changed before measurement cursor"
                )
            stream.seek(offset)
            while True:
                line = stream.readline()
                if not line or not line.endswith(b"\n"):
                    break
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("coordinator journal event must be an object")
                lease = state["leases"].get(row.get("token"))
                if lease is not None:
                    event = row.get("event")
                    if event in (
                        "pause_lease_ready",
                        "pause_lease_released",
                        "pause_target_restarted",
                        "pause_request_rejected",
                    ):
                        timestamp = _integer(
                            row.get("timestamp_ns"), "lease event time"
                        )
                        if event == "pause_lease_ready":
                            if lease["ready_ns"] is not None:
                                raise ValueError(
                                    "duplicate coordinator lease readiness"
                                )
                            lease["ready_ns"] = timestamp
                            if lease["kind"] == "measurement":
                                state["last_measurement_ready_ns"] = timestamp
                                state["pending_since_ns"] = timestamp
                        else:
                            ready = lease["ready_ns"]
                            if event == "pause_request_rejected" and ready is not None:
                                raise ValueError(
                                    "coordinator rejected an already-ready lease"
                                )
                            if ready is not None:
                                if timestamp < ready:
                                    raise ValueError(
                                        "coordinator lease time moved backwards"
                                    )
                                duration = timestamp - ready
                                kind = lease["kind"]
                                state[f"{kind}_gpu_ns"] += duration
                                share = lease["fraction"]
                                state["debt_ns"] += round(
                                    duration
                                    * (share if kind == "promotion" else share - 1)
                                )
                            del state["leases"][row["token"]]
                offset = stream.tell()
            state["journal_offset"] = offset
            state["journal_witness"] = _witness(stream, offset)
        if state != self.state:
            self.state = state
            if persist:
                self._save()

    def register_lease(self, token: str, kind: str, fraction: float) -> None:
        """Write ownership before the request can acquire a GPU."""
        self.refresh()
        self.reconcile_unrequested()
        if self.state["leases"]:
            raise ValueError("previous measurement-accounted GPU lease is unreconciled")
        if not token or kind not in ("promotion", "measurement"):
            raise ValueError("invalid measurement service registration")
        self.state["leases"][token] = {
            "kind": kind,
            "owner_pid": os.getpid(),
            "fraction": _fraction(fraction),
            "ready_ns": None,
        }
        self._save()

    def reconcile_unrequested(self) -> None:
        """Recover a crash after registration but before publishing the request.

        A lease that ever became ready requires its coordinator release event;
        process death alone never invents its release time or earns credit.
        """
        request = (
            json.loads(self.request_path.read_text())
            if self.request_path.exists()
            else {}
        )
        for token, lease in list(self.state["leases"].items()):
            if lease["ready_ns"] is not None or request.get("token") == token:
                continue
            try:
                os.kill(lease["owner_pid"], 0)
            except ProcessLookupError:
                del self.state["leases"][token]
                self._save()

    @property
    def pinned_job(self) -> dict[str, str] | None:
        value = self.state["pinned_job"]
        return dict(value) if value is not None else None

    def pin_job(
        self,
        candidate: str,
        baseline: str,
        *,
        candidate_manifest: Path | None = None,
        baseline_manifest: Path | None = None,
    ) -> None:
        job = {"candidate": candidate, "baseline": baseline}
        if not candidate or not baseline or candidate == baseline:
            raise ValueError("measurement matchup must contain distinct models")
        if self.pinned_job not in (None, job):
            raise ValueError("cannot replace an unfinished protected measurement")
        self.state["pinned_job"] = job
        # Existing artifact GC recognizes these top-level manifest references,
        # including a crash before the first arena resume file is published.
        if candidate_manifest is not None:
            self.state["candidate_manifest"] = str(candidate_manifest.resolve())
        if baseline_manifest is not None:
            self.state["baseline_manifest"] = str(baseline_manifest.resolve())
        self._save()

    def complete_job(self) -> None:
        self.state["pinned_job"] = None
        self.state["pending_since_ns"] = None
        self.state.pop("candidate_manifest", None)
        self.state.pop("baseline_manifest", None)
        self._save()

    def should_serve(self, *, due: bool, now_ns: int, max_wait_seconds: float) -> bool:
        self.refresh()
        _integer(now_ns, "decision time")
        if not math.isfinite(max_wait_seconds) or max_wait_seconds <= 0:
            raise ValueError("invalid measurement service wait bound")
        if not due:
            if self.state["pending_since_ns"] is not None or self.state["debt_ns"]:
                self.state["pending_since_ns"] = None
                # Idle service cannot accumulate a future GPU monopoly. Keep
                # actual counters, but reserve bandwidth only while backlogged.
                self.state["debt_ns"] = 0
                self._save()
            return False
        pending = self.state["pending_since_ns"]
        if pending is None:
            pending = now_ns
            self.state["pending_since_ns"] = pending
            self._save()
        if now_ns < pending:
            raise ValueError("measurement scheduling clock moved backwards")
        return self.state["debt_ns"] >= 0 or now_ns - pending >= max_wait_seconds * 1e9


def validate_measurement_service_state(
    state: object, run_identity: RunIdentity
) -> None:
    """Validate captured ledger data without reading files or mutating state."""
    MeasurementServiceLedger.validate_state(state, run_identity)


def measurement_service_status(
    root: Path,
    run_identity: RunIdentity,
    *,
    now_ns: int,
    target_fraction: float = 0.0,
) -> dict[str, object]:
    """Read settled service accounting without replaying or writing the ledger."""
    path = root / "arena" / "measurement-service.json"
    if not path.exists():
        return {
            "available": False,
            "status": "missing" if target_fraction else "disabled",
            "target_fraction": target_fraction,
            "path": str(path),
        }
    try:
        if path.is_symlink():
            raise ValueError("measurement service ledger must not be a symlink")
        state = json.loads(path.read_text())
        validate_measurement_service_state(state, run_identity)
    except (OSError, ValueError) as error:
        return {
            "available": False,
            "status": "invalid",
            "path": str(path),
            "reason": str(error),
        }
    promotion = state["promotion_gpu_ns"] / 1e9
    measurement = state["measurement_gpu_ns"] / 1e9

    def age(value: object) -> float | None:
        return max(0.0, (now_ns - value) / 1e9) if type(value) is int else None

    return {
        "available": True,
        "status": "complete"
        if state.get("accounting_complete", True)
        else "restored_with_uncredited_interval",
        "accounting_complete": state.get("accounting_complete", True),
        "path": str(path),
        "target_fraction": target_fraction,
        "promotion_gpu_seconds": promotion,
        "measurement_gpu_seconds": measurement,
        "observed_fraction": measurement / (promotion + measurement)
        if promotion + measurement
        else None,
        "debt_seconds": state["debt_ns"] / 1e9,
        "pending_wait_seconds": age(state["pending_since_ns"]),
        "last_service_age_seconds": age(state["last_measurement_ready_ns"]),
        "open_leases": len(state["leases"]),
        "pinned_job": state["pinned_job"],
        "scope": "settled-coordinator-ready-to-release-intervals-only",
        "reservation_scope": "while-independent-measurement-work-is-pending",
    }
