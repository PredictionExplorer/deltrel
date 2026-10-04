"""Bounded publication-renewal measurements, conditional on external premises.

No historical birth, learner work, preservation or execution verdict is issued.
The source-backed reporter writes cached details: only publication freshness is
measured. Raw records remain private. Old collector/birth wrappers are untouched.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import Any

from scripts import strength_freshness_cpu_collect_records as records
from scripts import strength_freshness_cpu_observed_kernel as kernel
from scripts import strength_freshness_cpu_observed_registration as registration
from scripts import strength_freshness_cpu_readonly as readonly

FORMAT = "strength-publication-renewal-measurement-v1"
MAX_ROUNDS = 256
MAX_PROOF_BYTES = 4 * 2**20
STREAMS = ("coordinator", *records.WORKERS, *records.COHORTS)
STAT_FIELDS = {
    "device",
    "inode",
    "mode",
    "uid",
    "gid",
    "bytes",
    "mtime_ns",
    "ctime_ns",
    "links",
}
CLOCK_FIELDS = {"boot_id", "monotonic_ns", "wall_ns"}
# Exact R3 producer phase vocabulary. This is a closed measurement subset;
# stopped/draining/failure/recovery phases cannot become healthy publications.
ACTOR_PHASES = {
    "waiting_for_champion",
    "replay_reconciliation",
    "champion_selfplay_plateau",
    "champion_selfplay_stale",
    "selfplay",
    "cohort_complete",
    "selfplay_completed",
    "selfplay_policy_salvaged",
    "selfplay_policy_published",
    "selfplay_refill",
    "selfplay_cohort",
}
PHASES = {
    "learner": {"training", "replay_wait", "update_to_data_wait"},
    "arena-promotion": {
        "arena",
        "arena_batch",
        "waiting_for_gpu_pause",
        "arena_gpu_lease_ready",
        "arena_gpu_lease_released",
        "waiting_for_candidate",
        "bootstrapped_champion",
        "candidate_superseded",
        "awaiting_new_candidate",
        "arena_inter_wave_cooldown",
        "historical_crossplay",
        "arena_loading_candidate",
        "arena_loading_champion",
        "arena_search_start",
        "promoted",
        "arena_terminal",
        "arena_continue",
        "model_gc",
        "candidate_terminal",
    },
}
COUNTERS = ("progress", "step", "cumulative_games")
NEGATIVE_KEYS = {
    "failure",
    "failure_reason",
    "failure_class",
    "error",
    "error_type",
    "hardware_failure_reason",
    "hardware_failure_class",
}
NONFINITE_COUNTERS = {
    "nonfinite_loss_count",
    "nonfinite_gradient_count",
    "nonfinite_gradient_tensors",
}


class RenewalRefusal(ValueError):
    """Fixed safe reason only; never arbitrary private exception text."""


class RenewalIncomplete(RenewalRefusal):
    """Insufficient observations; no production stall or repair claim."""


def require(ok: object, reason: str) -> None:
    if not ok:
        raise RenewalRefusal(reason)


def same(a: Any, b: Any) -> bool:
    return registration.encoded(a) == registration.encoded(b)


def digest(value: Any) -> str:
    return registration.sha(registration.encoded(value))


@dataclass(frozen=True)
class PublicationFence:
    _private: bytes = field(repr=False)

    def safe_summary(self) -> dict[str, Any]:
        value = self.safe_proof()
        return {
            "format": FORMAT,
            "kind": "fence",
            "streams": 34,
            "sha256": registration.sha(self._private),
            "read_end": value["read_end"],
            "publication_renewed": False,
            "preservation_passed": False,
        }

    def safe_proof(self) -> dict[str, Any]:
        return json.loads(self._private)


@dataclass(frozen=True)
class RenewalPoll:
    round: int
    renewed: int

    def safe_summary(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "kind": "observations-ready" if self.renewed == 34 else "pending",
            "round": self.round,
            "renewed": self.renewed,
            "required": 34,
            "preservation_passed": False,
            "physical_work_proven": False,
        }


@dataclass(frozen=True)
class RenewalEvidence:
    _private: bytes = field(repr=False)

    def private_copy(self) -> dict[str, Any]:
        return json.loads(self._private)

    def safe_proof(self) -> dict[str, Any]:
        """Bounded numeric/digest transcript, not independent replay of raw rows."""
        return self.private_copy()

    def safe_summary(self) -> dict[str, Any]:
        value = self.private_copy()
        return {
            k: value[k]
            for k in (
                "format",
                "registration_sha256",
                "window_sha256",
                "external_premises_sha256",
                "verified_champions_sha256",
                "fence_sha256",
                "observations_sha256",
                "rounds",
                "streams",
                "read_end",
                "publication_renewed",
                "physical_work_proven",
                "historical_lifetime_proven",
                "writer_qualified",
                "preservation_passed",
                "execution_authorized",
            )
        }


def _negative(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in NEGATIVE_KEYS:
                require(item is None, "publication-negative-evidence")
            if key in NONFINITE_COUNTERS:
                require(
                    type(item) is int and item == 0, "publication-nonfinite-evidence"
                )
            if key == "global_norm_finite":
                require(item is True, "publication-nonfinite-evidence")
            if key == "event":
                raise RenewalRefusal("publication-unknown-event")
            _negative(item)
    elif isinstance(value, list):
        for item in value:
            _negative(item)


class PublicationRenewalTracker:
    """One fixed window and one retained IO; errors permanently poison this chain.

    Kernel objects must have been emitted by the qualified-code observer instance.
    Its private issuance seal prevents accidental synthetic/cross-context use; it
    does not grant external source/access/writer qualification.
    """

    def __init__(
        self,
        reg: registration.ObservedRegistration,
        io: readonly.ReadOnlyIO,
        *,
        expected_window: dict[str, Any],
        external_premises: dict[str, Any],
        verified_champions: dict[str, str],
    ):
        self._registration, self._io = reg, io
        self._original_io = io
        self._reg = reg.private_copy()
        self._window = kernel.validate_expected_window(
            expected_window, boot_id=self._reg["kernel_context"]["boot_id"]
        )
        self._premises = kernel.validate_external_premises(reg, external_premises)
        require(
            isinstance(verified_champions, dict) and bool(verified_champions),
            "champion-proof-map",
        )
        require(
            all(
                records._identity(k) and records._sha(v)
                for k, v in verified_champions.items()
            ),
            "champion-proof-map",
        )
        self._champions = deepcopy(verified_champions)
        require(
            set(self._reg["publication_writers"]) == set(STREAMS),
            "publication-inventory",
        )
        self._initial: kernel.KernelSnapshot | None = None
        self._last_kernel: kernel.KernelSnapshot | None = None
        self._baseline: dict[str, Any] = {}
        self._latest: dict[str, Any] = {}
        self._renewed: set[str] = set()
        self._history: list[dict[str, Any]] = []
        self._io_audits: list[dict[str, Any]] = []
        self._finished_records_digest: str | None = None
        self._safe_history: list[dict[str, Any]] = []
        self._proof_bytes = 0
        self._kernel_brackets: list[dict[str, Any]] = []
        self._last_clock: dict[str, Any] | None = None
        self._failure: str | None = None
        self._finished = False
        self._round = 0
        self._fence: PublicationFence | None = None
        self._io_deadline = io.deadline
        self._io_maximum = io.maximum_bytes
        self._io_scope = io.scope
        self._consumed = io._consumed

    def _live(self) -> None:
        require(self._failure is None, "renewal-chain-refused")
        require(not self._finished, "renewal-chain-finished")
        require(
            self._io.deadline == self._io_deadline
            and self._io.maximum_bytes == self._io_maximum
            and self._io.scope is self._io_scope
            and self._io._consumed >= self._consumed,
            "renewal-io-context-changed",
        )
        self._consumed = self._io._consumed

    def _clock(self, value: Any, *, advance: bool = True) -> dict[str, Any]:
        require(
            isinstance(value, dict) and set(value) == CLOCK_FIELDS, "publication-clock"
        )
        require(
            value["boot_id"] == self._window["phase_start"]["boot_id"]
            and all(
                type(value[k]) is int and value[k] > 0
                for k in ("monotonic_ns", "wall_ns")
            ),
            "publication-clock",
        )
        for key in ("monotonic_ns", "wall_ns"):
            require(
                self._window["phase_start"][key]
                <= value[key]
                < self._window["deadline"][key],
                "publication-original-deadline",
            )
            if self._last_clock is not None:
                require(
                    self._last_clock[key] <= value[key], "publication-clock-regression"
                )
        if advance:
            self._last_clock = deepcopy(value)
        return value

    def _context(self, snapshot: kernel.KernelSnapshot) -> dict[str, Any]:
        kernel.require_snapshot_context(
            snapshot, self._registration, self._io, self._window, self._premises
        )
        if self._initial is not None:
            kernel.same_kernel_observations(self._initial, snapshot)
        if self._last_kernel is not None:
            kernel.same_kernel_observations(self._last_kernel, snapshot)
        value = snapshot.private_copy()
        self._clock(value["read_start"])
        self._clock(value["read_end"])
        self._last_kernel = snapshot
        self._kernel_brackets.append(snapshot.safe_summary())
        return value

    def _observation(self, name: str, owners: dict[str, Any]) -> dict[str, Any]:
        spec = self._reg["publication_writers"][name]
        obs = self._io.read_publication(spec["key"])
        raw, audit = obs.value, deepcopy(dict(obs.audit))
        self._io_audits.append(deepcopy(audit))
        # Retain hashes/stat facts even when a later payload check refuses.
        self._history.append(audit)
        require(
            type(raw) is bytes
            and audit["operation"] == "read-publication"
            and audit["subject"] == spec["key"],
            "publication-observation-kind",
        )
        require(
            same(audit["raw"], {"sha256": registration.sha(raw), "bytes": len(raw)})
            and type(audit["offset"]) is int
            and audit["offset"] == 0
            and type(audit["end_offset"]) is int
            and audit["end_offset"] == len(raw),
            "publication-raw-binding",
        )
        self._clock(audit["read_start"])
        self._clock(audit["read_end"])
        stat = audit["stat_before"]
        require(
            isinstance(stat, dict)
            and set(stat) == STAT_FIELDS
            and all(type(v) is int and v >= 0 for v in stat.values()),
            "publication-stat",
        )
        require(
            stat["inode"] > 0
            and stat["links"] > 0
            and stat["bytes"] == len(raw)
            and all(stat[k] == spec[k] for k in ("uid", "gid", "mode")),
            "publication-access",
        )
        require(
            all(
                same(stat, audit[k])
                for k in ("named_before", "stat_after", "named_after")
            ),
            "publication-incoherent-snapshot",
        )
        safe = {
            k: deepcopy(audit[k])
            for k in (
                "operation",
                "read_start",
                "read_end",
                "raw",
                "named_before",
                "stat_before",
                "stat_after",
                "named_after",
                "offset",
                "end_offset",
            )
        }
        safe.update(stream=name, round=self._round, audit_sha256=digest(audit))
        row = records.parse_json(raw)
        teacher = None
        _negative(row)
        owner = owners[spec["owner_role"]]
        require(
            type(row.get("schema_version")) is int and row["schema_version"] == 1,
            "publication-schema",
        )
        if name == "coordinator":
            stamp = row["timestamp_ns"]
            self._coordinator(row, owners)
            counters = {
                role: row["workers"][role]["restart_count"] for role in records.WORKERS
            }
        else:
            records._heartbeat(row, worker=spec["worker_literal"], pid=owner["pid"])
            stamp = row["heartbeat_ns"]
            if (
                spec["record_kind"] == "cohort-heartbeat"
                or spec["owner_role"] == "actor-cpu-ring4"
            ):
                allowed = ACTOR_PHASES
            elif spec["owner_role"] in records.ACTORS:
                allowed = {"shared_cohorts"}
            else:
                allowed = PHASES[spec["owner_role"]]
            require(row["phase"] in allowed, "publication-unknown-phase")
            counters = {key: row[key] for key in COUNTERS if key in row}
            require(
                "progress" in counters
                and all(type(v) is int and v >= 0 for v in counters.values()),
                "publication-counter",
            )
            counters["progress_ns"] = row["progress_ns"]
            is_cohort = spec["record_kind"] == "cohort-heartbeat"
            if is_cohort or spec["owner_role"] == "actor-cpu-ring4":
                # Scalar ActorSupervisor emits model_role/version. Its requested
                # role exists only with a work lease; absence is not a candidate.
                requested_ok = (
                    row.get("requested_model_role") == "champion"
                    if is_cohort or "requested_model_role" in row
                    else True
                )
                require(
                    row.get("model_role") == "champion"
                    and row.get("model_version") in self._champions
                    and requested_ok,
                    "publication-champion-proof",
                )
                teacher = {
                    "model_identity": row["model_version"],
                    "proof_sha256": self._champions[row["model_version"]],
                }
            if is_cohort:
                require(
                    type(row["cumulative_games"]) is int
                    and row["cumulative_games"] >= 0
                    and type(row["cohort"]) is int
                    and row["cohort"] >= 0,
                    "publication-cohort-counter",
                )
        require(
            type(stamp) is int and 0 < stamp <= audit["read_end"]["wall_ns"],
            "publication-future-stamp",
        )
        value = {
            "audit": audit,
            "version": stat,
            "sha256": registration.sha(raw),
            "stamp": stamp,
            "counters": counters,
            "row": row,
            "row_sha256": digest(row),
        }
        previous = self._latest.get(name)
        if previous is not None:
            require(
                stat["device"] == previous["version"]["device"],
                "publication-device-changed",
            )
            require(
                stamp >= previous["stamp"]
                and set(previous["counters"]) <= set(counters)
                and all(counters[k] >= v for k, v in previous["counters"].items()),
                "publication-counter-regression",
            )
        safe.update(
            producer_stamp_ns=stamp, counters=deepcopy(counters), teacher=teacher
        )
        self._proof_bytes += len(registration.encoded(safe))
        require(self._proof_bytes <= MAX_PROOF_BYTES, "renewal-proof-byte-bound")
        self._safe_history.append(safe)
        return value

    def _coordinator(self, row: dict[str, Any], owners: dict[str, Any]) -> None:
        require(
            row["state"] == "running"
            and row["draining"] is False
            and type(row["coordinator_pid"]) is int
            and row["coordinator_pid"] == owners["coordinator"]["pid"]
            and all(
                row[k] is None
                for k in (
                    "failure",
                    "hardware_failure_reason",
                    "hardware_failure_class",
                )
            ),
            "publication-coordinator-state",
        )
        require(
            isinstance(row["workers"], dict)
            and set(row["workers"]) == set(records.WORKERS),
            "publication-worker-roster",
        )
        for role, worker in row["workers"].items():
            expected_role = (
                "learner"
                if role == "learner"
                else "arena"
                if role == "arena-promotion"
                else "actor"
            )
            require(
                worker["role"] == expected_role
                and worker["state"] == "running"
                and type(worker["pid"]) is int
                and worker["pid"] == owners[role]["pid"]
                and type(worker["restart_count"]) is int
                and worker["restart_count"]
                == self._reg["policy"]["expected_processes"][role]["restarts"]
                and all(
                    worker[k] is None
                    for k in ("failure_reason", "failure_class", "failure_exit_code")
                ),
                "publication-worker-state",
            )

    def _read_round(self, owners: dict[str, Any], *, baseline: bool) -> None:
        for name in STREAMS:
            value = self._observation(name, owners)
            if not baseline:
                first = self._baseline[name]
                require(self._initial is not None, "renewal-fence-required")
                assert self._initial is not None
                if (
                    value["stamp"] > first["stamp"]
                    and value["stamp"]
                    > self._initial.private_copy()["read_end"]["wall_ns"]
                    and value["sha256"] != first["sha256"]
                    and not same(value["version"], first["version"])
                ):
                    self._renewed.add(name)
            self._latest[name] = value
        clock_observation = self._io.clock()
        self._io_audits.append(deepcopy(dict(clock_observation.audit)))
        self._clock(dict(clock_observation.value))
        self._consumed = self._io._consumed

    def _fail(self, error: Exception) -> None:
        reason = (
            str(error)
            if isinstance(error, RenewalRefusal)
            else "renewal-observation-refused"
        )
        self._failure = reason
        if isinstance(error, RenewalIncomplete):
            raise RenewalIncomplete(reason) from None
        raise RenewalRefusal(reason) from None

    def fence(self, kernel_before: kernel.KernelSnapshot) -> PublicationFence:
        try:
            self._live()
            require(self._initial is None, "renewal-fence-once")
            value = self._context(kernel_before)
            self._initial = kernel_before
            self._read_round(value["owners"], baseline=True)
            self._baseline = deepcopy(self._latest)
            private = {
                "registration_sha256": self._registration.sha256,
                "window_sha256": digest(self._window),
                "read_end": self._last_clock,
                "observations": self._safe_history,
                "kernel_admission": kernel_before.safe_summary(),
            }
            self._fence = PublicationFence(registration.encoded(private))
            return self._fence
        except (
            ValueError,
            TypeError,
            KeyError,
            OSError,
            AttributeError,
            readonly.ReadRefusal,
        ) as error:
            self._fail(error)
            raise AssertionError("unreachable")

    def observe(self, kernel_current: kernel.KernelSnapshot) -> RenewalPoll:
        try:
            self._live()
            require(
                self._initial is not None and self._round < MAX_ROUNDS,
                "renewal-round-bound",
            )
            value = self._context(kernel_current)
            self._round += 1
            self._read_round(value["owners"], baseline=False)
            return RenewalPoll(self._round, len(self._renewed))
        except (
            ValueError,
            TypeError,
            KeyError,
            OSError,
            AttributeError,
            readonly.ReadRefusal,
        ) as error:
            self._fail(error)
            raise AssertionError("unreachable")

    def finish(self, kernel_after: kernel.KernelSnapshot) -> RenewalEvidence:
        try:
            self._live()
            require(self._initial is not None, "renewal-fence-required")
            self._context(kernel_after)
            clock_observation = self._io.clock()
            self._io_audits.append(deepcopy(dict(clock_observation.audit)))
            self._clock(dict(clock_observation.value))
            if len(self._renewed) != 34:
                raise RenewalIncomplete("renewal-incomplete")
            assert self._last_clock is not None and self._fence is not None
            require(
                all(
                    self._last_clock["wall_ns"] - row["stamp"]
                    <= self._reg["policy"]["maximum_age_ns"]
                    for row in self._latest.values()
                ),
                "renewal-final-age",
            )
            result = {
                "format": FORMAT,
                "registration_sha256": self._registration.sha256,
                "window_sha256": digest(self._window),
                "external_premises_sha256": digest(self._premises),
                "verified_champions_sha256": digest(self._champions),
                "fence_sha256": registration.sha(self._fence._private),
                "observations_sha256": digest(self._safe_history),
                "observations": self._safe_history,
                "kernel_brackets": self._kernel_brackets,
                "fence": self._fence.safe_proof(),
                "rounds": self._round,
                "streams": 34,
                "read_end": self._last_clock,
                "publication_renewed": True,
                "physical_work_proven": False,
                "historical_lifetime_proven": False,
                "writer_qualified": False,
                "preservation_passed": False,
                "execution_authorized": False,
                "latest": {
                    name: {
                        k: row[k] for k in ("stamp", "sha256", "version", "counters")
                    }
                    for name, row in self._latest.items()
                },
            }
            raw = registration.encoded(result)
            require(len(raw) <= MAX_PROOF_BYTES, "renewal-proof-byte-bound")
            self._finished_records_digest = digest(self._record_copy())
            self._finished = True
            return RenewalEvidence(raw)
        except (
            ValueError,
            TypeError,
            KeyError,
            OSError,
            AttributeError,
            readonly.ReadRefusal,
        ) as error:
            self._fail(error)
            raise AssertionError("unreachable")

    def _record_copy(self) -> dict[str, Any]:
        return {
            name: {
                "row": deepcopy(value["row"]),
                "row_sha256": value["row_sha256"],
                "raw_sha256": value["sha256"],
                "audit": deepcopy(value["audit"]),
            }
            for name, value in self._latest.items()
        }

    def _private_context(self) -> None:
        require(self._failure is None, "renewal-chain-refused")
        require(
            self._io is self._original_io
            and self._io.deadline == self._io_deadline
            and self._io.maximum_bytes == self._io_maximum
            and self._io.scope is self._io_scope
            and self._io._consumed >= self._consumed,
            "renewal-io-context-changed",
        )
        if self._last_kernel is not None:
            kernel.require_snapshot_context(
                self._last_kernel,
                self._registration,
                self._io,
                self._window,
                self._premises,
            )

    def private_audits(self) -> list[dict[str, Any]]:
        """Actual prior read audits in execution order; no IO or authority grant."""
        self._private_context()
        require(
            len(registration.encoded(self._io_audits)) <= MAX_PROOF_BYTES,
            "renewal-private-audit-bound",
        )
        return deepcopy(self._io_audits)

    def private_records(self) -> dict[str, Any]:
        """Finished parsed rows with source-attested raw hash links, never raw replay."""
        self._private_context()
        require(self._finished, "renewal-private-records-before-finish")
        value = self._record_copy()
        require(
            set(value) == set(STREAMS)
            and digest(value) == self._finished_records_digest
            and all(
                digest(row["row"]) == row["row_sha256"]
                and row["raw_sha256"] == row["audit"]["raw"]["sha256"]
                for row in value.values()
            ),
            "renewal-private-record-drift",
        )
        require(
            len(registration.encoded(value)) <= MAX_PROOF_BYTES,
            "renewal-private-record-bound",
        )
        return value
