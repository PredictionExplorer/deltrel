"""Bounded, non-barrier work bundles for independently running actor cohorts.

A bundle contains a fixed number of compatible leases. Any free producer may
claim the next lease; it never waits for another producer to finish a game.
The bundle owns one resource reservation until every lease has acquired its
own pin. Model/board choices are joint; conditional per-lease mode choices avoid
both single-mode bundles and correlations with the model schedule.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import stat
import threading
import time
from types import MappingProxyType
from typing import Any

from .runtime import atomic_json


def _schedule_key(value: Hashable) -> str:
    """Typed stable keys; never deserialize executable Python representations."""

    def encode(item):
        if type(item) is str:
            return ["str", item]
        if type(item) is int:
            return ["int", item]
        if type(item) is tuple:
            return ["tuple", [encode(child) for child in item]]
        raise ValueError("persistent work keys must be strings, integers or tuples")

    encoded = json.dumps(encode(value), ensure_ascii=True, separators=(",", ":"))
    if len(encoded) > 1024:
        raise ValueError("persistent work key is too large")
    return encoded


class PersistentWorkSchedule:
    """Fleet-wide assignment credit, durable across individual actor restarts.

    Ring choice precedes its conditional role choice. This avoids delaying a
    low-weight board behind every separate role/board combination. A one-time
    coverage credit serves each positive ring early, then is repaid under the
    unchanged weights. For equal quanta, each ring appears in the first N
    assignments. Counts represent assigned work, not completed games; a process
    failure after assignment can abandon at most its outstanding work quantum.

    The JSON file is atomically replaced under a separate cross-process lock.
    Callers choose one path/namespace/seed for all actor GPUs in the same run.
    Model loading and execution must happen after this short transaction.
    ``coverage_first`` only initializes a new ledger; existing pending coverage
    is preserved when a profile changes or the worker restarts.
    """

    SCHEMA_VERSION = 1
    MAX_SCOPES = 512
    MAX_CHOICES_PER_SCOPE = 128
    MAX_BYTES = 4 * 1024**2

    def __init__(
        self,
        path: str | Path,
        *,
        namespace: str,
        seed: int,
        coverage_first: bool = True,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        if not isinstance(namespace, str) or not namespace or len(namespace) > 512:
            raise ValueError("persistent work namespace must be nonempty and bounded")
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("persistent work seed must be an unsigned 64-bit integer")
        if type(coverage_first) is not bool:
            raise ValueError("persistent work coverage policy must be boolean")
        if (
            isinstance(lock_timeout_seconds, bool)
            or not math.isfinite(lock_timeout_seconds)
            or not 0 < lock_timeout_seconds <= 30
        ):
            raise ValueError("persistent work lock deadline must be in (0, 30]")
        self.path = Path(path)
        self.namespace = namespace
        self.seed = seed
        self.coverage_first = coverage_first
        self.lock_timeout_seconds = lock_timeout_seconds
        self._lock = threading.RLock()
        self._closed = False

    def _empty(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "namespace": self.namespace,
            "seed": self.seed,
            "coverage_first": self.coverage_first,
            "assignment_semantics": "reserved-work-units-not-completed-games",
            "transactions": 0,
            "scopes": {},
        }

    def _read(self) -> dict[str, Any]:
        if self.path.is_symlink():
            raise ValueError("persistent work state may not be a symbolic link")
        if not self.path.exists():
            return self._empty()
        if not self.path.is_file() or self.path.stat().st_size > self.MAX_BYTES:
            raise ValueError("persistent work state is not a bounded regular file")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        expected = self._empty()
        if not isinstance(payload, dict) or payload.keys() != expected.keys():
            raise ValueError("persistent work state schema is invalid")
        for key in (
            "schema_version",
            "namespace",
            "seed",
            "assignment_semantics",
        ):
            if (
                type(payload[key]) is not type(expected[key])
                or payload[key] != expected[key]
            ):
                raise ValueError(
                    f"persistent work state {key} differs from its authority"
                )
        if type(payload["transactions"]) is not int or payload["transactions"] < 0:
            raise ValueError("persistent work transaction count is invalid")
        if type(payload["coverage_first"]) is not bool:
            raise ValueError("persistent work initial coverage policy is invalid")
        scopes = payload["scopes"]
        if not isinstance(scopes, dict) or len(scopes) > self.MAX_SCOPES:
            raise ValueError("persistent work scopes are invalid")
        for scope, state in scopes.items():
            if (
                not isinstance(scope, str)
                or len(scope) > 1024
                or not isinstance(state, dict)
            ):
                raise ValueError("persistent work scope is invalid")
            if state.keys() != {
                "weights",
                "credits",
                "base_credits",
                "assigned_units",
                "segment",
                "coverage_initialized",
                "coverage_pending",
                "max_units",
                "choices",
            }:
                raise ValueError("persistent work credit schema is invalid")
            weights, credits, assigned = (
                state["weights"],
                state["credits"],
                state["assigned_units"],
            )
            base = state["base_credits"]
            pending = state["coverage_pending"]
            if (
                not isinstance(weights, dict)
                or not 1 <= len(weights) <= self.MAX_CHOICES_PER_SCOPE
                or not isinstance(credits, dict)
                or not isinstance(assigned, dict)
                or not isinstance(base, dict)
                or credits.keys() != weights.keys()
                or base.keys() != weights.keys()
                or assigned.keys() != weights.keys()
                or any(not isinstance(key, str) or len(key) > 1024 for key in weights)
                or any(
                    type(value) is not float or not math.isfinite(value) or value <= 0
                    for value in weights.values()
                )
                or abs(math.fsum(weights.values()) - 1.0) > 1e-12
                or any(
                    type(value) is not float or not math.isfinite(value)
                    for value in (*credits.values(), *base.values())
                )
                or any(
                    type(value) is not int or value < 0 for value in assigned.values()
                )
                or type(state["segment"]) is not int
                or state["segment"] < 0
                or type(state["choices"]) is not int
                or state["choices"] < 0
                or type(state["coverage_initialized"]) is not bool
                or (state["coverage_initialized"] and not payload["coverage_first"])
                or type(state["max_units"]) is not int
                or not 1 <= state["max_units"] <= 1_000_000
                or not isinstance(pending, list)
                or any(
                    not isinstance(key, str) or key not in weights for key in pending
                )
                or len(pending) != len(set(pending))
                or (pending and not state["coverage_initialized"])
            ):
                raise ValueError("persistent work credit values are invalid")
            total_units = sum(assigned.values())
            tolerance = max(1e-9, total_units * 1e-12)
            if abs(math.fsum(credits.values())) > tolerance or any(
                value < -state["max_units"] - tolerance
                or value > (len(weights) - 1) * state["max_units"] + tolerance
                for value in (*credits.values(), *base.values())
            ):
                raise ValueError(
                    "persistent work credit exceeds bounded assignment debt"
                )
            for key, weight in weights.items():
                expected_credit = base[key] + total_units * weight - assigned[key]
                if not math.isclose(
                    credits[key],
                    expected_credit,
                    rel_tol=1e-10,
                    abs_tol=tolerance,
                ):
                    raise ValueError(
                        "persistent work credit disagrees with assignment accounting"
                    )
        return payload

    @contextmanager
    def _transaction(self, *, write: bool):
        with self._lock:
            if self._closed:
                raise RuntimeError("persistent work schedule is closed")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self.path.with_name(self.path.name + ".lock")
            descriptor = os.open(
                lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
            )
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ValueError("persistent work lock is not a regular file")
                deadline = time.monotonic() + self.lock_timeout_seconds
                while True:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "persistent work schedule lock deadline expired"
                            ) from None
                        time.sleep(min(0.005, max(0.0, deadline - time.monotonic())))
                payload = self._read()
                yield payload
                if write:
                    payload["transactions"] += 1
                    if len(json.dumps(payload, separators=(",", ":"))) > self.MAX_BYTES:
                        raise ValueError(
                            "persistent work state exceeded its byte bound"
                        )
                    atomic_json(self.path, payload)
            finally:
                os.close(descriptor)

    def _choose(
        self,
        payload: dict[str, Any],
        scope: Hashable,
        weights: Mapping[Hashable, float],
        *,
        units: int,
        coverage: bool = False,
    ) -> Any:
        if type(units) is not int or not 1 <= units <= 1_000_000:
            raise ValueError("persistent work units must be positive bounded integers")
        if (
            not weights
            or len(weights) > self.MAX_CHOICES_PER_SCOPE
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in weights.values()
            )
        ):
            raise ValueError("persistent work weights must be finite and nonnegative")
        total = math.fsum(weights.values())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("persistent work weights need positive finite mass")
        keys = {_schedule_key(key): key for key, value in weights.items() if value > 0}
        normalized = {
            key: float(weights[original] / total)
            for key, original in sorted(keys.items())
        }
        if any(value <= 0 for value in normalized.values()):
            raise ValueError("persistent work normalized weight underflowed")
        scope_key = _schedule_key(scope)
        scopes = payload["scopes"]
        previous = scopes.get(scope_key)
        if previous is None and len(scopes) >= self.MAX_SCOPES:
            raise ValueError("persistent work scope bound exceeded")
        if previous is None or previous["weights"] != normalized:
            # Coverage is a one-time initialization, not a way to repeatedly
            # bootstrap rare work every time weights or availability change.
            initialized = previous is None and coverage and payload["coverage_first"]
            same_support = (
                previous is not None and previous["weights"].keys() == normalized.keys()
            )
            carried = previous if same_support else None
            base = (
                dict(carried["credits"])
                if carried is not None
                else dict.fromkeys(normalized, 0.0)
            )
            pending = (
                list(carried["coverage_pending"])
                if carried is not None
                else list(normalized)
                if initialized
                else []
            )
            previous = {
                "weights": normalized,
                "credits": dict(base),
                "base_credits": base,
                "assigned_units": dict.fromkeys(normalized, 0),
                "segment": 0 if previous is None else previous["segment"] + 1,
                "coverage_initialized": carried["coverage_initialized"]
                if carried is not None
                else initialized,
                "coverage_pending": pending,
                "max_units": max(units, carried["max_units"])
                if carried is not None
                else units,
                "choices": 0,
            }
            scopes[scope_key] = previous
        previous["max_units"] = max(previous["max_units"], units)
        credits = previous["credits"]
        for key, weight in normalized.items():
            credits[key] += weight * units
        selected = max(
            previous["coverage_pending"] or normalized,
            key=lambda key: (
                credits[key],
                hashlib.sha256(f"{self.seed}:{scope_key}:{key}".encode()).digest(),
            ),
        )
        credits[selected] -= units
        # Keep accumulated rounding drift from becoming artificial credit when
        # dynamic weights repeatedly rebase a segment with only a few choices.
        credits[selected] -= math.fsum(credits.values())
        if selected in previous["coverage_pending"]:
            previous["coverage_pending"].remove(selected)
        previous["assigned_units"][selected] += units
        previous["choices"] += 1
        return keys[selected]

    def choose(
        self, scope: Hashable, weights: Mapping[Hashable, float], *, units: int = 1
    ) -> Any:
        with self._transaction(write=True) as payload:
            return self._choose(payload, scope, weights, units=units)

    def choose_role_and_ring(
        self,
        role_weights: Mapping[str, float],
        ring_weights: Mapping[int, float],
        *,
        units: int = 1,
    ) -> tuple[str, int]:
        if any(type(role) is not str for role in role_weights) or any(
            type(ring) is not int for ring in ring_weights
        ):
            raise ValueError("work roles must be strings and rings must be integers")
        with self._transaction(write=True) as payload:
            ring = self._choose(
                payload,
                "rings",
                {key: value for key, value in ring_weights.items()},
                units=units,
                coverage=True,
            )
            role = self._choose(
                payload,
                ("roles", ring),
                {key: value for key, value in role_weights.items()},
                units=units,
            )
            return role, ring

    def snapshot(self) -> dict[str, Any]:
        with self._transaction(write=False) as payload:
            return payload

    def close(self) -> None:
        with self._lock:
            self._closed = True


class WeightedFairChoice:
    """Smooth weighted selection with deterministic, seed-specific tie breaks."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._weights: dict[Hashable, float] = {}
        self._credits: dict[Hashable, float] = {}

    def choose(self, weights: Mapping[Hashable, float]) -> Any:
        if not weights or any(
            not math.isfinite(value) or value < 0 for value in weights.values()
        ):
            raise ValueError("work weights must be finite, nonnegative and nonempty")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("work weights must have positive mass")
        normalized = {key: value / total for key, value in weights.items() if value > 0}
        if normalized != self._weights:
            # A curriculum/availability change begins a new weighted segment;
            # obsolete debt must not force work outside its current support.
            self._weights = normalized
            self._credits = dict.fromkeys(normalized, 0.0)
        for key, value in normalized.items():
            self._credits[key] += value
        selected = max(
            normalized,
            key=lambda key: (
                self._credits[key],
                hashlib.sha256(f"{self.seed}:{key!r}".encode()).digest(),
            ),
        )
        self._credits[selected] -= 1.0
        return selected


@dataclass(frozen=True)
class WorkBundle:
    metadata: Mapping[str, Any]
    acquire: Callable[[], Any]
    release_reservation: Callable[[], None]
    lease_metadata: tuple[Mapping[str, Any], ...] | None = None


@dataclass(frozen=True)
class WorkLease:
    bundle_id: int
    index: int
    metadata: Mapping[str, Any]
    resource: Any


class CompatibleWorkCoordinator:
    def __init__(
        self,
        *,
        cohort_count: int,
        seed: int,
        bundle_cohorts: int | None = None,
        schedule: PersistentWorkSchedule | None = None,
    ) -> None:
        if type(cohort_count) is not int or not 2 <= cohort_count <= 32:
            raise ValueError("compatible work requires 2..32 producer cohorts")
        self.cohort_count = cohort_count
        if bundle_cohorts is None:
            bundle_cohorts = cohort_count
        if type(bundle_cohorts) is not int or not 1 <= bundle_cohorts <= cohort_count:
            raise ValueError("bundle cohorts must be between one and producer count")
        self.bundle_cohorts = bundle_cohorts
        self.schedule = schedule
        self.random = random.Random(seed)
        self.choice = WeightedFairChoice(seed)
        self._mode_choices: dict[Hashable, WeightedFairChoice] = {}
        self._severity_choices: dict[Hashable, WeightedFairChoice] = {}
        self._lock = threading.Lock()
        self._bundle: WorkBundle | None = None
        self._bundle_id = 0
        self._next_index = 0
        self._closed = False
        self._issued = 0
        self._requested_roles: Counter[str] = Counter()
        self._actual_roles: Counter[str] = Counter()
        self._rings: Counter[str] = Counter()
        self._modes: Counter[str] = Counter()
        self._outstanding: dict[tuple[int, int], int] = {}
        self._planned: Counter[str] = Counter()
        self._completed: Counter[str] = Counter()
        self._planned_rings: Counter[str] = Counter()
        self._planned_modes: Counter[str] = Counter()
        self._completed_rings: Counter[str] = Counter()
        self._completed_modes: Counter[str] = Counter()
        self._completed_actual_roles: Counter[str] = Counter()
        self._started = 0
        self._dropped = 0
        self._cancelled_unstarted = 0

    def choose_role_and_ring(
        self,
        role_weights: Mapping[str, float],
        ring_weights: Mapping[int, float],
        *,
        units: int = 1,
    ) -> tuple[str, int]:
        if self.schedule is not None:
            return self.schedule.choose_role_and_ring(
                role_weights, ring_weights, units=units
            )
        return self.choice.choose(
            {
                (role, ring): rw * bw
                for role, rw in role_weights.items()
                for ring, bw in ring_weights.items()
            }
        )

    def choose_severity(
        self, key: Hashable, minimum: int, maximum: int, *, units: int = 1
    ) -> int:
        if not 2 <= minimum <= maximum <= 9:
            raise ValueError("invalid work handicap severity range")
        if self.schedule is not None:
            return self.schedule.choose(
                ("severity", key),
                dict.fromkeys(range(minimum, maximum + 1), 1.0),
                units=units,
            )
        if key not in self._severity_choices:
            self._severity_choices[key] = WeightedFairChoice(self.choice.seed)
        return self._severity_choices[key].choose(
            dict.fromkeys(range(minimum, maximum + 1), 1.0)
        )

    def choose_mode(
        self, role_and_ring: Hashable, weights: Mapping[str, float], *, units: int = 1
    ) -> str:
        if self.schedule is not None:
            return self.schedule.choose(
                ("mode", role_and_ring),
                {key: value for key, value in weights.items()},
                units=units,
            )
        if role_and_ring not in self._mode_choices:
            salt = int.from_bytes(
                hashlib.sha256(repr(role_and_ring).encode()).digest()[:8], "big"
            )
            self._mode_choices[role_and_ring] = WeightedFairChoice(
                self.choice.seed ^ salt
            )
        return self._mode_choices[role_and_ring].choose(
            {key: value for key, value in weights.items()}
        )

    def acquire(
        self, factory: Callable[["CompatibleWorkCoordinator"], WorkBundle]
    ) -> WorkLease:
        with self._lock:
            if self._closed:
                raise RuntimeError("work coordinator is closed")
            if self._bundle is None:
                created = factory(self)
                if (
                    created.lease_metadata is not None
                    and len(created.lease_metadata) != self.bundle_cohorts
                ):
                    created.release_reservation()
                    raise ValueError(
                        "per-lease metadata count differs from bundle cohort count"
                    )
                self._bundle = WorkBundle(
                    MappingProxyType(dict(created.metadata)),
                    created.acquire,
                    created.release_reservation,
                    tuple(
                        MappingProxyType(dict(item)) for item in created.lease_metadata
                    )
                    if created.lease_metadata is not None
                    else None,
                )
                self._bundle_id += 1
                self._next_index = 0
            bundle = self._bundle
            metadata = dict(bundle.metadata)
            if bundle.lease_metadata is not None:
                metadata.update(bundle.lease_metadata[self._next_index])
            metadata = MappingProxyType(metadata)
            # Acquire the individual model pin before releasing the final bundle
            # reservation, including when the first producer finished early.
            resource = bundle.acquire()
            lease = WorkLease(self._bundle_id, self._next_index, metadata, resource)
            self._next_index += 1
            self._issued += 1
            self._requested_roles[str(metadata["requested_model_role"])] += 1
            self._actual_roles[str(metadata["model_role"])] += 1
            self._rings[str(metadata["ring"])] += 1
            self._modes[str(metadata["mode_category"])] += 1
            games = int(metadata["games"])
            self._outstanding[(lease.bundle_id, lease.index)] = games
            self._planned[str(metadata["requested_model_role"])] += games
            self._planned_rings[str(metadata["ring"])] += games
            self._planned_modes[str(metadata["mode_category"])] += games
            if self._next_index == self.bundle_cohorts:
                self._bundle = None
                bundle.release_reservation()
            return lease

    def record_outcome(
        self,
        lease: WorkLease,
        *,
        requested: int,
        started: int,
        completed: int,
        dropped: int,
        cancelling: bool,
    ) -> int:
        """Return unissued quota for a fresh-model continuation, never new debt."""
        if (
            min(requested, started, completed, dropped) < 0
            or started != completed + dropped
            or started > requested
        ):
            raise ValueError("invalid compatible work game accounting")
        with self._lock:
            key = (lease.bundle_id, lease.index)
            if self._outstanding.get(key) != requested:
                raise ValueError("work outcome does not match outstanding quota")
            self._started += started
            self._dropped += dropped
            self._completed[str(lease.metadata["requested_model_role"])] += completed
            self._completed_rings[str(lease.metadata["ring"])] += completed
            self._completed_modes[str(lease.metadata["mode_category"])] += completed
            self._completed_actual_roles[str(lease.metadata["model_role"])] += completed
            remaining = requested - started
            if cancelling:
                self._cancelled_unstarted += remaining
                remaining = 0
            if remaining:
                self._outstanding[key] = remaining
            else:
                del self._outstanding[key]
            return remaining

    def metrics_snapshot(self, *, blocking: bool = True) -> dict[str, object]:
        # Work creation holds this lock through model loading. The actor control
        # loop must keep polling its pause gate while that operation is pending.
        if not self._lock.acquire(blocking=blocking):
            return {"snapshot_available": False, "reason": "work_coordinator_busy"}
        try:
            return {
                "snapshot_available": True,
                "closed": self._closed,
                "producer_cohorts": self.cohort_count,
                "bundle_cohorts": self.bundle_cohorts,
                "persistent_schedule": self.schedule is not None,
                "bundles_created": self._bundle_id,
                "leases_issued": self._issued,
                "pending_leases": self.bundle_cohorts - self._next_index
                if self._bundle is not None
                else 0,
                "requested_model_roles": dict(self._requested_roles),
                "actual_model_roles": dict(self._actual_roles),
                "rings": dict(self._rings),
                "mode_categories": dict(self._modes),
                "planned_games_by_requested_role": dict(self._planned),
                "completed_games_by_requested_role": dict(self._completed),
                "completed_games_by_actual_role": dict(self._completed_actual_roles),
                "planned_games_by_ring": dict(self._planned_rings),
                "completed_games_by_ring": dict(self._completed_rings),
                "planned_games_by_mode": dict(self._planned_modes),
                "completed_games_by_mode": dict(self._completed_modes),
                "game_accounting_scope": "completed_task_outcomes; active durable publications appear in actor metrics",
                "started_games": self._started,
                "dropped_games": self._dropped,
                "cancelled_unstarted_games": self._cancelled_unstarted,
                "outstanding_promised_games": sum(self._outstanding.values()),
                "pending_model_identity": self._bundle.metadata.get("model_identity")
                if self._bundle
                else None,
            }
        finally:
            self._lock.release()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            bundle, self._bundle = self._bundle, None
            if bundle is not None:
                bundle.release_reservation()
