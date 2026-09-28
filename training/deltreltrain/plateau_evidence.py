"""Durable promotion verdicts and checkpointed plateau-consumption receipts.

The mutable promotion status describes the arena's current job. It cannot serve
as a queue: a later candidate can replace it before the learner observes a
completed rejection. Small verdict files survive that replacement; recovery
receipts are saved in the same checkpoint as the learning-rate change.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TypedDict, cast

from .runtime import atomic_json

PLATEAU_RECEIPTS_KEY = "plateau_verdict_receipts"
_REJECTIONS = frozenset({"reject", "reject_ring_regression", "reject_max_pairs"})


class VerdictReceipt(TypedDict):
    champion_identity: str
    evaluation_contract_identity: str | None
    consumed_verdict_ids: list[str]
    learner_step: int
    applied_ns: int


class VerdictReceipts(TypedDict):
    schema_version: int
    scopes: dict[str, VerdictReceipt]


def _identity(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()


def scope_identity(champion: str, contract: str | None) -> str:
    return _identity(champion, contract)


def verdict_identity(champion: str, contract: str | None, candidate: str) -> str:
    return _identity(champion, contract, candidate)


@dataclass(frozen=True)
class PlateauVerdict:
    candidate_identity: str
    candidate_step: int
    champion_identity: str
    champion_step: int
    evaluation_contract_identity: str | None
    decision: str
    completed_ns: int

    @property
    def identity(self) -> str:
        return verdict_identity(
            self.champion_identity,
            self.evaluation_contract_identity,
            self.candidate_identity,
        )

    @property
    def conclusive(self) -> bool:
        return self.decision != "reject_max_pairs"

    @classmethod
    def parse(cls, payload: object) -> "PlateauVerdict | None":
        if not isinstance(payload, Mapping):
            return None
        names = tuple(cls.__dataclass_fields__)
        if any(name not in payload for name in names):
            return None
        values = {name: payload[name] for name in names}
        if (
            any(
                not isinstance(values[name], str) or not values[name]
                for name in ("candidate_identity", "champion_identity")
            )
            or any(
                type(values[name]) is not int or values[name] < 0
                for name in ("candidate_step", "champion_step", "completed_ns")
            )
            or values["completed_ns"] == 0
            or not isinstance(values["decision"], str)
            or values["decision"] not in _REJECTIONS
            or (
                values["evaluation_contract_identity"] is not None
                and not isinstance(values["evaluation_contract_identity"], str)
            )
        ):
            return None
        return cls(**values)


def record_verdict(arena: Path, verdict: PlateauVerdict) -> None:
    """Publish once per (champion, contract, candidate), including on retries."""

    path = arena / "plateau-verdicts" / f"{verdict.identity}.json"
    if path.is_file():
        existing = PlateauVerdict.parse(_read_json(path))
        if (
            existing is None
            or replace(existing, completed_ns=verdict.completed_ns) != verdict
        ):
            raise ValueError("persisted plateau verdict is invalid")
        return
    atomic_json(path, {"schema_version": 1, **asdict(verdict)})


def receipts_from_checkpoint_extra(extra: object) -> VerdictReceipts:
    payload = extra.get(PLATEAU_RECEIPTS_KEY) if isinstance(extra, Mapping) else None
    if payload is None:
        return {"schema_version": 1, "scopes": {}}
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("scopes"), Mapping)
    ):
        raise ValueError("checkpoint plateau verdict receipts are invalid")
    scopes: dict[str, VerdictReceipt] = {}
    for key, receipt in payload["scopes"].items():
        if not isinstance(receipt, Mapping):
            raise ValueError("checkpoint plateau verdict receipt is invalid")
        champion = receipt.get("champion_identity")
        contract = receipt.get("evaluation_contract_identity")
        consumed = receipt.get("consumed_verdict_ids")
        if (
            not isinstance(champion, str)
            or not champion
            or (contract is not None and not isinstance(contract, str))
            or key != scope_identity(champion, contract)
            or type(receipt.get("learner_step")) is not int
            or receipt["learner_step"] < 0
            or type(receipt.get("applied_ns")) is not int
            or receipt["applied_ns"] <= 0
            or not isinstance(consumed, list)
            or not consumed
            or any(not isinstance(item, str) or not item for item in consumed)
            or len(set(consumed)) != len(consumed)
        ):
            raise ValueError("checkpoint plateau verdict receipt is invalid")
        scopes[key] = cast(VerdictReceipt, dict(receipt))
    return {"schema_version": 1, "scopes": scopes}


def receipt_for_recovery(
    receipts: object, evidence: Mapping[str, object], *, learner_step: int
) -> VerdictReceipts:
    state = receipts_from_checkpoint_extra({PLATEAU_RECEIPTS_KEY: receipts})
    champion = evidence["champion_identity"]
    contract = evidence["evaluation_contract_identity"]
    verdict_ids = evidence["verdict_ids"]
    if (
        not isinstance(champion, str)
        or not champion
        or (contract is not None and not isinstance(contract, str))
        or not isinstance(verdict_ids, list)
        or not verdict_ids
        or any(not isinstance(value, str) or not value for value in verdict_ids)
    ):
        raise ValueError("plateau recovery evidence is invalid")
    key = scope_identity(champion, contract)
    prior = state["scopes"].get(key)
    consumed = set(prior["consumed_verdict_ids"] if prior else ())
    consumed.update(verdict_ids)
    state["scopes"][key] = {
        "champion_identity": champion,
        "evaluation_contract_identity": contract,
        "consumed_verdict_ids": sorted(consumed),
        "learner_step": learner_step,
        "applied_ns": time.time_ns(),
    }
    return state


def _read_json(path: Path) -> object:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


class PlateauVerdictStore:
    """Read compact verdicts and backfill pre-upgrade terminal arena results.

    A poll interval and per-file stat cache keep control checks cheap. Arena
    results are atomically replaced, so changed results are retried even when a
    previous read observed a partial evaluation or absent candidate manifest.
    """

    def __init__(self, arena: Path):
        self.arena = arena
        self._next_scan = 0.0
        self._signatures: dict[Path, tuple[int, int]] = {}
        self._verdicts: dict[str, PlateauVerdict] = {}

    def has_scope(self, champion: str, contract: str | None) -> bool:
        return any(
            item.champion_identity == champion
            and item.evaluation_contract_identity == contract
            for item in self._verdicts.values()
        )

    def _legacy_verdict(self, path: Path, payload: object) -> PlateauVerdict | None:
        if (
            not isinstance(payload, Mapping)
            or payload.get("result_kind") != "promotion"
            or payload.get("terminal") is not True
            or not isinstance(payload.get("promotion"), Mapping)
            or not isinstance(payload["promotion"].get("decision"), str)
            or payload["promotion"].get("decision") not in _REJECTIONS
        ):
            return None
        contract = payload.get("evaluation_contract")
        if contract is not None and not isinstance(contract, Mapping):
            return None
        values = {
            "candidate_identity": payload.get("candidate"),
            "champion_identity": payload.get("baseline"),
            "evaluation_contract_identity": (
                contract.get("identity") if isinstance(contract, Mapping) else None
            ),
            "decision": payload["promotion"]["decision"],
            "completed_ns": payload.get("completed_ns"),
        }
        # Older result files store immutable manifest paths instead of steps.
        # Never guess missing steps from the current mutable model pointers.
        for role in ("candidate", "champion"):
            step = payload.get(f"{role}_step")
            if type(step) is not int:
                manifest = payload.get(f"{role}_manifest")
                if not isinstance(manifest, str):
                    return None
                manifest_path = Path(manifest)
                if not manifest_path.is_absolute():
                    manifest_path = path.parent / manifest_path
                data = _read_json(manifest_path)
                if (
                    not isinstance(data, Mapping)
                    or data.get("model_identity") != values[f"{role}_identity"]
                ):
                    return None
                step = data.get("model_step")
            values[f"{role}_step"] = step
        return PlateauVerdict.parse(values)

    def _scan(self, poll_seconds: float) -> None:
        now = time.monotonic()
        if now < self._next_scan:
            return
        self._next_scan = now + poll_seconds
        paths = list((self.arena / "plateau-verdicts").glob("*.json"))
        paths.extend(self.arena.glob("*-vs-*.json"))
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                continue
            signature = (stat.st_mtime_ns, stat.st_size)
            if self._signatures.get(path) == signature:
                continue
            payload = _read_json(path)
            compact = path.parent.name == "plateau-verdicts"
            verdict = (
                PlateauVerdict.parse(payload)
                if compact
                else self._legacy_verdict(path, payload)
            )
            if verdict is None:
                if isinstance(payload, Mapping) and (
                    payload.get("result_kind") != "promotion"
                    or payload.get("terminal") is not True
                ):
                    self._signatures[path] = signature
                # Recheck unresolved evidence: a missing manifest can be
                # restored, or an interrupted write can finish on the next poll.
                continue
            self._signatures[path] = signature
            if not compact:
                record_verdict(self.arena, verdict)
            self._verdicts.setdefault(verdict.identity, verdict)

    def pending_recovery(
        self,
        *,
        champion_identity: str,
        champion_step: int,
        contract_identity: str | None,
        learner_step: int,
        receipts: object,
        required_rejections: int,
        count_inconclusive: bool,
        poll_seconds: float,
    ) -> dict[str, object] | None:
        self._scan(poll_seconds)
        state = receipts_from_checkpoint_extra({PLATEAU_RECEIPTS_KEY: receipts})
        receipt = state["scopes"].get(
            scope_identity(champion_identity, contract_identity)
        )
        consumed = set(receipt["consumed_verdict_ids"] if receipt else ())
        minimum_step = max(champion_step, receipt["learner_step"] if receipt else -1)
        applied_ns = receipt["applied_ns"] if receipt else 0
        verdicts = sorted(
            (
                item
                for item in self._verdicts.values()
                if item.champion_identity == champion_identity
                and item.champion_step == champion_step
                and item.evaluation_contract_identity == contract_identity
                and item.identity not in consumed
                and minimum_step < item.candidate_step <= learner_step
                and item.completed_ns > applied_ns
                and (count_inconclusive or item.conclusive)
            ),
            key=lambda item: (item.completed_ns, item.identity),
        )
        if len(verdicts) < required_rejections:
            return None
        latest = verdicts[-1]
        # Consume the entire observed backlog in one recovery. The next recovery
        # needs evidence from candidates trained after this one, even if older
        # candidates finish their arena jobs later.
        return {
            "champion_identity": champion_identity,
            "champion_step": champion_step,
            "evaluation_contract_identity": contract_identity,
            "candidate_identity": latest.candidate_identity,
            "candidate_step": latest.candidate_step,
            "verdict_ids": [item.identity for item in verdicts],
        }
