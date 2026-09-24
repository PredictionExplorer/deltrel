"""Plan resumable predecessor measurements and non-adjacent crossplay."""

from __future__ import annotations

import json
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .checkpoint import MODEL_MANIFEST_FORMAT, ModelManifest, load_model_manifest
from .checkpoint_manifest_cache import ControlManifestCache
from .config import HistoricalEvaluationConfig
from .runtime import RunIdentity

HISTORICAL_CROSSPLAY_RESULT_KIND = "historical_crossplay"
PROMOTION_RESULT_KIND = "promotion"


@dataclass(frozen=True, slots=True)
class HistoricalEvaluationPlan:
    candidate: ModelManifest
    baseline: ModelManifest
    result_path: Path
    previous: dict[str, object] | None
    # "measurement" links a promoted champion to its direct predecessor;
    # unfinished links remain due after newer champions are promoted.
    # "anchor" is periodic non-adjacent crossplay.
    kind: str = "anchor"


def arena_result_kind(result: Mapping[str, object]) -> str:
    """Classify legacy arena files as promotion results."""

    value = result.get("result_kind")
    return value if isinstance(value, str) else PROMOTION_RESULT_KIND


def load_historical_manifests(
    directory: str | Path,
    *,
    run_identity: RunIdentity,
    cache: ControlManifestCache | None = None,
    required_identities: frozenset[str] | None = None,
) -> dict[str, ModelManifest]:
    """Read verified publications, optionally reusing parsed metadata.

    Cache hits rehash pointer/manifest content and the current checkpoint bytes;
    file identities alone never prove payload integrity. Shortlisting avoids
    reading unrelated checkpoints. Weight loading verifies independently.
    """
    manifests: dict[str, ModelManifest] = {}
    for path in Path(directory).glob("manifest-*.json"):
        if required_identities is not None:
            # Read only the small, content-addressed manifest to shortlist
            # identities. No ModelManifest is returned from this metadata: an
            # admitted publication must still pass the full verifier below.
            encoded = path.read_bytes()
            checksum = hashlib.sha256(encoded).hexdigest()
            payload = json.loads(encoded)
            if (
                path.name != f"manifest-{checksum}.json"
                or not isinstance(payload, dict)
                or payload.get("format") != MODEL_MANIFEST_FORMAT
                or not isinstance(payload.get("model_identity"), str)
            ):
                raise ValueError(
                    "historical manifest metadata failed content-address verification"
                )
            if payload["model_identity"] not in required_identities:
                continue
        manifest = load_model_manifest(path) if cache is None else cache.load(path)
        if (
            required_identities is not None
            and manifest.model_identity not in required_identities
        ):
            raise ValueError("historical manifest identity changed during selection")
        if (
            manifest.run_id == run_identity.run_id
            and manifest.generation_family == run_identity.generation_family
        ):
            manifests[manifest.model_identity] = manifest
    return manifests


def load_arena_results(directory: str | Path) -> list[tuple[Path, dict[str, object]]]:
    results = []
    for path in Path(directory).glob("*.json"):
        if path.name == "promotion-status.json" or path.name.endswith(".resume.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            results.append((path, payload))
    return results


def select_historical_evaluation(
    *,
    config: HistoricalEvaluationConfig,
    champion: ModelManifest,
    manifests: Mapping[str, ModelManifest],
    arena_results: Sequence[tuple[Path, Mapping[str, object]]],
    results_directory: str | Path,
    anchor_identity: str | None = None,
    pinned_job: Mapping[str, str] | None = None,
) -> HistoricalEvaluationPlan | None:
    """Choose one resumable historical evaluation without mutating pointers."""

    if not config.enabled:
        return None
    transitions: dict[tuple[str, str], tuple[int, int, str, str]] = {}
    existing_crossplay: dict[tuple[str, str], tuple[Path, dict[str, object]]] = {}
    for path, result in arena_results:
        # Resume snapshots carry arena identities too, but are execution state,
        # not evidence that a transition or a measurement has completed.
        if path.name.endswith(".resume.json"):
            continue
        candidate = result.get("candidate")
        baseline = result.get("baseline")
        if not isinstance(candidate, str) or not isinstance(baseline, str):
            continue
        kind = arena_result_kind(result)
        if kind == HISTORICAL_CROSSPLAY_RESULT_KIND:
            key = (candidate, baseline)
            prior = existing_crossplay.get(key)
            if prior is None or _crossplay_order(path, result) > _crossplay_order(
                *prior
            ):
                existing_crossplay[key] = (path, dict(result))
            continue
        promotion = result.get("promotion")
        decision = promotion.get("decision") if isinstance(promotion, Mapping) else None
        completed_ns = result.get("completed_ns")
        if (
            kind == PROMOTION_RESULT_KIND
            and decision == "promote"
            and type(completed_ns) is int
            and candidate in manifests
            and baseline in manifests
            and candidate != baseline
        ):
            transition = (
                completed_ns,
                manifests[candidate].model_step,
                candidate,
                baseline,
            )
            key = (candidate, baseline)
            # Repeated result records for the same transition must not change
            # its chronological position or the promotion cadence.
            transitions[key] = min(transitions.get(key, transition), transition)
    promoted = sorted(transitions.values())
    direct_predecessor = next(
        (
            baseline
            for _, _, candidate, baseline in reversed(promoted)
            if candidate == champion.model_identity
        ),
        None,
    )
    promoted_identities = {baseline for _, _, _, baseline in promoted}
    promoted_identities.update(candidate for _, _, candidate, _ in promoted)
    promotion_count = len({candidate for _, _, candidate, _ in promoted})
    if config.measurement_service_fraction > 0:
        # The protected service measures the present frontier, not an unbounded
        # backlog of bootstrap-era edges. Once admitted, a matchup remains pinned
        # across promotions and restarts until all its configured pairs finish.
        candidate_id = champion.model_identity
        baseline_id = anchor_identity or direct_predecessor
        if pinned_job is not None:
            candidate_id = pinned_job["candidate"]
            baseline_id = pinned_job["baseline"]
        if baseline_id is None or candidate_id == baseline_id:
            return None
        if candidate_id not in manifests or baseline_id not in manifests:
            raise ValueError("protected measurement model manifest is missing")
        prior = existing_crossplay.get((candidate_id, baseline_id))
        if prior is not None and prior[1].get("terminal") is True:
            return None
        return HistoricalEvaluationPlan(
            candidate=manifests[candidate_id],
            baseline=manifests[baseline_id],
            result_path=prior[0]
            if prior
            else Path(results_directory)
            / f"crossplay-{candidate_id}-vs-{baseline_id}.json",
            previous=prior[1] if prior else None,
            kind="measurement",
        )
    if config.measure_direct_predecessor:
        # Finish existing work first, then fill missing ladder links from oldest
        # to newest. Neither group is restricted to the current champion: a
        # waiting promotion may advance it before a measurement finishes.
        ordered = sorted(
            promoted,
            key=lambda item: (
                (item[2], item[3]) not in existing_crossplay,
                item,
            ),
        )
        for _, _, candidate, baseline in ordered:
            prior = existing_crossplay.get((candidate, baseline))
            if prior is not None and bool(prior[1].get("terminal")):
                continue
            name = f"crossplay-{candidate}-vs-{baseline}.json"
            return HistoricalEvaluationPlan(
                candidate=manifests[candidate],
                baseline=manifests[baseline],
                result_path=prior[0]
                if prior is not None
                else Path(results_directory) / name,
                previous=prior[1] if prior is not None else None,
                kind="measurement",
            )
    if not promotion_count or promotion_count % config.every_promotions:
        return None

    eligible = [
        manifest
        for identity, manifest in manifests.items()
        if identity in promoted_identities
        and identity not in (champion.model_identity, direct_predecessor)
        and manifest.model_step < champion.model_step
    ]
    eligible.sort(key=lambda item: (item.model_step, item.model_identity), reverse=True)
    for baseline in eligible[: config.anchors_per_evaluation]:
        key = (champion.model_identity, baseline.model_identity)
        prior = existing_crossplay.get(key)
        if prior is not None:
            path, payload = prior
            if bool(payload.get("terminal")):
                continue
            return HistoricalEvaluationPlan(
                candidate=champion,
                baseline=baseline,
                result_path=path,
                previous=payload,
            )
        name = f"crossplay-{champion.model_identity}-vs-{baseline.model_identity}.json"
        return HistoricalEvaluationPlan(
            candidate=champion,
            baseline=baseline,
            result_path=Path(results_directory) / name,
            previous=None,
            kind="anchor",
        )
    return None


def _crossplay_order(
    path: Path, result: Mapping[str, object]
) -> tuple[bool, int, int, str]:
    """Prefer terminal evidence, then the most advanced deterministic resume."""

    pairs = result.get("pairs")
    completed_ns = result.get("completed_ns")
    return (
        bool(result.get("terminal")),
        len(pairs) if isinstance(pairs, list) else 0,
        completed_ns if type(completed_ns) is int else 0,
        str(path),
    )
