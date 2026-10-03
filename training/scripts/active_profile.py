#!/usr/bin/env python3
"""Read the current profile authority and validate registered recovery metadata."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from deltreltrain.config import ExperimentConfig, load_config
from deltreltrain.search_allocation_gate import validate_production_ring_allocations
from deltreltrain.strength_recovery import completed_screen, digest, load_plan

if __package__:
    from .validate_continuous_profile import validate_continuous_config
else:
    from validate_continuous_profile import validate_continuous_config


@dataclass(frozen=True)
class ActiveProfile:
    path: Path
    contents: bytes | None
    sha256: str | None
    expected_sha256: str | None
    source: str


def _read(path: Path, *, limit: int = 1024 * 1024) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"profile metadata is missing or unsafe: {path}")
    if path.stat().st_size > limit:
        raise ValueError(f"profile metadata exceeds its size limit: {path}")
    data = path.read_bytes()
    if len(data) > limit:
        raise ValueError(f"profile metadata exceeds its size limit: {path}")
    return data


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(_read(path))
    if not isinstance(value, dict):
        raise ValueError(f"profile metadata is not an object: {path}")
    return value


def resolve_active_profile(
    root: Path,
    explicit: Path | None = None,
    *,
    allow_missing: bool = False,
) -> ActiveProfile:
    """Resolve anew on every call; a broken authority never falls back."""
    root = root.expanduser().resolve()
    if explicit is not None and explicit.expanduser().is_symlink():
        raise ValueError("explicit profile must not be a symbolic link")
    requested = explicit.expanduser().resolve() if explicit is not None else None
    authority = root / "profile.sha256"
    for _ in range(3):
        registered = authority.exists() or authority.is_symlink()
        authority_bytes = _read(authority, limit=4096) if registered else None
        expected = None
        if authority_bytes is not None:
            text = authority_bytes.decode("utf-8").strip()
            fields = text.split(maxsplit=1)
            if (
                len(text.splitlines()) != 1
                or len(fields) != 2
                or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None
            ):
                raise ValueError("active profile checksum is malformed")
            expected, name = fields
            candidate = Path(name.removeprefix("*"))
            candidate = candidate if candidate.is_absolute() else root / candidate
            if candidate.is_symlink():
                raise ValueError("registered profile must not be a symbolic link")
            path = candidate.resolve()
            if root not in path.parents:
                raise ValueError("registered profile escaped its run root")
            if requested is not None and requested != path:
                raise ValueError(
                    "explicit profile differs from the registered active profile"
                )
        else:
            path = requested or root / "profile.yaml"
        if (
            not path.exists()
            and not path.is_symlink()
            and allow_missing
            and not registered
        ):
            return ActiveProfile(path, None, None, None, "legacy_fallback")
        contents = _read(path)
        actual = hashlib.sha256(contents).hexdigest()
        if (
            authority_bytes is not None
            and _read(authority, limit=4096) != authority_bytes
        ):
            continue
        if authority_bytes is None and (authority.exists() or authority.is_symlink()):
            continue
        if expected is not None and actual != expected:
            raise ValueError("active profile disagrees with its registered checksum")
        return ActiveProfile(
            path,
            contents,
            actual,
            expected,
            "registered_profile"
            if registered
            else "explicit_profile"
            if requested
            else "legacy_fallback",
        )
    raise ValueError("active profile authority changed repeatedly while reading")


def _profile_pin(path: Path, pin: dict[str, Any]) -> bytes:
    contents = _read(path)
    if len(contents) != pin.get("bytes") or hashlib.sha256(
        contents
    ).hexdigest() != pin.get("sha256"):
        raise ValueError(f"registered recovery profile pin differs: {path}")
    return contents


def _screen_registration(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], ExperimentConfig]:
    plan = load_plan(root)
    if plan is None:
        raise ValueError("recovery profile lacks its registration")
    if (
        plan.get("classification")
        != "coupled-champion-restart-and-higher-search-quality-screen"
        or plan.get("wall_budget_seconds") != 43200
        or plan.get("initial_replay_credit") != 0
    ):
        raise ValueError("unsupported recovery experiment registration")
    profile = root / "profile-elo-ablation.yaml"
    contents = _profile_pin(profile, plan["profile"])
    if hashlib.sha256(contents).hexdigest() != plan["profile_sha256"]:
        raise ValueError("recovery plan has inconsistent profile digests")
    config = load_config(profile)
    if _read(profile) != contents:
        raise ValueError("recovery profile changed during typed validation")
    run = _json(root / "run.json")
    metadata = _json(root / "ablation.json")
    anchor = metadata.get("anchor", {})
    if (
        metadata.get("report") != "deltreltrain-elo-ablation-branch"
        or metadata.get("treatment") != "strength-recovery"
        or metadata.get("profile_sha256") != plan["profile_sha256"]
        or Path(metadata.get("profile", "")).resolve() != profile
        or metadata.get("source_run_id") != run.get("run_id")
        or metadata.get("source_generation_family") != run.get("generation_family")
        or config.orchestration.run_id != run.get("run_id")
        or Path(config.orchestration.directories.root).resolve() != root
        or not isinstance(anchor, dict)
        or anchor.get("model_identity") != plan.get("anchor_identity")
        or anchor.get("model_step") != plan.get("anchor_step")
        or config.orchestration.training_objective != "ring10_pie"
        or config.orchestration.model_refresh.selfplay_source != "champion"
        or config.learner.target_updates_per_new_sample != 1.5
        or config.learner.minimum_replay_shard_id_exclusive
        != plan.get("replay_watermark")
    ):
        raise ValueError(
            "recovery profile disagrees with its registered run/experiment"
        )
    # This function intentionally validates only small control metadata and
    # profile bytes. Full model/replay/provenance integrity is the backup and
    # runtime admission authority, not a multi-GiB task every five seconds.
    return plan, metadata, config


def validate_profile_for_monitor(
    root: Path, selected: ActiveProfile
) -> tuple[ExperimentConfig, str]:
    root = root.resolve()
    config = load_config(selected.path)
    if selected.contents is None or _read(selected.path) != selected.contents:
        raise ValueError("active profile changed during typed validation")
    registration = root / "strength-recovery-plan.json"
    if not registration.exists() and not registration.is_symlink():
        validate_continuous_config(config)
        return config, "continuous_profile"
    if selected.expected_sha256 is None:
        raise ValueError(
            "registered recovery requires active profile checksum authority"
        )
    plan, metadata, original = _screen_registration(root)
    if selected.path == root / "profile-elo-ablation.yaml":
        if selected.sha256 != plan["profile_sha256"] or config != original:
            raise ValueError("active screen profile differs from its registration")
        classification = "registered_recovery_screen"
    elif selected.path == root / "profile-strength-freshness.yaml":
        from deltreltrain.strength_freshness import validate_installed

        freshness, expected = validate_installed(root)
        if selected.sha256 != freshness["target_profile"]["sha256"] or digest(
            config.as_dict()
        ) != digest(expected.as_dict()):
            raise ValueError("active freshness profile differs from its registration")
        classification = "registered_champion_only_freshness"
    else:
        continuation = _json(root / "strength-continuation-plan.json")
        if (
            continuation.get("format") != "deltreltrain.strength-continuation"
            or continuation.get("schema_version") != 1
            or continuation.get("plan_sha256")
            != digest({k: v for k, v in continuation.items() if k != "plan_sha256"})
            or continuation.get("recovery_plan_sha256") != plan["plan_sha256"]
            or continuation.get("target_profile_name")
            != "profile-strength-continuation.yaml"
            or selected.path != root / "profile-strength-continuation.yaml"
        ):
            raise ValueError("active continuation lacks an exact registered plan")
        target_pin = continuation["target_profile"]
        if selected.sha256 != target_pin.get("sha256") or len(
            selected.contents
        ) != target_pin.get("bytes"):
            raise ValueError("active continuation profile differs from its pin")
        target_input = Path(target_pin["path"])
        if target_input.resolve() != root / "strength-continuation-input.yaml":
            raise ValueError(
                "continuation input profile escaped its registered location"
            )
        if _profile_pin(target_input, target_pin) != selected.contents:
            raise ValueError("installed continuation differs from its prepared input")
        expected = replace(
            original,
            learner=replace(original.learner, candidate_interval_examples=3_000_000),
        )
        if config != expected:
            raise ValueError(
                "continuation changes more than the registered candidate cadence"
            )
        completed_screen(root)
        seal = _json(root / "strength-screen-seal.json")
        if (
            seal.get("sha256")
            != digest({k: v for k, v in seal.items() if k != "sha256"})
            or seal.get("continuation_plan_sha256") != continuation["plan_sha256"]
            or seal.get("recovery_plan_sha256") != plan["plan_sha256"]
            or seal.get("ablation") != metadata
            or seal.get("ablation_sha256")
            != hashlib.sha256(_read(root / "ablation.json")).hexdigest()
        ):
            raise ValueError("continuation does not match the sealed completed screen")
        migrations = _read(root / "continuous-migrations.jsonl").splitlines()
        head = json.loads(migrations[-1]) if migrations else {}
        if (
            head.get("from_profile_sha256") != plan["profile_sha256"]
            or head.get("to_profile_sha256") != selected.sha256
            or head.get("to_profile") != selected.path.name
            or head.get("to_source_commit") != continuation.get("source_commit")
            or head.get("run_id") != config.orchestration.run_id
        ):
            raise ValueError(
                "continuation migration receipt does not match the active profile"
            )
        classification = "registered_recovery_continuation"
    validate_production_ring_allocations(config, _fresh=True)
    current = resolve_active_profile(root)
    if current.path != selected.path or current.contents != selected.contents:
        raise ValueError("active profile changed during recovery admission")
    return config, classification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    selected = resolve_active_profile(args.run_root, args.profile)
    admission = (
        validate_profile_for_monitor(args.run_root, selected)[1]
        if args.validate
        else None
    )
    print(
        json.dumps(
            {
                "path": str(selected.path),
                "sha256": selected.sha256,
                "source": selected.source,
                "admission": admission,
            }
        )
        if args.json
        else selected.path
    )


if __name__ == "__main__":
    main()
