#!/usr/bin/env python3
"""Prepare an immutable live-replay profile without activating it."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Sequence

import yaml

from deltreltrain.config import ActorWorkSchedulingConfig, ExperimentConfig, load_config
from deltreltrain.selfplay import PolicyPublicationConfig


class ProfilePreparationError(ValueError):
    """The requested candidate cannot preserve its source controls safely."""


ALLOWED_CHANGES = frozenset(
    {
        "selfplay.policy_publication.enabled",
        "selfplay.policy_publication.first_decisions",
        "selfplay.policy_publication.interval_decisions",
        "learner.replay_refresh_seconds",
        "orchestration.model_refresh.work_scheduling.enabled",
        "orchestration.model_refresh.work_scheduling.games_per_lease",
        "orchestration.model_refresh.work_scheduling.coverage_first",
    }
)


def _checked_path(path: str | Path) -> Path:
    absolute = Path(path).expanduser().absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise ProfilePreparationError(
                f"symbolic links are not allowed: {component}"
            )
    return absolute.resolve(strict=False)


def _changes(before: Any, after: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(before, dict) and isinstance(after, dict):
        if before.keys() != after.keys():
            raise ProfilePreparationError("candidate changed the configuration fields")
        return [
            change
            for key in sorted(before)
            for change in _changes(
                before[key], after[key], f"{prefix}.{key}" if prefix else key
            )
        ]
    if type(before) is type(after) and before == after:
        return []
    return [{"path": prefix, "from": before, "to": after}]


def _config_sha(config: ExperimentConfig) -> str:
    serialized = json.dumps(
        config.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _candidate(
    source: ExperimentConfig,
    *,
    refresh_seconds: float,
    first_decisions: int,
    interval_decisions: int,
    shared_scheduling: bool,
) -> ExperimentConfig:
    if type(shared_scheduling) is not bool:
        raise ProfilePreparationError("shared_scheduling must be boolean")
    if source.selfplay.record_fast_policy_targets is not True:
        raise ProfilePreparationError(
            "source selfplay.record_fast_policy_targets must already be true; "
            "validate and enable fast policy targets explicitly before preparing live replay"
        )
    orchestration = source.orchestration
    if shared_scheduling:
        refresh = orchestration.model_refresh
        if refresh.work_scheduling.games_per_lease is not None:
            raise ProfilePreparationError(
                "source has an explicit work_scheduling.games_per_lease override; "
                "use --no-shared-scheduling to preserve it"
            )
        if (
            not orchestration.enabled
            or orchestration.device != "cuda"
            or not orchestration.actor_gpus
            or not refresh.inference.shared_batching
        ):
            raise ProfilePreparationError(
                "shared scheduling requires existing enabled shared CUDA actor inference; "
                "use --no-shared-scheduling for a CPU or simple profile"
            )
        try:
            orchestration = replace(
                orchestration,
                model_refresh=replace(
                    refresh,
                    work_scheduling=ActorWorkSchedulingConfig(
                        enabled=True, games_per_lease=None, coverage_first=True
                    ),
                ),
            )
        except ValueError as error:
            raise ProfilePreparationError(
                "source actor topology is not ready for shared scheduling: "
                f"{error}; use --no-shared-scheduling or validate compatible "
                "shared actor cohorts separately"
            ) from error
    try:
        return replace(
            source,
            selfplay=replace(
                source.selfplay,
                policy_publication=PolicyPublicationConfig(
                    enabled=True,
                    first_decisions=first_decisions,
                    interval_decisions=interval_decisions,
                ),
            ),
            learner=replace(source.learner, replay_refresh_seconds=refresh_seconds),
            orchestration=orchestration,
        )
    except ValueError as error:
        raise ProfilePreparationError(str(error)) from error


def prepare_profile(
    source: str | Path,
    output: str | Path,
    *,
    refresh_seconds: float = 60.0,
    first_decisions: int = 8,
    interval_decisions: int = 32,
    shared_scheduling: bool = True,
) -> dict[str, Any]:
    source_path, output_path = _checked_path(source), _checked_path(output)
    if not source_path.is_file() or not stat.S_ISREG(source_path.stat().st_mode):
        raise ProfilePreparationError(
            f"source must be a regular profile: {source_path}"
        )
    if source_path == output_path or os.path.lexists(output_path):
        raise ProfilePreparationError(
            f"output already exists; choose a new path: {output_path}"
        )
    if not output_path.parent.is_dir():
        raise ProfilePreparationError(
            f"output directory must already exist: {output_path.parent}"
        )
    source_bytes = source_path.read_bytes()
    source_identity = source_path.stat()
    original = load_config(source_path)
    candidate = _candidate(
        original,
        refresh_seconds=refresh_seconds,
        first_decisions=first_decisions,
        interval_decisions=interval_decisions,
        shared_scheduling=shared_scheduling,
    )
    changes = _changes(original.as_dict(), candidate.as_dict())
    if any(change["path"] not in ALLOWED_CHANGES for change in changes):
        raise ProfilePreparationError("candidate changed an unrelated control")
    source_utd = original.learner.target_updates_per_new_sample
    if candidate.learner.target_updates_per_new_sample != source_utd:
        raise ProfilePreparationError("candidate changed update-to-data allowance")
    data = yaml.safe_dump(candidate.as_dict(), sort_keys=False).encode()
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        if load_config(temporary) != candidate:
            raise ProfilePreparationError(
                "serialized candidate failed round-trip validation"
            )
        current = source_path.stat()
        if (current.st_dev, current.st_ino, current.st_mode) != (
            source_identity.st_dev,
            source_identity.st_ino,
            source_identity.st_mode,
        ) or source_path.read_bytes() != source_bytes:
            raise ProfilePreparationError(
                "source changed while preparing the candidate"
            )
        if _checked_path(output_path) != output_path:
            raise ProfilePreparationError("output path changed during preparation")
        # link() creates the destination only if absent, including under a race;
        # rename()/replace() could overwrite another writer's candidate.
        os.link(temporary, output_path, follow_symlinks=False)
        directory = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "schema_version": 1,
        "status": "prepared",
        "activated": False,
        "source": {
            "path": str(source_path),
            "profile_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "config_sha256": _config_sha(original),
        },
        "target": {
            "path": str(output_path),
            "profile_sha256": hashlib.sha256(data).hexdigest(),
            "config_sha256": _config_sha(candidate),
            "mode": "0444",
        },
        "changes": changes,
        "unchanged_utd": {
            "target_updates_per_new_sample": source_utd,
            "changed": False,
        },
        "replay_compatibility": {
            "default_manifest_schema_version": 5,
            "first_game_revision_manifest_schema_version": 6,
            "upgrade_trigger": "first successful append_game_revision commit",
            "rollback_requires": "schema-6-capable readers after the first revision, even with publication disabled",
            "position_credit": "prefix growth only; final enrichment does not re-credit published positions",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--refresh-seconds", type=float, default=60.0)
    parser.add_argument("--first-decisions", type=int, default=8)
    parser.add_argument("--interval-decisions", type=int, default=32)
    parser.add_argument("--no-shared-scheduling", action="store_true")
    args = parser.parse_args(argv)
    try:
        receipt = prepare_profile(
            args.source,
            args.output,
            refresh_seconds=args.refresh_seconds,
            first_decisions=args.first_decisions,
            interval_decisions=args.interval_decisions,
            shared_scheduling=not args.no_shared_scheduling,
        )
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "error", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
