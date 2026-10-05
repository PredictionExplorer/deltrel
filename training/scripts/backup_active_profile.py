#!/usr/bin/env python3
"""Snapshot a run using its current, checksum-verified profile authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__:
    from .active_profile import resolve_active_profile
    from .training_disaster_recovery import DisasterRecoveryError, create_snapshot
else:
    from active_profile import resolve_active_profile
    from training_disaster_recovery import DisasterRecoveryError, create_snapshot


def backup_active_profile(
    run_root: Path,
    backup_root: Path,
    *,
    expected_backup_mount: Path,
    replay_backup_retain: int = 3,
) -> Path:
    """Resolve per invocation; capture's state fence rejects a concurrent handoff.

    A changed or invalid registration must fail the snapshot, never silently
    reuse profile.yaml. The next scheduled invocation resolves authority again.
    All payload verification and atomic publication remain in create_snapshot.
    """
    selected = resolve_active_profile(run_root)
    return create_snapshot(
        run_root,
        selected.path,
        backup_root,
        expected_backup_mount=expected_backup_mount,
        replay_backup_retain=replay_backup_retain,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--expected-backup-mount", type=Path, required=True)
    parser.add_argument("--replay-backup-retain", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        snapshot = backup_active_profile(
            args.run_root,
            args.backup_root,
            expected_backup_mount=args.expected_backup_mount,
            replay_backup_retain=args.replay_backup_retain,
        )
    except (DisasterRecoveryError, OSError, ValueError) as error:
        print(json.dumps({"status": "error", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps({"status": "ok", "snapshot": str(snapshot)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
