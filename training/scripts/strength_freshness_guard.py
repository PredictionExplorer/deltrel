"""Read-only guard-plan validation; live service execution is not exposed.

The finite core and CPU fault harness live in deltreltrain.strength_freshness_guard.
A Linux adapter, actual unit plans, probe and runtime qualification are separate
gates. This entrypoint cannot reserve GPUs or invoke systemctl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from deltreltrain.strength_freshness_guard import (
    Refusal,
    digest,
    source_sha256,
    validate_plan,
)


def validate(path: Path, checksum: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2**20:
        raise Refusal("unsafe-guard-plan")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != checksum:
        raise Refusal("guard-plan-file-sha256")
    plan = json.loads(data)
    validate_plan(plan)
    if plan["guard_source_sha256"] != source_sha256():
        raise Refusal("executing-guard-source-mismatch")
    return {
        "status": "structure-validated-no-live-adapter-qualified",
        "file_sha256": checksum,
        "canonical_plan_sha256": digest(plan),
        "attempt_id": plan["attempt_id"],
        "live_execution_available": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate-plan")
    check.add_argument("--plan", type=Path, required=True)
    check.add_argument("--sha256", required=True)
    arguments = parser.parse_args()
    print(json.dumps(validate(arguments.plan, arguments.sha256), sort_keys=True))


if __name__ == "__main__":
    main()
