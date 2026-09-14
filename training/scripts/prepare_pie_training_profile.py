#!/usr/bin/env python3
"""Prepare the pie-even training profile locally; never activate a run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

import yaml

from startrain.config import load_config
from startrain.pie_policy import pie_training_config
from startrain.pie_promotion import pie_promotion_config


def prepare_profile(
    source_path: Path, output_path: Path, *, adaptive_promotion: bool = False
) -> dict[str, object]:
    source = load_config(source_path)
    candidate = pie_training_config(source)
    if adaptive_promotion:
        candidate = pie_promotion_config(candidate)
    # JSON converts dataclass tuples to plain YAML sequences without Python tags.
    payload = json.loads(json.dumps(candidate.as_dict(), allow_nan=False))
    encoded = (
        "# Pie is standard for even games. Handicap: ring 10 only, both modes.\n"
        "# Target training mix: 90% even / 10% handicap across all boards.\n"
        "# Prepared only; activation requires a stopped-run profile migration.\n"
        + yaml.safe_dump(payload, sort_keys=False)
    )
    # Publish only a complete, verified file. link() is an atomic no-overwrite
    # operation even if another preparer creates the same destination first.
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        if load_config(temporary) != candidate:
            raise ValueError("prepared profile failed round-trip validation")
        os.link(temporary, output_path, follow_symlinks=False)
        directory = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "source": str(source_path.resolve()),
        "source_config_sha256": hashlib.sha256(
            json.dumps(source.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "profile": str(output_path.resolve()),
        "profile_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "training_objective": candidate.orchestration.training_objective,
        "promotion_allocation": candidate.arena.allocation_policy,
        "deployment_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--adaptive-promotion",
        action="store_true",
        help="Prepare pie-heavy promotion with adaptive, bounded handicap checks.",
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            prepare_profile(
                args.config, args.output, adaptive_promotion=args.adaptive_promotion
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
