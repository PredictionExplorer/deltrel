#!/usr/bin/env python3
"""Prepare a frozen, conservative efficiency profile without activating it."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from collections.abc import Sequence
import tempfile

import yaml

from deltreltrain.config import load_config
from deltreltrain.efficiency_scheduling import efficiency_scheduling_config


def prepare_profile(source: Path, output: Path) -> dict[str, object]:
    source = source.expanduser().resolve(strict=True)
    output = output.expanduser().absolute()
    if source == output or output.exists() or output.is_symlink():
        raise FileExistsError("a prepared profile never overwrites an existing file")
    source_bytes = source.read_bytes()
    original = load_config(source)
    if original.orchestration.autonomous.enabled:
        raise ValueError("existing autonomous profiles cannot change their policy")
    historical = original.orchestration.historical_evaluation
    if not historical.enabled or not historical.measure_direct_predecessor:
        raise ValueError(
            "protected service requires configured independent measurement"
        )
    payload = deepcopy(yaml.safe_load(source_bytes))
    payload["orchestration"]["historical_evaluation"].update(
        measurement_service_fraction=0.2, measurement_max_wait_seconds=3600.0
    )
    payload["orchestration"]["model_refresh"].update(
        history_horizon_enabled=True, history_horizon_initial_seconds=3600.0
    )
    proposed = efficiency_scheduling_config(original)
    encoded = yaml.safe_dump(payload, sort_keys=False).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        if load_config(temporary) != proposed:
            raise ValueError("prepared efficiency profile failed round-trip validation")
        os.link(temporary, output, follow_symlinks=False)
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema_version": 1,
        "source_profile": str(source),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "prepared_profile": str(output),
        "prepared_sha256": hashlib.sha256(encoded).hexdigest(),
        "measurement_service_fraction": 0.2,
        "history_horizon_enabled": True,
        "deployment_performed": False,
        "model_optimizer_and_search_changed": False,
        "admission_receipt_required": True,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(prepare_profile(args.source_profile, args.output), indent=2))


if __name__ == "__main__":
    main()
