#!/usr/bin/env python3
"""Prepare additive prediction heads and immutable admission; never activate a run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

import yaml

from scripts.prepare_pie_policy_gate import _publish, _reference
from deltreltrain import search_allocation_gate as admission
from deltreltrain.auxiliary_policy import (
    auxiliary_training_config,
    validate_auxiliary_prediction_transition,
)
from deltreltrain.config import load_config


def prepare_auxiliary_gate(
    source_path: Path, target_path: Path, *, resume_receipt: bool = False
) -> dict[str, object]:
    if type(resume_receipt) is not bool:
        raise TypeError("resume_receipt must be boolean")
    source, target = load_config(source_path), load_config(target_path)
    validate_auxiliary_prediction_transition(source, target)
    root = Path(source.orchestration.directories.root).expanduser().resolve()
    source_reference = _reference(root, source_path)
    source_gate_path = admission.allocation_gate_path(source)
    gate_path = admission.allocation_gate_path(target)
    receipt_path = gate_path.with_suffix(".auxiliary-transition.json")
    if (
        gate_path.exists()
        or gate_path.is_symlink()
        or (not resume_receipt and (receipt_path.exists() or receipt_path.is_symlink()))
    ):
        raise FileExistsError(
            "auxiliary gate or receipt already exists; immutable artifacts are never overwritten"
        )
    gate_reference = _reference(root, source_gate_path)
    _, contents = admission._read_ref(root, gate_reference)
    original = admission._json(contents)
    if "auxiliary_prediction_transition" in original:
        raise ValueError("auxiliary transition receipts cannot be chained")
    admission.validate_production_ring_allocations(source, _fresh=True)
    receipt = {
        "format": admission.AUXILIARY_TRANSITION_FORMAT,
        "schema_version": 1,
        "classification": admission.AUXILIARY_TRANSITION_CLASS,
        "run_id": source.orchestration.run_id,
        "source_profile": source_reference,
        "source_gate": gate_reference,
        "source_config_sha256": admission.canonical_config_sha256(source),
        "target_config_sha256": admission.canonical_config_sha256(target),
        "measurement_scope": admission.AUXILIARY_TRANSITION_SCOPE,
        "new_objective_performance_qualified": False,
    }
    if resume_receipt and (receipt_path.exists() or receipt_path.is_symlink()):
        receipt_reference = _reference(root, receipt_path)
        if admission._json(admission._read_ref(root, receipt_reference)[1]) != receipt:
            raise ValueError(
                "existing auxiliary receipt differs from requested authority"
            )
    else:
        receipt_reference = _publish(root, receipt_path, receipt)
    new_gate = dict(original)
    new_gate["target_config_sha256"] = admission.canonical_config_sha256(target)
    new_gate["auxiliary_prediction_transition"] = receipt_reference
    verified: dict[str, tuple[int, ...]] = {}
    admission._validate_auxiliary_transition_gate(root, new_gate, target, verified)
    if not admission._unchanged(root, verified):
        raise ValueError("source evidence changed before auxiliary gate publication")
    published = _publish(root, gate_path, new_gate)
    admission.validate_production_ring_allocations(target, _fresh=True)
    return {
        "classification": admission.AUXILIARY_TRANSITION_CLASS,
        "source_config_sha256": receipt["source_config_sha256"],
        "target_config_sha256": receipt["target_config_sha256"],
        "original_gate": gate_reference,
        "receipt": receipt_reference,
        "gate": published,
        "measurement_scope": admission.AUXILIARY_TRANSITION_SCOPE,
        "new_objective_performance_qualified": False,
        "deployment_performed": False,
    }


def ensure_auxiliary_gate(source_path: Path, target_path: Path) -> None:
    """Recovery may reuse a fully verified immutable gate, never rewrite it."""
    source, target = load_config(source_path), load_config(target_path)
    validate_auxiliary_prediction_transition(source, target)
    if not any(
        row.full_probability < admission.FULL_PROBABILITY_FLOOR
        for row in target.selfplay.ring_search_allocations
    ):
        return
    gate_path = admission.allocation_gate_path(target)
    if gate_path.exists() or gate_path.is_symlink():
        admission.validate_production_ring_allocations(target, _fresh=True)
    else:
        prepare_auxiliary_gate(source_path, target_path, resume_receipt=True)


def prepare_profile(
    source_path: Path, output_path: Path, *, recovery: bool = False
) -> dict[str, object]:
    source = load_config(source_path)
    target = auxiliary_training_config(source, recovery=recovery)
    encoded = (
        "# Additive auxiliary predictions; official clinch completion supplies final labels.\n"
        "# Prepared only; activation requires a stopped-run profile migration.\n"
        + yaml.safe_dump(json.loads(json.dumps(target.as_dict())), sort_keys=False)
    ).encode()
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        if load_config(temporary) != target:
            raise ValueError("prepared auxiliary profile failed round-trip validation")
        os.link(temporary, output_path, follow_symlinks=False)
        directory = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    ensure_auxiliary_gate(source_path, output_path)
    return {
        "source": str(source_path.resolve()),
        "source_config_sha256": admission.canonical_config_sha256(source),
        "profile": str(output_path.resolve()),
        "profile_sha256": hashlib.sha256(encoded).hexdigest(),
        "target_config_sha256": admission.canonical_config_sha256(target),
        "recovery": recovery,
        "deployment_performed": False,
        "new_objective_performance_qualified": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            prepare_profile(args.source_profile, args.output, recovery=args.recovery),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
