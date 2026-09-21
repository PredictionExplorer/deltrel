#!/usr/bin/env python3
"""Prepare an immutable continuation of existing search admission; never activate.

The new objective is prospective. Original qualification reports retain their
original workload and scope; this command does not qualify new throughput or Elo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

from deltreltrain.config import load_config
from deltreltrain.pie_policy import validate_pie_policy_transition
from deltreltrain import search_allocation_gate as admission


def _reference(root: Path, path: Path) -> dict[str, str]:
    relative = str(path.absolute().relative_to(root))
    safe = admission._safe_path(root, relative)
    reference = {
        "path": relative,
        "sha256": hashlib.sha256(safe.read_bytes()).hexdigest(),
    }
    admission._read_ref(root, reference)
    return reference


def _publish(root: Path, path: Path, payload: dict) -> dict[str, str]:
    """Publish a complete read-only artifact without replacing any existing file."""
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=".policy-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def prepare_policy_gate(source_path: Path, target_path: Path) -> dict[str, object]:
    source, target = load_config(source_path), load_config(target_path)
    validate_pie_policy_transition(source, target)
    root = Path(source.orchestration.directories.root).expanduser().resolve()
    source_reference = _reference(root, source_path)
    source_gate_path = admission.allocation_gate_path(source)
    gate_path = admission.allocation_gate_path(target)
    receipt_path = gate_path.with_suffix(".policy-transition.json")
    if gate_path.exists() or receipt_path.exists():
        raise FileExistsError(
            "policy gate or receipt already exists; immutable artifacts are never overwritten"
        )
    admission.validate_production_ring_allocations(source)
    gate_reference = _reference(root, source_gate_path)
    _, contents = admission._read_ref(root, gate_reference)
    original = admission._json(contents)
    if "training_policy_transition" in original:
        raise ValueError("policy transition receipts cannot be chained")
    receipt = {
        "format": admission.POLICY_TRANSITION_FORMAT,
        "schema_version": 1,
        "classification": admission.POLICY_TRANSITION_CLASS,
        "run_id": source.orchestration.run_id,
        "source_profile": source_reference,
        "source_gate": gate_reference,
        "source_config_sha256": admission.canonical_config_sha256(source),
        "target_config_sha256": admission.canonical_config_sha256(target),
        "measurement_scope": admission.POLICY_TRANSITION_SCOPE,
        "new_objective_performance_qualified": False,
    }
    receipt_reference = _publish(root, receipt_path, receipt)
    new_gate = dict(original)
    new_gate["target_config_sha256"] = admission.canonical_config_sha256(target)
    new_gate["training_policy_transition"] = receipt_reference
    # Validate the full pinned closure before the target admission file appears.
    # A failure may leave an inert receipt; it cannot grant startup admission.
    verified: dict[str, tuple[int, ...]] = {}
    admission._validate_policy_transition_gate(root, new_gate, target, verified)
    if not admission._unchanged(root, verified):
        raise ValueError("source evidence changed before policy gate publication")
    published_gate = _publish(root, gate_path, new_gate)
    admission.validate_production_ring_allocations(target)
    return {
        "classification": admission.POLICY_TRANSITION_CLASS,
        "source_config_sha256": receipt["source_config_sha256"],
        "target_config_sha256": receipt["target_config_sha256"],
        "original_gate": gate_reference,
        "receipt": receipt_reference,
        "gate": published_gate,
        "measurement_scope": admission.POLICY_TRANSITION_SCOPE,
        "new_objective_performance_qualified": False,
        "deployment_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--target-profile", type=Path, required=True)
    arguments = parser.parse_args(argv)
    print(
        json.dumps(
            prepare_policy_gate(arguments.source_profile, arguments.target_profile),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
