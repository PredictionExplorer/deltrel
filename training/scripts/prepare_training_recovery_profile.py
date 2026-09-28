#!/usr/bin/env python3
"""Prepare one immutable recovery treatment and its inherited search evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import yaml

from scripts.prepare_pie_policy_gate import _publish, _reference
from startrain import search_allocation_gate as admission
from startrain.config import load_config
from startrain.training_recovery_policy import (
    protected_champion_config,
    reuse_config,
    validate_training_recovery_transition,
)


def prepare_recovery_gate(source_path: Path, target_path: Path) -> dict[str, object]:
    source, target = load_config(source_path), load_config(target_path)
    treatment = validate_training_recovery_transition(source, target)
    root = Path(source.orchestration.directories.root).expanduser().resolve()
    admission.validate_production_ring_allocations(source, _fresh=True)
    if not any(
        row.full_probability < admission.FULL_PROBABILITY_FLOOR
        for row in target.selfplay.ring_search_allocations
    ):
        return {"treatment": treatment, "search_exception_required": False}
    source_reference = _reference(root, source_path)
    gate_reference = _reference(root, admission.allocation_gate_path(source))
    original = admission._json(admission._read_ref(root, gate_reference)[1])
    gate_path = admission.allocation_gate_path(target)
    receipt_path = gate_path.with_suffix(".recovery-transition.json")
    receipt = {
        "format": admission.RECOVERY_TRANSITION_FORMAT,
        "schema_version": 1,
        "classification": admission.RECOVERY_TRANSITION_CLASS,
        "treatment": treatment,
        "run_id": source.orchestration.run_id,
        "source_profile": source_reference,
        "source_gate": gate_reference,
        "source_config_sha256": admission.canonical_config_sha256(source),
        "target_config_sha256": admission.canonical_config_sha256(target),
        "measurement_scope": admission.RECOVERY_TRANSITION_SCOPE,
        "new_objective_performance_qualified": False,
    }
    if receipt_path.exists() or receipt_path.is_symlink():
        reference = _reference(root, receipt_path)
        if admission._json(admission._read_ref(root, reference)[1]) != receipt:
            raise ValueError(
                "existing recovery admission differs from requested treatment"
            )
    else:
        reference = _publish(root, receipt_path, receipt)
    gate = {
        **original,
        "target_config_sha256": admission.canonical_config_sha256(target),
        "training_recovery_transition": reference,
    }
    verified: dict[str, tuple[int, ...]] = {}
    admission._validate_training_recovery_transition_gate(root, gate, target, verified)
    if not admission._unchanged(root, verified):
        raise ValueError("source evidence changed while preparing recovery treatment")
    if gate_path.exists() or gate_path.is_symlink():
        if (
            admission._json(admission._read_ref(root, _reference(root, gate_path))[1])
            != gate
        ):
            raise ValueError("existing recovery gate differs")
    else:
        _publish(root, gate_path, gate)
    admission.validate_production_ring_allocations(target, _fresh=True)
    return {**receipt, "receipt": reference, "gate": _reference(root, gate_path)}


def prepare_profile(
    source_path: Path,
    output_path: Path,
    *,
    treatment: str,
    activation_ns: int | None = None,
    fraction: float = 0.25,
    target_reuse: float = 2.0,
) -> dict[str, object]:
    source = load_config(source_path)
    if treatment == "champion":
        target = protected_champion_config(
            source,
            fraction=fraction,
            after_ns=(activation_ns if activation_ns is not None else time.time_ns()),
        )
    elif treatment == "reuse":
        target = reuse_config(source, target=target_reuse)
    else:
        raise ValueError("unknown recovery treatment")
    validate_training_recovery_transition(source, target)
    contents = yaml.safe_dump(json.loads(json.dumps(target.as_dict())), sort_keys=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or output_path.is_symlink():
        if output_path.is_symlink() or output_path.read_text() != contents:
            raise ValueError("existing recovery profile differs; never overwrite it")
        return {
            "profile": str(output_path.resolve()),
            "admission": prepare_recovery_gate(source_path, output_path),
            "deployment_performed": False,
            "strength_improvement_established": False,
        }
    descriptor, name = tempfile.mkstemp(
        prefix=".recovery-profile-", dir=output_path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(contents)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        if load_config(temporary) != target:
            raise ValueError("recovery profile failed round-trip validation")
        os.link(temporary, output_path, follow_symlinks=False)
        directory = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "profile": str(output_path.resolve()),
        "admission": prepare_recovery_gate(source_path, output_path),
        "deployment_performed": False,
        "strength_improvement_established": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--treatment", choices=("champion", "reuse"), required=True)
    parser.add_argument("--activation-ns", type=int)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--target-reuse", type=float, default=2.0)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_profile(
                args.source_profile,
                args.output,
                treatment=args.treatment,
                activation_ns=args.activation_ns,
                fraction=args.fraction,
                target_reuse=args.target_reuse,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
