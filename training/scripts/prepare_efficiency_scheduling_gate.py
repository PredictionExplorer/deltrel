#!/usr/bin/env python3
"""Publish immutable scheduling-only admission evidence without activating it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from scripts.prepare_pie_policy_gate import _publish, _reference
from deltreltrain.config import load_config
from deltreltrain.efficiency_scheduling import validate_efficiency_scheduling_transition
from deltreltrain import search_allocation_gate as admission


def prepare_scheduling_gate(
    source_path: Path, target_path: Path, *, resume_receipt: bool = False
) -> dict[str, object]:
    if type(resume_receipt) is not bool:
        raise TypeError("resume_receipt must be boolean")
    source, target = load_config(source_path), load_config(target_path)
    validate_efficiency_scheduling_transition(source, target)
    root = Path(source.orchestration.directories.root).expanduser().resolve()
    source_reference = _reference(root, source_path)
    source_gate_path = admission.allocation_gate_path(source)
    gate_path = admission.allocation_gate_path(target)
    receipt_path = gate_path.with_suffix(".scheduling-transition.json")
    if (
        gate_path.exists()
        or gate_path.is_symlink()
        or (not resume_receipt and (receipt_path.exists() or receipt_path.is_symlink()))
    ):
        raise FileExistsError(
            "scheduling gate or receipt already exists; never overwrite"
        )
    gate_reference = _reference(root, source_gate_path)
    _, contents = admission._read_ref(root, gate_reference)
    original = admission._json(contents)
    if "efficiency_scheduling_transition" in original:
        raise ValueError("scheduling transition receipts cannot be chained")
    admission.validate_production_ring_allocations(source, _fresh=True)
    receipt = {
        "format": admission.SCHEDULING_TRANSITION_FORMAT,
        "schema_version": 1,
        "classification": admission.SCHEDULING_TRANSITION_CLASS,
        "run_id": source.orchestration.run_id,
        "source_profile": source_reference,
        "source_gate": gate_reference,
        "source_config_sha256": admission.canonical_config_sha256(source),
        "target_config_sha256": admission.canonical_config_sha256(target),
        "measurement_scope": admission.SCHEDULING_TRANSITION_SCOPE,
        "new_objective_performance_qualified": False,
    }
    if resume_receipt and (receipt_path.exists() or receipt_path.is_symlink()):
        reference = _reference(root, receipt_path)
        if admission._json(admission._read_ref(root, reference)[1]) != receipt:
            raise ValueError(
                "existing scheduling receipt differs from requested authority"
            )
    else:
        reference = _publish(root, receipt_path, receipt)
    new_gate = {
        **original,
        "target_config_sha256": admission.canonical_config_sha256(target),
        "efficiency_scheduling_transition": reference,
    }
    verified: dict[str, tuple[int, ...]] = {}
    admission._validate_scheduling_transition_gate(root, new_gate, target, verified)
    if not admission._unchanged(root, verified):
        raise ValueError("source evidence changed before scheduling gate publication")
    published = _publish(root, gate_path, new_gate)
    admission.validate_production_ring_allocations(target, _fresh=True)
    return {
        **receipt,
        "receipt": reference,
        "gate": published,
        "deployment_performed": False,
    }


def ensure_scheduling_gate(source_path: Path, target_path: Path) -> None:
    """Resume publication using exact immutable evidence, never overwrite it."""
    source, target = load_config(source_path), load_config(target_path)
    validate_efficiency_scheduling_transition(source, target)
    if not any(
        row.full_probability < admission.FULL_PROBABILITY_FLOOR
        for row in target.selfplay.ring_search_allocations
    ):
        return
    path = admission.allocation_gate_path(target)
    if path.exists() or path.is_symlink():
        admission.validate_production_ring_allocations(target, _fresh=True)
    else:
        prepare_scheduling_gate(source_path, target_path, resume_receipt=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--target-profile", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            prepare_scheduling_gate(args.source_profile, args.target_profile), indent=2
        )
    )


if __name__ == "__main__":
    main()
