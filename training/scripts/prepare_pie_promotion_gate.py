#!/usr/bin/env python3
"""Preserve admitted self-play execution through a promotion allocation change.

Original reports retain their original workload and qualification scope. This
preparer publishes immutable evidence only; it never activates training or
claims new throughput, promotion latency, or playing-strength qualification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from scripts.prepare_pie_policy_gate import _publish, _reference
from deltreltrain.config import load_config
from deltreltrain.pie_promotion import validate_pie_promotion_transition
from deltreltrain import search_allocation_gate as admission


def prepare_promotion_gate(source_path: Path, target_path: Path) -> dict[str, object]:
    source, target = load_config(source_path), load_config(target_path)
    validate_pie_promotion_transition(source, target)
    root = Path(source.orchestration.directories.root).expanduser().resolve()
    source_reference = _reference(root, source_path)
    source_gate_path = admission.allocation_gate_path(source)
    gate_path = admission.allocation_gate_path(target)
    receipt_path = gate_path.with_suffix(".promotion-transition.json")
    if any(path.exists() or path.is_symlink() for path in (gate_path, receipt_path)):
        raise FileExistsError(
            "promotion gate or receipt already exists; immutable artifacts are never overwritten"
        )
    gate_reference = _reference(root, source_gate_path)
    _, contents = admission._read_ref(root, gate_reference)
    original = admission._json(contents)
    if "promotion_allocation_transition" in original:
        raise ValueError("promotion transition receipts cannot be chained")
    admission.validate_production_ring_allocations(source, _fresh=True)
    receipt = {
        "format": admission.PROMOTION_TRANSITION_FORMAT,
        "schema_version": 1,
        "classification": admission.PROMOTION_TRANSITION_CLASS,
        "run_id": source.orchestration.run_id,
        "source_profile": source_reference,
        "source_gate": gate_reference,
        "source_config_sha256": admission.canonical_config_sha256(source),
        "target_config_sha256": admission.canonical_config_sha256(target),
        "measurement_scope": admission.PROMOTION_TRANSITION_SCOPE,
        "new_objective_performance_qualified": False,
    }
    receipt_reference = _publish(root, receipt_path, receipt)
    new_gate = dict(original)
    new_gate["target_config_sha256"] = admission.canonical_config_sha256(target)
    new_gate["promotion_allocation_transition"] = receipt_reference
    # A failure may leave an inert receipt, but cannot publish an unvalidated
    # admission gate. Every inherited report is rehashed by this validation.
    verified: dict[str, tuple[int, ...]] = {}
    admission._validate_promotion_transition_gate(root, new_gate, target, verified)
    if not admission._unchanged(root, verified):
        raise ValueError("source evidence changed before promotion gate publication")
    published = _publish(root, gate_path, new_gate)
    admission.validate_production_ring_allocations(target)
    return {
        "classification": admission.PROMOTION_TRANSITION_CLASS,
        "source_config_sha256": receipt["source_config_sha256"],
        "target_config_sha256": receipt["target_config_sha256"],
        "original_gate": gate_reference,
        "receipt": receipt_reference,
        "gate": published,
        "measurement_scope": admission.PROMOTION_TRANSITION_SCOPE,
        "new_objective_performance_qualified": False,
        "deployment_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--target-profile", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            prepare_promotion_gate(args.source_profile, args.target_profile), indent=2
        )
    )


if __name__ == "__main__":
    main()
