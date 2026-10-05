"""Pure blueprint finalization with real proof readers and synthetic observations."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import strength_freshness_cpu_template_finalization as f
from test_strength_freshness_cpu_capture_request import admitted as source_admitted
from test_strength_freshness_cpu_capture_request import (
    observations as source_observations,
)

admitted = source_admitted
observations = source_observations


@pytest.fixture
def parts(admitted):
    args, files, plan, anchor, *_ = admitted
    original = copy.deepcopy(plan)
    c = f.completion
    proof_pin = original["preservation"]["before_execution"]
    value = c.parse(files[proof_pin["path"]])
    blueprint_plan = copy.deepcopy(original)
    for name in f.DYNAMIC:
        blueprint_plan["preservation"][name].update(sha256=f.EMPTY_SHA, bytes=1)
    blueprint = {
        "format": f.FORMAT,
        "schema_version": 1,
        "plan": blueprint_plan,
        "before_launch": value["artifacts"]["launch"],
        "before_registration": value["artifacts"]["registration"],
        "primitive_source_sha256": value["qualified_outer_source_sha256"],
        "guardian_contract_sha256": value["exec_contracts"]["guardian"],
    }
    bundle = {
        "phase": "before",
        "completion_pin": proof_pin,
        "artifacts": value["artifacts"],
        "evidence_pins": value["evidence_pins"],
    }
    kwargs = {
        "approved_outer_intent_sha256": value["outer_intent_sha256"],
        "qualified_outer_source_sha256": value["qualified_outer_source_sha256"],
        "enclosing": value["enclosing"],
        "reader": f.requests._PreworkReader(args["reader"]),
        "original_start": {
            "boot_id": anchor.boot_id,
            "monotonic_ns": int(anchor.started_monotonic * 1e9),
            "wall_ns": anchor.started_wall_ns,
        },
    }
    return blueprint, bundle, kwargs, original, files


def run(parts):
    bp, bundle, kw, *_ = parts
    raw = f.completion.encoded(bp)
    return f.finalize(raw, f.completion.sha(raw), f.completion.encoded(bundle), **kw)


def test_finalization_changes_only_three_future_output_pins(parts):
    bp, bundle, kw, original, files = parts
    bp_raw = f.completion.encoded(bp)
    result = run(parts)
    assert json.loads(result.template) == original
    assert result.blueprint_sha256 == f.completion.sha(bp_raw)
    assert result.template_sha256 == f.completion.sha(result.template)
    assert f.completion.encoded(bp) == bp_raw
    assert result.before_execution == files[bundle["completion_pin"]["path"]]
    assert (
        result.expected_before["outer_intent_sha256"]
        == kw["approved_outer_intent_sha256"]
    )
    leaked = result.expected_before
    leaked.clear()
    assert result.expected_before
    evidence = result.before_evidence
    evidence.clear()
    assert result.before_evidence


@pytest.mark.parametrize(
    "fault",
    [
        "intent",
        "source",
        "enclosing",
        "guardian",
        "request",
        "registration",
        "launch",
        "already-finalized",
        "future-proof",
        "wrong-output-path",
        "missing-proof",
        "proof-byte",
        "evidence-alias",
    ],
)
def test_hash_valid_substitutions_and_incomplete_before_refuse(parts, fault):
    bp, bundle, kw, original, files = parts
    if fault == "intent":
        kw["approved_outer_intent_sha256"] = "5" * 64
    if fault == "source":
        kw["qualified_outer_source_sha256"] = "5" * 64
    if fault == "enclosing":
        kw["enclosing"] = copy.deepcopy(kw["enclosing"])
        kw["enclosing"]["operator"]["start_ticks"] += 1
    if fault == "guardian":
        bp["guardian_contract_sha256"] = "5" * 64
    if fault in {"request", "registration", "launch"}:
        bundle["artifacts"] = copy.deepcopy(bundle["artifacts"])
        bundle["artifacts"][fault]["sha256"] = "5" * 64
    if fault == "already-finalized":
        bp["plan"]["preservation"]["before"] = original["preservation"]["before"]
    if fault == "future-proof":
        kw["original_start"] = dict(kw["original_start"], monotonic_ns=99 * 10**9)
    if fault == "wrong-output-path":
        bp["plan"]["preservation"]["before"]["path"] = "/input/old-before.json"
    if fault == "missing-proof":
        del bundle["completion_pin"]
    if fault == "proof-byte":
        files[bundle["completion_pin"]["path"]] += b"changed"
    if fault == "evidence-alias":
        bundle["evidence_pins"] = dict(
            bundle["evidence_pins"],
            supervisor_admission=bundle["evidence_pins"]["collector_family"],
        )
    with pytest.raises((ValueError, RuntimeError)):
        run(parts)


def test_blueprint_digest_is_external_not_self_authorized(parts):
    bp, bundle, kw, *_ = parts
    raw = f.completion.encoded(bp)
    with pytest.raises(ValueError, match="blueprint-byte-pin"):
        f.finalize(raw, "f" * 64, f.completion.encoded(bundle), **kw)


def test_expired_current_reader_is_not_renewed_by_historical_start(parts):
    _bp, _bundle, kw, *_ = parts
    kw["reader"].reader.deadline = 106
    with pytest.raises(f.requests.RequestRefusal, match="metadata-deadline"):
        run(parts)


def test_facade_and_finalizer_cold_import_are_stdlib_only():
    script = "from scripts import strength_freshness_cpu_install_runtime, strength_freshness_cpu_template_finalization; import sys; assert not any(x in ('torch','deltreltrain','deltrel_native') or x.endswith('_native') or x.endswith('train') for x in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-S", "-E", "-B", "-c", script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr.decode()
