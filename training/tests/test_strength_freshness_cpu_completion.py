from __future__ import annotations

import copy
import hashlib

import pytest
from scripts import strength_freshness_cpu_completion as c

BASE = 1_800_000_000_000_000_000


def valid_completion(phase="before"):
    """Synthetic kernel facts only. Return raw, expected, now, evidence, anchor."""
    start_ns = (100 if phase == "before" else 640) * 10**9

    def clock(seconds):
        return {
            "boot_id": "boot",
            "monotonic_ns": start_ns + seconds * 10**9,
            "wall_ns": BASE + start_ns + seconds * 10**9,
        }

    def identity(pid, ppid):
        return {
            "pid": pid,
            "start_ticks": pid * 10,
            "ppid": ppid,
            "pgid": pid,
            "sid": pid,
            "uid": 0,
            "boot_id": "boot",
            "cgroup": "/existing.scope",
            "pid_namespace_inode": 123,
        }

    sup, op, guardian, child = (
        identity(10, 1),
        identity(20, 10),
        identity(30, 20),
        identity(40, 30),
    )
    anchor = (
        None
        if phase == "before"
        else {
            "attempt_id": "fixture",
            "nonce": "a" * 32,
            "plan_sha256": "b" * 64,
            "boot_id": "boot",
            "started_monotonic": 100.0,
            "started_wall_ns": BASE + 100 * 10**9,
        }
    )
    budget = (
        c.before_budget(clock(0))
        if phase == "before"
        else c.after_budget(clock(0), anchor)
    )
    parent = (
        "/input"
        if phase == "before"
        else "/run/edgeconnect-cpuqual-" + ("a" * 32) + "/external"
    )

    def pin(path):
        return {"path": path, "sha256": "c" * 64, "bytes": 1}

    artifacts = {name: pin(parent + "/" + name + ".json") for name in c.ARTIFACTS}
    artifacts["capture"] = pin(parent + "/r3-" + phase + ".json")
    artifacts["receipt"] = pin(parent + "/r3-" + phase + ".receipt.json")
    artifacts["provenance"] = pin(
        parent + "/r3-" + phase + ".provenance-" + ("c" * 64) + ".json"
    )
    expected = {
        "nonce": "a" * 32,
        "boot_id": "boot",
        "outer_intent_sha256": "d" * 64,
        "qualified_outer_source_sha256": "e" * 64,
        "phase_start": clock(0),
        "budget": budget,
        "enclosing": {"operator": op, "supervisor": sup},
        "exec_contracts": {"guardian": "f" * 64, "collector": "1" * 64},
        "artifacts": artifacts,
        "plan_sha256": None if anchor is None else anchor["plan_sha256"],
        "anchor_sha256": None if anchor is None else c.sha(c.encoded(anchor)),
        "before_execution_pin": None
        if anchor is None
        else pin("/input/before-execution.json"),
    }
    binding = {
        "phase": phase,
        **{
            k: expected[k]
            for k in (
                "nonce",
                "boot_id",
                "outer_intent_sha256",
                "qualified_outer_source_sha256",
                "phase_start",
            )
        },
    }

    def admission(role, who, t):
        return {
            "role": role,
            "self": who,
            "clock": clock(t),
            "subreaper_before": 0,
            "subreaper_after": 1,
            "task_ids": [who["pid"]],
        }

    def family(role, owner, who, exec_hash, t):
        admit, born, released, terminal, closed = t
        return {
            "format": c.FAMILY,
            "role_admission": admission(role, owner, admit),
            "child_started": {
                "owner": owner,
                "child": who,
                "clock": clock(born),
                "exec_contract_sha256": exec_hash,
                "pidfd_target_pid": who["pid"],
                "gate_closed": True,
            },
            "child_released": {
                "child": who,
                "clock": clock(released),
                "exec_contract_sha256": exec_hash,
            },
            "child_terminal": {
                "owner": owner,
                "child": who,
                "clock": clock(terminal),
                "waitid": {"pid": who["pid"], "code": "CLD_EXITED", "status": 0},
                "waitpid": {"pid": who["pid"], "status": 0},
                "reaped": True,
            },
            "family_closed": {
                "owner": owner,
                "clock": clock(closed),
                "task_ids": [owner["pid"]],
                "direct_children": [],
                "retained_children": [],
            },
            "signals": [],
            "adopted_children": [],
            "other_terminals": [],
            "binding": binding,
            "budget": budget,
            "output": {
                "stdout_sha256": c.sha(b""),
                "stdout_bytes": 0,
                "stderr_sha256": c.sha(b""),
                "stderr_bytes": 0,
            },
            "natural_complete": True,
            "reason": None,
        }

    docs = {
        "collector_family": family(
            "guardian", guardian, child, "1" * 64, (4, 5, 6, 10, 11)
        ),
        "guardian_family": family(
            "operator", op, guardian, "f" * 64, (1, 2, 3, 12, 13)
        ),
        "supervisor_admission": admission("supervisor", sup, 0),
    }
    evidence = {name: c.encoded(doc) for name, doc in docs.items()}
    pins = {
        name: {
            "path": parent
            + "/execution-evidence/"
            + phase
            + "-"
            + name.replace("_", "-")
            + ".json",
            "sha256": c.sha(raw),
            "bytes": len(raw),
        }
        for name, raw in evidence.items()
    }
    value = {
        "format": c.FORMAT,
        "schema_version": 1,
        "status": "natural-exit-complete",
        "phase": phase,
        **copy.deepcopy(expected),
        "terminal_clock": clock(13),
        "published_clock": clock(14),
        "evidence_pins": pins,
    }
    return c.encoded(value), expected, clock(15), evidence, anchor


def validate(parts):
    raw, expected, now, evidence, anchor = parts
    return (
        c.validate_before(raw, expected, now, evidence=evidence)
        if anchor is None
        else c.validate_after(raw, expected, now, evidence=evidence, anchor=anchor)
    )


@pytest.mark.parametrize("phase", ["before", "after"])
def test_valid_two_producer_completion_is_immutable_and_enclosing_only(phase):
    parts = valid_completion(phase)
    result = validate(parts)
    assert (
        result.phase == phase and result.sha256 == hashlib.sha256(parts[0]).hexdigest()
    )
    assert result.original_deadline == parts[1]["budget"]["gate"]
    artifacts = result.artifacts
    artifacts.clear()
    assert result.artifacts
    assert set(result.value["enclosing"]) == {"operator", "supervisor"}
    assert (
        "operator_terminal" not in result.value
        and "supervisor_terminal" not in result.value
    )


def mutate_doc(parts, name, change):
    raw, expected, now, evidence, anchor = copy.deepcopy(parts)
    value = c.parse(raw)
    doc = c.parse(evidence[name])
    change(doc)
    evidence[name] = c.encoded(doc)
    value["evidence_pins"][name].update(
        sha256=c.sha(evidence[name]), bytes=len(evidence[name])
    )
    return c.encoded(value), expected, now, evidence, anchor


@pytest.mark.parametrize("name", ["collector_family", "guardian_family"])
@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["child_terminal"]["waitid"].update(code="CLD_KILLED"),
        lambda d: d["child_terminal"]["waitid"].update(status=1),
        lambda d: d["child_terminal"]["waitpid"].update(status=256),
        lambda d: d["child_terminal"].update(reaped=False),
        lambda d: d["family_closed"].update(direct_children=[99]),
        lambda d: d["family_closed"].update(retained_children=[99]),
        lambda d: d.update(signals=[{"pid": 99}]),
        lambda d: d.update(adopted_children=[{"pid": 99}]),
        lambda d: d.update(other_terminals=[{"pid": 99}]),
        lambda d: d["role_admission"].update(subreaper_after=0),
        lambda d: d["role_admission"].update(task_ids=[20, 21]),
        lambda d: d["child_started"].update(gate_closed=False),
        lambda d: d["child_started"].update(pidfd_target_pid=99),
    ],
)
def test_rehashed_raw_failure_cannot_be_certified(name, change):
    with pytest.raises(c.CompletionRefusal):
        validate(mutate_doc(valid_completion(), name, change))


@pytest.mark.parametrize(
    "field",
    ["outer_intent_sha256", "qualified_outer_source_sha256", "nonce", "boot_id"],
)
def test_original_binding_mismatch_refuses(field):
    raw, expected, now, evidence, anchor = valid_completion()
    value = c.parse(raw)
    value[field] = "f" * 64
    with pytest.raises(c.CompletionRefusal, match="completion-expectation-mismatch"):
        c.validate_before(c.encoded(value), expected, now, evidence=evidence)


def test_expired_or_backdated_publication_refuses():
    raw, expected, now, evidence, _ = valid_completion()
    value = c.parse(raw)
    value["published_clock"]["monotonic_ns"] = expected["budget"]["gate"][
        "monotonic_ns"
    ]
    with pytest.raises(c.CompletionRefusal):
        c.validate_before(c.encoded(value), expected, now, evidence=evidence)
    value = c.parse(raw)
    value["published_clock"]["wall_ns"] = value["phase_start"]["wall_ns"]
    with pytest.raises(c.CompletionRefusal):
        c.validate_before(c.encoded(value), expected, now, evidence=evidence)


def test_before_budget_cannot_be_renewed_even_with_matching_expected():
    raw, expected, now, evidence, _ = valid_completion()
    expected["budget"]["gate"]["wall_ns"] += 1
    value = c.parse(raw)
    value["budget"] = expected["budget"]
    with pytest.raises(c.CompletionRefusal, match="completion-original-budget"):
        c.validate_before(c.encoded(value), expected, now, evidence=evidence)


def test_after_anchor_is_real_authority_not_an_unbound_hash():
    parts = valid_completion("after")
    assert parts[-1] is not None
    parts[-1]["started_wall_ns"] += 1
    with pytest.raises(c.CompletionRefusal, match="completion-after-anchor"):
        validate(parts)


def test_missing_evidence_or_wrong_body_hash_refuses():
    raw, expected, now, evidence, _ = valid_completion()
    evidence["collector_family"] += b" "
    with pytest.raises(c.CompletionRefusal, match="completion-evidence-hash"):
        c.validate_before(raw, expected, now, evidence=evidence)
    del evidence["collector_family"]
    with pytest.raises(c.CompletionRefusal):
        c.validate_before(raw, expected, now, evidence=evidence)


def consistent_identity_change(parts, change):
    raw, expected, now, evidence, anchor = copy.deepcopy(parts)
    value = c.parse(raw)

    def visit(item):
        if isinstance(item, dict):
            if set(item) == c.IDENTITY:
                change(item)
            for v in item.values():
                visit(v)
        elif isinstance(item, list):
            for v in item:
                visit(v)

    visit(value)
    visit(expected)
    for name, body in evidence.items():
        doc = c.parse(body)
        visit(doc)
        evidence[name] = c.encoded(doc)
        value["evidence_pins"][name].update(
            sha256=c.sha(evidence[name]), bytes=len(evidence[name])
        )
    return c.encoded(value), expected, now, evidence, anchor


def test_consistently_nonroot_process_context_refuses():
    with pytest.raises(c.CompletionRefusal, match="completion-process-context"):
        validate(
            consistent_identity_change(
                valid_completion(), lambda identity: identity.update(uid=1000)
            )
        )


@pytest.mark.parametrize("pid", [20, 30, 40])
def test_consistent_child_birth_before_parent_refuses(pid):
    def change(identity):
        if identity["pid"] == pid:
            identity["start_ticks"] = 1

    with pytest.raises(c.CompletionRefusal, match="completion-process-birth-order"):
        validate(consistent_identity_change(valid_completion(), change))


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["output"].update(stdout_bytes=c.MAX_OUTPUT, stderr_bytes=1),
        lambda d: d["output"].update(stdout_sha256="a" * 64),
        lambda d: d["output"].update(stderr_sha256="a" * 64),
    ],
)
def test_rehashed_output_bound_or_false_empty_digest_refuses(change):
    with pytest.raises(c.CompletionRefusal, match="completion-output-pin"):
        validate(mutate_doc(valid_completion(), "collector_family", change))


def test_collector_contract_is_exact_outer_compact_encoding():
    from scripts import strength_freshness_cpu_outer as outer

    launch = {"path": "/input/launch.json", "sha256": "a" * 64, "bytes": 123}
    spec = outer.CaptureSpec(
        "after",
        "/venv/bin/python",
        "/control/training",
        launch["path"],
        launch["sha256"],
        555123456789,
    )
    assert c.collector_contract_sha256(
        spec.python, spec.control_root, launch, spec.deadline_monotonic_ns
    ) == outer.digest(spec.contract())


def test_fractional_anchor_uses_same_stored_float_projection_as_outer():
    from scripts import strength_freshness_cpu_outer as outer

    _, _, _, _, anchor = valid_completion("after")
    assert anchor is not None
    anchor["started_monotonic"] = 100.123456789
    start = {
        "boot_id": "boot",
        "monotonic_ns": 640123456789,
        "wall_ns": BASE + 640 * 10**9,
    }
    projected = outer.Clock(
        "boot", int(anchor["started_monotonic"] * 10**9), anchor["started_wall_ns"]
    )
    actual = outer.PhaseBudget.after(projected, outer.Clock(**start))
    assert actual.compact() == c.after_budget(start, anchor)


def test_incomplete_nested_family_raises_sanitized_refusal():
    parts = mutate_doc(
        valid_completion(), "collector_family", lambda d: d.pop("role_admission")
    )
    with pytest.raises(c.CompletionRefusal, match="completion-malformed"):
        validate(parts)
