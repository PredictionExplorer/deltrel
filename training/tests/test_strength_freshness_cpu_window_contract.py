"""Complete local producer->safe bodies; kernel/source approval is synthetic."""

from copy import deepcopy
import json
from typing import Any

import pytest

from scripts import strength_freshness_cpu_window_contract as m
from tests.test_strength_freshness_cpu_before_measurement import (
    measurement_fixture,
    SECRET,
)


def valid_bodies_fixture(tmp_path, monkeypatch, *, count=2, prefill=0, observations=0):
    session, write, _, metrics, registration, io, window, *_ = measurement_fixture(
        tmp_path, monkeypatch
    )
    if prefill:
        for _ in range(prefill):
            io._backend.advance(0.001)
            with metrics.open("ab") as stream:
                stream.write(
                    m.encoded(
                        {
                            "schema_version": 1,
                            "worker": "learner",
                            "event": "utd_wait",
                            "step": registration.private_copy()["policy"]["recipe"][
                                "bootstrap_step"
                            ],
                            "timestamp_ns": io._backend.wall_ns(),
                        }
                    )
                )
    session.begin()
    write(count)
    for _ in range(observations):
        session.observe()
    measured = session.finish().private_copy()
    expected = deepcopy(session._expected)
    result = m.encode_before_bodies(
        measured["capture_value"],
        measured["provenance_value_without_capture_pin"],
        expected=expected,
        registration=registration,
        output_root=str(tmp_path.resolve()),
    )
    interval = {
        "released": deepcopy(window["phase_start"]),
        "terminal": {
            **measured["capture_value"]["read_end"],
            "monotonic_ns": measured["capture_value"]["read_end"]["monotonic_ns"] + 1,
            "wall_ns": measured["capture_value"]["read_end"]["wall_ns"] + 1,
        },
    }
    pins = deepcopy(result["pins"])
    for role in ("request", "launch", "registration"):
        pins[role] = {
            "path": str(tmp_path.resolve() / (role + ".json")),
            "sha256": expected["bindings"][role + "_sha256"],
            "bytes": len(registration._raw) if role == "registration" else 10,
        }
    return result, expected, registration, interval, pins, measured


def validate(fixture):
    result, expected, registration, interval, pins, *_ = fixture
    return m.validate_before_bodies(
        result["capture"],
        result["receipt"],
        result["provenance"],
        expected=expected,
        registration=registration,
        producer_interval=interval,
        artifact_pins=pins,
    )


def repin(fixture, change):
    result, expected, reg, interval, pins, measured = fixture
    c = json.loads(result["capture"])
    p = json.loads(result["provenance"])
    r = json.loads(result["receipt"])
    change(c, p, r)
    result = deepcopy(result)
    pins = deepcopy(pins)
    result["capture"] = m.encoded(c)
    pins["capture"] = m.pin(pins["capture"]["path"], result["capture"])
    p["capture_pin"] = pins["capture"]
    result["provenance"] = m.encoded(p)
    root = str(__import__("pathlib").PurePosixPath(pins["capture"]["path"]).parent)
    pins["provenance"] = m.pin(
        root + "/r3-before.provenance-" + m.sha(result["provenance"]) + ".json",
        result["provenance"],
    )
    r.update(capture_pin=pins["capture"], provenance_pin=pins["provenance"])
    result["receipt"] = m.encoded(r)
    pins["receipt"] = m.pin(pins["receipt"]["path"], result["receipt"])
    return result, expected, reg, interval, pins, measured


def test_actual_measurement_safe_encoder_and_complete_consumer(tmp_path, monkeypatch):
    f = valid_bodies_fixture(tmp_path, monkeypatch)
    got = validate(f)
    assert (
        not got["execution_qualified"]
        and not got["writer_authority_granted"]
        and not got["full_preservation"]
    )
    assert all(
        SECRET not in raw.decode()
        for raw in (f[0]["capture"], f[0]["receipt"], f[0]["provenance"])
    )
    assert len(got["capture"]["common"]["owners"]) == 12
    assert len(got["provenance"]["renewal"]["latest"]) == 34
    assert (
        got["capture"]["metric_window"]["append_proof"]["minimum_serial_intervals"] == 1
    )
    got["capture"]["common"]["owners"].clear()
    assert len(validate(f)["capture"]["common"]["owners"]) == 12


@pytest.mark.parametrize(
    "fault",
    [
        "old-tag",
        "after",
        "schema-alias",
        "unknown",
        "bindings",
        "missing-kernel",
        "owner",
        "source",
        "cache",
        "units",
        "aux",
        "span-gap",
        "span-rewrite",
        "guard-growth",
        "prefix-suffix",
        "proof-label",
        "drop-metric",
        "metric-loss",
        "metric-ema",
        "metric-credit",
        "support",
        "witness",
        "renewal-counter",
        "renewal-teacher",
        "unread-projection",
        "privacy",
    ],
)
def test_coherently_repinned_semantic_mutants_refuse(tmp_path, monkeypatch, fault):
    f = valid_bodies_fixture(tmp_path, monkeypatch, prefill=1)

    def mutate(c, p, r):
        if fault == "old-tag":
            c["format"] = "strength-preservation-capture-measurement-v1"
        elif fault == "after":
            c["phase"] = "after"
        elif fault == "schema-alias":
            c["schema_version"] = True
        elif fault == "unknown":
            c["private"] = SECRET
        elif fault == "bindings":
            c["bindings"]["source_pins_sha256"] = "0" * 64
        elif fault == "missing-kernel":
            p["kernel_projections"] = []
        elif fault == "owner":
            c["common"]["owners"]["learner"]["start_ticks"] += 1
        elif fault == "source":
            p["kernel_projections"][0]["source_facts"]["source_pins"][
                next(iter(p["kernel_projections"][0]["source_facts"]["source_pins"]))
            ]["sha256"] = "0" * 64
        elif fault == "cache":
            p["kernel_projections"][0]["source_facts"]["cached_references"][
                next(
                    iter(
                        p["kernel_projections"][0]["source_facts"]["cached_references"]
                    )
                )
            ]["literal_stat"]["inode"] += 1
        elif fault == "units":
            p["kernel_projections"][0]["unit_facts"][
                next(iter(p["kernel_projections"][0]["unit_facts"]))
            ]["properties"]["NeedDaemonReload"] = "yes"
        elif fault == "aux":
            p["kernel_projections"][0]["auxiliaries"] = [
                {
                    "pid": 999,
                    "ppid": 998,
                    "start_ticks": 1,
                    "parent_role": "learner",
                    "kind": "torch-inductor-pool",
                }
            ]
        elif fault == "span-gap":
            p["metric_spans"]["originals"][0]["span"]["start"] += 1
        elif fault == "span-rewrite":
            p["metric_spans"]["rereads"][0]["span"]["sha256"] = "0" * 64
        elif fault == "guard-growth":
            p["metric_spans"]["end_guard"]["span"]["audit"]["named_after"]["bytes"] += 1
        elif fault == "prefix-suffix":
            c["metric_window"]["before_fence"]["fence"]["prefix"]["start"] += 1
        elif fault == "proof-label":
            c["metric_window"]["append_proof"]["qualifying_records"][0][
                "outcome_labels"
            ] += 1
        elif fault == "drop-metric":
            c["common"]["progress"]["metrics"].pop()
        elif fault == "metric-loss":
            c["common"]["progress"]["metrics"][0]["losses"] = {"private-label": 1}
        elif fault == "metric-ema":
            c["common"]["progress"]["metrics"][0]["ema"]["num_updates"] += 1
        elif fault == "metric-credit":
            c["common"]["progress"]["metrics"][0]["segment_updates_per_new_sample"] = 2
        elif fault == "support":
            c["common"]["support"]["monitor.service"]["active"] = "inactive"
        elif fault == "witness":
            p["support_witnesses"][next(iter(p["support_witnesses"]))]["absence"][
                "read_start"
            ]["monotonic_ns"] -= 1
        elif fault == "renewal-counter":
            p["renewal"]["observations"][1]["counters"]["progress"] += 1
        elif fault == "renewal-teacher":
            p["renewal"]["observations"][
                next(
                    i
                    for i, x in enumerate(p["renewal"]["observations"])
                    if x["teacher"] is not None
                )
            ]["teacher"]["proof_sha256"] = "0" * 64
        elif fault == "unread-projection":
            c["metric_window"]["before_fence"]["owner_before"][
                "kernel_projection_sha256"
            ] = "0" * 64
        else:
            p["privacy_scope"] = SECRET

    bad = repin(f, mutate)
    with pytest.raises(m.BodyRefusal) as error:
        validate(bad)
    assert SECRET not in str(error.value)


@pytest.mark.parametrize(
    "fault",
    [
        "late-release",
        "early-terminal",
        "missing-interval",
        "wrong-pin",
        "missing-six",
        "unknown-json",
        "duplicate-json",
    ],
)
def test_external_lifetime_and_raw_artifact_join_required(tmp_path, monkeypatch, fault):
    f: list[Any] = list(valid_bodies_fixture(tmp_path, monkeypatch))
    c = json.loads(f[0]["capture"])
    if fault == "late-release":
        f[3]["released"] = {
            **c["read_start"],
            "monotonic_ns": c["read_start"]["monotonic_ns"] + 1,
        }
    elif fault == "early-terminal":
        f[3]["terminal"] = {
            **c["read_end"],
            "monotonic_ns": c["read_end"]["monotonic_ns"] - 1,
        }
    elif fault == "missing-interval":
        f[3] = {}
    elif fault == "wrong-pin":
        f[4]["registration"]["sha256"] = "0" * 64
    elif fault == "missing-six":
        f[4].pop("launch")
    elif fault == "unknown-json":
        f[0]["capture"] = b'{"private":"' + SECRET.encode() + b'"}'
    else:
        f[0]["capture"] = b'{"x":1,"x":2}'
    with pytest.raises(m.BodyRefusal) as error:
        validate(f)
    assert SECRET not in str(error.value)


def test_all_five_selected_records_are_retained(tmp_path, monkeypatch):
    f = valid_bodies_fixture(tmp_path, monkeypatch, count=5)
    got = validate(f)
    metrics = got["capture"]["common"]["progress"]["metrics"]
    proof = got["capture"]["metric_window"]["append_proof"]
    assert len(metrics) == len(proof["qualifying_records"]) == 5


def test_control_source_digest_is_distinct_from_runtime_registered_pins(
    tmp_path, monkeypatch
):
    f = valid_bodies_fixture(tmp_path, monkeypatch)
    assert f[1]["bindings"]["source_pins_sha256"] == "f" * 64
    assert f[1]["bindings"]["source_pins_sha256"] != m.digest(
        f[2].private_copy()["source_pins"]
    )
    validate(f)


def test_intermediate_owner_after_B_before_E_is_valid_with_final_owner(
    tmp_path, monkeypatch
):
    f = valid_bodies_fixture(tmp_path, monkeypatch, observations=2)
    got = validate(f)
    b = got["capture"]["metric_window"]["before_fence"]
    e = got["capture"]["metric_window"]["end_fence"]
    assert (
        b["fence"]["audit"]["read_end"]["monotonic_ns"]
        <= b["owner_after"]["read_start"]["monotonic_ns"]
        < e["fence"]["audit"]["read_start"]["monotonic_ns"]
    )
    owners = got["provenance"]["owner_observations"]
    guard = got["provenance"]["metric_spans"]["end_guard"]
    assert (
        guard["span"]["audit"]["read_end"]["monotonic_ns"]
        <= owners[-1]["read_start"]["monotonic_ns"]
    )


@pytest.mark.parametrize("fault", ["B-owner-before-B", "final-owner-before-guard"])
def test_B_bracket_and_distinct_final_owner_are_required(tmp_path, monkeypatch, fault):
    f = valid_bodies_fixture(tmp_path, monkeypatch, observations=2)

    def mutate(c, p, r):
        b = c["metric_window"]["before_fence"]
        if fault == "B-owner-before-B":
            b["owner_after"] = deepcopy(b["owner_before"])
            c["metric_window"]["causal_fence_sha256"] = m.digest(b)
        else:
            p["owner_observations"].pop()

    with pytest.raises(m.BodyRefusal):
        validate(repin(f, mutate))


@pytest.mark.parametrize(
    "fault",
    [
        "restart-policy",
        "counter-regression",
        "teacher",
        "selected-ref",
        "source-interval",
    ],
)
def test_nested_consistency_hashes_do_not_hide_semantic_contradictions(
    tmp_path, monkeypatch, fault
):
    f = valid_bodies_fixture(tmp_path, monkeypatch)

    def mutate(c, p, r):
        proof = p["renewal"]
        rows = proof["observations"]
        if fault == "restart-policy":
            rows[0]["counters"]["learner"] += 1
        elif fault == "counter-regression":
            rows[1]["counters"]["progress"] += 100
        elif fault == "teacher":
            next(x for x in rows if x["teacher"] is not None)["teacher"][
                "proof_sha256"
            ] = "0" * 64
        elif fault == "selected-ref":
            row = c["metric_window"]["append_proof"]["qualifying_records"][0]
            row["step"] += 1
        else:
            p["source_checks"][-1]["read_end"] = deepcopy(
                p["source_checks"][-1]["read_start"]
            )
        proof["observations_sha256"] = m.digest(rows)
        proof["fence"]["observations"] = deepcopy(rows[:34])
        proof["fence_sha256"] = m.digest(proof["fence"])

    with pytest.raises(m.BodyRefusal):
        validate(repin(f, mutate))


@pytest.mark.parametrize("kind", ["depth", "raw-bytes", "oversize", "nan"])
def test_encoder_rejects_unbounded_or_private_value_types_without_leak(kind):
    value = {}
    if kind == "depth":
        for _ in range(66):
            value = {"child": value}
    elif kind == "raw-bytes":
        value = {"raw": SECRET.encode()}
    elif kind == "oversize":
        value = {"private": SECRET * (m.MAX_PROVENANCE // len(SECRET) + 1)}
    else:
        value = {"number": float("nan")}
    with pytest.raises(m.BodyRefusal) as error:
        m.encoded(value)
    assert SECRET not in str(error.value)


@pytest.mark.parametrize(
    "fault",
    [
        "fence-private-clock",
        "manager-private-stat",
        "support-bool-alias",
        "invented-final-clock",
    ],
)
def test_complete_safe_schema_and_actual_clock_cannot_be_replaced(
    tmp_path, monkeypatch, fault
):
    f = valid_bodies_fixture(tmp_path, monkeypatch)

    def mutate(c, p, r):
        if fault == "fence-private-clock":
            p["renewal"]["fence"]["read_end"] = SECRET
            p["renewal"]["fence_sha256"] = m.digest(p["renewal"]["fence"])
        elif fault == "manager-private-stat":
            for sp in p["support_provenance"]:
                sp["manager"]["namespaces"]["pid"]["stat"]["private"] = SECRET
                sp["manager_id"] = m.support.manager_id(sp["manager"])
            for witness in p["support_witnesses"].values():
                witness["proof"]["manager"]["namespaces"]["pid"]["stat"]["private"] = (
                    SECRET
                )
                witness["absence"]["manager_id"] = m.support.manager_id(
                    witness["proof"]["manager"]
                )
                witness["absence"]["raw_sha256"] = m.support.digest(witness["proof"])
                witness["sha256"] = m.support.digest(
                    {key: value for key, value in witness.items() if key != "sha256"}
                )
        elif fault == "support-bool-alias":
            c["common"]["support"]["monitor.service"]["enabled"] = int(
                c["common"]["support"]["monitor.service"]["enabled"]
            )
        else:
            c["read_end"]["monotonic_ns"] += 1
            c["read_end"]["wall_ns"] += 1
            p["read_end"] = deepcopy(c["read_end"])
            r["read_end"] = deepcopy(c["read_end"])
            proof = c["metric_window"]["append_proof"]
            proof["final_observation_monotonic_ns"] = c["read_end"]["monotonic_ns"]
            proof["final_observation_wall_ns"] = c["read_end"]["wall_ns"]
            proof["window_digest"] = m.digest(
                {
                    "start": f[1]["expected_window"]["phase_start"],
                    "deadline": f[1]["expected_window"]["deadline"],
                    "final": c["read_end"],
                }
            )

    with pytest.raises(m.BodyRefusal) as error:
        validate(repin(f, mutate))
    assert SECRET not in str(error.value)


def test_append_span_cap_matches_registered_reader(tmp_path, monkeypatch):
    f = valid_bodies_fixture(tmp_path, monkeypatch)
    p = json.loads(f[0]["provenance"])
    row = p["metric_spans"]["originals"][0]
    reg = f[2].private_copy()
    size = row["span"]["bytes"]
    reg["scope"]["tails"]["metrics"]["maximum_bytes"] = size
    m._append_observation(
        row, reg, f[1]["writer_binding"], f[1]["expected_window"], fence=False
    )
    reg["scope"]["tails"]["metrics"]["maximum_bytes"] = size - 1
    with pytest.raises(m.BodyRefusal, match="body-span-read-bound"):
        m._append_observation(
            row, reg, f[1]["writer_binding"], f[1]["expected_window"], fence=False
        )


def test_selected_record_cannot_exceed_parser_limit(tmp_path, monkeypatch):
    f = valid_bodies_fixture(tmp_path, monkeypatch)
    rows = json.loads(f[0]["capture"])["metric_window"]["append_proof"][
        "qualifying_records"
    ]
    size = max(row["end"] - row["start"] for row in rows)
    # A reduced parser limit exercises the exact boundary without a megabyte
    # fixture. The same imported constant constrains actual raw AppendProof parsing.
    c = json.loads(f[0]["capture"])
    p = json.loads(f[0]["provenance"])
    reg = f[2].private_copy()
    projections = {m.digest(k): k for k in p["kernel_projections"]}
    inventory = m.Counter(m.digest(a) for a in p["audit_inventory"])
    monkeypatch.setattr(m.append, "MAX_JSON", size)
    m._b_and_spans(c, p, reg, f[1], projections, inventory)
    monkeypatch.setattr(m.append, "MAX_JSON", size - 1)
    with pytest.raises(m.BodyRefusal, match="body-selected-record"):
        m._b_and_spans(c, p, reg, f[1], projections, inventory)


def test_nonzero_prefix_origin_cannot_claim_no_skipped_fragment(tmp_path, monkeypatch):
    f = valid_bodies_fixture(tmp_path, monkeypatch, prefill=700)
    proof = validate(f)["capture"]["metric_window"]["append_proof"]
    assert proof["coverage_start"] > 0
    assert proof["unparsed_pre_window_prefix_bytes"] > 0

    def mutate(c, p, r):
        c["metric_window"]["append_proof"]["unparsed_pre_window_prefix_bytes"] = 0

    with pytest.raises(m.BodyRefusal, match="body-prefix-fragment"):
        validate(repin(f, mutate))


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("Result", SECRET, "body-unit-result-enum"),
        ("SubState", SECRET, "body-unit-substate-enum"),
        ("MainPID", SECRET, "body-unit-number"),
        ("ActiveState", "failed", "body-observed-monitor-unhealthy"),
        ("SubState", "dead", "body-observed-monitor-unhealthy"),
        ("Result", "exit-code", "body-observed-monitor-unhealthy"),
    ],
)
def test_initial_support_private_or_negative_state_cannot_heal(
    tmp_path, monkeypatch, field, value, reason
):
    f = valid_bodies_fixture(tmp_path, monkeypatch)

    def mutate(c, p, r):
        assert len(p["support_provenance"]) >= 2
        p["support_provenance"][0]["unit_states"]["monitor.service"][field] = value

    with pytest.raises(m.BodyRefusal, match=reason) as error:
        validate(repin(f, mutate))
    assert SECRET not in str(error.value)


def test_initial_inactive_timer_cannot_heal(tmp_path, monkeypatch):
    f = valid_bodies_fixture(tmp_path, monkeypatch)
    reg = f[2].private_copy()
    name = next(k for k, v in reg["units"].items() if v["kind"] == "timer")

    def mutate(c, p, r):
        p["support_provenance"][0]["unit_states"][name]["ActiveState"] = "inactive"

    with pytest.raises(m.BodyRefusal, match="body-observed-timer-inactive"):
        validate(repin(f, mutate))


def test_active_oneshot_previous_result_is_not_new_completion(tmp_path, monkeypatch):
    f = valid_bodies_fixture(tmp_path, monkeypatch)
    sp = json.loads(f[0]["provenance"])["support_provenance"][0]
    props = deepcopy(sp["unit_states"]["monitor.service"])
    props.update(
        ActiveState="active",
        SubState="running",
        Result="exit-code",
        ExecMainStatus="1",
        InvocationID="a" * 32,
        InactiveExitTimestampMonotonic=str(sp["clock"]["monotonic_ns"] // 1000 - 1000),
    )
    # Isolated raw-state predicate: prior Result is not new completion while an
    # actual invocation remains active. This supplies no observed oneshot claim.
    maximum = f[2].private_copy()["policy"]["maximum_support_seconds"]
    m._support_unit_health("oneshot", props, sp["clock"], maximum)
    props["ActiveState"] = "failed"
    with pytest.raises(m.BodyRefusal, match="body-observed-support-failed"):
        m._support_unit_health("oneshot", props, sp["clock"], maximum)


@pytest.mark.parametrize(
    "slot",
    [
        "run",
        "continuation",
        "profile_authority",
        "profile",
        "run_source",
        "release_source",
        "source_manifest",
        "champion",
    ],
)
def test_static_claims_require_every_actual_authority_read(tmp_path, monkeypatch, slot):
    f = valid_bodies_fixture(tmp_path, monkeypatch)
    key = f[2].private_copy()["keys"][slot]

    def mutate(c, p, r):
        for source in p["source_checks"]:
            source["audit_inventory"] = [
                a
                for a in source["audit_inventory"]
                if not (a["operation"] == "read" and a["subject"] == key)
            ]
        p["audit_inventory"] = [
            a
            for a in p["audit_inventory"]
            if not (a["operation"] == "read" and a["subject"] == key)
        ]

    with pytest.raises(m.BodyRefusal, match="body-static-authority-read"):
        validate(repin(f, mutate))


@pytest.mark.parametrize(
    "slot", ["profile", "source_manifest", "run_source", "release_source"]
)
def test_static_authority_raw_hashes_join_known_policy_bytes(
    tmp_path, monkeypatch, slot
):
    f = valid_bodies_fixture(tmp_path, monkeypatch)
    key = f[2].private_copy()["keys"][slot]

    def mutate(c, p, r):
        for source in p["source_checks"]:
            for a in source["audit_inventory"]:
                if a["operation"] == "read" and a["subject"] == key:
                    a["raw"]["sha256"] = "0" * 64
        for a in p["audit_inventory"]:
            if a["operation"] == "read" and a["subject"] == key:
                a["raw"]["sha256"] = "0" * 64

    with pytest.raises(
        m.BodyRefusal, match="body-static-(authority-sha|source-commit)"
    ):
        validate(repin(f, mutate))


@pytest.mark.parametrize("stage", ["reporter", "source", "support", "guard"])
def test_finish_order_rejects_earlier_observations_as_final_checks(
    tmp_path, monkeypatch, stage
):
    f = valid_bodies_fixture(tmp_path, monkeypatch, observations=1)
    p = json.loads(f[0]["provenance"])
    m._finish_order(p)
    if stage == "reporter":
        p["renewal"]["observations"][-34:] = deepcopy(p["renewal"]["observations"][:34])
    elif stage == "source":
        p["source_checks"][-1] = deepcopy(p["source_checks"][0])
    elif stage == "support":
        p["support_provenance"][-1] = deepcopy(p["support_provenance"][0])
    else:
        p["metric_spans"]["end_guard"]["span"]["audit"]["read_start"] = deepcopy(
            p["read_start"]
        )
    with pytest.raises(m.BodyRefusal, match="body-clock-order"):
        m._finish_order(p)
