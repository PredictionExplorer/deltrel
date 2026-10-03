"""Pure schema fixtures. No writer/runtime qualification or target IO."""

from copy import deepcopy
import json
from typing import Any, cast

import pytest

from scripts import strength_freshness_cpu_observed_registration as m
from scripts import strength_freshness_cpu_collect_identity as old
from tests.test_strength_freshness_cpu_collect_identity import identity_fixture
from tests.test_strength_freshness_cpu_learner_window import sample_window
from tests.test_strength_freshness_cpu_preservation import observations

SECRET = "private-registration-sentinel"


def registration_fixture():
    legacy, io, _ = identity_fixture()
    policy, _, _, _ = getattr(observations, "__wrapped__")()
    r = {k: deepcopy(legacy[k]) for k in old.COMMON_FIELDS}
    policy.pop("learner_birth_upper_ns")
    policy.pop("physical_work_contract_sha256")
    policy.update(format=m.POLICY, contract=m.CONTRACT)
    policy["static"].update(
        runtime_name=old.RUNTIME, runtime_cgroup="/system.slice/" + old.RUNTIME
    )
    for row in policy["expected_processes"].values():
        row["cgroup"] = policy["static"]["runtime_cgroup"]
    policy["cohorts"] = sorted(m.COHORTS)
    r["cohorts"] = {
        name: {
            "key": name,
            "worker": name,
            "parent_role": name.rsplit("-cohort-", 1)[0],
        }
        for name in m.COHORTS
    }
    for name in m.COHORTS:
        r["scope"]["files"][name] = {"path": "/private-run/status/" + name + ".json"}
    r["policy"] = policy
    policy["support"] = {
        name: {
            "kind": spec["kind"],
            "definition_sha256": "a" * 64,
            "environment_sha256": "b" * 64,
            "boot_links_sha256": "c" * 64,
            "enabled": True,
        }
        for name, spec in r["units"].items()
        if name != old.RUNTIME
    }
    policy["expected_monitors"] = {
        "monitor.service": policy["expected_monitors"]["monitor.service"]
    }
    r["scope"]["tails"]["metrics"]["path"] = "/private-run/learner/metrics.jsonl"
    for key, p in [
        ("writer-evidence", "/private-input/writer.json"),
        ("access", "/private-input/access.json"),
        ("legacy-policy", "/private-input/legacy-policy.json"),
    ]:
        r["scope"]["files"][key] = {"path": p}
    sources = {
        "learner": m.append.LEARNER_SOURCE,
        "training": m.append.TRAINING_SOURCE,
        "runtime": m.RUNTIME_SOURCE,
        "actor": m.ACTOR_SOURCE,
        "coordinator": m.COORDINATOR_SOURCE,
    }
    for name, digest in sources.items():
        key = "source-" + name
        r["scope"]["files"][key] = {"path": "/private-source/" + name + ".py"}
        r["source_pins"][key] = {"sha256": digest, "bytes": 100}
    _, expected, _ = sample_window()
    writer = deepcopy(expected["binding"])
    writer.update(
        metrics_path="/private-run/learner/metrics.jsonl",
        output_root="/private-run/learner",
        run_id=policy["static"]["run_id"],
        recipe_sha256=m.sha(m.append.encoded(policy["recipe"])),
        native_sha256=policy["static"]["native_sha256"],
        command_sha256=r["origins"]["learner"]["argv_sha256"],
    )
    writer["file_identity"]["path"] = writer["metrics_path"]
    owner = writer["owner_key"]
    process = policy["expected_processes"]["learner"]
    for key in ("pid", "start_ticks", "cgroup", "invocation_id", "origin_sha256"):
        owner[key] = process[key]
    owner.update(
        ppid=policy["expected_processes"]["coordinator"]["pid"],
        boot_id="fixture-boot",
        uid=1000,
        pid_namespace_inode=42,
        time_namespace_inode=43,
        clock_ticks_per_second=100,
    )
    evidence = m.append.encoded(writer)
    r.update(
        format=m.FORMAT,
        schema_version=1,
        contract=m.CONTRACT,
        kernel_context={
            "boot_id": "fixture-boot",
            "clock_ticks_per_second": 100,
            "namespace_expectations": {
                "self": {"pid": 42, "time": 43},
                "pid1": {"pid": 42, "time": 43},
                "owners": {role: {"pid": 42, "time": 43} for role in m.ROLES},
            },
            "credentials": {role: {k: 1000 for k in m.UID_FIELDS} for role in m.ROLES},
        },
        learner_writer={
            "binding": writer,
            "evidence_pin": {
                "path": "/private-input/writer.json",
                "sha256": m.sha(evidence),
                "bytes": len(evidence),
            },
        },
        legacy_policy_reference_pin={
            "path": "/private-input/legacy-policy.json",
            "sha256": "a" * 64,
            "bytes": 100,
        },
    )
    pubs = {}
    for name in {"coordinator", *old.WORKERS, *m.COHORTS}:
        if name == "coordinator":
            vals = (r["keys"]["coordinator"], "coordinator", "coordinator-status", None)
        elif name in old.WORKERS:
            vals = (r["heartbeats"][name], name, "worker-heartbeat", name)
        else:
            vals = (
                r["cohorts"][name]["key"],
                r["cohorts"][name]["parent_role"],
                "cohort-heartbeat",
                name,
            )
        pubs[name] = {
            **dict(zip(("key", "owner_role", "record_kind", "worker_literal"), vals)),
            "source_contract_sha256": m.PUBLICATION_CONTRACT,
            "access_qualification_pin": {
                "path": "/private-input/access.json",
                "sha256": "b" * 64,
                "bytes": 100,
            },
            "uid": 1000,
            "gid": 1000,
            "mode": 0o600,
        }
    r["publication_writers"] = pubs
    # Private material exists, but neither repr nor public summary can expose it.
    r["origins"]["learner"]["environment"]["PRIVATE_TEST"] = {
        "present": True,
        "value_sha256": m.sha(SECRET.encode()),
        "value_bytes": len(SECRET),
    }
    scope = old.scope_from(r["scope"])
    return r, scope, evidence, cast(Any, io), legacy


def parse(r, scope, evidence):
    raw = m.encoded(r)
    return m.parse_observed_registration(
        raw, approved_sha256=m.sha(raw), scope=scope, writer_evidence=evidence
    )


def test_strict_new_registration_is_private_detached_and_not_a_grant():
    r, scope, evidence, io, _ = registration_fixture()
    parsed = parse(r, scope, evidence)
    assert io.calls == []
    summary = parsed.safe_summary()
    assert summary["publications"] == 34 and summary["owners"] == 12
    assert summary["schema_validated"] is True
    assert not any(
        summary[k]
        for k in (
            "writer_qualified",
            "runtime_qualified",
            "preservation_passed",
            "execution_authorized",
        )
    )
    assert "/private-" not in repr(parsed) + json.dumps(summary) and SECRET not in repr(
        parsed
    ) + json.dumps(summary)
    copy = parsed.private_copy()
    copy["kernel_context"]["boot_id"] = "changed"
    assert parsed.private_copy()["kernel_context"]["boot_id"] == "fixture-boot"
    assert parsed.private_writer_evidence() == evidence


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(format=old.FORMAT),
        lambda r: r.update(schema_version=True),
        lambda r: r.update(contract="legacy-v2"),
        lambda r: r.update(birth_reference={}),
        lambda r: r["policy"].update(learner_birth_upper_ns=1),
        lambda r: r["policy"].update(physical_work_contract_sha256="a" * 64),
        lambda r: r["learner_writer"]["binding"].update(trusted_writer=True),
        lambda r: r["kernel_context"].update(offset_upper_ns=1),
        lambda r: r.update(approved_sha256="a" * 64),
    ],
)
def test_cross_contract_or_fabricated_authority_fields_refuse(change):
    r, scope, evidence, io, _ = registration_fixture()
    change(r)
    with pytest.raises(m.ObservedRegistrationRefusal):
        parse(r, scope, evidence)
    assert io.calls == []


def test_legacy_constructor_still_rejects_new_format_and_null_birth_before_io():
    r, _, _, io, legacy = registration_fixture()
    with pytest.raises(old.CollectionRefusal):
        old.IdentityCollector(r, io)
    legacy["birth_reference"]["qualification_sha256"] = None
    with pytest.raises(old.CollectionRefusal):
        old.IdentityCollector(legacy, io)
    assert io.calls == []


def test_unapproved_bytes_and_duplicate_private_keys_refuse_safely():
    r, scope, evidence, _, _ = registration_fixture()
    raw = m.encoded(r)
    with pytest.raises(m.ObservedRegistrationRefusal, match="approval"):
        m.parse_observed_registration(
            raw, approved_sha256="0" * 64, scope=scope, writer_evidence=evidence
        )
    raw = b'{"format":"' + SECRET.encode() + b'","format":"duplicate"}'
    with pytest.raises(m.ObservedRegistrationRefusal) as error:
        m.parse_observed_registration(
            raw, approved_sha256=m.sha(raw), scope=scope, writer_evidence=evidence
        )
    assert SECRET not in str(error.value)


@pytest.mark.parametrize(
    "kind",
    [
        "recipe",
        "run",
        "path",
        "owner",
        "namespace",
        "uid",
        "source",
        "native",
        "evidence-bytes",
        "raw-scalar-alias",
    ],
)
def test_external_writer_policy_source_binding_refuses(kind):
    r, scope, evidence, _, _ = registration_fixture()
    writer = r["learner_writer"]["binding"]
    if kind == "recipe":
        r["policy"]["recipe"]["bootstrap_step"] += 1
    elif kind == "run":
        writer["run_id"] = "another"
    elif kind == "path":
        writer["file_identity"]["path"] = "/other/metrics.jsonl"
    elif kind == "owner":
        writer["owner_key"]["start_ticks"] += 1
    elif kind == "namespace":
        writer["owner_key"]["pid_namespace_inode"] += 1
    elif kind == "uid":
        writer["owner_key"]["uid"] += 1
    elif kind == "source":
        r["source_pins"]["source-runtime"]["sha256"] = "0" * 64
    elif kind == "native":
        writer["native_sha256"] = "not-a-digest"
    elif kind == "evidence-bytes":
        evidence += b" "
    else:
        doc = json.loads(evidence)
        doc["schema_version"] = 1.0
        evidence = m.encoded(doc)
    if kind not in ("evidence-bytes", "raw-scalar-alias"):
        evidence = m.append.encoded(writer)
    if kind != "evidence-bytes":
        r["learner_writer"]["evidence_pin"].update(
            sha256=m.sha(evidence), bytes=len(evidence)
        )
    with pytest.raises(m.ObservedRegistrationRefusal):
        parse(r, scope, evidence)


@pytest.mark.parametrize(
    "kind",
    [
        "roster",
        "fsuid",
        "uidbool",
        "ns-drift",
        "hzbool",
        "duplicate-pid",
        "gpu",
        "cohort",
        "scope-pin",
        "limits",
    ],
)
def test_fixed_owner_population_namespace_credentials_and_bounds(kind):
    r, scope, evidence, _, _ = registration_fixture()
    k = r["kernel_context"]
    if kind == "roster":
        k["credentials"].pop("monitor")
    elif kind == "fsuid":
        k["credentials"]["learner"]["filesystem"] = 0
    elif kind == "uidbool":
        k["credentials"]["learner"]["real"] = True
    elif kind == "ns-drift":
        k["namespace_expectations"]["owners"]["learner"]["time"] += 1
    elif kind == "hzbool":
        k["clock_ticks_per_second"] = True
    elif kind == "duplicate-pid":
        r["policy"]["expected_processes"]["learner"]["pid"] = r["policy"][
            "expected_processes"
        ]["coordinator"]["pid"]
    elif kind == "gpu":
        r["policy"]["gpu_roles"][r["policy"]["gpu_uuids"][0]] = "actor-cpu-ring4"
    elif kind == "cohort":
        r["policy"]["cohorts"].pop()
    elif kind == "scope-pin":
        r["learner_writer"]["evidence_pin"]["path"] = "/unregistered/writer.json"
    else:
        r["policy"]["maximum_seconds"] = 601
    with pytest.raises(m.ObservedRegistrationRefusal):
        parse(r, scope, evidence)


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "role",
        "kind",
        "worker",
        "hash",
        "uid",
        "mode",
        "path-alias",
        "access",
    ],
)
def test_publication_writer_contract_is_closed_and_path_specific(kind):
    r, scope, evidence, _, _ = registration_fixture()
    p = r["publication_writers"]["actor-gpu-1-cohort-0"]
    if kind == "missing":
        r["publication_writers"].pop("coordinator")
    elif kind == "role":
        p["owner_role"] = "learner"
    elif kind == "kind":
        p["record_kind"] = "arbitrary-parser"
    elif kind == "worker":
        p["worker_literal"] = "another"
    elif kind == "hash":
        p["source_contract_sha256"] = "0" * 64
    elif kind == "uid":
        p["uid"] = 0
    elif kind == "mode":
        p["mode"] = 0o666
    elif kind == "path-alias":
        key = r["cohorts"]["actor-gpu-1-cohort-0"]["key"]
        r["scope"]["files"][key]["path"] = r["scope"]["files"][
            r["heartbeats"]["learner"]
        ]["path"]
        scope = old.scope_from(r["scope"])
    else:
        p["access_qualification_pin"]["path"] = "/unregistered/access.json"
    with pytest.raises(m.ObservedRegistrationRefusal):
        parse(r, scope, evidence)


def test_well_formed_conflicting_native_byte_identity_refuses():
    r, scope, _, _, _ = registration_fixture()
    writer = r["learner_writer"]["binding"]
    writer["native_sha256"] = "f" * 64
    evidence = m.append.encoded(writer)
    r["learner_writer"]["evidence_pin"].update(
        sha256=m.sha(evidence), bytes=len(evidence)
    )
    with pytest.raises(m.ObservedRegistrationRefusal, match="writer-policy-binding"):
        parse(r, scope, evidence)


def test_impossible_all_equal_uid_vector_refuses():
    r, scope, evidence, _, _ = registration_fixture()
    r["kernel_context"]["credentials"]["learner"] = {key: 2**32 for key in m.UID_FIELDS}
    with pytest.raises(m.ObservedRegistrationRefusal, match="uid-consistency"):
        parse(r, scope, evidence)


def test_consistently_rehashed_wrong_raw_command_digest_refuses():
    r, scope, _, _, _ = registration_fixture()
    writer = r["learner_writer"]["binding"]
    writer["command_sha256"] = "e" * 64
    evidence = m.append.encoded(writer)
    r["learner_writer"]["evidence_pin"].update(
        sha256=m.sha(evidence), bytes=len(evidence)
    )
    with pytest.raises(m.ObservedRegistrationRefusal, match="writer-policy-binding"):
        parse(r, scope, evidence)
