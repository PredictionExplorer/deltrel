from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

from scripts import strength_freshness_cpu_capture_cli as m
from tests.test_strength_freshness_cpu_collector import sample as core_sample, BASE
from tests.test_strength_freshness_cpu_preservation import observations, window


def protected(path, value):
    raw = value if isinstance(value, bytes) else m.encoded(value)
    if path.exists():
        path.chmod(0o600)  # Deliberately mutate only this owned temporary fixture.
    path.write_bytes(raw)
    path.chmod(0o444)
    return m.file_pin(path, raw)


def launch_value(parent, phase="before") -> dict[str, Any]:
    value = dict(
        format=m.LAUNCH,
        schema_version=1,
        phase=phase,
        physical_kind="learner_metrics",
        request=dict(path=str(parent / "request.json"), sha256="a" * 64, bytes=3),
        registration=dict(
            path=str(parent / "registration.json"), sha256="b" * 64, bytes=3
        ),
    )
    if phase == "before":
        value.update(
            input_root=str(parent), verified_champions={"sha256-" + "a" * 64: "b" * 64}
        )
    else:
        value.update(
            plan=dict(path=str(parent / "plan.json"), sha256="c" * 64, bytes=3),
            anchor=dict(path=str(parent / "anchor.json"), sha256="d" * 64, bytes=3),
        )
    return value


class Runtime:
    def __init__(self, io):
        self.io = io
        self.owner_uid = os.geteuid()
        self.deadlines = []

    def monotonic_ns(self):
        return self.io.ns

    def wall_ns(self):
        return BASE + self.io.ns

    def clock(self, reader):
        reader.check()
        return self.io.clock().value

    def make_io(self, scope, deadline):
        assert scope == self.io.scope
        self.deadlines.append(deadline)
        self.io.deadline = deadline
        self.io.maximum_bytes = m.requests.RUNTIME_BUDGET
        self.io.budget = m.requests.RUNTIME_BUDGET
        return self.io


def prepared_before(tmp_path, monkeypatch):
    parent = tmp_path.resolve()
    parent.chmod(0o700)
    reg, io = cast(Any, core_sample).__wrapped__()
    source = protected(parent / "control-source.py", b"control source")
    reg["scope"]["files"]["source"]["path"] = source["path"]
    io.scope = m.collector.identities.scope_from(reg["scope"])
    io.maximum_bytes = m.requests.RUNTIME_BUDGET
    regpin = protected(parent / "registration.json", reg)
    value = dict(
        format=m.requests.FORMAT,
        schema_version=2,
        phase="before",
        registration_sha256=regpin["sha256"],
        nonce="a" * 32,
        source_pins=[source],
        limits=dict(
            started={"boot_id": "boot", "monotonic_ns": io.ns, "wall_ns": BASE + io.ns},
            deadline_monotonic_ns=120_000_000_000,
            deadline_wall_ns=BASE + 120_000_000_000,
            metadata_bytes=m.requests.METADATA_BUDGET,
            runtime_bytes=m.requests.RUNTIME_BUDGET,
        ),
    )
    reqpin = protected(parent / "request.json", value)
    config = launch_value(parent)
    config.update(request=reqpin, registration=regpin)
    pin = protected(parent / "launch.json", config)
    checks = []

    def qualified_source_fixture(reader, request):
        # The immutable import bootstrap is simulated here. Exact origin refusal
        # is tested separately; all metadata reads and all collector helpers run.
        checks.append(request["source_pins"])
        for row in request["source_pins"]:
            reader.read(row)

    monkeypatch.setattr(m, "verify_sources", qualified_source_fixture)
    return pin, Runtime(io), reg, value, checks


def execute(pin, rt, deadline=120_000_000_000):
    return m.execute_launch(pin["path"], pin["sha256"], deadline, runtime=rt)


def test_before_real_metadata_core_and_atomic_publication(tmp_path, monkeypatch):
    pin, rt, reg, value, checks = prepared_before(tmp_path, monkeypatch)
    result = execute(pin, rt)
    assert result["status"] == "complete" and not result["execution_qualified"]
    capture = json.loads(Path(result["capture_pin"]["path"]).read_bytes())
    receipt = json.loads(Path(result["receipt_pin"]["path"]).read_bytes())
    provenance = json.loads(Path(result["provenance_pin"]["path"]).read_bytes())
    assert set(receipt) == m.requests.RECEIPT_FIELDS
    assert (
        receipt["capture_pin"] == result["capture_pin"]
        and receipt["status"] == "complete"
    )
    assert (
        receipt["derivations"]["support_witnesses"] == provenance["support_witnesses"]
    )
    assert (
        provenance["binding"]["registration_file_sha256"]
        == value["registration_sha256"]
    )
    assert provenance["binding"]["capture_pin"] == result["capture_pin"]
    assert (
        m.preservation.digest(provenance["raw_inventory"])
        == receipt["raw_inventory_sha256"]
    )
    prefix = provenance["core_provenance"]["raw_inventory_count"]
    assert (
        m.preservation.digest(provenance["raw_inventory"][:prefix])
        == provenance["core_provenance"]["raw_inventory_sha256"]
    )
    assert capture["clock"] == m.collector.identities.capture_clock(receipt["read_end"])
    assert provenance["preservation_verification"] is None
    assert rt.deadlines == [120.0] and len(checks) == 2
    assert all(
        (Path(row["path"]).stat().st_mode & 0o777) == 0o444
        for row in result.values()
        if isinstance(row, dict)
    )
    assert "private-sentinel-value" not in json.dumps(provenance)
    with pytest.raises(m.LaunchRefusal, match="output-already-exists"):
        execute(pin, rt)


@pytest.mark.parametrize(
    "field", ["phase", "deadline", "source", "registration", "owner", "launch-hash"]
)
def test_binding_failures_precede_capture_and_publication(tmp_path, monkeypatch, field):
    pin, rt, reg, value, _ = prepared_before(tmp_path, monkeypatch)
    if field == "phase":
        cfg = json.loads(Path(pin["path"]).read_bytes())
        cfg["phase"] = "after"
        pin = protected(Path(pin["path"]), cfg)
    if field == "deadline":
        value["limits"]["deadline_monotonic_ns"] = 119_000_000_000
    if field == "registration":
        value["registration_sha256"] = "f" * 64
    if field in {"deadline", "registration"}:
        reqpin = protected(tmp_path / "request.json", value)
        cfg = json.loads(Path(pin["path"]).read_bytes())
        cfg["request"] = reqpin
        pin = protected(Path(pin["path"]), cfg)
    if field == "owner":
        (tmp_path / "launch.json").chmod(0o644)
    if field == "launch-hash":
        pin = {**pin, "sha256": "f" * 64}
    if field == "source":

        def source_refusal(*args):
            raise m.requests.RequestRefusal("executed-module-origin")

        monkeypatch.setattr(m, "verify_sources", source_refusal)
    with pytest.raises((m.LaunchRefusal, m.requests.RequestRefusal)):
        execute(pin, rt)
    assert rt.deadlines == []
    assert not list(tmp_path.glob("r3-before*"))


def test_actual_module_origin_requires_self_and_all_custom_helpers():
    modules = m.custom_modules()
    paths = set()
    for module in modules:
        assert isinstance(module.__file__, str)
        paths.add(str(Path(module.__file__).resolve()))
    value = {
        "source_pins": [dict(path=p, sha256="a" * 64, bytes=1) for p in sorted(paths)]
    }
    checked = m.requests.executed_sources(value, modules=modules)
    assert str(Path(m.__file__).resolve()) in checked
    value["source_pins"] = [
        x for x in value["source_pins"] if x["path"] != str(Path(m.__file__).resolve())
    ]
    value["source_pins"].append(
        dict(
            path="/shadow/strength_freshness_cpu_capture_cli.py",
            sha256="a" * 64,
            bytes=1,
        )
    )
    with pytest.raises(m.requests.RequestRefusal, match="executed-module-origin"):
        m.requests.executed_sources(value, modules=modules)


def test_restamp_revalidates_real_v2_without_changing_producer_timestamps():
    values = cast(Any, observations).__wrapped__()
    policy, before, after, verified = values
    limits = window(values)
    admission = SimpleNamespace(
        before=before,
        cleanup_clock=limits["cleanup_clock"],
        anchor=m.requests.lifecycle.Anchor.from_dict(limits["attempt"]),
    )
    old: dict[str, Any] = dict(
        boot_id="boot", monotonic_ns=110_000_000_000, wall_ns=after["clock"]["wall_ns"]
    )
    measured = {"capture": after, "provenance": {"read_end": old}}
    now = {
        **old,
        "monotonic_ns": old["monotonic_ns"] + 1_250_000_000,
        "wall_ns": old["wall_ns"] + 1_250_000_000,
    }
    frozen = copy.deepcopy(measured)
    final, projection, verdict = m.restamp(
        measured,
        now=now,
        policy=policy,
        verified_champions=verified,
        admission=admission,
    )
    assert measured == frozen and projection["added_support_age_seconds"] == 2
    assert final["capture"]["physical_work"] == after["physical_work"]
    assert final["capture"]["progress"] == after["progress"]
    assert verdict is not None
    assert (
        verdict["audit_clock"] == final["capture"]["clock"]
        and verdict["status"] == "passed"
    )
    expired = {
        **now,
        "monotonic_ns": now["monotonic_ns"] + 121 * 10**9,
        "wall_ns": now["wall_ns"] + 121 * 10**9,
    }
    with pytest.raises(m.preservation.PreservationViolation):
        m.restamp(
            measured,
            now=expired,
            policy=policy,
            verified_champions=verified,
            admission=admission,
        )


def publication(parent, check=lambda: None):
    return m.publish_files(
        parent,
        [
            ("r3-before.provenance-" + "a" * 64 + ".json", b"proof"),
            ("r3-before.receipt.json", b"receipt"),
            ("r3-before.json", b"capture"),
        ],
        owner_uid=os.geteuid(),
        check=check,
    )


def test_real_filesystem_sidecar_precedes_capture_and_is_no_clobber(tmp_path):
    parent = tmp_path.resolve()
    parent.chmod(0o700)
    order = []

    def check():
        order.append((parent / "r3-before.receipt.json").exists())

    publication(parent, check)
    assert (parent / "r3-before.json").read_bytes() == b"capture"
    assert any(order)
    with pytest.raises(m.LaunchRefusal, match="output-already-exists"):
        publication(parent)
    assert (parent / "r3-before.receipt.json").read_bytes() == b"receipt"


def test_interruption_after_sidecar_preserves_orphan_and_blocks_retry(tmp_path):
    parent = tmp_path.resolve()
    parent.chmod(0o700)

    def crash():
        if (parent / "r3-before.receipt.json").exists():
            raise RuntimeError("injected-stop")

    with pytest.raises(RuntimeError):
        publication(parent, crash)
    assert (parent / "r3-before.receipt.json").read_bytes() == b"receipt"
    assert not (parent / "r3-before.json").exists()
    with pytest.raises(m.LaunchRefusal, match="output-already-exists"):
        publication(parent)
    assert not list(parent.glob(".*.tmp-*"))


@pytest.mark.parametrize("kind", ["symlink", "mode", "existing-symlink"])
def test_publication_rejects_unprotected_or_aliased_parents(tmp_path, kind):
    parent = tmp_path.resolve()
    parent.chmod(0o700)
    if kind == "symlink":
        alias = parent / "alias"
        alias.symlink_to(parent, target_is_directory=True)
        with pytest.raises(m.LaunchRefusal):
            publication(alias)
    elif kind == "mode":
        parent.chmod(0o755)
        with pytest.raises(m.LaunchRefusal, match="output-parent-protection"):
            publication(parent)
    else:
        (parent / "r3-before.json").symlink_to(parent / "missing")
        with pytest.raises(m.LaunchRefusal, match="output-already-exists"):
            publication(parent)


def test_provenance_structure_is_bounded_and_not_truncated():
    with pytest.raises(m.LaunchRefusal, match="publication-structure-limit"):
        m.encoded([0] * 250001)
    value = {}
    for _ in range(65):
        value = {"nested": value}
    with pytest.raises(m.LaunchRefusal, match="publication-structure-limit"):
        m.encoded(value)
    with pytest.raises(m.LaunchRefusal, match="publication-json"):
        m.encoded({"x": float("inf")})


def test_core_refusal_writes_safe_bound_sidecar_without_capture(tmp_path, monkeypatch):
    pin, rt, _, _, _ = prepared_before(tmp_path, monkeypatch)

    def refuse(*args, **kwargs):
        raise m.collector.CaptureRefusal(
            "observed-fault",
            {
                "raw_inventory": [],
                "failed_operation": {"stdout_sha256": "a" * 64, "stdout_bytes": 20},
            },
        )

    monkeypatch.setattr(m.collector, "collect_capture", refuse)
    with pytest.raises(m.collector.CaptureRefusal):
        execute(pin, rt)
    receipt = json.loads((tmp_path / "r3-before.receipt.json").read_text())
    assert receipt["status"] == "refused" and receipt["capture_pin"] is None
    assert not (tmp_path / "r3-before.json").exists()
    provenance = json.loads(
        Path(receipt["derivations"]["provenance_pin"]["path"]).read_text()
    )
    assert (
        provenance["partial_core_provenance"]["failed_operation"]["stdout_sha256"]
        == "a" * 64
    )


def test_exhausted_budget_gets_no_late_failure_write(tmp_path, monkeypatch):
    pin, rt, _, _, _ = prepared_before(tmp_path, monkeypatch)

    def exhausted(*args, **kwargs):
        raise m.collector.CaptureRefusal("capture-byte-budget", {"raw_inventory": []})

    monkeypatch.setattr(m.collector, "collect_capture", exhausted)
    with pytest.raises(m.collector.CaptureRefusal):
        execute(pin, rt)
    assert not list(tmp_path.glob("r3-before*"))


def test_fresh_import_does_not_load_runtime_or_accelerator_modules():
    training = Path(__file__).resolve().parents[1]
    script = 'import sys;sys.path.insert(0,sys.argv[1]);import scripts.strength_freshness_cpu_capture_cli;assert not any(n.split(".")[0] in {"torch","numpy","startrain","deltreltrain","star_native","deltrel_native"} or n in {"scripts.strength_freshness_linux","scripts.strength_freshness_cpu_qualification"} for n in sys.modules)'
    result = subprocess.run(
        [sys.executable, "-S", "-E", "-B", "-c", script, str(training)],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr.decode()


def prepared_after(tmp_path, monkeypatch, proof_fault=False):
    before_pin, rt, reg, before_request, _ = prepared_before(tmp_path, monkeypatch)
    before_result = execute(before_pin, rt)
    before = json.loads(Path(before_result["capture_pin"]["path"]).read_bytes())
    inputs = tmp_path / "after-input"
    inputs.mkdir(mode=0o700)
    external = tmp_path / "external"
    external.mkdir(mode=0o700)
    plan = protected(
        inputs / "plan.json",
        {
            "fixture": "admission simulated; root request semantic chain covered separately"
        },
    )
    mark = before["clock"]
    anchor = m.requests.lifecycle.Anchor(
        "fixture",
        "a" * 32,
        plan["sha256"],
        "boot",
        mark["monotonic"] + 1,
        mark["wall_ns"] + 10**9,
    )
    cleanup = {
        "boot_id": "boot",
        "monotonic": anchor.started_monotonic + 1,
        "wall_ns": anchor.started_wall_ns + 10**9,
    }
    rt.io.ns = int((cleanup["monotonic"] + 1) * 1e9)
    deadline = rt.io.ns + 20 * 10**9
    regpin = protected(inputs / "registration.json", reg)
    value = copy.deepcopy(before_request)
    value.update(
        phase="after",
        plan_sha256=plan["sha256"],
        anchor_sha256=m.requests.lifecycle.digest(anchor.as_dict()),
        before_pin=before_result["capture_pin"],
        policy_pin=protected(inputs / "policy.json", reg["policy"]),
        cleanup_pin=protected(
            inputs / "cleanup.json", {"fixture": "qualified cleanup seam"}
        ),
        before_request_pin=m.file_pin(
            tmp_path / "request.json", (tmp_path / "request.json").read_bytes()
        ),
        before_receipt_pin=before_result["receipt_pin"],
    )
    value["limits"].update(
        started={
            "boot_id": "boot",
            "monotonic_ns": rt.io.ns,
            "wall_ns": BASE + rt.io.ns,
        },
        deadline_monotonic_ns=deadline,
        deadline_wall_ns=BASE + deadline,
    )
    reqpin = protected(inputs / "request.json", value)
    config = launch_value(inputs, "after")
    config.update(
        request=reqpin,
        registration=regpin,
        plan=plan,
        anchor=protected(inputs / "anchor.json", anchor.as_dict()),
        physical_kind="actor_broker",
    )
    launchpin = protected(inputs / "launch.json", config)
    proof = SimpleNamespace(calls=0, checked_wall_ns=0)

    def recheck():
        proof.calls += 1
        rt.io.ns += 2 * 10**9
        proof.checked_wall_ns = BASE + rt.io.ns
        if proof_fault:
            raise m.requests.RequestRefusal("proof-singleton-raced")

    proof.recheck = recheck

    def admit(**kwargs):
        assert kwargs["value"] == value and kwargs["plan_pin"] == plan
        assert kwargs["reader"].deadline <= deadline / 1e9
        return m.requests.AfterAdmission(
            anchor,
            cleanup,
            reg["policy"],
            before,
            json.loads(Path(before_result["receipt_pin"]["path"]).read_bytes())[
                "derivations"
            ]["support_witnesses"],
            config.get("verified_champions", {"sha256-" + "a" * 64: "b" * 64}),
            str(external / "r3-after.json"),
            cast(m.requests.ReadOnlyLog, proof),
        )

    monkeypatch.setattr(m.requests, "admit_after", admit)
    return launchpin, rt, deadline, proof, external


def test_after_real_core_v2_and_publication_recheck_proof_before_final_clock(
    tmp_path, monkeypatch
):
    pin, rt, deadline, proof, external = prepared_after(tmp_path, monkeypatch)
    result = execute(pin, rt, deadline)
    body = json.loads(Path(result["capture_pin"]["path"]).read_bytes())
    provenance = json.loads(Path(result["provenance_pin"]["path"]).read_bytes())
    assert body["format"] == "strength-freshness-cpu-r3-after-v1"
    assert Path(result["capture_pin"]["path"]) == external / "r3-after.json"
    assert (
        proof.calls == 1
        and body["capture"]["clock"]["wall_ns"] >= proof.checked_wall_ns
    )
    assert provenance["preservation_verification"]["status"] == "passed"
    assert (
        provenance["preservation_verification"]["audit_clock"]
        == body["capture"]["clock"]
    )
    assert provenance["final_clock_projection"]["added_support_age_seconds"] >= 2
    assert rt.deadlines[-1] == deadline / 1e9


def test_after_late_proof_failure_cannot_publish_capture(tmp_path, monkeypatch):
    pin, rt, deadline, proof, external = prepared_after(
        tmp_path, monkeypatch, proof_fault=True
    )
    with pytest.raises(m.requests.RequestRefusal, match="proof-singleton-raced"):
        execute(pin, rt, deadline)
    assert proof.calls == 1 and not (external / "r3-after.json").exists()
    receipt = json.loads((external / "r3-after.receipt.json").read_bytes())
    assert receipt["status"] == "refused" and receipt["capture_pin"] is None


def test_racing_capture_writer_is_never_overwritten_or_removed(tmp_path, monkeypatch):
    parent = tmp_path.resolve()
    parent.chmod(0o700)
    original = m.os.link

    def race(src, dst, **kwargs):
        if dst == "r3-before.json":
            (parent / dst).write_bytes(b"other-writer")
        return original(src, dst, **kwargs)

    monkeypatch.setattr(m.os, "link", race)
    with pytest.raises(FileExistsError):
        publication(parent)
    assert (parent / "r3-before.json").read_bytes() == b"other-writer"
    assert (parent / "r3-before.receipt.json").read_bytes() == b"receipt"


def test_post_commit_fsync_failure_never_retracts_visible_capture(
    tmp_path, monkeypatch
):
    parent = tmp_path.resolve()
    parent.chmod(0o700)
    original = m.os.fsync

    def fail(fd):
        if (parent / "r3-before.json").exists():
            raise OSError("injected-durability-error")
        return original(fd)

    monkeypatch.setattr(m.os, "fsync", fail)
    with pytest.raises(OSError):
        publication(parent)
    assert (parent / "r3-before.json").read_bytes() == b"capture"
    assert (parent / "r3-before.receipt.json").read_bytes() == b"receipt"


def test_provenance_orphan_refuses_before_new_collection(tmp_path, monkeypatch):
    pin, rt, _, _, _ = prepared_before(tmp_path, monkeypatch)
    (tmp_path / ("r3-before.provenance-" + "a" * 64 + ".json")).write_bytes(b"orphan")
    with pytest.raises(m.LaunchRefusal, match="output-already-exists"):
        execute(pin, rt)
    assert rt.deadlines == []
    assert not (tmp_path / "r3-before.json").exists()


def test_arbitrary_exception_data_is_absent_from_failure_artifacts(
    tmp_path, monkeypatch
):
    pin, rt, _, _, _ = prepared_before(tmp_path, monkeypatch)

    class PrivateFailure(Exception):
        audit = {"private": "do-not-publish-secret"}

    def fail(*args, **kwargs):
        raise PrivateFailure("do-not-publish-secret")

    monkeypatch.setattr(m.collector, "collect_capture", fail)
    with pytest.raises(PrivateFailure):
        execute(pin, rt)
    assert not (tmp_path / "r3-before.json").exists()
    for path in tmp_path.glob("r3-before*"):
        assert b"do-not-publish-secret" not in path.read_bytes()


def test_original_deadline_expiry_does_not_allow_failure_publication(
    tmp_path, monkeypatch
):
    pin, rt, _, _, _ = prepared_before(tmp_path, monkeypatch)

    def expire(*args, **kwargs):
        rt.io.ns = 120_000_000_000
        raise m.collector.CaptureRefusal(
            "absolute-read-deadline", {"raw_inventory": []}
        )

    monkeypatch.setattr(m.collector, "collect_capture", expire)
    with pytest.raises(m.collector.CaptureRefusal):
        execute(pin, rt)
    assert not list(tmp_path.glob("r3-before*"))


def test_oversized_provenance_refuses_without_truncation(tmp_path, monkeypatch):
    pin, rt, _, _, _ = prepared_before(tmp_path, monkeypatch)
    monkeypatch.setattr(m, "PROVENANCE_LIMIT", 10)
    with pytest.raises(m.LaunchRefusal, match="provenance-output-size"):
        execute(pin, rt)
    assert not list(tmp_path.glob("r3-before*"))


def test_stdout_error_never_echoes_launch_path_or_private_error(capsys):
    result = m.main(
        [
            "--launch",
            "private-input-path",
            "--launch-sha256",
            "a" * 64,
            "--deadline-monotonic-ns",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert result == 1 and captured.out == ""
    assert captured.err == '{"status":"refused","reason":"capture-cli-refused"}\n'


@pytest.mark.parametrize("deadline", [True, 10**30, 0])
def test_launch_deadline_not_renewable_or_ambiguous(tmp_path, monkeypatch, deadline):
    pin, rt, _, _, _ = prepared_before(tmp_path, monkeypatch)
    with pytest.raises(m.LaunchRefusal):
        execute(pin, rt, deadline)
    assert rt.deadlines == []


def test_failure_sidecar_is_immutable_and_cannot_become_complete_on_retry(
    tmp_path, monkeypatch
):
    pin, rt, _, _, _ = prepared_before(tmp_path, monkeypatch)

    def fail(*args, **kwargs):
        raise m.collector.CaptureRefusal("observed-fault", {"raw_inventory": []})

    monkeypatch.setattr(m.collector, "collect_capture", fail)
    with pytest.raises(m.collector.CaptureRefusal):
        execute(pin, rt)
    old = (tmp_path / "r3-before.receipt.json").read_bytes()
    with pytest.raises(m.LaunchRefusal, match="output-already-exists"):
        execute(pin, rt)
    assert (tmp_path / "r3-before.receipt.json").read_bytes() == old


def test_published_before_receipt_and_provenance_compose_with_real_admission_readers(
    tmp_path, monkeypatch
):
    launchpin, rt, reg, before_request, _ = prepared_before(tmp_path, monkeypatch)
    result = execute(launchpin, rt)
    reader = m.requests.PinnedReader(
        deadline=120, owner_uid=os.geteuid(), monotonic=lambda: rt.monotonic_ns() / 1e9
    )
    before = m.requests.records.parse_json(reader.read(result["capture_pin"]))
    receipt = m.requests.records.parse_json(reader.read(result["receipt_pin"]))
    request_pin = m.file_pin(
        tmp_path / "request.json", (tmp_path / "request.json").read_bytes()
    )
    anchor = m.requests.lifecycle.Anchor(
        "fixture",
        "a" * 32,
        "f" * 64,
        "boot",
        before["clock"]["monotonic"] + 1,
        before["clock"]["wall_ns"] + 10**9,
    )
    witnesses = m.requests.validate_before_receipt(
        receipt,
        before_request=before_request,
        before_request_pin=request_pin,
        before_pin=result["capture_pin"],
        registration=reg,
        before=before,
        anchor=anchor,
    )
    provenance = m.requests.read_before_provenance(
        reader,
        receipt=receipt,
        receipt_pin=result["receipt_pin"],
        before_request_pin=request_pin,
        before_pin=result["capture_pin"],
        registration=reg,
    )
    assert witnesses == provenance["support_witnesses"]
    assert provenance["binding"]["capture_pin"] == result["capture_pin"]
    assert reader.consumed <= m.requests.METADATA_BUDGET


def test_after_admission_uses_same_fresh_clock_to_tighten_original_reader(
    tmp_path, monkeypatch
):
    pin, rt, deadline, proof, _ = prepared_after(tmp_path, monkeypatch)
    original_clock = rt.clock
    calls = 0

    def skew(reader):
        nonlocal calls
        value = original_clock(reader)
        calls += 1
        if calls == 2:
            value["wall_ns"] += 2_000_000
        return value

    rt.clock = skew
    original_admit = m.requests.admit_after
    observed = []

    def admit(**kwargs):
        effective = m.requests.effective_deadline(kwargs["value"], kwargs["now"])
        assert kwargs["reader"].deadline <= effective
        observed.append((kwargs["reader"].deadline, effective))
        return original_admit(**kwargs)

    monkeypatch.setattr(m.requests, "admit_after", admit)
    result = execute(pin, rt, deadline)
    assert result["status"] == "complete" and proof.calls == 1
    assert observed and observed[0][0] < deadline / 1e9
    assert rt.deadlines[-1] == observed[0][0]
