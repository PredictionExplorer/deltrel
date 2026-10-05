"""Pure/fake-kernel composition tests. Never launch a target process."""

from dataclasses import asdict, replace
from pathlib import Path
import runpy
from typing import Any

import pytest

from scripts import strength_freshness_cpu_outer as o
from scripts import strength_freshness_cpu_outer_runtime as r

FakeKernel = runpy.run_path(
    str(Path(__file__).with_name("test_strength_freshness_cpu_outer.py"))
)["FakeKernel"]


def pin(path, raw=b"{}"):
    return {"path": str(path), "sha256": r.sha(raw), "bytes": len(raw)}


class MemoryStore:
    def __init__(self):
        self.files = {}
        self.deadline_ns = 999 * o.SECOND
        self.roots = (Path("/input"),)
        self.published = []

    def bind_deadline(self, original, mono, wall, clock_source):
        self.deadline_ns, self.deadline_wall_ns = mono, wall
        self.clock_source = clock_source
        self.original = original
        self.check()

    def check(self):
        now = self.clock_source()
        r.require(
            now.boot_id == self.original.boot_id
            and now.monotonic_ns < self.deadline_ns
            and now.wall_ns < self.deadline_wall_ns,
            "store-deadline",
        )

    def read(self, expected, **kwargs):
        raw = self.files[expected["path"]]
        assert pin(expected["path"], raw) == expected
        return raw

    def publish(self, path, raw, **kwargs):
        assert str(path) not in self.files
        self.files[str(path)] = raw
        self.published.append(str(path))
        return pin(path, raw)

    def fixed(self, path, **kwargs):
        return pin(path, self.files[str(path)]), self.files[str(path)]

    def ensure_directory(self, path, *, parent):
        assert path.parent == parent
        self.roots = tuple(set(self.roots) | {path})

    def directory(self, path):
        assert path in self.roots

    def directory_empty(self, path):
        return not any(Path(p).parent == path for p in self.files)

    def exists(self, path):
        return str(path) in self.files


def admission(role="guardian") -> Any:
    k = FakeKernel()
    k.t = o.SECOND
    store: Any = MemoryStore()
    cfg = {
        "nonce": "b" * 32,
        "preflight_start": asdict(k.clock()),
        "input_root": "/input",
        "control_root": "/qualified/control",
        "python": {"path": "/qualified/python"},
        "authorization_path": "/input/outer.json",
        "source_pins": [
            {
                "path": "/qualified/control/scripts/strength_freshness_cpu_outer.py",
                "sha256": "d" * 64,
            },
            {
                "path": "/qualified/control/scripts/strength_freshness_cpu_outer_runtime.py",
                "sha256": "e" * 64,
            },
        ],
    }
    return r.Admission(
        b"{}",
        "c" * 64,
        {},
        cfg,
        k.owner,
        o.Handle(k.processes[50], 99),
        k,
        store,
        pin("/input/session.json"),
        role,
    )


def guardian_frame(a, phase="before"):
    start = a.preflight_start
    anchor = (
        None
        if phase == "before"
        else {
            "attempt_id": "fixed",
            "nonce": a.outer["nonce"],
            "plan_sha256": "f" * 64,
            "boot_id": "boot",
            "started_monotonic": 1.0,
            "started_wall_ns": start.wall_ns,
        }
    )
    return {
        "format": r.FRAME,
        "schema_version": 1,
        "role": "guardian",
        "nonce": a.outer["nonce"],
        "authorization_sha256": a.approved_intent_sha256,
        "parent": asdict(a.parent.identity),
        "phase_start": asdict(start),
        "supervisor": asdict(replace(a.parent.identity, pid=25, start_ticks=1, ppid=1)),
        "phase": phase,
        "budget": r.completion.before_budget(asdict(start))
        if phase == "before"
        else r.completion.after_budget(asdict(start), anchor),
        "launch": pin("/input/launch.json"),
        "anchor": anchor,
    }


def test_guardian_actual_composition_uses_reviewed_family_engine():
    a = admission()
    frame = guardian_frame(a)
    a.reader.files[frame["launch"]["path"]] = b"{}"
    value = r.run_guardian(a, frame)
    assert value["natural_complete"] is True and value["reason"] is None
    assert value["child_terminal"]["waitpid"] == {"pid": 200, "status": 0}
    assert a.kernel.reaped == [200] and not a.kernel.children()
    r.completion._family(
        value,
        role="guardian",
        owner=asdict(a.identity),
        child=value["child_started"]["child"],
        exec_hash=value["child_started"]["exec_contract_sha256"],
        binding=value["binding"],
        budget=frame["budget"],
    )


def test_guardian_orphan_is_not_published_as_natural_completion():
    a = admission()
    a.kernel.orphans = 1
    frame = guardian_frame(a)
    a.reader.files[frame["launch"]["path"]] = b"{}"
    with pytest.raises(r.RuntimeRefusal, match="owned-child-incomplete"):
        r.run_guardian(a, frame)
    assert not a.kernel.children() and not a.reader.published


@pytest.mark.parametrize(
    "change",
    [
        lambda x: x.update(role="operator"),
        lambda x: x.update(authorization_sha256="f" * 64),
        lambda x: x.update(nonce="f" * 32),
        lambda x: x.update(callback="os.system"),
        lambda x: x["budget"]["work"].update(monotonic_ns=999999999999),
    ],
)
def test_role_frame_does_not_offer_generic_action_or_new_budget(change):
    a = admission()
    frame = guardian_frame(a)
    change(frame)
    with pytest.raises(r.RuntimeRefusal):
        r.validate_frame(
            frame,
            role="guardian",
            authorization_sha256=a.approved_intent_sha256,
            nonce=a.outer["nonce"],
        )
    assert not a.kernel.forked


def test_fixed_guardian_contract_has_no_intent_hash_cycle():
    a = admission()
    frame = r.encode(guardian_frame(a))
    first = o.RoleSpec(
        "before",
        "guardian",
        "/qualified/python",
        "/qualified/control",
        "/input/outer.json",
        "1" * 64,
        105 * o.SECOND,
        frame,
        a.role_source_hashes,
    )
    second = replace(
        first,
        phase="after",
        authorization_sha256="2" * 64,
        deadline_monotonic_ns=555 * o.SECOND,
    )
    assert first.argv() != second.argv()
    assert first.contract() == second.contract()
    assert first.contract()["argv_template"][-1] == "@approved-outer-intent-sha256"
    assert o.digest(first.contract()) == o.guardian_contract_sha256(
        first.python, first.control_root, first.authorization_path, a.role_source_hashes
    )
    assert (
        replace(first, authorization_path="/input/other.json").contract()
        != first.contract()
    )
    assert (
        replace(first, qualified_sources=("f" * 64, "e" * 64)).contract()
        != first.contract()
    )


@pytest.mark.parametrize("role", ["shell", "helper", "collector", "cleanup"])
def test_role_spec_cannot_select_other_program(role):
    with pytest.raises(o.OuterRefusal):
        o.RoleSpec(
            "before",
            role,
            "/qualified/python",
            "/qualified/control",
            "/input/outer.json",
            "a" * 64,
            1,
            b"{}",
            ("b" * 64, "c" * 64),
        ).argv()


def test_session_window_only_extends_from_early_original_handoff():
    start = o.Clock("boot", o.SECOND, 10**15)
    window = o.BudgetWindow(start)
    assert window.compact()["work"]["monotonic_ns"] == 119 * o.SECOND
    recorded = replace(
        start, monotonic_ns=110 * o.SECOND, wall_ns=start.wall_ns + 109 * o.SECOND
    )
    now = replace(
        recorded, monotonic_ns=111 * o.SECOND, wall_ns=recorded.wall_ns + o.SECOND
    )
    after = window.handoff(recorded, now)
    assert after.start == start
    assert (
        after.compact()["work"]["monotonic_ns"]
        == recorded.monotonic_ns + 598 * o.SECOND
    )
    assert (
        after.compact()["gate"]["monotonic_ns"] <= start.monotonic_ns + 720 * o.SECOND
    )
    with pytest.raises(o.OuterRefusal, match="repeated-handoff"):
        after.handoff(recorded, now)


@pytest.mark.parametrize("kind", ["late", "future", "boot", "wall"])
def test_window_refuses_renewal_or_unobserved_start(kind):
    start = o.Clock("boot", o.SECOND, 10**15)
    now = replace(
        start, monotonic_ns=119 * o.SECOND, wall_ns=start.wall_ns + 118 * o.SECOND
    )
    recorded = replace(
        start, monotonic_ns=100 * o.SECOND, wall_ns=start.wall_ns + 99 * o.SECOND
    )
    if kind == "future":
        now = start
    elif kind == "boot":
        now = replace(start, boot_id="another")
    elif kind == "wall":
        now = replace(start, wall_ns=start.wall_ns - 1)
    with pytest.raises(o.OuterRefusal):
        o.BudgetWindow(start).handoff(recorded, now)


def test_ack_is_detached_content_binding_not_a_clock_claim():
    a = admission()
    enclosing = {
        "operator": asdict(a.identity),
        "supervisor": asdict(a.parent.identity),
    }
    expected: dict[str, Any] = dict(
        outer_intent_sha256=a.approved_intent_sha256,
        nonce=a.outer["nonce"],
        start_pin=pin("/input/dummy-start.json"),
        before_execution_pin=pin("/input/before-execution.json"),
        enclosing=enclosing,
    )
    value = o.dummy_start_ack(**expected)
    assert "clock" not in value and value["format"] == o.ACK_FORMAT
    checked = o.validate_dummy_start_ack(value, **expected)
    checked["enclosing"]["operator"]["pid"] = 999
    assert value["enclosing"]["operator"]["pid"] == a.identity.pid
    with pytest.raises(o.OuterRefusal):
        o.validate_dummy_start_ack({**value, "nonce": "f" * 32}, **expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("nonce", "bad"),
        ("start_pin", pin("/else/dummy-start.json")),
        ("before_execution_pin", pin("/input/other.json")),
    ],
)
def test_ack_rejects_uncommitted_path_or_identity(field, value):
    a = admission()
    expected: dict[str, Any] = dict(
        outer_intent_sha256=a.approved_intent_sha256,
        nonce=a.outer["nonce"],
        start_pin=pin("/input/dummy-start.json"),
        before_execution_pin=pin("/input/before-execution.json"),
        enclosing={
            "operator": asdict(a.identity),
            "supervisor": asdict(a.parent.identity),
        },
    )
    expected[field] = value
    with pytest.raises(o.OuterRefusal):
        o.dummy_start_ack(**expected)


@pytest.mark.parametrize("mode", sorted(o.HELPER_MODES))
def test_helper_contract_is_fixed_site_enabled_and_bounded(mode):
    spec = o.HelperSpec(
        mode,
        "/qualified/python",
        "/qualified/control",
        "/input/outer.json",
        "a" * 64,
        100,
        b"{}",
    )
    assert spec.argv()[:3] == ("/qualified/python", "-s", "-B")
    assert (
        spec.argv()[3]
        == "/qualified/control/scripts/strength_freshness_cpu_install_helper.py"
    )
    assert spec.contract()["env"]["PYTHONPATH"] == "/qualified/control"
    assert spec.contract()["resources"]["address_space_bytes"] == 16 * 2**30
    assert spec.contract()["resources"]["stdout_stderr_bytes"] == 8 * 2**20


def test_helper_total_cap_includes_two_second_cleanup():
    start = o.Clock("boot", o.SECOND, 10**15)
    end = replace(
        start,
        monotonic_ns=start.monotonic_ns + 10 * o.SECOND,
        wall_ns=start.wall_ns + 10 * o.SECOND,
    )
    budget = o.HelperBudget("audit", start, start, end)
    budget.validate()
    assert budget.compact()["cleanup"]["monotonic_ns"] - budget.work_ns == 2 * o.SECOND
    with pytest.raises(o.OuterRefusal):
        replace(
            budget, ceiling=replace(end, monotonic_ns=end.monotonic_ns + 1)
        ).validate()


def test_collector_contract_stays_compatible_and_parent_hard_limit_is_explicit():
    spec = o.CaptureSpec(
        "before",
        "/qualified/python",
        "/qualified/control",
        "/input/launch.json",
        "a" * 64,
        100,
    )
    launch_pin = {"path": spec.launch_path, "sha256": spec.launch_sha256, "bytes": 1}
    assert o.digest(spec.contract()) == r.completion.collector_contract_sha256(
        spec.python, spec.control_root, launch_pin, 100
    )
    role = o.RoleSpec(
        "session",
        "operator",
        spec.python,
        spec.control_root,
        "/input/outer.json",
        "a" * 64,
        100,
        b"{}",
        ("b" * 64, "c" * 64),
    )
    assert role.contract()["resources"]["address_space_hard_bytes"] == 16 * 2**30
    assert role.contract()["resources"]["address_space_bytes"] == 512 * 2**20
    assert r.resource_policy()["address_space_tuples"]["guardian"] == [
        512 * 2**20,
        512 * 2**20,
    ]


def test_real_store_no_clobber_and_nonblocking_wrong_file_type(tmp_path):
    import os

    tmp_path.chmod(0o700)
    s = r.ProtectedStore(
        roots=(tmp_path,),
        deadline_ns=100,
        owner_uid=os.geteuid(),
        monotonic_ns=lambda: 1,
    )
    s.bind_deadline(o.Clock("boot", 0, 0), 100, 100, lambda: o.Clock("boot", 1, 1))
    p = s.publish(tmp_path / "proof.json", b"{}")
    assert s.read(p) == b"{}"
    with pytest.raises(FileExistsError):
        s.publish(tmp_path / "proof.json", b"changed")
    fifo = tmp_path / "fifo.json"
    os.mkfifo(fifo, 0o444)
    with pytest.raises(r.RuntimeRefusal, match="input-protection"):
        s.read(pin(fifo))


def test_interpreter_cap_is_separate_from_ordinary_source_counter(tmp_path):
    import os

    tmp_path.chmod(0o700)
    p = tmp_path / "python"
    p.write_bytes(b"ELF")
    p.chmod(0o444)
    s = r.ProtectedStore(
        roots=(tmp_path,),
        deadline_ns=100,
        owner_uid=os.geteuid(),
        monotonic_ns=lambda: 1,
    )
    s.consumed = 32 * 2**20
    assert (
        s.read(pin(p, b"ELF"), source=True, maximum=64 * 2**20, interpreter=True)
        == b"ELF"
    )
    assert s.consumed == 32 * 2**20 and s.interpreter_consumed == 3
    with pytest.raises(r.RuntimeRefusal, match="byte-budget"):
        s.read(pin(p, b"ELF"))


def test_null_actual_admission_never_becomes_a_target_fork(monkeypatch):
    monkeypatch.setattr(r.sys, "platform", "darwin")
    with pytest.raises(r.RuntimeRefusal, match="root-linux"):
        r.load_admission("/input/outer.json", "a" * 64, role="supervisor")


def test_final_audit_is_only_named_cpu_scope():
    doc = {
        "status": "passed_cpu_scope",
        "plan_sha256": "a" * 64,
        "anchor_sha256": "b" * 64,
        "evidence_pins": {"external_audit": pin("/input/audit.json")},
    }
    assert r.final_audit(r.encode(doc)) == doc
    with pytest.raises(r.RuntimeRefusal):
        r.final_audit(r.encode({**doc, "status": "aggregate-complete"}))


def test_release_clock_is_gate_initiation_even_for_fast_child():
    a = admission()
    old = a.kernel.release

    def release(fd):
        old(fd)
        a.kernel.t += o.SECOND

    a.kernel.release = release
    frame = guardian_frame(a)
    a.reader.files[frame["launch"]["path"]] = b"{}"
    report = r.run_guardian(a, frame)
    assert report["child_released"]["clock"]["monotonic_ns"] == o.SECOND
    assert report["child_terminal"]["clock"]["monotonic_ns"] >= 2 * o.SECOND


@pytest.mark.parametrize("failure", ["deadline", "parent"])
def test_gate_write_delay_cannot_report_natural_success(failure):
    a = admission()
    old = a.kernel.release

    def release(fd):
        old(fd)
        if failure == "deadline":
            a.kernel.t += 106 * o.SECOND
        else:
            a.kernel.parent_dead_at = a.kernel.t

    a.kernel.release = release
    frame = guardian_frame(a)
    a.reader.files[frame["launch"]["path"]] = b"{}"
    with pytest.raises(o.OuterRefusal, match="post-release"):
        r.run_guardian(a, frame)
    assert a.kernel.reaped == [200] and not a.kernel.children()


@pytest.mark.parametrize("drift", ["wall", "boot", "regressed"])
def test_bound_store_checks_actual_dual_clock_on_every_operation(tmp_path, drift):
    import os

    tmp_path.chmod(0o700)
    current = o.Clock("boot", 1, 1)
    store = r.ProtectedStore(
        roots=(tmp_path,),
        deadline_ns=100,
        owner_uid=os.getuid(),
        monotonic_ns=lambda: 1,
    )
    store.bind_deadline(o.Clock("boot", 0, 0), 100, 100, lambda: current)
    if drift == "wall":
        current = replace(current, wall_ns=100)
    elif drift == "boot":
        current = replace(current, boot_id="different")
    else:
        current = replace(current, wall_ns=0)
    with pytest.raises(r.RuntimeRefusal):
        store.publish(tmp_path / "no.json", b"{}")
    assert not (tmp_path / "no.json").exists()


def test_unbound_bootstrap_store_cannot_publish(tmp_path):
    import os

    tmp_path.chmod(0o700)
    store = r.ProtectedStore(
        roots=(tmp_path,),
        deadline_ns=100,
        owner_uid=os.getuid(),
        monotonic_ns=lambda: 1,
    )
    with pytest.raises(r.RuntimeRefusal, match="before-admission"):
        store.publish(tmp_path / "no.json", b"{}")


def test_ack_read_crossing_handoff_cutoff_refuses():
    a = admission("operator")
    enclosing = {
        "operator": asdict(a.identity),
        "supervisor": asdict(a.parent.identity),
    }
    start_pin, before_pin = (
        pin("/input/dummy-start.json"),
        pin("/input/before-execution.json"),
    )
    raw = r.encode(
        r.dummy_start_ack(
            outer_intent_sha256=a.approved_intent_sha256,
            nonce=a.outer["nonce"],
            start_pin=start_pin,
            before_execution_pin=before_pin,
            enclosing=enclosing,
        )
    )
    a.reader.files["/input/dummy-start.ack.json"] = raw
    old = a.reader.fixed

    def fixed(path, **kwargs):
        answer = old(path, **kwargs)
        a.kernel.wall_jump += 118 * o.SECOND
        return answer

    a.reader.fixed = fixed
    with pytest.raises(r.RuntimeRefusal, match="handoff-deadline"):
        r.await_start_ack(a, start_pin, before_pin, enclosing)


def publication_fixture(monkeypatch):
    from tests.test_strength_freshness_cpu_completion import valid_completion

    raw, expected, now, evidence, _ = valid_completion()
    a = admission("operator")
    a.approved_intent_sha256 = expected["outer_intent_sha256"]
    a.outer["nonce"] = expected["nonce"]
    a.outer["preflight_start"] = expected["phase_start"]
    a.outer["source_pins"][0]["sha256"] = expected["qualified_outer_source_sha256"]
    current = [r.clock(now)]
    a.kernel.clock = lambda: current[0]
    docs = {key: r.completion.parse(body) for key, body in evidence.items()}
    budget = o.PhaseBudget.before(r.clock(expected["phase_start"]))
    launch = expected["artifacts"]["launch"]
    calculated = r.completion.collector_contract_sha256(
        a.outer["python"]["path"], a.outer["control_root"], launch, budget.work_ns
    )
    for event in ("child_started", "child_released"):
        docs["collector_family"][event]["exec_contract_sha256"] = calculated
    monkeypatch.setattr(r, "bind_outputs", lambda *args: expected["artifacts"])
    kwargs: dict[str, Any] = dict(
        phase="before",
        budget=budget,
        launch_pin=launch,
        guardian_family=docs["guardian_family"],
        collector_family=docs["collector_family"],
        supervisor_admission=docs["supervisor_admission"],
        enclosing=expected["enclosing"],
        guardian_contract="f" * 64,
        anchor=None,
        plan_pin=None,
        before_execution=None,
    )
    return a, current, kwargs, raw, expected, evidence


@pytest.mark.parametrize("late", [False, True])
def test_completion_clock_sample_is_after_all_evidence_publications(monkeypatch, late):
    a, current, kwargs, *_ = publication_fixture(monkeypatch)
    old = a.reader.publish

    def publish(path, raw, **options):
        answer = old(path, raw, **options)
        if "execution-evidence" in str(path):
            current[0] = replace(
                current[0], wall_ns=current[0].wall_ns + (40 if late else 1) * o.SECOND
            )
        return answer

    a.reader.publish = publish
    if late:
        with pytest.raises(r.RuntimeRefusal, match="publication-late"):
            r.publish_completion(a, **kwargs)
        assert len(a.reader.published) == 3
        assert "/input/before-execution.json" not in a.reader.files
    else:
        r.publish_completion(a, **kwargs)
        marker = r.completion.parse(a.reader.files["/input/before-execution.json"])
        assert marker["published_clock"] == asdict(current[0])
        assert a.reader.published[-1] == "/input/before-execution.json"


def test_handoff_fresh_clock_after_proof_reads_cannot_backdate_ack(monkeypatch):
    a, current, _, raw, expected, evidence = publication_fixture(monkeypatch)
    start = r.clock(r.completion.parse(raw)["published_clock"])
    before_pin = pin("/input/before-execution.json", raw)
    a.reader.files[before_pin["path"]] = raw
    for name, p in r.completion.evidence_pins(raw).items():
        a.reader.files[p["path"]] = evidence[name]
    value = dict(
        format="strength-freshness-dummy-start-v1",
        schema_version=1,
        outer_intent_sha256=a.approved_intent_sha256,
        before_execution_pin=before_pin,
        clock=asdict(start),
    )
    a.reader.files["/input/dummy-start.json"] = r.encode(value)
    old = a.reader.read

    def read(p, **options):
        answer = old(p, **options)
        if p["path"].endswith("supervisor-admission.json"):
            current[0] = replace(
                current[0], wall_ns=a.preflight_start.wall_ns + 118 * o.SECOND
            )
        return answer

    a.reader.read = read
    with pytest.raises(r.RuntimeRefusal, match="handoff-deadline"):
        r.admit_dummy_handoff(
            a, o.BudgetWindow(a.preflight_start), expected["enclosing"]
        )
    assert "/input/dummy-start.ack.json" not in a.reader.files


def test_actual_interpreter_metadata_and_deleted_image_refuse(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace

    python = tmp_path / "python"
    python.write_bytes(b"ELF-image")
    python.chmod(0o775)
    store = r.ProtectedStore(
        roots=(tmp_path,),
        deadline_ns=100,
        owner_uid=os.getuid(),
        monotonic_ns=lambda: 1,
    )
    store.bind_deadline(o.Clock("boot", 0, 0), 100, 100, lambda: o.Clock("boot", 1, 1))
    registered = {
        **pin(python, b"ELF-image"),
        "resolved_path": str(python),
        "metadata": r.file_identity(python.stat()),
    }
    monkeypatch.setattr(r.sys, "executable", str(python))
    monkeypatch.setattr(r, "os", SimpleNamespace(**vars(os)))
    real_open = os.open
    monkeypatch.setattr(r.os, "readlink", lambda path: str(python))
    monkeypatch.setattr(
        r.os,
        "stat",
        lambda path, *args, **kwargs: os.stat(
            python if path == "/proc/self/exe" else path, *args, **kwargs
        ),
    )
    monkeypatch.setattr(
        r.os,
        "open",
        lambda path, flags, *args, **kwargs: real_open(
            python if path == "/proc/self/exe" else path, flags, *args, **kwargs
        ),
    )
    r.current_interpreter(store, registered)
    assert store.interpreter_consumed == 2 * len(b"ELF-image")
    registered["metadata"]["inode"] += 1
    with pytest.raises(r.RuntimeRefusal, match="image-identity"):
        r.current_interpreter(store, registered)
    registered["metadata"]["inode"] -= 1
    with pytest.raises(r.RuntimeRefusal, match="interpreter-account-failed"):
        r.current_interpreter(store, registered)
    assert store.interpreter_consumed == 2 * len(b"ELF-image")
    # Independent negative case, not a reset or retry on the failed owner.
    fresh = r.ProtectedStore(
        roots=(tmp_path,),
        deadline_ns=100,
        owner_uid=os.getuid(),
        monotonic_ns=lambda: 1,
    )
    fresh.bind_deadline(o.Clock("boot", 0, 0), 100, 100, lambda: o.Clock("boot", 1, 1))
    monkeypatch.setattr(r.os, "readlink", lambda path: str(python) + " (deleted)")
    with pytest.raises(r.RuntimeRefusal, match="image-path"):
        r.current_interpreter(fresh, registered)


def test_helper_liveness_mandatory_during_facade_polling():
    a = admission("operator")
    helper = r.HelperExecutor(a, "/input/outer.json")
    helper.bind_dummy_start(a.preflight_start)
    helper.check_alive()
    a.kernel.parent_dead_at = a.kernel.t
    with pytest.raises(r.RuntimeRefusal, match="supervisor-lost"):
        helper.check_alive()


def test_operator_does_not_reach_installer_before_ack(monkeypatch):
    a = admission("operator")
    frame = {
        **r.frame_base(a, "operator", a.preflight_start),
        "parent": asdict(a.parent.identity),
        "supervisor": asdict(a.parent.identity),
    }
    a.reader.files["/input/supervisor-admission.json"] = b"{}"
    a.outer["before_launch"] = pin("/input/launch.json")
    bundle = r.encode(
        dict(
            phase="before",
            completion_pin=pin("/input/before-execution.json"),
            artifacts={},
            evidence_pins={},
        )
    )
    monkeypatch.setattr(r, "capture_phase", lambda *args, **kwargs: bundle)

    def record(*args):
        raw = r.encode({"clock": asdict(a.preflight_start)})
        return a.reader.publish(Path("/input/dummy-start.json"), raw)

    monkeypatch.setattr(r, "original_start_record", record)

    def refuse(*args):
        raise r.RuntimeRefusal("missing-ack")

    monkeypatch.setattr(r, "await_start_ack", refuse)
    called = []
    with pytest.raises(r.RuntimeRefusal, match="missing-ack"):
        r.run_operator(
            a, frame, installer_factory=lambda *args, **kwargs: called.append(True)
        )
    assert called == [] and a.reader.published == ["/input/dummy-start.json"]


@pytest.mark.parametrize(
    "change",
    [
        dict(cgroup="/other.scope"),
        dict(uid=1000),
        dict(start_ticks=1),
        dict(pid_namespace_inode=200),
    ],
)
def test_actual_parent_context_refuses_unqualified_lifetime(change):
    a = admission()
    own = replace(a.identity, **change)
    with pytest.raises(r.RuntimeRefusal, match="parent-lifetime"):
        r.validate_parent_context(
            own, a.parent.identity, asdict(a.parent.identity), asdict(a.parent.identity)
        )


def test_facade_actual_origin_must_match_exact_pinned_path(monkeypatch):
    from scripts import strength_freshness_cpu_install_runtime as facade

    a = admission("operator")
    filename = "/qualified/control/scripts/strength_freshness_cpu_install_runtime.py"
    p = pin(filename)
    a.outer["source_pins"].append(p)
    a.reader.files[filename] = b"{}"
    monkeypatch.setattr(
        facade,
        "__file__",
        "/qualified/control/shadow/strength_freshness_cpu_install_runtime.py",
    )
    with pytest.raises(r.RuntimeRefusal, match="installer-import-origin"):
        r.actual_installer_factory(a)
    monkeypatch.setattr(facade, "__file__", filename)
    assert r.actual_installer_factory(a) is facade.Installer


def test_site_startup_additions_and_cached_large_drift_are_refused(tmp_path):
    root = tmp_path / "site"
    root.mkdir()
    startup, binary = root / "startup.pth", root / "large.so"
    startup.write_bytes(b"import approved\n")
    binary.write_bytes(b"not-read-large-library")
    startup_pin = pin(startup, startup.read_bytes())
    binary_meta = r.file_identity(binary.stat())
    cache = dict(
        format="strength-freshness-qualified-file-cache-v1",
        status="independently-qualified",
        evidence_kind="real-host-content-hash",
        synthetic=False,
        path=str(binary),
        sha256="f" * 64,
        metadata=binary_meta,
    )
    cache_raw = r.encode(cache)
    cache_pin = pin("/input/cache.json", cache_raw)
    inventory = dict(
        format="strength-freshness-qualified-site-files-v1",
        schema_version=1,
        root=str(root),
        startup_directories=[
            dict(path=str(root), entries=sorted([startup.name, binary.name]))
        ],
        startup_files=[
            dict(
                path=str(startup),
                sha256=startup_pin["sha256"],
                metadata=r.file_identity(startup.stat()),
            )
        ],
        cached_files=[
            dict(
                path=str(binary),
                sha256="f" * 64,
                metadata=binary_meta,
                cache_receipt=cache_pin,
            )
        ],
    )
    raw = r.encode(inventory)
    p = pin("/input/site-inventory.json", raw)
    store: Any = MemoryStore()
    store.bind_deadline(o.Clock("boot", 0, 0), 100, 100, lambda: o.Clock("boot", 1, 1))
    store.files.update(
        {
            p["path"]: raw,
            cache_pin["path"]: cache_raw,
            str(startup): startup.read_bytes(),
        }
    )
    r.current_site_inventory(store, {"site_inventory_pin": p})
    # There is deliberately no binary payload in the reader; cached path is stat-only.
    addition = root / "unexpected.pth"
    addition.write_bytes(b"new startup")
    with pytest.raises(r.RuntimeRefusal, match="addition-or-removal"):
        r.current_site_inventory(store, {"site_inventory_pin": p})
    addition.unlink()
    binary.write_bytes(b"changed")
    with pytest.raises(r.RuntimeRefusal, match="current-file-drift"):
        r.current_site_inventory(store, {"site_inventory_pin": p})


def test_supervisor_natural_operator_exit_is_only_candidate_scope(monkeypatch):
    a = admission("supervisor")
    a.kernel.stdout = [
        r.encode(
            {
                "format": "strength-freshness-operator-result-v1",
                "status": "passed_cpu_scope",
            }
        ),
        b"",
    ]
    calls = []

    def handoff(actual, window, enclosing):
        calls.append(enclosing)
        return a.preflight_start

    monkeypatch.setattr(r, "admit_dummy_handoff", handoff)
    result = r.run_supervisor(a, authorization_path="/input/outer.json")
    assert result["status"] == "candidate_cpu_scope"
    assert result["outside_caller_terminal_audit_required"] is True
    assert result["execution_qualified"] is False
    assert a.kernel.reaped == [200] and not a.kernel.children()
    assert calls[0]["operator"]["pid"] == 200


def test_supervisor_readiness_failure_reaps_owned_child(monkeypatch):
    a = admission("supervisor")
    a.kernel.ready = False
    with pytest.raises(o.OuterRefusal, match="not-ready"):
        r.run_supervisor(a, authorization_path="/input/outer.json")
    assert a.kernel.reaped == [200] and not a.kernel.children()
    assert "/input/outer-result.json" not in a.reader.files


def test_operator_original_start_precedes_all_installation_and_after(monkeypatch):
    a = admission("operator")
    frame = {
        **r.frame_base(a, "operator", a.preflight_start),
        "parent": asdict(a.parent.identity),
        "supervisor": asdict(a.parent.identity),
    }
    a.reader.files["/input/supervisor-admission.json"] = b"{}"
    a.outer["before_launch"] = pin("/input/launch.json")
    events = []
    anchor = dict(
        boot_id="boot",
        nonce=a.outer["nonce"],
        plan_sha256="e" * 64,
        attempt_id="test",
        started_monotonic=1.0,
        started_wall_ns=a.preflight_start.wall_ns,
    )
    prepared = dict(
        plan_pin=pin("/input/plan.json"),
        anchor_pin=pin("/input/anchor.json"),
        after_path="/run/edgeconnect-cpuqual-"
        + a.outer["nonce"]
        + "/external/r3-after.json",
        authorization_root="/input",
        cleanup_proof_root="/input/proofs",
    )

    def capture(*args, phase, **kwargs):
        events.append(phase)
        return r.encode(
            dict(
                phase=phase,
                completion_pin=pin("/input/before-execution.json"),
                artifacts={},
                evidence_pins={},
            )
        )

    def original(*args):
        events.append("original-start")
        return a.reader.publish(
            Path("/input/dummy-start.json"),
            r.encode({"clock": asdict(a.preflight_start)}),
        )

    def ack(*args):
        events.append("ack")

    class Installer:
        def __init__(self, raw, checksum, enclosing, *, intent_pin, helper_executor):
            assert events == ["before", "original-start", "ack"]
            assert enclosing["operator"] == asdict(a.identity)
            self.helper = helper_executor

        def finalize_template_and_install(self, bundle, original_start):
            self.helper.check_alive()
            events.append("install")
            assert original_start == asdict(a.preflight_start)
            return r.encode(prepared)

        def await_checked_cleanup(self):
            events.append("cleanup")
            return r.encode(
                dict(
                    cleanup_pin=pin("/input/cleanup.json"),
                    clock=asdict(a.preflight_start),
                )
            )

        def final_audit_and_retire(self):
            events.append("audit")
            return r.encode(
                dict(
                    status="passed_cpu_scope",
                    plan_sha256=prepared["plan_pin"]["sha256"],
                    anchor_sha256=r.sha(r.encode(anchor)),
                    evidence_pins={"external_audit": pin("/input/audit.json")},
                )
            )

    monkeypatch.setattr(r, "capture_phase", capture)
    monkeypatch.setattr(r, "original_start_record", original)
    monkeypatch.setattr(r, "await_start_ack", ack)
    monkeypatch.setattr(r, "arm_alarm", lambda *args: None)
    monkeypatch.setattr(
        r,
        "after_launch",
        lambda *args: (
            pin("/input/after-launch.json"),
            anchor,
            o.PhaseBudget.after(a.preflight_start, a.preflight_start),
        ),
    )
    result = r.run_operator(a, frame, installer_factory=Installer)
    assert events == [
        "before",
        "original-start",
        "ack",
        "install",
        "cleanup",
        "after",
        "audit",
    ]
    assert result["status"] == "passed_cpu_scope"
