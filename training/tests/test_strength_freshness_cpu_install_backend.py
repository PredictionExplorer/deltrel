"""Closed transaction CPU/fault tests; only private temporary files are written."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from scripts import strength_freshness_cpu_install_backend as m
from scripts import strength_freshness_cpu_install as render
from scripts import strength_freshness_cpu_lifecycle as life
from test_strength_freshness_cpu_install import fixture_inputs


class FakeHost:
    def __init__(self, prepared, files):
        self.prepared, self.files = prepared, files
        self.t = prepared.anchor.started_monotonic
        self.trace = []
        self.states = {}
        self.jobs = {}
        self.links = {}
        self.members = {}
        self.fail_arm = False
        self.failed_start = None
        self.advance_reload = 0

    def clock(self):
        a = self.prepared.anchor
        return life.Clock(
            a.boot_id,
            self.t,
            a.started_wall_ns + round((self.t - a.started_monotonic) * 1e9),
        )

    def observe(self, names):
        rows = {}
        for name in names:
            present = (
                self.files.stat(
                    self.prepared.plan.value["units"][name]["installed_path"]
                )
                is not None
            )
            state = self.states.get(name, "inactive")
            rows[name] = {
                "name": name,
                "unit_list": [name] if present else [],
                "unit_files": [name] if present else [],
                "active": state,
                "main_pid": 44 if state == "active" else 0,
                "job_id": self.jobs.get(name, 0),
                "jobs": [self.jobs[name]] if name in self.jobs else [],
                "links": self.links.get(name, {}),
                "members": self.members.get(name, []),
                "cgroup_exists": state == "active",
            }
        return rows

    def fresh(self, names, deadline):
        self.trace.append("fresh")
        return self.observe(names)

    def inert(self, names, deadline):
        self.trace.append("inert")
        return self.observe(names)

    def reload(self, deadline):
        self.trace.append("reload")
        self.t += self.advance_reload

    def verify_sources(self, prepared, deadline):
        self.trace.append("sources")

    def start(self, name, deadline):
        self.trace.append(("start", name))
        if name == self.failed_start:
            raise m.InstallRefusal("fake-start-outcome-unknown")
        self.states[name] = "active"

    def arm(self, prepared, deadline):
        self.trace.append("arm")
        if self.fail_arm:
            raise m.InstallRefusal("fake-arm-failed")
        return {"path": "/proof/arm.json", "sha256": "a" * 64, "bytes": 1}

    def recheck_arm(self, pin, deadline):
        self.trace.append("arm-recheck")
        assert self.states[self.prepared.start_order[0]] == "active"

    def started(self, name, deadline):
        self.trace.append(("started", name))
        return {"name": name, "observed_clock": self.clock().__dict__}

    def inspect_cleanup(self, prepared, deadline):
        self.trace.append("inspect-only")
        return {"status": "pending"}

    def audit(self, prepared, deadline):
        self.trace.append("audit-only")
        return {"status": "observed", "execution_qualified": False}


@pytest.fixture
def setup(monkeypatch, tmp_path):
    x = fixture_inputs(monkeypatch)
    raw = render.q.encode(x.plan)
    prepared = render.prepare(
        raw,
        render.q.sha(raw),
        life.Clock(x.plan["boot_id"], 100, 10**18),
        x.sources,
        before_execution=x.before,
        expected_before=x.expected,
        before_evidence={},
        approved_outer_intent_sha256="7" * 64,
    )
    files = m.Files._private_test_root(tmp_path)
    for path in [prepared.plan.value["input_root"], "/run", "/etc/systemd/system"]:
        files.path(path).mkdir(parents=True, exist_ok=True, mode=0o700)
    host = FakeHost(prepared, files)
    from scripts import strength_freshness_cpu_outer as outer

    owner = {
        "pid": 20,
        "start_ticks": 20,
        "ppid": 10,
        "pgid": 20,
        "sid": 20,
        "uid": 0,
        "boot_id": prepared.anchor.boot_id,
        "cgroup": "/scope",
        "pid_namespace_inode": 99,
    }
    enclosing = {
        "operator": owner,
        "supervisor": dict(owner, pid=10, start_ticks=10, ppid=1, pgid=10, sid=10),
    }
    start_body = {
        "format": "strength-freshness-dummy-start-v1",
        "schema_version": 1,
        "outer_intent_sha256": "7" * 64,
        "before_execution_pin": prepared.plan.value["preservation"]["before_execution"],
        "clock": {
            "boot_id": prepared.anchor.boot_id,
            "monotonic_ns": 100 * 10**9,
            "wall_ns": 10**18,
        },
    }
    start_path = str(Path(prepared.plan.value["input_root"]) / "dummy-start.json")
    start_raw = render.q.encode(start_body)
    start_pin = {
        "path": start_path,
        "sha256": m.sha(start_raw),
        "bytes": len(start_raw),
    }
    ack_raw = render.q.encode(
        outer.dummy_start_ack(
            outer_intent_sha256="7" * 64,
            nonce=prepared.anchor.nonce,
            start_pin=start_pin,
            before_execution_pin=start_body["before_execution_pin"],
            enclosing=enclosing,
        )
    )
    ack_path = str(Path(prepared.plan.value["input_root"]) / "dummy-start.ack.json")
    ack_pin = {"path": ack_path, "sha256": m.sha(ack_raw), "bytes": len(ack_raw)}
    for path, raw in ((start_path, start_raw), (ack_path, ack_raw)):
        files.path(path).write_bytes(raw)
        files.path(path).chmod(0o444)
    backend = m.Backend(
        prepared,
        host,
        files,
        intent_sha256="7" * 64,
        work_deadline=life.Clock(prepared.anchor.boot_id, 158, 10**18 + 58 * 10**9),
        start_authorization=m.StartAuthorization(ack_pin, start_pin, enclosing),
    )
    return backend, host, files


def rows(backend, files):
    return [
        json.loads(line)
        for line in files.read(backend.journal_path, m.MAX_JOURNAL_BYTES).splitlines()
    ]


def test_order_exact_artifacts_and_original_clock(setup):
    backend, host, files = setup
    result = backend.prepare_install()
    assert (
        result["status"] == "installed-and-armed"
        and result["execution_qualified"] is False
    )
    starts = [x[1] for x in host.trace if isinstance(x, tuple) and x[0] == "start"]
    assert starts == list(backend.prepared.start_order)
    assert host.trace.index("sources") < host.trace.index("arm")
    assert host.trace.count("arm-recheck") == 3
    journal = rows(backend, files)
    assert journal[0]["event"] == "installation-intent"
    assert all(row["binding"]["anchor"] == backend.anchor.as_dict() for row in journal)
    for item in backend.artifacts.values():
        assert files.read(item.path, len(item.data)) == item.data
        assert (files.stat(item.path).st_mode & 0o777) == item.mode
        created = next(
            i
            for i, r in enumerate(journal)
            if r["event"] == "created-inode" and r["data"]["path"] == item.path
        )
        intent = next(
            i
            for i, r in enumerate(journal)
            if r["event"] == "create-intent" and r["data"]["pin"]["path"] == item.path
        )
        assert intent < created


@pytest.mark.parametrize(
    "fault", ["file", "symlink", "job", "member", "link", "scratch"]
)
def test_preflight_refuses_existing_namespace_without_start_or_delete(
    setup, fault, tmp_path
):
    backend, host, files = setup
    item = backend.prepared.installed[0]
    if fault == "file":
        files.path(item.path).write_bytes(b"foreign")
    elif fault == "symlink":
        other = tmp_path / "foreign"
        other.write_bytes(b"safe")
        files.path(item.path).symlink_to(other)
    elif fault == "scratch":
        files.path(backend.plan["scratch_root"]).mkdir()
    else:
        name = backend.names[0]
        getattr(host, {"job": "jobs", "member": "members", "link": "links"}[fault])[
            name
        ] = {"foreign": "x"} if fault == "link" else [77]
    with pytest.raises(m.InstallRefusal):
        backend.prepare_install()
    assert not any(isinstance(x, tuple) and x[0] == "start" for x in host.trace)
    if fault in {"file", "symlink"}:
        assert files.path(item.path).exists()


def test_failed_arm_keeps_watchdog_and_never_starts_controls(setup):
    backend, host, _ = setup
    host.fail_arm = True
    with pytest.raises(m.InstallRefusal, match="arm-failed"):
        backend.prepare_install()
    assert [x for x in host.trace if isinstance(x, tuple) and x[0] == "start"] == [
        ("start", backend.prepared.start_order[0])
    ]
    with pytest.raises(m.InstallRefusal, match="not-inert"):
        backend.prearm_cleanup()
    assert host.states[backend.prepared.start_order[0]] == "active"


def test_control_start_intent_blocks_prearm_even_after_unknown_result(setup):
    backend, host, _ = setup
    host.failed_start = backend.prepared.start_order[1]
    with pytest.raises(m.InstallRefusal, match="outcome-unknown"):
        backend.prepare_install()
    host.states.clear()
    with pytest.raises(m.InstallRefusal, match="prearm-control-started"):
        backend.prearm_cleanup()


def test_crash_after_complete_file_has_exact_inode_cleanup_and_retains_journal(
    setup, monkeypatch
):
    backend, host, files = setup
    original = files.create
    count = 0

    def crash(item, journal):
        nonlocal count
        original(item, journal)
        count += 1
        if count == 3:
            raise RuntimeError("simulated crash")

    monkeypatch.setattr(files, "create", crash)
    with pytest.raises(RuntimeError):
        backend.prepare_install()
    result = backend.prearm_cleanup()
    assert len(result["removed"]) == 3
    assert files.stat(backend.journal_path) is not None
    assert backend.prearm_cleanup()["removed"] == []
    assert not any(isinstance(x, tuple) and x[0] == "start" for x in host.trace)


@pytest.mark.parametrize(
    "fault", ["unrecorded-inode", "partial", "replacement", "wrong-mode"]
)
def test_unknown_partial_or_changed_file_cannot_be_deleted(setup, monkeypatch, fault):
    backend, _host, files = setup
    original = m.Journal.record
    created_path = []

    def crash(journal, event, data):
        if event == "created-inode":
            created_path.append(data["path"])
            if fault == "unrecorded-inode":
                raise RuntimeError("before durable identity")
            original(journal, event, data)
            raise RuntimeError("before bytes")
        original(journal, event, data)

    monkeypatch.setattr(m.Journal, "record", crash)
    with pytest.raises(RuntimeError):
        backend.prepare_install()
    monkeypatch.setattr(m.Journal, "record", original)
    path = created_path[0]
    physical = files.path(path)
    item = backend.artifacts[path]
    if fault == "replacement":
        physical.unlink()
        physical.write_bytes(item.data)
        physical.chmod(item.mode)
    elif fault == "wrong-mode":
        physical.write_bytes(item.data)
        physical.chmod(0o644)
    with pytest.raises(m.InstallRefusal):
        backend.prearm_cleanup()
    assert physical.exists()


def test_original_deadline_is_not_renewed_after_slow_reload(setup):
    backend, host, files = setup
    host.advance_reload = 60
    with pytest.raises(m.InstallRefusal, match="deadline"):
        backend.prepare_install()
    assert not any(isinstance(x, tuple) and x[0] == "start" for x in host.trace)
    assert rows(backend, files)[-1]["event"] == "reload-intent"


def test_repeated_prepare_never_overwrites_journal_or_rearms(setup):
    backend, host, files = setup
    backend.prepare_install()
    before = files.read(backend.journal_path, m.MAX_JOURNAL_BYTES)
    starts = list(host.trace)
    with pytest.raises(FileExistsError):
        backend.prepare_install()
    assert files.read(backend.journal_path, m.MAX_JOURNAL_BYTES) == before
    assert host.trace == starts


def test_unknown_installed_path_cannot_expand_backend_scope(setup):
    backend, host, files = setup
    malicious = replace(
        backend.prepared,
        installed=(
            *backend.prepared.installed,
            render.Artifact("/etc/foreign", b"x", 0o644),
        ),
    )
    with pytest.raises(m.InstallRefusal, match="closed-targets"):
        m.Backend(
            malicious,
            host,
            files,
            intent_sha256="7" * 64,
            work_deadline=backend.work_deadline,
            start_authorization=backend.start_authorization,
        )


def test_read_operations_do_not_install_or_start(setup):
    backend, host, _ = setup
    assert backend.inspect_cleanup() == {"status": "pending"}
    assert backend.audit()["execution_qualified"] is False
    assert host.trace == ["inspect-only", "audit-only"]


def test_same_inode_rewrite_after_completed_receipt_is_not_owned_cleanup(
    setup, monkeypatch
):
    backend, _host, files = setup
    original = files.create
    written = []

    def crash(item, journal):
        original(item, journal)
        written.append(item)
        raise RuntimeError("after completed record")

    monkeypatch.setattr(files, "create", crash)
    with pytest.raises(RuntimeError):
        backend.prepare_install()
    item = written[0]
    path = files.path(item.path)
    before_inode = path.stat().st_ino
    path.chmod(0o600)
    path.write_bytes(item.data)
    path.chmod(item.mode)
    assert path.stat().st_ino == before_inode
    with pytest.raises(m.InstallRefusal, match="file-version"):
        backend.prearm_cleanup()
    assert path.exists()


def test_concurrent_prearm_journal_lock_refuses_without_mutation(setup, monkeypatch):
    backend, _host, files = setup

    def crash(*args):
        raise RuntimeError("before any artifact")

    monkeypatch.setattr(files, "create", crash)
    with pytest.raises(RuntimeError):
        backend.prepare_install()
    journal = m.Journal(
        files, backend.journal_path, backend.binding, backend.check, create=False
    )
    before = files.read(backend.journal_path, m.MAX_JOURNAL_BYTES)
    try:
        with pytest.raises(BlockingIOError):
            backend.prearm_cleanup()
        assert files.read(backend.journal_path, m.MAX_JOURNAL_BYTES) == before
    finally:
        journal.close()


@pytest.mark.parametrize("fault", ["missing", "wrong-mode", "changed-bytes"])
def test_backend_requires_actual_immutable_start_ack_before_journal_or_setup(
    setup, fault
):
    backend, host, files = setup
    ack = files.path(backend.start_authorization.ack_pin["path"])
    if fault == "missing":
        ack.unlink()
    elif fault == "wrong-mode":
        ack.chmod(0o600)
    else:
        ack.chmod(0o600)
        ack.write_bytes(ack.read_bytes() + b"changed")
        ack.chmod(0o444)
    with pytest.raises((m.InstallRefusal, FileNotFoundError)):
        backend.prepare_install()
    assert host.trace == []
    assert files.stat(backend.journal_path) is None


@pytest.mark.parametrize("job", ["", "0", "7"])
def test_real_site_observer_uses_systemd255_empty_scalar_job_and_exact_join(job):
    name = "edgeconnect-cpuqual-" + "a" * 32 + "-example.service"
    installed = "/etc/systemd/system/" + name

    class IO:
        def command(self, argv, deadline):
            assert deadline == 150
            if argv == list(render.q.linux.JOBS_COMMAND):
                return f"7 {name} start waiting\n" if job == "7" else ""
            if argv[1] == "list-units":
                return f"{name} loaded inactive dead fixture\n"
            if argv[1] == "list-unit-files":
                return f"{name} disabled enabled\n"
            assert argv[1:3] == ["show", name]
            return (
                "\n".join(
                    [
                        f"Id={name}",
                        "LoadState=loaded",
                        "ActiveState=inactive",
                        "SubState=dead",
                        f"Job={job}",
                        f"FragmentPath={installed}",
                        "DropInPaths=",
                        "MainPID=0",
                    ]
                )
                + "\n"
            )

        def members(self, path, deadline):
            return ()

        def boot_links(self, name, deadline):
            return {}

        def exists(self, path):
            return False

    host = object.__new__(m.SiteHost)
    host.plan = {"units": {name: {"installed_path": installed}}}
    host.__dict__["io"] = IO()
    observed = host.inert((name,), 150)[name]
    assert observed["properties"]["Job"] == job
    assert observed["job_id"] == (7 if job == "7" else 0)
    assert bool(observed["jobs"]) == (job == "7")
