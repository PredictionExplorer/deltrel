"""Collector IO contract tests: fake filesystem/process facts, never live /proc/GPU."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
from scripts import strength_freshness_cpu_readonly as r

BOOT = "11111111-1111-1111-1111-111111111111"
UNIT = "edgeconnect-fixture.service"
GROUP = "/system.slice/" + UNIT


def proc_stat(pid=31, start=90, ppid=12, state="S"):
    fields = [state, str(ppid)] + ["0"] * 17 + [str(start)] + ["0"] * 8
    return f"{pid} (worker (odd) name) ".encode() + " ".join(fields).encode() + b"\n"


class Fake:
    """Every external value is explicit; unknown reads never touch the host."""

    def __init__(self):
        self.ns = 100_000_000_000
        self.files = {
            "/proc/sys/kernel/random/boot_id": (BOOT + "\n").encode(),
            "/run/fixture/heartbeat.json": b'{"heartbeat_ns":123}\n',
            "/run/fixture/metrics.jsonl": b'partial\n{"step":1}\n{"step":2}\nlast',
        }
        self.dirs = {
            "/sys/fs/cgroup" + GROUP: [],
            "/etc/systemd/system": [],
            "/run/systemd/system": [],
        }
        self.files["/sys/fs/cgroup" + GROUP + "/cgroup.procs"] = b"31\n"
        self.proc_data = {
            31: {
                "stat_before": proc_stat(),
                "stat_after": proc_stat(),
                "cgroup_before": ("0::" + GROUP + "\n").encode(),
                "cgroup_after": ("0::" + GROUP + "\n").encode(),
                "exe": "/qualified/python",
                "cwd": "/qualified/training",
                "cmdline": b"python\0--secret\0private-argv\0",
                "environ": b"PASSWORD=private-env\0PYTHONPATH=/qualified\0",
                "maps": b"mapped /qualified/native.so\n",
            }
        }
        self.calls = []
        self.stdout = b"Id=" + UNIT.encode() + b"\nEnvironment=TOKEN=private-env\n"
        self.stderr = b""
        self.returncode = 0
        self.links = {}
        self.stats = {}

    def monotonic_ns(self):
        return self.ns

    def wall_ns(self):
        return 10**18 + self.ns

    def boottime_ns(self):
        return self.ns + 12345

    def hertz(self):
        return 100

    def read_file(self, path, maximum, deadline, *, tail=False):
        assert self.ns / 1e9 < deadline
        self.calls.append(("read", path))
        if path not in self.files:
            raise FileNotFoundError(path)
        raw = self.files[path]
        if not tail:
            r.require(len(raw) <= maximum, "file-size-limit")
        offset = max(0, len(raw) - maximum) if tail else 0
        data = raw[offset:]
        info = {"inode": 1, "bytes": len(raw)}
        return data, {
            "stat_before": info,
            "stat_after": info,
            "offset": offset,
            "end_offset": len(raw),
        }

    def proc(self, pid, deadline, *, details=False, expected=None, maps=True):
        self.calls.append(("proc", pid, details))
        assert pid in self.proc_data, "no real/proc fallback"
        v = dict(self.proc_data[pid])
        if details:
            actual = r._process_identity(pid, v["stat_before"], v["cgroup_before"])
            r.require(
                expected is not None
                and actual["start_ticks"] == expected.start_ticks
                and actual["cgroup"] == expected.cgroup,
                "private-process-admission-drift",
            )
        if not maps:
            v.pop("maps", None)
        return (
            v
            if details
            else {
                k: v[k]
                for k in ("stat_before", "stat_after", "cgroup_before", "cgroup_after")
            }
        )

    def directory(self, path, deadline):
        self.calls.append(("directory", path))
        assert path in self.dirs, "no real directory fallback"
        return self.dirs[path]

    def command(self, argv, deadline):
        self.calls.append(("command", argv))
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
        }

    def namespace(self, pid, deadline):
        self.calls.append(("namespace", pid))
        return {
            key: {"literal": key + ":[123]", "stat": {"device": 4, "inode": 123}}
            for key in ("pid", "time")
        }

    def link(self, path, deadline):
        self.calls.append(("link", path))
        return self.links[path]

    def stat_path(self, path, deadline, *, follow=False):
        self.calls.append(("stat", path))
        return self.stats[path]


def scope():
    return r.ReadScope(
        {UNIT: "service", "fixture.timer": "timer"},
        ("multi-user.target",),
        {"heartbeat": r.FileKey("/run/fixture/heartbeat.json")},
        {"metrics": r.FileKey("/run/fixture/metrics.jsonl", 32)},
        {"native": r.CachedFile("/qualified/native.so", "/qualified/native.so")},
    )


@pytest.fixture
def fixture():
    backend = Fake()
    return r.ReadOnlyIO(scope(), deadline=110, backend=backend), backend


def test_closed_reads_keep_raw_bytes_and_missing_properties(fixture):
    io, backend = fixture
    got = io.read("heartbeat")
    assert got.value == backend.files["/run/fixture/heartbeat.json"]
    assert got.audit["raw"]["sha256"] == r.sha(got.value)
    result = io.query("unit", UNIT)
    assert "EnvironmentFiles" not in result.value
    assert "private-env" not in repr(result) and "private-env" not in json.dumps(
        result.audit
    )
    assert result.audit["read_start"]["boot_id"] == BOOT


def test_scope_copies_inputs_and_rejects_unknown_properties():
    units = {UNIT: "service"}
    spec = r.ReadScope(units, (), {}, {}, {})
    units["other.service"] = "service"
    assert "other.service" not in spec.units
    with pytest.raises(r.ReadRefusal, match="property-not-allowlisted"):
        replace(
            spec,
            property_sets={
                "service": ("ArbitraryDanger",),
                "timer": r.TIMER_PROPERTIES,
            },
        )


@pytest.mark.parametrize(
    "name", ["--user.service", "-Hhost.service", "x;stop.service", "../bad.service"]
)
def test_unit_option_and_path_injection_refused(name):
    with pytest.raises(r.ReadRefusal):
        r.ReadScope({name: "service"}, (), {}, {}, {})


@pytest.mark.parametrize(
    "kind,target",
    [
        ("stop", UNIT),
        ("unit", "foreign.service"),
        ("registered-target", "rescue.target"),
        ("gpu-owners", "all"),
        ("jobs", UNIT),
    ],
)
def test_no_unknown_command_or_target_reaches_backend(fixture, kind, target):
    io, backend = fixture
    with pytest.raises(r.ReadRefusal):
        io.query(kind, target)
    assert not backend.calls


@pytest.mark.parametrize("key", ["/etc/passwd", "../metrics", "checkpoint"])
def test_arbitrary_paths_are_not_file_keys(fixture, key):
    io, backend = fixture
    with pytest.raises(r.ReadRefusal):
        io.read(key)
    assert not backend.calls


def test_no_model_or_replay_bytes_can_be_registered():
    for name in ("checkpoint.pt", "shard.npz", "manifest.sqlite3", "native.so"):
        with pytest.raises(r.ReadRefusal, match="model-replay"):
            r.ReadScope({UNIT: "service"}, (), {"x": r.FileKey("/run/" + name)}, {}, {})


def test_tail_preserves_slice_offset_and_partial_lines_for_normalizer(fixture):
    io, backend = fixture
    result = io.tail("metrics")
    assert result.value == backend.files["/run/fixture/metrics.jsonl"][-32:]
    assert (
        result.audit["offset"] == len(backend.files["/run/fixture/metrics.jsonl"]) - 32
    )
    assert result.value.endswith(b"last")


def test_process_admission_and_private_origin_values(fixture):
    io, backend = fixture
    members = io.members(UNIT).value
    assert members == (r.ProcessAdmission(31, 90, GROUP),)
    result = io.process(members[0])
    assert result.value["environ"].startswith(b"PASSWORD=")
    assert result.value["start_ticks"] == 90 and result.value["ppid"] == 12
    assert "private-env" not in repr(result) and "private-argv" not in json.dumps(
        result.audit
    )
    with pytest.raises(r.ReadRefusal, match="not-admitted"):
        io.process(r.ProcessAdmission(999, 1, GROUP))
    assert not any(c[:2] == ("proc", 999) for c in backend.calls)


@pytest.mark.parametrize("fault", ["start", "group", "during"])
def test_pid_reuse_or_movement_refuses_private_origin_read(fixture, fault):
    io, backend = fixture
    admitted = io.members(UNIT).value[0]
    if fault == "group":
        backend.proc_data[31]["cgroup_before"] = backend.proc_data[31][
            "cgroup_after"
        ] = b"0::/foreign\n"
    else:
        backend.proc_data[31]["stat_after"] = proc_stat(start=91)
        if fault == "start":
            backend.proc_data[31]["stat_before"] = proc_stat(start=91)
    with pytest.raises(r.ReadRefusal):
        io.process(admitted)


def test_foreign_membership_never_admits_a_pid(fixture):
    io, backend = fixture
    backend.proc_data[31]["cgroup_before"] = backend.proc_data[31]["cgroup_after"] = (
        b"0::/foreign\n"
    )
    with pytest.raises(r.ReadRefusal, match="foreign-cgroup"):
        io.members(UNIT)
    with pytest.raises(r.ReadRefusal, match="not-admitted"):
        io.process(r.ProcessAdmission(31, 90, GROUP))


def test_namespace_and_birth_clocks_remain_distinct(fixture):
    io, backend = fixture
    got = io.birth_bracket().value
    assert got["boottime_before_ns"] != got["monotonic_before_ns"]
    assert got["clock_ticks_per_second"] == 100
    assert io.namespaces("self").value["time"]["literal"] == "time:[123]"
    assert io.namespaces(1).value["pid"]["stat"]["inode"] == 123
    with pytest.raises(r.ReadRefusal):
        io.namespaces(12345)


def test_boot_change_or_expired_budget_does_not_return_snapshot(fixture):
    io, backend = fixture
    io.clock()
    backend.files["/proc/sys/kernel/random/boot_id"] = (
        "2" * 8 + "-1111-1111-1111-111111111111\n"
    ).encode()
    with pytest.raises(r.ReadRefusal, match="boot-changed"):
        io.read("heartbeat")
    backend.ns = 111_000_000_000
    with pytest.raises(r.ReadRefusal, match="deadline"):
        io.query("jobs")


def test_stat_reuse_never_claims_a_new_content_hash(fixture):
    io, backend = fixture
    backend.stats["/qualified/native.so"] = {
        "device": 4,
        "inode": 8,
        "bytes": 999,
        "mode": 0o444,
    }
    result = io.stat_cached("native")
    assert result.audit["content_hashed"] is False
    assert not any(c == ("read", "/qualified/native.so") for c in backend.calls)


def test_boot_links_include_aliases_and_literal_targets(fixture):
    io, backend = fixture
    parent = "/etc/systemd/system/multi-user.target.wants"
    backend.dirs["/etc/systemd/system"] = [
        {"name": "multi-user.target.wants", "directory": True, "symlink": False}
    ]
    backend.dirs[parent] = [
        {"name": "unexpected-alias.service", "directory": False, "symlink": True}
    ]
    path = parent + "/unexpected-alias.service"
    backend.links[path] = {
        "literal": "../" + UNIT,
        "resolved": "/etc/systemd/system/" + UNIT,
    }
    backend.stats[path] = {"inode": 3, "bytes": len(UNIT) + 3}
    result = io.boot_links(UNIT)
    assert result.value[0]["literal"] == "../" + UNIT
    assert result.value[0]["path"] == path


def test_fresh_interpreter_import_has_no_training_or_torch_modules():
    path = Path(r.__file__).resolve()
    source = f"""import importlib.util,sys
spec=importlib.util.spec_from_file_location("private_readonly",{str(path)!r})
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
assert not any(n=="torch" or n.startswith(("torch.","deltreltrain","startrain")) for n in sys.modules)
print("stdlib-only")
"""
    result = subprocess.run(
        [sys.executable, "-B", "-S", "-E", "-c", source],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    assert result.stdout.strip() == "stdlib-only"


@pytest.mark.parametrize("mode", ["timeout", "overflow"])
def test_real_local_metadata_child_is_bounded_and_reaped(tmp_path, mode):
    path = Path(r.__file__).resolve()
    pidfile = tmp_path / "pid"
    program = (
        f"""import os,time
open({str(pidfile)!r},"w").write(str(os.getpid()))
"""
        + ("print('x'*1200000,flush=True)\n" if mode == "overflow" else "")
        + "time.sleep(10)\n"
    )
    source = f"""import importlib.util,sys,subprocess,signal,time,os
spec=importlib.util.spec_from_file_location("private_readonly",{str(path)!r})
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
signal.signal(signal.SIGCHLD,signal.SIG_DFL)
original=subprocess.Popen
class Factory:
 def __call__(self,argv,**kwargs):
  assert argv==("systemctl","get-default")
  return original([sys.executable,"-B","-S","-E","-c",{program!r}],**kwargs)
m.subprocess.Popen=Factory()
start=time.monotonic()
try:m.System().command(("systemctl","get-default"),start+1)
except m.ReadRefusal as e:
 assert str(e) in ("command-timeout","command-output-limit"),str(e)
 assert time.monotonic()-start<1.5
else:raise AssertionError("unbounded helper passed")
pid=int(open({str(pidfile)!r}).read())
try:os.kill(pid,0)
except ProcessLookupError:pass
else:raise AssertionError("owned helper survived")
print("bounded-and-reaped")
"""
    result = subprocess.run(
        [sys.executable, "-B", "-S", "-E", "-c", source],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    assert result.stdout.strip() == "bounded-and-reaped"


def test_complete_source_stat_inventory_has_explicit_large_key_bound(fixture):
    io, backend = fixture
    cached = {
        str(i): r.CachedFile(f"/qualified/source-{i}.py", f"/qualified/source-{i}.py")
        for i in range(1024)
    }
    spec = replace(scope(), cached=cached)
    for entry in cached.values():
        backend.stats[entry.literal] = {"inode": 8, "bytes": 10, "mode": 0o444}
    io = r.ReadOnlyIO(spec, deadline=110, backend=backend)
    observed = io.stat_cached("1023")
    assert observed.audit["content_hashed"] is False
    cached["overflow"] = r.CachedFile("/x", "/x")
    with pytest.raises(r.ReadRefusal, match="inventory-limit"):
        replace(spec, cached=cached)


def test_auxiliary_origin_can_omit_maps_without_faking_empty_maps(fixture):
    io, backend = fixture
    admission = io.members(UNIT).value[0]
    observed = io.process(admission, maps=False)
    assert observed.value["maps"] is None
    assert observed.audit["maps_read"] is False
    assert io.process(admission).value["maps"] == backend.proc_data[31]["maps"]


def test_deadline_failure_audit_never_invents_a_boot_recheck(fixture):
    io, backend = fixture

    def failed(argv, deadline):
        backend.ns = 111_000_000_000
        raise r.ReadRefusal(
            "command-timeout", {"stdout_sha256": "a" * 64, "stdout_bytes": 0}
        )

    backend.command = failed
    with pytest.raises(r.ReadRefusal, match="command-timeout") as caught:
        io.query("jobs")
    assert caught.value.audit["read_start"]["boot_id"] == BOOT
    assert caught.value.audit["read_end"]["boot_id"] is None
    assert caught.value.audit["boot_rechecked"] is False


def test_low_capture_limit_refuses_without_unbounded_output(fixture):
    previous, fake = fixture
    io = r.ReadOnlyIO(
        previous.scope, deadline=previous.deadline, backend=fake, maximum_bytes=100
    )
    with pytest.raises(r.ReadRefusal, match="capture"):
        io.read("heartbeat")


def test_syscall_boundary_itself_has_no_arbitrary_or_mutating_command():
    for command in (
        ("sh", "-c", "true"),
        ("systemctl", "stop", UNIT),
        ("systemctl", "show", "--user", "--all", "--no-pager", "--property=Id"),
    ):
        with pytest.raises(r.ReadRefusal, match="not-readonly"):
            r.System().command(command, 1)


def manager_fake():
    fake = Fake()
    fake.proc_data[1] = {
        "stat_before": proc_stat(pid=1, start=1, ppid=0),
        "stat_after": proc_stat(pid=1, start=1, ppid=0),
        "cgroup_before": b"0::/init.scope\n",
        "cgroup_after": b"0::/init.scope\n",
    }
    return fake


def test_fixed_manager_identity_does_not_admit_or_read_private_pid1():
    fake = manager_fake()
    io = r.ReadOnlyIO(scope(), deadline=200, backend=fake)
    observed = io.manager_identity()
    assert observed.value["pid"] == 1 and observed.value["start_ticks"] == 1
    assert observed.value["boot_id"] == BOOT
    assert [c for c in fake.calls if c[0] == "proc"] == [
        ("proc", 1, False),
        ("proc", 1, False),
    ]
    assert not io._admitted
    assert "environ" not in json.dumps(observed.audit) and "cmdline" not in json.dumps(
        observed.audit
    )


def test_manager_identity_refuses_namespace_mismatch():
    fake = manager_fake()
    original = fake.namespace

    def namespace(pid, deadline):
        value = original(pid, deadline)
        if pid == 1:
            value["pid"]["literal"] = "pid:[999]"
        return value

    fake.namespace = namespace
    with pytest.raises(r.ReadRefusal, match="manager-namespace-mismatch"):
        r.ReadOnlyIO(scope(), deadline=200, backend=fake).manager_identity()


def test_manager_identity_refuses_changed_lifetime_between_brackets():
    fake = manager_fake()
    original = fake.proc
    count = 0

    def proc(pid, deadline, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            fake.proc_data[1]["stat_before"] = proc_stat(pid=1, start=2, ppid=0)
            fake.proc_data[1]["stat_after"] = proc_stat(pid=1, start=2, ppid=0)
        return original(pid, deadline, **kwargs)

    fake.proc = proc
    with pytest.raises(r.ReadRefusal, match="manager-lifetime-raced"):
        r.ReadOnlyIO(scope(), deadline=200, backend=fake).manager_identity()


def test_jobs_query_uses_systemd255_closed_plain_full_vector():
    fake = Fake()
    io = r.ReadOnlyIO(scope(), deadline=200, backend=fake)
    io.query("jobs")
    assert (
        "command",
        (
            "systemctl",
            "list-jobs",
            "--all",
            "--no-pager",
            "--no-legend",
            "--plain",
            "--full",
        ),
    ) in fake.calls


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, r.MAX_CAPTURE_BYTES + 1])
def test_per_capture_budget_cannot_expand_or_be_ambiguous(budget):
    with pytest.raises(r.ReadRefusal, match="capture-budget-range"):
        r.ReadOnlyIO(scope(), deadline=200, backend=Fake(), maximum_bytes=budget)


def test_smaller_metadata_budget_is_shared_across_repeated_operations():
    io = r.ReadOnlyIO(scope(), deadline=200, backend=Fake(), maximum_bytes=1000)
    io.clock()
    with pytest.raises(r.ReadRefusal, match="capture-output-budget"):
        for _ in range(20):
            io.clock()
    assert io.maximum_bytes == 1000


def test_default_metadata_budget_remains_32mib():
    assert (
        r.ReadOnlyIO(scope(), deadline=200, backend=Fake()).maximum_bytes == 32 * 2**20
    )


def test_real_command_gate_accepts_only_corrected_job_query():
    assert r._readonly_vector(
        (
            "systemctl",
            "list-jobs",
            "--all",
            "--no-pager",
            "--no-legend",
            "--plain",
            "--full",
        )
    )
    assert not r._readonly_vector(
        ("systemctl", "list-jobs", "--all", "--no-pager", "--output=json")
    )
    assert not r._readonly_vector(
        (
            "systemctl",
            "list-jobs",
            "--all",
            "--no-pager",
            "--no-legend",
            "--plain",
            "--full",
            "foreign.service",
        )
    )
