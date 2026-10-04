"""Real private-file IO with synthetic writer/clock admission, not target proof."""

from copy import deepcopy

import pytest

from scripts import strength_freshness_cpu_learner_window as proof
from scripts import strength_freshness_cpu_readonly as readonly
from tests.test_strength_freshness_cpu_learner_window import (
    BASE,
    event,
    metric,
    sample_window,
)

BOOT = "11111111-2222-3333-4444-555555555555"


def clock(seconds):
    return {
        "boot_id": BOOT,
        "monotonic_ns": seconds * proof.SECOND,
        "wall_ns": BASE + seconds * proof.SECOND,
    }


class ClockedFiles(readonly.System):
    """Keep actual stat/open/pread behavior; replace only observation clocks."""

    seconds = 0

    def monotonic_ns(self):
        return self.seconds * proof.SECOND

    def wall_ns(self):
        return BASE + self.monotonic_ns()

    def read_file(self, path, maximum, deadline, *, tail=False, charge, allowance):
        if path == "/proc/sys/kernel/random/boot_id":
            raw = (BOOT + "\n").encode()
            readonly.require(len(raw) <= allowance, "capture-byte-budget")
            charge(len(raw))
            return raw, {}
        return super().read_file(
            path, maximum, deadline, tail=tail, charge=charge, allowance=allowance
        )


def envelope(observation):
    return {"value": observation.value, "audit": dict(observation.audit)}


def collected_window(
    tmp_path, *, phase="after", crossed_failure=False, grow_after_end=False
):
    window, expected, _ = sample_window(phase=phase)
    expected_window = deepcopy(
        {
            key: window[key]
            for key in ("phase", "phase_start", "deadline", "cleanup_clock")
        }
    )
    for key in ("phase_start", "deadline", "cleanup_clock"):
        if expected_window[key] is not None:
            expected_window[key]["boot_id"] = BOOT
    path = tmp_path / "metrics.jsonl"
    first = (
        proof.encoded(event(error="PRIVATE_FAILURE"))
        if crossed_failure
        else proof.encoded(event())
    )
    cut = len(first) // 2 if crossed_failure else len(first)
    path.write_bytes(first[:cut])
    backend = ClockedFiles()
    scope = readonly.ReadScope(
        {"fixture.service": "service"},
        (),
        {},
        {"metrics": readonly.FileKey(str(path), readonly.MAX_TAIL_BYTES)},
        {},
    )
    io = readonly.ReadOnlyIO(scope, deadline=100, backend=backend)
    backend.seconds = 2
    before = envelope(io.append_fence("metrics"))
    if crossed_failure:
        with path.open("ab") as stream:
            stream.write(first[cut:])
    if phase == "after":
        with path.open("ab") as stream:
            stream.write(proof.encoded(event(6)))
        backend.seconds = 10
        causal = envelope(io.append_fence("metrics"))
    else:
        causal = deepcopy(before)
    with path.open("ab") as stream:
        stream.write(proof.encoded(metric(1010, 9)))
        stream.write(proof.encoded(metric(1020, 13)))
    backend.seconds = 14
    end = envelope(io.append_fence("metrics"))
    if grow_after_end:
        with path.open("ab") as stream:
            stream.write(proof.encoded(event(15, error="PRIVATE_LATE_FAILURE")))
    identity = before["value"]["file_identity"]
    span = {
        "file_identity": identity,
        "start": before["value"]["prefix"]["start"],
        "end": end["value"]["size_at_fstat"],
    }
    backend.seconds = 16
    original = envelope(io.read_append_range("metrics", **span))
    backend.seconds = 18
    reread = envelope(io.read_append_range("metrics", **span))
    binding = expected["binding"]
    binding.update(
        metrics_path=str(path),
        output_root=str(path.parent),
        file_identity=deepcopy(identity),
    )
    binding["owner_key"]["boot_id"] = BOOT
    binding["owner_key"]["uid"] = identity["uid"]
    writer = proof.encoded(binding)
    expected.update(evidence_sha256=proof.sha(writer), evidence_bytes=len(writer))
    for key in ("phase_start", "deadline", "final_clock", "cleanup_clock"):
        if window[key] is not None:
            window[key]["boot_id"] = BOOT
    for observation in window["owner_observations"]:
        observation["owner_key"] = deepcopy(binding["owner_key"])
        observation["writer_evidence_sha256"] = expected["evidence_sha256"]
        observation["read_start"]["boot_id"] = BOOT
        observation["read_end"]["boot_id"] = BOOT
    window.update(
        before_fence=before,
        causal_fence=causal,
        end_fence=end,
        segments=[original],
        rereads=[reread],
    )
    return window, expected, writer, expected_window


@pytest.mark.parametrize("phase", ["before", "after"])
def test_private_real_io_composes_with_conditional_append_proof(tmp_path, phase):
    window, expected, writer, expected_window = collected_window(tmp_path, phase=phase)
    result = proof.verify_append(
        window,
        expected_writer=expected,
        writer_evidence=writer,
        expected_window=expected_window,
    )
    public = result.public()
    assert public["minimum_serial_intervals"] == 1
    assert public["full_preservation"] is False
    assert public["execution_qualified"] is False
    assert public["historical_birth_qualified"] is False
    assert str(tmp_path) not in repr(result)
    assert str(tmp_path) not in str(public)


def test_failure_crossing_initial_fence_is_not_hidden_by_later_work(tmp_path):
    window, expected, writer, expected_window = collected_window(
        tmp_path, crossed_failure=True
    )
    with pytest.raises(proof.AppendRefusal) as error:
        proof.verify_append(
            window,
            expected_writer=expected,
            writer_evidence=writer,
            expected_window=expected_window,
        )
    assert "PRIVATE_FAILURE" not in str(error.value)


def test_changed_reread_bytes_cannot_be_relabelled_as_stable_history(tmp_path):
    window, expected, writer, expected_window = collected_window(tmp_path)
    reread = window["rereads"][0]
    raw = reread["value"]["raw"].replace(b'"outcome":0.5', b'"outcome":0.6', 1)
    assert raw != reread["value"]["raw"]
    reread["value"].update(raw=raw, sha256=proof.sha(raw))
    reread["audit"]["raw"] = {"sha256": proof.sha(raw), "bytes": len(raw)}
    with pytest.raises(proof.AppendRefusal):
        proof.verify_append(
            window,
            expected_writer=expected,
            writer_evidence=writer,
            expected_window=expected_window,
        )


def test_unread_growth_after_fixed_eof_cannot_claim_complete_window(tmp_path):
    window, expected, writer, expected_window = collected_window(
        tmp_path, grow_after_end=True
    )
    assert (
        window["segments"][0]["value"]["size_before"]
        > window["end_fence"]["value"]["size_at_fstat"]
    )
    with pytest.raises(proof.AppendIncomplete) as error:
        proof.verify_append(
            window,
            expected_writer=expected,
            writer_evidence=writer,
            expected_window=expected_window,
        )
    assert "PRIVATE_LATE_FAILURE" not in str(error.value)


def test_conditional_append_module_does_not_admit_missing_historical_birth():
    from scripts import strength_freshness_cpu_collect_identity as identity
    from tests.test_strength_freshness_cpu_collect_identity import (
        collector,
        identity_fixture,
    )

    registration, io, _ = identity_fixture()
    registration["birth_reference"]["qualification_sha256"] = None
    with pytest.raises(identity.CollectionRefusal):
        collector(registration, io)
    assert io.calls == []
