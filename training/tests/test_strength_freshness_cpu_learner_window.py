"""Pure synthetic append fixtures, never writer or target qualification."""

from copy import deepcopy
import json

import pytest

from scripts import strength_freshness_cpu_learner_window as m

BASE = 1800000000000000000


def clock(seconds):
    return {
        "boot_id": "fixture-boot",
        "monotonic_ns": int(seconds * m.SECOND),
        "wall_ns": BASE + int(seconds * m.SECOND),
    }


def metric(step, stamp):
    return {
        "schema_version": 1,
        "worker": "learner",
        "timestamp_ns": BASE + stamp * m.SECOND,
        "step": step,
        "losses": {"policy": 1.0, "outcome": 0.5},
        "gradient_norm": 1.0,
        "gradient_pre_clip_norm": 1.0,
        "gradient_post_clip_norm": 0.5,
        "nonfinite_loss_count": 0,
        "nonfinite_gradient_count": 0,
        "gradient_diagnostics": {
            "global_norm_finite": True,
            "nonfinite_gradient_tensors": 0,
            "batch": {
                "rows": 512,
                "six_mode_unknown": {},
                "label_availability": {"policy": 512, "outcome": 500},
            },
        },
        "learning_rates": [0.1, 0.01, 0.001],
        "ema": {"decay": 0.9999, "num_updates": step - 1000},
        "replay_minimum_shard_id_exclusive": 1,
        "utd_segment_baseline_committed_replay_samples": 10,
        "utd_segment_baseline_examples_consumed": 20,
        "utd_segment_target_updates_per_new_sample": 1.5,
        "segment_updates_per_new_sample": 1.0,
        "policy_batch_metrics": {"unknown_provenance_rows": 0},
    }


def event(stamp=1, **extra):
    return {
        "schema_version": 1,
        "worker": "learner",
        "timestamp_ns": BASE + stamp * m.SECOND,
        "step": 1000,
        "event": "utd_wait",
        **extra,
    }


def sample_window(
    *,
    phase="after",
    rows=None,
    between=None,
    b_partial=False,
    f_partial=False,
    trailing=b"",
):
    """Return (window, expected_writer, private writer bytes), exact IO envelopes."""
    recipe = {
        "learning_rates": [0.1, 0.01, 0.001],
        "ema_decay": 0.9999,
        "bootstrap_step": 1000,
        "replay_minimum_shard_id_exclusive": 1,
        "utd_segment_baseline_committed_replay_samples": 10,
        "utd_segment_baseline_examples_consumed": 20,
        "utd_segment_target_updates_per_new_sample": 1.5,
    }
    owner = {
        "boot_id": "fixture-boot",
        "pid": 123,
        "ppid": 100,
        "start_ticks": 500,
        "cgroup": "/fixture.service",
        "invocation_id": "a" * 32,
        "pid_namespace_inode": 42,
        "time_namespace_inode": 43,
        "clock_ticks_per_second": 100,
        "origin_sha256": "b" * 64,
        "uid": 1000,
    }
    binding = {
        "format": m.WRITER,
        "schema_version": 1,
        "source_contract_sha256": m.SOURCE_CONTRACT,
        "source_commit": m.R3_COMMIT,
        "learner_source_sha256": m.LEARNER_SOURCE,
        "training_source_sha256": m.TRAINING_SOURCE,
        "metrics_key": "metrics",
        "metrics_path": "/fixture/learner/metrics.jsonl",
        "run_id": "fixture-run",
        "output_root": "/fixture/learner",
        "owner_key": owner,
        "file_identity": {
            "path": "/fixture/learner/metrics.jsonl",
            "device": 1,
            "inode": 2,
            "uid": 1000,
            "gid": 1000,
            "mode": 0o644,
        },
        "command_sha256": "c" * 64,
        "imports_sha256": "d" * 64,
        "native_sha256": "e" * 64,
        "access_sha256": "f" * 64,
        "qualification_sha256": "1" * 64,
        "recipe_sha256": m.sha(m.encoded(recipe)),
    }
    writer = m.encoded(binding)
    expected = {
        "binding": binding,
        "evidence_sha256": m.sha(writer),
        "evidence_bytes": len(writer),
    }
    prefix = m.encoded(event())
    middle = m.encoded(event(6)) if between is None else m.encoded(between)
    row_values = [metric(1010, 9), metric(1020, 13)] if rows is None else rows
    suffix = (
        b"".join(
            m.encoded(row) if not isinstance(row, bytes) else row for row in row_values
        )
        + trailing
    )
    raw = prefix + (middle if phase == "after" else b"") + suffix
    b_size = len(prefix) // 2 if b_partial else len(prefix)
    f_size = len(prefix) + (len(middle) if phase == "after" else 0)
    if f_partial:
        f_size += len(m.encoded(row_values[0])) // 2
    if phase == "before":
        f_size = b_size
    identity = {
        "path": binding["metrics_path"],
        "device": 1,
        "inode": 2,
        "uid": 1000,
        "gid": 1000,
        "mode": 0o644,
    }

    def observed(kind, start, end, size, at):
        data = raw[start:end]
        stat = {
            **{k: v for k, v in identity.items() if k != "path"},
            "bytes": size,
            "mtime_ns": BASE,
            "ctime_ns": BASE,
            "links": 1,
        }
        value = {"registered_key": "metrics", "file_identity": deepcopy(identity)}
        span = {"start": start, "end": end, "raw": data, "sha256": m.sha(data)}
        if kind == "append-fence":
            value.update(
                size_at_fstat=size,
                prefix=span,
                line_boundary=end == 0 or data.endswith(b"\n"),
            )
        else:
            value.update(**span, size_before=size, size_after=size)
        audit = {
            "operation": kind,
            "subject": "metrics",
            "read_start": clock(at),
            "read_end": clock(at + 1),
            "raw": {"sha256": m.sha(data), "bytes": len(data)},
            "named_before": dict(stat),
            "stat_before": dict(stat),
            "stat_after": dict(stat),
            "named_after": dict(stat),
            "offset": start,
            "end_offset": end,
        }
        return {"value": value, "audit": audit}

    b = observed("append-fence", 0, b_size, b_size, 2)
    f = (
        deepcopy(b)
        if phase == "before"
        else observed("append-fence", 0, f_size, f_size, 10)
    )
    e = observed("append-fence", 0, len(raw), len(raw), 14)
    original = observed("append-range", 0, len(raw), len(raw), 16)
    reread = observed("append-range", 0, len(raw), len(raw), 18)

    def own(t):
        return {
            "owner_key": deepcopy(owner),
            "writer_evidence_sha256": expected["evidence_sha256"],
            "read_start": clock(t),
            "read_end": clock(t + 1),
        }

    observations = [own(0), own(8), own(20)] if phase == "after" else [own(0), own(20)]
    w = {
        "format": m.WINDOW,
        "schema_version": 1,
        "phase": phase,
        "phase_start": clock(8 if phase == "after" else 0),
        "deadline": clock(100),
        "cleanup_clock": clock(7) if phase == "after" else None,
        "final_clock": clock(22),
        "before_fence": b,
        "causal_fence": f,
        "end_fence": e,
        "segments": [original],
        "rereads": [reread],
        "owner_observations": observations,
        "recipe": recipe,
    }
    return w, expected, writer


def verify(parts):
    w, expected, writer = parts
    return m.verify_append(
        w,
        expected_writer=expected,
        writer_evidence=writer,
        expected_window={
            k: deepcopy(w[k])
            for k in ("phase", "phase_start", "deadline", "cleanup_clock")
        },
    )


@pytest.mark.parametrize("phase", ["before", "after"])
def test_two_wholly_new_serial_records_give_only_conditional_proof(phase):
    parts = sample_window(phase=phase)
    proof = verify(parts)
    public = proof.public()
    assert (
        public["minimum_serial_intervals"] == 1
        and len(public["qualifying_records"]) == 2
    )
    assert (
        public["full_preservation"] is False and public["execution_qualified"] is False
    )
    assert (
        public["historical_birth_qualified"] is False
        and public["writer_authority_granted"] is False
    )
    assert "/fixture" not in json.dumps(public) and "fixture-run" not in repr(proof)
    public["status"] = "tampered"
    assert proof.public()["status"] != "tampered"
    assert public["raw_bytes_checked_including_rereads"] == len(parts[2]) + sum(
        len(parts[0][k]["value"]["prefix"]["raw"])
        for k in ("before_fence", "causal_fence", "end_fence")
    ) + sum(
        len(x["value"]["raw"]) for k in ("segments", "rereads") for x in parts[0][k]
    )


def test_m1_prepared_before_fence_is_allowed_but_cannot_alone_prove_work():
    w, expected, writer = sample_window()
    assert (
        metric(1010, 9)["timestamp_ns"]
        < w["causal_fence"]["audit"]["read_start"]["wall_ns"]
    )
    verify((w, expected, writer))
    with pytest.raises(m.AppendIncomplete, match="two-new"):
        verify(sample_window(rows=[metric(1010, 9)]))


def test_mid_record_fence_excludes_straddling_loss_but_keeps_two_later_rows():
    p = verify(
        sample_window(
            rows=[metric(1010, 9), metric(1020, 12), metric(1030, 13)], f_partial=True
        )
    ).public()
    assert [row["step"] for row in p["qualifying_records"]] == [1020, 1030]
    with pytest.raises(m.AppendIncomplete):
        verify(sample_window(f_partial=True))


def test_initial_partial_record_is_reconstructed_and_checked():
    verify(sample_window(b_partial=True))
    # Negative completed initial row, even though it began before B, is observed.
    parts = sample_window(b_partial=True)
    w, expected, writer = parts
    # Rebuild via the common bytes, with unchanged-length event replacement.
    original = w["segments"][0]["value"]["raw"]
    assert b"utd_wait" in original
    changed = original.replace(b"utd_wait", b"bad_wait", 1)
    replace_bytes(w, changed)
    with pytest.raises(m.AppendRefusal, match="unsupported-event"):
        verify((w, expected, writer))


def replace_bytes(w, raw):
    """Keep all declared observations consistent while mutating semantic bytes."""
    for key in ("before_fence", "causal_fence", "end_fence"):
        env = w[key]
        v = env["value"]
        span = v["prefix"]
        b = raw[span["start"] : span["end"]]
        span.update(raw=b, sha256=m.sha(b))
        v["line_boundary"] = span["end"] == 0 or b.endswith(b"\n")
        env["audit"]["raw"] = {"sha256": m.sha(b), "bytes": len(b)}
    for key in ("segments", "rereads"):
        for env in w[key]:
            v = env["value"]
            b = raw[v["start"] : v["end"]]
            v.update(raw=b, sha256=m.sha(b))
            env["audit"]["raw"] = {"sha256": m.sha(b), "bytes": len(b)}


@pytest.mark.parametrize(
    "negative",
    [
        event(6, error="private failure text"),
        event(6, phase="nonfinite_abort"),
        event(6, event="unknown_event"),
        event(6, nonfinite_gradient_count=1),
        event(
            6,
            gradient_diagnostics={
                "global_norm_finite": False,
                "nonfinite_gradient_tensors": 0,
            },
        ),
    ],
)
def test_negative_before_causal_fence_is_never_filtered_out(negative):
    with pytest.raises(m.AppendRefusal) as error:
        verify(sample_window(between=negative))
    assert "private failure text" not in str(error.value)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x["ema"].update(num_updates=0),
        lambda x: x.update(learning_rates=[1, 2, 3]),
        lambda x: x.update(segment_updates_per_new_sample=9),
        lambda x: x.update(utd_segment_baseline_examples_consumed=999),
        lambda x: x["gradient_diagnostics"].update(global_norm_finite=False),
        lambda x: x["gradient_diagnostics"]["batch"]["label_availability"].update(
            policy=0
        ),
        lambda x: x["policy_batch_metrics"].update(unknown_provenance_rows=1),
        lambda x: x.update(step=True),
        lambda x: x.update(event="unsupported"),
        lambda x: x["losses"].update(policy=float("nan")),
    ],
)
def test_full_loss_recipe_and_negative_checks(mutation):
    row = metric(1020, 13)
    mutation(row)
    if not all(not isinstance(v, float) or v == v for v in row["losses"].values()):
        raw = json.dumps(row).encode() + b"\n"
        parts = sample_window(rows=[metric(1010, 9), raw])
    else:
        parts = sample_window(rows=[metric(1010, 9), row])
    with pytest.raises(m.AppendRefusal):
        verify(parts)


@pytest.mark.parametrize(
    "kind",
    [
        "inode",
        "owner",
        "source",
        "gap",
        "overlap",
        "rewrite",
        "truncate",
        "boot",
        "future",
        "clock",
        "cleanup",
        "deadline",
        "old-format",
        "trusted-bool",
    ],
)
def test_structural_identity_clock_range_and_authority_refusals(kind):
    w, expected, evidence = sample_window()
    if kind == "inode":
        w["end_fence"]["value"]["file_identity"]["inode"] += 1
    elif kind == "owner":
        w["owner_observations"][-1]["owner_key"]["start_ticks"] += 1
    elif kind == "source":
        expected["binding"]["learner_source_sha256"] = "0" * 64
        evidence = m.encoded(expected["binding"])
        expected.update(evidence_sha256=m.sha(evidence), evidence_bytes=len(evidence))
    elif kind == "gap":
        w["segments"][0]["value"]["start"] += 1
    elif kind == "overlap":
        w["segments"].append(deepcopy(w["segments"][0]))
    elif kind == "rewrite":
        w["rereads"][0]["value"]["raw"] = b"x" + w["rereads"][0]["value"]["raw"][1:]
    elif kind == "truncate":
        w["segments"][0]["audit"]["stat_after"]["bytes"] = 0
    elif kind == "boot":
        w["final_clock"]["boot_id"] = "other"
    elif kind == "future":
        w["segments"][0]["audit"]["read_end"]["wall_ns"] = BASE + 5 * m.SECOND
    elif kind == "clock":
        w["rereads"][0]["audit"]["read_start"] = clock(1)
    elif kind == "cleanup":
        w["cleanup_clock"] = clock(10)
    elif kind == "deadline":
        w["deadline"] = clock(200)
    elif kind == "old-format":
        w["format"] = "strength-preservation-v2"
    else:
        expected["binding"]["trusted_writer"] = True
    with pytest.raises(m.AppendRefusal):
        verify((w, expected, evidence))


def test_final_partial_and_credit_wait_are_incomplete_not_training_failure():
    with pytest.raises(m.AppendIncomplete, match="unfinished"):
        verify(sample_window(trailing=b'{"unfinished":'))
    with pytest.raises(m.AppendIncomplete, match="two-new"):
        verify(sample_window(rows=[event(13)]))


def test_expired_original_deadline_never_renews():
    parts = sample_window()
    parts[0]["final_clock"] = clock(100)
    with pytest.raises(m.AppendIncomplete, match="deadline"):
        verify(parts)


def test_record_duplicate_step_timestamp_and_malformed_private_utf8():
    for rows in (
        [metric(1010, 9), metric(1010, 13)],
        [metric(1010, 9), metric(1020, 9)],
        [metric(1010, 9), b"\xffSECRET\n"],
    ):
        with pytest.raises(m.AppendRefusal) as error:
            verify(sample_window(rows=rows))
        assert "SECRET" not in str(error.value)


def test_empty_unchanged_file_is_incomplete():
    w, e, b = sample_window(phase="before", rows=[])
    # It has only a pre-window safe event, no optimizer record.
    with pytest.raises(m.AppendIncomplete):
        verify((w, e, b))


def test_consistently_hashed_reread_change_is_still_refused():
    parts = sample_window()
    env = parts[0]["rereads"][0]
    v = env["value"]
    raw = v["raw"].replace(b"utd_wait", b"bad_wait", 1)
    v.update(raw=raw, sha256=m.sha(raw))
    env["audit"]["raw"] = {"sha256": m.sha(raw), "bytes": len(raw)}
    with pytest.raises(m.AppendRefusal, match="observed-rewrite"):
        verify(parts)


def test_rereads_count_against_total_byte_cap_before_record_parsing():
    parts = sample_window()
    w = parts[0]
    raw = b" " * (12 * 2**20) + b"\n"
    size = len(raw)
    for key in ("segments", "rereads"):
        env = w[key][0]
        env["value"].update(
            end=size, raw=raw, sha256=m.sha(raw), size_before=size, size_after=size
        )
        env["audit"].update(end_offset=size, raw={"sha256": m.sha(raw), "bytes": size})
        for k in ("named_before", "stat_before", "stat_after", "named_after"):
            env["audit"][k]["bytes"] = size
    env = w["end_fence"]
    v = env["value"]
    prefix = raw[-65536:]
    v.update(
        size_at_fstat=size,
        line_boundary=True,
        prefix={
            "start": size - len(prefix),
            "end": size,
            "raw": prefix,
            "sha256": m.sha(prefix),
        },
    )
    env["audit"].update(
        offset=size - len(prefix),
        end_offset=size,
        raw={"sha256": m.sha(prefix), "bytes": len(prefix)},
    )
    for k in ("named_before", "stat_before", "stat_after", "named_after"):
        env["audit"][k]["bytes"] = size
    with pytest.raises(m.AppendRefusal, match="total-byte-budget"):
        verify(parts)


def test_genuinely_empty_regular_file_gives_incomplete():
    parts = sample_window(phase="before", rows=[])
    w = parts[0]
    for key in ("before_fence", "causal_fence", "end_fence"):
        env = w[key]
        env["value"].update(
            size_at_fstat=0,
            line_boundary=True,
            prefix={"start": 0, "end": 0, "raw": b"", "sha256": m.sha(b"")},
        )
        env["audit"].update(
            offset=0, end_offset=0, raw={"sha256": m.sha(b""), "bytes": 0}
        )
        for k in ("named_before", "stat_before", "stat_after", "named_after"):
            env["audit"][k]["bytes"] = 0
    for key in ("segments", "rereads"):
        env = w[key][0]
        env["value"].update(
            start=0, end=0, raw=b"", sha256=m.sha(b""), size_before=0, size_after=0
        )
        env["audit"].update(
            offset=0, end_offset=0, raw={"sha256": m.sha(b""), "bytes": 0}
        )
        for k in ("named_before", "stat_before", "stat_after", "named_after"):
            env["audit"][k]["bytes"] = 0
    with pytest.raises(m.AppendIncomplete, match="two-new"):
        verify(parts)


def test_two_old_records_do_not_qualify_merely_because_read_after_cleanup():
    parts = sample_window()
    w = parts[0]
    w["causal_fence"] = deepcopy(w["end_fence"])
    w["causal_fence"]["audit"]["read_start"] = clock(10)
    w["causal_fence"]["audit"]["read_end"] = clock(11)
    with pytest.raises(m.AppendIncomplete, match="two-new"):
        verify(parts)


def test_partial_line_failure_is_scanned_before_incomplete_verdict():
    with pytest.raises(m.AppendRefusal, match="recorded-failure"):
        verify(sample_window(between=event(6, error="failure"), trailing=b"{"))


def test_current_file_identity_must_match_admitted_writer_not_only_itself():
    parts = sample_window()
    w = parts[0]
    for key in ("before_fence", "causal_fence", "end_fence"):
        env = w[key]
        env["value"]["file_identity"]["inode"] = 777
        for k in ("named_before", "stat_before", "stat_after", "named_after"):
            env["audit"][k]["inode"] = 777
    for key in ("segments", "rereads"):
        env = w[key][0]
        env["value"]["file_identity"]["inode"] = 777
        for k in ("named_before", "stat_before", "stat_after", "named_after"):
            env["audit"][k]["inode"] = 777
    with pytest.raises(m.AppendRefusal, match="file-values"):
        verify(parts)


def test_external_original_window_cannot_be_renewed_by_evidence():
    w, e, b = sample_window()
    admitted = {
        k: deepcopy(w[k]) for k in ("phase", "phase_start", "deadline", "cleanup_clock")
    }
    w["deadline"] = clock(101)  # Still inside105, but not the admitted deadline.
    with pytest.raises(m.AppendRefusal, match="original-window-binding"):
        m.verify_append(
            w, expected_writer=e, writer_evidence=b, expected_window=admitted
        )


def test_recipe_is_bound_by_independent_writer_premise():
    parts = sample_window()
    parts[0]["recipe"]["bootstrap_step"] = 999
    with pytest.raises(m.AppendRefusal, match="admitted-recipe"):
        verify(parts)


def test_parseable_first_prewindow_fragment_negative_is_not_discarded():
    parts = sample_window(between=event(1, error="old observed failure"))
    w = parts[0]
    raw = w["segments"][0]["value"]["raw"]
    cut = len(m.encoded(event()))
    size = w["causal_fence"]["value"]["size_at_fstat"]
    w["before_fence"]["value"]["size_at_fstat"] = size
    for env in (w["before_fence"], w["causal_fence"], w["end_fence"]):
        v = env["value"]
        end = size if env is w["before_fence"] else v["prefix"]["end"]
        data = raw[cut:end]
        v["prefix"] = {"start": cut, "end": end, "sha256": m.sha(data), "raw": data}
        v["line_boundary"] = True
        env["audit"].update(
            offset=cut, end_offset=end, raw={"sha256": m.sha(data), "bytes": len(data)}
        )
        if env is w["before_fence"]:
            for k in ("named_before", "stat_before", "stat_after", "named_after"):
                env["audit"][k]["bytes"] = size
    for key in ("segments", "rereads"):
        env = w[key][0]
        data = raw[cut:]
        env["value"].update(start=cut, raw=data, sha256=m.sha(data))
        env["audit"].update(offset=cut, raw={"sha256": m.sha(data), "bytes": len(data)})
    with pytest.raises(m.AppendRefusal, match="recorded-failure"):
        verify(parts)


def test_same_size_changed_stat_metadata_is_observed_rewrite():
    parts = sample_window()
    parts[0]["rereads"][0]["audit"]["stat_after"]["mtime_ns"] += 1
    with pytest.raises(m.AppendRefusal, match="metadata-rewrite"):
        verify(parts)


def test_growth_after_fixed_eof_is_incomplete_and_does_not_move_endpoint():
    parts = sample_window()
    w = parts[0]
    env = w["rereads"][0]
    fixed = w["end_fence"]["value"]["size_at_fstat"]
    env["value"]["size_before"] = env["value"]["size_after"] = fixed + 100
    for k in ("named_before", "stat_before", "stat_after", "named_after"):
        env["audit"][k]["bytes"] = fixed + 100
    with pytest.raises(m.AppendIncomplete, match="growth-after-fixed-end"):
        verify(parts)
    assert w["end_fence"]["value"]["size_at_fstat"] == fixed


def test_public_end_coverage_clock_is_not_later_final_observation_clock():
    proof = verify(sample_window()).public()
    assert proof["coverage_end_monotonic_ns"] < proof["final_observation_monotonic_ns"]
    assert proof["coverage_end_wall_ns"] < proof["final_observation_wall_ns"]


def test_same_size_metadata_rewrite_between_reads_refuses_even_if_bytes_match():
    parts = sample_window()
    audit = parts[0]["rereads"][0]["audit"]
    for key in ("named_before", "stat_before", "stat_after", "named_after"):
        audit[key]["mtime_ns"] += 1
    with pytest.raises(m.AppendRefusal, match="metadata-rewrite"):
        verify(parts)


def test_before_fence_alias_does_not_invent_regression_on_concurrent_append():
    parts = sample_window(phase="before")
    w = parts[0]
    b = w["before_fence"]
    for key in ("stat_after", "named_after"):
        b["audit"][key]["bytes"] += 1
    w["causal_fence"] = deepcopy(b)  # Same observed fence, not an earlier new read.
    proof = verify(parts).public()
    assert len(proof["qualifying_records"]) == 2 and proof["full_preservation"] is False


@pytest.mark.parametrize("alias", [True, 1.0])
def test_writer_evidence_numeric_alias_does_not_match_admitted_binding(alias):
    w, expected, raw = sample_window()
    doc = json.loads(raw)
    doc["schema_version"] = alias
    raw = m.encoded(doc)
    expected.update(evidence_sha256=m.sha(raw), evidence_bytes=len(raw))
    with pytest.raises(m.AppendRefusal, match="writer-binding"):
        verify((w, expected, raw))


def test_expected_original_clock_float_alias_refuses():
    w, e, b = sample_window()
    expected = {
        k: deepcopy(w[k]) for k in ("phase", "phase_start", "deadline", "cleanup_clock")
    }
    expected["deadline"]["monotonic_ns"] = float(expected["deadline"]["monotonic_ns"])
    with pytest.raises(m.AppendRefusal, match="original-window-binding"):
        m.verify_append(
            w, expected_writer=e, writer_evidence=b, expected_window=expected
        )


def test_owner_observation_float_alias_refuses():
    parts = sample_window()
    parts[0]["owner_observations"][0]["owner_key"]["uid"] = 1000.0
    with pytest.raises(m.AppendRefusal, match="owner-source-change"):
        verify(parts)
