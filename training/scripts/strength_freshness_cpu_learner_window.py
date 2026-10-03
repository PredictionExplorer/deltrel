"""Conditional serial learner append proof; no IO, CLI or preservation verdict.

Expected writer and observation provenance is independently admitted by a future
consumer. That consumer must bind B to its committed BEFORE artifact, never a
caller-selected clean suffix. This component does not establish that admission.
Matching its JSON/hash does not grant writer authority or historical birth proof.
Only observed replacement/truncation/changed bytes are rejected; an independently
qualified append-only writer/access premise remains essential between samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
from typing import Any, Mapping

WINDOW = "strength-preservation-learner-append-window-v1"
PROOF = "strength-preservation-conditional-learner-append-proof-v1"
WRITER = "strength-preservation-learner-writer-binding-v1"
SOURCE_CONTRACT = "2de1d3009e6b78084da1ab9bc312609484532b58f022d8d6483157d619e4a635"
R3_COMMIT = "7e77c135bb101f08d1534f0ca5406cb309e2e3d2"
LEARNER_SOURCE = "365d0e2dc87ff7cf16bf12d5e6d2b02efaebf1ecfad11430eb29fb9f1db3196f"
TRAINING_SOURCE = "8753c48cd1c1c05dbb9235bd9bd1a2f9ae810fc6d412ba93dc8d2e7c896cd80b"
MAX_RAW = 24 * 2**20
MAX_JSON = 2**20
MAX_ROWS = 4096
MAX_SPANS = 128
MAX_PREFIX = 65536
SECOND = 10**9
CLOCK = {"boot_id", "monotonic_ns", "wall_ns"}
IDENTITY = {"path", "device", "inode", "uid", "gid", "mode"}
STAT = {
    "device",
    "inode",
    "mode",
    "uid",
    "gid",
    "bytes",
    "mtime_ns",
    "ctime_ns",
    "links",
}
OWNER = {
    "boot_id",
    "pid",
    "start_ticks",
    "ppid",
    "cgroup",
    "invocation_id",
    "pid_namespace_inode",
    "time_namespace_inode",
    "clock_ticks_per_second",
    "origin_sha256",
    "uid",
}
SAFE_EVENTS = {
    "replay_window_consumed",
    "replay_window_allocated",
    "replay_window_refreshed",
    "replay_loader_pool_started",
    "replay_loader_pool_rebound",
    "replay_loader_pool_shutdown",
    "recovery_checkpoint",
    "recovery_gc",
    "replay_gc",
    "replay_reconciliation",
    "promotion_candidate",
    "selfplay_snapshot",
    "selfplay_model_gc",
    "utd_wait",
}
RECIPE = {
    "learning_rates",
    "ema_decay",
    "bootstrap_step",
    "replay_minimum_shard_id_exclusive",
    "utd_segment_baseline_committed_replay_samples",
    "utd_segment_baseline_examples_consumed",
    "utd_segment_target_updates_per_new_sample",
}


class AppendRefusal(ValueError):
    """Fixed public reason only; no private row, path, argv or exception text."""


class AppendIncomplete(ValueError):
    """Insufficient bounded evidence, never a training-failure/action verdict."""


def require(ok: object, code: str) -> None:
    if not ok:
        raise AppendRefusal(code)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encoded(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def integer(x, minimum=0):
    return type(x) is int and x >= minimum


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(x):
    return isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x) is not None


def shape(value, keys, code):
    require(isinstance(value, dict) and set(value) == set(keys), code)
    return value


def clock(value):
    shape(value, CLOCK, "append-clock-shape")
    require(
        isinstance(value["boot_id"], str)
        and 0 < len(value["boot_id"]) <= 64
        and integer(value["monotonic_ns"])
        and integer(value["wall_ns"], 1),
        "append-clock-values",
    )
    return value


def order(a, b):
    clock(a)
    clock(b)
    require(
        a["boot_id"] == b["boot_id"]
        and a["monotonic_ns"] <= b["monotonic_ns"]
        and a["wall_ns"] <= b["wall_ns"],
        "append-clock-order",
    )


def canonical_path(value):
    require(
        isinstance(value, str)
        and value.startswith("/")
        and str(PurePosixPath(value)) == value
        and ".." not in PurePosixPath(value).parts,
        "append-path",
    )


def bounded(value, budget, depth=0):
    budget[0] -= 1
    require(depth <= 64 and budget[0] >= 0, "append-node-budget")
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        require(math.isfinite(value), "append-nonfinite")
    elif isinstance(value, dict):
        require(all(type(k) is str for k in value), "append-json-key")
        for item in value.values():
            bounded(item, budget, depth + 1)
    elif isinstance(value, list):
        for item in value:
            bounded(item, budget, depth + 1)
    else:
        raise AppendRefusal("append-json-value")


def parse(raw, nodes):
    require(type(raw) is bytes and 0 < len(raw) <= MAX_JSON, "append-json-bytes")

    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, "append-duplicate-json-key")
            result[k] = v
        return result

    def constant(_):
        raise AppendRefusal("append-nonfinite")

    try:
        row = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise AppendRefusal("append-invalid-json") from None
    require(isinstance(row, dict), "append-row-object")
    bounded(row, nodes)
    return row


@dataclass(frozen=True)
class AppendProof:
    """Detached public numeric/hash summary; raw input is never retained/repr'd."""

    _safe: bytes = field(repr=False)

    def public(self) -> dict:
        return json.loads(self._safe)

    @property
    def sha256(self) -> str:
        return sha(self._safe)


def _writer(expected, evidence, nodes):
    shape(
        expected,
        {"binding", "evidence_sha256", "evidence_bytes"},
        "append-expected-writer",
    )
    require(
        type(evidence) is bytes
        and integer(expected["evidence_bytes"], 1)
        and len(evidence) == expected["evidence_bytes"] <= MAX_JSON
        and sha(evidence) == expected["evidence_sha256"],
        "append-writer-evidence-pin",
    )
    binding = expected["binding"]
    shape(
        binding,
        {
            "format",
            "schema_version",
            "source_contract_sha256",
            "source_commit",
            "learner_source_sha256",
            "training_source_sha256",
            "metrics_key",
            "metrics_path",
            "run_id",
            "output_root",
            "owner_key",
            "file_identity",
            "command_sha256",
            "imports_sha256",
            "native_sha256",
            "access_sha256",
            "qualification_sha256",
            "recipe_sha256",
        },
        "append-writer-fields",
    )
    require(
        encoded(parse(evidence, nodes)) == encoded(binding), "append-writer-binding"
    )
    require(
        binding["format"] == WRITER
        and type(binding["schema_version"]) is int
        and binding["schema_version"] == 1
        and binding["source_contract_sha256"] == SOURCE_CONTRACT
        and binding["source_commit"] == R3_COMMIT
        and binding["learner_source_sha256"] == LEARNER_SOURCE
        and binding["training_source_sha256"] == TRAINING_SOURCE,
        "append-source-premise",
    )
    require(
        all(
            digest(binding[k])
            for k in (
                "command_sha256",
                "imports_sha256",
                "native_sha256",
                "access_sha256",
                "qualification_sha256",
                "recipe_sha256",
            )
        ),
        "append-writer-provenance",
    )
    canonical_path(binding["metrics_path"])
    canonical_path(binding["output_root"])
    require(
        binding["metrics_key"] == "metrics"
        and PurePosixPath(binding["metrics_path"])
        == PurePosixPath(binding["output_root"]) / "metrics.jsonl"
        and isinstance(binding["run_id"], str)
        and 0 < len(binding["run_id"]) <= 256,
        "append-writer-path",
    )
    identity = shape(binding["file_identity"], IDENTITY, "append-admitted-file")
    require(
        identity["path"] == binding["metrics_path"]
        and all(integer(identity[k]) for k in IDENTITY - {"path"}),
        "append-admitted-file-values",
    )
    owner = shape(binding["owner_key"], OWNER, "append-owner-fields")
    require(
        all(
            integer(owner[k], 1)
            for k in (
                "pid",
                "ppid",
                "start_ticks",
                "pid_namespace_inode",
                "time_namespace_inode",
                "clock_ticks_per_second",
            )
        )
        and integer(owner["uid"])
        and isinstance(owner["boot_id"], str)
        and isinstance(owner["invocation_id"], str)
        and re.fullmatch("[0-9a-f]{32}", owner["invocation_id"]) is not None
        and digest(owner["origin_sha256"]),
        "append-owner-key",
    )
    canonical_path(owner["cgroup"])
    return binding


def _recipe(recipe):
    shape(recipe, RECIPE, "append-recipe-fields")
    rates = recipe["learning_rates"]
    require(
        isinstance(rates, list)
        and len(rates) == 3
        and all(finite(x) and x > 0 for x in rates)
        and finite(recipe["ema_decay"])
        and 0 <= recipe["ema_decay"] < 1
        and all(
            integer(recipe[k])
            for k in (
                "bootstrap_step",
                "replay_minimum_shard_id_exclusive",
                "utd_segment_baseline_committed_replay_samples",
                "utd_segment_baseline_examples_consumed",
            )
        )
        and finite(recipe["utd_segment_target_updates_per_new_sample"])
        and recipe["utd_segment_target_updates_per_new_sample"] > 0,
        "append-recipe-values",
    )


def stat_continuity(a, b):
    require(a["bytes"] <= b["bytes"], "append-observed-truncation")
    require(
        a["bytes"] != b["bytes"]
        or (a["mtime_ns"], a["ctime_ns"]) == (b["mtime_ns"], b["ctime_ns"]),
        "append-observed-metadata-rewrite",
    )


def _observation(envelope, kind, writer, raw_count):
    shape(envelope, {"value", "audit"}, "append-observation-fields")
    value = envelope["value"]
    audit = envelope["audit"]
    common = {"registered_key", "file_identity"}
    shape(
        value,
        common
        | (
            {"size_at_fstat", "prefix", "line_boundary"}
            if kind == "append-fence"
            else {"start", "end", "sha256", "raw", "size_before", "size_after"}
        ),
        "append-value-fields",
    )
    shape(
        audit,
        {
            "operation",
            "subject",
            "read_start",
            "read_end",
            "raw",
            "named_before",
            "stat_before",
            "stat_after",
            "named_after",
            "offset",
            "end_offset",
        },
        "append-audit-fields",
    )
    require(
        audit["operation"] == kind
        and audit["subject"] == value["registered_key"] == writer["metrics_key"],
        "append-operation",
    )
    order(audit["read_start"], audit["read_end"])
    identity = shape(value["file_identity"], IDENTITY, "append-file-identity")
    require(
        identity == writer["file_identity"]
        and identity["path"] == writer["metrics_path"]
        and all(integer(identity[k]) for k in IDENTITY - {"path"})
        and identity["mode"] <= 0o7777,
        "append-file-values",
    )
    sizes = []
    for name in ("named_before", "stat_before", "stat_after", "named_after"):
        st = shape(audit[name], STAT, "append-stat-fields")
        require(
            all(integer(v) for v in st.values())
            and st["links"] > 0
            and all(st[k] == identity[k] for k in IDENTITY - {"path"}),
            "append-stat-identity",
        )
        sizes.append(st["bytes"])
    require(sizes == sorted(sizes), "append-observed-truncation")
    stats = [
        audit[k] for k in ("named_before", "stat_before", "stat_after", "named_after")
    ]
    require(
        all(
            a["bytes"] != b["bytes"]
            or (a["mtime_ns"], a["ctime_ns"]) == (b["mtime_ns"], b["ctime_ns"])
            for a, b in zip(stats, stats[1:])
        ),
        "append-observed-metadata-rewrite",
    )
    if kind == "append-fence":
        span = shape(
            value["prefix"], {"start", "end", "sha256", "raw"}, "append-prefix-fields"
        )
        require(
            type(value["line_boundary"]) is bool
            and integer(value["size_at_fstat"])
            and value["size_at_fstat"] == audit["stat_before"]["bytes"]
            and span["end"] == value["size_at_fstat"]
            and len(span["raw"]) <= MAX_PREFIX,
            "append-fence-size",
        )
    else:
        span = value
        require(
            value["size_before"] == audit["stat_before"]["bytes"]
            and value["size_after"] == audit["stat_after"]["bytes"],
            "append-range-size",
        )
    require(
        integer(span["start"])
        and integer(span["end"])
        and span["start"] <= span["end"] <= sizes[1]
        and type(span["raw"]) is bytes
        and len(span["raw"]) == span["end"] - span["start"]
        and sha(span["raw"]) == span["sha256"],
        "append-span",
    )
    shape(audit["raw"], {"sha256", "bytes"}, "append-raw-pin")
    require(
        integer(audit["raw"]["bytes"])
        and integer(audit["offset"])
        and integer(audit["end_offset"])
        and audit["raw"] == {"sha256": span["sha256"], "bytes": len(span["raw"])}
        and audit["offset"] == span["start"]
        and audit["end_offset"] == span["end"],
        "append-raw-join",
    )
    raw_count[0] += len(span["raw"])
    require(raw_count[0] <= MAX_RAW, "append-total-byte-budget")
    if kind == "append-fence":
        require(
            value["line_boundary"] == (span["end"] == 0 or span["raw"].endswith(b"\n")),
            "append-boundary-bytes",
        )
    return value, audit, span


def _negative(row):
    require(
        row.get("phase") != "nonfinite_abort"
        and all(
            row.get(k) is None
            for k in ("error", "error_type", "failure", "failure_reason")
        ),
        "append-recorded-failure",
    )
    for k in ("nonfinite_loss_count", "nonfinite_gradient_count"):
        if k in row:
            require(integer(row[k]) and row[k] == 0, "append-recorded-failure")
    require(
        row.get("phase") in (None, "training", "update_to_data_wait"),
        "append-unsupported-phase",
    )
    if "gradient_diagnostics" in row:
        observed = row["gradient_diagnostics"]
        require(
            isinstance(observed, dict)
            and observed.get("global_norm_finite") is True
            and integer(observed.get("nonfinite_gradient_tensors"))
            and observed["nonfinite_gradient_tensors"] == 0,
            "append-gradient-diagnostic",
        )


def _loss(row, recipe):
    _negative(row)
    if "event" in row:
        require(row["event"] in SAFE_EVENTS, "append-unsupported-event")
    if "losses" not in row:
        require(row.get("event") in SAFE_EVENTS, "append-unsupported-event")
        return False, 0
    required = {
        "step",
        "losses",
        "gradient_norm",
        "gradient_pre_clip_norm",
        "gradient_post_clip_norm",
        "nonfinite_loss_count",
        "nonfinite_gradient_count",
        "gradient_diagnostics",
        "learning_rates",
        "ema",
        "replay_minimum_shard_id_exclusive",
        "utd_segment_baseline_committed_replay_samples",
        "utd_segment_baseline_examples_consumed",
        "utd_segment_target_updates_per_new_sample",
        "segment_updates_per_new_sample",
        "policy_batch_metrics",
    }
    require(required <= set(row), "append-loss-fields")
    require(
        integer(row["step"], 1)
        and isinstance(row["losses"], dict)
        and row["losses"]
        and all(finite(v) for v in row["losses"].values())
        and all(
            finite(row[k]) and row[k] >= 0
            for k in (
                "gradient_norm",
                "gradient_pre_clip_norm",
                "gradient_post_clip_norm",
            )
        ),
        "append-loss-finite",
    )
    d = row["gradient_diagnostics"]
    require(
        isinstance(d, dict)
        and d.get("global_norm_finite") is True
        and integer(d.get("nonfinite_gradient_tensors"))
        and d["nonfinite_gradient_tensors"] == 0,
        "append-gradient-diagnostic",
    )
    ema = row["ema"]
    require(
        row["learning_rates"] == recipe["learning_rates"]
        and isinstance(row["learning_rates"], list)
        and all(finite(v) for v in row["learning_rates"])
        and isinstance(ema, dict)
        and finite(ema.get("decay"))
        and ema.get("decay") == recipe["ema_decay"]
        and integer(ema.get("num_updates"))
        and ema["num_updates"] == row["step"] - recipe["bootstrap_step"],
        "append-rates-ema",
    )
    require(
        all(
            integer(row[k]) and row[k] == recipe[k]
            for k in (
                "replay_minimum_shard_id_exclusive",
                "utd_segment_baseline_committed_replay_samples",
                "utd_segment_baseline_examples_consumed",
            )
        )
        and finite(row["utd_segment_target_updates_per_new_sample"])
        and row["utd_segment_target_updates_per_new_sample"]
        == recipe["utd_segment_target_updates_per_new_sample"]
        and finite(row["segment_updates_per_new_sample"])
        and 0
        <= row["segment_updates_per_new_sample"]
        <= recipe["utd_segment_target_updates_per_new_sample"] + 1e-9,
        "append-replay-credit",
    )
    batch = d.get("batch")
    policy = row["policy_batch_metrics"]
    require(
        isinstance(batch, dict)
        and {"rows", "six_mode_unknown", "label_availability"} <= set(batch)
        and not batch.get("six_mode_unknown")
        and integer(batch.get("rows"), 1)
        and isinstance(policy, dict)
        and integer(policy.get("unknown_provenance_rows"))
        and policy["unknown_provenance_rows"] == 0,
        "append-policy-provenance",
    )
    assert isinstance(batch, dict)
    labels = batch.get("label_availability")
    require(
        isinstance(labels, dict)
        and all(
            integer(labels.get(k)) and labels[k] <= batch["rows"]
            for k in ("policy", "outcome")
        )
        and labels["policy"] > 0,
        "append-labels",
    )
    assert isinstance(labels, dict)
    return True, labels["outcome"]


def verify_append(
    window: Mapping[str, Any],
    *,
    expected_writer: Mapping[str, Any],
    writer_evidence: bytes,
    expected_window: Mapping[str, Any],
) -> AppendProof:
    """Validate private evidence conditional on the caller's independent writer admission."""
    try:
        return _verify(window, expected_writer, writer_evidence, expected_window)
    except (AppendRefusal, AppendIncomplete):
        raise
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
        AttributeError,
    ):
        raise AppendRefusal("append-invalid-schema") from None


def _verify(w, expected, evidence, expected_window):
    shape(
        w,
        {
            "format",
            "schema_version",
            "phase",
            "phase_start",
            "deadline",
            "cleanup_clock",
            "final_clock",
            "before_fence",
            "causal_fence",
            "end_fence",
            "segments",
            "rereads",
            "owner_observations",
            "recipe",
        },
        "append-window-fields",
    )
    require(
        w["format"] == WINDOW
        and type(w["schema_version"]) is int
        and w["schema_version"] == 1
        and w["phase"] in ("before", "after"),
        "append-window-format",
    )
    shape(
        expected_window,
        {"phase", "phase_start", "deadline", "cleanup_clock"},
        "append-expected-window-fields",
    )
    require(
        encoded({k: w[k] for k in expected_window}) == encoded(expected_window),
        "append-original-window-binding",
    )
    nodes = [250000]
    writer = _writer(expected, evidence, nodes)
    _recipe(w["recipe"])
    require(
        sha(encoded(w["recipe"])) == writer["recipe_sha256"], "append-admitted-recipe"
    )
    start, end, final = (
        clock(w["phase_start"]),
        clock(w["deadline"]),
        clock(w["final_clock"]),
    )
    order(start, end)
    order(start, final)
    require(
        0 < end["monotonic_ns"] - start["monotonic_ns"] <= 105 * SECOND
        and 0 < end["wall_ns"] - start["wall_ns"] <= 105 * SECOND,
        "append-original-window",
    )
    if (
        final["monotonic_ns"] >= end["monotonic_ns"]
        or final["wall_ns"] >= end["wall_ns"]
    ):
        raise AppendIncomplete("append-deadline-exhausted")
    count = [len(evidence)]
    b, ba, bp = _observation(w["before_fence"], "append-fence", writer, count)
    f, fa, fp = _observation(w["causal_fence"], "append-fence", writer, count)
    e, ea, ep = _observation(w["end_fence"], "append-fence", writer, count)
    require(
        b["file_identity"] == f["file_identity"] == e["file_identity"]
        and b["size_at_fstat"] <= f["size_at_fstat"] <= e["size_at_fstat"],
        "append-fence-continuity",
    )
    order(ba["read_end"], fa["read_start"] if w["phase"] == "after" else fa["read_end"])
    order(fa["read_end"], ea["read_start"])
    order(start, fa["read_start"])
    if w["phase"] == "before":
        require(
            w["cleanup_clock"] is None and w["before_fence"] == w["causal_fence"],
            "append-before-fence",
        )
    else:
        cleanup = clock(w["cleanup_clock"])
        order(cleanup, fa["read_start"])
        require(
            cleanup["monotonic_ns"] < fa["read_start"]["monotonic_ns"],
            "append-fence-not-after-cleanup",
        )
    if w["phase"] == "after":
        stat_continuity(ba["named_after"], fa["named_before"])
    stat_continuity(fa["named_after"], ea["named_before"])
    previous_stat = ea["named_after"]
    observed_growth = any(
        ea[k]["bytes"] > e["size_at_fstat"]
        for k in ("named_before", "stat_before", "stat_after", "named_after")
    )
    originals = []
    last = ea["read_end"]
    cover_start = bp["start"]
    cover_end = e["size_at_fstat"]
    for name in ("segments", "rereads"):
        rows = w[name]
        require(
            isinstance(rows, list) and 1 <= len(rows) <= MAX_SPANS, "append-range-count"
        )
        cursor = cover_start
        seen = []
        for item in rows:
            value, audit, span = _observation(item, "append-range", writer, count)
            require(
                value["file_identity"] == b["file_identity"]
                and span["start"] == cursor
                and (
                    span["end"] > cursor
                    or (cover_start == cover_end and len(rows) == 1)
                )
                and span["end"] <= cover_end,
                "append-range-gap-or-overlap",
            )
            stat_continuity(previous_stat, audit["named_before"])
            previous_stat = audit["named_after"]
            observed_growth = observed_growth or any(
                audit[k]["bytes"] > cover_end
                for k in ("named_before", "stat_before", "stat_after", "named_after")
            )
            order(last, audit["read_start"])
            last = audit["read_end"]
            cursor = span["end"]
            seen.append((span, audit))
        require(cursor == cover_end, "append-range-missing-suffix")
        if name == "segments":
            originals = seen
        else:
            require(
                len(seen) == len(originals)
                and all(
                    all(a[0][k] == b_[0][k] for k in ("start", "end", "sha256", "raw"))
                    for a, b_ in zip(originals, seen)
                ),
                "append-observed-rewrite",
            )
    order(last, final)
    raw = b"".join(span["raw"] for span, _ in originals)
    for span in (bp, fp, ep):
        require(
            cover_start <= span["start"] <= span["end"] <= cover_end
            and raw[span["start"] - cover_start : span["end"] - cover_start]
            == span["raw"],
            "append-prefix-rewrite",
        )
    obs = w["owner_observations"]
    require(isinstance(obs, list) and 2 <= len(obs) <= 16, "append-owner-observations")
    prev = None
    for item in obs:
        shape(
            item,
            {"owner_key", "writer_evidence_sha256", "read_start", "read_end"},
            "append-owner-observation-fields",
        )
        require(
            encoded(item["owner_key"]) == encoded(writer["owner_key"])
            and item["writer_evidence_sha256"] == expected["evidence_sha256"],
            "append-owner-source-change",
        )
        order(item["read_start"], item["read_end"])
        if prev is not None:
            order(prev, item["read_start"])
        prev = item["read_end"]
    order(obs[0]["read_end"], ba["read_start"])
    order(last, obs[-1]["read_start"])
    order(obs[-1]["read_end"], final)
    if w["phase"] == "after":
        require(
            any(
                all(
                    w["cleanup_clock"][axis]
                    <= x["read_start"][axis]
                    <= x["read_end"][axis]
                    <= fa["read_start"][axis]
                    for axis in ("monotonic_ns", "wall_ns")
                )
                for x in obs[1:-1]
            ),
            "append-post-cleanup-owner-readmission",
        )
    require(writer["owner_key"]["boot_id"] == final["boot_id"], "append-owner-boot")
    offset = cover_start
    skipped = 0
    if offset > 0:
        cut = bp["raw"].find(b"\n")
        require(cut >= 0, "append-unrecoverable-prefix-line")
        skipped = cut + 1
        # Boundary is not proven. A parseable whole fragment can still contain
        # explicit observed negatives; it never contributes M1/M2 or history.
        try:
            fragment = parse(raw[:skipped], nodes)
        except AppendRefusal as error:
            if str(error) in ("append-nonfinite", "append-duplicate-json-key"):
                raise
        else:
            _negative(fragment)
        raw = raw[skipped:]
        offset += skipped
    selected = []
    all_losses = 0
    previous_stamp = -1
    previous_step = -1
    loss_step = -1
    rows = 0
    complete = raw.rsplit(b"\n", 1)[0] + b"\n" if b"\n" in raw else b""
    for line in complete.splitlines(keepends=True):
        require(line.endswith(b"\n") and len(line) <= MAX_JSON, "append-line-bound")
        begin = offset
        offset += len(line)
        rows += 1
        require(rows <= MAX_ROWS, "append-row-budget")
        row = parse(line, nodes)
        require(
            type(row.get("schema_version")) is int
            and row["schema_version"] == 1
            and row.get("worker") == "learner"
            and integer(row.get("timestamp_ns"), 1),
            "append-record-header",
        )
        stamp = row["timestamp_ns"]
        observed = next(a for s, a in originals if s["end"] >= offset)
        require(
            previous_stamp < stamp <= observed["read_end"]["wall_ns"],
            "append-record-clock-order",
        )
        previous_stamp = stamp
        if "step" in row:
            require(
                integer(row["step"]) and row["step"] >= previous_step,
                "append-step-regression",
            )
            previous_step = row["step"]
        is_loss, labels = _loss(row, w["recipe"])
        if is_loss:
            require(row["step"] > loss_step, "append-loss-step-order")
            loss_step = row["step"]
            all_losses += 1
            if begin >= f["size_at_fstat"] and final["wall_ns"] - stamp <= 120 * SECOND:
                selected.append(
                    {
                        "start": begin,
                        "end": offset,
                        "step": row["step"],
                        "timestamp_ns": stamp,
                        "sha256": sha(line),
                        "outcome_labels": labels,
                    }
                )
    require(len(raw) - len(complete) <= MAX_JSON, "append-partial-line-bound")
    if not e["line_boundary"] or len(complete) != len(raw):
        raise AppendIncomplete("append-unfinished-final-line")
    if observed_growth:
        raise AppendIncomplete("append-observed-growth-after-fixed-end")
    if len(selected) < 2:
        raise AppendIncomplete("append-two-new-loss-records-required")
    require(
        sum(row["outcome_labels"] for row in selected) > 0,
        "append-missing-outcome-labels",
    )
    result = {
        "format": PROOF,
        "schema_version": 1,
        "status": "conditional-serial-append-observed",
        "full_preservation": False,
        "execution_qualified": False,
        "historical_birth_qualified": False,
        "source_contract_sha256": SOURCE_CONTRACT,
        "writer_evidence_sha256": expected["evidence_sha256"],
        "owner_sha256": sha(encoded(writer["owner_key"])),
        "file_identity_sha256": sha(encoded(b["file_identity"])),
        "phase": w["phase"],
        "before_offset": b["size_at_fstat"],
        "causal_offset": f["size_at_fstat"],
        "final_offset": cover_end,
        "coverage_end_monotonic_ns": ea["read_end"]["monotonic_ns"],
        "coverage_end_wall_ns": ea["read_end"]["wall_ns"],
        "final_observation_monotonic_ns": final["monotonic_ns"],
        "final_observation_wall_ns": final["wall_ns"],
        "coverage_start": cover_start,
        "unparsed_pre_window_prefix_bytes": skipped,
        "raw_bytes_checked_including_rereads": count[0],
        "complete_records_checked": rows,
        "loss_records_checked": all_losses,
        "qualifying_records": selected,
        "minimum_serial_intervals": 1,
        "window_digest": sha(
            encoded({"start": start, "deadline": end, "final": final})
        ),
        "history_scope": "sampled-since-before-fence-only",
        "writer_authority_granted": False,
    }
    safe = encoded(result)
    require(len(safe) <= MAX_JSON, "append-public-proof-bound")
    return AppendProof(safe)
