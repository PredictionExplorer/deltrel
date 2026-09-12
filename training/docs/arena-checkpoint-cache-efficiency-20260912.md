# Arena, checkpoint and graph-cache efficiency — September 12, 2026

Implementation is complete; production activation and isolated H100 results are
pending. This release removes repeated learner checkpoint verification, adds
optional exact arena endings, and prepares a measured increase in actor graph-cache
capacity. No end-to-end throughput or Elo/hour gain is established yet.

## Changes and evidence

**Learner control metadata.** The learner now caches successfully verified model
publications when polling champion/candidate control metadata. The initial read
still performs the existing complete verification. Cache reuse requires unchanged
resolved paths, device/inode, size, modification time and change time for the
pointer, immutable manifest and checkpoint. Replacements or corruption invalidate
the entry. Dependencies are captured before verification and checked afterward;
an overlapping atomic publication receives one retry, while persistent churn fails.
A failed replacement cannot fall back to a formerly verified entry. Weight loading
and recovery retain independent full verification.

A read-only probe against the live immutable checkpoints at steps 156,256 and
200,203 measured approximately **385 ms → 0.270 ms** per pair of control reads
(100 cached repetitions; the first cached read still took 384 ms). Returned
verified manifests matched. This is component latency with CUDA hidden, not a
training throughput multiplier. The cache is automatic in the new learner runtime.

**Exact arena endings.** `arena.exact_clinch_termination` defaults to **false**.
When enabled, the arena stops a game only when the native extremal-completion proof
establishes the same winner for every remaining legal continuation. Proofs operate
on copies; stored actions remain the actual played prefix. An available pie swap
prevents early finalization. Stable per-game/pair seeds are required, and subtree
reuse is rejected with this option. Search budgets, rules and statistical tests
retain their existing definitions.

Offline inspection of 192 completed production arena games found an earlier exact
proof in every game, preserving their winners. Their omitted tails account for
16,477 searched moves and 7,195,904 of 21,205,248 nominal simulations (33.9%). The
implemented resume reader independently revalidated all 192 shortened completions
with the option enabled and disabled. These are historical work counts and CPU
proof checks; H100 timing and a full arena comparison remain pending.

**Actor graph capacity.** The production 16-entry cache can hold the sixteen
current batch buckets for one board size. A shared model serving multiple board
sizes can evict and recapture useful graphs. The candidate changes only
`orchestration.model_refresh.inference.cuda_graph_max_entries` from **16 to 32**.
It retains the current 48 GiB process allowance, divided into **8 GiB per model**
with four actor cohorts and six registry slots. This candidate is not enabled by
the implementation or by a bare profile edit.

## Graph benchmark and admission

Run the benchmark twice, once for each immutable actor/champion model named by the
existing search-allocation admission artifact. Use that artifact's original frozen
`baseline_profile`, not a newly edited target profile. The GPU must already be
exclusive; the benchmark does not pause other jobs. From `training/`:

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark_graph_cache_capacity.py \
  --config /path/to/original-baseline-profile.yaml \
  --manifest /path/to/immutable-model-manifest.json \
  --actor-gpu-id 1 --device cuda:7

PYTHONPATH=. .venv/bin/python scripts/benchmark_graph_cache_capacity.py \
  --config /path/to/original-baseline-profile.yaml \
  --manifest /path/to/immutable-model-manifest.json \
  --actor-gpu-id 1 --device cuda:7 \
  --graph-cache-bytes 8589934592 --repeats 2 --cycles 3 \
  --execute --output /path/to/new-graph-capacity-report.json
```

The first command prints a pinned plan. Use a separate new output path for each
model. Default scenarios are `ring10-control`, `mixed-6-10`, `mixed-8-10` and
`mixed-6-8-10`, covering all sixteen production buckets. The three-board trace has
48 distinct keys and deliberately retains pressure beyond either entry limit.
These deterministic stress traces are not measured production request frequencies.

Arms run serially in alternating order. Each begins with an empty graph cache;
every cold capture, subsequent recapture, validation, transfer and inference return
is timed. Graphs remain resident across cycles. Only prediction caches are cleared
between repeated fixtures, preventing prediction hits from hiding graph churn.
Compilation is primed at matching physical shapes and reported separately, as is
producer preparation. Reports include exact response fingerprints, per-cycle
counters, cold/steady totals, peak memory, retained bytes and GPU owner observations.

Admission recomputes evidence rather than trusting saved success flags. It requires
both original frozen models, the full production traces, exact outputs, complete
work/capture accounting, verified H100 ownership, unchanged byte limits, and median
throughput ratios of at least 1.0 in every scenario. At least one mixed case must
reduce captures. Baseline model/config hashes, inference settings, compilation,
precision and producer topology remain bound to the evidence. Original search
quality and full-target-production evidence remains mandatory.

## Compatibility, deployment and rollback

The disabled arena option preserves existing configuration authority through an
explicit default-omission rule. Its enabled value remains distinct. Arena result
and resume schema versions stay unchanged; the new reader accepts old snapshots
and independently proves every stored completion from its recorded actions.
Enabling or disabling new early endings may preserve valid in-progress evaluation
state at a stopped, verified migration boundary. Stored early completions remain
valid when new early endings are disabled.

The existing allocation-gate schema accepts optional `graph_cache_reports` references
only for the exact measured 16→32 capacity change. Backup dependency collection
includes those reports alongside the original baseline, selections, model manifests
and search reports. Changed evidence invalidates the verification cache. Missing
or invalid evidence prevents admission and backup validation.

Deploy through the normal graceful checkpoint, verified backup and profile-migration
procedure. Retain a rollback configuration on the **new reader-capable runtime**:
disable `arena.exact_clinch_termination` and restore the original 16-entry graph
capacity with its matching allocation-gate authority. Do not roll back to a reader
that cannot validate already-recorded early completions or the run's existing live
replay revisions.

Focused tests cover control-cache invalidation and replacement races; classic,
double, handicap and pie arena paths; stable seeds, interruption/resume, forged
winners and rollback; graph byte pressure, exact outputs, incomplete/corrupt
evidence, admission and backup dependencies. The graph benchmark and prior bucket
suite passed 47 CPU tests, Ruff and Pyright; 48 real native fixture shapes also
passed a local smoke check. These checks do not substitute for the pending H100
qualification and production readiness evidence.
