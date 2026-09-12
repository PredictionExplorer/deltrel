# Training pipeline efficiency — September 12, 2026

This implementation removes redundant work while preserving model arithmetic,
search budgets, replay eligibility, target weights, and arena decisions. The
objective remains measured playing strength per provisioned wall-clock hour.
Component latency and memory savings are not equivalent to Elo improvements.

The baseline is `c607372`, after the recorded graph-cache activation. This is a
local implementation and validation report; these changes have not been deployed
to the H100 server. The earlier unqualified local-message kernels remain disabled.
The rebuilt native wheel is installed in the local training environment; its
binary SHA-256 matches the isolated candidate used for the native comparisons.

## Changes

- **Replay readiness:** reuse exact eligibility-count queries while the SQLite
  manifest and all eligibility arguments remain unchanged. External commits and
  local writes invalidate the bounded eight-entry cache. Explicit transactions
  bypass it, preserving publication, WAL snapshot, rollback, quarantine, and GC
  behavior. Selected replay spans, sampling probabilities, and UTD credit are
  unchanged.
- **Arena bookkeeping:** build independent resume snapshots without encoding and
  decoding the entire history before the durable writer encodes it again. Tests
  compare serialized snapshots against the original implementation, including
  interrupted native games. Per-move checkpoint callbacks and durable write
  cadence are unchanged. Avoid full native scoring when no terminal result needs
  it, including cases with an already proven clinch winner.
- **Inference returns:** gather legal logits directly from the owned float32 CPU
  array for trusted native CSR requests. This removes temporary tensor creation
  and dispatch; score/outcome arithmetic, model execution, cache contents,
  request ordering, and returned-list ownership are unchanged. Mixed native and
  general requests retain their separate validation paths.
- **Trusted training losses:** construct margin, ownership, and alive validation
  gathers only when target validation is requested. The trusted replay path
  previously created and discarded three Boolean-index results per batch. This
  removes data-dependent intermediate shapes and potential CUDA synchronization
  without changing any loss, diagnostic, or gradient calculation. Public input
  validation remains enabled by default.
- **Native search storage:** encode child indices with a nonzero optional handle.
  Each edge occupies 32 instead of 40 bytes on 64-bit targets: 20% less edge
  storage, not 20% less total process memory. Index zero, large indices, subtree
  remapping, and transposition reuse remain supported.
- **Native response submission:** validate every pending session before changing
  any tree, then submit independent sessions in parallel only with multiple
  threads, multiple sessions, at least 8,192 policy logits, and at least two
  response rows per pending session. Small batches and ordinary single-row
  leaf updates retain serial submission. The row-count condition was added
  during Linux qualification after the initial logit-only guard regressed in
  long searches. Target-host measurements for this refinement are separate
  from the original Mac measurements below.
- **Benchmark binary identity:** recognize both the normal package wrapper and
  a directly loaded native extension. The full regression exposed an existing
  package-only assumption when testing the isolated candidate binary. Both
  layouts now resolve the actual compiled artifact; a missing binary still
  fails validation.
- **Current-profile startup:** when stored autonomous provenance exactly equals
  freshly computed current provenance, skip enumerating historical configuration
  hashes. Different hashes still use the original compatibility logic, and all
  other provenance fields must still match. Six focused tests cover the fast
  path, legacy fallback, byte preservation, and altered metadata. No startup
  latency multiplier is claimed.

## Measured CPU results

The final Python measurements ran sequentially after native builds and
benchmarks finished, before the full regression suite. All comparisons use
identical inputs and counterbalanced order on this Apple Silicon workstation.
Raw observations and source hashes are retained in
[the evidence record](cpu-efficiency-evidence-20260912.json).

| Component | Baseline | Updated | Scope |
| --- | ---: | ---: | --- |
| Two same-step replay ring-count checks | 11.816 ms | 5.911 ms | 30,000 synthetic shard records; aggregate scans halved |
| Two ring/segment readiness checks | 19.070 ms | 9.615 ms | Same manifest; all counts identical |
| Arena resume snapshot | 5.566 ms | 0.446 ms | 240 games / 192 completions, 456,174 serialized bytes; disk excluded |
| Trusted loss construction | 2.925 ms | 2.177 ms | B512, 275 nodes, CPU forward only; three unused indexing operations removed |
| Cached native inference return, 32 rows | 0.357 ms | 0.252 ms | Same float32 predictions; no timed neural work |
| Cached native inference return, 128 rows | 1.299 ms | 0.920 ms | Same scope; one/eight-row cases also improved |

The native comparison with a 53-candidate cap (limited by the simulation budget)
matched state-fingerprint, request-order, and result traces. Ring-10 native
next-request plus submission throughput improved **1.067× /
1.198×** at 32 roots for budgets 27/640, and **1.122× / 1.332×** at 128 roots.
The broader 16-case comparison and separate compact-only isolation also matched
those traces. Single-thread controls ranged from **0.967× to 1.082×**, so this is
not a claim that every search shape became faster. Edge storage is 20% smaller
regardless of timing. The cheap synthetic evaluator and four native threads do
not reproduce H100 model execution or whole-actor throughput.
The [native report](native-search-memory-and-submit-20260912.md) retains the
complete comparison matrix, controls, build commands, and rejected experiment.

These ratios must not be multiplied together or reported as a whole-training
speedup. In particular, the loss timing excludes neural backward computation,
and the inference timing isolates CPU return work shared by cache hits and
misses. No new equal-time Elo result is claimed.

## Reproduction

Run from `training/` with the existing environment and a release native extension:

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark_cpu_hotpaths.py \
  --baseline-ref c607372 --repeats 8 --iterations 50 \
  --output /tmp/cpu-hotpaths-new.json
PYTHONPATH=. .venv/bin/python scripts/benchmark_arena_bookkeeping.py \
  --repeats 8 --iterations 100 --output /tmp/arena-bookkeeping-new.json
PYTHONPATH=. .venv/bin/python scripts/benchmark_replay_eligibility.py \
  --shards 30000 --checks-per-step 2 --repeats 7
```

The first benchmark loads the two original Python modules from the pinned git
ref. Its inference measurements use warmed caches and assert that no neural work
runs during timing. Loss timings cover loss construction, excluding model
forward/backward and optimizer work; losses and gradients are checked separately.
The other benchmarks retain the original snapshot algorithm or bypass only the
new replay cache. All arms alternate order and verify equal outputs.

For native comparisons, build baseline and candidate release extensions in
separate directories, then run `scripts/benchmark_native_submit.py` with explicit
`--baseline`, `--candidate`, and a new `--output` path. It checks semantic request
traces, actions, visits, Q values, and policy targets, and records extension hashes.
The synthetic evaluator is deliberately cheap; native timings do not predict
GPU actor throughput.

## H100 and strength acceptance

CUDA production qualification and an equal-wall-time strength comparison remain
required before claiming an Elo/hour improvement. Use the existing
[ablation protocol](training-ablation-protocol.md) and
[equal-time runbook](elo-per-hour-ablation-runbook.md), with identical starting
checkpoints, frozen replay/search settings, and the same six-cell evaluation
objective. Charge setup, evaluation, waiting, restart, and recovery time to the
provisioned wall budget. Keep publication and optimizer clocks comparable.

First verify exact inference/search responses, losses, and gradients on the H100
execution paths, then measure complete actor and arena throughput with realistic
models and batch shapes. The learner's existing fresh-data limit can absorb a
faster loss step as additional wait; a local step speedup alone is not evidence
of faster learning. Do not increase UTD or reduce search budgets based solely on
these CPU measurements.

## Verification

All 100 Rust workspace tests and strict Clippy passed. The shared Rust engine also
passes a `wasm32-unknown-unknown` build check for the browser. Whole-project Python
Ruff and Pyright checks passed. After fixing direct-extension identification,
23 affected benchmark tests passed; a further 77 native/inference/benchmark tests
passed against the normally installed candidate wheel.

The full CPU selection passed in four isolated groups: **467 + 732 + 949 + 949 =
3,097 tests**, with no final group failures. The six additional provenance
fast-path cases also passed, for **3,103 CPU tests**. Coverage from these five
runs was combined: **83.3314% overall**, passing the 74% total requirement and
all 28 per-file floors. The final Ruff and whole-project Pyright checks passed.

The test launcher initially needed a multiprocessing guard; affected groups were
rerun in full with that guard before accepting their results or coverage. The
earlier interrupted serial run and failed launcher runs are not counted toward
the final pass totals. CPU tests cannot certify CUDA graph capture, H100
throughput, or multi-GPU production readiness.
