# Fresh-data starvation correction — September 11, 2026

Deployment is complete. The new release resumed checkpoint **190,377** with zero
discarded optimizer steps. All ten workers remained healthy during an uninterrupted
20-minute comparison. Fresh-position throughput rose **38.3%** and learner updates
rose **39.5%** against the prospective baseline.

The release is `7d874bba9206e07c4b17156e2dba169758d35720`, built at
`/home/ubuntu/deltrel-releases/variant-deltrelvation-7d874bb`. Its application
implementation is commit `916cbfd`; subsequent commits isolate experimental
compiler tests and annotate a Triton DSL construct for static checking. Neither
follow-up changes executable application code.

## Selected correction

The candidate profile changes four fields:

| Setting | Existing | Candidate |
| --- | --- | --- |
| Ring-10 search allocation | Fast 53/full 640, full probability 0.35, fast policy weight 0.2 | Fast 27/full 640, full probability 0.23, fast policy weight 0.11094619666048239 |
| First-visit prediction batch | 1 | 8 |
| Replay loader workers | 16 | 4 |
| Prefetch batches per worker | 4 | 2 |

The per-ring override changes the fast cap and allocation only on ring 10.
First-visit batching retains the sequential search mutation/backup order.
The model, BF16 compute, FP32 master weights, optimizer, LR/EMA clocks, 1.5
update-to-data target, full-search caps, PDA rules and arena budgets stay fixed.
The new policy weight preserves the **nominal expected** full/fast policy-weight
ratio; it does not make the normalized training objective or batch variance
identical.

The learner forecasts bounded wakeups from the committed-position rate, retaining
exact integer/Fraction credit authorization. Enrichment and retries earn no extra
position credit. Faster credit probes do not repeat full replay scans or durable
heartbeats. The loader now drains only its already-issued tensor transfers before
stopping pinning/workers, with a ten-second drain deadline and errors preserved.
New CPU-side batch diagnostics expose effective policy rows, deep-target weight
share and all-fast batches without GPU synchronization.

Reducing prefetch cuts speculative queued batches from 64 to eight. Isolated
measurements found no steady refresh-latency improvement; this saves work and
memory and must not be counted as a large throughput multiplier.

## Measured search gate

Two immutable models were tested on independent holdouts, each containing 126
positions balanced across six variants and three game phases, plus 16 actual
swap opportunities. Games from both preliminary selections were excluded.
Reference values came only from visited actions of the unchanged deep search;
they are finite learned search estimates, not ground-truth strength.

| Candidate | Actor upper-95% incremental regret | Champion upper-95% incremental regret | Decision |
| --- | ---: | ---: | --- |
| Fast 13 | 0.02655 | 0.01144 | Reject: actor exceeds the predeclared 0.02 limit |
| Fast 27, width 8 | 0.01972 | 0.00461 | Pass the stated mean-regret screen |

Assessment coverage was 100%; there were no new swap failures. One fast-27 actor
position still had a large 0.804 reference-Q loss. Passing the mean-confidence
screen does not establish tactical equivalence or an Elo improvement.

Warm repeats reset prediction caching without discarding CUDA graphs. Measured
repeats had zero graph captures, warmups, evictions, fallbacks or validation work.
At full probability 0.23, the selected allocation projects **1.598× / 1.593×**
position throughput for the actor/champion models. Conservative observed-repeat
bounds preserve deep-target production by **3.09% / 2.57%**. These are projections
from separate balanced fast/full waves; actual mixed production must be measured.
The benchmark's prediction-cache capacity exceeded the per-model production cap;
there were no measured cache hits, but full-wave bookkeeping cost may differ.

Quartering deep-search budgets failed the preliminary quality screen. Both
source-class and project-first compiled inference prototypes failed their declared
numerical gates and remain disabled. Source-class diagnostic timing was 22–29%
faster, which is not an adopted or production performance gain.

## Configuration admission and validation

A below-floor ring allocation requires an exact-config artifact under
`status/search-allocation-gates`. Validation binds the raw baseline profile,
models, independent selections, search contract and report hashes; it recomputes
quality and timing criteria. It cannot be enabled by a bare success flag.
Unchanged artifacts use a file-identity cache, measured at 0.52 ms per validation.
Backups preserve the gate and all dependencies, including the raw baseline YAML.
Original-root restore is tested; relocation of a gated profile fails explicitly.

Candidate profile SHA-256:
`6cf8ceba830b4be34368f92824d2b8d32e8b866696ec810d03c9e7c2252a6d93`.
Canonical config SHA-256:
`78b6b4ed7961791492ed43628aa99c67010e5fa4ab051fd144dd0b3f905ad9e1`.
Admission artifact SHA-256:
`b9f033f8638eb63a505bdd7f63412580b83e6d53cc35bad72e9b347a7726c5b6`.

The complete server CPU/native-required run passed 2,780 tests and skipped 17;
two late compiler tests failed because the new experimental tests left global
Dynamo variants behind. A test-local public compiler reset fixed the isolation.
The final release passed a 126-test combined compiler/training sequence, including
both former failures, with six CUDA skips. All other application code is
byte-identical or AST-identical to the complete run; retained qualification
artifacts document the coverage rather than adding overlapping test counts.
Ruff and server Pyright pass. All 538 archived source files and all 103 installed
package versions were checked. The Rust source/dependencies and native binary
are unchanged from the previously validated production release; native SHA-256:
`a860d66c128cedbd3429765180459bdcdad34104d4ec520633f1025ff26abf1b`.

The deployment controller and corrected smoke wrapper passed 40 local tests,
including independent review. Recovery refuses unrelated or incomplete migration
history. Before migration, recovery uses the old release; after a migration
attempt it restores the baseline settings on the new reader-capable release.

## Baselines and live measurement

The uninterrupted 11:36–11:50 UTC baseline sampled 79.17% UTD wait and measured
348,560 unique positions/hour and 1,024 updates/hour. A separate prospective
11:59:17–12:09:39 UTC interval measured 394,377 unique positions/hour, 128,900 full
policy targets/hour, 1,153 updates/hour and 69.27% sampled UTD wait. The latter
classified all 68,072 new logical positions across 1,789 game heads, with zero
missing data; enrichment was counted separately. Device step time was 0.402 s.
Use the latter interval for a full-target comparison and retain both windows to
show the variation under unchanged settings. Old code did not record actual
UTD sleep; sampled phase fractions are not invented sleep durations.

The live observer uses read-only SQLite counters and per-game prefix differences.
It reads only NPZ metadata needed to classify newly appended targets, never creates
ReplayStore or replay pins, and reports explicit bounds if data is unavailable.
The uninterrupted post-warmup interval below establishes worker health, unique
position throughput, full-target production, learner progress and explicit sleep.
No Elo/hour multiplier is established by these checks.


The post-deployment interval was **13:17:14–13:37:41 UTC** (1,226.98 seconds),
with coordinator 2071498 and unchanged worker identities. All 245 monitor
observations were healthy, with no restarts, inference failures or telemetry gaps.
All 185,877 new positions were classified from 2,214 game heads; prefix differences
matched the committed counter exactly. There were no missing/invalid payloads.

| Metric | Prospective baseline | Deployed observation |
| --- | ---: | ---: |
| Unique positions/hour | 394,377 | 545,368 |
| Learner updates/hour | 1,153 | 1,608 |
| Full-search targets/hour, fleet | 128,900 | 129,526 |
| Full-search targets/hour, ring 10 | 117,852 | 109,665 |
| Device step | 0.402 s | 0.406 s |
| Sampled UTD phase | 69.27% | 65.16% |
| Directly measured UTD sleep | Not recorded | 37.43% |

Overall full-target production was effectively flat (+0.5%); ring-10 full-target
production was 6.95% lower in this short observation. Board allocation, model
identities and game phases confound a before/after comparison. This observation
does not establish a per-ring teacher-rate improvement or an Elo/hour gain.

The direct wait timer recorded 459.24 seconds across 366 completed wait events.
The previous roughly 80% figure was a heartbeat-phase estimate: the code announces
`training` after the step, so the first update after waiting can remain labelled
as waiting. Five-second sampling adds aliasing. It would be incorrect to describe
80% versus 37.43% as a like-for-like measured reduction. Eight fully contained old
replay windows account for 276 seconds of requested sleep over 490.73 seconds
(56.24%); their incomplete coverage does not recover exact old whole-window sleep.

The deployed batch diagnostics cover 540 updates: all metadata was present,
mean effective policy rows were 263.99 per batch, mean full-target policy-weight
share was 75.45%, and no sampled batch was all-fast. Final outcome labels appeared
in 51.9% of diagnostic sample rows, versus 61.0% in the baseline; recent unfinished
games and prior interruption-only policies remain in replay. This is a sampled
label-availability observation, not a data-loss or learning-quality conclusion.

Both exact stopped-boundary and post-readiness off-server verification passed.
The pre-migration snapshot SHA is
`2f6fbfa1c075a3803ffef190b37113243377c63b0cdb80373fd439d4353464e1`;
the post-readiness snapshot SHA is
`51cc1af8746e7a1e2bca9e42dfb432617be0bab5af6e8d0b9b6095960e449c68`.
The workload, monitoring, reporting and backup services use the new
release/profile; their existing timers are active. The backup namespace and
disabled global continuity automation are preserved.

An earlier pre-migration attempt passed the GPU checks but failed when its helper
tried to overwrite the immutable neural smoke report. Automatic recovery restored
the old release from checkpoint 190,106. The corrected helper now retains separate
immutable neural evidence and pinned-loader XML/logs, then atomically publishes a
new combined report. The successful attempt preserved checkpoint
`46ce93fa07a892ceaca22391fe6077ad6b06ae7189e5a7020e1211ae0dc35354`
at step 190,377 and completed without changing the frozen learning/arena controls.

An independent final audit passed at 13:48 UTC: source/profile/gate and every
referenced backup artifact agree, all original worker identities remain healthy,
and no restarts or inference/graph validation failures were observed. Detailed
measurements and the audit are retained in
[the evidence record](starvation-correction-evidence-20260911.json).
