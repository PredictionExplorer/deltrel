# Streaming replay validation

The first efficiency deployment's startup diagnostics showed each actor cohort
validating the complete replay publication ledger. The old validator issued two
additional SQLite lookups for every publication. This follow-up streams one joined
query while retaining the existing counters, digests, immutable context, game
identity, shard metadata, run/family filters and transaction checks. Missing games
still fail; garbage-collected shards remain valid when their durable descriptors
are valid. It does not cache or skip validation.

The joins first prove the canonical single-column primary keys and declared types
for publication, game and shard identities. Review found that an unguarded join
could multiply rows in a malformed table and make forged counters appear valid.
Explicit key checks close that regression. Noncanonical replacement schemas now
fail earlier and more strictly than the old validator; valid production schemas
retain the existing semantics and indexed access plan.

## Evidence

The predeclared performance gate was at least 10% lower median full-manifest wall
time, with all validation retained and differential tests passing. The final
guarded function passed on the target host:

| Measure | Previous validator | Streaming validator |
| --- | ---: | ---: |
| Median wall time | 33.3028 s | 29.4437 s |
| Statements on 806,717 publications | 1,613,442 | 10 |
| Maximum observed process RSS | 548,748 KiB | 549,188 KiB |

Median elapsed time decreased **11.59%**. This is a component measurement, not a
whole-startup or Elo/hour gain. Six fresh processes ran in counterbalanced order
against the same immutable 1,732,681,728-byte stopped manifest, with warm OS cache,
CUDA hidden and CPU library threads capped at one. Training remained active;
the production database was never opened for writing by the benchmark.

The [machine-readable result](replay-validation-benchmark-20260923.json) records
every run, the baseline commit, scope and candidate function digest. The earlier
unguarded screen remains server-side evidence and is not the adopted candidate.

Qualification passed 215 differential, corruption, property, revision, backup,
publication and retry tests, plus Ruff and Pyright. The frozen pre-change validator
is an independent test oracle. Regressions cover duplicate join keys with inflated
counters, missing/composite keys, noncanonical unique-index-only keys, wrong key
types, query count, transaction ownership and both scoped and unscoped validation.

## Source-only deployment recovery

The follow-up also fixes recovery after a source-only deployment commits its
authority but fails to become ready. Previously, recovery attempted an invalid
same-commit no-op migration. It now verifies the pinned source, candidate and
installed profile bytes, compares typed canonical recovery settings, records an
immutable forward-recovery intent and starts the authoritative compatible reader
from the latest stopped checkpoint. It does not reset the epoch or service ledger.

The recovery change passed 73 focused tests, including formatting-only YAML,
interrupted installation/readiness, tampered authority and intent, and first-time
auxiliary-head compatibility. Profile-changing recovery retains its existing
compatible-settings migration when one is needed.

Activation used a fresh deployment plan and profile basename with identical
training settings. The migration reported `kind=source-only`, `changes=[]`,
unchanged epoch and zero discarded learner updates. The source-only release
completed at 02:26:04 UTC on September 24, and its independent canary passed.
See the [deployment report](elo-efficiency-deployment-20260923.md) for qualification,
checkpoint continuity and precisely scoped preservation/backup evidence.
