# Elo efficiency: measurement and reliability release

This release implements the first production engineering stage of the
[efficiency plan](elo-efficiency-plan-20260923.md). Experimental changes to
learning reuse, optimizer clocks, architecture, precision and search budgets
remain gated by independent strength evidence and a declared compute budget.
No Elo/hour multiplier is claimed by this release.

## Implemented behavior

- Independent strength work receives protected service even with promotion
  candidates waiting. The pilot reserves 20% of evaluation service while a
  measurement is pending, using actual coordinator-confirmed GPU lease time.
  The one-hour wait bound applies at safe lease boundaries, not within kernels.
- Measurement selection targets the current champion/epoch instead of first
  draining bootstrap-era work. An admitted matchup stays pinned across restart
  and later promotions. The existing search/statistical contracts are unchanged.
- Activating protected service creates an explicit epoch at the stopped boundary,
  preserving the prior epoch in the migration archive. Pre-boundary pilot results
  cannot masquerade as new-epoch learning gains.
- Current strength reports identify the champion and age of independent evidence.
  Missing strength never falls back to an old bootstrap or promotion Elo estimate.
- Actor pause control no longer blocks on diagnostic work-scheduler metadata while
  another thread loads a model. Durable phase, cohort and broker diagnostics make
  future handoff failures diagnosable.
- Cancellation, rejected requests, actor restart and graceful shutdown settle
  accounted leases only through authoritative coordinator events. Release evidence
  is durable before acknowledgment; an unreaped owner never earns a false release.
- History admission and refill account for forecast game duration and learner
  progress, preserving active games and existing labels. New actor metrics include
  PID identity. Current-state monitoring excludes retired workers and old processes.
- Recent learner throughput exposes its actual observed interval, fresh-position
  rate and measured wait/update components. These are not GPU-utilization claims.
- Calibration splits whole immutable games, deduplicates logical position revisions,
  and bootstraps game-level observations. Per-head weight/mask accounting is stable
  across evaluation chunks. Updated calibration evidence uses schema 2; schema 1
  cannot pass the new selection gate retroactively.

## Integrity findings addressed during qualification

The target host reproduced same-size writes whose device, inode, size, mtime and
ctime all remained unchanged. A pre-existing writable mmap also changed bytes
without filesystem-change notifications. Neither stat signatures nor notification
streams prove payload integrity on this host.

Control metadata therefore reuses parsed objects only after checking current
pointer/manifest contents and **fully verifying checkpoint bytes on every lookup**.
Actual weight loading retains its independent validation. Failed verification
cannot fall back to a previous successful cache entry.

Protected planning avoids unnecessary verification by selecting the required
identities first and skipping manifest work while a pinned job has no service due.
The read-count regression uses 42 historical publications: each warm planning pass
verifies the single needed checkpoint instead of all 42. This is an I/O-work
reduction, not an observed whole-training or Elo/hour multiplier.

## Admission, preservation and recovery

The profile preparation tool changes only four declared scheduling settings:

| Setting | Prepared value |
| --- | --- |
| `historical_evaluation.measurement_service_fraction` | `0.2` |
| `historical_evaluation.measurement_max_wait_seconds` | `3600.0` |
| `model_refresh.history_horizon_enabled` | `true` |
| `model_refresh.history_horizon_initial_seconds` | `3600.0` |

Typed disabled defaults retain the old canonical profile identity. Enabled
settings require an immutable scheduling-transition receipt that revalidates the
full existing auxiliary, promotion and search-admission evidence chain. The
receipt does not claim a new search-quality or playing-strength qualification.

Disaster snapshots and stopped archives preserve the scheduling ledger, the exact
coordinator-journal prefix, pinned model artifacts and epoch anchor. A disaster
restore explicitly marks an unclosed GPU interval as uncredited; it does not
invent a duration or discard settled debt. Graceful shutdown records release only
after the actual owner process group has been reaped.

The running production lineage still uses the previous StarTrain/EdgeConnect
identities. Its release is a tested backport of the same changes, preserving those
identities and its existing native binary/dependencies. This avoids coupling an
efficiency rollout to the separate Deltrel checkpoint/replay identity migration.

Deployment uses the existing supervised graceful controller: qualify source and
profile while training runs, drain support jobs, checkpoint and stop workers,
preserve the stopped run, run the bounded CUDA smoke, migrate, resume the exact
checkpoint, require sustained readiness and restore support services. Recovery uses
a reader that remains compatible with the newest durable state.

## Validation record

Qualification includes full local and target-host CPU/native suites, targeted
adversarial cache and lease tests, lint/type checks, the stopped-run CUDA smoke,
and post-deployment state/backup verification. Final commit identities, suite counts,
cutover checkpoint and deployment outcome are recorded in the deployment evidence
document once the supervised rollout completes.

The initial broad local run found three integration failures: an outdated
calibration helper caller and two fixtures requiring explicit treatment of the new
disabled defaults. These were corrected and their focused regressions passed.
The initial target-host run exposed the real checkpoint-cache integrity defect
described above; all 28 cache tests then passed on that host. Failed qualification
evidence is retained rather than relabeled as success.

## Remaining gated work

Arena graph/batching changes, persistent evaluator residency, typed asynchronous
inference, dedicated-evaluator topology, learning reuse and distilled actors remain
separate benchmark/experiment stages. They are not silently enabled by this release.
The existing warm-start runner is not a full-state continuation runner for a pure
reuse trial: optimizer state and LR/EMA/replay-age clocks need explicit support.
See [trial readiness](protected-strength-measurement-20260923.md#learning-trial-readiness).

The first live canary must establish continued learner/self-play progress, successful
protected measurement slices, clean lease settlement, fresh identity-correct
telemetry and a verified disaster snapshot. Independent Elo/hour remains unavailable
until the required new-epoch measurement evidence exists.
