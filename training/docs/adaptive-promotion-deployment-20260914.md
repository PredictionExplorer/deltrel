# Adaptive promotion deployment — September 14, 2026

Deployed successfully on September 15 at **00:43:29 UTC** (September 14 in
Chicago). All five managed services, including training, use commit
`4f6027c50117eee8b66b59dfebbef7257d47fe12` from the immutable
`variant-adaptive-promotion-20260914-v3` release. The active profile is
`profile-adaptive-promotion-20260914-v3.yaml`.

The [evidence record](adaptive-promotion-deployment-evidence-20260914.json)
contains the release, profile, checkpoint, test, GPU, preservation, migration,
and recovery records. The [policy description](adaptive-pie-promotion.md)
documents the sampling and statistical design.

## Active behavior and preserved state

The production profile differs by exactly `arena.allocation_policy`:
`equal_cells` became `adaptive_pie`. Promotion keeps its 45/45/5/5 score weights,
starts with 32 games, uses 90/10 ordinary continuation rounds, and adds bounded
handicap reviews when warranted. All evaluation remains on ring 10, with pie
for even games and both move modes represented.

The new promotion contract is
`sha256-400dc692f581a442f033b24d3edf709be6aec3e57941db30c854f77a84fcd820`.
Its initial durable allocation and all 32 started games were independently
inspected. Candidate 288097 is being evaluated against champion 229501. Older
results remain in their original contract namespace; a stale last-decision
status file is not evidence that the current worker is evaluating that old
candidate.

Model, optimizer, EMA, training mixture, replay settings, search budgets,
publication cadence, and UTD target are unchanged. The UTD segment and strength
epoch files match their predeployment SHA-256 hashes exactly. The separate
1024-simulation strength contract remains
`sha256-b62ec3697a8134c1017892a6950870656ab2e22b5764979ee6b89ad08440c8d8`.

All ten workers exited cleanly at final step **299595**, with **153392640**
consumed examples. The stopped heartbeat and verified checkpoint agreed:
**zero uncheckpointed updates or examples were discarded**. Training resumed
that exact checkpoint:
`d24164b4e85da177213b5ec02516c745a0749af85d2b8f7f2c6e6a9fc1508b15`.
Sustained readiness passed with zero worker restarts. A subsequent independent
audit observed step 300174 and verified the active executable, environment,
working directory, Python entrypoint, and import paths.

## Final cutover

| Boundary | September 15, UTC |
| --- | --- |
| Graceful stop requested | 00:27:01 |
| Clean stop and checkpoint verified | 00:28:35 |
| Preservation snapshot complete | 00:28:46 |
| Required production GPU checks complete | 00:29:31 |
| Separate graph diagnostic complete | 00:30:06 |
| New workload launch requested | 00:30:31 |
| Sustained readiness and support restoration complete | 00:43:29 |

Learning advanced during startup, before the final readiness declaration.
Readiness includes actor warmup and the shared GPU's evaluation handoff; the
entire interval is not idle learner time.

## Verification and data retention

The server passed 612 tests on the initial implementation, 190 focused tests
after the shutdown correction, and 177 tests after correcting GPU-check scope.
These suites overlap and should not be summed. Source manifests prove that
each subsequent release changed only the stated correction and its tests/docs.
All 103 dependency versions and the native binary stayed unchanged.

The required H100 check used the actual BF16/compiled promotion runtime, which
has CUDA graphs disabled. It exercised 18-game cohorts in both move modes,
256 simulations per root, shared inference, and two successive search waves
with preserved histories. Production batches used the expected 32-row padding;
inference completed without failures. A separate graph worker passed strict
reference comparisons with 24-row padding. Both workers rechecked the pinned
checkpoint and profile, stopped-worker inventory, and GPU ownership. No
training replay or weights were written by either check.

Three stopped-run archives are retained and fully checksum-verified:

| Checkpoint step | Replay shards | Files |
| ---: | ---: | ---: |
| 298764 | 48900 | 49288 |
| 299179 | 52467 | 52856 |
| 299595 | 51880 | 52270 |

These snapshots overlap. They use independent SQLite backups and hard links
to immutable replay/model artifacts, preserving earlier data even if normal
retention later removes its live path. Completed outcomes and flushed policy
prefixes remain available; unfinished games were not assigned invented outcomes.
Their union retains **60356 distinct immutable replay files**. The latest
snapshot contains 4098815 ready sample rows; logical positions are not summed
across overlapping revisions or snapshots.

The first backup from the new release ran from **00:43:29 to 00:52:49 UTC** and
exited successfully. Its committed catalog contains 49649 files, the exact
active profile, and the new admission evidence. Its catalog SHA-256 is
`67dafbecc9811896fc9046f030a988513bd3ba1567087598f841f4ffa4a0e3da`.
The latest pointer, catalog bytes, and commit marker agree. All normal timers
are active, the eight GPUs report healthy hardware, and loss/gradient nonfinite
counters are zero. The only remaining monitor warning is the expected absence
of a completed connected strength measurement.

## Issues found and corrected

The first attempt caught a pre-existing shutdown telemetry defect. The final
heartbeat reported the correct step 298764 but retained 152966656 examples
from the preceding UTD wait; the verified checkpoint contained 152967168.
The checkpoint, one completed replay batch, checkpoint publication, and stopped
timestamps proved the missing 512 examples were already trained. Original raw
records were preserved before correcting only telemetry and resuming the same
checkpoint. No profile migration had occurred.

The learner now publishes step and examples together, and shutdown refreshes
both from the learner. The controller also has a strict, evidence-backed path
for older releases with this known stale-counter pattern. It retains a
read-only proof inside the run so backup and restoration preserve the audit
trail. Unexplained mismatches still block deployment.

The second attempt was stopped by a supplementary graph comparison in a
mixed production/graph test process (`policy_logits` maximum absolute
difference 0.125). This occurred before profile migration. Training resumed
its unchanged checkpoint, and the failure record was retained. Audits found
no omitted rule inputs; the numerical root cause was not established.

The final check tests the actual production runtime first, then the graph
diagnostic in an isolated process. Both passed. Production inference code and
settings were not changed, and comparison tolerances were not loosened. The
isolated result does not explain or erase the earlier mixed-process discrepancy,
nor does it establish parity for every future model or initialization order.

The backup service previously used a separate `93c8f25` release. Deployment now
pins per-service source paths and moves that service into the unified release,
so its parser and complete nested admission-evidence support match the new
profile. The corrected backup-header cache remains included.

Actual Elo/hour improvement is not established by deployment or synthetic
cost estimates. It must be measured from the independent strength ladder.
