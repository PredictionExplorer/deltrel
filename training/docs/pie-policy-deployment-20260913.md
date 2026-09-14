# Pie policy deployment — September 13, 2026

**Deployed and verified.** The training cutover completed on September 14 at
00:57:17 UTC (September 13 local time). Training runs from commit
`c2904f933109909695a8b8da179426a18ccdd252`; the separately deployed backup tool
uses `93c8f25119858d9024913909c144c583ebc837fb`. The
[evidence ledger](pie-policy-deployment-evidence-20260913.json) records checkpoint,
profile, source, archive, test, and live-policy verification.

The previous source was
`variant-cpu-efficiency-20260912` release, commit
`8d0d643d30430afd9cd7c662fe4648aedecfca3e`, using
`profile-cpu-efficiency-20260912.yaml`. The new profile is derived from those
exact production settings, retaining model, optimizer, EMA, search budgets,
actor topology, CUDA graph capacity, and live policy publication.

The [new policy](pie-even-training.md) makes pie standard for even games,
targets 90% even / 10% handicap with both move modes equally represented, and
restricts handicap to ring 10.

## Verified result

- All ten workers stopped cleanly at learner step **268,457**, preserving
  **137,449,984 consumed examples**. The final heartbeat and checkpoint matched:
  **zero uncheckpointed steps or examples were discarded**.
- Training resumed that exact checkpoint, SHA-256
  `bfc9d3aeb8b3090cd20bad03203668c51db3126b8fa02f608cd6650c6d29ffbd`.
  Sustained worker readiness passed with zero restarts. Hardware was healthy,
  and observed loss/gradient nonfinite counters remained zero.
  Final inspection observed step **269,630** with the original new worker PIDs.
  The first subsequent recovery checkpoint at **269,457** was independently
  verified to retain optimizer, scheduler, EMA, the pie default, and scoped UTD.
- The independent, fully checksum-verified archive contains **43,722 replay
  shards** and **3,563,698 ready logical positions**: 1,550,701 in the allowed
  game categories and 2,012,997 retired-category positions. This count covers
  available retained data, not the run's much larger lifetime generation count.
  Existing off-host backup history was retained too.
- Actual selected replay contained only pie games on rings 4/6/8 and pie plus
  both handicap modes on ring 10. Its observed board-weighted handicap share was
  **9.999968%**, with only integer-row rounding from 10%. Each mode received an
  equal share within its category, to within one row.
- The new four-category weighted arena and a distinct strength epoch are active.
  Monitoring and all three protection timers were restored. The first off-host
  backup after policy activation completed successfully.

| Checkpointed deployment boundary | UTC, September 14 |
| --- | --- |
| Graceful stop requested | 00:43:33 |
| All workers stopped and final checkpoint verified | 00:45:40 |
| Local replay/weight preservation complete | 00:45:50 |
| CUDA/native/BF16 smoke passed | 00:46:52 |
| New workload launch requested | 00:47:15 |
| Sustained readiness and support restoration complete | 00:57:17 |

Local preservation took approximately **9.4 seconds** after the verified stop.
Its full checksum pass ran after learning resumed. Readiness includes actor
warmup and the shared GPU's initial arena handoff; it is not all idle GPU time.
The learner resumed updates before the final readiness declaration.

The target host passed **324 training regression tests** (one CUDA case was
reserved for explicit hardware qualification). The bounded CUDA check covered
12 allowed board/variant cases, with graph/reference logits and values exactly
matching, no graph fallbacks, and finite BF16 learner gradients/losses. The smoke
modified only an in-memory model copy. The separately staged backup tool passed
**49 server tests**, including the real timestamp-resolution regression, plus
checks against four actual historical catalogs.

## Issues corrected before deployment

- Production replay contains much more handicap than the new target. The
  prior fallback would fill a requested million-row ring-10 window with every
  allowed old sample, initially overrepresenting handicap. The new objective
  shrinks windows to the available pie/handicap and classic/double proportions.
  A representative 335k pie +335k handicap pool now selects 379,666 rows,
  comprising 335k pie and 44,666 handicap. All unused data remains stored.
- Excluded historical replay is protected from routine new-objective garbage
  collection. The stopped snapshot also preserves the available eligible data.
- The prior source-only deployment controller cannot roll back checkpoints
  containing the new pie default and scoped UTD metadata. The new controller
  uses the compatible reader for inverse profile migrations and never restores
  older learned weights. An interrupted metadata transaction has an explicit
  before/after journal and can be repaired only while the checkpoint is unchanged.

## Evidence boundaries

The existing search allocation was qualified on the old six-category workload.
Its original gate, models, reports, and cache32 admission remain pinned and
unchanged. A separate policy-transition receipt permits only the exact new
training-policy transformation; it does not claim new performance qualification.

A diagnostic reweighting of the original frozen positions gave an actor
reference-Q regret upper95 estimate of approximately 0.0532, versus 0.01972 under
the original aggregate. One pie-double position accounts for much of the
uncertainty. The champion estimate was approximately 0.00794. There were no new
swap failures. These old-model, small-sample diagnostics are not new Elo or
wall-time measurements, and the 90/10 efficacy remains unqualified. A later
search allocation change requires its own evidence; this deployment keeps the
already running execution settings.

## Backup delay and correction

The pre-existing backup completed naturally before the stop. Its long delay
came from repeatedly parsing 930 historical snapshot catalogs totaling 2.63 GB.
A subsequent backup-only release caches parsed, immutable headers within each
operation while rehashing current file bytes on every reuse. It was installed
without restarting the trainer. [Details and benchmark](backup-header-cache-20260914.md).
The first complete off-host backup using the corrected tool ran from 01:29:47
to 01:36:49 UTC and exited successfully; its catalog hash and commit marker were
verified. That approximately seven-minute duration is an operational observation
over a different snapshot, not a controlled whole-backup speedup measurement.

Target-host testing rejected the first metadata-only cache: real content changes
could retain identical timestamps on the host's coarse filesystem clock. That
version was never activated. The corrected release retains SHA-256 verification
and passed the unchanged corruption test plus forced same-signature race tests.
Its roughly 5.93× synthetic header-processing improvement is not a whole-backup
or Elo/hour measurement.

## Preservation and validation procedure

Build and CPU-qualify an isolated immutable release while training continues.
Drain the variant-network support jobs, then request a graceful stop. Require
all workers to exit cleanly and the final learner heartbeat to match the recovery
checkpoint exactly. Completed games retain outcome targets; unfinished games
flush their latest policy prefixes without invented outcomes.

The stopped preservation archive uses an independent SQLite backup plus hard
links to immutable replay/checkpoint artifacts on the same local filesystem.
It retains source profile, controls, and arena evidence. Source files are never
deleted or modified by snapshot creation. The archive's full checksum pass and
off-host disaster backup run after training resumes to reduce idle GPU time.

After the bounded CUDA/native/learner smoke, apply the validated profile migration,
switch service paths, and resume the exact checkpoint. Require sustained healthy
workers and advancing learning before restoring normal protection services.
Verify scoped credit, actual selected replay proportions, retained data, the
new four-category arena contract, and the new strength epoch. This establishes
correct deployment and continued training, not a measured Elo/hour improvement.
