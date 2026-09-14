# Pie policy deployment — September 13, 2026

Deployment preparation is in progress. The source is the running
`variant-cpu-efficiency-20260912` release, commit
`8d0d643d30430afd9cd7c662fe4648aedecfca3e`, using
`profile-cpu-efficiency-20260912.yaml`. The new profile is derived from those
exact production settings, retaining model, optimizer, EMA, search budgets,
actor topology, CUDA graph capacity, and live policy publication.

The [new policy](pie-even-training.md) makes pie standard for even games,
targets 90% even / 10% handicap with both move modes equally represented, and
restricts handicap to ring 10.

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

## Preservation and validation plan

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
