# Frozen recovery calibration

The opt-in `recovery-effective-control`, `recovery-effective-moderate`, and
`recovery-effective-high` arms in `scripts.run_frozen_replay_optimizer_calibration`
support the current `ring10_pie` network. Legacy optimizer-calibration arms keep
their existing contracts. Recovery arms are diagnostics; none promotes a model.

Every arm starts from the same verified champion **EMA** weights, copies those
weights into a fresh EMA, and creates empty optimizer state. Both learning rates
are required command arguments. A fresh constant schedule has no warmup, cosine
decay, inherited schedule position, or governor multiplier. The result records
the effective configuration, first-step updates, final actual learning rates,
and a complete resumable candidate checkpoint. Changing either rate on resume
fails the run-contract check.

Use a frozen replay snapshot and frozen champion publication, outside the live
run, and reserve the selected GPU before launching. The runner does not reserve
GPU ownership. Run one arm per reserved device; do not share an actor GPU without
the normal worker ownership handoff. The CLI initializes isolated compile caches
before importing the compiler. Invoking the Python function directly is only
supported for CPU or compilation-disabled runs.

```sh
python -m scripts.run_frozen_replay_optimizer_calibration \
  --config /frozen/profile.yaml --champion /frozen/learner/champion.json \
  --replay-root /frozen/replay --replay-cutoff 9000000 \
  --minimum-replay-shard-id-exclusive 8900000 \
  --replay-model-identity sha256-REPLACE-WITH-FROZEN-CHAMPION-IDENTITY \
  --arm recovery-effective-moderate \
  --effective-muon-lr 0.002 --effective-adamw-lr 0.00003 \
  --steps 2000 --batch-size 512 --max-samples 262144 \
  --evaluation-batch-size 64 --gradient-diagnostic-rows 8 \
  --budget-h100-hours 2 --output-dir /diagnostics/recovery-moderate
```

Replace the example paths, cutoff bounds, and identity with verified snapshot
values. Freeze the control's measured effective rates explicitly; do not copy
its YAML reference rates. The moderate pair above and optional `0.008/0.00012`
high-rate arm are hypotheses, not defaults or qualified production settings.
All arms must use the same source profile, replay filters, seed, batch, steps,
evaluation settings, and immutable implementation. `--dry-run` checks the pins
and partition without creating an output directory. `--stop-after-steps` permits
an operational pause; restart with identical arguments to resume. Paused wall
time still counts against the original arm's two-H100-hour ceiling.
Budget enforcement is cooperative: checks occur between training updates,
held-out forwards, and gradient heads. An operation already running may finish;
once the deadline is reached, later evaluation work does not start and the
existing checkpoint is retained with `budget_exhausted` status.

Selection uses only finalized ring-10 pie/handicap rows, optionally from one exact
teacher identity. The lower shard cutoff bounds the source scan. Hashes and
logical game/model/variant identities of selected shards are validated; unrelated
live publication rows are not exhaustively rescanned. Latest revisions of a
logical game/ply deduplicate. Selection uses 45/45/5/5 row quotas across pie classic,
pie double, handicap classic, and handicap double. Every cell must have enough
rows and at least two distinct games. A deterministic game split keeps every
selected game's positions on one side. Incomplete cell coverage is an error.

Held-out loss is computed per game with each head's actual target-weight
denominator, then position-weighted within each cell. The reported macro endpoint
uses 45/45/5/5 cell weights regardless of game length or held-out allocation.
Both raw and EMA models are evaluated against the frozen champion. This is loss
calibration, not playing-strength evidence. Optional gradient diagnostics use
at most 32 fixed training rows, do not alter weights or `.grad`, and report
weighted per-head norms and pairwise cosines over shared parameters. They are
small-batch descriptions and do not establish general gradient conflict.

```sh
python -m scripts.compare_recovery_calibration \
  --result /diagnostics/recovery-control/result.json \
  --result /diagnostics/recovery-moderate/result.json \
  --result /diagnostics/recovery-high/result.json \
  --output /diagnostics/recovery-comparison.json
```

The comparator verifies hashes and common contracts, uses paired whole-game
resampling within each cell, and keeps the predeclared cell weights. It requires
at least eight held-out games per cell by default and adjusts the one-sided
confidence level for the number of treatments. Only the EMA endpoint suggests
a follow-up strength-screen arm; raw results are descriptive. An insufficient
or nonpositive screen produces no recommendation. The next decision still needs
complete paired arena evaluations, fixed search effort, and the unchanged
promotion contract. Throughput, losses, and this bootstrap cannot authorize
production promotion.
