# Training recovery without increasing model size

The 17,467,840-parameter network retains its architecture, rules, search budgets,
and four-category 45/45/5/5 objective. This release repairs missed plateau
responses and adds bounded exposure to fresh games from the retained champion.
It also provides separate raw/EMA and full-state replay-reuse experiments.
Implementation correctness is not evidence of improved playing strength.

## Durable plateau decisions

Completed rejection evidence is independent of the mutable latest-candidate
pointer and current arena status. Verdicts are scoped to champion and evaluation
contract; the first scan can recover pre-upgrade terminal results. A recovery
receipt is saved in the same checkpoint as the learning-rate reduction. Restart,
new candidate publication, and ongoing arena sessions cannot lose or double-count
that response. A further reduction requires qualifying candidates trained after
the previous reduction, rather than consuming an old backlog repeatedly.

The existing floored, noncompounding governor policy remains authoritative. The
live profile's 0.5 reduction, 0.25 floor, and optimizer-state-clearing setting are
preserved. Weights and EMA are retained. The in-flight arena keeps its exact
models and accumulated results when the learner changes its rate.

## Protected champion replay

See [the replay contract](protected-champion-replay.md). The first treatment uses
a 25% maximum protected share and a six-hour publication window. Only the
verified current champion can receive the exception. Its identity and model step
remain truthful. Game publication origins survive prefix/final revisions; legacy
rows with unknown origins and pre-activation games receive no exception. Other
stale models remain excluded. Shortages use ordinary replay rather than waiting
for champion games. No historical replay is credited again to the UTD budget.

## Separate experiments

The [averaging diagnostic](checkpoint-averaging-diagnostic.md) freezes raw and EMA
weights from one recovery checkpoint and one champion. Both arms use the same
four rule categories, openings, reversed seats and search budget. Sessions are
resumable and require exclusive GPU ownership. The stopped deployment smoke
runs two bounded sessions; incomplete games are preserved and never presented
as a completed strength measurement. Promotion state is never changed.

The reuse treatment changes 1.5 to 2.0 consumed training examples per fresh replay
position. Model, optimizer moments, EMA values, and schedule progress resume from
the complete checkpoint. LR age advances by 0.75 reference updates per update;
EMA decay becomes the original decay raised to 0.75. Candidate and self-play
publication intervals scale from 7.5M/1.5M to 10M/2M consumed examples. Replay age
and plateau lag limits scale proportionally. Migration starts a prospective UTD
segment; there is no catch-up credit. Reversal to 1.5 preserves the reference
clock and learned state. Malformed or incompatible clocks fail before state load.

Prepare each treatment separately, from an immutable source profile:

```sh
python -m scripts.prepare_training_recovery_profile \
  --source-profile /run/active-profile.yaml \
  --output /run/status/recovery-inputs/protected.yaml \
  --treatment champion --activation-ns <fixed-UTC-nanoseconds>

python -m scripts.prepare_training_recovery_profile \
  --source-profile /run/status/recovery-inputs/protected.yaml \
  --output /run/status/recovery-inputs/reuse-2.yaml --treatment reuse
```

Preparation is not activation. Exact, checksummed transition receipts preserve
the previous search-admission chain without claiming that the learning treatment
has passed a strength experiment. Receipts, verdicts, and their dependencies are
included in stopped preservation and disaster recovery. Profile and source
migrations use the supervised graceful controller.

## Rollout and interpretation

Initial production activation keeps reuse 1.5 and EMA's existing reference decay,
so the corrective changes can be observed before a separate reuse comparison.
The 2.0 profile and diagnostic tool are deployed as explicit experiments. Compare
fixed endpoints against the same champion at equal search budgets and report
classic, double, and both handicap categories separately. Do not infer strength
from GPU utilization, training loss alone, or a partial arena. No larger model,
weaker promotion threshold, or changed game-type objective is introduced.

Qualification and the final deployment outcome are recorded in the accompanying
deployment evidence after target-host testing and the live canary complete.
