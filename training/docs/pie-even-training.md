# Pie-even training policy

Deployed on September 14, 2026 UTC, following local implementation on September
13. See the [verified deployment record](pie-policy-deployment-20260913.md).

The `ring10_pie` objective trains both classic (one stone per turn) and Double
*Star (two stones per turn). Every even game uses the pie rule: the responder
can keep the opening or swap sides. Handicap games never use pie and are
restricted to the largest board, ring 10. No no-pie even games or smaller-board
handicap games enter this objective's self-play or learner replay.

## Allocation

The global learner target is 90% even games and 10% handicap, with an equal
classic/double split within each. The existing board allocation is preserved:
5% each for rings 4, 6, and 8; 85% for ring 10.

| Board | Conditional even share | Conditional handicap share |
| --- | ---: | ---: |
| Rings 4, 6, 8 | 100% | 0% |
| Ring 10 | 15/17 (88.2353%) | 2/17 (11.7647%) |

This produces global shares of 45% classic-pie, 45% double-pie, 5%
classic-handicap, and 5% double-handicap. Simply setting 10% handicap within
ring 10 would produce only 8.5% globally. Both random and coordinated actor
paths use the same board-conditioned allocation. The auxiliary CPU actor
supplies only even pie games on its smaller board.

These are scheduling and sampled-position targets, not percentages of GPU
time. Completed game proportions vary with game duration, restricted auxiliary
actors, and available replay. If a required replay stratum is short, the
pie objective shrinks the window to preserve the declared segment and mode
proportions; it waits if a required mode has no data. It does not fill missing
pie examples with extra handicap. Both handicap modes retain the existing severity
range and playout-doubling compensation.

## Replay and recovery

Existing replay is retained. A strict board/variant filter excludes no-pie
even games and small-board handicap before readiness, selection, window reuse,
and fresh-data accounting. It applies to completed games and live policy
prefixes. Selecting zero quota for an old segment alone would be insufficient,
because shortage fallback could otherwise select it.
Under the new objective, routine replay cleanup preserves excluded ready and
superseded files as historical data, outside the active per-ring retention quota.

Fresh-data credit has a durable `ring10_pie` scope. It counts eligible logical
positions once, survives replay garbage collection, and gives no additional
credit for later outcome enrichment. Migration starts a new update-to-data
segment at the stopped checkpoint's consumed-example count and the eligible
replay counter. It does not reuse accumulated credit from retired games. The
learner rejects an unprepared legacy baseline under the new objective.

Model architecture, rule inputs, optimizer, and EMA are unchanged. A pie-default
change is compatible with an existing rules-v3 checkpoint only when the rest
of its game configuration and admitted variant family agree. Historical rules
and replay readers remain available; old games are not relabeled as pie games.

## Evaluation

Promotion and the separate strength ladder use four ring-10 categories:
classic-pie and double-pie each have weight 0.45; classic-handicap and
double-handicap each have weight 0.05. Both handicap modes cover the complete
configured severity cycle. Paired games reverse the candidate's seat.

Evaluation still schedules equal numbers of pairs per category to retain
handicap regression evidence. The aggregate score uses the 90/10 weights.
Its Hoeffding confidence sequence accounts for unequal weights: for `L`
severities and normalized cell weights `w`, each complete cycle has effective
pair count `L / sum(w²)`. The actual pair count remains separately reported.
Per-category regression guards are retained.

The versioned `pie_even` evaluation contract includes the new cell set, weights,
and statistical method. Old results and partial games remain in their original
namespaces and cannot enter this contract. Migration also starts a new strength
epoch anchored to the retained champion, with the historical search budget.
Elo/hour remains unavailable until the new measurement has sufficient evidence.

## Preparing a profile

`configs/h100-8gpu-pie-even.yaml` is a validated reference profile based on the
earlier largest-board template. It is not a replacement for the current
server's optimized execution profile. Prepare the actual candidate from a copy
of that frozen active profile:

```sh
cd training
PYTHONPATH=. .venv/bin/python scripts/prepare_pie_training_profile.py \
  --config /absolute/path/to/copied-active-profile.yaml \
  --output /absolute/path/to/new-pie-profile.yaml
```

The preparation command writes a new file, refuses to overwrite an existing
file, and performs no deployment. It retains the source's optimizer, model,
search budgets, actor topology, batching, inference settings, and publication
cadence. Existing hardware/search admission checks still apply.
For a source with an already admitted search allocation, the separate
`prepare_pie_policy_gate.py` tool can pin an exact policy-only continuation:
it revalidates the original source gate and every underlying artifact, permits
no execution changes, and records that performance on the new distribution is
unqualified. The original six-category measurements are not relabeled as new
90/10 evidence. Actual Elo/hour must be measured under the new contract.

At the later deployment, use the established checkpointed stop and profile
migration procedure. The migrator backs up the old UTD segment and strength
epoch, verifies the eligible replay boundary again before applying, preserves
old arena evidence, and rolls back the metadata transaction if a write fails.
Run the state preflight against the new profile before restarting workers.
Rollback must also use a reader that understands scoped credit and the pie-default
checkpoint compatibility; an older binary is not assumed to read new checkpoints.
