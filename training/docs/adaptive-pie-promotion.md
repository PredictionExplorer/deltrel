# Pie-heavy promotion

`arena.allocation_policy: adaptive_pie` changes promotion sampling and its
statistical test together. It requires `pie_even` evaluation on ring 10. The
score remains 45% classic-pie, 45% double-pie, 5% classic-handicap, and 5%
double-handicap. Training still uses its existing 90/10 mixture across boards.

## Work allocation

The initial check covers every configured handicap severity in both modes,
with the same number of even-game pairs. The current four-severity profile
therefore starts with four pairs per category: **16 pairs / 32 games**.
Each pair reverses the candidate's seat under matched opening conditions.
Even games always use pie; handicap games never do.

Ordinary continuation rounds add nine pairs to each pie category and one pair
to each handicap category: **90% even pairs / 10% handicap pairs** after the
initial check. These are game allocations, not GPU-time fractions. The initial
coverage and targeted extra checks mean the complete evaluation's handicap
fraction can exceed 10%.

Extra handicap work is limited to one complete severity cycle per allocation
and at most half of the total pair budget. This exceptional review allowance
is large enough to detect a severe regression in both handicap modes under
the existing statistical guard; ordinary continuation stays at 90/10. A category qualifies when
its observed score is below the regression floor and its uncertainty still
crosses that floor, or when reducing its uncertainty could change the aggregate
decision and has better estimated benefit per search cost than another pie
pair. The estimate uses the fixed search/PDA budgets and an approximate
inverse-square-root uncertainty reduction. It is a scheduling heuristic, not
a measured optimum. Before promotion, a suspected handicap regression receives
the same bounded additional review while budget remains. Once this review
starts, it continues with handicap-only allocations while the suspicion remains,
even if the extra evidence temporarily weakens the aggregate promotion result.
It ends when the suspicion clears, the candidate is rejected, or its bounded
allowance is exhausted. This prevents switching back to unnecessary
pie games during an unresolved targeted review.

The maximum budget is the old number of categories times
`max_pairs_per_ring`: the current profile remains **160 pairs / 320 games**.
Individual pie categories may now exceed 40 pairs. Reaching the budget without
statistical evidence remains an inconclusive rejection. Exhausting additional
review does not prove that every handicap category is safe.

## Statistical validity

Each category has a separate anytime confidence sequence for diagnostics and
regression guards. Pie evidence updates after each contiguous complete pair.
Handicap severity still follows the fixed 2/4/6/9 sequence, and a conservative
partial-cycle bound allows its guard to update without waiting for another
category. A missing pair leaves later pairs outside that category's statistical
prefix until it finishes.

For a handicap cell with `L` severities, `n = qL + r` observations and cell mean
at most `mu`, the sum of its expected scores is at most
`qL*mu + min(r, L*mu)`. Substituting this upper bound into the existing Hoeffding
mixture gives a process dominated by the valid process for the true fixed
severity means. At complete cycles it reduces to the previous per-cell test.
The lower-tail test applies the same construction to `1 - score`.
This assumes independent pair seed streams and a stationary mean within each
severity; the two games in one reversed pair may be arbitrarily dependent.
The confidence-sequence framework is described by
[Howard et al.](https://arxiv.org/abs/1810.08240).

The aggregate uses one joint Hoeffding mixture over the two pie strata and the
eight handicap mode/severity strata. Pair coefficients are fixed before play:
initial handicap pairs receive coefficient 1/9, and other pairs coefficient 1.
This discounts the extra initial coverage in the aggregate test, while keeping
its full evidence in the regression guards. For weighted observed sum `S`,
squared coefficient sum `V`, and coefficient mass `A_s` in each stratum, the
null expectation is bounded by
`M(mu) = max sum(A_s * mean_s)` subject to the fixed objective mean being at
most `mu` and each stratum mean lying in [0,1]. Sorting strata by `A_s / w_s`
solves this fractional-knapsack problem exactly. Each mixture term is
`exp(lambda * (S - M(mu)) - lambda² * V / 8)`, no greater than the valid
true-mean process for every null parameter. Rejection applies the construction
to `1 - score`. This avoids the substantial power loss of summing four separate
confidence bounds.

Aggregate decisions use only **completed, previously committed allocations**.
The next allocation is selected using outcomes from completed previous work.
This prevents asynchronous game duration or completion order from selecting a
favorable subset of outcomes. An ordinary continuation needs 20 pairs / 40
games, even when its handicap severities have not completed another cycle;
it never waits for four continuation rounds to form one severity cycle.
Per-category regression vetoes can act earlier under their independent anytime
bounds. Summaries without a verified completed allocation expose no actionable
aggregate test. Handicap point estimates give each severity equal weight.
Estimates are descriptive under adaptive stopping; valid bounds determine
promotion.

Promotion requires the aggregate lower bound to exceed the score corresponding
to 0 Elo. Rejection requires the upper bound to fall below the score
corresponding to +35 Elo. The latter is a separate rejection hypothesis, not a
minimum gain required for promotion. A separately controlled per-category test
still vetoes a demonstrated regression below -100 Elo.
With alpha and beta both 0.05, the aggregate bounds have at least 90% simultaneous
two-sided coverage; each direction has 95% coverage. Aggregate e-values and
confidence bounds come from the same joint test.

## Resuming and preserving evidence

Every allocation is durably recorded before games start. A restart finishes
its existing targets before choosing another ordinary round. Absolute
per-category indices preserve seeds, severity coverage, completed outcomes,
and partially played seat reversals. Result metadata reports actual completed
shares separately from the fixed score weights.

The new versioned contract has separate result, resume, and allocation files.
Old promotion evidence is retained in its original namespace and is not
reinterpreted under a new test. The continuous-profile migrator supports this
change at its established stopped-run boundary. It preserves replay, learner
and optimizer state, update-to-data credit, and the existing strength epoch.
Historical 1024-simulation crossplay keeps its original equal allocation and
contract, including in monitoring and strength reports.

Shutdown publishes the learner step and consumed-example counter together.
For older releases that leave a stale sample counter after an update-to-data
wait, deployment can reconcile stopped telemetry only when the verified
checkpoint, the last wait, the same replay window's completed batches, and the
checkpoint publication prove the exact difference. Raw records and a bounded
metric tail are preserved before the status field is corrected. Checkpoint,
replay, training credit, and original heartbeat timestamps remain unchanged;
an unexplained mismatch still stops deployment.

## Verification

Deterministic controller fixtures exercise the real statistical test and the
durable scheduler together. A candidate winning every pie pair and splitting
every handicap pair promotes after 36 pairs, with no extra handicap review.
With the same pie outcomes but losses in both seats of every handicap pair,
the candidate is rejected by the actual regression guard at 108 pairs; the
targeted review remains active when its aggregate evidence dips. These fixtures
verify decisions and budget behavior, not playing strength or H100 throughput.

Separate tests enumerate linear-program vertices to check the null-expectation
solver, verify that composite evidence never exceeds the corresponding
true-mean test process, and exclude uncommitted outcomes from global evidence.
Fault tests cover corrupted or missing allocation records, partial seats,
completion-order changes, and a crash between saving a plan and saving a result.

The [seeded synthetic comparison](adaptive-promotion-benchmark-20260914.json)
uses 256 trials per gain scenario, a 160-pair maximum, and constant 0.5 handicap
pair scores. Each synthetic pie pair is a maximally correlated Bernoulli
observation. It compares the statistical tests under controlled allocations;
it does not reproduce the full production review and cost-selection policy.

| Pie pair-score mean | Old promotion rate | New promotion rate | Old mean cost proxy | New mean cost proxy |
| --- | ---: | ---: | ---: | ---: |
| 0.65 | 37.5% | 58.6% | 260.6 | 165.3 |
| 0.75 | 93.4% | 100.0% | 160.1 | 100.8 |

The search-cost proxy is about 37% lower in these two cases. It accounts for PDA
budgets but excludes actual game lengths, inference batching, and GPU time.
A separate heterogeneous-null stress test used outcome-dependent extra
allocations and observed zero false promotions in 1,000 trials; this is an
empirical check, not a replacement for the mathematical error bound. Source
digests in the report pin the tested code. Reproduce it with:

```sh
cd training
PYTHONPATH=. .venv/bin/python scripts/benchmark_adaptive_promotion.py \
  --trials 256 --null-trials 1000 --seed 71341 \
  --output /absolute/path/to/new-benchmark.json
```

## Preparing activation

Prepare from a copy of the actual active profile to retain its measured search,
inference, hardware, and learner settings:

```sh
cd training
PYTHONPATH=. .venv/bin/python scripts/prepare_pie_training_profile.py \
  --config /absolute/path/to/copied-active-profile.yaml \
  --output /absolute/path/to/new-profile.yaml \
  --adaptive-promotion
```

The tool publishes a complete validated file, refuses to overwrite a file, and
does not deploy. Frozen reference profiles and their admission receipts remain
unchanged. For the optimized production profile, retain the complete admitted
search-evidence chain with an allocation-only receipt:

```sh
PYTHONPATH=. .venv/bin/python scripts/prepare_pie_promotion_gate.py \
  --source-profile /absolute/path/to/registered-active-profile.yaml \
  --target-profile /absolute/path/to/new-profile.yaml
```

This receipt permits exactly the promotion allocation change. It rehashes the
original evidence, including the previous training-policy and controlled-cache
receipts, and makes no new throughput or strength qualification claim. Backup
and offline restore retain and verify that full chain. When support services
use a separate release, the deployment plan pins each service's source release
so its parser and backup code move with the new profile.

Deployment must follow the existing release verification, admission,
checkpointed stop, migration, and resume process. A changed sampling schedule
alone is not evidence of better Elo per hour; compare actual decision cost and
the independent strength ladder after activation.
