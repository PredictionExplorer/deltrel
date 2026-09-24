# Protected independent strength measurement

The opt-in scheduler reserves evaluation service for independent measurements
even when promotion candidates are continuously available. It changes scheduling,
not search budgets, statistical tests, game rules, or champion publication.

`orchestration.historical_evaluation.measurement_service_fraction` defaults to
`0.0`, preserving background-only scheduling. A positive fraction requires
enabled orchestration, enabled predecessor measurements, and coordinator-managed
GPU pause sharing. `measurement_max_wait_seconds` defaults to 3600. The existing
profile migration/admission process remains required before enabling a treatment.

At a lease boundary, a pending measurement receives service when its durable
service debt is nonnegative or its waiting interval has reached the configured
bound. The latter can exceed the fractional reservation: the fraction is not an
absolute cap. Shared actor catch-up cooldowns remain in force and begin after
release acknowledgement. The wait bound cannot interrupt kernels, override a
handoff failure, or guarantee that a complete matchup finishes within that time.

The reservation applies while meaningful measurement work is pending. When the
selector proves no job is due, scheduling debt resets without changing actual
service counters. Debt carries across completed jobs when more work is queued;
an idle period cannot bank hours of exclusive evaluation time.

## Measurement selection and scientific meaning

The protected selector favors the current champion against the frozen
`strength-epoch.json` anchor. Without an epoch it selects the current champion's
direct predecessor. It does not first reconstruct every unfinished bootstrap-era
link. Historical evidence remains on disk.

Once admitted, the exact candidate/baseline pair is pinned until its configured
measurement finishes, including across newer promotions and process restarts.
The ledger retains manifest paths so existing artifact collection and disaster
backup machinery protect both models before the first arena snapshot is written.

At a new epoch, the current champion may be below the epoch's learner boundary.
Its predecessor matchup can qualify measurement operation, but the strength
report excludes that pre-boundary model from new-epoch learning gains. An anchor
rating of zero is a definition; a rate requires later independent evidence.

`current_strength` is the authoritative current-objective report section. It
identifies the deployed champion, separately reports point/rate availability,
and records the measurement timestamp and age. A freshly generated report does
not reset the age of old evidence. Promotion estimates and old bootstrap
headlines remain explicit historical diagnostics. The text monitor never uses a
promotion estimate to fill a missing balanced-strength headline, and invalidates
a report whose champion has changed.

## Service accounting and recovery

`arena/measurement-service.json` is owned by the single promotion supervisor.
It stores settled promotion/measurement GPU nanoseconds, service debt, waiting
time, the pinned matchup, outstanding token ownership, and an append cursor with
the SHA-256 of the consumed coordinator-journal prefix.

Only coordinator-confirmed ready-to-release intervals earn service credit.
Request waits, actor cooldowns, and nominal session durations do not. Registration
is durable before a lease request; coordinator events settle it after normal
release, cancellation before admission, or actor-restart recovery. Replaying a
release after a process restart cannot count the same interval twice.

The read-only `measurement_service` status reports these settled counters,
observed share, debt seconds, pending wait, open leases and accounting completeness.
An active unsettled lease is visible but has not yet entered the settled totals.

Disaster snapshots preserve the ledger plus the coordinator-journal prefix and
model dependencies. Restoration may supply a byte-bound
`arena/measurement-service-restore.json` receipt. The worker first consumes all
captured release evidence, then explicitly records any still-open interval as
uncredited. It preserves settled counters, debt and the pinned matchup; it never
invents an end timestamp. Receipt consumption is atomic and idempotent. The
monitor exposes the resulting accounting gap. Missing/corrupt ordinary runtime
state cannot silently invoke this disaster-only recovery path.

## Learning-trial readiness

The existing champion warm-start experiment runner resets optimizer and scheduler
state. It is not yet a full-state continuation runner for a pure UTD trial.
Before a 1.5-versus-2.0 trial can execute, register and implement continuation of
raw weights, EMA, optimizer state, reference scheduler age and prospective replay
credit. Preserve publication cadence per fresh sample (7.5M to 10M learner
examples for candidates; 1.5M to 2M for actor snapshots).

Also declare EMA, learning-rate and model-age clocks: an unchanged update-based
clock changes its fresh-data horizon when UTD changes. The replay/history lag
cutoff therefore needs the same explicit clock treatment. These are execution
prerequisites, not settings that the current warm-start runner silently supports.
No learning trial is launched or authorized by enabling this scheduler.
