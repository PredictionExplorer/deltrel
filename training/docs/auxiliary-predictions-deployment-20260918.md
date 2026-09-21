# Auxiliary prediction deployment — September 18, 2026

**Deployed and verified.** The eight-H100 training service runs source commit
`7a077107b1ebba55494b09f80689989e773b9dde` from the immutable
`variant-auxiliary-predictions-20260918-v3` release. Its active profile is
`profile-auxiliary-predictions-20260918.yaml`.

The [prediction contract](auxiliary-predictions.md) describes the new targets,
official ending, compatibility and serving behavior. The compact
[deployment evidence](auxiliary-predictions-deployment-evidence-20260918.json)
records the actual checkpoint, hardware, tests, training observations and backup.

## Behavior

The existing network now learns five additional outputs:

- the opponent's next observed reply, including a pie swap;
- its own second placement in a regular Double turn;
- each player's final shores, networks and controlled corners.

Final targets use the official completed board: after a mathematical clinch,
the remaining cells are filled with the losing side's stones and scored.
Self-play does not continue beyond the existing ending. Corner-count
probabilities also provide the chance of receiving the cape-bonus bonus.

The new heads add 65,065 parameters. The trained trunk and all six existing
heads retain their original weights. Search omits the added heads at ordinary
leaves, and its utility, budgets, game mixture and publication cadence are
unchanged. Initial auxiliary loss weights are 0.1 for each future-move target,
0.05 for shores and networks, and 0.025 for corners.

## Preserved training state

The clean cutover checkpoint is **step 425,629**, epoch **10,988**, with
**217,922,048 consumed examples**. Its SHA-256 is
`fd886b27b9b5a56a1cb98c018621f4157e49e178eb4d7b5583bdbdbbdc2c65e1`.
The learner resumed that exact file: **zero uncheckpointed updates or examples
were discarded**.

The H100 test verified all **255 existing parameter tensors**, all **255
existing optimizer parameter states**, the EMA averages and update count, the
scheduler clock, and progress metadata. Only the twelve new parameter tensors
and their optimizer states start fresh. The source used stateless global
gradient clipping and had no saved adaptive clipping history to restore.

The UTD segment and strength epoch remain byte-for-byte identical to their
predeployment versions. Historical model artifacts and the current champion
retain their verified original architectures; older replay remains readable.

## Cutover timeline

| Event | September 18, UTC |
| --- | --- |
| Prior backup completed | 22:20:27 |
| Graceful stop requested | 22:20:35 |
| All workers stopped and checkpoint verified | 22:22:47 |
| Stopped-run preservation complete | 22:23:00 |
| H100 qualification | 22:23:03–22:24:04 |
| New workload launch requested | 22:24:40 |
| New training process started | 22:25:44 |
| Sustained readiness and support restoration complete | 22:35:08 |
| First new-release disaster backup completed | 22:51:40 |

The readiness interval includes replay reconciliation and actor startup.
Learning began before the final readiness declaration.

The subsequent audit observed **step 426,181**, all ten workers healthy,
zero worker restarts, healthy hardware, and zero nonfinite loss/gradient
counters in observed training metrics. Final-count supervision was observed
at step **425,630**; both future-move heads first received observed supervised
labels at step **425,790**. Unfinished prefixes correctly omit final counts.

## Validation and durability

- Final local native/CPU suite: **3,762 passed**, 17 hardware/soak tests
  deselected; no failures.
- Frontend: **254 tests passed**, TypeScript checks, lint and production build
  passed. Additional swap and readiness regressions passed.
- Target host: 442 core tests, followed by 311 tests for replay and smoke
  corrections; the final smoke/controller revision passed its focused suite.
  These scopes overlap and must not be summed.
- H100: twelve BF16 primary-output comparisons were bitwise equal after the
  additive upgrade. Native searches, detailed predictions and omission of
  auxiliary leaf computation passed.
- One compiled BF16 optimizer update used the real **512-position batch
  shape**, repeating eight distinct verified replay fixtures solely for
  runtime qualification. All five target types were present, gradients were
  finite, and peak learner allocation was **66,786,128,896 bytes**. No training
  checkpoint or replay was written by the test.
- The preserved stopped-run archive passed full checksum verification for
  **56,637 files**, including **56,153 replay shards**. The earlier recovery
  archive was verified too.
- The first successful backup under the new release contains **52,516 files /
  13,800,261,492 bytes**. Its catalog SHA-256 is
  `5b358504614b2cf955a5e368e8f40e4d220ae9e8bc478a714ff26c5374c742ea`.
  The latest pointer, catalog bytes and commit marker agree. The catalog
  includes the active auxiliary profile and its full admission evidence.

## Issues corrected during implementation

The initial GPU qualification could not find completed examples in a bounded
scan of the newest shards: shutdown had just published unfinished prefixes.
No profile migration occurred. The controller resumed the prior release from
step 425,165 with no lost updates. The corrected selector queries finalized
games separately for each mode, validates their complete trajectories, and
was checked against the preserved production snapshot before retrying.

Other fixes included clearing cached component labels when their source masks
are removed, keeping canonical hashes compatible with disabled defaults,
preserving interrupted-game destination indices, accepting mathematically
zero softmax-bias gradients in the smoke, and suppressing a second-stone
forecast when the selected action is a pie swap.

A subsequent deployment-tooling review fixed recovery for future rollouts
whose source already has auxiliary heads. Those rollouts retain the original
auxiliary settings instead of trying to repeat the one-way addition. Commit
`28410aff86d6a15f5fce7c98b295df7222f4ff6d` passed 64 focused tests and is also
installed as a separate immutable operator tool under
`/home/ubuntu/deltrel-operator-tools/auxiliary-predictions-28410aff`.
It does not change the running training release or require another restart.

## Frontend availability

The normal network-game estimate panel now shows final component forecasts,
corner-bonus probabilities, and applicable future moves. It is no longer
hidden behind developer tools. The opt-in analysis API remains compatible
with existing clients.

The local MPS service and real Classic/Double play were checked successfully.
Its existing champion 332,044 predates the new heads, so it correctly reports
these additional predictions as unavailable. Upgraded models expose them only
after all five heads have observed supervised training. A newly published
upgraded checkpoint must be loaded locally to see the new numbers; deployment
does not silently replace a verified champion with a training candidate.

The release establishes correct training continuation and target use. Improved
Elo per wall-clock hour remains a measurement to make from later evaluation.
