# Champion recovery: experiments and deployment, 2 October 2026

The previous learner had advanced to step 844329 without replacing the retained
step-566428 champion. The deployed recovery starts from that champion's EMA,
corrects two pie-search errors, and gathers fresh champion-generated replay.
No Elo improvement has yet been established. Frozen losses, search-reference
quality and throughput below are diagnostic measurements, not playing strength.

## Corrections and initial training recipe

Pie search previously mixed swap-inclusive values with conditional KEEP values
before applying the opening minimax transform. Swap-available policy expansion
now continues through a placement without backing up the incompatible value or
consuming a simulation. The actor also uses equal search strength for both pie
seats, as required by that transform. Rules and feature identities are unchanged;
240 native feature cases matched the previous release byte for byte.

The twelve-hour screen resets optimizer moments and both raw/EMA weights to
champion EMA. Rates are explicitly constant: Muon 0.000533667 and AdamW
0.000008005, with no warmup and EMA decay 0.9999. Replay reuse remains 1.5.
Only shards newer than 8735527 are eligible, with zero inherited replay credit.
The objective and 17.5M-parameter architecture remain unchanged. Ring-10 search
uses 53 fast / 640 full simulations, 35% full searches, and unsoftened targets.

GPU 0 learns, GPUs 1–6 generate self-play, and GPU 7 owns promotion evaluation.
The screen retains candidates at 2, 6 and 12 elapsed hours. A tested server
controller then changes only the ordinary publication cadence to three million
examples and continues training. Startup, restarts and handoff time remain in
the respective elapsed-time accounting. See [the runbook](strength-recovery-screen.md).

## Completed experiments

| Experiment | Evidence | Decision |
| --- | --- | --- |
| Three learning-rate arms | 1,000 BF16 H100 updates each; 184,000 samples, whole-game split. EMA composite: champion 4.70817; low 4.69172; moderate 4.69327; high 4.70054. Higher-rate raw endpoints regressed sharply. | Use low rates and EMA. |
| Fresh search validation | Independent 51-game, zero-PDA pie holdout: champion composite 4.70572; low-rate EMA 4.69072. | Supports the conservative choice; playing-strength validation still required. |
| Optimizer ratio followups | Half-Muon raw composite 4.62549 on that fresh holdout. Adjusted one-sided lower improvement bound versus low-rate EMA 0.01266. Double-Adam offers no compelling advantage. | Reserve a paired strength screen for half-Muon raw; do not adopt from loss alone. |
| Search/target sensitivity | 48 positions from 16 disjoint games, all four objective cells, ordinary and dense 2,048-simulation references. Full-640 action agrees with dense reference 33/48 versus 22/48 for fast-53. Every tested softening treatment worsened average target quality. | Keep stronger full search and leave optional target softening disabled. |
| Old latest raw versus EMA | Each completed 20/32 games against champion under corrected 256-simulation search. Even winning every unfinished game could not raise planned weighted point scores above 24.375% raw / 35.625% EMA. | Stop these arms for deterministic point-score futility and preserve unfinished states. This is not a confidence statement about true strength. |
| Shared graph geometry | Resident batch-512 step 0.524 → 0.418 seconds, 1.255× throughput; 8.85 GB less GPU memory; numerical gate passed. | Already enabled in production. Full calibration wall-time benefit is smaller because materialization dominates. |

The composite is policy plus 0.25 soft-policy loss, outcome loss and 0.25
score-margin loss. Report fields named `value` include the weighted score term.
Frozen calibration's handicap coverage was limited to severities 2 and 3.
Fresh independent collection produced 203 zero-PDA pie games / 42,265 positions;
a further handicap set contains 23 games / 4,610 positions across severities
4, 6 and 9, with one classic-severity-9 cohort explicitly censored.

Auxiliary gradient measurements gave no sufficient reason to remove losses.
The inference batching/compile benchmark remains queued; no unmeasured speedup
has been activated. These diagnostics support a coupled recovery experiment,
not a causal claim that any single change raises Elo.

## Release and recovery evidence

The immutable live release is
`/home/ubuntu/edgeconnect-releases/variant-strength-recovery-20261002-v3`,
production commit `7e77c135bb101f08d1534f0ca5406cb309e2e3d2`, corresponding to
main commit `79c1ee8a5885b368e4c1e047fd3bf2bf50af8129`. Production retains the
StarTrain/EdgeConnect names and existing rules/features hashes. The native
binary SHA-256 is
`04a441609de0f272de493b4bcca9c9992fce6c9ef3d9d20e72196f8a73606c4e`;
search identity is `gumbel-completed-q-v3-conditional-keep`.

Qualification passed 4,305 target-host CPU/native regression tests plus Ruff and
Pyright. Search validation also includes 49 Rust tests, seven WASM tests, 41
browser tests, TypeScript/lint checks and CUDA inference parity. Later operator
and champion-only replay changes have their own focused qualification; they
must not be represented as part of this immutable deployed release.

The active run is `/home/ubuntu/edgeconnect-runs/strength-recovery-20261002`,
owned by `edgeconnect-strength-recovery-20261002.service`. Initial profile hash:
`cf68aaad2e71c422e0f7b1086db88e952d676803e3abd39817e5ca8abb68fa2f`.
Recovery plan hash:
`efd086917ea569b22dd415cb58e5366f22674553261be6cb9e340f8e7eff7511`.
The elapsed screen began at nanosecond timestamp `1790931126840849890`.
Read `profile.sha256` for current authority rather than assuming that the initial
profile is still active.

The old run stopped cleanly at step 844329, with all workers exiting zero.
Its final recovery checkpoint is
`sha256-ff831ae8da5cfe2e1ae2897378c3a18688f3ca7a70eea5368c9a01e863a801e3.pt`.
The stopped local preservation verified 51,971 files. The final off-host
snapshot independently verified 47,451 objects at 08:20 UTC. Its catalog hash is
`338c486c589be963527d3f8c774e8be7948642245a6bb777c9dba4a874820538`.
The prior champion and complete previous run remain recoverable.

## Continued work and operational limits

Cold start exposed incomplete ring-by-mode coverage: fresh positions alone do
not satisfy the learner until every active ring meets its balanced-mode quota.
Also, an isolated fork inherited the parent's migration journal and source
authority. The normalization operator archives the exact historical bytes and
records the already-qualified source under a clean stop; it does not invent a
migration or reset weights, replay, credit or experiment time.

The new champion-only freshness mode is tested but not enabled in R3. Qualify an
explicit transition before the learner is 120,000 steps ahead of its teacher;
otherwise the old actor fallback would start using candidates. Do not solve
this by falsifying checkpoint steps or silently weakening freshness bounds.

Detailed server receipts live under
`/home/ubuntu/edgeconnect-rollouts/strength-recovery-20261002`. Durable local
operator state and copied evidence live in the ignored
`training/runs/strength-recovery-20261002/` directory. The ongoing supervisor
checks actual service ownership and receipts before acting, avoids overlapping
GPU experiments, and reports meaningful results or failures. A positive Elo
claim requires completed paired games under a matched search contract.
