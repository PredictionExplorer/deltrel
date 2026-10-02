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

## First elapsed-time endpoint

The two-hour endpoint was retained at step 568263, after 1,835 fresh updates
and 7,206.85 elapsed seconds. Its checkpoint SHA-256 is
`520394a7be087f0c549c1d90bb38f0e908bad7dccf487a77bac476cf5c2573e6`.
The checkpoint, manifest and snapshot receipt were independently hashed in the
off-host backup. GPU 7 began the matched promotion evaluation automatically.
The first completed block contained only 16 reversed-seat pairs: its weighted
score was 0.3375, with an anytime confidence interval of [0, 0.83163]. The
evaluation continued without promotion. This early block is unfavorable but
does not establish the model's strength or an Elo change.
At the next recorded observation, 12:41 UTC, the completed boundary had grown
to 36 pairs / 72 games with weighted score 0.49832 and interval
[0.18221, 0.85625]. The decision remained inconclusive. The change illustrates
why the first small block should not be treated as a reliable strength estimate.

Startup ring-10 batches initially had few outcome labels. By 12:41 UTC,
sampled gradient-diagnostic batches had outcome/score targets for 80.78% of
ring-10 positions over ten minutes and 83.59% over thirty minutes, versus
19.18% in the ten-minute window at 10:33 UTC. These are sampled batch counts
that include exact-clinch labels, not a census of finalized games. No batch
in the latest ten-minute sample was entirely without outcome labels.

A subsequent CPU diagnostic froze ten ordered-ply positions from each of the
51 previously held-out zero-PDA games. All 510 positions received identical
FP32 treatment for champion EMA, two-hour EMA and two-hour raw weights. The
selection's SHA-256 is
`98f71ad3df12ae92275beb346f18efcbdb76744c9a53b89f3ff2bfed15b53c7b`.
Source integrity, calibration partition disjointness and exclusion from active
replay were verified, including after scoring.

| Loss on the frozen CPU probe | Champion EMA | Two-hour EMA | Two-hour raw |
| --- | ---: | ---: | ---: |
| Policy cross-entropy | 2.34219 | 2.32358 | 2.27259 |
| Outcome cross-entropy | 0.62271 | 0.62151 | 0.64174 |
| Score cross-entropy | 1.78403 | 1.78235 | 1.79195 |
| Weighted composite | 4.36138 | 4.33912 | 4.30916 |

The paired-game EMA composite improvement was 0.02226, with an exploratory,
unadjusted 95% interval of [0.01356, 0.03069]. Its value-component improvement
was only 0.00162, interval [-0.00294, 0.00570]. The raw value mean worsened by
0.02101, but its interval included zero. These results support better policy
fitting and do not provide evidence of an EMA value collapse. They do not
justify switching to raw weights or changing rates. Absolute losses from this
selected FP32 probe are not directly comparable to the earlier full-data BF16
measurements. The diagnostic consumed 276.9 wall seconds, 1,047 CPU seconds
and no GPU time; its driver, receipts and results are preserved in
`training/runs/strength-recovery-20261002/evidence/two-hour-cpu-probe-01/`.

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

The [live canary](strength-recovery-canary-20261002.json) reached step 566870:
442 fresh optimizer updates. Thirty consecutive recorded metric intervals had
finite losses/gradients, the exact calibrated rates and EMA, the fresh replay
threshold, the prospective credit baseline and no unknown policy provenance.
All 24 GPU cohorts used the retained champion and all ten workers were healthy
with zero worker restarts in the current coordinator attempt.

The new run's first disaster snapshot completed at 09:48 UTC and passed
independent full verification at 09:53 UTC. Catalog SHA-256:
`99c64f9e82d4ed216b8b2dc0c809d05d9adc45b925da486801bb987dafd0be6f`.
Completed experiment artifacts also have a separately verified off-host
archive: 12,216 files / 12,881,182,411 bytes, manifest SHA-256
`d65f581fbd970d990dd062eb36dddb3a996d7d914bffcdca97436ca9453c3f88`.

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
That repair completed under one clean stop: all ten workers exited zero, replay
prefixes were flushed, and the same R3 release resumed with its original clock.
The new assignments supplied the missing classic ring-6/ring-8 coverage.
The normalization receipt hash is
`dd2957c438f62adb5118e6e953af0a10b86dd2d9fdf06d1ef1887c26251456aa`.

A separate operations package provides monitoring every 15 seconds, strength
reports every 15 minutes and disaster snapshots every 14 minutes. It passed 90
target-host tests; the authority repair passed six. The immutable R3 training
source was not edited. A later review identified a clean-budget completion
integrity-record gap. Future runners now record that check directly. For the
immutable current release, an `OnFailure` operator was armed without restarting
training; 14 compatibility tests passed against R3, and 62 checks cover the
future runner and repair. The handler requires the exact clean twelve-hour
exit-78 invocation, pinned sources, retained endpoint and a fresh successful
integrity check. It preserves original metadata and file ownership, and cannot
restart an unrelated failure or loop across later service invocations. Its
installed helper SHA-256 is
`4ef88ced72066d4ffc38244689d4fb0c304627650c116aa32702a59a522b6a56`.

The new champion-only freshness mode is tested but not enabled in R3. Qualify an
explicit transition before the learner is 120,000 steps ahead of its teacher;
otherwise the old actor fallback would start using candidates. Do not solve
this by falsifying checkpoint steps or silently weakening freshness bounds.

A separate R4 release at production commit
`7dca37252714bbe0c52d35a380ec175d743d1938` passed CPU/native qualification:
4,362 tests, Ruff, Pyright, dependency audit and source/environment checks.
The exact-node receipt retains 4,295 unchanged passing cases and records 67
fresh passes after a historical-hash test fixture correction. Its native binary
is identical to R3. R4 is inactive, lacks a CUDA qualification receipt and must
not replace R3 until an explicit profile/controller transition is tested.
The qualification receipt's file SHA-256 is
`dae84288cefaefbbc811bc3e3a1bad44af61c590b66307e12c496d173a570d44`.

Later repository validation changes are separate from both release pins.
Python CI uses two test workers with bounded inner thread pools and combined
coverage. Exhaustive browser tests are split into bounded cases without
dropping coverage; platform-specific visual baselines retain provenance.
Playwright 1.63 restores optimized Firefox WASM under its debugger, allowing
the unchanged real-model Standard-move test to finish. Statistical evidence
accumulation preserves scalar operation order, checked against the previous
implementation. Historical profile enumeration now deduplicates convergent
omission paths at each stage while preserving the exact ordered payloads and
hashes; 395 parity, mutation-isolation and authority checks passed with branch
coverage and two workers. None of these changes weakens promotion thresholds or changes
the live training recipe. Final-head GitHub CI remains the merge gate.

Detailed server receipts live under
`/home/ubuntu/edgeconnect-rollouts/strength-recovery-20261002`. Durable local
operator state and copied evidence live in the ignored
`training/runs/strength-recovery-20261002/` directory. The ongoing supervisor
checks actual service ownership and receipts before acting, avoids overlapping
GPU experiments, and reports meaningful results or failures. A positive Elo
claim requires completed paired games under a matched search contract.
