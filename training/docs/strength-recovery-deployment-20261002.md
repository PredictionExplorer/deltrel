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
At the 13:22 UTC observation, the completed boundary had grown to 56 pairs /
112 games with weighted score 0.53608 and interval [0.29420, 0.81034]. The
decision remained inconclusive. The change illustrates
why the first small block should not be treated as a reliable strength estimate.

An evaluation-throughput audit found no substantial scheduler idle time. The
completed continuation wave's broker was busy for 98% of its elapsed time;
neural round trips consumed 68%, and 34.9% of physical batch rows were padding.
The arena already uses compiled BF16 inference. Its CUDA graphs are disabled.
The next proposed experiment therefore compares graph execution against that
actual control, initially holding padding sizes fixed. It has an eight-minute
adapter gate followed only if useful by a separate, at-most-45-minute search
comparison. The driver and GPU reservation still require qualification; no
graph speedup has been measured or activated. The existing finite cooldown
can retain a CUDA context, so it is not by itself proof of exclusive GPU use.

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
All 24 GPU cohorts used the retained champion. The coordinator and its nine
supervised workers were healthy, with zero worker restarts in that attempt.

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
That repair completed under one clean stop: all workers exited zero, replay
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
coverage and two workers. A subsequent optimization skips provable no-op
scheduling copies, normalizes autonomous epochs once, and rejects unrelated
provenance edits before enumeration. It passed 671 affected checks; 141
independent comparisons matched the previous implementation. The two slow
resume cases fell from about 41 seconds to 13 seconds locally with the same
coverage settings and unchanged 60-second limits. A further 1,794 tests in
unaffected modules passed. None of these changes weakens promotion thresholds or changes
the live training recipe. Final-head GitHub CI remains the merge gate.

The complete CI run for runtime-code head
`002cc0f51d9cc96cf5798f1b5e3b0bc2aea16d0d` passed all six jobs. It includes
4,516 Python/native tests, all 28 coverage floors, 1,012 web tests and 107 browser
end-to-end tests, with five existing browser-specific or opt-in skips. The
Python suite took 1,138 seconds with two workers and branch coverage. CUDA,
multi-GPU and soak tests remain separate qualifications. The evidence is
[CI run 37013445481](https://github.com/PredictionExplorer/deltrel/actions/runs/37013445481).

Detailed server receipts live under
`/home/ubuntu/edgeconnect-rollouts/strength-recovery-20261002`. Durable local
operator state and copied evidence live in the ignored
`training/runs/strength-recovery-20261002/` directory. The ongoing supervisor
checks actual service ownership and receipts before acting, avoids overlapping
GPU experiments, and reports meaningful results or failures. A positive Elo
claim requires completed paired games under a matched search contract.

## Completed screen and publication qualification

The two-hour checkpoint, step 568263, exhausted its 320-game allocation without
promotion: weighted score 55.09%, with a 90% global anytime interval of
41.98–68.53%. The six-hour checkpoint, step 572377, earned promotion against
566428 at 152 games / 76 reversed-seat pairs: weighted score 73.96%, interval
52.94–93.91%, promotion evidence 93.79 above the required 20, and no cell vetoes.
All 152 native winner histories, four allocation plans, exact R3 statistics and
the champion pointer were independently checked. The relative logistic Elo
estimate is +181, with a wide +20 to +475 interval under this arena contract;
it is not an absolute rating or an isolated causal effect of one change.

The twelve-hour endpoint was retained at step 578551. The qualified completion
helper handled the expected clean exit 78, sealed the screen, and restarted the
same immutable R3. Twenty receipt, seal, profile and clock checks passed. Only
candidate cadence changed to three million examples; weights, optimizer, EMA,
counters, replay credit and original experiment time were preserved. The
registered continuation profile SHA-256 is
`605ceefef88e9606a7ff3e9305b9c9a0f6418de59ebad5225fb3f0830ff55f1a`.
All 24 GPU actor cohorts became productive on champion 572377 after cached replay
validation, without worker restarts. Normal update-to-data waits remain expected.

The proof-preserving publication bridge migrated the promoted checkpoint to
`sha256-255dbca3a0ee33fbcc6916b961430c39903471b3bb75bc1de2662949a78b160d`.
Independent verification compared all 956 tensor leaves and training metadata
against the fixed historical rename map, and passed strict loading and
relocation. The original checkpoint and promotion proof remain intact.

The separately qualified cloud runtime uses main `2c00a32` and v3 search. Its
CPU stage passed 16 real semantic cases. Exclusive GPU qualification attempt 02
passed the same 16 cases, one and eight concurrent Standard requests, and one
Deep request. The operator restored the previous production service in 92.47
seconds total. Optional eight-way Deep was skipped by its conservative budget
guard. Attempt 01 had refused before any outage; the narrow systemd empty-field
adapter correction was reviewed and tested before the distinct second attempt.
These checks establish functional runtime qualification, not additional Elo or
general capacity. They do not perform the final public cutover.

The browser release adds a runtime-bound manifest, migrated ONNX and immutable
v3 WASM assets while retaining exact deployed legacy bytes for existing clients.
The parsed manifest and executable bytes are hash-bound before use. Chromium,
Firefox and WebKit passed real CPU/WASM Standard searches and cache/reload
checks; these are not WebGPU qualifications. Public browser deployment remains
pending authentication and exact-model WebGPU qualification. The cloud cutover
described below subsequently completed; consult current operator receipts before
asserting actual serving ownership.

`cutover_cloud_champion.py` implements that separate transaction. The reviewed
initial invocation supplies an explicit plan checksum; a boot-enabled guard
uses the checksum recorded in its protected prepared state. Guard startup is
verified before disabling production autostart or changing the YAML and current
symlink. It restores the exact old pair unless a complete acceptance commit
proves the new pair, authenticated/public health and one full public Standard
request. Original service enablement is restored before retiring the guard.
Unknown owners or file contents are never overwritten. The same-boot budget is
900 seconds, with guard recovery beginning by 540 seconds; a fresh boot records
a separate recovery budget and never resumes the old clock or trusts old PIDs.
The controller and acceptance checker passed 46 and 52 focused CPU tests,
including native v3 legality, with independent implementation review. The
acceptance checker verifies its helper bytes before execution and sends no
bearer token, redirects or environment proxy traffic.

All six CI jobs passed at `071a5d4`: 4,654 Python/native tests, 1,023 web tests,
107 browser E2E tests, coverage floors, Rust/WASM, audits and container checks.
See [CI run 37076251006](https://github.com/PredictionExplorer/deltrel/actions/runs/37076251006).
The publication qualification archive contains 257 logical files in 235 unique
objects, totaling 303,065,030 bytes, with the original champion checkpoint
referenced from its verified disaster-recovery object. A separate process read
back every object. Its manifest SHA-256 is
`463aba482e55edad904c7b3471fe403e64574652bf13d53c6051e4951f334163`, under
`/lambda/nfs/texas-north-fs/edgeconnect-experiments/strength-recovery-20261002/publication-572377-qualification-01`.

## Continuation and freshness recovery preparation, October 3

The final cloud transaction, `572377-cutover-01`, completed in 28.55 seconds.
Public Standard acceptance completed all 544 simulations / 16 candidates and
returned a native-legal move. Independent closure verified champion 572377,
the v3 runtime, service enablement, exact configuration/current-model pair,
native mapping and sole GPU ownership. Its guard is retired and disabled;
the completed plan must never be rerun. Browser serving still uses 566428/v2.

The twelve-hour checkpoint 578551 completed 320 games / 160 pairs without a
conclusive improvement: weighted score 55.97%, 90% interval 42.31–68.85%, and
`reject_max_pairs` with `conclusive=false`. All 320 native histories and nine
allocation boundaries were independently checked. Champion 572377 remains
retained. Later candidates use their own identity-bound official results;
asynchronous heartbeat fields can still describe a preceding evaluation.

All six jobs passed at `64b108de716cb9d8ce1cf453b02849bb8779c12a`: 4,893
Python/native tests, 1,023 web tests, 107 browser E2E tests and all 28 coverage
floors, plus Rust/WASM, audits and container checks. See
[CI run 37101135097](https://github.com/PredictionExplorer/deltrel/actions/runs/37101135097).
This evidence applies to that exact commit; later changes require their own
qualification.

The R4 restored-checkpoint probe passed a separate target-host CPU run against
step 585411 and a pinned, real 512-row batch. It restored model, optimizer,
scheduler, EMA, clipping state and continuation metadata, with exact batch
recollation. Independent lifecycle verification passed 54 checks. Work took
8.06 seconds and the process lifecycle 10.32 seconds, within a 120-second work
allowance plus 10 seconds for cleanup. CUDA remained uninitialized; this run
performed no optimizer update or native neural inference. The checkpoint has
its own archive-owned retention link, independent of disaster-recovery garbage
collection.

R4 remains inactive. Its transition must preserve the sealed screen, original
continuation clock, optimizer/EMA/counters, replay credit, evaluation resumes
and candidate cadence. The only intended profile changes are champion-only
replay freshness and its one fixed timestamp. Actual stopped-boundary CUDA
proof must precede durable migration intent. After intent, recovery proceeds
forward on R4; a backup delay after a productive canary must preserve training
and report incomplete proof rather than claim completion.

The recovery helpers separate these responsibilities:

- `activate_strength_freshness.py reprepare` abandons a retained pre-intent
  boundary through an append-only, numbered receipt. It preserves previous
  evidence and refuses any durable intent or authority drift. A new attempt
  still needs its own exact boundary and CUDA proof.
- The finite guard records ownership using boot identity, systemd invocation,
  cgroup and PID/start time. Same-boot guardian re-entry preserves its original
  deadline and phase. Boot admission rebases effective deadlines within the
  original expiry and cannot launch a runtime. Existing proof files survive
  recovery; a fresh recovered canary can retire with incomplete proof instead
  of issuing a second backup. Missing telemetry alone cannot certify progress
  or justify stopping an otherwise proven productive runtime.
- `strength_freshness_progress.py` joins fresh telemetry to current process
  lifetimes and externally verified champion identities. It distinguishes
  ordinary startup and replay-credit waits from current worker failures or
  nonfinite updates. Pending peers cannot hide attributable negative evidence.
- `strength_freshness_auxiliary.py` recognizes closed argument forms for
  persistent compiler and loader children of verified workers. Those children
  never certify progress. Their package and import-environment provenance still
  require separate qualification; unknown children remain pending.
- `strength_freshness_linux.py` is the concrete host-adapter preparation. Its
  execution entry points require separately pinned target-host qualification
  and protected authorization. CPU/fault tests are not Linux/systemd, CUDA,
  GPU-exclusion or live-handoff qualification.
- `strength_freshness_units.py` records support-service update intent before
  writes, checks persistent startup links and drains registered processes and
  jobs. Recovery can finish only the recorded, known file/environment changes;
  it preserves the original deadline and cannot authorize runtime actions.

The registered operator state records exact source pins, review/test receipts,
current service ownership and remaining execution gates. A proposed 45-minute
handoff ceiling is an upper bound, not an armed reservation or planned outage.
Keep R3 productive while preparing the remaining qualifications, warn at a
teacher lag of 90,000 steps and resolve before 108,000, ahead of the 120,000-step
fallback.
