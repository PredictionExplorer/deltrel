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

The corrected head `6b5d88ffa763c80dfece3a1948469fc8cf6def17` passed all six CI
jobs: 5,186 Python/native tests, 1,023 web tests and 107 browser E2E tests, plus
coverage floors, Rust/WASM, audits and container checks. The preceding run
exposed two test fixtures that resolved fake process IDs through real Linux
`/proc` paths. The correction supplies explicit fake origins and a regression
that forbids those reads; production validation is unchanged. See
[CI run 37113784678](https://github.com/PredictionExplorer/deltrel/actions/runs/37113784678).

The local legacy control bundle contains 627 files: 600 unchanged R4 base files,
17 explicit control/proof overlays and 10 test files. Its namespace rendering
passed 403 CPU tests with 30 native-dependent skips, followed by 31 affected
tests after an AST-identical formatting correction. An independent reader checked
the complete tar inventory and namespace/source bindings. Control imports may
use explicitly inventoried copies of the original R4 modules; learner, actor
and probe imports must still use the original immutable release. Local absence
of the legacy native extension remains a target qualification prerequisite.

The reviewed target CPU protocol has a 600-second total window and twelve
registered dummy units. A separate dispatcher owns all case work and fixture
children; the observer only reads facts and appends evidence. One fixed oneshot
sleep unit serves sequentially as the queued-start barrier and the support
transaction specimen. Its authorization never changes, and its queued duties
must finish and drain before the registered support file variants are applied.

Dispatcher admission checks the actual process start and independent systemd
runtime/stop limits against the original 390/405-second endpoints. A result must
be produced before the work cutoff; the observer checks natural exit and seals
that source-bound proof by 405. Later consumers validate the recorded proof
without presenting its old observation time as current. Cleanup independently
stops and drains the dispatcher before workload file removal. A missing or
failed dispatcher barrier still permits bounded stop attempts for known owned
workloads, but never permits deletion or aggregate success. Cleanup remains due
by 540, observer termination by 575, publisher termination by 595, and external
read-only auditing by 600. Setup delay never renews any endpoint.

Dispatcher, observer and publisher definitions remain as three explicitly
inventoried inert inputs, disabled and without boot links. The observer has
only scratch/evidence write access. The dispatcher, cleanup, publisher and fixed
support guard retain narrowly reviewed `/etc/systemd/system` parent access for
atomic replacement and removal of their fixed dummy resources. Closed dispatch
still checks exact unit names, paths, known bytes and owned links. Actual mount,
unit, job, descendant and retirement behavior remain target qualification gates.
The installer, immutable legacy rendering, complete source/import/environment
inventory, actual collector and frozen target plan still require independent
review before any target arming. No production unit or GPU experiment is enabled.

Preservation schema v2 corrects two unsupported telemetry assumptions. A
periodically rewritten heartbeat is not an inference completion timestamp, and
a long-lived cohort search need not refresh its semantic progress timestamp
every 120 seconds. The new pure verifier instead requires two ordered finite
learner metric events after cleanup, or condition-locked broker phase/counter
pairs proving subsequent physical inference for all six actors. Exact R3 source
ordering and the one-broker-per-process entrypoint bind these alternatives.
Heartbeat freshness, ownership, source/profile/native pins, finite metrics,
recipe/EMA/replay credit, known teachers and nondecreasing counters remain
required. Counter resets cannot be hidden with offsets or champion changes.
This does not qualify the actual collector: job-age provenance, raw observation
joins and origin/environment/boot digest encodings remain separate requirements.

Read-only host inspection also explained why persistent Inductor compiler
workers carry a different import path from their parents: Torch materializes
`sys.path` into `PYTHONPATH` and an unset library path into an empty string. An
optional policy now pins those two exact compiler values after the closed
compiler command and parent checks succeed. Other child types still require
strict parent inheritance. Saved R3 observations and pure matcher tests do not
qualify the future R4 process environment or package closure.

The initial `75b8799` joint local check passed 459 CPU/native cases across the harness
and affected adapter, proof, support and compiler-policy tests, with no skips.
Ruff, formatting and Pyright also passed on the same unchanged source bytes.
This is the preparation code milestone, not an actual target-host experiment.

Full CI on `75b8799` passed five jobs but stopped the Python/native job after
4,677 passes on one real-fork fixture that inherited a non-default SIGCHLD
handler from its shared test process. The production guard correctly refused
that state. The fixture now runs in a fresh bounded interpreter, with an
explicit foreign-handler regression and no weakening of the runtime guard.
The browser job recorded 106 clean passes, one successful WebKit setup-reload
retry and five skips; the saved error preceded the gameplay/persistence body,
so no persistence bug is established and its assertions remain unchanged.

The separate-dispatcher and preservation-v2 update passed a final joint run of
560 CPU/native tests with two workers, zero failures and zero skips. Ruff,
formatting and Pyright passed on the same source bytes. Independent reviews
closed the dispatcher terminal-grace mismatch and verified the source-backed
work proof. These results qualify local preparation only; actual target
rendering, collection, Linux behavior and CUDA remain unqualified.

The following collector components use only the Python standard library:
closed, bounded file/process/systemd/NVML reads; canonical unit, environment,
boot and process-origin digests; process ownership and kernel birth joins; and
strict normalization of the existing R3 metadata and metric records. Private
command and environment bytes stay in memory. Cached source/native references
are checked against qualified file identities without claiming a new payload
hash. Unknown processes, changed imports, reused-PID heartbeats, unsupported
records and unavailable time provenance refuse collection.

The digest contract excludes a timer's changing next-elapse values from its
static definition and explicitly marks service environment as inapplicable to
timer units. Worker restart counts come from the coordinator's worker record;
controller, coordinator and monitor restart counts come from their owning unit.
Kernel birth bounds remain conditional on the independently qualified boot and
wall-clock mapping. They are not inferred from a heartbeat timestamp.

These are locally tested collection components, not an armed collector or a
target execution receipt. The final before/after orchestration must still bind
requests and cleanup proof, recheck owners and source closure after producer
sampling, derive support-job ages from actual manager observations, and stamp
the capture after those reads. The original deadline, full legacy import and
runtime inventory, target registration and independent outer observer also
remain required before the bounded Linux qualification. No production source,
profile, unit, model or GPU allocation changed during this preparation.

Full CI for `2a71e55` passed all six jobs: 5,599 Python/native tests,
1,023 web tests and 107 clean browser E2E tests plus five skips. All 28
coverage floors, Rust/WASM checks, audits and the container check passed.
The older failed heads were retained as evidence and were not retried.

The collector-component joint check passed 478 local CPU/fault tests with two
workers, zero failures and zero skips, plus Ruff, formatting and Pyright. It
includes actual identity-helper composition with the kernel-join component,
strict private-data redaction, malformed-record refusal and current-lifetime
negative evidence retained before freshness filtering. Independent component
reviews and the final source bindings are preserved separately from execution
qualification. No runnable target collector or new GPU authority is implied.


The before/after capture workflow is now implemented locally. The standard-library
CLI reads a hash-pinned launch manifest, retains the original absolute deadline,
and uses separate eight-MiB metadata and twenty-four-MiB runtime read budgets.
The launch manifest remains an input from an independently qualified launcher;
it does not authorize itself. Actual custom module origins must match pinned
source paths. Private process command/environment bytes stay in memory.

Request schema v2 closes a pre-work provenance gap. The prospective plan now
commits the before request and receipt alongside the before capture and policy.
The after request must name those exact pins. The before phase stays plan-free,
so this ordering introduces no digest cycle. A bounded read-only facade checks
the existing lifecycle cleanup chain without constructing an evidence writer;
the planned workload set, dispatcher owner/definition, original Anchor, source
and time bounds must agree. Dispatcher properties use the same narrow stable
projection as the host adapter, preserving unknown static fields and ignore-error
semantics. Metadata opens are nonblocking before their regular-file check, so a
substituted FIFO refuses instead of waiting for a writer.

Support collection uses actual manager/process identity and coherent unit/job
observations. A previous no-job witness must come from the plan-committed before
receipt, its retained provenance and matching observation inventory. It cannot
be manufactured from a job's first-seen time. Final support-state contradictions
refuse, and later capture timing only extends existing age upper bounds. Producer
heartbeat/progress timestamps are never rewritten. A source check found that
systemd 255 emits a plain four-column jobs table and a scalar `Job` ID; the
collector and older controller/dummy adapters now consume those actual formats
with closed query vectors and strict joining, rather than assuming JSON output.

A compact, bounded provenance artifact is published first, then its receipt,
then the capture file as the commit marker. Existing outputs are never replaced;
interrupted partial artifacts remain evidence. Refused receipts can be preserved
only after output/source authority is admitted and while the original limits
permit it. Expired budgets or storage failures cannot be described as complete
failure closure. The outer finite launcher still owns terminal observation.

Local tests exercise real protected temporary files, no-clobber publication,
request/provenance binding, the existing cleanup semantics, real helper
composition and fault cases over explicit fake host observations. The published
before artifacts also pass the actual request validators. Positive CLI
source/bootstrap and after-authority setup remains simulated; these tests do not
qualify an actual Linux target, interpreter/import/native closure or outer
watchdog. An additive immutable legacy rendering, exact host registration and
independently reviewed prospective plan remain required before target execution.
No training unit, source, profile, model or GPU reservation changed.

All six CI jobs passed for committed head `4f4534f`: 5,888 Python/native tests,
1,023 web tests and 107 clean browser E2E tests plus five skips, all 28 coverage
floors, Rust/WASM, audits and the container check. That result does not qualify
this subsequent capture-workflow revision; its own full CI remains a separate gate.

The final joint local workflow check passed 1,060 CPU/fault tests with two
workers, zero failures and zero skips, plus Ruff, formatting and Pyright on
unchanged source bytes. Earlier 1,059-case evidence is retained separately;
the final run includes the nonblocking FIFO regression. These checks preserve
the remaining target/bootstrap/outer-launcher qualification limits above.


Read-only inspection of the actual systemd 255 host confirmed another textual
representation detail: its four services omit empty `ExecStartPre`, `ExecStop`
and `ExecStopPost` lines even with `--all`. Independent D-Bus reads showed empty
execution-command arrays for all twelve property/service combinations. The
collector now accepts each omission only with its exact registered versioned
rule, preserving the original observation and separate absence derivation.
Missing `ExecStart`, unknown properties/rules and required nonempty commands
still refuse. The controller and cleanup-proof projection received the matching
narrow `ExecStartPre` correction. The final joint check passed 666 CPU/fault
cases with zero failures/skips, clean static checks and independent reviews.
No production source or unit was patched.

An exact local legacy rendering of `f269d4a` is retained separately: 656 files,
61 explicit overlays, 183 namespace replacements, 600 unchanged base files and
52 unchanged protected runtime/native-build files. Its 1,316 passing CPU cases
and 31 native-dependent skips do not qualify native or target execution. A
single translated test needed an independently recorded AST-identical whitespace
reflow; bytecode generated by tests is excluded from the sealed source bundle.
The historical rendering retains the now-understood optional-property blocker;
a separately identified additive revision is required for the correction.

The proposed external installer design keeps the twelve dummy units and their
device sandboxes unchanged. A separately supervised read-only metadata process
needs NVML access to observe GPU owners. The reviewed design uses nested Linux
subreapers and pidfds to retain owned descendants across detached sessions; it
now has locally tested process primitives; the full executable composition and
Linux qualification remain outstanding. A before producer-completion proof
is precommitted, and the observer requires an after execution gate
in addition to the capture marker. The separate preflight allowance is 120
seconds including setup and cleanup; the dummy envelope remains 600 seconds
including all dummy setup. This is a 120-plus-600 allowance, not 600 overall.
The proposed after work/closure/gate bounds are 555/560/565, leaving observer
consumption until 570. Caller/session-death survival must be tested on the real
launch context; setsid alone is not that proof. No additional unit, scope,
cgroup, device-policy relaxation or GPU reservation is authorized by this design.

Committed `f269d4a` passed all six CI jobs: 6,099 Python/native tests, 1,023 web
and 107 clean browser E2E tests plus five skips, with all other gates passing.
The optional-property correction has its own subsequent CI gate.


The same host inspection exposed systemd's quoted rendering of a mount-unit
identifier containing literal `\xHH` escapes. The selected dependency-list
normalizer now decodes only the manager's outer double-quote/backslash form,
then validates and sorts unit identities. It does not decode unit-level hex
escapes into characters or path separators. Malformed/unknown quote and escape
forms, empty or duplicate words, invalid unit names and oversized input refuse.
The raw property observations remain unchanged. The source-backed encoding
addendum and actual safe fixture are retained; 196 affected local tests and
static checks passed. Historical f269/d6 bundles remain separately identified.


The external-process preparation now includes a bounded subreaper/pidfd library,
pure producer-completion checks and deterministic setup-artifact rendering.
The library keeps a forked capture child behind an exec gate until its direct
ownership, PID birth and pidfd are joined. It rechecks the supervising parent
before release and retains wait/reap evidence, adopted descendants and original
deadlines. Local fault tests use an explicit fake kernel; actual Linux launch
admission refuses. A runnable supervisor/operator/guardian composition, durable
failure publication, higher-parent takeover and actual caller/session survival
still need implementation and independent target qualification.

The completion contract admits only natural zero exits and reaping of both the
collector and its phase guardian, with empty owned families and exact source,
intent, process, artifact and clock bindings. It never certifies the future exit
of a still-running operator or supervisor. The plan commits `before_execution`,
and an after capture needs the fixed `r3-after.execution.json` gate in addition
to its commit marker. The after collector contract is reconstructed from admitted
inputs; its guardian program/resource contract remains the one committed by the
before proof. A future phase-specific guardian command requires another reviewed
contract, rather than accepting a new claim from the after receipt.

At revision `df09faf4`, the pure renderer checks the before completion proof and approved template,
changes only the absolute watchdog deadline in supplied unit definitions, and
produces the plan, original Anchor and fixed role authorizations. Both support
stages must have exact environment variants and complete pinned source bytes.
File aliases, ancestor collisions and overlap with preserved or protected inputs
refuse. This has no filesystem/systemd backend or CLI. Its final template digest
is a separate post-BEFORE admission: embedding that digest in the earlier intent
would create a hash cycle. The actual outer template-finalization authority is
an explicit remaining implementation gate; hashing a proposal is not approval.

Read-only evidence supports all twelve current process identities but does not
qualify historical wall-clock birth bounds for the eleven runtime producers.
The registration's missing qualification stays null. A separately named
prospective observation-window design is retained for review; it has narrower
historical claims, does not replace the current contract and cannot fill that
null. Neither local tests nor preserved source bundles admit a target experiment
or live R4 transition. The prior legacy bundle predates these new sources and
cannot inherit their qualification.

Committed `c4ef7173` passed all six CI jobs on its first attempt: 6,199
Python/native tests, 1,023 web tests and 107 clean browser E2E tests plus five
skips, all 28 coverage floors, Rust/WASM, audits and container checks. Its final
logs are independently preserved off-host. The external-process revision has a
separate final local check and latest-head CI gate.


The final joint check for this revision passed 1,121 CPU/fault tests with two
workers, zero failures/skips, unchanged source bytes, Ruff, formatting and
Pyright. Eight brand-audit tests and the complete 938-file audit also passed.
The earlier 1,119-case packet is retained separately; the final run includes the
actual imported-driver origin check and its positive/shadowed-origin cases.
Request-side, process-family and renderer reviews retain their explicit synthetic
and pure-function limits. These counts do not qualify a Linux target, hardware
execution, the missing outer installer or a live training transition.


Revision `8151937f` adds fixed executable supervisor, operator and
phase-guardian entrypoints, a standard-library installer facade, a separately
supervised site-enabled helper and a closed file/service backend. No operation
or callback is selected by arbitrary JSON code. The helper is needed because
qualification imports reach the training package; keeping those imports out of
the supervision processes preserves their small, standard-library bootstrap.
Collector/guardian limits remain separate from the helper's virtual-address
allowance. Parent soft and hard limits are explicit so a helper does not depend
on an undeclared privilege to raise an inherited hard ceiling. These are resource
contracts to qualify, not claims of observed target memory use or an aggregate
cgroup limit.

The original dummy-start record is retained before rendering or setup. The
supervisor must acknowledge that exact record, completed BEFORE proof and
current enclosing identities before helper writes. The preflight handoff,
cleanup and outside audit cutoffs are 118/119/120 seconds; the dummy operator,
supervisor cleanup and outside audit cutoffs are 598/599/600, also bounded by the
original aggregate 718/719/720 limits. Existing dispatcher, cleanup, capture,
observer and publisher boundaries remain unchanged. A process cannot certify
its own future exit: successful dummy audit remains distinct from an outside
caller's actual supervisor termination and reaping evidence.

The blueprint precommits the BEFORE launch/request/registration and replaces
only three fixed-path output placeholders after checking actual completed
producer evidence. The guardian program contract normalizes only the separately
bound authorization digest value, avoiding an intent/blueprint/guardian hash
cycle; actual argv, source, frame and intent bindings remain exact. The backend
uses exclusive file creation and records durable creation and completed-file
identities. Unknown, partial, replaced or modified residues remain incomplete.
Once a control-start intent exists, prearm cleanup cannot retire the independent
cleanup resources. These source implementations have not installed dummy units
or modified production.

Actual host applicability remains mandatory. The existing R4 interpreter's
observed ownership/mode differs from a generic root-readonly-file assumption.
An exact independently approved interpreter metadata and byte identity contract,
including the actual loaded `/proc/self/exe` image, must be checked without
chmod, copying or silently replacing the live runtime. This is not an immutable
interpreter claim. Helper startup also needs current small startup-file hashes,
qualified large-library cache identities, allowed site inventory, actual import
origins and the original source/session/resource proof. Cached large-library
identity is not a fresh content hash. No missing qualification is synthesized.

The combined current-tree check passed 1,255 CPU/fault cases with two workers,
zero failures/skips, stable source bytes and clean Ruff/format/Pyright. Eight
brand tests and the complete 948-file audit passed after adding one exact quoted
legacy-package token exception for the new runtime's import guard. The runtime
bytes and other audit prohibitions remain unchanged. Independent source review
covers producer/consumer composition, filesystem crash cases, interpreter and
site identity, parent liveness and dual-clock publication checks; actual target
bootstrap, Linux behavior, session survival, helper-site qualification and the
outside caller's terminal audit remain separate gates.

Committed `df09faf4` passed all six CI jobs on attempt one: 6,372 Python/native
tests, 1,023 web tests, 107 clean browser E2E tests plus five skips, all 28 coverage
floors, Rust/WASM, audits and container checks. Its final small archive was
independently read back. That result applies to the prior committed source; the
new executable revision requires its own latest-head CI result.

The outside-caller audit now supplies the fixed terminal launcher. Its approved
intent must already identify the actual caller lifetime; admission checks that
identity along with the existing interpreter, source, session, site, environment
and resource contracts. The caller launches only the fixed supervisor command,
retains its pidfd and records actual natural exit, reaping and an empty owned
family. Its result covers those completed descendant events. It does not claim
the caller's own future exit or manufacture target qualification.

The caller consumes the original START record and supervisor ACK, then joins
the supervisor's output bytes to its retained result, the operator's output
hash, complete BEFORE/AFTER producer proofs, final CPU audit, plan and source
pins. Historical producer evidence remains historical; current kernel terminal
observations supply the final closure. Supervisor work ends at the original
119/599 cutoffs; caller closure and publication end at 120/600, bounded by the
original aggregate 720 seconds. The separate child alarm is an aggregate
backstop, not an extension of those checked deadlines. Raw START timestamps
remain raw; stored Anchor-derived nanoseconds use the existing declared floor
projection, without a round-trip equality assumption or added tolerance.

Actual Linux/session/bootstrap and resource qualification remains outstanding.
The readiness assessment sequences a separately reviewed harmless CPU witness
routine, exact additive staging and independent real receipt verification
before the normal runtime can consume those prerequisites. The twelve-unit
dummy protocol and production R3 sources, profile and services stay unchanged.
The separately named learner-append observation-window proposal is approved
only for local implementation preparation and independent review. Its initial
slice is a bounded append-proof verifier and registered file-range reads; old
historical-birth gates stay unconditional, and a new success-producing route
requires explicit end-to-end contract selection and consumer review.

Committed `8151937f` passed all six CI jobs on attempt one: 6,506 Python/native
tests, 1,023 web tests, 107 clean browser E2E tests plus five skips, all 28 coverage
floors, Rust/WASM, audits and container checks. These results apply to that
committed source; the caller audit has its own affected-check and CI evidence.
Its final affected check passed 269 cases with two workers, zero failures/skips,
stable source bytes and clean Ruff, formatting and Pyright. The repository brand
audit passed across 950 files. The earlier 32-case author packet and its 35-case
admission-coverage revision remain distinct evidence; their counts are not added
to the final run. Unchanged collector suites were not repeated.

The first learner-window component now implements conditional append evidence
and narrow reads of the already registered `metrics.jsonl` slot. Independently
admitted writer/file/source, recipe and original time-window bindings are
required. The reader retains an observed EOF and private prefix bytes, then
reads and rereads fixed contiguous ranges with actual file/name/descriptor
metadata and bounded clocks. Bytes remain charged even when a later check
refuses. The public proof contains safe numeric facts and hashes, not raw log
content or private paths.

The pure verifier checks every complete captured record for failures and
nonfinite or contradictory data before choosing two loss records wholly after
the causal fence. The first may have been prepared before that fence; its
synchronous append precedes the subsequent serial learner work represented by
the second. A parseable initial pre-window fragment is conservatively screened
for explicit negatives without claiming a proven record boundary. Unrecoverable
fragments, observed replacement, rewrite or truncation, contradictory owners,
sources or clocks refuse. Observed growth beyond the final fixed EOF is
incomplete; the reader does not chase it or renew the deadline.

This result proves only conditional serial work within the observed window.
It neither establishes the writer's qualification nor proves exhaustive earlier
history, current full preservation or playing strength. Unobserved filesystem
changes between samples remain covered by the explicitly qualified append-only
writer/access premise. Existing registration, collector, CLI and historical
birth gates remain unchanged. A full observed-window capture still needs a
separate explicit contract discriminator, owner/static/support/recipe checks,
transitive proof retention and independent consumer review.

The `7f279e18` CI run passed five jobs, but its Python/native job stopped after
5,424 passes and one fake-kernel caller test reached the strict ambient SIGCHLD
guard. The exact nondefault handler and its earlier setter were not identified.
The test now supplies a module-local signal facade; it changes neither the real
process disposition nor the shared signal module. Explicit ignored/callable
handler cases retain the production refusal. That failed head was not retried;
the corrected head requires its own full CI result.

The final affected run passed 184 cases with two workers, zero failures/skips,
stable source bytes and clean Ruff, formatting and Pyright. It includes actual
private-file reader-to-proof composition and the signal-fixture correction.
The complete 954-file brand audit also passed. Separate author runs and earlier
proof revisions remain preserved rather than combined into a larger count.
No target qualification, live source/profile/service change, GPU work or R4
activation accompanied this local preparation.

The observed-window registration now has a separate strict parser. It checks
externally approved raw bytes, exact format and contract, common scope/source/
environment/boot/cache invariants, the private writer evidence, the complete
publication-writer inventory and the current-kernel namespace/HZ/credential
expectations. Raw command and native byte digests join the existing argv and
native policy pins. Import/access/qualification digests remain externally
established premises, not facts proved by equality. Its frozen private result
reports schema validation only; it grants no writer, runtime or execution
qualification and enables no collector or CLI success route.

Shared validation stays in the existing identity module. The historical
constructor still requires its exact header and mandatory birth qualification
before any IO. New registrations reject old birth fields rather than filling
them with an observation time. The new fixed credential read accepts only a
previously admitted process token, reads status through its retained proc
directory, and checks PID/start/parent/cgroup around the read. It reports the
four literal real/effective/saved/filesystem UIDs without exposing raw status.
The reader does not infer credentials from a username or proc-directory owner;
the strict R3 registration separately requires the qualified UID consistency.

The final affected check passed 258 cases with two workers, zero failures/skips,
stable source bytes and clean Ruff, formatting and Pyright. Private-proc
reader/schema composition confirms both honest unequal-UID measurement and
strict schema refusal. The complete 959-file brand audit passed. Actual renewal
collection, authenticated historical-B reconstruction, the end-to-end contract
discriminator and safe producer/consumer attestation remain separate work.
No target staging, service, source, profile, GPU or R4 activation occurred.

Committed `bf798cee` passed all six CI jobs on attempt one: 6,647 Python/native
tests, 1,023 web tests, 107 clean browser E2E tests plus five skips, all 28 coverage
floors, Rust/WASM, audits and container checks. The failed `7f279e18` run remains
preserved as a failure; its test isolation correction was validated on this new
head. This first registration revision requires its own exact-head CI result.

Candidate 596131 completed its 320-game/160-pair allocation at a 62.8849638%
weighted score and a 90% anytime interval of 49.8425431–76.3893410%. The frozen
decision is `reject_max_pairs`, with `conclusive=false`; champion 572377 remains
retained. All 320 recorded native winner histories and nine allocation boundaries
were verified with the exact unchanged R3 helpers in 9.06 CPU seconds, with no
CUDA initialization, search, inference or checkpoint loading. Checkpoint
dependencies are retained through catalog-bound, same-inode hardlinks without
copying or rehashing their payloads. The arena has advanced to candidate 601991;
its asynchronous game count is not a complete strength decision.

The `d69b559f` CI run completed with five successful jobs and one web-quality
failure. One GameScreen test observed the thinking label before the asynchronous
request import/call had arrived, then read an empty mock-call list. The test now
awaits the actual request-call barrier, as neighboring tests already do. All 55
GameScreen tests, targeted ESLint and project TypeScript checks pass. Production
behavior, search budgets, timeouts and assertions are unchanged; the failed head
was not retried. The corrected head requires its own CI result.

That failed run still completed 6,744 Python/native tests and all 28 coverage
floors. Its successful browser job had 106 clean passes, five skips and one
Firefox history-review test passing on its automatic retry. The recorded first
failure and unconfirmed cause remain preserved; it is not reported as a clean
107-test run, and no browser assertion or timeout was changed from that evidence.
