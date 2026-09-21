# Training efficiency audit — September 12, 2026 UTC

Training is operating, but maximum Elo per wall-clock hour is not established. This audit found several concrete, inexpensive improvements in the deployed implementation. The strongest measured outcome-preserving opportunity is to stop arena games once the existing rules engine proves their winner. The most striking control-loop waste is repeatedly hashing two unchanged model checkpoints.

This was a read-only production audit. No training code, profiles, services, checkpoints, or replay were changed. Evidence comes from live telemetry, source inspection, small CPU-only probes, and offline replay of actual evaluation games. No additional H100 workloads competed with training. Local HEAD is `9d8553d`; the active application release is `7d874bba9206e07c4b17156e2dba169758d35720`, implementing `916cbfd`.

**Current progress and limits**

The representative interval is September 11 **21:50:23 UTC** through September 12 **00:50:23 UTC**: 2,161 observations at five-second spacing, without gaps. The final independent health check at approximately 00:56 UTC found learner step **205,876**, the original learner/actor process identities, and healthy hardware. Actor GPU 7 was deliberately paused for evaluation.

| Measurement | Evidence |
| --- | --- |
| Training progress during the three-hour interval | 201,815 → 205,731; **1,305 updates/hour** |
| Unique committed positions | Approximately **446,019/hour**, from learner counter endpoints |
| Explicit UTD sleep | **41.55%** of the three-hour interval |
| Median logged device step | **0.406 seconds** |
| Average GPU busy samples | Learner **12.26%**; actor GPUs 1–6 **43.48–49.68%**; shared GPU 7 **33.18%** |
| Nonfinite loss/gradient counters | Zero in the sampled training metrics |
| Process continuity | No process changes during the three-hour interval |
| Earlier arena errors | Two recovered GPU-ready acknowledgement timeouts after deployment; the current restart counter is not a lifetime total |
| Latest completed balanced gate cycle | Candidate 185,554 versus champion 156,256: **+104.37 Elo**, interval **−9.29 to +246.34**, 96 pairs; inconclusive |
| Comparable 1,024-simulation strength measurement | No completed connected six-cell measurement for the selected contract |

GPU busy percentages are not achieved FLOP utilization. The fresh-data rate is an observation, not an A/B comparison against today's earlier 545k/hour sample. Model identities, board allocation, game phases, and arena sharing vary.

The learner is about 20,000 steps ahead of the checkpoint currently under evaluation. The champion has not changed since September 10 at 23:12 UTC. Training may be improving, and the current gate estimate is encouraging, but the available evidence cannot quantify the latest release's Elo/hour. Identical current and epoch-anchor champions imply no realized champion-frontier change during that epoch; they do not imply zero improvement in the unevaluated learner.

**1. Stop mathematically decided arena games — first implementation recommendation**

Self-play already calls the exact extremal-completion proof. Arena plays to a full board. The proof guarantees the winner under every remaining continuation, so this is suitable for winner-only evaluation without weakening search or introducing heuristic resignation.

An offline native replay of the current saved arena checked **192 completed games**. All 192 proof winners matched the recorded final winner, across all six modes and all tested handicap severities. It identified:

- **16,477 of 52,681 searched moves** that could have been skipped: **31.28%**.
- **7,195,904 of 21,205,248 nominal simulations**, accounting for the actual turn's handicap PDA: **33.93%**.
- A median of **87.5 saved moves/game**. Handicap-9 nominal savings were **41.28–42.77%**.
- A CPU proof sweep taking **0.394 seconds** for 240 saved games and 3,311 batched calls. The additional 48 games were unfinished and contributed no claimed savings.

That is approximately **1.51× arena search-work headroom**, not a measured wall-time or Elo multiplier. It also removes expensive tails when few games remain active. Faster evaluation can advance the champion sooner and release the shared GPU sooner.

Implementation: reuse [native complete_clinches](../crates/deltrel-py/src/lib.rs#L891) in the [arena move loop](../deltreltrain/arena.py#L2255), backed by [the existing proof](../crates/deltrel-engine/src/scoring.rs#L301). Preserve actual move histories and reconstruct proof finalization during resume. Explicitly guard pending pie swaps and keep statistical stopping on completed balanced cycles. Do not substitute the optimal-play endgame solver: unlike a proven clinch, optimal play can change the winner relative to the actual agents' continuations.

Validation: exact outcome and pair-accounting tests, interruption/resume and pie tests, followed by a fixed-model H100 comparison over identical games. Charge proof overhead and complete arena elapsed time.

**2. Cache verification of unchanged learner checkpoints — smallest substantial code cleanup**

Every outer learner loop calls plateau control. Under the live `reduce_lr_keep_weights` setting, [the plateau decision](../deltreltrain/learner.py#L4763) loads both champion and candidate manifests even when neither has changed. [Manifest loading](../deltreltrain/checkpoint.py#L1732) hashes the entire referenced checkpoint.

Both live checkpoints are about **212.7 MB**. Three warm, read-only server repeats measured approximately **0.192 seconds each**, or **0.384 seconds and 425.4 MB of cached file reads per plateau pass**. The three-hour logs contain at least 7,159 corresponding loop passes. Multiplying the isolated measurement by those events suggests about **2,749 seconds, or 25% of elapsed time**, and about **3 TB of cached bytes processed**. This is a cost extrapolation, not direct profiling of the production process, and these bytes should not be called physical disk I/O.

Keep verification, but reuse verified immutable manifests while the pointer, manifest, and checkpoint file identities remain unchanged. Reverify on change or replacement; retain explicit verification at actual load/recovery boundaries. Actors already have a related changed-pointer cache at [actor.py](../deltreltrain/actor.py#L1983).

This is an excellent engineering fix, but removing it alone need not increase training steps: the current 1.5 UTD cap still limits updates to available new data. It removes needless latency and increases capacity available for a subsequent UTD experiment.

**3. Stop graph-cache churn — smallest configuration experiment**

The actor graph cache allows **16 shapes per model**. The enabled small-batch bucket scheme can require all 16 slots for just one board size. The same model sometimes concurrently serves multiple board sizes.

Live counters recorded **7,456 captures** in three hours. GPU 3 alone had 3,101 captures and 3,029 evictions; GPU 5 had 1,707 captures and 1,659 evictions. GPU 3 recaptured in approximately 39% of five-second intervals. Warmup rows alone were 1.72% of its useful-plus-padding row volume, excluding capture and validation costs.

Test **16 → 32 entries**, preserving existing byte limits. Byte limits may still evict large graphs, so this is not a promise to eliminate all churn. Benchmark concurrent ring-6/ring-8/ring-10 traces for the same model, including model switches; a single-ring benchmark would miss the defect. Check memory headroom, exact outputs, captures, and useful rows/second.

**4. Reject history work that will expire before publication**

The history selector checks whether a model is eligible now, without enough allowance for game duration. One ring-10 task began with model step **83,011**, learner step **202,962**, and only **49 learner updates** before its 120k-step age limit. It ran for **78.8 minutes**, performing **3,328,618 simulations**.

Of its 22,135 published positions, five-second telemetry brackets place publication callbacks for **21,623 after expiry** and 512 before. Callbacks follow durable commits, so the exact number committed after expiry still needs ledger verification. The task-completion metric labels the entire task ineligible. The implicated rows represent about **1.75%** of completed-task volume; that is an indicative waste estimate, not a proven discard count.

At [history selection](../deltreltrain/actor.py#L1813), require age headroom based on measured task duration and learner progress; stop refilling an expiring task and select younger history while preserving the role mix. Do not increase the global replay-age limit to hide the waste. Validate publication-time eligibility, historical diversity, and policy provenance.

**5. Give arena the existing fast inference path and keep its batches populated**

Actors enable CUDA graphs through per-actor overrides; [arena evaluator construction](../deltreltrain/promotion.py#L170) inherits global `cuda_graphs: false`. Arena also retains first-visit width 1 while actors use 8.

One current 300.8-second arena slice had **zero graph replays**, 17,839 physical neural calls, **31% padding**, and about 1,421 logical evaluator rows/second. Its preparation and return counters are large, although overlapping counters must not be added as a clean wall-time decomposition.

The four-pair wave also splits handicap severities into groups of just two games. Completed rows are removed without replacement. A tail slice advanced 64 moves in 300 seconds, compared with 1,354 moves in a fresh slice. Different PDA budgets confound that comparison; **21× is not an implementation speedup estimate**.

First screen arena-specific cached graph execution and first-visit batching with unchanged model outputs/search semantics. Then add bounded rolling lookahead across pair/cycle boundaries, retaining fixed seeds and statistical decisions only on completed balanced cycles. This is a larger change than the first four, with potentially greater evaluation throughput impact.

**6. Remove repeated replay scans and redundant trusted-batch work**

[Window validation](../deltreltrain/learner.py#L2743) sums eligible replay and checks every selected path, often at least twice per outer loop. A current window contains approximately **23,641 spans**. Warm read-only probes measured **0.087 seconds per eligibility/path sweep**. Bound these checks by revision, relevant step boundaries, and a validation cadence; maintain GC pins and fail correctly on missing data.

The freshness check [constructs a successor selection](../deltreltrain/learner.py#L2857), discards it, and then selects again when opening the new window. Warm selection cost was **0.60–0.62 seconds**. The pinning selection runs under a write transaction. Reuse a safely validated successor or shorten transactional work. A first 4.66-second probe included cold topology construction and is **not** the production steady-state cost.

There are also three unused Boolean-index tensors in [loss validation](../deltreltrain/losses.py#L412), constructed even for trusted batches. Move them inside the validation branch. Native first-visit submission also validates/submits independent sessions serially at [deltrel-py](../crates/deltrel-py/src/lib.rs#L2126), unlike the legacy parallel path. An alternating CPU probe found 38.7% more submit time for width 8, but the native time is small relative to neural inference; this is a lower-priority cleanup, not a major fleet multiplier.

**7. Establish a usable Elo/hour feedback loop, then tune learning reuse**

The current measurement scheduler postpones historical evaluation while candidates are ready, and prioritizes old predecessor links. The active 1,024-simulation crossplay is still an early champion against bootstrap, with zero completed pairs. Give a small, bounded share of evaluation time to comparable measurements from the current epoch anchor; prioritize evidence relevant to the current frontier. Keep promotion and measurement search budgets distinct.

Once exact overhead fixes are in place, the most promising small learning experiment is **UTD 1.5 versus 2.0**, with a further arm only if capacity and results justify it. GPU 0 has substantial spare time. However, 1.5 is not demonstrated to be too low merely because it waits.

Current policy diagnostics show approximately **175 effective policy rows per 512-row batch**, while full-search rows contribute about **79% of policy weight**. Investigate weight-aware sampling with correct probability compensation as a separate experiment, preserving mode balance and measuring duplicate/deep-target reuse. This is a statistical-efficiency opportunity, not evidence of 2.9× stronger learning.

Use the same starting checkpoint and equal whole-machine wall budgets, preserve publication cadence per fresh sample, explicitly control LR/EMA clocks, and evaluate on the same six-cell objective. Report confidence intervals and charge evaluation, waiting, and restarts. Increased optimizer steps or GPU busy time alone is not success.

**Recommended implementation order**

Implement exact arena clinches, learner manifest-verification caching, and the graph-cache experiment first. Add the history eligibility margin next. Then improve arena batching/inference and replay housekeeping. Establish comparable strength measurements before selecting a higher UTD or changing target sampling.

Earlier improvements—live policy publication, shared scheduling, fast-27 allocation, first-visit width 8 for actors, four loader workers, lazy native state export, and shared geometry—are already deployed and are not new recommendations. The local-message prototypes that failed numerical gates remain unsuitable for immediate activation. No evidence supports calling the current learner a 10× easy kernel optimization opportunity.

The compact [evidence record](training-efficiency-audit-evidence-20260912.json) retains measured summaries, source identities, and offline proofs. Full temporary observations are under `/tmp/deltrel-audit-20260912` on this workstation.
