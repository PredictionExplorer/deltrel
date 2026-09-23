# Plan: materially higher Elo per wall-clock hour

**Status:** proposed implementation plan; September 23, 2026. No production changes or performance experiments were performed while preparing this plan.

**Working assumptions:** optimize the existing eight-H100 server first; preserve the current playing-strength objective. Hardware changes and objective changes are separate optional decisions. This plan proposes future work, not activation of any treatment.

## 1. Decision and success criteria

Prioritize the learning feedback loop: trustworthy strength measurement, faster evaluation, more useful self-play, and better learning from each position. Learner kernel speed is a secondary opportunity under the current data limit.

The primary outcome is independently measured improvement in champion strength per **total provisioned wall-clock hour**, with pair-valid uncertainty. Keep the existing conservative lower-bound metric alongside the point estimate. Count initialization, compilation, evaluation, waiting, checkpointing, migration and recovery until resource release; do not stop the clock at the final optimizer update. Report provisioned GPU-hours and cost separately, particularly for hardware changes.

The current ring-10 objective is 45% Classic pie, 45% Double pie, 5% Classic handicap and 5% Double handicap. Handicap severity levels remain equally weighted within each category. Preserve the training ring mixture, game rules, swap behavior and evaluation search contract unless a separately registered experiment changes them.

**Program ambition:** pursue a substantial compound improvement, with 2× Elo/hour as a stretch target rather than a forecast. **Adoption standard:** retain the repository's three-seed confirmation requirement—positive treatment lower bound minus control upper bound in each seed, plus at least 20% median point Elo/hour improvement. A component speedup is not evidence that this outcome has been achieved.

Use three evidence classes:

1. **Correctness-preserving engineering:** unchanged game/search/training semantics, validated by deterministic differential tests and representative performance measurements.
2. **Resource and scheduling treatments:** altered allocation of hardware or evaluation time; judge the full system, including lost self-play capacity and feedback delay.
3. **Learning treatments:** changed reuse, sampling, search targets or model behavior; require controlled strength experiments.

Do not multiply isolated speedup estimates into an Elo forecast.

## 2. Evidence baseline and its limits

The September 23 live audit used deployed source `7a077107b1ebba55494b09f80689989e773b9dde`, with `profile-auxiliary-predictions-20260918.yaml`. The server still uses `edgeconnect-*` paths and service names; the local repository is renamed `deltrel`. The planning checkout was `01345afcbd637d56c0ff4053ae89ab75d3912010`. Freeze and reconcile these identities before implementation or benchmarking.

| Observation | Implication |
| --- | --- |
| 1,326 learner updates/hour over approximately 24 hours; 1,392/hour in the latest hour | Stable progress, rather than a stalled learner |
| Approximately 126 new replay positions/second over 24 hours | Current supply to the learner |
| Batch size 512; consumed-example/new-position ratio 1.5 | Predicted limit: `126 × 1.5 × 3600 / 512 ≈ 1,329 updates/hour`, almost exactly the observed rate |
| Approximately 60% explicit fresh-data-credit sleep; 14.5% measured device update time | Faster learner kernels alone will not materially raise updates at the unchanged data limit; these are timing categories, not a FLOP-efficiency measurement |
| Approximately 2,375 self-play games/hour in the latest hour | Useful baseline, to be broken down by mode, ring, target quality and freshness |
| Independent 1,024-simulation strength results unfinished; official Elo/hour null | The optimization target currently lacks a functioning measurement loop |
| A recent candidate waited 10.79 hours before evaluation; its predecessor took about 14.83 elapsed hours for 320 games | Validation latency materially delays model feedback |
| Arena uses five-minute sessions and five-minute cooldowns; approximately 44% of GPU 7's wall time was active evaluation | Resource sharing is a deliberate tradeoff requiring a system-level comparison |
| Arena graphs disabled, first-visit batching width 1; one 300-second slice had 36.2% padding and zero graph replays | Concrete inference/batching candidates; the single slice does not establish a speedup |
| Two recovered 930-second GPU handoff timeouts in the preceding day | Diagnose the blocked transition; increasing the timeout is not a fix |
| Backup source cutoff approximately 36 minutes old, with a new backup running | Recovery freshness needs explicit measurement and an operational target |

Already deployed: exact arena clinches, four actor cohorts per GPU, actor rolling slots, live policy publication, actor first-visit batching, graph-cache expansion, persistent scheduling credits, model-verification caching and several replay/CPU optimizations. Do not repackage old audit recommendations as new work. The older roadmap's model and experiment baseline is not the current auxiliary-head production baseline.

## 3. Phase 0 — establish an honest measurement and profiling system

### 3.1 Restore independent strength measurement

The current scheduler runs historical strength work only when no promotion candidate is ready; that work also stops when a candidate appears. Enabling historical evaluation is insufficient: it is already enabled and can starve indefinitely.

Introduce durable promotion and independent-measurement queues with explicit service accounting. Initially pilot a reservation of **20% of the existing evaluation budget** for measurement, preserving the total self-play/evaluation allocation. Account for actual held GPU seconds, not nominal session counts. Preserve service debt, priorities and interrupted jobs across restart. Guarantee a bounded wait to the next measurement slice even under continuous candidate arrivals.

The reservation is a starting experiment, not a promise that all measurements finish daily. Use a complete four-category, four-pair-per-category pilot to estimate actual 1,024-simulation pair costs. Its 32 games validate operation and cost, not statistical precision. Set completion deadlines and capacity from the measured pair-cost distribution and desired uncertainty.

Create an explicit new measurement epoch with an immutable current-champion anchor, without rewriting older epochs or deleting their evidence. Baseline relative strength is zero by definition; a rate still requires subsequent measurements. Optionally bridge to the old epoch using fresh evidence. Avoid spending all available evaluation time reconstructing ancient links before measuring current progress. If an anchor becomes too weak for an informative finite estimate, register a bridge anchor and retain the link's uncertainty.

At predeclared wall-clock checkpoints—initially daily, subject to the cost pilot—pin the deployed champion's identity and evaluate it on fresh holdout opening/seed streams. Keep promotion evidence separate from independent measurement. Add occasional fixed-schedule learner snapshots as a diagnostic to distinguish learning progress from promotion lag; they do not replace the primary champion metric.

Freeze the estimator: apply the Elo transform to the fixed weighted score as the current implementation does; do not silently substitute a weighted average of category Elo values. Use the existing paired uncertainty machinery with explicit error allocation across categories, links, times and seeds. Nonlinear finite-sample Elo estimates are not claimed to be unbiased. Repeatedly selecting promising checkpoints on their evaluation outcomes is not a holdout protocol.

**Exit gate:** a complete current four-category measurement between distinct immutable models, honest point/interval reporting, immutable timestamps, and a starvation test proving service under endless promotion arrivals. Missing evidence remains visibly unavailable; old-lineage headline numbers cannot fill the gap.

### 3.2 Make telemetry authoritative

Define one active-worker inventory keyed by run, generation, role, PID and model identity. Retired actor-lane records currently remain in some legacy monitor summaries; exclude them from active aggregates. Preserve historical data under an explicit historical view.

Track disjoint critical-path spans and overlapping-work counters separately:

- Actor search, feature packing, broker queueing, model refresh, H2D, kernels, D2H, postprocessing, native submission and replay publication.
- Learner UTD wait, replay wait, replay selection/validation, loading, copies, optimizer work, diagnostics and checkpoint/control overhead.
- Evaluation queue wait, GPU handoff, model load/compile/warmup, state restore, useful search, durable save and cooldown.
- Backup capture, NFS metadata, object reads/copies/hashes, catalog construction, verification, publication and retries.

Report useful eligible positions, effective full-policy supervision, finalized labels, unique games, replay reuse, and age at generation/publication/consumption. Raw rows, games and GPU busy percentages alone are insufficient.

Use short representative CPU/GPU traces, not continuous heavy profiling. Nsight tools were not found on the server; qualify them in a benchmark environment before use. Measure profiler overhead with an unprofiled control. Include warm/cold paths, small and large boards, expensive handicap games, model switches and backup overlap. NVIDIA recommends realistic workloads and focused traces; PyTorch documents additional tracing overhead.[1][2][3]

**Exit gate:** reproducible baseline spanning a full candidate lifecycle and backup cycle, plus a bottleneck report that attributes time without adding overlapping counters.

## 4. Phase 1 — make evaluation fast, reliable and fresh

### 4.1 Fix the actual GPU handoff fault

Add phase durations and blocked-owner information to the existing cooperative pause protocol: producer safe points, outstanding broker work, model refresh/compile, CUDA completion and acknowledgement tokens. Reproduce the observed stall before changing cancellation or retry behavior.

Readiness must still mean every producer is parked, outstanding inference is settled, CUDA work is complete, and the acknowledgement belongs to the current lease. Test stale acknowledgements, owner exit, stuck producers, refresh waits, cancellation before adoption and rapid reacquisition. A restarted evaluator must retain every committed pair and allocation.

### 4.2 Remove measured per-session overhead

Every session currently loads both models, rebuilds evaluators, restores game histories and later releases evaluator state and CUDA caches. The deadline includes setup. Instrument each phase, then retain a bounded evaluator/model cache across leases if that saves material wall time.

Key caches by immutable model, precision, feature/rules contract and execution settings. Enforce combined actor/evaluator memory budgets and lease ownership; retaining a model must never permit inference outside its lease. Consider keeping an unfinished native search in memory across cooperative pauses, while retaining durable move-level crash recovery.

The learner currently reserves almost the entire GPU 0 memory budget. Its idle intervals do not establish that another workload can safely share that GPU.

### 4.3 Improve arena inference and occupancy

Use the existing complete-workload arena benchmark to test, separately:

1. Arena-specific CUDA graphs and shape buckets, including capture cost and sparse tails.
2. First-visit batching with unchanged legal requests, search budgets and seed streams.
3. Refill from unfinished games/pairs **inside the already committed allocation**.
4. Broker batching limits selected from real request distributions and padding costs.

Do not speculate across adaptive allocation boundaries without a separate precommitted statistical design. Faster-finishing results must not select the next allocation or stopping decision. Preserve role reversal, handicap PDA, pie behavior and exact-clinch outcomes.

Measure complete fixed-model matchups, including setup, checkpointing, expensive tails and all required pairs—not just neural rows/second. Existing admission and numerical rules remain in force; failed comparisons cannot be relabeled successful.

### 4.4 Compare resource allocations only after obvious overhead is removed

Compare optimized GPU sharing with **one learner + six actor GPUs + one dedicated evaluator GPU** on the same machine. Dedicated evaluation could eliminate repeated handoffs and substantially increase evaluation service, but sacrifices some self-play supply. Measure that tradeoff instead of assuming it helps.

Keep the current rule that an evaluation with durable evidence finishes before being replaced. The scheduler already supersedes obsolete unstarted candidates; preserve that behavior. After faster evaluation is measured, set promotion publication cadence relative to observed service capacity, while keeping self-play snapshot cadence separate. Avoid an unstable queue whose arrival rate exceeds completion rate.

**Proposed engineering targets:** halve candidate publication-to-decision latency and seek 2× complete-pair throughput for representative workloads. These are targets, not forecasts. Adoption requires intact evidence and a favorable full-system result, not merely more evaluator GPU time.

## 5. Phase 2 — increase useful self-play production

### 5.1 Prevent avoidable stale work

History selection currently checks age at selection but can choose a model with insufficient headroom to finish its games. A live cohort began with only 298 learner steps of eligibility remaining and later exceeded the 120,000-step lag boundary.

Add a conservative horizon based on recent game duration and learner progress; stop refilling an aging cohort and switch to eligible history while preserving the intended model-role distribution. Measure exact publication-time and consumption-time eligibility. The observed completed-task waste was small in a bounded sample, so treat this as a targeted correctness/efficiency fix rather than the major speedup.

### 5.2 Optimize the inference-to-search boundary

The current broker holds device ownership through synchronous result transfer and Python postprocessing. Prototype owned, typed result buffers across the Python/Rust boundary, removing unnecessary scalar boxing, copies and repeated reductions. Then test safe overlap of CPU search/packing, GPU execution and result consumption.

Require explicit buffer lifetimes, bounded backpressure, cancellation ownership and model pinning. Preserve results and utility/PDA semantics, including detailed predictions. A buffer optimization must not expose storage that another request can overwrite.

Sweep cohort concurrency, games per cohort, inference batch limits, NUMA/thread placement and broker wait budgets one factor at a time, then confirm the selected combination. Four cohorts and rolling slots already exist. More concurrency is useful only while it improves retained target production without cache churn, memory pressure or worse freshness.

Rank kernel work by measured fleet critical-path cost. Candidate exact optimizations include remaining local-aggregation/layout work, inference-only projection fusion and typed return paths. Preserve training optimizer parameter identities: fusing Muon parameters would change its update rule. Keep previously rejected numerical shortcuts quarantined.

**Proposed engineering target:** at least 25% more eligible, correctly distributed fresh positions per wall hour for a selected runtime treatment, without sacrificing supervision quality. Confirm full-game and whole-fleet behavior; a single shape or isolated kernel result is insufficient.

## 6. Phase 3 — improve learning per unit of self-play compute

These are experiments, not automatic configuration changes. Prioritize a small number of strong hypotheses.

| Order | Experiment | Mechanism and controls |
| --- | --- | --- |
| 1 | Current reuse ratio 1.5 versus 2.0 | Use existing learner headroom. Freeze the clock contract first. Preserve starting checkpoint, replay boundary, search and objective; measure overfitting, calibration and independent strength. Advance beyond 2.0 only if evidence supports it. |
| 2 | Separately chosen learning-rate/EMA treatment | After the reuse experiment's clocks are controlled, test a distinct LR or EMA hypothesis. Do not attribute an unintended change in decay, averaging age or publication timing to reuse alone. |
| 3 | Replay quality and freshness | Study duplicated logical positions, old-model share, finalized versus policy-only supervision and effective full-search policy weight. Preserve ring/mode quotas and document any sampling correction or intended objective change. |
| 4 | Search quality per second | Revisit fast/full-search allocation, uncertainty-directed search and subtree reuse with an explicit additional-budget policy. Faster generation must retain or improve eventual strength. Preserve the current tested allocation as control. |
| 5 | Cheaper actors or model architecture | Screen a moderate distilled actor against the full model at equal wall-time search strength; charge distillation and teacher refresh. Only then consider architecture or precision changes. |

For the first reuse experiment, keep publication frequency fixed per newly generated position. Raising reuse from 1.5 to 2.0 therefore changes the candidate interval from 7.5 million to 10 million consumed examples, and the self-play snapshot interval from 1.5 million to 2 million. Preserve the starting effective learning rate and LR/EMA progression in fresh-data units through a versioned prospective segment; validate this clock conversion before the trial. A treatment that instead keeps update-based clocks is a different, explicitly coupled experiment. Start a prospective credit segment; do not award catch-up credit for historical replay. Its arithmetic allowance is one-third more optimizer updates at the same data rate, not one-third more Elo.

Preserve the complete optimizer, scheduler, EMA and replay state in both continuation arms. Existing champion warm-start helpers can reset optimizer/scheduler state; using that reset only for treatment would invalidate the comparison. If both arms deliberately reset, identify the experiment as a warm-start study.

EMA deserves its own diagnostic: decay 0.9999 corresponds to approximately 6,931 updates, or 5.2 hours at the current rate. Compare raw and current EMA weights from one recovery checkpoint before testing a shorter averaging horizon. Preserve all other clocks and allow enough settling time. Similarly, auxiliary-loss changes need a current hypothesis and per-head diagnostics rather than wholesale repetition of old-lineage null experiments.

Fix the calibration split before relying on it: the current frozen-replay tool splits individual row identifiers. Split by immutable game identity, deduplicate logical positions and revisions, and group related openings where appropriate. Use game-level resampling for correlated positions. Better held-out loss is a diagnostic, not the final adoption gate.

Keep the current auxiliary outputs as the baseline. Any change to their loss weights, supervision, model size, search algorithm or precision is separately versioned and evaluated. Do not combine a search-quality change with a runtime optimization and attribute the result to the latter.

Large research directions—distilled actors, search-light action-value supervision and alternative batched search—may offer the largest gains, but have substantially greater risk. KataGo supplies relevant evidence that search/data allocation and auxiliary learning can matter; its reported gains are not forecasts for this system, and this project already implements several related ideas.[4]

Two existing negative results remain binding evidence: quartering deep-search budgets failed a preliminary quality gate, and source-class/project-first local-message prototypes failed numerical gates. Do not deploy them as shortcuts. A materially different new experiment needs fresh admission evidence.

Secondary exact capacity work includes safely reusing a pinned successor replay window, avoiding repeated unchanged shard-path scans, and deferring native leaf-edge allocation. Profile before implementing. A fused relational-attention backward path is a later learner-memory project with strict BF16/gradient parity; it might enable a future resource-sharing design, but does not remove the present data limit.

## 7. Code quality is part of every work package

The repository already has substantial Python/native, Rust, browser, mutation, coverage and hardware checks. Build on them.

- Extract touched responsibilities from the large learner, promotion and orchestration modules into typed components: evaluation scheduling, GPU lease management, replay-credit accounting, inference execution and artifact verification. Keep thin orchestration entry points and explicit side-effect boundaries.
- Represent scheduling and lease transitions as explicit states with validated events. Separate pure policy decisions from process/filesystem operations so failure paths can be tested deterministically.
- Reuse inference execution components across actors and arena while retaining separate, frozen configuration. Avoid copying optimized actor code into a second diverging arena implementation.
- Give metrics and durable events versioned schemas, unambiguous units, process identity and ownership. Publish current state and historical data separately.
- Preserve backward readers and migration contracts for checkpoints, replay, allocation ledgers and resume data. Refactor and change behavior in separate reviewable commits.
- Add invariant, property-based and crash-injection tests where the change affects correctness: seed/order determinism, role pairs, D5 symmetry, label availability, no duplicate UTD credit, buffer lifetime, cancellation, restart and restore. Extend risk-weighted coverage in changed modules rather than chasing an arbitrary global percentage.
- Add representative performance regression records with immutable fixtures, counterbalanced runs, memory bounds and prospectively chosen margins. Require exclusive GPU ownership for hardware tests; CI must not compete silently with production.
- Record an architecture decision and an experiment/evidence receipt for each material change. Update stale runbooks and remove obsolete compatibility branches only through an explicit deprecation path.

Avoid a broad rewrite. Each refactor should enable a specific measured improvement, remove duplicated policy or make an important invariant enforceable.

## 8. Durability and operational efficiency

Preserve integrity while reducing measured backup and resume overhead.

The arena currently serializes a growing full snapshot after searched-move batches under a shared lock. If profiling shows material cost, introduce a versioned per-game append journal with periodic compaction. Every acknowledged move and completed pair remains durable; replay is idempotent, torn records are detected, and compaction survives crashes. Do not merely save less frequently and silently lose recovery guarantees.

Backup header reuse and retirement-race protections already exist. First distinguish NFS metadata, copying, hashing, catalog-history scanning and retries. Then test bounded concurrency, less repeated catalog work, and retention through the existing verified dependency/GC machinery. Do not weaken checksums, publish before verification, or revive timestamp-only trust caching.

Propose a recovery-point objective of **under 30 minutes at the 95th percentile**, with an explicit hard alert at 60 minutes; confirm feasibility against measured full backup and restore costs. Display capture age, completion age and oldest protected source time separately. Run a restore drill covering learner/optimizer/EMA, replay, arena moves/pairs, allocation ledgers and measurement-scheduler state.

## 9. Experiment and rollout protocol

1. **Freeze control.** Capture production code/native binary/environment/profile digests, exact checkpoint, replay cutoff, rules/search/measurement contracts and hardware topology. Reconcile the newer local checkout before selecting a release.
2. **Register the hypothesis.** Name the single factor, primary metric, uncertainty method, wall/GPU-hour ceiling, seeds, stopping rules and admission criteria before results are known.
3. **Use the cheapest valid screen.** Static analysis and deterministic replay first; isolated target-host fixed-workload benchmark next; whole-pipeline canary after that. Do not run unbounded diagnostic work on live GPUs.
4. **Separate engineering from learning evidence.** Exact implementation parity can justify a runtime fix with a representative throughput gain. Changed targets, reuse or resource allocation need whole-system strength evidence before claiming Elo/hour improvement.
5. **Use adequate trial length.** The current candidate cadence is roughly eleven hours and evaluation can take another long interval. An eight-hour arm can end before meaningful champion evidence exists. Start power/cost planning around 24–48 hours per arm, then set the actual budget using empirical pair outcomes and game costs. Do not keep extending losing trials until they look positive.
6. **Screen then confirm.** One seed for ranking/futility, then only control and the selected treatment on seeds 17, 18 and 19. Use common immutable raw weights, EMA, optimizer moments, scheduler state, replay cutoff and accounting; compatible paired evaluation fixtures; isolated replay; counterbalanced arm order; and equal full-machine wall budgets. Keep learner histories and identities separate. Reserve fresh confirmation opening/seed streams distinct from exploratory screening.
7. **Keep the current adoption gate.** Require positive conservative treatment-versus-control advantage in every confirmation seed and at least 20% median point Elo/hour improvement. If power analysis says the gate is unaffordable, report that and design any prospective policy revision explicitly; do not weaken it after seeing results. Inconclusive results retain control.
8. **Charge all costs.** One 24-hour eight-GPU arm uses 192 provisioned GPU-hours. Three control/treatment seed pairs require at least 1,152 GPU-hours at that length, before any additional final evaluation or recovery overhead. Agree a campaign ceiling before execution and avoid a broad Cartesian parameter sweep.
9. **Deploy incrementally.** Shadow scheduler decisions, verify candidate release and native artifacts, take a verified stopped-boundary checkpoint/backup, migrate compatible state, resume exactly, and canary through representative evaluation/backup behavior. Roll back code/settings while preserving valid learned state and durable evidence. Never return to a reader unable to load newly written formats.

Pin each arm's champion at a preregistered endpoint. Promotions after that endpoint cannot improve its recorded endpoint score. Reserve a final measurement interval inside the total fixed arm budget, or explicitly version a protocol that allows later independent evaluation of the pinned endpoint and charges all additional provisioned time/resources. The current ablation runner excludes measurements completed after its cutoff; do not silently change that rule. Keep the endpoint identity, measurement cutoff and resource-release time separate in reports.

Hard rollback/stop conditions include changed deterministic outcomes for a purported exact optimization, invalid/nonfinite training, replay corruption or duplicate credit, lost evaluation evidence, ownership/memory violations, repeated unrecovered handoff failures, and unacceptable recovery freshness. Performance and statistical futility conditions must also be preregistered.

## 10. Ordered implementation backlog

| Package | Deliverable | Dependency | Completion evidence |
| --- | --- | --- | --- |
| A | Baseline manifest, authoritative telemetry, profiling fixtures | None | Full lifecycle cost accounting and no stale-worker aggregates |
| B | Protected measurement queue, explicit epoch and honest Elo/hour report | A | Complete current measurement; starvation/restart tests |
| C | Diagnosed GPU handoff fix | A | Reproduced fault fixed; ownership/failure tests and sustained canary |
| D | Arena execution benchmarks and selected exact optimizations | A, C | Same outcomes/statistics; faster complete workloads; bounded memory |
| E | Actor typed-buffer/overlap work and stale-history prevention | A | Same target/search semantics; improved eligible production |
| F | Optimized shared versus dedicated evaluation topology | B, D | Independent whole-machine strength/cost comparison |
| G | Reuse and learning-clock experiment, then selected sampling/search treatment | B, E; adequate replay split | Registered seed screen and three-seed confirmation |
| H | Resume/backup persistence optimization where profiling warrants | A | Measured lower overhead, recovery-point target and verified restore |
| I | Distilled actor/model/precision research | B, G | Equal-time strength plus end-to-end training evidence |

Code-quality work is embedded in A–I, not deferred to a cleanup project. B, C and the actor profiling/prototype work can proceed in parallel after the shared measurement contracts are frozen. Do not combine them into one production rollout.

After B–E, review actual evidence and re-rank the remaining work. The plan succeeds by delivering measured strength faster with a maintainable, verifiable system—not by completing every speculative item.

## 11. Optional hardware track

Hardware changes are outside the initial implementation sequence. After fixing scheduling and measuring real workloads, compare a separate evaluator, additional actor capacity, and a different accelerator only with representative model/search benchmarks and current provider pricing. Charge every added device, transfer, startup and evaluation hour. Preserve an equal-cost comparison alongside the primary equal-wall-time comparison. Do not move evaluation onto the nearly full learner GPU without a proven memory/ownership design, or buy more learner compute while fresh data remains the binding limit.

## Evidence and implementation references

Local references are relative to this document:

- [Existing scientific/adoption contract and prior null experiments](model-improvement-roadmap.md)
- [Latest production auxiliary deployment](auxiliary-predictions-deployment-20260918.md)
- [Already deployed arena/cache work](arena-checkpoint-cache-efficiency-20260912.md)
- [Already deployed CPU work](cpu-efficiency-deployment-20260912.md)
- [Adaptive promotion statistical contract](adaptive-pie-promotion.md)
- [Promotion scheduling and per-session execution](../deltreltrain/promotion.py)
- [Historical job selection](../deltreltrain/historical_evaluation.py)
- [Independent strength implementation](../deltreltrain/balanced_strength.py)
- [Strength report](../scripts/strength_efficiency_report.py)
- [Actor selection and execution](../deltreltrain/actor.py)
- [Inference execution](../deltreltrain/inference.py) and [broker](../deltreltrain/inference_batching.py)
- [Actor pause protocol](../deltreltrain/actor_pause.py) and [coordinator](../deltreltrain/orchestration.py)
- [Arena loop and resume](../deltreltrain/arena.py), [occupancy benchmark](../scripts/benchmark_arena_occupancy.py)
- [Calibration data split](../scripts/run_frozen_replay_optimizer_calibration.py)
- [Three-seed adoption enforcement](../scripts/compare_elo_ablation_seeds.py)
- [Backup implementation](../scripts/training_disaster_recovery.py), [header-cache correctness](backup-header-cache-20260914.md)
- [Existing CI](../../.github/workflows/ci.yml) and [hardware validation](../../.github/workflows/hardware.yml)

Primary external references, used for methodology rather than transferred speedup claims:

1. [NVIDIA CUDA Best Practices: realistic application profiling](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
2. [NVIDIA Nsight Systems: focused profiling and NVTX ranges](https://docs.nvidia.com/nsight-systems/UserGuide/)
3. [PyTorch profiler: scheduling and overhead](https://docs.pytorch.org/docs/main/profiler)
4. [Wu, Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565)
