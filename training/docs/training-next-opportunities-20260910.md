# Next training and inference opportunities

## Assessment

There are several new, concrete opportunities. The strongest conservative changes remove redundant graph messages, avoid unused search-tree allocations, and replace the separate attention-bias backward computation. The clearest 10× work reduction is in the website's unusually large default search budget, but retaining playing strength at the smaller budget must be demonstrated.

A 10× improvement in end-to-end training or Elo/hour is not established by this audit. Some opportunities offer order-of-magnitude reductions in a specific allocation or feedback delay; those quantities must not be presented as whole-system speedups. The strongest larger training experiment combines a cheaper distilled actor with less search, while retaining the current large learner and measuring equal-time strength.

Production code, settings, services and replay were not changed for this audit. The deployed source remains `e8e877cd7292dd44558e280c8117e6242820c2ce`, with the optimized profile already enabled. CPU probes and existing server telemetry provide the measurements below. No new isolated H100 kernel benchmark was run.

## Current bottleneck, after the previous rollout

The common 17:00:01–17:59:56 UTC interval contains 720 monitor observations with stable process identities and no observation gaps.

| Measurement | Observation | Implication |
| --- | ---: | --- |
| Learner progress | 171,934 → 173,083; 1,150.6 steps/hour | Faster than the earlier observed period, with changed models/data |
| Median logged device step | 0.407 seconds | The shared-geometry improvement is visible in production timing |
| UTD wait | 81.45% of the interval | Fresh self-play still limits permitted updates |
| Learner GPU utilization | 9.97% | Making only the learner faster does not remove the data-supply limit |
| Learner resident GPU memory | 65.62 GiB | More capacity is available than before the rollout |
| GPU actors 1–6 utilization | 51.94% mean | Requires kernel-level attribution, not an assumption of compute saturation |
| Current broker busy fraction | 98–100% over a separate 131-second sample | Producers already keep the inference owner occupied |
| Actor process CPU consumption | 1.25–1.33 cores each, little runnable delay | Adding CPU cores alone lacks supporting evidence |
| Actor padding | 14.1–16.2% | Removing every padded row has only about 1.16–1.19× row-work headroom |
| Actor prediction-cache hits | 1.66–2.94% | A larger cache alone is not an obvious large gain |

The earlier ef7 period observed 939 steps/hour and a 0.503-second median device step. These are observational comparisons with different checkpoints, trajectories and arena occupancy. Aggregate useful neural rows/second did not improve in this shorter interval; that does not isolate the effect of the batch-bucket change.

The five-minute replay refresh is functioning: eleven freshness refreshes occurred at 300.5–302.3 seconds. The prior short-file, six-mode selection and stale-window defects should not be counted again as new opportunities.

## 1. Resolve the website's split search defaults

**Priority: high for interactive inference. Small implementation, but a strength experiment is required.**

Ordinary non-developer website play does not pass the stored search settings. The server client consequently supplies **4,096 simulations and 32 candidates**, while the UI/store and advertised server default use **512/16**. The explicit client request takes precedence over the server default. See [GameScreen](../../src/components/GameScreen.tsx#L317) and [server client](../../src/lib/deltrel/ai/server-client.ts#L20).

The two numeric public environment overrides are absent locally. Deployment-time browser environment values were not verified. Git history shows the discrepancy existed in the original implementation; later tests deliberately preserve developer-only budget overrides. It is a confirmed split-default inconsistency, not a proven accidental regression.

| Candidate default | Nominal simulation-work reduction from 4,096 |
| --- | ---: |
| 1,024 | 4× |
| 512 | 8× |
| 384 | 10.67× |

Compare these budgets on identical frozen models, six-mode positions and color-reversed games. Measure move latency, tactical disagreements and equal-time strength. Then unify the ordinary client with an explicit advertised preset or capability-aware budget. The reduction in simulations is arithmetic; equal strength and proportional latency reduction are not established.

The local browser path is separate. It uses its published model manifest, whose actual remote contents were not inspected. The repository's export configuration recommends 64 simulations and 16 candidates. Do not apply the 4,096 finding to every browser execution path.

## 2. Factor local messages by source node and edge class

**Priority: highest conservative model-kernel experiment. The mathematical redundancy is confirmed.**

Production uses the `mean` local operator. Its message is a function of the normalized source node and edge class; it does not depend on the destination. Nevertheless, [LocalEdgeBlock](../deltreltrain/model.py#L343) gathers a wide tensor and projects/activates each repeated neighbor slot independently.

Ring 10 has **550 distinct source/edge-class pairs**, versus **1,925 padded slots**. About 19.5% of slots are invalid. Compute the source projection once, form the required class-specific activated messages, and fuse their destination aggregation.

At batch 512, the existing FP32 neighbor gather is **1,443.75 MiB** and the message tensor is **721.875 MiB per local block**. The unique source/class table would be **206.25 MiB**. There are sixteen local blocks.

The neighbor projection accounts for 32.59% of analytic forward multiply-accumulates. Projecting once removes 27.93% of whole-model MACs, corresponding to approximately **1.39× arithmetic headroom** before memory effects. This is not a measured latency multiplier.

A local FP64 probe across all four rings, including nonzero rule conditioning, produced identical outputs and input/parameter-gradient differences no larger than 4.44e-16. The proof applies to `mean`; `source_gated` introduces destination dependence.

The earlier projection-order-only rewrite regressed in its H100 experiment. The new candidate must eliminate the repeated class nonlinearities and large intermediates, not merely move one linear layer. Compare existing, projection-only and factorized/fused variants at actor B32/64/128 and learner B512, using identical checkpoints and gradient oracles.

## 3. Replace the dense attention-bias backward workaround

**Priority: high for training time and memory. A concrete vendor API may avoid a bespoke kernel.**

The shared-geometry rollout reduced bias broadcasting, but [the custom VJP](../deltreltrain/attention_bias_autograd.py#L31) still recomputes attention probabilities and materializes dense FP32 intermediates. At B512 and sequence length 276, each matrix has **468,025,344 elements**, or **1,785.375 MiB**. Five named intermediates can remain live simultaneously: approximately **8.72 GiB** during a block's backward pass.

NVIDIA's cuDNN Graph API exposes `sdpa_backward` with both `bias` and a `dBias` output. Its documentation covers BF16, GQA and broadcast bias. The exact combination—12 query heads, three KV heads, head width 32, N106/276 and a shared per-head bias—must pass a supported-plan check before adoption. The server has cuDNN **9.20.0**, but its Python frontend package is not installed. [NVIDIA attention API](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html).

First test a fused forward/backward plan that returns Q/K/V and bias gradients. If unsupported, a tiled opaque VJP or Triton FlexAttention is the next option. A simpler memory-only intermediate step is careful reuse/release of fresh temporary buffers; input tensors must remain unmodified.

Preserve the existing FP32 bias-gradient oracle and the previously failing compiled BF16 ring-6 regression case. Compare whole-step latency, peak allocated memory, workspace, and all gradients. This workaround is absent during inference, so removing it does not directly accelerate actor prediction.

“Enable FlashAttention” is not itself a finding: [PyTorch 2.13 already selects fused backends for supported cases](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp). Its newer FA4 integration still directs captured trainable-buffer gradients to the Triton backend. [PyTorch backend guidance](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/).

FlashBias is a separate approximation/reparameterization experiment. Its learned-bias approach uses truncated SVD for inference or low-rank factors for training. Our arbitrary relation tables are not guaranteed low rank: a structural random-table probe produced rank 276 and 79.5% reconstruction error at rank 32. That is not the trained model's spectrum. Its implementation's head-layout requirements also complicate 12/3 GQA. [FlashBias paper](https://papers.nips.cc/paper_files/paper/2025/file/1dc3d70df51a218497529df998a8a8ce-Paper-Conference.pdf).

## 4. Stop allocating full edge records on leaves never traversed

**Priority: high for a bounded native-memory improvement.**

[Tree expansion](../crates/deltrel-search/src/tree.rs#L838) immediately softmaxes the policy and allocates a **40-byte Edge for every legal action**, including leaves that are never visited again.

A release-mode CPU probe using the actual tree implementation, a ring-10 position after three placements and a synthetic evaluator produced:

| Budget | Edge allocation per root | On never-traversed leaves |
| --- | ---: | ---: |
| 53 | 585,400 bytes | 574,520 bytes |
| 640 | About 6.86 MB | About 3.79 MB |

For the 53-budget search, 53 of 54 nodes have no outgoing visits. Retaining logits/value immediately and constructing edges only on traversal can replace a 40-byte unused edge with roughly a four-byte logit. That is approximately **10× smaller untouched-leaf policy storage**, not a 10× reduction of total tree memory or wall time.

A smaller first patch can replace `Option<usize>`—16 bytes in this layout—with a checked compact child handle. The measured structure could shrink from 40 to approximately 32 bytes. A reusable contiguous edge arena can then reduce allocator traffic.

Require exact request ordering, visits, Q values, policy targets and cancellation/reuse behavior with a deterministic evaluator. Then measure native time and peak RSS under realistic roots/budgets; the synthetic probe is not a trained-network performance measurement.

## 5. Finish removing redundant native exports and copies

**Priority: high for small, exact implementation changes; moderate whole-system ceiling.**

[pack_requests](../crates/deltrel-py/src/lib.rs#L3225) still eagerly creates the legacy `StateData` export: roughly two dozen vectors and recomputed state hashes. The trusted compact-key inference path does not read this object. Make that export lazy while preserving compatibility for callers that request it.

[Selected-feature packing](../crates/deltrel-py/src/lib.rs#L1282) deep-clones cached rows before copying them into the final flat buffers. A borrowed-row packer can remove the intermediate clone while keeping public writable buffers isolated from immutable cached data.

The next larger bridge experiment is buffer-based submission instead of Python float lists. Current inference copies packed predictions to the CPU, turns rows into bytes, reconstructs a CPU tensor, applies softmax there and converts legal logits into Python lists. See [GPU-to-CPU return](../deltreltrain/inference.py#L881) and [postprocessing](../deltreltrain/inference.py#L1107).

A typed native buffer API could avoid repeated scalar boxing and copying. GPU-side outcome/score reduction could also avoid transferring all 303 score logits on ordinary non-detailed requests. Detailed APIs, cache semantics, utility-weight changes and numerical parity must be preserved.

Return processing occupies only **8.8–9.7%** of actor wall time in the measured window. Eliminating it entirely has an idealized **1.10–1.11×** serial speed ceiling. Asynchronous transfer/postprocessing may improve overlap, but the current logs do not isolate pure GPU time from transfer time.

## 6. Reduce scheduling and supervision delay

**Priority: high for freshness and learning efficiency; this does not directly remove equivalent amounts of compute.**

During 17–18 UTC, **all 387,403 GPU-generated durable rows were candidate/ring10**. There were no fresh GPU rows for rings 6 or 8; CPU ring 4 added 5,141 rows. Coordinator snapshots showed only one or two bundles per GPU after roughly two hours.

The [coordinator](../deltreltrain/cohort_work.py#L31) schedules joint role/ring weights using bundles of four 128-game leases. A local reproduction with the actual weights and seven GPU seeds showed that **every GPU's first eight bundle choices are ring 10**. The first non-ring10 choice is bundle nine; the first particular ring-6/8 choice can be as late as bundle nineteen. These are scheduling choices, not a claim that every preceding bundle must finish before the next begins.

Test smaller compatible scheduling quanta, persistent scheduling credits and fleet-level phase staggering. Preserve compatibility pooling where it improves batching: indiscriminately mixing models/rings could reduce GPU throughput. Track useful eligible rows and neural cost by role/ring/mode, rather than judging balance solely by lease counts.

Publication task age was **53.1 minutes median**, with a 75.9-minute p90. Task age is not the exact residence time of each row, but it reveals a long feedback pipeline. Already-computed policy targets can exist well before game outcomes.

The existing policy-only interruption support makes **live policy-target streaming with later outcome enrichment** a concrete next design. A minutes-scale policy feedback target could reduce that component of delay by an order of magnitude. Terminal labels must remain unavailable until known; logical row IDs must prevent double publication and duplicate UTD credit. Later label versions need their own freshness signal. This is a data-model change with explicit accounting requirements, not a one-line switch or proven Elo multiplier.

## 7. Use the existing search-execution options for targeted experiments

The implemented first-visit batching option remains disabled in production. Native tests with 64 roots and a 256-row cap gave:

| Simulations | Width-1 calls | Width-8 calls | Logical request rows |
| --- | ---: | ---: | ---: |
| 53 | 54 | 15 | 3,456 in both |
| 640 | 641 | 603 | 41,024 in both |

Width 64 packed worse than width 8 in these actor tests. Single-root 53/53 search can collapse from 54 calls to two at width 64, but that is not the default website configuration. Its real serving cap and candidate count matter. Under the nominal 65% fast/35% full actor mixture, the first-visit work has only about a **1.26× ideal total-search ceiling** if its entire cost disappeared.

Subtree reuse also needs a coherent additional-budget policy. A native reproduction retained 146 nodes and 145 visits but still requested 641 new neural rows for the next 640-budget search. Current budgets mean **additional work**, so turning reuse on alone does not save simulations. Compare reuse with a reduced additional budget, separately accounting for inherited and fresh evidence.

## 8. The larger route: cheaper actors and fewer full-network evaluations

The code already contains a [teacher/replay distillation pipeline](../deltreltrain/distill.py#L394). This makes a cheaper actor experiment more concrete than building a second learning stack from scratch. It currently targets browser models and needs actor-specific model loading, measured execution settings and a reliable distillation data path.

Exact parameter counts for valid configurations:

| Width / groups / query heads / KV heads | Parameters |
| --- | ---: |
| Current 384 / 8 / 12 / 3 | 17,402,775 |
| Candidate 256 / 4 / 8 / 2 | 4,013,463 |
| Candidate 192 / 4 / 8 / 2 | 2,348,183 |
| Existing browser configuration 96 / 3 / 6 / 2 | 485,895 |

Parameter ratios are not latency or strength ratios. Start with the moderate student, preserve the large learner, distill all important heads and all six modes, and compare equal-time search strength. A small actor can propose moves; a large model can selectively verify difficult positions. That hybrid is a hypothesis requiring calibration, not exact speculative decoding.

Native search already produces action Q and visit information that [decision recording](../deltreltrain/selfplay.py#L1496) does not preserve as action-value supervision. Store genuinely visited action values, confidence/visit masks and utility/PDA context. Do not treat completed-Q placeholders as observed targets. This supports a cheap per-action Q head and subsequent search-light trials.

RMCTS is a particularly relevant new search experiment: its January 2026 preprint evaluates breadth-first prior-guided trees in large batches and reports about 3× faster batched search/training in its studied games, with much larger single-root results. It changes exploration and target construction, and its smaller networks/RTX3080 measurements do not transfer directly to this H100/Gumbel setup. [RMCTS paper](https://arxiv.org/html/2601.01301v1).

KLENT provides a search-free action-value-learning direction; its wall-clock Go evidence is much more modest than an automatic 4× promise. RGSC provides a way to restart from informative self-generated states; it reports stronger play across three other board games. Both need new training experiments, and consecutive same-player placements must be handled explicitly. [KLENT](https://arxiv.org/html/2602.10894v2), [RGSC](https://arxiv.org/abs/2602.20809).

These approaches could plausibly produce a combined 10× computational-work reduction. Only controlled learning curves can show whether target quality and playing strength survive. Their factors must not be multiplied into an Elo forecast.

## Additional bounded opportunities and rejected shortcuts

- Pack Q/K/V projections into one GEMM for inference: three projections in each of eight global blocks imply sixteen avoidable launches per forward. Preserve separate parameter identities for training, because combining Muon parameters changes its matrix operation.
- Screen FP8 on the global SwiGLU linears. They represent about 35% of forward MACs; even doubling that part yields only about 1.21× arithmetic improvement overall. Quantization/scaling overhead must be included. [NVIDIA precision guidance](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/speedups.html).
- Do not prioritize a complex incremental scorer: the local scoring probe measured only about 2–4 microseconds/state. Game states already have fixed-capacity storage and shared immutable topology.
- Do not assume generation is currently aging out before commit. All 75 completed postactivation tasks in the checked set were eligible. Old GPU-history work was scarcely represented, so horizon-aware eligibility remains a future guard rather than measured current waste.
- Do not raise UTD and call the added optimizer steps stronger learning. The fresh-data repair makes an explicit UTD sweep more meaningful, but publication, EMA and learning-rate clocks must be controlled.
- No Nsight Systems or Nsight Compute executable is installed on the server. Exact kernel attribution remains outstanding; GPU utilization percentages are not a FLOP-efficiency measurement.

## Recommended sequence

1. Screen the ordinary website's 4,096 budget against 384/512/1,024 on frozen models and positions.
2. Implement isolated prototypes for lazy native exports, borrowed feature packing and deferred leaf edges; validate exact native behavior.
3. Compare source/class fused local messages and cuDNN fused bias gradients on H100, charging complete execution and workspace.
4. Fix coarse scheduling phases and design policy-first publication without duplicate credit.
5. Run a moderate distilled-actor experiment and a controlled lower-search/RMCTS comparison from common checkpoints.

The first group offers clear engineering opportunities. The last group is the credible route toward 10×+ overall performance, with a corresponding need to establish retained strength.

The [evidence summary](training-next-opportunities-evidence-20260910.json) retains the telemetry aggregate, CPU-probe results, source identities and research URLs.
