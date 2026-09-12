# Graceful CPU efficiency rollout — September 12, 2026

**Deployed and verified.** The controller completed successfully at 12:20 UTC.
An independent final check at 12:24 UTC found learner step **219,255**, exact
resumption from **218,423**, ten healthy workers, zero worker restarts or failed
inference requests, restored protection services, and a verified backup of the
new configuration. The detailed
[evidence ledger](cpu-efficiency-deployment-evidence-20260912.json) retains the
source identities, receipts, and CPU qualification results.

The release is `8d0d643d30430afd9cd7c662fe4648aedecfca3e`, staged at
`/home/ubuntu/edgeconnect-releases/variant-cpu-efficiency-20260912`.
It contains all changes from the
[implementation report](cpu-efficiency-implementation-20260912.md), plus a
server-qualified refinement to native parallel submission. The previous
production release is `b21e120418f41ca5402ff4dfb034dd96396e9f14`.

## Preserved training contract

The new profile is byte-identical to the current profile, with SHA-256
`a3c41d7b92b10b0eb66e17a09979f07c5371b5e3b254c08f2dd3fa8040e9527a` and canonical
configuration `7f6617e3e1c31ad3f89cb63cc91e9b587333b2907bf6bee95d8599beb942f3e3`.
The source-only migration recorded `changes=[]`, no UTD segment change, no
new strength epoch, and zero discarded uncheckpointed steps. The existing
graph32 admission evidence and memory limits remain pinned. All 103 Python
package names/versions match the previous environment; the native binary is
rebuilt from the new source.

## Native correction during server qualification

The initial logit-count-only guard improved larger native workloads on the Mac,
but the Linux qualification exposed a long-search regression. After first-visit
prefetch ends, many single-row sessions still exceed 8,192 total policy logits;
parallel scheduling then repeats for small per-session jobs.

The corrected guard requires at least two response rows per pending session in
addition to multiple sessions, multiple threads, and 8,192 logits. Ordinary
single-row leaf updates retain serial submission. Validation still completes
for every session before any tree is changed. The 40-to-32-byte edge storage
improvement remains in place.

The final paired comparisons used four native threads, CPU affinity 32–103,
ring 10, first-visit width 8, and a 53-candidate cap limited by the simulation
budget. Ratios are baseline time divided by candidate time for native
`next_requests` plus `submit` only.

| Roots | Budget 27 | Budget 640 |
| ---: | ---: | ---: |
| 32 | 0.995× | 1.010× |
| 64 | 0.965× | 1.022× |
| 128 | 1.018× | 1.017× |

Request ordering, state fingerprints, visits, values, and policy outputs matched.
These are near-parity component measurements with a 20% edge-storage reduction,
not a universal native speedup. The earlier Linux long-search ratios of
0.868–0.905× were rejected. The original Mac timing ratios must not be used as
server throughput or Elo/hour gains.

## Qualification and recovery

The refined source passed 222 CPU tests on the server (one CUDA test intentionally
skipped while training was active), the Rust workspace tests, and 95 isolated
controller/backup/readiness tests. Sixty native/inference tests also passed
against the rebuilt local extension.

Both stopped H100 checks passed. The frozen production model comparison covered
ring 10 at batch 16, with eight finalized and eight policy-only legal prefixes.
All seven loss tensors and 255 parameter-gradient tensors matched numerically
exactly across old/new code in FP32 and BF16, with validated and trusted targets.
Every maximum absolute difference was zero. Native BF16 inference responses
also matched for cold prediction, cache hits, and CUDA graph replay; no graph
fallback or validation failure occurred. The comparison took 22.87 seconds and
did not change production weights or replay. Its tensor gate uses `torch.equal`,
not a signed-zero-sensitive byte comparison; the raw report's “bitwise” label
should be interpreted with that limitation. This is execution equivalence, not
an H100 throughput or Elo experiment.

The controller ran under systemd and preserved the exact stopped checkpoint:
SHA-256 `f808d0e5f61e219489127ceeabab88c929bcc8fadfae6ecc8f6274496bebba42`,
**111,832,576 consumed examples**, epoch **2,353**. All frozen learner and arena
control hashes remained unchanged across migration.

| Boundary | UTC |
| --- | --- |
| Graceful stop requested, after the in-flight backup drained | 11:33:42 |
| All workers stopped and checkpoint verified | 11:35:57 |
| New release launch requested | 11:52:48 |
| Sustained healthy readiness confirmed | 11:57:12 |
| Post-deployment backup/report complete | 12:20:30 |
| Independent final source/worker/backup audit | 12:24:38 |

The exact stopped snapshot took **938.86 seconds**. It reused 5.57 GB without
upload, recorded 83,021 verification-cache hits, and fully hashed 1.42 GB of
other objects. These are component counters; they do not identify hashing as
the dominant backup cost. The post-readiness backup verified **30,179 objects**
and **9.16 GB** of catalog data, snapshot SHA-256
`3366e035fa63df49e32812c8b9a3813830717976b1a7f358dbafdfccd58e6769`.
Training continued while that final backup was created and verified.

The final audit checked all 575 release source files, the native binary loaded
by actor/arena processes, profile and gate hashes, fresh heartbeats, GPU health,
graph memory/entry bounds, and active monitoring/report/backup units. The normal
GPU-7 arena pause remains valid healthy behavior.

The retained recovery path checks that the old reader can load the newest
stopped checkpoint before any inverse source-only migration. Rollback does not revert learned
state or change graph capacity to 16. Monitoring and backup unit states are
restored; the unrelated disabled continuity timer and existing seed18 barrier
remain unchanged. Rollback was not needed for this deployment.

No whole-training throughput or equal-wall-time Elo improvement is claimed.
