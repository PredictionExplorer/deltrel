# Live replay rollout — September 11, 2026 (UTC)

## Deployment status

**Deployment is complete and independently verified.** Training restarted with
all three live-data settings enabled at **02:18:04 UTC**. The controller completed
its readiness, accounting and backup checks at **02:37:19 UTC**. An independent
audit at **06:06:12 UTC** passed every check after nearly four hours of operation.

The learner resumed from the exact activation checkpoint at step **180,377** and
advanced to **184,405**. All ten workers and the coordinator were healthy with
zero restarts, zero inference failures and healthy GPU hardware. Monitoring,
reporting and both backup timers were active; temporary build/check/deployment
processes were stopped.

Application source: `70058b1ff5ca34ecf20b0b1112bbe93e9e817ee1`.
Immutable release: `/home/ubuntu/edgeconnect-releases/variant-live-replay-70058b1`.
Server evidence: `/home/ubuntu/edgeconnect-rollouts/live-replay-20260911`.
The compact retained record is
[the deployment evidence](live-replay-deployment-evidence-20260911.json).

## Validation before cutover

- The server's complete native-required Python suite passed: **2,522 passed,
  12 skipped**. CUDA was hidden during these tests while production continued.
- **99 Rust tests**, strict workspace Clippy, Ruff and Pyright passed.
- All **513 archived source files** passed checksum verification before and
  after building; the release was then made root-owned and read-only.
- Python and all **103 installed package versions** match the previous release.
- The source controller passed 21 recovery/readiness cases. The activation
  controller passed 24 cases; exact stopped-snapshot verification passed another
  21 cases. Controller files are separately hashed and retained with the evidence.
- The production-model H100 smoke passed in **12.67 seconds**, retaining BF16.
  It verified the rebuilt native binary, lazy state export, trusted inference,
  cache reuse, native search, a policy-only CUDA update, final-label enrichment
  and an idempotent retry. The temporary replay store granted two distinct
  position credits, two enriched labels and zero retry credits. Production model
  files were read-only inputs.

The native binary SHA-256 is
`a860d66c128cedbd3429765180459bdcdad34104d4ec520633f1025ff26abf1b`.
The build initially needed the standalone Rust test runner's Python library path
and the pinned toolchain's Clippy component. Those build-environment issues were
resolved before cutover; successful application tests were retained.

## Source-only boundary

All ten workers exited normally with exit code zero. The durable checkpoint was
step **180,002**, with **92,161,024 examples consumed**, SHA-256
`3c49171f1287fc725834f7fbb09069dd6388aaca1dd1ce03e14daf29c2c0f799`.

Verified local and off-server backups preceded migration. The source-only
migration discarded **zero optimizer steps**, changed no profile settings or UTD
segment, and preserved all checked learner/arena control files. The recorded
configuration hash remained
`6af7df9bf422e2befbcf3c4a19a9090271996cb4183c80e466aaac0c33ce57b1`.

The new runtime resumed the exact checkpoint and passed sustained readiness.
The disabled-publication baseline profile is
`/home/ubuntu/edgeconnect-runs/variant-network/profile-live-replay-baseline-20260911.yaml`,
with unchanged profile SHA-256
`1db1a8b29a177e9b92a455d1468247e88e5fcfc5e74ce0aef1d9867a09265d62`.
Fresh local and off-server backups also passed after readiness.

## Activation and live evidence

The second graceful stop preserved step **180,377**, **92,353,024 consumed
examples**, and checkpoint SHA-256
`e2a7bcd362eb85cb36da9ed4d22494135b5c2885b8d5d4ac61c059a639d39eb5`.
All ten workers again exited with code zero. Its replay boundary contains
**68,419,884 distinct committed positions**, schema 5, and zero publication
revisions/enriched rows. This is the exact baseline for verifying new logical
credit after activation.

The active profile changes exactly:

| Setting | Previous | Activated |
| --- | --- | --- |
| `selfplay.policy_publication.enabled` | false | true |
| `orchestration.model_refresh.work_scheduling.enabled` | false | true |
| `learner.replay_refresh_seconds` | 300 | 60 |

First publication remains eight decisions, with 32 additional decisions between
later prefixes. Existing 128-game lease quotas and rolling refills remain in
place. Model, precision, search budgets, optimizer, LR/EMA clocks, evaluation and
the 1.5 update-to-data ratio are preserved. Previously enabled geometry sharing,
small graph buckets and interruption-policy preservation stay enabled.

The target profile is
`/home/ubuntu/edgeconnect-runs/variant-network/profile-live-replay-20260911.yaml`.
Its SHA-256 is
`e7f6420f08d1b5efdad9490d1ee0521041840a717936c25ffbc61556109458da`;
its canonical configuration SHA-256 is
`6fba967f63a761a643fa2ecb1636606ae1fb9c34d2667f7c1539c20d870b9536`.

Activation discarded **zero optimizer steps**, made no UTD transition, and
preserved all checked learner/arena control files. Live revision accounting was
validated against one read snapshot:

| Board size | Distinct new positions | Earlier positions enriched with final labels |
| --- | ---: | ---: |
| 4 | 38,840 | 25,472 |
| 6 | 30,717 | 25,376 |
| 8 | 47,358 | 40,864 |
| 10 | 1,106,432 | 933,448 |
| Total | **1,223,347** | **1,025,160** |

The committed-position increase matched the sum of durable game heads exactly.
Enrichment and revision counters also matched; enrichment did not mint extra
position credit. Four sampled pending/final payloads passed checksum, target and
identity validation without modifying production pins. The initial activation
gate had already observed fresh prefixes from every board size and completed
label enrichment before declaring readiness.

Both local schema-6 replay backup and off-server snapshot verification passed.
The post-activation verified disaster snapshot covers **16,071 objects** and
**22,129,935,845 catalog bytes**, with manifest SHA-256
`2cd38d2c970dd28c594674d27ba8fc02e328efadbf73f19db850b019300b922b`.
Normal backup jobs continued afterward in the same namespace.

The first independent audit exposed only an audit-helper assumption that timers
have a process ID field. The helper was corrected and passed ten focused checks;
the complete audit then passed. No application or service change was needed.

## Observed performance limits

A read-only screen of 05:00:00–05:59:55 UTC covered 720 monitor observations,
stable worker identities, and no nonfinite losses/gradients or graph validation
failures. It observed 312,265 newly published positions, 309,104 enriched
positions, and 1,246,745 physical rows written (about 4.0 times new-position
volume). All 55 replay freshness refreshes observed enrichment; window age at
refresh was 60.36–62.82 seconds, with a 61.37-second median.

| Metric | September 10, 17–18 UTC audit | September 11, 05–06 UTC |
| --- | ---: | ---: |
| Learner update-to-data wait | 81.45% | 80.25% |
| Median device step | 0.407 s | 0.407 s |
| Optimizer steps/hour | 1,150.6 | 901.3 |
| Useful neural rows/s, GPUs 1–6 | 18,471 | 17,042 |
| Learner replay wait | 19 s / 0.53% | 150 s / 4.17% |

The delivery changes are functioning, but the learner remains limited by fresh
data production. This observational comparison does **not** demonstrate a
throughput gain: model identities, tasks and scheduling differ between windows.
It also reveals a measurable cost of the shorter refresh interval and revision
writes. In completed GPU-task samples, replay append time was 2.82% of summed
task wall time versus 0.056% previously; those tasks can begin outside the
measurement window, so this is not an exact estimate of GPU idle time.

The deployment is verified for correctness, continuity and operational health.
Improved Elo per hour still requires a controlled learning comparison; these
results do not establish a 10× gain or elimination of the original data wait.

## Recovery and operational notes

After the first revision commits manifest schema 6, rollback must retain a
schema-6-capable runtime. The activation controller reverses the three profile
changes on this same release and preserves durable checkpoint/replay state.
It never returns the upgraded manifest to the older E8 reader. Controller
recovery also stops isolated GPU-check descendants before restarting training.

All workload, preflight, monitor, report and backup service paths are updated
together. Normal disaster backups retain the existing namespace
`/lambda/nfs/texas-north-fs/edgeconnect-dr/variant-network`.
The global continuity service/timer remain inactive, with the timer disabled.

Normal snapshot creation currently takes approximately 11–12 minutes because
the backup tool repeatedly parses roughly 700 historical catalogs. This
activation's stopped-boundary verification used the existing explicit
`verify --snapshot` option: it first pins the new committed snapshot, checks its
exact stopped profile/source/checkpoint authority, then performs full payload
verification. This avoided two extra historical scans. Snapshot creation and
verification of backups after training resumes retain the normal behavior.

See [the design and operating contract](live-replay-publication.md) for revision
accounting, bounded physical write amplification and rollback compatibility.
No reduction in the previous 81% wait or Elo/hour multiplier is established by
the deployment checks alone.
