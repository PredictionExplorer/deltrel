# Training recovery deployment — September 28, 2026

**Deployed and live canary passed.** The existing eight-H100 service runs the
identity-preserving production commit `659430fc856bb44f005a0898c96f5e0cfd981c15`.
The corresponding main implementation is `78868b6fe5b906616b17b9272bfd7ff90a6fd6ce`.
The model remains 17,467,840 parameters; rules, search budgets, precision,
game-type weights, and champion publication criteria are unchanged.

See the [implementation contract](training-recovery-20260928.md) and
[machine-readable evidence](training-recovery-deployment-evidence-20260928.json).

## Active behavior

- Completed rejection evidence survives candidate replacement and restart. The
  earned plateau response applied exactly once, setting the multiplier to **0.5**.
- Fresh self-play from champion **566,428** remains eligible despite its model
  age. The exception is capped at **25% per selected ring/mode cell**, with a
  six-hour game-publication window and a fixed activation boundary.
- The first canary selected **3,584 protected positions**: 1,536 classic-pie and
  2,048 double-pie. This was **0.157% of the selected replay window**, while new
  data was arriving, and every per-cell cap passed. Selection is not a claim
  that exactly 25% of consumed updates already came from the champion.
- Production reuse remains **1.5**. A separate **2.0** profile is prepared with
  full-state resume, prospective credit, and preserved LR/EMA/cadence data clocks.
- Raw and EMA diagnostic arms share frozen checkpoint/champion bytes and matched
  game settings. Both bounded GPU sessions executed and saved resumable action
  histories. **Neither completed a paired game**, so there is no averaging
  winner or strength conclusion. EMA decay remains unchanged.

The restored checkpoint includes the original model, EMA, optimizer and scheduler
state. After exact resume, the existing configured plateau policy intentionally
cleared optimizer moments when applying its rate cut; it retained weights and
EMA. This is distinct from discarding training progress or rewinding the learner.

## Checkpoint continuity and live health

The preserved boundary is step **743,910**, epoch **24,508**, with
**380,881,920 consumed examples**. Checkpoint SHA-256:
`f72fa1cb31f8f2c8c9c47797528b43b7deffc36e3a046ae8f10474b228b331ff`
(213,775,175 bytes). The migrator and live `resumed_from` pointer agree on that
exact file. **Zero uncheckpointed steps or examples were discarded.**

The independent canary reached step **744,033**, advancing 123 updates / 62,976
examples. All ten workers and eight GPUs were healthy, with zero worker restarts,
zero observed nonfinite losses/gradients, and the valid arena lease on GPU 7.
The strength epoch, UTD segment, stopped coordinator journal prefix, measurement
accounting, and pinned measurement identity survived. Monitoring, report and
backup timers were restored.

All times below are **September 29 UTC** (September 28 in Chicago):

| Event | Time |
| --- | --- |
| Graceful stop requested after the running backup completed | 00:37:30 |
| Source workers stopped | 00:39:33 |
| CUDA qualification began | 00:39:50 |
| New workload start requested | 00:46:12 |
| Sustained readiness and support restoration completed | 00:59:39 |
| Independent live canary passed | 01:01:12 |

The startup interval included replay-ledger validation and cohort reconciliation.
The learner resumed before all actor cohorts completed that initialization.

## Qualification and preservation

- **4,166 target-host CPU/native tests passed**, with zero failures, errors or
  skips. Ruff, Pyright, source checksums and profile validation passed.
- Actual production CUDA inference/native search, checkpoint/resume behavior,
  and the separate shadow graph parity check passed. Diagnostic child processes
  used one explicitly reserved GPU and released ownership between stages.
- All 103 installed package versions and the native extension bytes matched the
  preceding release. Native SHA-256:
  `d023eb194523ae6f7239fd9ee7f9661059baf47aa8ccb3e00e2e7de8f2fada33`.
- Independent stopped-archive verification hashed **53,876 files** and checked
  **53,091 replay shards**, their ledger rows/counters and the exact checkpoint.
- The first local broad run had one outdated historical-hash fixture among
  4,222 selected tests. The fixture was corrected to omit the new disabled
  defaults; the affected suite and later integration checks passed. The final
  immutable production release passed the complete target-host suite above.

The first post-deployment disaster backup completed successfully at **01:29:40
UTC**. Independent verification checked every referenced payload in **51,793
catalog files / 51,791 objects / 22,433,003,785 bytes**, the exact deployed
profile, all **28 admission dependencies**, and **18 plateau verdicts**. Snapshot
SHA-256: `9828b3ff53b5989875e11be104e2ffd539a04c5fe0a79758c9ee9032029556a9`.
The replay capture began at 00:59:44 UTC, after deployment. A new timer invocation
started immediately after the successful writer, so its active state was not
misreported as failure or used to infer completion. This was full payload and
recovery-dependency verification; it did not execute a production restore.

## Operational locations

- Release: `/home/ubuntu/edgeconnect-releases/variant-training-recovery-20260928`
- Active profile: `/home/ubuntu/edgeconnect-runs/variant-network/profile-training-recovery-20260928.yaml`
- Evidence: `/home/ubuntu/edgeconnect-rollouts/training-recovery-20260928`
- Prepared reuse trial: `/home/ubuntu/edgeconnect-runs/variant-network/status/training-recovery-20260928/reuse-2-profile.yaml`

The production lineage retains StarTrain/EdgeConnect artifact identities. The
tested compatibility backport avoids combining this recovery with a rules,
checkpoint, replay or native-binary identity migration. No claim of improved Elo
is made until subsequent, comparable arena evidence supports it.
