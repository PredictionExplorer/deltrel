# Elo efficiency: production deployment evidence

The first engineering stage of the [efficiency plan](elo-efficiency-plan-20260923.md)
is committed and running on the existing eight-GPU server. It restores protected
independent strength measurement, fixes the reproduced cooperative handoff stall,
prevents admission of history expected to expire, and strengthens reporting,
calibration and recovery. It does not establish an Elo/hour improvement yet.

**Latest production source:** `1ad289502b0a7aa6428b4f2750b81b58054fc68a`,
the qualified source-only runtime follow-up completed at 02:26:04 UTC. The initial
release and follow-up evidence are distinguished below.

## Source and qualification

| Item | Evidence |
| --- | --- |
| Plan commit | `7b75c9d` |
| Main implementation | `d2028876a4bde2d7c2c0faee0013bbcaefc8b616` |
| Production compatibility branch | `codex/training-efficiency-production` |
| Initial deployed implementation | `db5470806d899519be77a2f18e2c2de5fe885fba` |
| Previous production source | `7a077107b1ebba55494b09f80689989e773b9dde` |
| Local regression suite | 3,954 passed; zero failures, errors or skips |
| Target-host regression suite | 3,930 passed; zero failures, errors or skips |
| Static checks | Ruff and Pyright passed in both source lineages |
| CUDA qualification | Native search and recovery smoke passed; shadow graph parity passed |

The production backport preserves the live StarTrain/EdgeConnect rules, features,
checkpoint and replay identities. It does not couple this release to the separate
Deltrel rename/migration. Native code, dependencies, model architecture, optimizer,
search budgets and precision remain at the qualified production baseline. CUDA
graphs were tested in shadow and were not activated.

The final Linux suite includes a corrected legacy-name test fixture. Earlier failed
qualification attempts remain recorded: Linux exposed a real checkpoint integrity
cache defect, which was fixed before deployment. Same-size writes and mmap mutation
cannot bypass the new complete checkpoint digest check.

Release: `/home/ubuntu/edgeconnect-releases/variant-elo-efficiency-20260923-v2`.
Server evidence: `/home/ubuntu/edgeconnect-rollouts/elo-efficiency-20260923-v2`.

| Artifact | SHA-256 |
| --- | --- |
| Sealed source archive | `6e59921f52762f4af27976c5458f0284dad965cdf6561dc6e5bbdc42d8d30252` |
| Native extension | `d023eb194523ae6f7239fd9ee7f9661059baf47aa8ccb3e00e2e7de8f2fada33` |
| Active profile bytes | `d811d2246dd0364a5fcc9eba5dad7778ad7fb0600734f26014c6ae420da474b3` |

## Checkpointed cutover

All times below are UTC on September 24, 2026 (September 23 in Chicago).
The supervised deployment waited for the existing backup, requested graceful
shutdown at 00:43:27, and observed all old workers stopped by 00:45:56. It preserved
the stopped run, passed the CUDA smoke, performed the admitted scheduling migration,
and restarted training. Sustained readiness and support-service restoration completed
at 01:02:14.

The exact resumed checkpoint was step **587,954**, containing **301,032,448**
consumed examples. Its SHA-256 is
`d4f8aa41fd48135ea146b7e95209028dc5a75089864bcd9f970b750706ab171e`
(213,774,983 bytes). **Zero learner steps and zero consumed examples were discarded.**
The controller verified the resumed checkpoint identity, rather than inferring
continuity from a later step number.

The initial release's profile is
`/home/ubuntu/edgeconnect-runs/variant-network/profile-elo-efficiency-20260923.yaml`.
The migration changes only the four scheduling settings documented in the
[release notes](elo-efficiency-release-20260923.md), with an immutable admission
receipt and explicit strength epoch.

## Live canary and preservation

The canary observed all ten workers healthy with zero new restarts and all eight
GPUs healthy. The learner reached step 588,234, advancing 280 updates and 143,360
examples after the preserved checkpoint.

Independent measurement received a coordinator-confirmed **301.148670485-second**
GPU slice, preserved 242 searched moves across 32 game states, and returned GPU 7
to self-play. The ledger recorded the exact settled interval, no active lease and
complete accounting. No completed measurement pairs were claimed from this first
slice. Promotion evidence remained resumable.

All 29 active actor streams reported current process identities; 13 retired rows
were excluded. The report correctly identified champion 537,130 and left current
independent strength and Elo/hour unavailable pending sufficient evidence.

Independent full verification of the stopped archive checked **56,440 files**,
including **55,763 replay shards**, 157 model artifacts and 427 arena files. Exact
model, EMA and optimizer state was preserved. All 112 started promotion games
retained their durable action prefixes and completed results; at the boundary,
108 games / 54 pairs / 23,109 searched moves were complete.

The full archive verification receipt is
`evidence/preserved-archive-full-verification.json` under the server evidence root.
The first post-deployment disaster snapshot completed at 01:24:18 UTC after
22 minutes 4 seconds. Its committed catalog SHA-256 is
`26ef796ab61156c71181b5ce66057557d03d55ee4bb9594fd751fdf033fb6027`,
covering 53,593 files / 53,591 objects and 17.84 GB of referenced content.
Independent verification checked the commit marker, every object's metadata,
semantic recovery dependencies, full hashes of the required control artifacts,
the ledger and exact journal prefix, the epoch anchor, and all 25 dependencies in
the policy → promotion → auxiliary → scheduling admission chain. This check did
not repeat exhaustive reads of every remote replay payload; the separate stopped
archive verification above did validate every archived file.

The replay cutoff was 22 minutes old at verification. This single observation
does not establish the plan's proposed percentile recovery-point objective.
The receipt is `evidence/postdeployment-disaster-snapshot-verification.json` under
the server evidence root. The regular backup timer continued running afterward.

## Qualified source-only follow-up

The [streaming replay validator](replay-validation-efficiency-20260923.md) reduced
median full-manifest validation time from 33.3028 to 29.4437 seconds (**11.59% less
time**) on 806,717 publications. This is a component measurement; it is not an
observed whole-startup or Elo/hour gain. Canonical key checks preserve join
cardinality, and malformed replacement schemas fail closed.

Main commits `e2ba105` and `2305545` add forward deployment recovery and the
qualified validator. Production runtime commit
`1ad289502b0a7aa6428b4f2750b81b58054fc68a` preserves the existing lineage.
All **4,068 target-host tests passed**, with no failures, errors or skips. Ruff and
Pyright passed; all 677 source hashes verified. All 103 package versions and the
native binary remained unchanged. The stopped-boundary CUDA smoke also passed.

The controller drained a successful disaster backup before stopping training.
That snapshot's catalog/commit marker and successful writer exit were checked:
`3055bc44e45085fef795fd948aa5f9b20d4a0692dfa5f530eabfc25504f95327`,
53,597 files, completed at 02:08:30 UTC. This receipt does not claim a new exhaustive
remote payload scan. The prior independent disaster verification and fully hashed
v2 local archive retain their separately stated scopes above.

The follow-up preserved **57,043 files**, including **56,356 replay shards**,
and resumed exact checkpoint **589,660** / **301,905,920 examples**. Checkpoint
SHA-256: `f3d1ed5962117fcaa0b3f9ff2cb43770a17fc7ca35de1e0a19903ef4b48b917e`
(213,775,175 bytes). No updates or examples were discarded. The v3 archive's
critical control artifacts and checkpoint were verified; a second exhaustive
replay/model payload rescan was not performed.

Migration evidence records `kind=source-only` and `changes=[]`. Exact profile bytes,
strength epoch and UTD accounting are unchanged. The saved 3,873,924-byte coordinator
journal prefix, 1,355.210442548 seconds of promotion service, 602.143260378 seconds
of measurement service, scheduling debt and pinned matchup survived. All durable
action prefixes and completed results survived for 152 promotion games and 32
measurement games.

The supervised rollout completed at **02:26:04 UTC**. The independent canary passed
at **02:26:46 UTC**, with learner step **589,783**, all ten workers healthy, zero
restarts and all eight GPUs healthy. GPU 7 was correctly leased to evaluation.
Monitoring and backup timers were restored; the first v3 disaster-backup process
started at 02:26:04 UTC with the correct release and profile. Its completion is not
claimed by this canary; durable formats did not change in this follow-up.

Current release:
`/home/ubuntu/edgeconnect-releases/variant-elo-efficiency-20260923-v3`.
Current profile:
`/home/ubuntu/edgeconnect-runs/variant-network/profile-elo-efficiency-runtime-20260923.yaml`.
It retains SHA-256 `d811d2246dd0364a5fcc9eba5dad7778ad7fb0600734f26014c6ae420da474b3`.
The sealed source archive SHA-256 is
`7e4fdbda06ddedf93c56723c59e88e2cb14a846511e9b4cfffe68f11430c14ce`.

The [machine-readable follow-up evidence](elo-efficiency-runtime-deployment-evidence-20260923.json)
contains qualification, controller, canary, preservation and scoped backup receipts.
The controller's source-only preparation enforces identical profile bytes and gives
its bounded recovery path a two-hour stop-post allowance. A subsequent test-only
commit improves the recovery fixture to publish a new immutable checkpoint; it
does not change the deployed runtime.

## Scope and remaining gates

This is the first validated engineering release, not completion of every research
option in the plan. The measurement cost pilot must finish before reliable current
Elo/hour and measurement capacity can be assessed. A component speedup must not be
reported as a measured Elo/hour gain.

Learning reuse, optimizer clocks, architecture, precision, search-budget and
dedicated-evaluator experiments remain unlaunched. They require the declared
campaign budget, complete measurement evidence and the plan's existing adoption
gates. No new hardware was provisioned. Later exact performance improvements are
recorded separately with their own validation and activation evidence.

A bounded CPU-only [backup header probe](backup-efficiency-next-benchmarks-20260923.md)
completed during the safe drain interval. It measured catalog parsing and retained
memory without changing runtime behavior. That evidence identifies the next backup
profiling priority; it is not an adopted backup optimization or a full-backup gain.
