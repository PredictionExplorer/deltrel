# Live replay publication and actor scheduling

This change lets the learner consume search policies while their games are still
running. Completed games subsequently enrich those same positions with final
labels. It also shares scheduling progress across GPU actors so restarts and
independent GPU schedules do not repeatedly postpone lower-weight board sizes.

The motivating production observation was 81.45% learner update-to-data waiting
in the 17:00–18:00 UTC window on September 10, 2026. Publication task age was
53.1 minutes at the median; that is a task-age measurement, not exact per-position
latency. These changes address avoidable delivery delay and native packing overhead.
They do not remove the need to generate new positions or establish an Elo gain.

## Configuration

All new behavior is opt-in. Existing profiles retain completed-game publication,
their existing scheduling, and a 300-second replay refresh interval. A live
publication profile uses:

```yaml
selfplay:
  record_fast_policy_targets: true
  policy_publication:
    enabled: true
    first_decisions: 8
    interval_decisions: 32
learner:
  replay_refresh_seconds: 60.0
orchestration:
  model_refresh:
    work_scheduling:
      enabled: true
      games_per_lease: null
      coverage_first: true
```

Merge these fields into the existing profile; this is not a complete profile.
Fast policy targets must already be enabled so every original decision has a
valid policy target. The preparation tool checks this prerequisite and shared
CUDA actor topology without changing unrelated controls:

```sh
python scripts/prepare_live_replay_profile.py \
  --source /path/to/current-profile.yaml \
  --output /path/to/new-live-replay-profile.yaml
```

The output must be a new path. The tool creates a read-only candidate and emits
its hashes and exact semantic changes. It does not activate the candidate or
change the update-to-data allowance. Use `--no-shared-scheduling` for profiles
without shared GPU actor inference. Configuration compatibility accepts omitted
new fields only when their values are the exact disabled defaults; it does not
hide an enabled feature from run/profile authority checks.

## Publication and accounting

`SelfPlayActor` publishes a full prefix after the first eight decisions of a
game, then after each additional 32 decisions. Each position retains its original
game ID and ply. `ReplayStore.append_game_revision` stores every revision as a
self-contained immutable NPZ, atomically advances the game's visible head, and
supersedes its previous revision. Selection exposes the latest eligible revision
once. An already-open loader can finish using its pinned older payload.

Pending rows contain the existing policy and soft-policy targets. Outcome,
score, spatial and teacher losses remain masked until their targets exist.
Finalization preserves the original state and search policy, supplies the
available final targets, and applies normal completed-game sample weighting.
Existing clinch/exact-endgame target rules still apply.

For example, an 8-row prefix, a 40-row prefix and a 45-row completed game produce:

| Publication | New-position credit | Enriched earlier positions | Physical rows written |
| --- | ---: | ---: | ---: |
| First prefix | 8 | 0 | 8 |
| Extended prefix | 32 | 0 | 40 |
| Completed game | 5 | 40 | 45 |
| Total | 45 | 40 | 93 |

Final enrichment with no prefix growth grants **zero** additional position
credit. Retries also grant zero credit. The existing update-to-data budget is
still computed from distinct committed positions, never the sum of physical
revision rows. Lowering the refresh interval does not increase that budget.

Full prefixes preserve compatibility with the replay loader and avoid a new
fragment-assembly data path. They cost extra serialization and storage: with
8/32 thresholds, a 275-position game writes 1,499 rows across ten revisions
(about 5.45 times its final size before garbage collection). Publication is
bounded by board size and configurable thresholds. Do not lower these thresholds
without measuring append time, disk traffic and actor throughput.

Clean shutdown flushes any unpublished tail under the same game ID and retains
already-published policies from incomplete games. It never fabricates outcomes
or gives the retained prefix a second abandoned-game identity. Unexpected
process failure can lose only the unpublished in-memory tail; committed
prefixes remain usable. This feature does not resume interrupted games.

## Freshness and failure handling

The learner observes position counts, publication revision and enrichment counts
from the same transaction as its selected replay snapshot. A freshness probe
can reopen the window for changed final labels even when position credit did
not increase. Probes also check that eligible selection actually changed and
has valid batch capacity, avoiding reopen loops for irrelevant publications.
An enrichment-only refresh still waits if the legitimate update allowance is
exhausted.

Real loader selection and its garbage-collection pin are installed atomically.
This prevents a superseded payload from being collected between selection and
loader opening. Pins are released on allocation/open failures and normal window
closure. Distributed selection/probe failures are broadcast to other ranks.

The store checks the actor lease, game/model identity, contiguous plies,
immutable state/policy prefix, payload integrity and finalization rules before
granting credit. Durable publication heads remain as tombstones after payload
collection, so an exact finalized retry cannot resurrect data or claim credit.
If a commit acknowledgement is lost, the store keeps the possibly committed
payload. The producer retries an I/O or SQLite operational failure once and
reconciles its local metrics only for that known uncertain invocation. Semantic
errors and stale leases fail immediately. A second operational failure is
surfaced rather than retried indefinitely.

## Shared work scheduling and native overhead

GPU actors share `status/work-schedule.json`, with short locked transactions and
atomic replacement. The ledger chooses a board size first, then a role within
that board; conditional mode choices retain their own durable progress. Each
positive-weight board receives initial coverage, charged against its real
weighted allowance. Restarting an actor retains this progress. Counts represent
assigned work, not completed games or equal neural cost.

With shared scheduling enabled, the coordinator schedules one lease at a time
instead of a four-cohort bundle. On the current server profile this reduces the
assignment quantum from 512 to 128 games. `games_per_lease: null` preserves the
existing 128-game task quota and rolling refills. Explicit quotas are validated
against actor batch size. Compatibility pooling and weighted long-run choices
remain in place; the scheduler does not force incompatible requests into one
GPU batch.

The native evaluation batch also delays the legacy Python state export until a
caller requests it. Cached feature packing borrows immutable rows instead of
deep-cloning them before packing. Public writable buffers remain independent.
These are exact representation changes; search budgets and policies are
unchanged. Local CPU probes measured 6–19% faster cached packing, not that gain
for complete search or production training.

## Activation, backup and rollback

Replay payloads remain version 5. New runtime code reads manifest versions 5
and 6. Ordinary appends leave a version-5 manifest unchanged; the first successful
game-revision commit upgrades its marker to version 6. Version 6 records logical
position credit and publication heads that older readers do not understand.

Deploy the schema-6-capable runtime to **all** replay readers, writers, preflight,
backup, restore and calibration tools before enabling publication. First retain
a validated rollback release that can read schema 6 with publication disabled.
Use the existing graceful checkpoint/drain and explicit profile migration
workflow to activate the prepared profile. Merely installing this code or
preparing a profile does not enable live publication.

After the first version-6 publication, rollback must use a schema-6-capable
runtime. Disabling the feature is supported; returning to a schema-5-only binary
against the upgraded manifest is not. Never reconstruct committed position
credit from physical shard row counts.

Manifest backup and disaster recovery preserve the publication tables,
counters, game tombstones and scheduler JSON. Superseded payloads can be omitted
from a new backup; their tombstones remain valid. The scheduler lock file is
ephemeral and excluded. Restore validates logical heads and credit without
requiring already-collected payloads.

## Validation and production measurement

The complete native-required Python suite passed with **2,508 passed and 12
skipped**. Focused publication/retry tests exercise all six game modes, clean
shutdown, retries before and after an actual commit, finalization, real learner
updates before any completed game, outcome enrichment, persistent loader
workers, selection/GC races and pin cleanup. Rust binding tests, strict workspace
Clippy, Ruff and Pyright also passed. Backup/restore and profile compatibility
have dedicated regression coverage.

Production activation and an after-activation performance window have not been
performed as part of this implementation. Compare matched steady-state windows
before attributing a throughput improvement:

- Distinct eligible rows/hour and their distribution by board, role and mode.
- Policy-first rows, enriched rows and completed-game rows; do not treat physical
  writes as newly generated data. Actor publication events expose
  `published_policy_first_samples`, `published_enriched_samples`,
  `published_written_samples` and cumulative equivalents.
- `pending_policy_rows`, which counts published positions still lacking final
  enrichment and includes retained incomplete games; it is not an active-game
  queue length.
- Learner update-to-data wait, device step time and replay refresh reasons,
  including enrichment with no new position credit.
- Actor replay append time/bytes, GPU inference throughput, loader overhead and
  disk growth. Faster visibility must not come at the expense of fresh-data
  production.
- Playing strength against fixed opponents per elapsed training hour. Earlier
  policies change the temporary target mix; throughput alone cannot establish
  an Elo improvement.

Some update-to-data waiting remains expected when search is the limiting
producer. This change removes avoidable publication and scheduling delay while
preserving the existing learning budget; it does not promise to eliminate the
observed 81% wait or deliver a 10× training gain.
