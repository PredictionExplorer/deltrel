# Twelve-hour champion recovery screen

This is a coupled recovery experiment, not a claim of improved Elo or a
one-factor optimizer comparison. It restarts raw and EMA weights from the
retained champion EMA, with fresh optimizer moments and an explicitly selected
learning rate. It keeps the 17.5M model and the existing game objective.

The profile uses champion-only self-play, reuse 1.5, a fresh shard watermark,
and zero initial replay credit. Old rejected-branch positions cannot enter as
the new branch catches up in step count. The conservative search allocation is
65% fast / 35% full, 32/384 simulations at reference ring 6 (scaled to ring 10),
without the old per-ring reduced-full-search waivers. This deliberately raises
search quality and does not carry old qualification receipts into corrected pie
search. The normal search-admission validator still runs.

Even pie games explicitly use zero asymmetric PDA. The keep/swap identity
assumes zero PDA; the corrected actor enforces it too. Handicap PDA is unchanged.

GPU 0 remains the learner, GPUs 1–6 generate games, and GPU 7 is a dedicated
evaluator. Promotion keeps the same 256-simulation game/search/statistical
contract. Independent measurements use 1024 simulations and a new epoch anchored
at the starting champion. Old arenas and measurement queues are archived within
the fork. No parent replay, checkpoint, profile or model pointer is modified.

The learning-rate CLI parameters and warmup length are required. The scheduler
stays at the selected rates after warmup, and automatic plateau resets are
disabled for the fixed screen. Select rates from calibration; do not copy the
old reference rates into a fresh scheduler.

## Prepare after stopping the source

Run these commands from the tested release's training directory. Preparation
pins the exact source profile, replay database, champion pointer/manifest/bytes
and implementation files. Output and destination must both be outside the
source and outside each other. Existing output is never overwritten.

```sh
python -m scripts.prepare_strength_recovery prepare \
  --source-profile /source/profile.yaml \
  --destination /runs/strength-recovery \
  --output /experiments/strength-recovery-inputs \
  --muon-lr CALIBRATED_MUON_RATE --adamw-lr CALIBRATED_ADAMW_RATE \
  --warmup-steps SELECTED_WARMUP
python -m scripts.prepare_strength_recovery apply \
  --plan /experiments/strength-recovery-inputs/strength-recovery-plan.json
```

Application uses the existing isolated fork and champion-warm-start tools. If
application fails after the fork has been copied, preserve that directory for
inspection; the command does not overwrite or silently rebuild an existing
fork. Qualification and backups of the new root precede starting it.

The fork includes an independent copy of every pinned source, profile and
implementation input under `strength-recovery-provenance/`. Its catalog is bound
to the experiment plan. Restart verifies implementation bytes in the running
release; disappearing preparation directories cannot redirect local inputs.
Keep the full tested release and native extension with deployment backups too.

Disaster and stopped-run backups include this provenance and every completed
timed snapshot's receipt, immutable manifest and checkpoint. Independent backup
verification checks the complete reference chain and payload hashes. A corrupt
dependency prevents publishing a new backup. Restore these experiments to their
original run path: hashed profiles, endpoint times and provenance are not
silently rewritten by profile relocation. Exact-root restore and repeated backup
are covered by integration tests.

## Run and resume

```sh
python -m scripts.run_strength_recovery --run-root /runs/strength-recovery
```

Use an owned service or process group for this command. The proven ablation
runner starts and stops its coordinator/process group, requests graceful
checkpointing, and escalates within the configured teardown grace. Its
twelve-hour wall clock includes startup and restart downtime. Teardown time and
confirmed resource release are reported separately; they remain part of any
provisioned-cost calculation. Budget completion cannot start another training
attempt. Transient restarts retain the original deadline. A signal does not
leave workers running.

The learner publishes candidates at the first optimizer safe point after 2, 6,
and 12 elapsed hours, independent of updates/hour. The ordinary example cadence
is disabled by a distant threshold. Shutdown's final publication can capture
the twelve-hour endpoint. Each captured manifest and checkpoint is hard-linked
under `strength-recovery-snapshots/<scheduled-seconds>/`, preserving it against
normal retention. The receipt records the actual capture time. If a restart
misses multiple endpoints, only the latest elapsed endpoint is captured and
earlier endpoints are explicitly reported missing. The tools never label one
late checkpoint as multiple historical snapshots.

## Continue automatically after the screen

To keep the qualified lineage training without a desktop connection, prepare
the continuation controller before starting the screen:

```sh
python -m scripts.run_strength_recovery_continuation prepare \
  --run-root /runs/strength-recovery --source-commit VERIFIED_RELEASE_COMMIT
python -m scripts.run_strength_recovery_continuation run \
  --run-root /runs/strength-recovery
```

The controller owns the screen runner and the later orchestrator. Run it under
one server service with `KillMode=control-group`, `Restart=on-failure`, and
`RestartPreventExitStatus=78`. A retryable screen crash returns 75 and retains
its original deadline. Fatal validation/worker errors return 78. A live GPU
owner blocks startup; stop other experiments before assigning all eight GPUs.

At clean twelve-hour completion it verifies checkpoint integrity and confirmed
worker release, requires the retained twelve-hour endpoint, and seals the screen
metadata. It then uses the existing profile migrator to change only publication
cadence to three million examples. Rates, optimizer state, EMA, search, teacher
source and champion remain unchanged. A narrowly scoped migration exception
admits this exact completed-screen continuation; other continuous-profile
validation is unchanged. An already-applied migration is verified and reused
after restart. A partial or conflicting handoff fails closed.

The continuation runs until an operator stop or a fatal failure, with its own
clock and process-attempt accounting in `strength-continuation-state.json`.
Completed screen metadata is never restarted or extended, and later updates
cannot fabricate missing screen snapshots. A signal checkpoints and releases
the owned process group. The controller retries only a bounded number of
transient exits; it never falls back to older search code.

Configure monitoring and backups to read the active profile filename from
`profile.sha256` at invocation time. The migration installs
`profile-strength-continuation.yaml`; a service hard-coded to the screen profile
will have the wrong active checksum after handoff. Both original and continuation
profiles are retained by the recovery backup dependency closure.

## Independently evaluate a frozen endpoint

Use this after training stops or after reserving a separate idle GPU for the
whole evaluation session. GPU 7 is owned by promotion while training runs; an
external diagnostic must not overlap that owner. The tool requires one full
GPU UUID and checks that no compute processes own the device before loading.

```sh
python -m scripts.evaluate_strength_recovery prepare \
  --run-root /runs/strength-recovery --snapshot-seconds 43200 \
  --output /experiments/recovery-12h-256 \
  --simulations 256 --pairs-per-cell 16 --wall-budget-hours 8
CUDA_VISIBLE_DEVICES=GPU-FULL-UUID \
python -m scripts.evaluate_strength_recovery run \
  --output /experiments/recovery-12h-256 --exclusive-device \
  --device cuda:0 --session-seconds 300
```

Repeat the `run` command to resume unfinished games. A plan pins its checkpoints,
search settings, implementations and original champion; subsequent champion
promotion cannot redirect it. It uses the production 64 considered actions,
complete reversed-seat pairs, and four game categories. No promotion pointer
is written. A new 1024-simulation plan uses the same endpoint and anchor but
separate results. Four pairs per cell is a 32-game pilot; sixteen is 128 games.
Match both budgets and anchors when comparing a control.

The endpoint wall budget starts on first execution, persists separately from
session state, and includes downtime between sessions. Budget exhaustion returns
incomplete evidence and cannot be mistaken for a strength verdict. It does not
grant a fresh budget on retry. A session ends at a safe search boundary, so
model loading and an in-flight search may overrun its requested slice. Run the
session under the host's owned service timeout if a hard external kill bound is
required; already persisted action histories remain resumable.
