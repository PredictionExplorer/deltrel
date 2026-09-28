# Raw versus EMA diagnostic

`scripts.compare_checkpoint_averaging` compares raw and EMA weights from one
recovery checkpoint against the EMA weights of one frozen champion. The four
ring-10 cells are classic with pie, double with pie, classic handicap, and double
handicap. Both arms use identical opening/seat/search seeds and fixed search
effort. Handicaps cycle through 2, 4, 6, and 9 stones.

Freeze the input checkpoints once, while the named immutable recovery artifact
still exists. Run from the training directory with the training environment:

```bash
python -m scripts.compare_checkpoint_averaging \
  --source-run-root /training/run \
  --checkpoint /training/run/learner/recovery/checkpoint.pt \
  --champion-checkpoint /training/run/learner/models/champion.pt \
  --output-dir /training/diagnostics/averaging-20260928 \
  --pairs-per-cell 8 --simulations 256 --plan-only
```

The example checkpoint names must be replaced by the immutable artifact paths
resolved from the recovery and champion pointers. Do not pass a pointer JSON as
a checkpoint. Output must be outside the source run. The plan pins both copies,
the search contract, precision, and evaluation implementation hashes; moving
live pointers or source garbage collection cannot change a resumed diagnostic.
The script does not write to the source run, publish models, or produce
promotion decisions.

Reserve one GPU for the entire session using the training service's ownership
mechanism or by stopping its workers gracefully. The command checks the GPU is
idle before loading models; the caller must prevent new workers from claiming
it until the command exits. Address it with its complete UUID from
`nvidia-smi --query-gpu=uuid --format=csv,noheader`:

```bash
CUDA_VISIBLE_DEVICES=GPU-REPLACE-WITH-FULL-UUID \
python -m scripts.compare_checkpoint_averaging \
  --output-dir /training/diagnostics/averaging-20260928 \
  --device cuda:0 --exclusive-device --session-seconds 300
```

Each invocation runs one arm. Repeating this command alternates unfinished arms
by session count. `--arm raw` or `--arm ema` explicitly chooses one. The time
limit requests a graceful stop; an in-flight model load/search and durable
checkpoint write may finish after the deadline. SIGTERM and SIGINT request the
same stop. Existing arena resume support preserves completed games and searched
moves, and resumes an interrupted search from its current move.

`summary.json` reports each arm against the common champion, per-cell paired
confidence intervals, and the raw-minus-EMA score on matching completed pairs.
The difference is descriptive, not a significance test. Partial handicap cycles
and small completed samples warrant particular caution. `terminal: true` in the
summary means both fixed budgets finished. No result automatically changes EMA
decay or training settings. The default is 64 games per arm (128 total); increase
the predeclared budget in a new output directory if the completed result is too
uncertain.

Resume only with runtime flags (`--output-dir`, device ownership, arm, session
duration). Input paths and search settings cannot be replaced on a resumed plan.
Use the same evaluation implementation for all sessions; a code hash mismatch
requires finishing with the original release or starting a new diagnostic.
