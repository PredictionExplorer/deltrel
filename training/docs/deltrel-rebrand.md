# Deltrel identity and model migration

Deltrel retains every node id, edge, turn rule, scoring calculation, symmetry,
feature position, action slot, and numeric feature value. Display coordinates use an independent spatial notation contract: letters run
left to right and numbers bottom to top, with the full board spanning A–Y and
ranks 1–24. The rules-v3 canonical bytes retain their historical label clause
only for compatibility; it is not the current display notation. Label updates
do not require model migration or restarting a serving champion.
Shoreline nodes, capes, and established waterway networks replace the earlier
terminology. The checked-in cross-language vectors preserve all numeric gameplay
values, including every full-game state, score, and graph adjacency.

The canonical rules fingerprint is `fnv1a64:46e4fbcff4e17fd3`; production feature
schema v4 is `058eb071d77948a7`. Identity changes intentionally invalidate old
manifests and protocol messages even though the tensor layout is unchanged.
Normal loaders do not silently accept earlier identities.

## Existing production checkpoints

Migrate a copied, stopped production checkpoint explicitly, from `training/`:

```sh
python -m deltreltrain.rebrand /absolute/source.pt /absolute/deltrel.pt
```

The converter accepts only the immediately preceding production rules and
feature fingerprints (`a5d932b0ef8354e8`, `cb0e1e89a6ce3540`). It preserves every
tensor, optimizer moment, EMA weight, scheduler value, and training step. The
three auxiliary output heads are identified by their unique output dimensions
(102 shoreline logits, 52 network logits, 12 cape logits) and receive the new
parameter names; their loss settings and gradient-clipping histories receive the
same names. Persisted gradient-clipping format identifiers are updated without
resetting adaptive history. Optimizer parameter order is preserved and its routing checksum
is verified before conversion and recomputed afterward. The output records the
migration and must pass the normal checkpoint validator before atomic publication.
Neither input nor an existing output is overwritten.

Models without auxiliary heads can migrate their default disabled count losses.
A headless checkpoint with nonzero earlier count-loss settings is rejected
because the source contains no output shapes to identify those settings safely.

This command migrates checkpoint identity only. Replay shards, checkpoint
manifests, champion pointers, old run directories, and historical rollout evidence
are not rewritten. Preserve those artifacts separately. Use a new Deltrel run
directory and regenerate checkpoint/champion manifests so their content hashes
refer to the converted checkpoint. Re-export browser models and publish a new
manifest with the current rules and feature identifiers. Rebuild the native and
WebAssembly packages from the renamed crates. Do not edit a signed or checksummed
model manifest in place. Older feature-v3 teacher checkpoints are outside this
converter's scope and remain rejected until separately converted and validated
for lineage transfer.

Deployment entry points are `deltreltrain-*` and `deltrelserve`, the native module
is `deltrel_native`, and environment variables use `DELTREL` or `DELTRELTRAIN`.
Update service units, filesystem paths, environment settings, and external model
URLs together with the application release. A deployed service with an old rules
fingerprint is incompatible until upgraded; strict compatibility checks remain.

## Restoring an already running local engine

A serving snapshot can live outside `training/runs`. Before selecting an archived
checkpoint, inspect the configured upstream's `/v2/health` response and the exact
configuration used by its running process. A healthy service can still be rejected
by the Deltrel client when that process was started before the identity migration.
Changing source files does not update a running Python process or its loaded model.

The verified local Deltrel snapshot is
`~/.local/share/deltrel/champion-478534-identity-v1/`. Its champion remains at step
478,534, with all 267 model tensors and 267 EMA tensors preserved exactly. The new
content-addressed checkpoint and champion pointer have current schema identifiers;
`identity-migration.json` records their relationship to the original publication.
The original snapshot is retained unchanged.

From the repository root, start the migrated service with:

```sh
training/.venv/bin/deltrelserve --config "$HOME/.local/share/deltrel/champion-478534-identity-v1/deltrelserve-mac.yaml"
```

This serves the trained champion on MPS at `http://127.0.0.1:8082`. Configure
`.env.local` with `DELTREL_AI_SERVER_URL=http://127.0.0.1:8082` and restart the web
server so it reads that setting. The same-origin `/v2/health` route must report
`ready: true`, rules fingerprint `fnv1a64:46e4fbcff4e17fd3`, and feature fingerprint
`058eb071d77948a7`. A command started in a terminal needs to be started again after
the machine restarts; this repository does not install a login service implicitly.

The current local instance runs as a detached process. Its PID is recorded in
`~/.local/share/deltrel/champion-478534-identity-v1/deltrelserve.pid`, and its log is
`~/.local/share/deltrel/champion-478534-identity-v1/logs/deltrelserve-network-output-8082.log`. To stop
that instance safely, read the PID file, confirm `ps -p <PID> -o pid=,args=` names
this snapshot's `deltrelserve-mac.yaml`, then send `kill -TERM <PID>` to that exact
process. The previous service on port 8080 and its original snapshot were retained
unchanged during the cutover.
