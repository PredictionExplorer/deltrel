# Controlled graph-cache activation — September 12, 2026

**Deployed and verified.** The 32-entry cache is live. Final verification observed
step **214,099**, exact resumption from checkpoint **213,452**, zero worker
restarts or inference failures, restored monitoring/backups, and a successfully
verified backup of the new configuration. The [evidence ledger](graph-cache-controlled-activation-evidence-20260912.json)
retains source identities, checkpoint and snapshot hashes, test scopes and observed
production results.

The user requested enabling the larger cache if it improves performance and deploying it gracefully. The target changes only `orchestration.model_refresh.inference.cuda_graph_max_entries` from 16 to 32. The 48 GiB process allowance and 8 GiB per-model allocation remain unchanged. The first two efficiency improvements stay enabled.

## Why the deployment is justified as a controlled trial

Both correctly configured H100 studies showed about 39–41% faster inference in synthetic two-board traces, with identical responses, actor math and memory limits. Those are not fleet throughput gains. The earlier strict and bounded policies remain failed; their report bytes and classifications have not been altered.

The new live evidence found recurring capacity pressure on an actual model serving ring 8 and ring 10. During 06:27:48–06:37:43 UTC, GPU 2 recorded 329 graph captures and evictions. A fresh pre-deployment window, 07:17:58–07:27:58 UTC, found the mixed workload on GPU 5 instead, with 261 captures and evictions. Most other workers used a single board size and needed little or no recapture.

The expected fleet benefit under this mix is modest—a fraction of a percent in avoided work, rather than 40%. Its exact wall-time and Elo effects are not established. The observed workload moves between GPUs, so the post-deployment comparison follows model/board context rather than a fixed GPU number.

A separately pinned controlled-activation receipt records the actual user request, rationale, original failed reports and live evidence. It preserves the original search-quality checks and verifies all original output, math, memory, owner and timing accounting. Both two-board median gains must remain above 5%, and the original conservative full-target production ratio multiplied by the worst observed execution factor (capped at one) must remain at least one for each frozen model. The observed products are 1.0037 and 1.0091. This arithmetic is an engineering safeguard, not a statistical lower bound for the live workload.

## Deployment and recovery

The code release is `b21e120418f41ca5402ff4dfb034dd96396e9f14`, with 559 source files. All 103 packages, the native binary and the actor/model/inference/learner/arena execution files match the previous production release. Only admission and backup evidence handling changed in runtime code.

The target profile is `profile-cache-activation-20260912.yaml`, canonical configuration `7f6617e3e1c31ad3f89cb63cc91e9b587333b2907bf6bee95d8599beb942f3e3`. The former profile/source remain available for pre-migration recovery; after migration, rollback uses the new reader and restores the 16-entry setting while preserving the checkpoint.

Preparation verified 12.54 GB across 27,885 immutable backup objects while training remained active. The scoped stopped-snapshot helper reuses only recent, unchanged identities on the same host and mount. It fully checks new, changed or unproven objects before publishing a new snapshot. Normal periodic backups and restores retain their ordinary verification paths. The fresh snapshot is bound to the exact stopped learner pointer and unchanged profile/source authority.

The graceful stop saved checkpoint **213,452**, SHA256 `60dc55f59c7d9ed50365b91a862d5aaaa1b6728efb3bc9d3e945810a19f96f01`, with **109,287,424** consumed examples. The H100 smoke and new stopped-checkpoint snapshot passed. Migration recorded zero discarded learner steps and exactly one configuration change. The new runtime resumed that exact checkpoint and passed sustained readiness at approximately 07:56:50 UTC. The subsequent backup, full integrity verification and warmed production observations completed successfully.

The stopped snapshot took 899.5 seconds. It reused 4.93 GB without upload and recorded 78,083 successful verification-cache hits, while fully hashing 1.78 GB of other objects. The full preparation/snapshot process still required substantial time; this change does not establish that checkpoint hashing was the dominant remaining backup cost.

## Validation

The final source passed **196** targeted server tests covering original and controlled admission, corrupt/incomplete evidence, backup capture and restore. The temporary deployment/recovery bundle passed **70** tests on the server, including exact resumed-checkpoint identity, backup invalidation/full fallback and corrupt-object rejection before snapshot publication. These are separate scopes. Ruff and the configured Pyright checks passed.

The post-activation plan requires sustained worker readiness, exact checkpoint resumption, advancing learning, unchanged memory bounds and an observed cache/capture comparison after warmup. Changes in models, board mix, pauses and counter lifetimes must remain visible in the report; no component metric is presented as Elo/hour.

## Warmed production observation

The 07:57:04–08:07:09 UTC window contains 122 observations across 605 seconds, with stable worker PIDs and no restarts, inference failures, graph fallbacks, graph-validation failures or monitor errors. Across the actor processes there were **zero graph evictions**. The 52 new captures corresponded to initial cache fills; 46 occurred when GPU 7's actor started after its normal arena pause.

One current model on GPU 2 retained **32 graphs for rings 6 and 8**, using **5.57 GiB**. This verifies that the additional capacity is used within the unchanged 8 GiB per-model limit. The largest sampled actor-process graph allocation was **21.85 GiB**, below the unchanged 48 GiB allowance.

The preceding 16-entry window had 264 evictions, and useful neural-row throughput rose from 17,927 to 24,078 rows/second in the later window. These are **observed counts and rates, not a causal cache-speedup measurement**: boards, models, game phases and arena occupancy changed. The verified post window had a 6/8 model, with no corresponding 6/10 or 8/10 case matching the earlier benchmarked workload. Stale publication context from earlier process generations was excluded from matched-workload claims. No fleet throughput or Elo/hour improvement is quantified from this comparison.
