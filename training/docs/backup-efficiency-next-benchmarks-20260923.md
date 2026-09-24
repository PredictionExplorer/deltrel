# Next bounded backup benchmarks

This is a read-only source audit and bounded metadata probe, not a complete-backup
attribution or an adopted runtime change.
The first new-profile disaster backup completed in 22 minutes 4 seconds. A late
process sample observed approximately 148 GB of cumulative logical reads and
38.8 GiB RSS, with zero physical `read_bytes`. These observations do not establish
an NFS or storage bottleneck. Training remained active during capture.

The final streaming publication validator is qualified separately; its 11.59%
component improvement must not be extrapolated into a whole-backup speedup.

## 1. Historical catalog retention and repeated traversal

`_snapshot_header` retains both the immutable JSON payload and a separately parsed
catalog. Cache hits still rehash document bytes. With existing run/global latest
pointers and one run directory, ordinary creation can traverse historical headers
seven times across pointer validation and enumeration. This is the strongest
source-level hypothesis for the observed large resident set.

Instrument unique headers, retained catalog entries, traversal count, bytes
rehashed, JSON parsing/freezing time and RSS around each traversal. The first
benchmark uses fresh CPU-only processes with 8 and 32 preserved headers,
at most 512 MiB serialized input and a three-minute cap each. Two unchanged
validation passes distinguish retention growth from repeat hashing/parsing cost.

That bounded probe completed at 02:02:53 UTC on September 24, before the rollout's
workload stop. Both processes exited normally using unchanged deployed v2 helpers,
one CPU, lowest scheduling priority, CUDA hidden, a 180-second per-process bound,
8 GiB address-space limit and 7 GiB RSS watchdog. No replay/model payloads were read.

| Preserved headers | Serialized input | Catalog entries retained | First-pass wall / CPU | Second-pass wall / CPU | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 82.05 MB | 422,639 | 6.048 / 5.995 s | 0.124 / 0.114 s | 0.841 GiB |
| 32 | 325.74 MB | 1,677,859 | 23.851 / 23.751 s | 0.491 / 0.446 s | 1.713 GiB |

Logical reads were approximately twice the serialized size on the first pass and
once on the second. Retained memory above the imported-runtime baseline was
3.9–4.1 times serialized header size. These observations support CPU-intensive
initial parsing and substantial catalog retention. The existing backup overlapped
the probe; the result does not quantify their share of the full backup duration or
establish a storage bottleneck. Full per-case observations and source identity are
in the [benchmark receipt](backup-header-retention-benchmark-20260923.json).

References: `_snapshot_header`, `_resolve_latest`,
`_snapshot_headers`, and `create_snapshot` in
[training_disaster_recovery.py](../scripts/training_disaster_recovery.py).

## 2. Repeated replay database verification

A normal successful CLI snapshot validates publications during the live-source
backup check, copied-backup check, staged snapshot verification and final CLI
report verification: four passes. It also performs three complete SQLite
integrity checks. `full_objects=False` still performs database and semantic checks.
Retries can add more work.

Measure exclusive CPU/wall time and logical reads for each SQLite integrity check,
publication validation, ready-shard reconciliation and complete verification pass.
Record capture-attempt counts and the exact changed fence fields. Run the existing
document verifier twice in fresh processes against one preserved snapshot, capped
at five minutes each, with phase instrumentation. No verification pass should be
removed merely because a prior phase succeeded: first prove that the same immutable
bytes and dependency closure are still being verified at the required boundary.

References: [replay_manifest_backup.py](../scripts/replay_manifest_backup.py),
`_validate_replay_database`, and final CLI verification in
[training_disaster_recovery.py](../scripts/training_disaster_recovery.py).

## 3. Non-replay objects copied before deduplication

`_store_object` reads, hashes, writes and fsyncs a temporary object before checking
whether the destination already exists, then hashes that destination. Replay
shards have a separate reuse path. Model-manifest loading can introduce another
checkpoint hash before copying.

Measure bytes/time by artifact kind, already-present destinations, temporary bytes
written, fsync time and checksum bytes per digest. A bounded next benchmark can
capture at most 20 preserved artifacts totaling 2 GiB into a private scratch store,
then repeat capture using unchanged validation. Keep source checksums, destination
verification, publication ordering, race handling and retention semantics intact.

Do not reintroduce timestamp-only or notification-only trust. Qualification already
proved that same-stat writes and pre-existing writable mmap changes can evade such
shortcuts on the production host. Any eventual implementation needs corruption and
interruption tests plus a representative complete-operation benchmark.
