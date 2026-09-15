# Replay retirement during disaster recovery capture

Replay cleanup commits removal of a shard's ledger row before unlinking its
immutable file. A consistent SQLite backup can therefore legitimately refer to
a file that cleanup removes before the offsite backup opens it. The September
15 failures followed this ordering; the affected paths had already disappeared
from the live ledger.

The backup now handles that ordering explicitly:

1. Capture the online SQLite backup and its checksum while holding the existing
   local backup-service lock. Keep that lock until the ledger is copied into the
   offsite content-addressed object store. This lock does not prevent training
   writes or hold a SQLite transaction during the offsite copy.
2. Enumerate ready shards from the archived ledger bytes. Local backup rotation
   can safely proceed after the ledger copy completes.
3. Copy each immutable shard normally. Once its source descriptor is open,
   unlinking the source name cannot invalidate the bytes being copied.
4. If inspection or opening reports that this particular source file is
   missing, check the live ledger, including its WAL. Any row with the original
   ID or path makes this an integrity failure. Ready, superseded, quarantined,
   and conflicting rows are never silently skipped.
5. For confirmed retirement, reuse a previously archived object only after
   verifying its immutable-file metadata and complete SHA-256 checksum. The
   original archived ledger and its full shard catalog remain intact.
6. If no archived object exists, restart the entire capture with a fresh ledger.
   Previously copied objects remain available for reuse. At most four capture
   attempts are allowed; exhaustion leaves the last successful backup intact.

Destination I/O errors, unsafe paths, checksum mismatches, and missing files
still registered in the live ledger remain fatal. Retirement recovery never
edits the archived ledger, invents training credit, or drops a referenced shard
from a catalog. The state fence also detects replacement of the live manifest
and changes to the restore marker without treating ordinary WAL writes as a
reason to retry.

Snapshot documents are fully validated in a private staging file before an
exclusive, atomic publication into the snapshot namespace. Validation failure
or interruption cannot expose an invalid final document. Publication then
updates per-run latest, global latest, and the commit marker in that order.
Existing snapshots, commit markers, latest pointers, and source training data
are preserved. Database readers explicitly close their connections on success
and failure.

The regression suite exercises real replay cleanup at deterministic capture
boundaries, missing active files, corrupt and unsafe artifacts, retry
exhaustion, cancellation, local retention, and a concurrent training writer
during a blocked backup copy. This is an operational reliability change; it
does not change training allocation, search budgets, promotion, or Elo scoring.
