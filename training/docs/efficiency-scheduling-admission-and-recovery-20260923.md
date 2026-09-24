# Scheduling admission and recovery — September 23, 2026

The scheduling transition enables a 20% share of existing evaluation service
for independent measurements, a 3,600-second wait bound checked at lease
boundaries, and the history eligibility horizon with a 3,600-second initial
estimate. These are the only four profile fields permitted by the transition.
Model, losses, optimizer, targets, search budgets, promotion statistics, actor
resource allocation and inference execution remain covered by the source
profile. No throughput or playing-strength claim is added.

`prepare_efficiency_scheduling_gate.py` publishes an immutable outer receipt.
It pins the exact source profile and its configuration-named gate, requires the
target to equal the declared transform, and revalidates all original reports
and inherited policy, promotion and auxiliary receipts. Unknown changes,
same-byte source aliases, recursive scheduling receipts and rewritten report
inheritance fail validation. Exact disabled defaults preserve old configuration
hashes; enabled/nondefault settings retain distinct authority.
`ensure_scheduling_gate` can finish an interrupted publication only by reusing
the identical immutable receipt and revalidating the entire source chain.

Disaster snapshots copy the atomic measurement service ledger first, followed
by a bounded, complete-record prefix of the append-only coordinator journal.
The ledger's byte cursor and SHA-256 witness must match that prefix. Snapshot
verification requires the journal, the pinned matchup's immutable manifests
and checkpoints, and an explicitly referenced strength-epoch anchor. Journal
bytes are not path-rewritten during relocation because the cursor witnesses
those exact bytes.

After restoring and relocating metadata in its private staging directory,
restore writes a one-use receipt bound to the final ledger bytes and captured
journal. The scheduler first consumes any captured release events exactly
once. If the old host disappeared during a lease with no captured release,
the receipt permits removing only those named unresolved tokens: settled
totals, service debt and the pinned job remain intact; the unfinished interval
receives no invented GPU credit and accounting is marked incomplete. The
receipt remains as audit evidence and its consumption is recorded atomically.
Old snapshots without measurement scheduling remain readable.

Local stopped-run preservation also copies the coordinator journal byte for
byte when a measurement ledger exists. Before publishing the archive it checks
the witnessed prefix and projects ledger recovery in a private scratch copy.
An unsettled lease without terminal evidence prevents publication. Offline
verification repeats that check and requires the pinned matchup and epoch
anchor manifests and checkpoints in the inventory. This covers later
deployments, when the scheduler already has service debt and an active job.

During graceful shutdown, the coordinator records the lease's terminal event
only after the exact arena owner process group has been reaped. The initial
shutdown-request event earns no credit. A previously released lease is not
recorded twice, and an owner whose group survives does not receive a fabricated
release. Thus ordinary restart can consume the final journal suffix without
requiring a disaster-restore exception.
