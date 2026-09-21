# Backup retirement hardening deployment — September 15, 2026

The change fixes the race between a frozen replay ledger and subsequent removal
of an older shard. It also protects the local ledger copy from backup rotation,
closes database readers explicitly, detects replay database replacement, and
validates snapshot documents before publication.

The deployed backup-tool commit is
`c796ba9a61e208e2ad2a263b63dc427e856a38e5`, staged at
`/home/ubuntu/deltrel-releases/backup-retirement-20260915`.
The release contains 634 tracked source files verified against the source
archive. Its 103 dependency versions and native binary are identical to the
running training release. Both local and server suites passed 134 tests with
no failures or skips. Formatting, Ruff, and full-project Pyright also passed
locally.

The deployment changes only
`deltrel-deltreltrain-variant-network-disaster-backup.service`.
Training and the other support services remain on
`variant-adaptive-promotion-20260914-v3`, runtime commit
`4f6027c50117eee8b66b59dfebbef7257d47fe12`.
The active profile remains `profile-adaptive-promotion-20260914-v3.yaml`,
SHA256 `9eefd9dcad3c9ba35a956d383e1ca24ef7b711e54108a81caa81bea687e1660a`.

The controller captures source/profile authority, training PID/start time,
worker restarts, UTD segment, strength epoch, and original unit/timer state.
It pauses the backup timer in a supervised service and lets the current backup
finish. It then atomically installs the prepared unit, starts a new backup,
and restores the original timer state. An ExecStopPost recovery path restores
the original unit and timer if activation is interrupted before its completion
marker. Neither recovery nor activation stops the trainer or an active backup.
Five controller fault simulations covered an old failed backup, install failure,
reload failure, wait timeout, and interruption after unit replacement.

Before activation, the new tool completed an end-to-end full checksum
verification of an existing offsite snapshot: 48,289 catalog files,
23,519,417,578 bytes, SHA256
`1da0cb7de4580011b5a31dc371ed386fc4116aaf5ad150808983349e345edd3c`.
Verification took approximately 157 seconds.

Operational scripts and original unit bytes are retained at
`/home/ubuntu/deltrel-rollouts/backup-retirement-20260915`.
The frozen deployment-controller SHA256 is
`9166e1efe272a78bcfbe9791f4a55635a9ffce2c5a254632bc8ce2e0e29e32b3`.

The previous backup completed successfully at 17:04:06 UTC. The new backup
started immediately afterward as PID 3081437. A deployment-only check then
misinterpreted `systemctl`'s refreshed ExecStart runtime annotations as a change
to the trainer command. Automatic recovery restored the old unit configuration
and active timer, while leaving the new backup process and trainer running.

The corrected controller compares the configured executable/arguments separately
from PID, start time, and restart count. Seven targeted regression tests passed,
including rejection of actual command and process changes. Its frozen SHA256 is
`2d7b967ad83153b2c2689edc3a241b28ab20fa912434dd21e98a54457ba10766`.

At 17:08:42 UTC the corrected controller reinstated the new unit configuration
without restarting the already-running new backup. Before and after this
operation it verified the process's exact command, working directory, Python
import path, cgroup, boot ID, and start ticks. The backup retained PID 3081437
and start ticks 255886406. The timer is active and enabled. Initial error and
recovery evidence remains in the rollout's `first-activation-attempt` directory.

At this point the trainer retained PID 2918356, all ten workers had zero
restarts, and learner progress had advanced to step 322034. The profile,
profile authority, source identity, UTD segment, and strength epoch matched
the pre-deployment record.

The first new backup completed successfully at 17:14:32 UTC. Its catalog
contains 47,992 files totaling 23,449,434,331 bytes, with snapshot SHA256
`772e9c9cac335124d66febabb2221a32e592f8ab20385639e5af96b3c14a8787`.
The immutable commit marker and matching latest pointer were checked. A
separate end-to-end verification reread and hashed every catalog payload,
finishing successfully at 17:17:17 UTC after approximately 105 seconds.
The successful service report came from the verified new process, PID 3081437.

The final audit at 17:18:39 UTC found the same trainer process, all ten workers
healthy with zero restarts, and learner step 322251 versus 321773 before the
cutover. All recorded training authority/accounting hashes remained unchanged.
The restored timer had already started its next regular backup at 17:18:31 UTC
using the new release. The rollout issued no replay shard deletions, replay
resets, trainer stops, or checkpoint rewrites.

The failed temporary deployment unit's journal was preserved before clearing
only that unit's failed status. The recovered production backup service and
its schedule are active. This release is qualified for backup integrity and
continuity; it does not constitute a measurement of Elo/hour.

[Structured deployment evidence](backup-retirement-deployment-evidence-20260915.json)
records source hashes, tests, the recovered controller error, process identity,
unchanged training state, and both full backup verifications.
[Design and failure handling](backup-retirement-safety.md) describes the replay
retirement and publication contracts.
