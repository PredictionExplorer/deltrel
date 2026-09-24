# Cooperative GPU handoff diagnostics — September 23, 2026

Read-only production inspection found a failed GPU-7 lease requested at
20:33:42 UTC and cancelled at 20:49:11 UTC after the configured 930-second
readiness deadline. The actor resumed with the same PID; the arena exited with
transient code 75, restarted, and acquired its next lease about 25 seconds after
requesting it. Coordinator events establish a wait for actor quiescence, but
the detailed five-second telemetry from that incident had already rotated.
The historical blocking resource is therefore **not established**.

Code inspection found a separate reproducible control-loop hazard: the actor's
main thread reads compatible-work metrics under the work coordinator's lock,
which also spans model creation. A slow model operation can therefore block
pause polling and heartbeat progress merely because telemetry wants that lock.
The control loop now requests a nonblocking snapshot and explicitly reports
`snapshot_available: false` while the coordinator is busy. Work acquisition,
model pinning, and outcome accounting retain their existing synchronization.

The regression integration test holds that lock while requesting a pause and
requires parking, matching-token release, and shutdown to remain responsive.
Injecting the former blocking read makes the same test fail its five-second
pause deadline; the corrected implementation passes. This reproduces the code
hazard, not necessarily the historical incident.

Pending handoffs now expose the wait stage, its start time, parked/unparked
cohort identifiers, and sparse locations of blocked Python frames. Locations
contain only file basenames, line numbers and function names; locals, tensors
and absolute paths are not captured. The inference broker distinguishes batch
collection, device-lock wait and inference, and reports oldest request age.
Coordinator events retain the observations on stage changes and at most once
per 30 seconds while unchanged, so an incident remains diagnosable after monitor
rotation. Diagnostic fields do not participate in readiness authorization.

Readiness still requires the matching worker PID and lease token, every live
producer parked, no queued or owned inference jobs, synchronized CUDA, and a
fresh actor heartbeat. Cancellation and release still require the coordinator's
ownership proof; this change neither shortens nor extends the timeout, signals
the actor with SIGSTOP, nor discards an unfinished self-play trajectory.

The service-accounting integration also requires every successful release,
including actor restart, to publish a canonical durable journal event before
the released acknowledgement becomes visible. This closes the zero-cooldown
race where the next lease could be registered before the preceding interval was
journaled. Owner-validated cancellation before the coordinator's first poll
now settles its never-admitted token exactly once without parking an actor or
recording GPU service. Unknown owners remain rejected. Neither correction
invents a ready or release timestamp for a lost host.
