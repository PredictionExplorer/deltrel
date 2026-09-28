# Protected champion replay

The ordinary model-step lag limit can exclude a retained champion even when its
newly generated self-play remains stronger than the current learner. The opt-in
exception preserves that teacher while keeping a majority of current replay.

The learner settings are `protected_champion_fraction` (default `0.0`, maximum
`0.5`), `protected_champion_max_age_seconds` (default `21600.0`), and
`protected_champion_after_ns` (default `null`). Enabling a positive fraction
requires an explicit, fixed activation timestamp. A 0.25 fraction admits at most
one protected row for every three ordinary rows in each selected ring/segment/mode
cell. Missing protected data is filled with ordinary data; the learner never
waits for the protected share to fill. A shortage of ordinary data limits the
protected allocation instead of allowing it to take over the window.

The current champion pointer is verified at selection time and must belong to
the same run and generation family. The exception requires the exact champion
identity and original model step; every other model retains the existing lag
limit. Promotion automatically replaces the protected identity. A champion
within the normal model-step window needs no exception and follows the existing
sampling policy.

Protected rows must come from games first published after activation and within
the rolling age window. Replay stores the first publication time in additive,
nullable manifest columns. New games record it once, and later prefix revisions
and final-label enrichment retain that original time. Historical records with
unknown origin remain NULL and cannot use the exception. No historical rows are
relabelled, model steps are never advanced artificially, and selection does not
alter cumulative unique-position credit. The existing publication ledger grants
credit only for new logical positions.

Actors retain their configured candidate/champion/history proportions once the
exception is active, including when the champion has exceeded the ordinary lag
window. The replay cap is independent of actor selection, so configured teacher
generation share is not a promise of an exact learner share. Ring, game-category,
and classic/double quotas continue to apply.

The source publication schema and NPZ payload format are unchanged. The new
metadata columns are preserved by SQLite backups; upgraded legacy databases do
not backfill unknown timestamps. Validation checks publication/shard origin
agreement during its existing single ledger scan. Disabled configuration fields
are omitted from serialized profile authority to preserve existing hashes.

Actor commit telemetry reports ordinary eligibility separately. Rows from an
aged champion awaiting the protected age/cap checks are marked
`protected_pending_selection`; their old aggregate eligibility boolean/counts
are unknown, rather than incorrectly declaring them unusable. This does not
claim they were consumed. Learner replay-window diagnostics record the exact
selected source counts, protected model identity/step, per-cell fractions, and
cap check. The monitor exposes that allocation with its age independently of
the latest training-loss record. Its scope is the selected window, not the
number of optimizer updates that have consumed it.
