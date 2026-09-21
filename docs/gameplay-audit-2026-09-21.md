# Gameplay and AI audit — September 21, 2026

## Scope and reproduction

Reviewed the web application's setup, controllers, AI preparation and search,
game transitions, move history, persistence, scoring contracts, inspection UI,
and server request boundaries. Verification uses the production Next.js build,
the checked-in champion step 478,534, the shipped WASM engine, and the available
local MPS champion service. The supplied Vercel preview required authentication,
so its exact deployed revision was not independently inspected.

### Maximum strength

The current browser release's Deep setting performs **64 simulations**, considering
up to **8 root candidates**. Automated real-model browser checks inspect the returned
analysis and require exactly 64 total root visits. Shipped-WASM tests also verify
8, 32, and 64 visits, that larger budgets evaluate additional positions, and that
cached repeats preserve the complete search while reusing neural evaluations.

A real Mini-board request through the local application's server proxy used the online
champion's Maximum preset: **4,096 simulations / 64 candidates**, returned **4,096
root visits**, and took **40.9 seconds** on the available MPS service. Wall-clock
speed is not evidence of search depth: cached evaluations and terminal
continuations legitimately need less inference.

The inspector now distinguishes the neural **Search input value** from the
**Searched root value**. Browser search has no score-utility adjustment, so its
input value equals the model value by design. The searched value incorporates
simulated continuations; changing those API semantics would be incorrect.

A separate Firefox CPU reproduction exposed a real timeout defect: Full-board
Deep search was interrupted by the client's fixed 90-second deadline before it
could move. The default deadline is now an **inactivity watchdog**: only strictly
increasing completed simulations for the matching request and stable budget can
renew it. Stale, duplicate, regressing, or preparation messages cannot renew it.
Explicit caller-supplied deadlines remain absolute. The UI displays actual search
progress and ignores updates from old or paused requests. Initial setup/root
evaluation and each interval without completed simulations still have a 90-second
limit; this is not an unlimited timeout.

The same real Firefox Full-board Deep case completed after the fix in **455.0
seconds**, reporting **64 simulations and 64 root visits**. Its first visible
progress update was one completed simulation. This intentionally slow regression
is retained as `e2e/slow-search.spec.ts`, enabled with `DELTREL_SLOW_AI_TESTS=1`
and `--project=firefox`; it is excluded from routine runs by default. These
timings describe the tested CPU fallback, not all browsers or devices.

### Nine-stone handicap

The reported failure was not reproduced in the current rules or worker pipeline.
Real-model AI-versus-AI tests complete all nine first-player placements and the
opponent's reply in both Classic and Double on the Full board. Additional
component tests exercise every request and turn transition. The rules/WASM
compatibility tests cover historical configurations on every supported board;
new handicap games remain restricted to the Full board by product policy.

Confirmed adjacent failures were fixed: waiting for an unprepared engine appeared
as thinking, handicap instructions incorrectly described ordinary two-stone
turns, and a stale active response could leave automatic play silently stalled.

## Fixes

- Neutral Player 1 / Player 2 quick-start names and migration of the legacy
  You / Champion pair; custom names and AI-versus-AI controller selections survive
  preparation and champion reselection.
- Accurate preparation, unavailable-engine, retry, and handicap status; keyboard
  navigation for handicap choices.
- Retryable recovery for interrupted or stale active AI responses. Paused games
  reject late responses, and rapid human clicks cannot play the AI's turn.
- Real, accessible search progress and a progress-based default timeout so slow
  healthy searches are not discarded at 90 seconds; strict caller deadlines,
  cancellation, stalled-engine detection, and budget validation remain enforced.
- Store-level legality validation, copied actions, protection for completed games
  and setup state, consistent redo/rematch state, and bounded saved-history parsing.
- Browser-storage errors no longer throw out of a legal move. Persistence remains
  best-effort if the browser denies writes; refreshing cannot restore an unwritten game.
- Deep follows a release's actual published maximum. Re-preparation notices changed
  search/output metadata even when model weights are unchanged. Partial tensor
  allocation failures release already-created inputs.
- Bounded streaming of capability responses, cleanup of rejected response bodies,
  correct cancellation/timeout classification, and rejection of upstream redirects
  by the fixed-target proxy.
- Browser tests start their own production server, preventing a stale or unrelated
  localhost application from being silently tested. Real-model scenarios run
  sequentially within each browser to avoid competing CPU inference workloads.

## Validation

Added property tests include 100 complete randomized games with an independently calculated turn
schedule, 80 randomized edit/save/reload histories, and 40 seeded WASM comparisons
between standard and session search. Existing independent scoring, symmetry,
completion-bound, and conformance tests remain in place.

| Check | Result |
| --- | --- |
| `npm run check` | Branding/assets, TypeScript, ESLint, coverage gates, model/WASM integrity, and production build passed |
| Unit/component/integration tests | 539 passed in 42 files; 89.68% line and 88.35% branch coverage |
| Production browser suite | 116 passed across Chromium, Firefox, WebKit, and responsive layouts; 2 native-touch injection cases skipped outside Chromium |
| Core game mutation tests | 94.44% score: 152 killed, 1 timed out, 9 survived, none uncovered; configured gate passed |
| `cargo test --workspace --locked` | 101 passed |
| Native-enabled Python serving/search/conformance tests | 123 passed |
| Real Firefox Full-board Deep regression | Passed with all 64 visits in 455.0 seconds; the original 90-second cutoff was reproduced before the fix |
| `npm audit` | No reported vulnerabilities, including development dependencies |

The surviving game mutants include redundant typed-array bounds checks and
defensive copies; the mutation score is reported without suppressing them. The
single updated visual baseline was inspected and reflects the warm-pearl palette
change committed separately while this audit was in progress.

The browser suite completed both entire nine-stone AI-versus-AI openings on all
three engines. Firefox's Classic and Double openings took 8.4 and 9.0 minutes on
this CPU fallback, respectively, so the tests allow an appropriate whole-opening
budget while still failing promptly on an engine error. The separately validated
slow Deep test is opt-in; it was excluded from the routine suite run above.

No training weights, rule fingerprints, numeric move IDs, or saved-game schema
were changed. GPU training, multi-GPU orchestration, and long-running soak tests
were outside this local gameplay audit. Automated tests and review reduce risk;
they do not establish the absence of all possible defects.
