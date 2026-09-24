# Deltrel

A two-player strategy game of waterways, connected networks, and a shared shoreline.
Rules design by Ea Ea; Deltrel provides an original marine visual identity and notation. Human and AI play support classic and Double Deltrel.
Even games always use the pie rule; handicap games use the Full board without pie.

## The estuary

The responsive SVG board combines coastal contours, shallow-water gradients, curved
channels, and tactile clay/pearl markers. The central confluence preserves every
connection between the five innermost nodes. Decorative motion honors reduced
motion preferences and pauses when the page is hidden. All playable positions remain
keyboard accessible, with screen-reader labels and stable placement targets.

## Coordinates

Letters identify columns from left to right; numbers identify ranks from bottom
to top. For example, `G4` names the marked point inside column G and rank 4.
Coordinates appear only when you hover or keyboard-focus a point. On touchscreens,
press and hold to inspect without playing; a quick tap still places a stone.
There are no permanent axis labels or grid guides. The curved coast
leaves some cells empty; only marked points are playable.

| Board | Columns | Ranks |
| --- | --- | --- |
| Mini | A–K | 1–10 |
| Small | A–O | 1–15 |
| Medium | A–U | 1–19 |
| Full | A–Y | 1–24 |

Display notation has its own version. Numeric move IDs, saved games, AI models,
and the rules fingerprint remain unchanged when labels change.

## Features

- **Both variants**: classic Deltrel (one stone per turn) and Double Deltrel (two stones per
  turn; the first player places a single stone on the opening turn).
- **Four supported boards**: Mini (4 rings), Small (6), Medium (8), and
  Full (10). Other ring counts are rejected at every rules and API boundary.
- **Exact scoring, live**: shores (occupied and enclosed), the cape bonus for holding three
  or more corners, the ±2 × network-difference award, and the cape tie-break — recomputed
  after every stone with a union-find + flood-fill engine over a CSR adjacency (microseconds
  per evaluation, so the score panel is always current).
- **Complete games**: placement is mandatory until the board is full, with undo/redo,
  pie even games, Full-board handicap openings of up to nine stones, influence overlays, and a
  final score reveal.
- Games persist in `localStorage`, so a refresh resumes play.

## Deltrel AI

The repository includes a self-play training and inference stack for classic and Double
Deltrel on 4-, 6-, 8-, and 10-ring boards, played by one variant-capable network. The new
training policy targets 90% pie even games and 10% handicap games, with handicap confined
to the 10-ring board in both modes. Existing game histories retain their original rules.

- Rust rules, scoring, symmetry and Gumbel tree search, exposed to Python as
  `deltrel_native`;
- a PyTorch graph ResTNet with node policy, binary outcome, score, ownership, and
  alive heads;
- replay, learner, candidate/champion arenas and single-host 4/8-H100 orchestration;
- a private GPU `deltrelserve` backend and an ONNX + Rust/WASM browser runtime.

The site ships a verified browser export of champion step **566,428**, with its
trained EMA weights and all eleven network outputs. Browser play needs no private
AI service. This is not a claim of superhuman playing strength. See the
[training operator guide](training/README.md) and
[production H100 training runbook](training/docs/production-h100-training-runbook.md), then the
[target-host benchmark results](training/docs/h100-target-host-benchmark-results.md) and
[serving/distillation details](training/docs/serving-and-distillation.md).

### Playing in the browser

Each player has two choices: **Human** or **AI**. AI always runs in a background
browser worker using the full published champion. Choose AI for one player to
play against it, or for both players to watch self-play.

A fresh page automatically checks the current published manifest and prepares
the **72.5 MB FP32 model**, plus its browser runtime. Verified cached model bytes
are reused; a changed release downloads under a new content-addressed identity.
Download, integrity checking, and initialization show their progress and can be
cancelled or retried. Preparation never changes player selections or a saved game.
The prepared release remains pinned for the page session, including worker
restarts after pauses or idle disposal. Reload to pick up a newly published model.

WebGPU is used when supported, with a WebAssembly CPU fallback. Both execute the
same full FP32 model and requested search budget. No separate Python service,
application, account, or extension is required. Private browsing or cache eviction
can require another download; storage failure does not block play.

**AI strength** is available in setup and during play:

| Level | Simulations | Candidate moves |
| --- | ---: | ---: |
| Quick | 128 | 8 |
| Standard (default) | 512 | 16 |
| Deep | 4,096 | 64 |

Open **Custom search budget** to enter positive whole-number simulation and
candidate limits, then choose **Apply custom budget**. Custom values can exceed
Deep within native representation limits. Available legal moves bound the actual
candidate set. Settings are saved; changes apply to the next search while a search
already running keeps its original budget.

The browser uses the champion's score-margin utility, native seed contract,
Gumbel search constants, and pie-swap decision. The search value is the win/loss
value plus 0.05 times normalized expected score margin, clamped to [-1, 1].
The displayed win probability remains the model's unmodified outcome prediction.
FP32 export avoids half-precision conversion; numerical rounding across inference
backends can still change close decisions. See the
[release evidence](docs/browser-champion-566428-release.md).

All board sizes, both variants, pie swaps, handicap games, and AI-versus-AI play
are supported. Progress reports actual completed simulations. Slow searches can
continue beyond 90 seconds while progressing; 90 seconds without progress stops
a stalled engine. **Pause AI** remains available throughout. Larger budgets take
more time and memory, especially on phones.

Champion publication is **manual**: export and validate a confirmed checkpoint,
then publish its immutable ONNX/WASM assets and replace the manifest last. The
website automatically loads that published release; it does not poll the training
server or run a promotion schedule. Follow the
[model release instructions](training/README.md#export-and-publish-the-trained-browser-champion).
The native service remains a development/reference tool, outside public gameplay.

### Inspecting the engine

The **Engine estimate** panel shows each player's win probability and expected
final points during human–AI and AI–AI games. Forecasts update after each completed
search. The panel identifies the analyzed position and keeps the last completed
forecast visible while the next search runs.

Expand the panel's sections to inspect every search candidate, final-count
forecasts, and all network outputs: move and soft-move policies, win/loss and
score-margin distributions, ownership and living-network probabilities at every
point, future moves, and final shore, network, and cape counts. Tables include
every entry, with filtering, pagination, masks, and optional raw logits. The full
analysis is also available as downloadable JSON. The published AI exposes all eleven outputs. Older six-head browser
exports remain supported and clearly mark unavailable auxiliary outputs.

Use **Pause AI** to stop automatic play and **Analyze position** to inspect a
position without making a move, including positions selected from move history.
During automatic play, AI shows a progress bar with the actual completed
simulation count.
Use **Resume AI** to continue. The latest 64 analyzed positions are kept in memory
for the current game and matched to their exact history; an uncached position can
be analyzed again. Reloading the page restores the game, but not this analysis cache.

Win probabilities come from the outcome distribution. Expected points come from
the score-margin distribution and the board's fixed final total. The independent
count-based forecast is shown separately because those predictions can disagree.
See the [network-output contract](training/docs/network-output.md) for output
semantics and the optional serving API.

## Development

```bash
npm ci
npm run dev              # development server at http://localhost:3000
npm test                 # Vitest suite
npm run check:brand      # prevent retired names, symbols, paths, and coordinates
npm run test:coverage    # unit/component coverage with risk-weighted floors
npm run test:e2e         # production build + Chromium/Firefox/WebKit flows
npm run lint             # ESLint
npm run typecheck        # strict TypeScript
npm run build            # production Next.js build
npm run start            # serve the production build
npm run build:deltrel-wasm  # optional local-AI Rust/WASM artifacts
npm run test:browser-assets # verify packaged ONNX runtime deployment assets
node scripts/export-deltrel-conformance.mjs # regenerate conformance-v3.json
python training/scripts/export_feature_fixture.py # regenerate features-v4.json
```

Browser tests start their own production server and reject an occupied port;
use `PORT=3218 npm run test:e2e` when port 3000 is in use. They include real-model
Deep search and nine-stone AI-versus-AI openings in both variants. See the
[September 2026 gameplay audit](docs/gameplay-audit-2026-09-21.md) for findings
and validation scope.

The expensive Full-board Deep-search regression on Firefox is opt-in:
`DELTREL_SLOW_AI_TESTS=1 PORT=3218 npx playwright test e2e/slow-search.spec.ts --project=firefox`.
It uses the real model and may take several minutes on the CPU fallback.

CI also builds the native Python extension, runs pytest with per-module coverage
floors, Rust property and WASM contract suites, mutation jobs, dependency audits,
and a container smoke. CUDA, NCCL, and soak tests are separate hardware tiers; see
[testing and H100 validation](training/docs/testing-and-h100-validation.md).

The rules engine lives in `src/lib/deltrel/`:

- `board.ts` — pentagonal mesh generation, spatial letter/number coordinates (letters across, numbers upward), CSR adjacency, layout
- `scoring.ts` — the scoring engine (see the file header for the exact rule semantics)
- `game.ts` — turn protocol for both modes, handicap openings, the pie swap, and the
  replayable action log with retained placement history

Scoring is cross-validated against an independent reference implementation. Full supported
boards have no contested shores, totals sum to `5 × rings + 1`, and the odd nonzero margin
always identifies exactly one winner.

## Deploying to Vercel

```bash
npm i -g vercel
vercel         # preview
vercel --prod  # production
```

Human play and browser AI need no private service or Vercel environment variables.
The trained model and rules/search WASM are checked in. `npm run build` automatically
copies the matching ONNX runtime files from the locked dependency into public
assets, so Vercel does not need Python or Rust to build the site. Immutable model
and runtime files are cached; the current-model manifest is revalidated.

Optional server AI uses
same-origin Next.js routes at `/v2/move`, `/v2/analyze`, and `/v2/health`; configure
the deployment with server-only `DELTREL_AI_SERVER_URL` and, when enabled by `deltrelserve`,
`DELTREL_AI_BEARER_TOKEN`. Never expose the bearer token through a `NEXT_PUBLIC_*`
variable.

### Site identity

Set `DELTREL_SITE_URL` to the deployed site's absolute `https://` URL for social
preview links. On Vercel, the production project URL is used automatically when
that setting is absent. Local development falls back to `http://localhost:3000`
(or the configured `PORT`). The web app manifest, home-screen icon, and share
image use the Deltrel identity.

### Existing saves and trained models

Existing browser saves are copied into Deltrel's storage namespace and validated
by replay before use; a failed copy leaves the previous save intact. Coordinates
are display labels, so stored numeric moves retain their meaning.

Existing model checkpoints require the explicit, validated migration described in
[the Deltrel migration guide](training/docs/deltrel-rebrand.md). The migration
writes a new checkpoint while preserving learned tensors and training state.
Runtime loaders continue to reject incompatible contracts.
