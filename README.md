# Deltrel

A two-player strategy game of waterways, connected networks, and a shared shoreline.
Rules design by Ea Ea; Deltrel provides an original marine visual identity and notation. Human and AI play support classic and Double Deltrel.
Even games always use the pie rule; handicap games use the Full board without pie.

## The estuary

The responsive SVG board combines coastal contours, shallow-water gradients, curved
channels, and tactile clay/seafoam markers. The central confluence preserves every
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

The site ships a verified browser export of champion step **478,534**, with its
trained EMA weights and all eleven network outputs. Browser play needs no private
AI service. This is not a claim of superhuman playing strength. See the
[training operator guide](training/README.md) and
[production H100 training runbook](training/docs/production-h100-training-runbook.md), then the
[target-host benchmark results](training/docs/h100-target-host-benchmark-results.md) and
[serving/distillation details](training/docs/serving-and-distillation.md).

### Playing in the browser

Choose **Download browser AI** to prepare the current published champion, then
begin a game. The first preparation downloads the **37.6 MB model**, plus the
browser runtime. A progress bar shows model bytes received, followed by integrity
checking and engine initialization. Preparation can be cancelled and retried.
No application, extension, account, or other installation is needed.

Moves run in a background worker on the player's device. WebGPU is used when
supported, with a single-threaded WebAssembly CPU fallback. The browser checks
the latest published manifest when preparing and reuses a complete, SHA-256
verified cached model when available. Private browsing, storage limits, or browser
cache eviction can require another download; storage failure does not block play.
Refreshing a saved game asks the player to prepare the browser AI again, reusing
the cached model rather than silently downloading it on page load.

Browser AI strength is available in setup and during play: **Quick** uses eight
search simulations, **Balanced** uses 32, and **Deep** uses 64. Each level uses
the same trained model, within the published release's limits. Choices are saved
for future games. Changing strength keeps the current search running with its
original budget and applies the new setting to the next search.

The default is Quick to keep the full champion responsive. More search costs
more time, especially on phones. All board sizes,
both variants, pie swaps, handicap games, and AI-versus-AI play are supported.
Rules and learned weights match the published champion; the browser export uses
FP16 numerical precision. New trained champions must be exported, validated, and
published using the [model release instructions](training/docs/serving-and-distillation.md).

Downloading or preparing the engine preserves an existing AI-versus-AI setup.
Game status and scoring identify each controller explicitly; the automatic human
label “You” is displayed as an AI name when that side is controlled by an engine.

The browser reuses one root evaluation for search and inspection, avoiding a
duplicate model pass. See the [measured performance comparison](training/docs/browser-search-performance-20260921.md)
for timing results, exact-output checks, and a reproducible benchmark.

### Playing the full champion locally

The full champion can run on your Mac's GPU through `deltrelserve`. Use a verified
champion snapshot exported with the [local serving instructions](training/docs/serving-and-distillation.md#mac-local-champion-service),
then start its service:

```bash
training/.venv/bin/deltrelserve --config /absolute/path/to/snapshot/deltrelserve-mac.yaml
```

Set `DELTREL_AI_SERVER_URL=http://127.0.0.1:8080` in `.env.local` and run
`npm run dev -- --hostname 127.0.0.1 --port 3001`. Open
`http://127.0.0.1:3001`, choose the champion opponent, and select one of the two
variants and three openings. Thinking-time choices use the same full model;
larger search budgets take longer. The separate browser AI option requires a
published browser model; the current release includes one.

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
analysis is also available as downloadable JSON. Both the server champion and
the shipped browser champion expose all eleven outputs. Older six-head browser
exports remain supported and clearly mark unavailable auxiliary outputs.

Use **Pause AI** to stop automatic play and **Analyze position** to inspect a
position without making a move, including positions selected from move history.
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
