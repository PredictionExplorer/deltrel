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
- a private GPU `deltrelserve` backend and a distilled ONNX + Rust/WASM browser runtime.

No trained model is checked in. Server and local AI choices remain unavailable until an
operator trains a champion or publishes a distilled browser model. This is implemented
training infrastructure, not a claim of superhuman playing strength. See the
[training operator guide](training/README.md) and
[production H100 training runbook](training/docs/production-h100-training-runbook.md), then the
[target-host benchmark results](training/docs/h100-target-host-benchmark-results.md) and
[serving/distillation details](training/docs/serving-and-distillation.md).

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
published lightweight model.

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
node scripts/export-deltrel-conformance.mjs # regenerate conformance-v3.json
python training/scripts/export_feature_fixture.py # regenerate features-v4.json
```

CI also builds the native Python extension, runs pytest with per-module coverage
floors, Rust property and WASM contract suites, mutation jobs, dependency audits,
and a container smoke. CUDA, NCCL, and soak tests are separate hardware tiers; see
[testing and H100 validation](training/docs/testing-and-h100-validation.md).

The rules engine lives in `src/lib/deltrel/`:

- `board.ts` — pentagonal mesh generation, sequential letter coordinates (A–Z, AA onward), CSR adjacency, layout
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

Human play and published local-AI assets need no private service. Server AI uses
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
