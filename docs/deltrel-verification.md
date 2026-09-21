# Deltrel release verification

Initial rebrand validated locally on September 20, 2026; coordinate inspection
and engine inspector follow-ups validated on September 21, 2026.

## Identity and presentation

- The current source, filenames, public assets, packages, configuration, fixtures,
  and deployment examples use Deltrel terminology.
- `npm run check:brand` prevents retired branding, coordinate notation, and visual
  symbols from returning to version-controlled source. Its own detection tests
  distinguish branding from unrelated programming terms.
- Production HTML, application bundles, and public text assets were scanned
  separately. All eight visual baselines were inspected and refreshed.
- Chrome, Firefox, and Safari checks cover branding, full-board spatial file/rank coordinates,
  keyboard controls, saved games, AI interactions, dialogs, and accessibility.
- Layout checks cover widths from 320 to 1440 pixels, with reduced-motion support
  and stable board geometry. Decorative motion pauses when the document is hidden.

## Rules and compatibility

The numerical conformance fingerprint remains
`4c4129690ca2390713c454c73acbef7c561da8dc7a2a9b536fbe377b4fbd60db`.
The automated golden-vector test verifies unchanged graph adjacency, scores,
turns, and complete game histories. Feature tensors retain their original numeric
values. Coordinates and named contract identifiers change; game rules do not.

Browser saves migrate by namespace fingerprint with strict replay validation.
Checkpoint migration is explicit and writes a new file; normal loaders remain
strict. An independent test built a real checkpoint using the original source,
then migrated and loaded it using the new source. All 64 model tensors, 64
optimizer states, 64 EMA tensors, and 64 adaptive-clipping histories matched,
including nonzero auxiliary loss settings and training progress.

Historical training archives are preserved unchanged. The earlier local teacher
checkpoint has a different rules/feature contract and was correctly rejected
without changing its checksum. See the
[migration guide](../training/docs/deltrel-rebrand.md) for supported artifacts and
deployment steps.

## Validation results

| Check | Result |
| --- | --- |
| Clean `npm ci` | Passed |
| `npm run check` | Branding audit, typecheck, lint, coverage, production build passed |
| TypeScript unit/component suite | 439 passed; 87.99% line coverage |
| Production Playwright suite | 110 passed across three browser engines and responsive layouts, including real browser champion inference; 2 native-touch injection cases skipped outside Chromium |
| Updated screenshot baselines | All 8 reviewed; subsequent comparisons passed |
| Native-enabled Python validation | 3,771 unique current cases passed; 17 hardware/soak skips |
| Rust engine, search, and native bridge | 100 passed; Clippy passed with warnings treated as errors |
| Node/WebAssembly contracts | 6 passed; browser runtime rebuilt |
| Python Ruff and typecheck | Passed |
| Engine inspector service/inference validation | 92 focused Python tests passed |
| Browser champion export and publication | 39 focused Python tests passed; 50 FP32 parity positions and 40 real-browser positions verified |
| Browser runtime deployment assets | 3 tests passed; full artifact integrity and actual WASM rules contract verified before production build |
| Dependency audit | Zero reported vulnerabilities after compatible updates |
| Conformance export | Byte-for-byte match with checked-in fixture |

The initial full Python run found one expected identity digest that needed the
new namespace. After its correction, all 28 related and migration cases passed;
the unique passing count above includes that repaired case and the new migration
tests. GPU training, multi-GPU operation, and long-running soak tests require the
target hardware and were not executed locally.

## Sandy coastline and live champion follow-up

The board now has a broad, independently shaped sand beach, fine grain, dune
shading, a wet shoreline, foam, and turquoise shallows. A shared view-box constant
keeps pointer placement aligned with the larger coast; resized and letterboxed
pointer tests pass. Mobile coordinate lettering was increased. Visual checks at
320px and 1280px confirmed the beach remains clear and the setup header stays
visible when the layout changes.

The unavailable engine was a still-running service reporting the earlier
identity. Its actual production checkpoint was outside the repository's archived
runs. A new copy of champion step 478,534 was migrated and its publication rebuilt,
preserving all 267 model tensors and 267 EMA tensors exactly. The detached local
service now runs on port 8082, and the app proxy reports it ready with the current
rules and feature identifiers. Real inference was verified across all four board
sizes, Classic, Double, pie, and handicap. Browser play also verified an opening
swap and a subsequent normal champion move. Startup and log locations are in the
[migration guide](../training/docs/deltrel-rebrand.md).

The original service and checkpoint remain intact. No external deployment or
training run was changed.

## Spatial coordinate follow-up

Coordinates now use letters from left to right and numbered ranks from bottom to
top. Every node falls inside exactly one named map cell on all four board sizes;
the Full board spans A–Y and ranks 1–24. The board keeps its shoreline uncluttered: no permanent axis labels or grid
guides are drawn. A single coordinate badge appears beside a hovered or
keyboard-focused point. On touchscreens, holding a point for inspection suppresses
placement on release; an ordinary quick tap still places a stone. Movement and
pointer cancellation stop the inspection, so scrolling cannot turn into a move.
Empty cells remain unplayable.

The independently versioned `deltrel.board-notation.v2` contract uses identical
integer arithmetic in TypeScript, Rust, and Python. All 610 node coordinates
were checked for uniqueness, spatial order, cell containment, and round trips.
Invalid cells are rejected; lowercase and surrounding whitespace are accepted
consistently. Move history and engine candidate labels use the same board mapping.

The immutable rules-v3 fingerprint and model feature fingerprints did not change.
Existing saves retain numeric move IDs. Before/after topology and feature tensor
byte comparisons are identical, and the live champion remained connected without
a restart, checkpoint conversion, or model change. Focused backend verification
passed 37 Rust and 81 Python/native tests, plus Clippy, Ruff, and type checks.
The updated frontend suite and all three browser engines passed as listed above.

Touch inspection additionally covers short taps, pen input, held-point release,
scroll movement, cancellation, empty coastline, timer cleanup, compatibility
mouse/focus events, and returning to keyboard navigation. A real Chromium touch
sequence verified that holding displays a coordinate without making a move and
that the next ordinary tap still places exactly one stone. Ordinary touch taps
were verified in Chromium, Firefox, and WebKit. The held-touch injection is skipped
in the other two engines because that test uses Chromium's native input protocol.

## Engine inspector follow-up

Both players' win probabilities and expected final points are visible during
human–AI and AI–AI play. Expandable sections expose every root search candidate,
all eleven champion network outputs, full distributions and masks, raw logits,
count forecasts, model/search diagnostics, and a complete JSON export. Browser
models expose their six exported base outputs and mark the auxiliary outputs
unavailable. Probabilities come from the outcome head; score-margin and count-based
point estimates are presented separately.

Each estimate is bound to its analyzed position, history branch, configuration,
and player perspective. The previous result stays visible during the next search,
with an explicit earlier-position label. A bounded in-memory history supports
review, and a read-only analysis request can fill missing positions. Pause/resume,
undo/redo, branching, rematches, errors, and late replies have regression coverage.
Browser-engine inspection requires pausing automatic play so its shared worker
cannot cancel a playing request.

The service's optional root-output flag preserves existing API callers. Output
collection adds one root-only forward pass, with no changes to search policy,
selected actions, leaf caches, or model weights. All output shapes, masks,
activations, finite values, and normalization are validated. A narrow fallback
supports older services that reject the new flag. Responses are bounded to 1 MiB;
the real Full-board champion response was approximately 164 KB.

Focused Python service/inference validation passed 92 tests, including legacy
responses, malformed outputs, model locks, and checkpoint compatibility. The live
champion at step 478,534 produced all eleven outputs across every board size,
Classic, Double, pie, and handicap. A production-browser AI–AI match completed;
another match verified pause, read-only reanalysis with unchanged occupancy, full
network output, and resume. Responsive inspection at 320px had no horizontal
overflow, and the new browser tests include accessibility checks.

The local app now uses the updated service on port 8082. The superseded port-8081
service was stopped after verification. The original port-8080 service and all
checkpoint archives remain unchanged.

## Public browser champion release

The repository now includes a direct FP16 export of champion step 478,534:
`deltrel-champion-478534-1aa623983d22.fp16.onnx` (37,577,312 bytes), with all eleven
trained heads. Its SHA-256 is
`1aa623983d22f33844690b0ede5e5372743f893612ee0f92e9bd762b489e33f5`.
The source checkpoint stays outside public assets. Export metadata records the
original checkpoint identity, training step, and parity results without local
filesystem paths.

Fifty real positions cover every board size, both variants, handicap, pie swaps,
and near-full boards. Against the original FP32 EMA model, the largest probability
difference was 0.002048 (about 0.205 percentage points), and the largest expected
margin difference was 0.00786 points. Dynamic batch inference also passed. A real
GPU-disabled Chromium run loaded the FP16 model using single-threaded WebAssembly
and evaluated all four sizes; no GPU, cross-origin isolation, application, or
extension installation is required.

The public setup offers an explicit browser AI download with model bytes,
percentage, verification, initialization, cancel, and retry states. A page visit
only checks availability. Saved games require explicit preparation after reload.
Verified models are cached by content digest; corrupted, oversized, or incomplete
downloads are rejected, and storage denial or quota failure still allows play
from memory. Cancellation and stale worker messages are covered by regression
tests. The latest manifest is revalidated when preparing; games use a consistent
loaded model between explicit preparations.

The full champion defaults to eight simulations and four considered candidates,
with published limits of 64 and eight. Older saved settings are clamped to those
limits. WebGPU is attempted first and WebAssembly provides the CPU fallback.
Inference and search run in a worker, and full auxiliary forecasts retain the same
player-perspective and applicability checks as the server inspector.

A production build with no server AI configured served the real browser model,
showed all eleven outputs, and played a Full-board AI–AI game through 71 placements
and a pie swap. Pause and reload preserved that position; explicit preparation
reused its cached model. The existing trained server and private checkpoints are
unchanged.

Vercel builds copy the exact locked ONNX runtime files automatically. The model
and rules/search WASM are tracked in Git. Every production build verifies the
model's manifest, SHA-256, byte count, and actual WASM rules contract before
compilation. Production responses were checked for correct immutable caching on
versioned model/runtime files and revalidation on the mutable manifest. Public
browser play requires no private service or deployment environment variables.

Real-model browser tests cover Chromium, Firefox, and WebKit without mocking
inference: no model GET before a user action, local replies, all eleven heads,
and no private move/analyze requests. Chromium and Firefox also verify cache
reuse with model downloads blocked. The installed WebKit test browser evicts
CacheStorage on reload even on a static SVG page without app JavaScript; its
test verifies explicit retry and real inference after that eviction instead of
assuming storage persistence that the browser does not provide.
