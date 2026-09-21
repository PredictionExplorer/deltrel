# Deltrel release verification

Validated locally on September 20, 2026.

## Identity and presentation

- The current source, filenames, public assets, packages, configuration, fixtures,
  and deployment examples use Deltrel terminology.
- `npm run check:brand` prevents retired branding, coordinate notation, and visual
  symbols from returning to version-controlled source. Its own detection tests
  distinguish branding from unrelated programming terms.
- Production HTML, application bundles, and public text assets were scanned
  separately. All eight visual baselines were inspected and refreshed.
- Chrome, Firefox, and Safari checks cover branding, full-board A–JO coordinates,
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
| TypeScript unit/component suite | 276 passed; 86.12% line coverage |
| Production Playwright suite | 97 passed across three browser engines and responsive layouts |
| Updated screenshot baselines | All 8 reviewed; subsequent comparisons passed |
| Native-enabled Python validation | 3,771 unique current cases passed; 17 hardware/soak skips |
| Rust engine, search, and native bridge | 100 passed; Clippy passed with warnings treated as errors |
| Node/WebAssembly contracts | 6 passed; browser runtime rebuilt |
| Python Ruff and typecheck | Passed |
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
service now runs on port 8081, and the app proxy reports it ready with the current
rules and feature identifiers. Real inference was verified across all four board
sizes, Classic, Double, pie, and handicap. Browser play also verified an opening
swap and a subsequent normal champion move. Startup and log locations are in the
[migration guide](../training/docs/deltrel-rebrand.md).

The original service and checkpoint remain intact. No external deployment or
training run was changed.
