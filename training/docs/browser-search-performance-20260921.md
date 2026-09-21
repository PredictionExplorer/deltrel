# Browser search performance — September 21, 2026

Removing the duplicate root model forward reduced fresh, prepared CPU/WASM search
latency by about 10% at the 8-visit setting on this machine. An immediate repeat
of the exact same position benefits much more from the new one-entry root report
cache. These are separate workloads; the repeat figures do not describe every
move in an ordinary game.

## Deployment-like comparison

Three paired repetitions per case, with no COOP/COEP headers, matching the deployed
application. Times are median worker round trips after the model has been prepared.
Model download and initialization are excluded and recorded separately in the
[evidence](browser-search-performance-20260921.json).

| Board and position | Workload | Before | After | Reduction |
| --- | --- | ---: | ---: | ---: |
| Mini, 15 placements, 8 visits / 4 candidates | Fresh runtime | 575.5 ms | 520.9 ms | 9.5% |
| Mini, same settings | Immediate identical-root repeat | 56.3 ms | 1.0 ms | 98.2% |
| Full, 82 placements, 8 visits / 4 candidates | Fresh runtime | 3,248.3 ms | 2,921.3 ms | 10.1% |
| Full, same settings | Immediate identical-root repeat | 321.9 ms | 1.9 ms | 99.4% |

The runtime is prepared in a new worker before each fresh sample, so its neural
prediction and root-report caches begin empty. The repeat runs immediately in
that same worker with the same position and budget. During actual play, a new
position may already have a cached leaf evaluation; its benefit can consequently
be smaller than the fresh-runtime improvement above. Repeats also require the
worker to remain alive.

## Higher-budget controlled comparison

An additional three-repetition run used cross-origin-isolation headers for both
builds and one WASM inference thread. This provides a controlled relative
comparison at 32 visits; its absolute timings are not presented as measurements
of the deployment's nonisolated configuration. The evidence also retains its
8-visit control cases.

| Board and position | Workload | Before | After | Reduction |
| --- | --- | ---: | ---: | ---: |
| Mini, 15 placements, 32 visits / 8 candidates | Fresh runtime | 1,885.6 ms | 1,838.9 ms | 2.5% |
| Mini, same settings | Immediate identical-root repeat | 56.2 ms | 1.5 ms | 97.4% |
| Full, 82 placements, 32 visits / 8 candidates | Fresh runtime | 11,298.3 ms | 10,953.2 ms | 3.1% |
| Full, same settings | Immediate identical-root repeat | 339.0 ms | 4.0 ms | 98.8% |

The smaller percentage gain at a larger budget is expected: removing one root
forward saves roughly one model evaluation while preserving all search visits.

## What was held constant

- Baseline source: commit `f14f186`, built from an isolated archive.
- Candidate: that same archive with only `src/workers/deltrel-ai.worker.ts`
  replaced by the optimized worker. The candidate worker source SHA-256 is
  `368cb47484c4ebb9fd8aae81937470c0d9a86b73a3e8c41dfef78a966b9c3d07`.
- Model: the real 37,577,312-byte champion-478534 FP16 artifact, SHA-256
  `1aa623983d22f33844690b0ede5e5372743f893612ee0f92e9bd762b489e33f5`.
- Actual built Next.js worker bundles, game WASM, ONNX Runtime, and worker
  `prepare`/`choose` messages. No substitute evaluator or search implementation.
- Fixed legal conformance positions, deterministic native search seeds, identical
  requested budgets, and one WASM inference thread.
- Apple M4 Max, 16 logical CPUs, macOS/arm64, headless Chromium 149.0.7827.55.
  Baseline/candidate execution order alternated between repetitions. Other heavy
  tests and browser AI activity were paused during the measurements.

Every result was parsed through the production decision validator. Native root
visits equaled the requested budget. Canonical SHA-256 comparisons of complete
decisions, excluding only request ids and timing fields, matched exactly across
baseline/candidate, all repetitions, and fresh/repeated requests. This covers
chosen actions, root visits/policies/Q values, outcomes, score estimates,
auxiliary predictions, and all eleven network output heads. The improvement was
not obtained by reducing search effort or changing the model.

The headless WebGPU probe returned **no available adapter**. There are therefore
no WebGPU timing claims. The harness records both adapter capabilities and the
backend actually selected by the worker; a WebGPU-to-WASM fallback is not reported
as GPU performance. These measurements do not establish playing strength or Elo.

## Reproduce

Install the repository's development dependencies and Playwright Chromium first.
Build both source trees before running the benchmark. To isolate the worker change:

```sh
repository_root="$PWD"
baseline_dir="$(mktemp -d)"
candidate_dir="$(mktemp -d)"
git archive f14f186 | tar -x -C "$baseline_dir"
git archive f14f186 | tar -x -C "$candidate_dir"
ln -s "$repository_root/node_modules" "$baseline_dir/node_modules"
ln -s "$repository_root/node_modules" "$candidate_dir/node_modules"
cp src/workers/deltrel-ai.worker.ts "$candidate_dir/src/workers/deltrel-ai.worker.ts"
npm --prefix "$baseline_dir" run build
npm --prefix "$candidate_dir" run build
node training/scripts/benchmark_browser_search.mjs \
  --baseline "$baseline_dir" --candidate "$candidate_dir" \
  --baseline-revision f14f186 --candidate-revision working-worker \
  --repetitions 3 --budgets 8,32 --providers wasm,webgpu \
  --isolation none --output /tmp/deltrel-browser-search-benchmark.json
```

Use `--isolation coi` to reproduce the isolated control. The harness serves only
temporary test pages and existing build assets on ephemeral localhost ports; it
adds no shipped route. Model hashes, worker hashes, actual backends, each sample,
preparation timing, visit counts, semantic digests, and medians are recorded.
A partially completed run retains `complete: false` and should not be used as a
finished comparison. The worker's neural computations are untouched; CPU mode
is selected only by hiding WebGPU from the test worker before loading its bundle.
