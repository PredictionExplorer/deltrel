# Browser champion 566,428 — FP32 release

This release makes browser inference the public game's only AI controller.
The model is the full confirmed champion, with original EMA weights and all
eleven prediction heads. It is not distilled or quantized.

| Artifact | Verified identity |
| --- | --- |
| Training champion step | 566,428 |
| Original training checkpoint SHA-256 | `876b5a7bf3efe843d0c3ae26b5ea14a358980759d73a1f9f7ed2c490c03478be` |
| Losslessly migrated checkpoint SHA-256 | `47a8e20edb462330bd7d2877e6a2fc3de15f3c0d9d88c7e4c6406c854521d42b` |
| Browser artifact | `deltrel-champion-566428-20f52f268869.fp32.onnx` |
| Browser SHA-256 | `20f52f268869396de096ce23419ca071c69a43d0f5002e4ee2e03d984255a8bb` |
| Browser bytes | 72,474,137 |
| Search/rules package | `wasm-46e4fbcff4e17fd3-champion-v1` |

The source pointer had role `champion` and a completed, conclusive `promote`
result. Identity migration preserved all 267 model tensors, 267 EMA tensors,
and the original checkpoint. Private checkpoint files remain outside the public
release and Git.

FP32 export compared fifty legal positions against the original EMA across all
supported sizes and variants. Maximum probability error was 1.41e-6; maximum
expected-margin error was 2.55e-6 points. The manifest records the full per-head
validation results. These are numerical validation measurements, not a playing
strength or speed benchmark.

The browser uses the native champion seed derivation and search core. Its search
utility is `clamp(P(win) - P(loss) + 0.05 * E[margin] / 151, -1, 1)`; displayed
outcome probabilities remain unmodified. Standard uses 512 simulations and 16
candidates. Quick uses 128/8, Deep 4,096/64, and custom budgets remain supported.
Search execution defaults to one first-visit row and no subtree reuse, matching
the reference service configuration. Native and browser backends can still
round floating-point computations differently.

A fresh page checks the published manifest, downloads or verifies the identified
model, and pins the release for that page session. Cancellation and idle worker
disposal retain the pinned identity. A newer release is picked up on reload.
Champion export and publication remain manual; no training poller, promotion
watcher, or scheduled deployment is installed.

Automated validation covers export integrity, rules/search contracts, feature and
seed parity, browser worker behavior, and UI/persistence regressions. Interactive
gameplay and performance testing are reserved for the user for this release.
