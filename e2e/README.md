# Browser layout snapshots

The `layout-chromium` project checks viewport bounds, overflow, stable board and
status geometry, focus, and screenshots against a production build. Screenshot
baselines live in `__screenshots__/layout.spec.ts/<platform>/`, where `platform`
is Node's `process.platform`: `linux` for CI and `darwin` for macOS. Native
scrollbars differ between these platforms; do not copy one platform's images
over another's. Windows baselines have not been qualified.

Run the current platform's checks with:

```sh
npm ci
npx playwright install chromium
npm run build
npx playwright test --project=layout-chromium
```

For an intentional visual change, run the same command with `--update-snapshots=all`,
then inspect every changed image before committing it. Keep the existing layout
assertions and screenshot difference threshold. A missing baseline fails the
normal comparison; it is not an approved new image.

The **Layout baseline proposal** workflow builds and captures Linux images using
Node 22 and the locked Playwright version. It runs automatically when its own
workflow file changes in a pull request and supports manual dispatch after it is
merged. Its artifact includes source/version provenance, image hashes, and the
test report. This workflow proposes images only: it never commits them, and a
successful capture does not replace the normal browser CI gate. Require a
successful layout run and visually review the images before copying the `linux`
directory into this repository. A failed proposal may contain partial images.

## Firefox real-model inference

Use Playwright 1.63.0 or newer for the real browser-AI test. Earlier Firefox
Juggler debuggers disabled the optimizing WASM compiler in observed workers.
The [upstream fix](https://github.com/microsoft/playwright/blob/v1.63.0/browser_patches/firefox/juggler/content/Runtime.js#L54-L57)
enables unobserved WASM/asm.js execution; both flags are present in the shipped
Firefox revision 1543. Stable 1.62.1 still lacks the fix.

The October 2, 2026 reproduction used Node 22.23.3, macOS ARM64, ONNX Runtime
1.27.0, and the published `deltrel-champion-566428-20f52f268869` FP32 model:

| Check | Playwright 1.61.1 / Firefox 1532 | Playwright 1.63.0 / Firefox 1543 |
| --- | --- | --- |
| Five real model forwards, mean | 1,553 ms | 56.2 ms |
| First Standard move | Timed out at 300 s, 145/544 simulations | 48.35 s |
| Standard move after reload/cache reuse | Not reached | 44.94 s |

The forward comparison used the same four-ring Classic pie position after
placing node 0, the same Asyncify WASM runtime, and one inference thread. Policy
and outcome arrays matched exactly. The complete production UI test then passed
with its existing timeouts, real worker/model, verified cache, all network heads,
and both searches at 544 simulations / 16 considered actions. Its assertions
also require exactly 544 root visits and no remote inference. Keep the client's
90-second search-inactivity timeout and these strength checks when investigating
future slow runs.
