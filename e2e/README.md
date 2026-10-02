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
