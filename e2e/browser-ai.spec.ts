import { SELF_PLAY_PATH } from './self-play-helpers';
import { expect, test } from '@playwright/test';
import type { DeltrelAiAnalysis } from '../src/lib/deltrel/ai/decision';
import { selectAiStrength } from './browser-ai-helpers';
import { installAiWorkerFixture } from './ai-worker-fixture';
import publishedModel from '../public/models/deltrel/manifest.json';

// Model inference is CPU-bound in headless Firefox. Keep these real-model
// scenarios sequential within each browser instead of competing for its budget.
test.describe.configure({ mode: 'default' });

for (const mode of ['classic', 'double'] as const) {
  test(`${mode}: the self-play UI completes a nine-stone handicap before the opponent moves`, async ({ page, context }) => {
    test.setTimeout(45_000);
    // Exercise UI turn sequencing through the worker protocol without running
    // ten Full-board searches. Native engine tests cover handicap search rules.
    const ai = await installAiWorkerFixture(page, { heldCalls: [10, 11] });
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    await context.route('**/v2/health', route => route.fulfill({
      status: 503, contentType: 'application/json', body: '{}',
    }));
    await page.goto(SELF_PLAY_PATH);
    await page.getByRole('button', { name: 'Full, 10 rings', exact: true }).click();
    await page.getByRole('button', { name: mode === 'classic' ? /^Classic Deltrel,/ : /^Double Deltrel,/ }).click();
    await page.getByRole('button', { name: 'Handicap opening', exact: true }).click();
    await page.getByRole('radio', { name: '9 handicap stones', exact: true }).click();
    const first = page.getByRole('combobox', { name: 'Player 1 controller' });
    await expect(first.locator('option[value="local"]')).toBeEnabled();
    await first.selectOption('local');
    await page.getByRole('combobox', { name: 'Player 2 controller' }).selectOption('local');
    await page.getByRole('textbox', { name: 'Player 1 name' }).fill('Drift');
    await page.getByRole('textbox', { name: 'Player 2 name' }).fill('Tide');
    await selectAiStrength(page, 'Standard');
    await expect(page.getByRole('button', { name: 'AI selected', exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Begin the game', exact: true }).click();
    // Every one of the nine opening placements must complete before side two
    // receives a move request; hold its response to inspect that boundary.
    await expect.poll(() => ai.requests.length).toBe(10);
    expect(ai.requests.slice(0, 9).map(request => ({
      toMove: request.state.toMove,
      opening: request.state.opening,
      movesLeft: request.state.movesLeft,
    }))).toEqual(Array.from({ length: 9 }, (_, index) => ({ toMove: 0, opening: true, movesLeft: 9 - index })));
    expect(ai.requests[9].state.toMove).toBe(1);
    await expect(page.locator('[data-move-chip]')).toHaveCount(9);
    await ai.releaseCall(10);
    await expect(page.locator('[data-move-chip="9"]')).toBeVisible();
    await expect(page.getByRole('main').getByRole('alert')).toHaveCount(0);
    await page.getByRole('button', { name: 'Pause AI', exact: true }).click();
    await expect(page.locator('[data-game-status="paused"]')).toBeVisible();
    for (let index = 0; index < 9; index++) {
      await expect(page.locator(`[data-move-chip="${index}"]`)).toHaveAccessibleName(new RegExp(`^Go to move ${index + 1}: Drift `));
    }
    await expect(page.locator('[data-move-chip="9"]')).toHaveAccessibleName(/^Go to move 10: Tide /);
    await expect(page.getByRole('main').getByRole('alert')).toHaveCount(0);
    expect(errors).toEqual([]);
  });
}


async function inspectModelCache() {
  if (typeof caches === 'undefined') return { supported: false, entries: [] };
  try {
    const cache = await caches.open('deltrel-models-v1');
    const requests = await cache.keys();
    const entries = [];
    for (const request of requests) {
      const response = await cache.match(request);
      if (!response) continue;
      const bytes = await response.arrayBuffer();
      const digest = await crypto.subtle.digest('SHA-256', bytes);
      entries.push({ url: request.url, bytes: bytes.byteLength,
        sha256: Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('') });
    }
    return { supported: true, entries };
  } catch (error) { return { supported: false, entries: [], error: String(error) }; }
}

// This exercises the published model, real worker, WASM search and ONNX runtime.
// No remote inference is permitted; the browser uses the shipped model.
test('preserves Standard strength through preparation, plays two real Mini searches locally, and reuses its verified cache', async ({ page, context, browserName }, testInfo) => {
  // Both moves use the public Standard preset, including on the CPU fallback.
  // The real client's 90-second inactivity timeout still detects stalled work.
  const searchTimeout = browserName === 'firefox' ? 300_000 : 180_000;
  test.setTimeout(2 * searchTimeout + 180_000);
  let releaseDownload!: () => void;
  const downloadGate = new Promise<void>(resolve => { releaseDownload = resolve; });
  let blockModelDownload = false;
  let modelGets = 0;
  const remoteInference: string[] = [];
  const runtimeAssets: string[] = [];

  context.on('request', request => {
    const path = new URL(request.url()).pathname;
    if (request.method() === 'GET' && path.startsWith('/onnxruntime/')) runtimeAssets.push(path);
  });
  await context.route('**/v2/health', route => route.fulfill({
    status: 503,
    contentType: 'application/json',
    body: JSON.stringify({ status: 'unavailable' }),
  }));
  await context.route(/\/v2\/(?:move|analyze)(?:\?.*)?$/, async route => {
    remoteInference.push(route.request().url());
    await route.abort('blockedbyclient');
  });
  await context.route(/\/models\/deltrel\/[^/?]+\.onnx(?:\?.*)?$/, async route => {
    if (route.request().method() === 'GET') {
      modelGets++;
      if (blockModelDownload) {
        await route.abort('blockedbyclient');
        return;
      }
      await downloadGate;
    }
    // Availability probes use HEAD and must complete while the GET is held.
    await route.continue();
  });

  await page.goto(SELF_PLAY_PATH);
  await expect(page.getByRole('heading', { name: 'Deltrel', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Mini, 4 rings', exact: true }).click();
  await page.getByRole('button', { name: 'Classic Deltrel, 1 stone per turn', exact: true }).click();
  const playerOneController = page.getByRole('combobox', { name: 'Player 1 controller' });
  const playerTwoController = page.getByRole('combobox', { name: 'Player 2 controller' });
  await expect(playerOneController.locator('option[value="local"]')).toBeEnabled();
  await playerOneController.selectOption('local');
  await playerTwoController.selectOption('local');
  await page.getByRole('textbox', { name: 'Player 1 name' }).fill('Drift');
  await page.getByRole('textbox', { name: 'Player 2 name' }).fill('Tide');
  await selectAiStrength(page, 'Standard');
  await expect.poll(() => modelGets).toBe(1);
  await expect(page.getByRole('progressbar', { name: 'AI preparation' })).toBeVisible();
  releaseDownload();
  await expect(page.getByRole('button', { name: 'AI selected', exact: true })).toBeVisible({ timeout: 120_000 });
  expect(modelGets).toBe(1);
  expect(runtimeAssets.some(path => path.endsWith('.wasm'))).toBe(true);
  await expect(playerOneController).toHaveValue('local');
  await expect(playerTwoController).toHaveValue('local');
  await expect(page.getByRole('textbox', { name: 'Player 1 name' })).toHaveValue('Drift');
  await expect(page.getByRole('textbox', { name: 'Player 2 name' })).toHaveValue('Tide');
  await expect(page.getByRole('button', { name: 'Standard AI strength', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('button', { name: 'Quick AI strength', exact: true })).toHaveCount(0);
  await expect(page.getByText('Custom search budget', { exact: true })).toHaveCount(0);
  const cacheAfterDownload = await page.evaluate(inspectModelCache);
  const expectedModel = publishedModel.artifacts.onnx;
  const cachedEntry = cacheAfterDownload.entries.find(entry => entry.sha256 === expectedModel.sha256);
  expect(cachedEntry).toMatchObject({ bytes: expectedModel.bytes, sha256: expectedModel.sha256 });
  await testInfo.attach('model-cache-after-download', {
    contentType: 'application/json', body: JSON.stringify(cacheAfterDownload, null, 2),
  });

  await page.getByRole('button', { name: 'Begin the game', exact: true }).click();
  await expect(page.getByText('AI versus AI', { exact: true })).toBeVisible();
  await expect(page.getByText('Human versus AI', { exact: true })).toHaveCount(0);
  await expect(page.locator('[data-game-status="human"]')).toHaveCount(0);
  await expect(page.getByRole('button', { name: /empty .* may place here/i })).toHaveCount(0);
  // Wait for a genuine AI action, then stop autoplay before inspecting its result.
  // Count actions rather than occupied nodes because the pie reply may be a swap.
  await expect(page.getByRole('progressbar', { name: 'AI search progress', exact: true })).toBeVisible();
  await expect(page.getByRole('progressbar', { name: 'AI search progress', exact: true })).toHaveAttribute('max', '544');
  await expect(page.locator('[data-move-chip="0"]')).toBeVisible({ timeout: searchTimeout });
  const estimate = page.getByRole('region', { name: 'Engine estimate', exact: true });
  await estimate.getByRole('button', { name: 'Pause AI', exact: true }).click();
  await expect(page.locator('[data-game-status="paused"]')).toBeVisible();
  const pausedPly = await page.locator('[data-move-chip]').count();
  expect(pausedPly).toBeGreaterThanOrEqual(1);
  const scores = page.getByRole('region', { name: 'Current player scores' });
  await expect(scores.getByRole('heading', { name: 'Drift', exact: true })).toBeVisible();
  await expect(scores.getByRole('heading', { name: 'Tide', exact: true })).toBeVisible();
  await expect(scores.getByText('AI', { exact: true })).toHaveCount(2);
  await expect(scores.getByText('Human', { exact: true })).toHaveCount(0);
  await expect(estimate.getByRole('article', { name: 'Drift forecast' })).toContainText(/\d+(?:\.\d+)?%/);
  await expect(estimate.getByRole('article', { name: 'Tide forecast' })).toContainText(/\d+(?:\.\d+)?%/);
  await estimate.getByRole('button', { name: /^All network outputs/ }).click();
  const chooser = estimate.getByLabel('Network output', { exact: true });
  await expect(chooser.locator('option')).toHaveCount(11);
  await chooser.selectOption('ownership');
  await expect(estimate.getByRole('table', { name: 'Ownership at every point' })).toBeVisible();
  await expect(estimate.getByRole('table', { name: 'Ownership at every point' })).toContainText(/\d+(?:\.\d+)?%/);
  await estimate.getByRole('button', { name: /^Raw engine output/ }).click();
  const report = JSON.parse(await estimate.getByRole('textbox', { name: 'Raw engine output' }).inputValue()) as {
    analysis: DeltrelAiAnalysis;
    context: { source: string; applied: boolean };
    board: { rings: number };
  };
  expect(report.context).toMatchObject({ source: 'local', applied: true });
  expect(report.board.rings).toBe(4);
  expect(report.analysis.simulations).toBe(544);
  expect(report.analysis.maxConsidered).toBe(16);
  expect(report.analysis.rootVisits.reduce((sum, visits) => sum + visits, 0)).toBe(544);
  const network = report.analysis.networkOutput!;
  expect(network.nodeCount).toBe(50);
  expect(network.perspective).toBe(report.analysis.perspective);
  expect(network.auxiliaryStatus).toBe('ready');
  expect(Object.values(network.heads).filter(Boolean)).toHaveLength(11);
  expect(network.heads.scoreMargin!.probabilities).toHaveLength(303);
  expect(network.heads.ownership!.probabilities).toHaveLength(150);
  expect(network.heads.policy!.probabilities.reduce((sum, probability) => sum + probability, 0)).toBeCloseTo(1, 5);
  expect(network.heads.outcome!.probabilities.reduce((sum, probability) => sum + probability, 0)).toBeCloseTo(1, 5);
  expect(report.analysis.predictions?.finalCounts).toHaveLength(2);
  expect(remoteInference).toEqual([]);

  // Reload with Standard unchanged and verify the next real search uses it too.
  await expect(page.locator('[data-move-chip]')).toHaveCount(pausedPly);
  blockModelDownload = true;
  await page.reload();
  const cacheAfterReload = await page.evaluate(inspectModelCache);
  await testInfo.attach('model-cache-after-reload', {
    contentType: 'application/json', body: JSON.stringify(cacheAfterReload, null, 2),
  });
  const cachePersisted = cacheAfterReload.entries.some(entry =>
    entry.bytes === expectedModel.bytes && entry.sha256 === expectedModel.sha256);
  if (browserName !== 'webkit') expect(cachePersisted).toBe(true);
  await expect(page.locator('[data-move-chip]')).toHaveCount(pausedPly);
  const preparationProgress = page.getByRole('progressbar', { name: 'AI preparation' });
  const browserPreparation = page.getByRole('region', { name: 'AI on this device', exact: true });
  let expectedModelGets = 1;
  if (cachePersisted) {
    await expect(browserPreparation).toBeHidden({ timeout: 30_000 });
    expect(modelGets).toBe(1);
  } else {
    // On this macOS WebKit build, even Cache.put('/probe', new Response('x'))
    // on a script-free /icon.svg page loses its entry on reload. Verify the
    // product's storage-eviction recovery instead of skipping real inference.
    expect(browserName).toBe('webkit');
    testInfo.annotations.push({ type: 'cache-persistence', description: 'WebKit evicted a verified model on reload; real retry/redownload is exercised.' });
    await expect(browserPreparation.getByRole('alert')).toContainText('could not be downloaded');
    await expect(page.locator('[data-move-chip]')).toHaveCount(pausedPly);
    expect(modelGets).toBe(2); // The deliberately blocked cache-miss request.
    blockModelDownload = false;
    await browserPreparation.getByRole('button', { name: 'Retry AI', exact: true }).click();
    await expect(preparationProgress).toBeVisible();
    await expect(browserPreparation).toBeHidden({ timeout: 30_000 });
    expectedModelGets = 3;
    expect(modelGets).toBe(expectedModelGets);
  }
  await expect(page.getByText('AI versus AI', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Standard AI strength', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Resume AI', exact: true }).click();
  await expect(page.locator(`[data-move-chip="${pausedPly}"]`)).toBeVisible({ timeout: searchTimeout });
  const resumedEstimate = page.getByRole('region', { name: 'Engine estimate', exact: true });
  await resumedEstimate.getByRole('button', { name: 'Pause AI', exact: true }).click();
  await expect(page.locator('[data-game-status="paused"]')).toBeVisible();
  await expect(resumedEstimate.getByRole('article', { name: 'Tide forecast' })).toContainText(/\d+(?:\.\d+)?%/);
  await resumedEstimate.getByRole('button', { name: /^Raw engine output/ }).click();
  const resumedReport = JSON.parse(await resumedEstimate.getByRole('textbox', { name: 'Raw engine output' }).inputValue()) as { analysis: DeltrelAiAnalysis };
  expect(resumedReport.analysis.simulations).toBe(544);
  expect(resumedReport.analysis.maxConsidered).toBe(16);
  expect(resumedReport.analysis.rootVisits.reduce((sum, visits) => sum + visits, 0)).toBe(544);
  await expect(page.getByText('Human versus AI', { exact: true })).toHaveCount(0);
  expect(modelGets).toBe(expectedModelGets);
  expect(remoteInference).toEqual([]);
});
