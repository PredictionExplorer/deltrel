import { expect, test } from '@playwright/test';
import type { DeltrelAiAnalysis } from '../src/lib/deltrel/ai/decision';
import publishedModel from '../public/models/deltrel/manifest.json';


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
// Only the optional private server is disabled; inference is never mocked.
test('downloads the browser champion on request, plays locally, and reuses its verified cache', async ({ page, context, browserName }, testInfo) => {
  test.setTimeout(90_000);
  let downloadAllowed = false;
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
      if (!downloadAllowed || blockModelDownload) {
        await route.abort('blockedbyclient');
        return;
      }
    }
    await route.continue();
  });

  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Deltrel', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Mini, 4 rings', exact: true }).click();
  await page.getByRole('button', { name: 'Classic Deltrel, 1 stone per turn', exact: true }).click();
  await page.getByRole('textbox', { name: 'Player 1 name' }).fill('Ada');
  const download = page.getByRole('button', { name: 'Download browser AI', exact: true });
  await expect(download).toBeEnabled();
  expect(modelGets).toBe(0);

  downloadAllowed = true;
  await download.click();
  await expect(page.getByRole('progressbar', { name: 'Browser AI preparation' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Browser AI selected', exact: true })).toBeVisible({ timeout: 45_000 });
  expect(modelGets).toBe(1);
  expect(runtimeAssets.some(path => path.endsWith('.wasm'))).toBe(true);
  await expect(page.getByRole('combobox', { name: 'Player 1 controller' })).toHaveValue('human');
  await expect(page.getByRole('combobox', { name: 'Player 2 controller' })).toHaveValue('local');
  await page.getByRole('button', { name: 'Begin the game', exact: true }).click();
  await page.getByRole('button', { name: /empty .*Ada may place here/i }).first().click();
  // Pie swapping is an action even though it does not increase occupied nodes.
  await expect(page.locator('[data-move-chip]')).toHaveCount(2, { timeout: 30_000 });

  const estimate = page.getByRole('region', { name: 'Engine estimate', exact: true });
  await expect(estimate.getByRole('article', { name: 'Ada forecast' })).toContainText(/\d+(?:\.\d+)?%/);
  await expect(estimate.getByRole('article', { name: 'Champion forecast' })).toContainText(/\d+(?:\.\d+)?%/);
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
  expect(report.analysis.perspective).toBe(1);
  const network = report.analysis.networkOutput!;
  expect(network.nodeCount).toBe(50);
  expect(network.auxiliaryStatus).toBe('ready');
  expect(Object.values(network.heads).filter(Boolean)).toHaveLength(11);
  expect(network.heads.scoreMargin!.probabilities).toHaveLength(303);
  expect(network.heads.ownership!.probabilities).toHaveLength(150);
  expect(network.heads.policy!.probabilities.reduce((sum, probability) => sum + probability, 0)).toBeCloseTo(1, 5);
  expect(network.heads.outcome!.probabilities.reduce((sum, probability) => sum + probability, 0)).toBeCloseTo(1, 5);
  expect(report.analysis.predictions?.finalCounts).toHaveLength(2);
  expect(remoteInference).toEqual([]);

  const cacheBeforeReload = await page.evaluate(inspectModelCache);
  const expectedModel = publishedModel.artifacts.onnx;
  const cachedEntry = cacheBeforeReload.entries.find(entry => entry.sha256 === expectedModel.sha256);
  expect(cachedEntry).toMatchObject({ bytes: expectedModel.bytes, sha256: expectedModel.sha256 });
  await testInfo.attach('model-cache-before-reload', {
    contentType: 'application/json', body: JSON.stringify(cacheBeforeReload, null, 2),
  });

  // Undo pauses the AI with its same position ready to be replayed after reload.
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(page.locator('[data-move-chip]')).toHaveCount(1);
  blockModelDownload = true;
  await page.reload();
  const cacheAfterReload = await page.evaluate(inspectModelCache);
  await testInfo.attach('model-cache-after-reload', {
    contentType: 'application/json', body: JSON.stringify(cacheAfterReload, null, 2),
  });
  const cachePersisted = cacheAfterReload.entries.some(entry =>
    entry.bytes === expectedModel.bytes && entry.sha256 === expectedModel.sha256);
  if (browserName !== 'webkit') expect(cachePersisted).toBe(true);
  const prepare = page.getByRole('button', { name: 'Prepare browser AI', exact: true });
  await expect(prepare).toBeEnabled();
  expect(modelGets).toBe(1);
  await expect(page.locator('[data-move-chip]')).toHaveCount(1);
  await prepare.click();
  const preparationProgress = page.getByRole('progressbar', { name: 'Browser AI preparation' });
  await expect(preparationProgress).toBeVisible();
  await expect(preparationProgress).toBeHidden({ timeout: 30_000 });
  const browserPreparation = page.getByRole('region', { name: 'Browser AI', exact: true });
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
    await expect(page.locator('[data-move-chip]')).toHaveCount(1);
    expect(modelGets).toBe(2); // The deliberately blocked cache-miss request.
    blockModelDownload = false;
    await browserPreparation.getByRole('button', { name: 'Retry browser AI', exact: true }).click();
    await expect(preparationProgress).toBeVisible();
    await expect(browserPreparation).toBeHidden({ timeout: 30_000 });
    expectedModelGets = 3;
    expect(modelGets).toBe(expectedModelGets);
  }
  await page.getByRole('button', { name: 'Resume AI', exact: true }).click();
  await expect(page.locator('[data-move-chip]')).toHaveCount(2, { timeout: 30_000 });
  await expect(page.getByRole('region', { name: 'Engine estimate' }).getByRole('article', { name: 'Champion forecast' })).toContainText(/\d+(?:\.\d+)?%/);
  expect(modelGets).toBe(expectedModelGets);
  expect(remoteInference).toEqual([]);
});
