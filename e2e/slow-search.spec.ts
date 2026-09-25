import { SELF_PLAY_PATH } from './self-play-helpers';
import { expect, test } from '@playwright/test';
import { selectAiStrength } from './browser-ai-helpers';
import type { DeltrelAiAnalysis } from '../src/lib/deltrel/ai/decision';

// Full-board Standard inference is deliberately expensive on the CPU fallback.
// The default suite covers inactivity deadlines and real Standard searches on Mini;
// opt into this expensive regression when validating runtime or timeout changes.
test.skip(({ browserName }) => browserName !== 'firefox' || process.env.DELTREL_SLOW_AI_TESTS !== '1',
  'Run explicitly with DELTREL_SLOW_AI_TESTS=1 on Firefox.');

test('Full-board Standard search on the Firefox CPU fallback', async ({ page, context }, testInfo) => {
  test.setTimeout(45 * 60_000);
  await context.route('**/v2/health', route => route.fulfill({ status: 503, contentType: 'application/json', body: '{}' }));
  await page.goto(SELF_PLAY_PATH);
  await page.getByRole('button', { name: 'Full, 10 rings', exact: true }).click();
  const controller = page.getByRole('combobox', { name: 'Player 1 controller' });
  await expect(controller.locator('option[value="local"]')).toBeEnabled();
  await controller.selectOption('local');
  await page.getByRole('combobox', { name: 'Player 2 controller' }).selectOption('local');
  await selectAiStrength(page, 'Standard');
  await expect(page.getByRole('button', { name: 'AI selected', exact: true })).toBeVisible({ timeout: 120_000 });
  const started = Date.now();
  await page.getByRole('button', { name: 'Begin the game', exact: true }).click();
  const progress = page.getByRole('progressbar', { name: /^AI.* search progress$/ });
  await expect(progress).toBeVisible({ timeout: 95_000 });
  await expect(progress).toHaveAttribute('max', '544');
  // The bar is visible at zero immediately; wait for genuine worker progress.
  await expect.poll(async () => Number(await progress.getAttribute('value')), { timeout: 95_000 }).toBeGreaterThan(0);
  const firstProgress = Number(await progress.getAttribute('value'));
  expect(firstProgress).toBeGreaterThan(0);
  await expect(page.locator('[data-move-chip="0"]').or(page.getByRole('main').getByRole('alert')).first()).toBeVisible({ timeout: 40 * 60_000 });
  await expect(page.getByRole('main').getByRole('alert')).toHaveCount(0);
  await page.getByRole('button', { name: 'Pause AI', exact: true }).click();
  await expect(page.locator('[data-move-chip]')).toHaveCount(1);
  const estimate = page.getByRole('region', { name: 'Engine estimate', exact: true });
  await estimate.getByRole('button', { name: /^Raw engine output/ }).click();
  const { analysis } = JSON.parse(await estimate.getByRole('textbox', { name: 'Raw engine output' }).inputValue()) as { analysis: DeltrelAiAnalysis };
  expect(analysis.simulations).toBe(544);
  expect(analysis.maxConsidered).toBe(16);
  expect(analysis.rootVisits.reduce((sum, value) => sum + value, 0)).toBe(544);
  const evidence = { browser: 'Firefox CPU fallback', rings: 10, modelVersion: analysis.modelVersion,
    simulations: analysis.simulations, rootVisits: analysis.rootVisits.reduce((sum, value) => sum + value, 0),
    firstProgress, elapsedMs: Date.now() - started, timingMs: analysis.timingMs };
  await testInfo.attach('slow-search-evidence', { contentType: 'application/json', body: JSON.stringify(evidence) });
});
