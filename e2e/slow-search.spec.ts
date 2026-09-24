import { expect, test } from '@playwright/test';
import { setAiSearchBudget } from './browser-ai-helpers';
import type { DeltrelAiAnalysis } from '../src/lib/deltrel/ai/decision';

// Real Full-board CPU inference can take several minutes. The default suite
// covers deadline behavior with fake clocks and real-model searches with small custom budgets;
// opt into this expensive regression when validating runtime or timeout changes.
test.skip(({ browserName }) => browserName !== 'firefox' || process.env.DELTREL_SLOW_AI_TESTS !== '1',
  'Run explicitly with DELTREL_SLOW_AI_TESTS=1 on Firefox.');

test('Full-board 64-simulation search on the Firefox CPU fallback', async ({ page, context }, testInfo) => {
  test.setTimeout(900_000);
  await context.route('**/v2/health', route => route.fulfill({ status: 503, contentType: 'application/json', body: '{}' }));
  await page.goto('/');
  await page.getByRole('button', { name: 'Full, 10 rings', exact: true }).click();
  const controller = page.getByRole('combobox', { name: 'Player 1 controller' });
  await expect(controller.locator('option[value="local"]')).toBeEnabled();
  await controller.selectOption('local');
  await page.getByRole('combobox', { name: 'Player 2 controller' }).selectOption('human');
  await setAiSearchBudget(page, 64, 8);
  await expect(page.getByRole('button', { name: 'AI selected', exact: true })).toBeVisible({ timeout: 120_000 });
  const started = Date.now();
  await page.getByRole('button', { name: 'Begin the game', exact: true }).click();
  const progress = page.getByRole('progressbar', { name: /^AI.* search progress$/ });
  await expect(progress).toBeVisible({ timeout: 95_000 });
  await expect(progress).toHaveAttribute('max', '64');
  const firstProgress = Number(await progress.getAttribute('value'));
  expect(firstProgress).toBeGreaterThan(0);
  await expect(page.locator('[data-move-chip="0"]').or(page.getByRole('main').getByRole('alert')).first()).toBeVisible({ timeout: 780_000 });
  await expect(page.getByRole('main').getByRole('alert')).toHaveCount(0);
  await expect(page.locator('[data-move-chip]')).toHaveCount(1);
  const estimate = page.getByRole('region', { name: 'Engine estimate', exact: true });
  await estimate.getByRole('button', { name: /^Raw engine output/ }).click();
  const { analysis } = JSON.parse(await estimate.getByRole('textbox', { name: 'Raw engine output' }).inputValue()) as { analysis: DeltrelAiAnalysis };
  expect(analysis.simulations).toBe(64);
  expect(analysis.rootVisits.reduce((sum, value) => sum + value, 0)).toBe(64);
  const evidence = { browser: 'Firefox CPU fallback', rings: 10, modelVersion: analysis.modelVersion,
    simulations: analysis.simulations, rootVisits: analysis.rootVisits.reduce((sum, value) => sum + value, 0),
    firstProgress, elapsedMs: Date.now() - started, timingMs: analysis.timingMs };
  await testInfo.attach('slow-search-evidence', { contentType: 'application/json', body: JSON.stringify(evidence) });
});
