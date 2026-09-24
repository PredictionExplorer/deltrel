import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';
import { installAiWorkerFixture } from './ai-worker-fixture';

async function setup(page: Page) {
  await page.goto('/');
  await expect(page.getByRole('region', { name: 'AI', exact: true })
    .getByRole('status')).toHaveText('Ready on this device');
  await page.getByRole('button', { name: 'Mini, 4 rings' }).click();
  await page.getByRole('textbox', { name: 'Player 1 name' }).fill('Ada');
  await page.getByRole('textbox', { name: 'Player 2 name' }).fill('Grace');
}

test('all network outputs are inspectable without playing a move', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const ai = await installAiWorkerFixture(page);
  await setup(page);
  await page.getByRole('button', { name: 'Begin the game' }).click();
  const panel = page.getByRole('region', { name: 'Engine estimate' });
  expect((await panel.boundingBox())!.height).toBeGreaterThan(200);
  await panel.getByRole('button', { name: 'Analyze position' }).click();
  await expect(panel.getByRole('article', { name: 'Ada forecast' })).toContainText('80.0%');
  await expect(panel.getByRole('article', { name: 'Grace forecast' })).toContainText('20.0%');
  await expect(panel.getByRole('article', { name: 'Ada forecast' })).toContainText('10.5');
  await expect(page.getByRole('group', { name: /0 of 50 nodes occupied/ })).toBeVisible();
  expect(ai.requests).toHaveLength(1);

  await panel.getByRole('button', { name: /^All network outputs/ }).click();
  const chooser = panel.getByLabel('Network output', { exact: true });
  await expect(chooser.locator('option')).toHaveCount(11);
  await chooser.selectOption('ownership');
  await expect(panel.getByRole('table', { name: 'Ownership at every point' })).toBeVisible();
  await panel.getByLabel('Find a point or value').fill('G4');
  await expect(panel.getByRole('table', { name: 'Ownership at every point' }).getByRole('row')).toHaveCount(2);
  await panel.getByLabel('Show raw logits').check();
  await expect(panel.getByText('logit +0.00000')).toHaveCount(3);
  await chooser.selectOption('scoreMargin');
  await expect(panel.getByRole('img', { name: 'Final score margin probability distribution' })).toBeVisible();
  await expect(panel.getByText('1–12 of 303')).toBeVisible();
  await panel.getByRole('button', { name: /^Raw engine output/ }).click();
  const raw = JSON.parse(await panel.getByRole('textbox', { name: 'Raw engine output' }).inputValue());
  expect(Object.keys(raw.analysis.networkOutput.heads)).toHaveLength(11);
  expect(raw.analysis.networkOutput.heads.scoreMargin.probabilities).toHaveLength(303);
  expect(raw.analysis.networkOutput.heads.ownership.probabilities).toHaveLength(150);
  expect(raw.context.source).toBe('local');
  expect(raw.context.applied).toBe(false);
  expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1);
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
});

test('AI-versus-AI forecasts persist while paused and manual inspection stays read-only', async ({ page }) => {
  const ai = await installAiWorkerFixture(page, { heldCalls: [2, 4] });
  await setup(page);
  await page.getByRole('combobox', { name: 'Player 1 controller' }).selectOption('local');
  await page.getByRole('combobox', { name: 'Player 2 controller' }).selectOption('local');
  await page.getByRole('button', { name: 'Begin the game' }).click();
  const panel = page.getByRole('region', { name: 'Engine estimate' });
  await expect(panel.getByRole('article', { name: 'Ada forecast' })).toContainText('80.0%');
  await expect.poll(() => ai.requests.length).toBe(2);
  await panel.getByRole('button', { name: /^All network outputs/ }).click();
  await panel.getByRole('button', { name: 'Pause AI' }).click();
  await expect(page.locator('[data-game-status="paused"]')).toBeVisible();
  await ai.releaseCall(2);
  expect(ai.discardedRequests.has(ai.requests[1].requestId)).toBe(true);
  await expect(page.getByRole('img', { name: /1 of 50 nodes occupied/ })).toBeVisible();
  await expect(panel.getByRole('button', { name: /^All network outputs/ })).toHaveAttribute('aria-expanded', 'true');
  await panel.getByRole('button', { name: 'Analyze position' }).click();
  await expect(panel.getByRole('article', { name: 'Grace forecast' })).toContainText('80.0%');
  await expect(panel.getByText(/After move 1 · Grace to move/)).toBeVisible();
  await expect(page.getByRole('img', { name: /1 of 50 nodes occupied/ })).toBeVisible();
  await page.getByRole('button', { name: 'Resume AI' }).click();
  await expect.poll(() => ai.requests.length).toBe(4);
  expect(ai.requests[3].state.stones).toEqual(ai.requests[2].state.stones);
  await panel.getByRole('button', { name: 'Pause AI' }).click();
  await expect(page.locator('[data-game-status="paused"]')).toBeVisible();
  await ai.releaseCall(4);
  expect(ai.discardedRequests.has(ai.requests[3].requestId)).toBe(true);
  await expect(page.getByRole('img', { name: /1 of 50 nodes occupied/ })).toBeVisible();
});
