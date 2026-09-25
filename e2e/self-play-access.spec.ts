import { expect, test } from '@playwright/test';
import { installAiWorkerFixture } from './ai-worker-fixture';
import { SELF_PLAY_PATH } from './self-play-helpers';

for (const path of ['/', '/?selfplay=true', '/?selfplay=wrong']) {
  test(`public visit ${path} keeps a human seat and only offers Standard or Deep`, async ({ page }) => {
    const ai = await installAiWorkerFixture(page, { holdMoves: true });
    await page.goto(path);
    const first = page.getByRole('combobox', { name: 'Player 1 controller' });
    const second = page.getByRole('combobox', { name: 'Player 2 controller' });
    await expect(first.locator('option[value="local"]')).toBeEnabled();
    await first.selectOption('local');
    await expect(second.locator('option[value="local"]')).toHaveCount(0);
    await expect(second).toHaveValue('human');
    await expect(page.getByRole('button', { name: 'Quick AI strength' })).toHaveCount(0);
    await expect(page.getByText('Custom search budget', { exact: true })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Standard AI strength' })).toHaveAttribute('aria-pressed', 'true');
    await page.getByRole('button', { name: 'Begin the game' }).click();
    await expect(page.getByRole('progressbar', { name: 'AI search progress' })).toBeVisible();
    await expect(page.getByRole('progressbar', { name: 'AI search progress' })).toHaveAttribute('max', '544');
    await expect.poll(() => ai.requests.length).toBe(1);
    await expect(page.getByText('Human versus AI', { exact: true })).toBeVisible();
    await expect(page.getByRole('region', { name: 'Engine estimate' })).toHaveCount(0);
  });
}

test('the special link permits self-play, but ordinary navigation and reload revoke it', async ({ page }) => {
  await installAiWorkerFixture(page, { holdMoves: true });
  await page.goto(SELF_PLAY_PATH);
  const first = page.getByRole('combobox', { name: 'Player 1 controller' });
  await expect(first.locator('option[value="local"]')).toBeEnabled();
  await first.selectOption('local');
  await page.getByRole('combobox', { name: 'Player 2 controller' }).selectOption('local');
  await page.getByRole('button', { name: 'Deep AI strength' }).click();
  await page.getByRole('button', { name: 'Begin the game' }).click();
  await expect(page.getByText('AI versus AI', { exact: true })).toBeVisible();
  await expect(page.getByRole('progressbar', { name: 'AI search progress' })).toHaveAttribute('max', '4096');
  await page.reload();
  await expect(page.getByText('AI versus AI', { exact: true })).toBeVisible();
  await page.goto('/');
  await expect(page.getByText('Human versus AI', { exact: true })).toBeVisible();
  await expect(page.getByText('AI versus AI', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Engine estimate' })).toHaveCount(0);
  await page.reload();
  await expect(page.getByText('Human versus AI', { exact: true })).toBeVisible();
});
