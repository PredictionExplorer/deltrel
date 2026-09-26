import { expect, test } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { reachFourRingClinch } from './helpers';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.getByRole('textbox', { name: 'Player 1 name' }).fill('Ada');
  await page.getByRole('textbox', { name: 'Player 2 name' }).fill('Grace');
  await page.getByRole('button', { name: 'Mini, 4 rings' }).click();
  await page.getByRole('button', { name: 'Begin the game' }).click();
});

test('shares, imports and reviews a game without overwriting the active match', async ({ page }) => {
  await page.getByRole('button', { name: /^Node A10, empty/ }).click();
  await page.getByRole('button', { name: /Steal it/ }).click();
  await page.getByRole('button', { name: /^Node B10, empty/ }).click();
  await page.getByRole('button', { name: 'Share game', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Game library' });
  const notation = await dialog.getByRole('textbox', { name: 'Game notation' }).inputValue();
  expect(notation).toContain('1. A10 2. swap');
  expect(notation).toContain('[Termination "unfinished"]');
  await dialog.getByRole('button', { name: 'Import game', exact: true }).click();
  await dialog.getByRole('textbox', { name: 'Game notation' }).fill(notation);
  await dialog.getByRole('button', { name: 'Import and review' }).click();
  await expect(dialog.getByText('Saved position')).toBeVisible();
  await dialog.getByRole('button', { name: 'Jump to the empty board' }).click();
  await expect(dialog.getByRole('img', { name: /0 of 50 nodes occupied/ })).toBeVisible();
  await dialog.getByRole('button', { name: 'Play replay' }).click();
  await expect(dialog.getByText('Viewing move 1 of 3')).toBeVisible();
  await dialog.getByRole('button', { name: 'Pause replay' }).click();
  await dialog.getByRole('button', { name: 'Close game library' }).click();
  await expect(page.getByRole('group', { name: /2 of 50 nodes occupied/ })).toBeVisible();
  await page.reload();
  await expect(page.getByRole('group', { name: /2 of 50 nodes occupied/ })).toBeVisible();
  await page.getByRole('button', { name: 'New game', exact: true }).click();
  await page.getByRole('button', { name: 'Game library', exact: true }).click();
  await expect(dialog.getByRole('button', { name: 'Review Ada vs Grace' })).toHaveCount(2);
  await expect(dialog.getByRole('button', { name: 'Saved games 2' })).toBeVisible();
});

test('resignation and proven clinch are preserved as distinct portable results', async ({ page }) => {
  await page.getByRole('button', { name: /^Node A10, empty/ }).click();
  await page.getByRole('button', { name: 'Resign Grace', exact: true }).click();
  await page.getByRole('dialog', { name: /resign/i }).getByRole('button', { name: /resign/i }).click();
  await page.getByRole('dialog', { name: 'Game over' }).getByRole('button', { name: /Review/ }).click();
  await page.getByRole('button', { name: 'Share game', exact: true }).click();
  const library = page.getByRole('dialog', { name: 'Game library' });
  const resignation = await library.getByRole('textbox', { name: 'Game notation' }).inputValue();
  expect(resignation).toContain('[Termination "resignation"]');
  expect(resignation).toContain('[Result "1-0"]');
  await library.getByRole('button', { name: 'Close game library' }).click();
  await page.getByRole('button', { name: 'New game', exact: true }).click();
  await page.getByRole('button', { name: 'Begin the game' }).click();
  await reachFourRingClinch(page);
  await page.getByRole('dialog', { name: /cannot be caught/i }).getByRole('button', { name: /End/ }).click();
  await page.getByRole('dialog', { name: 'Game over' }).getByRole('button', { name: /Review/ }).click();
  await page.getByRole('button', { name: 'Share game', exact: true }).click();
  const clinch = await library.getByRole('textbox', { name: 'Game notation' }).inputValue();
  expect(clinch).toContain('[Termination "clinch"]');
  expect(clinch).not.toContain('[Result "*"]');
});

test('phone library and review remain accessible and fit the viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('button', { name: /^Node A10, empty/ }).click();
  await page.getByRole('button', { name: 'Game library', exact: true }).click();
  const library = page.getByRole('dialog', { name: 'Game library' });
  await expect(library.getByRole('button', { name: 'Review Ada vs Grace' })).toBeVisible();
  expect((await new AxeBuilder({ page }).include('dialog').analyze()).violations).toEqual([]);
  await library.getByRole('button', { name: 'Review Ada vs Grace' }).click();
  expect((await new AxeBuilder({ page }).include('dialog').analyze()).violations).toEqual([]);
  expect(await library.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: '/tmp/deltrel-library-phone.png' });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.screenshot({ path: '/tmp/deltrel-library-desktop.png' });
});
