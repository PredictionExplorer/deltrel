import { expect, test } from '@playwright/test';

test('serves Deltrel identity, marine assets, and a matching installation manifest', async ({ page, request }) => {
  await page.goto('/');
  await expect(page).toHaveTitle('Deltrel — Connection runs deep');
  await expect(page.locator('meta[name="application-name"]')).toHaveAttribute('content', 'Deltrel');
  await expect(page.locator('meta[property="og:site_name"]')).toHaveAttribute('content', 'Deltrel');
  await expect(page.locator('[data-water-scene]')).toBeVisible();
  await expect(page.locator('[data-shoreline]')).toHaveCount(1);
  const manifestResponse = await request.get('/manifest.webmanifest');
  expect(manifestResponse.ok()).toBe(true);
  const manifest = await manifestResponse.json();
  expect(manifest.short_name).toBe('Deltrel');
  expect(manifest.theme_color).toBe('#0b393c');
  const icon = await request.get(manifest.icons[0].src);
  expect(icon.ok()).toBe(true);
  expect(icon.headers()['content-type']).toContain('image/svg+xml');
  const retired = String.fromCharCode(115, 116, 97, 114);
  expect(await page.content()).not.toMatch(new RegExp(`\\b${retired}(?:\\b|board|field|train|serve)`, 'i'));
});

test('uses letter coordinates throughout a full board and persisted move history', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Full, 10 rings' }).click();
  await page.getByRole('button', { name: 'Begin the game' }).click();
  const board = page.getByRole('group', { name: /Deltrel board with 10 rings/ });
  const nodes = board.getByRole('button');
  await expect(nodes).toHaveCount(275);
  const labels = await nodes.evaluateAll((elements) => elements.map((element) =>
    element.getAttribute('aria-label')?.match(/^Node ([A-Z]+),/)?.[1]));
  expect(labels.every(Boolean)).toBe(true);
  expect(new Set(labels).size).toBe(275);
  expect(labels.slice(0, 3)).toEqual(['A', 'B', 'C']);
  expect(labels.slice(25, 28)).toEqual(['Z', 'AA', 'AB']);
  expect(labels.at(-1)).toBe('JO');
  await nodes.last().focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('region', { name: 'Move history' })).toContainText('JO');
  await page.reload();
  await expect(page.getByRole('button', { name: /^Node JO, Player 1 stone/ })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Move history' })).toContainText('JO');
});
