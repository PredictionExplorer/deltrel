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

test('uses spatial file/rank coordinates throughout the board and persisted history', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Full, 10 rings' }).click();
  await page.getByRole('button', { name: 'Begin the game' }).click();
  const board = page.getByRole('group', { name: /Deltrel board with 10 rings/ });
  const nodes = board.getByRole('button');
  await expect(nodes).toHaveCount(275);
  const labels = await nodes.evaluateAll((elements) => elements.map((element) =>
    element.getAttribute('aria-label')?.match(/^Node ([A-Z][1-9]\d?),/)?.[1]));
  expect(labels.every(Boolean)).toBe(true);
  expect(new Set(labels).size).toBe(275);
  expect(labels.slice(0, 3)).toEqual(['N10', 'L10', 'L11']);
  expect(labels.at(-1)).toBe('U2');
  await expect(board.locator('[data-coordinate-axes]')).toHaveCount(0);
  await expect(board.locator('[data-coordinate-guides]')).toHaveCount(0);
  await expect(board.locator('[data-coordinate-tooltip]')).toHaveCount(0);
  await nodes.first().press('End');
  await expect(board.locator('[data-coordinate-tooltip="U2"]')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(board.locator('[data-coordinate-tooltip]')).toHaveCount(0);
  await page.keyboard.press('Enter');
  await expect(page.getByRole('region', { name: 'Move history' })).toContainText('U2');
  await page.reload();
  await expect(page.getByRole('button', { name: /^Node U2, Player 1 stone/ })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Move history' })).toContainText('U2');
});

test.describe('touch coordinate inspection', () => {
  test.use({ hasTouch: true, viewport: { width: 390, height: 844 } });

  test('a quick tap still places exactly one stone', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: 'Mini, 4 rings' }).click();
    await page.getByRole('button', { name: 'Begin the game' }).click();
    await page.getByRole('button', { name: /^Node G4,/ }).tap();
    await expect(page.getByRole('group', { name: /1 of 50 nodes occupied/ })).toBeVisible();
    await expect(page.locator('[data-coordinate-tooltip]')).toHaveCount(0);
  });

  test('a native touch hold inspects without playing and the next tap works', async ({ page, context, browserName }) => {
    test.skip(browserName !== 'chromium', 'Native held-touch injection requires the Chromium input protocol.');
    await page.goto('/');
    await page.getByRole('button', { name: 'Mini, 4 rings' }).click();
    await page.getByRole('button', { name: 'Begin the game' }).click();
    const node = page.getByRole('button', { name: /^Node G4,/ });
    const box = await node.boundingBox();
    expect(box).not.toBeNull();
    const touch = { x: box!.x + box!.width / 2, y: box!.y + box!.height / 2 };
    const session = await context.newCDPSession(page);
    try {
      await session.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [touch] });
      await expect(page.locator('[data-coordinate-tooltip="G4"]')).toBeVisible();
      await expect(page.getByRole('group', { name: /0 of 50 nodes occupied/ })).toBeVisible();
      await session.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
      await expect(page.locator('[data-coordinate-tooltip]')).toHaveCount(0);
      // Give delayed compatibility mouse events time to surface an unwanted move.
      await page.waitForTimeout(450);
      await expect(page.getByRole('group', { name: /0 of 50 nodes occupied/ })).toBeVisible();
      await page.touchscreen.tap(touch.x, touch.y);
      await expect(page.getByRole('group', { name: /1 of 50 nodes occupied/ })).toBeVisible();
    } finally {
      await session.detach();
    }
  });
});
