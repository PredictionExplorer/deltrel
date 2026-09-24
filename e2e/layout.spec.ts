import { expect, test, type Page } from '@playwright/test';
import { fillFourRingGame, reachFourRingClinch } from './helpers';
import { installAiWorkerFixture } from './ai-worker-fixture';

const viewports: readonly {
  name: string;
  width: number;
  height: number;
  screenshot?: boolean;
}[] = [
  { name: 'small-phone', width: 320, height: 568 },
  { name: 'phone', width: 390, height: 844, screenshot: true },
  { name: 'tablet', width: 768, height: 1024 },
  { name: 'small-laptop', width: 1024, height: 768 },
  { name: 'laptop', width: 1280, height: 720, screenshot: true },
  { name: 'desktop', width: 1440, height: 900 },
];

async function openFreshSetup(page: Page, waitForAi = true) {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await expect(
    page.getByRole('heading', { level: 1, name: 'Deltrel' }),
  ).toBeVisible();
  if (waitForAi) await expect(page.getByRole('region', { name: 'AI', exact: true })
    .getByRole('status')).toHaveText('Ready on this device');
}

async function bounds(page: Page, selector: string) {
  const box = await page.locator(selector).boundingBox();
  if (!box) throw new Error(`No bounding box for ${selector}`);
  return box;
}

function expectStableBox(
  before: Awaited<ReturnType<typeof bounds>>,
  after: Awaited<ReturnType<typeof bounds>>,
  label: string,
) {
  for (const dimension of ['x', 'y', 'width', 'height'] as const) {
    expect(
      Math.abs(before[dimension] - after[dimension]),
      `${label} ${dimension} changed from ${JSON.stringify(before)} to ${JSON.stringify(after)}`,
    ).toBeLessThanOrEqual(1);
  }
}

for (const viewport of viewports) {
  test(`keeps AI gameplay stable at ${viewport.name}`, async ({ page }) => {
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    const ai = await installAiWorkerFixture(page, { modelVersion: 'layout-model', holdMoves: true });
    await openFreshSetup(page);
    await expect(page.getByRole('button', { name: 'Quick AI strength', exact: true })).toBeEnabled();

    const setupOverflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(setupOverflow).toBeLessThanOrEqual(1);
    if (viewport.screenshot) {
      await expect(page).toHaveScreenshot(`setup-${viewport.name}.png`, {
        fullPage: true,
      });
    }

    await page.getByRole('button', { name: /^Double Deltrel/i }).click();
    await page.getByRole('button', { name: /^Mini, 4 rings$/i }).click();
    const controller = page.getByRole('combobox', {
      name: 'Player 1 controller',
    });
    await expect(controller.locator('option[value="local"]')).toBeEnabled();
    await controller.selectOption('local');
    await page.getByRole('button', { name: 'Begin the game' }).click();

    await expect(page.locator('[data-game-status="thinking"]')).toBeVisible();
    expect(await page.evaluate(() => window.scrollY)).toBeLessThanOrEqual(1);
    const stageBefore = await bounds(page, '[data-board-stage]');
    const boardBefore = await bounds(page, '[data-board-stage] svg');
    const statusBefore = await bounds(page, '[data-game-status]');
    expect(Math.abs(boardBefore.width - boardBefore.height)).toBeLessThanOrEqual(1);
    expect(boardBefore.x).toBeGreaterThanOrEqual(0);
    expect(boardBefore.x + boardBefore.width).toBeLessThanOrEqual(viewport.width + 1);

    ai.releaseMoves();
    await expect(page.locator('[data-game-status="human"]')).toBeVisible();
    await expect(page.getByText('Player 2 to play')).toBeVisible();
    await expect(
      page.getByRole('group', {
        name: /Deltrel board with 4 rings, 1 of 50 nodes occupied/i,
      }),
    ).toBeVisible();
    expect(ai.requests).toHaveLength(1);

    const stageAfter = await bounds(page, '[data-board-stage]');
    const boardAfter = await bounds(page, '[data-board-stage] svg');
    const statusAfter = await bounds(page, '[data-game-status]');
    expectStableBox(stageBefore, stageAfter, 'board stage');
    expectStableBox(boardBefore, boardAfter, 'board');
    expectStableBox(statusBefore, statusAfter, 'status');

    const gameOverflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(gameOverflow).toBeLessThanOrEqual(1);

    const actionDock = page.locator('[data-action-dock]');
    if (viewport.width >= 1024 && viewport.height >= 640) {
      const documentOverflow = await page.evaluate(
        () => document.documentElement.scrollHeight - document.documentElement.clientHeight,
      );
      expect(documentOverflow).toBeLessThanOrEqual(1);
      const actionBox = await actionDock.boundingBox();
      expect(actionBox).not.toBeNull();
      expect(actionBox!.y + actionBox!.height).toBeLessThanOrEqual(viewport.height + 1);
    } else {
      await actionDock.scrollIntoViewIfNeeded();
      await expect(actionDock).toBeInViewport();
    }

    if (viewport.screenshot) {
      await expect(page).toHaveScreenshot(`game-${viewport.name}.png`, {
        fullPage: true,
      });
    }
  });
}

test('keeps the setup preview fixed while AI preparation completes', async ({ page }) => {
  await page.setViewportSize({ width: 768, height: 1024 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const ai = await installAiWorkerFixture(page, { holdPreparation: true });
  await openFreshSetup(page, false);
  await page.getByRole('button', { name: /^Double Deltrel/i }).click();
  const controller = page.getByRole('combobox', { name: 'Player 1 controller' });
  await expect(controller.locator('option[value="local"]')).toBeEnabled();
  await controller.selectOption('local');
  await expect(page.getByRole('button', { name: 'Begin the game' })).toBeDisabled();
  await expect(page.getByRole('progressbar', { name: 'AI preparation' })).toBeVisible();

  const before = await bounds(page, '[data-setup-preview]');
  ai.releasePreparation();
  await expect(page.getByRole('button', { name: 'Begin the game' })).toBeEnabled();
  await expect(page.getByRole('progressbar', { name: 'AI preparation' })).toBeHidden();
  const after = await bounds(page, '[data-setup-preview]');
  expectStableBox(before, after, 'setup preview');
});

test('keeps dialogs in bounds and restores focus on a small phone', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 568 });
  await installAiWorkerFixture(page);
  await openFreshSetup(page);
  await page.getByRole('button', { name: /^Mini, 4 rings$/i }).click();
  await page.getByRole('button', { name: 'Begin the game' }).click();

  const rulesButton = page.getByRole('button', { name: 'Rules' });
  await rulesButton.click();
  const dialog = page.getByRole('dialog', { name: 'How to play Deltrel' });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole('button', { name: 'Close rules' })).toBeFocused();
  await expect(page).toHaveScreenshot('rules-phone.png');
  const dialogBox = await dialog.boundingBox();
  expect(dialogBox).not.toBeNull();
  expect(dialogBox!.x).toBeGreaterThanOrEqual(0);
  expect(dialogBox!.y).toBeGreaterThanOrEqual(0);
  expect(dialogBox!.x + dialogBox!.width).toBeLessThanOrEqual(321);
  expect(dialogBox!.y + dialogBox!.height).toBeLessThanOrEqual(569);
  expect(await page.evaluate(() => document.body.style.overflow)).toBe('hidden');

  await dialog.getByRole('button', { name: 'Close rules' }).click();
  await expect(dialog).toBeHidden();
  await expect(rulesButton).toBeFocused();
  expect(await page.evaluate(() => document.body.style.overflow)).toBe('');
});

test('keeps the game-over result usable on a phone', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await installAiWorkerFixture(page);
  await openFreshSetup(page);
  await page.getByRole('button', { name: /^Mini, 4 rings$/i }).click();
  await page.getByRole('button', { name: 'Begin the game' }).click();
  await fillFourRingGame(page);

  const dialog = page.getByRole('dialog', { name: 'Game over' });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole('button', { name: 'Rematch' })).toBeFocused();
  await expect(page).toHaveScreenshot('game-over-phone.png');
  const dialogBox = await dialog.boundingBox();
  expect(dialogBox).not.toBeNull();
  expect(dialogBox!.x).toBeGreaterThanOrEqual(0);
  expect(dialogBox!.y).toBeGreaterThanOrEqual(0);
  expect(dialogBox!.x + dialogBox!.width).toBeLessThanOrEqual(391);
  expect(dialogBox!.y + dialogBox!.height).toBeLessThanOrEqual(845);
});

test('keeps the clinch decision and proof controls usable on a small phone', async ({
  page,
}) => {
  await page.setViewportSize({ width: 320, height: 568 });
  await installAiWorkerFixture(page);
  await openFreshSetup(page);
  await page.getByRole('button', { name: /^Mini, 4 rings$/i }).click();
  await page.getByRole('button', { name: 'Begin the game' }).click();
  await reachFourRingClinch(page);

  const dialog = page.getByRole('dialog', { name: /cannot be caught/i });
  await expect(dialog).toBeVisible();
  await expect(
    dialog.getByRole('button', { name: 'Continue playing' }),
  ).toBeFocused();
  const dialogBox = await dialog.boundingBox();
  expect(dialogBox).not.toBeNull();
  expect(dialogBox!.x).toBeGreaterThanOrEqual(0);
  expect(dialogBox!.y).toBeGreaterThanOrEqual(0);
  expect(dialogBox!.x + dialogBox!.width).toBeLessThanOrEqual(321);
  expect(dialogBox!.y + dialogBox!.height).toBeLessThanOrEqual(569);
  await expect(page).toHaveScreenshot('clinch-phone.png');

  await dialog.getByRole('button', { name: 'Show proof board' }).click();
  await expect(page.getByRole('region', { name: 'Clinch proof board' })).toBeVisible();
  const mobileControls = page.locator('[data-mobile-clinch-controls]');
  await expect(mobileControls).toBeVisible();
  const controlsBox = await mobileControls.boundingBox();
  expect(controlsBox).not.toBeNull();
  expect(controlsBox!.x).toBeGreaterThanOrEqual(0);
  expect(controlsBox!.x + controlsBox!.width).toBeLessThanOrEqual(321);
  expect(controlsBox!.y + controlsBox!.height).toBeLessThanOrEqual(569);
  await expect(page).toHaveScreenshot('clinch-proof-phone.png');

  await mobileControls.getByRole('button', { name: 'End now' }).click();
  const result = page.getByRole('dialog', { name: 'Game over' });
  await result.getByRole('button', { name: 'Review proof' }).click();
  await expect(page.getByRole('button', { name: 'Result' })).toHaveCount(1);
});
