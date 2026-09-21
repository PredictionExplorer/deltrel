import { expect, test, type Page } from '@playwright/test';

async function startMiniDoubleGame(page: Page) {
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.getByRole('button', { name: /^Double Deltrel/i }).click();
  await page.getByRole('button', { name: /^Mini, 4 rings$/i }).click();
  await page.getByRole('button', { name: 'Begin the game' }).click();
  await expect(
    page.getByRole('group', { name: /Deltrel board with 4 rings/i }),
  ).toBeVisible();
}

async function place(page: Page, label: string) {
  await page
    .getByRole('button', {
      name: new RegExp(
        `^Node ${label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}, empty`,
        'i',
      ),
    })
    .click();
}

test.beforeEach(async ({ page }) => {
  await startMiniDoubleGame(page);
});

test('projected territory does not cross out rescuable stones', async ({ page }) => {
  await place(page, 'E4');
  await place(page, 'I1');
  await place(page, 'G1');

  await expect(page.locator('[data-provably-dead-stone]')).toHaveCount(0);
  await expect(
    page.getByRole('button', {
      name: /Node E4, Player 1 stone.*not currently part of a living network/i,
    }),
  ).toBeVisible();
  await expect(page.locator('[data-stone-node="1"]')).toHaveAttribute(
    'opacity',
    '1',
  );

  await page.getByRole('checkbox', { name: 'Show influence' }).check();
  await expect(page.locator('[data-stone-node="1"]')).toHaveAttribute(
    'opacity',
    '0.35',
  );
  await expect(page.locator('[data-provably-dead-stone]')).toHaveCount(0);
});

test('a walled group is crossed only once rescue becomes impossible', async ({
  page,
}) => {
  await place(page, 'E1');
  await place(page, 'F1');
  await place(page, 'E2');
  await place(page, 'D8');
  await place(page, 'E9');
  await place(page, 'D2');

  await expect(page.locator('[data-provably-dead-stone="33"]')).toHaveCount(0);
  await place(page, 'C1');
  await expect(page.locator('[data-provably-dead-stone="33"]')).toBeVisible();
  await expect(
    page.getByRole('button', { name: /Node E1.*provably dead/i }),
  ).toBeVisible();

  await page.getByRole('button', { name: 'Undo' }).click();
  await expect(page.locator('[data-provably-dead-stone="33"]')).toHaveCount(0);
  await page.getByRole('button', { name: 'Redo' }).click();
  await expect(page.locator('[data-provably-dead-stone="33"]')).toBeVisible();
});
