import { expect, type Page } from '@playwright/test';

/** Use the same supported strength controls as a player. */
export async function selectAiStrength(page: Page, strength: 'Standard' | 'Deep') {
  const choice = page.getByRole('button', { name: `${strength} AI strength`, exact: true });
  await choice.click();
  await expect(choice).toHaveAttribute('aria-pressed', 'true');
}
