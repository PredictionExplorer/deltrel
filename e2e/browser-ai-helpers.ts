import { expect, type Page } from '@playwright/test';

/** Bound real-model test work independently of the public strength presets. */
export async function setAiSearchBudget(page: Page, simulations: number, maxConsidered: number) {
  const summary = page.getByText('Custom search budget', { exact: true });
  const details = summary.locator('..');
  if (await details.getAttribute('open') === null) await summary.click();
  await page.getByRole('spinbutton', { name: 'Simulations', exact: true }).fill(String(simulations));
  await page.getByRole('spinbutton', { name: 'Candidate moves', exact: true }).fill(String(maxConsidered));
  const apply = page.getByRole('button', { name: 'Apply custom budget', exact: true });
  if (await apply.isEnabled()) await apply.click();
  await expect(apply).toBeDisabled();
  await summary.click();
}
