import { expect, test, type Page } from '@playwright/test';
import type { AnalyzeRequestV3 } from '../src/lib/star/ai/server-client';

const health = {
  status: 'ok',
  service_version: '2.0.0',
  api_schema_version: 3,
  model: {
    ready: true,
    role: 'champion',
    model_version: 'e2e-full-champion',
    model_step: 185554,
    model_identity: `sha256:${'a'.repeat(64)}`,
  },
  search: {
    defaults: { simulations: 512, max_considered: 16 },
    maximums: { simulations: 4096, max_considered: 32 },
    presets: {
      quick: { simulations: 64, max_considered: 8 },
      strong: { simulations: 512, max_considered: 16 },
      maximum: { simulations: 4096, max_considered: 32 },
    },
  },
  rules: {
    schema_id: 'edgeconnect.star.rules.v3',
    version: 3,
    hash: 'fnv1a64:a5d932b0ef8354e8',
  },
  features: {
    schema_id: 'edgeconnect.star.model-features.external.v3',
    version: 4,
    hash: 'cb0e1e89a6ce3540',
  },
  actions: {
    schema_id: 'edgeconnect.star.action-layout.nodes-only.v1',
    types: ['place', 'swap'],
  },
};

async function mockChampion(page: Page, swap = false) {
  const requests: AnalyzeRequestV3[] = [];
  await page.route('**/v2/health', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(health) }),
  );
  await page.route('**/v2/move', async (route) => {
    const body = route.request().postDataJSON() as AnalyzeRequestV3;
    requests.push(body);
    const action = body.stones.findIndex((stone) => stone === -1);
    const score = new Array<number>(303).fill(0);
    score[151] = 1;
    const requestId = route.request().headers()['x-request-id'];
    if (!requestId) throw new Error('server request omitted X-Request-ID');
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      headers: { 'X-Request-ID': requestId },
      body: JSON.stringify({
        schema_version: 3,
        request_id: requestId,
        action: { code: action, kind: 'place', node: action },
        root_actions: [{ code: action, kind: 'place', node: action }],
        root_policy: [1],
        root_q: [0.25],
        root_visits: [body.search.simulations],
        outcome: { loss: 0.25, win: 0.75 },
        value: 0.5,
        search_value: 0.25,
        root_value: 0.2,
        variant: { mode: body.mode, handicap: body.handicap, pie: body.pie },
        swap_available: body.swap_available,
        // The public API has placement-only policy rows; swapping is a
        // separate recommendation consumed by the browser's action adapter.
        swap_recommended: swap && body.swap_available,
        history_known: true,
        score_belief: {
          support_min: -151,
          support_max: 151,
          expected_margin: 0,
          probabilities: score,
        },
        model_version: health.model.model_version,
        model_step: health.model.model_step,
        timing_ms: {
          queue: 0,
          model_reload: 0,
          inference_search: 1,
          total: 1,
        },
      }),
    });
  });
  return requests;
}

async function openFreshSetup(page: Page) {
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await expect(page.getByRole('heading', { name: '✳Star' })).toBeVisible();
}

async function selectGame(
  page: Page,
  mode: 'classic' | 'double',
  opening: 'Even (pie)' | 'Handicap',
) {
  await page.getByRole('textbox', { name: 'Player 1 name' }).fill('Ada');
  await page.getByRole('textbox', { name: 'Player 2 name' }).fill('Champion');
  await page.getByRole('button', {
    name: mode === 'classic' ? /^Classic \*Star,/ : /^Double \*Star,/,
  }).click();
  const rings = opening === 'Handicap' ? 10 : 4;
  await page.getByRole('button', {
    name: opening === 'Handicap' ? 'Full, 10 rings' : 'Mini, 4 rings',
    exact: true,
  }).click();
  await page.getByRole('button', { name: `${opening} opening`, exact: true }).click();
  if (opening === 'Handicap') {
    await page.getByRole('radio', { name: '4 handicap stones', exact: true }).click();
  }
  await expect(page.locator('[data-selected-game-mode]')).toHaveText(
    `${mode === 'classic' ? 'Classic' : 'Double'} · ${opening === 'Handicap' ? '4-stone handicap' : opening} · ${rings} rings`,
  );
}

async function chooseChampionOpponent(page: Page) {
  await expect(page.getByRole('status').filter({ hasText: 'Ready to play' })).toBeVisible();
  await page.getByRole('button', { name: 'Play against the champion', exact: true }).click();
  await expect(page.getByRole('combobox', { name: 'Player 1 controller' })).toHaveValue('human');
  await expect(page.getByRole('combobox', { name: 'Player 2 controller' })).toHaveValue('server');
  await page.getByRole('button', { name: 'Strong champion search', exact: true }).click();
}

async function placeHumanOpening(page: Page, stones: number) {
  for (let index = 0; index < stones; index += 1) {
    await page.getByRole('button', { name: /empty .*Ada may place here/i }).first().click();
  }
}

test('a compatible server AI capability drives one validated atomic move', async ({ page }) => {
  const requests = await mockChampion(page);
  await openFreshSetup(page);

  await page.getByRole('button', { name: /Double \*Star/ }).click();
  const playerOneController = page.getByRole('combobox', {
    name: 'Player 1 controller',
  });
  await expect(playerOneController.locator('option[value="server"]')).toBeEnabled();
  await playerOneController.selectOption('server');
  await page.getByRole('button', { name: 'Begin the game' }).click();

  await expect(page.getByRole('button', { name: /stone on/ })).toHaveCount(1);
  expect(requests).toHaveLength(1);
});

for (const mode of ['classic', 'double'] as const) {
  test(`${mode} resets a Full-board handicap to pie when switching to Mini`, async ({ page }) => {
    const requests = await mockChampion(page);
    await openFreshSetup(page);
    await selectGame(page, mode, 'Handicap');
    await page.getByRole('button', { name: 'Mini, 4 rings', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Standard opening', exact: true })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Handicap opening', exact: true })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Even (pie) opening', exact: true })).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByRole('radiogroup', { name: 'Handicap stones' })).toHaveCount(0);
    await chooseChampionOpponent(page);
    await page.getByRole('button', { name: 'Begin the game' }).click();
    await placeHumanOpening(page, 1);
    await expect(page.getByText('Ada to play', { exact: true })).toBeVisible();
    expect(requests[0]).toMatchObject({ rings: 4, mode, handicap: 1, pie: true, swap_available: true });
  });

  for (const opening of ['Even (pie)', 'Handicap'] as const) {
    test(`${mode} ${opening.toLowerCase()} opens against the full champion with the selected rules`, async ({ page }) => {
      const requests = await mockChampion(page);
      await openFreshSetup(page);
      await selectGame(page, mode, opening);
      await chooseChampionOpponent(page);
      await page.getByRole('button', { name: 'Begin the game' }).click();

      const openingStones = opening === 'Handicap' ? 4 : 1;
      await placeHumanOpening(page, openingStones);
      const replyStones = mode === 'classic' ? 1 : 2;
      await expect(page.getByRole('button', { name: /stone on/ })).toHaveCount(openingStones + replyStones);
      await expect(page.getByText('Ada to play', { exact: true })).toBeVisible();
      expect(requests).toHaveLength(replyStones);
      expect(requests[0]).toMatchObject({
        rings: opening === 'Handicap' ? 10 : 4,
        mode,
        handicap: openingStones,
        pie: opening === 'Even (pie)',
        opening: false,
        to_move: 1,
        moves_left: replyStones,
        swap_available: opening === 'Even (pie)',
        swapped: false,
        pda: 0,
        search: { simulations: 512, max_considered: 16 },
      });
      expect(requests[0].stones.filter((stone) => stone === 0)).toHaveLength(openingStones);
      expect(requests[0].history.handicap_stones).toHaveLength(openingStones);
      if (mode === 'double') {
        expect(requests[1]).toMatchObject({ to_move: 1, moves_left: 1, swap_available: false });
        expect(requests[1].history.current_turn).toHaveLength(1);
      }
      // Next's route announcer lives outside the game and also uses role=alert.
      await expect(page.getByRole('main').getByRole('alert')).toHaveCount(0);
    });
  }

  test(`${mode} pie honors the champion's swap and keeps it after reload`, async ({ page }) => {
    const requests = await mockChampion(page, true);
    await openFreshSetup(page);
    await selectGame(page, mode, 'Even (pie)');
    await chooseChampionOpponent(page);
    await page.getByRole('button', { name: 'Begin the game' }).click();
    await placeHumanOpening(page, 1);

    const swap = page.getByRole('button', { name: 'Go to move 2: Champion swapped sides', exact: true });
    await expect(swap).toBeVisible();
    await expect(page.getByRole('button', { name: /Champion stone on/ })).toHaveCount(1);
    await expect(page.getByText('Ada to play', { exact: true })).toBeVisible();
    expect(requests).toHaveLength(1);
    expect(requests[0]).toMatchObject({ mode, pie: true, handicap: 1, swap_available: true });

    await page.reload();
    await expect(swap).toBeVisible();
    await expect(page.getByRole('button', { name: /Champion stone on/ })).toHaveCount(1);
    await expect(page.getByText('Ada to play', { exact: true })).toBeVisible();
    expect(requests).toHaveLength(1);
  });

  test(`${mode} champion completes all four handicap opening placements`, async ({ page }) => {
    const requests = await mockChampion(page);
    await openFreshSetup(page);
    await selectGame(page, mode, 'Handicap');
    const first = page.getByRole('combobox', { name: 'Player 1 controller' });
    await expect(first.locator('option[value="server"]')).toBeEnabled();
    await first.selectOption('server');
    await page.getByRole('combobox', { name: 'Player 2 controller' }).selectOption('human');
    await page.getByRole('button', { name: 'Begin the game' }).click();

    await expect(page.getByRole('button', { name: /Ada stone on/ })).toHaveCount(4);
    await expect(page.getByText('Champion to play', { exact: true })).toBeVisible();
    expect(requests).toHaveLength(4);
    requests.forEach((request, index) => {
      expect(request).toMatchObject({ mode, handicap: 4, pie: false, to_move: 0, opening: true, moves_left: 4 - index });
      expect(request.history.handicap_stones).toHaveLength(index);
    });
  });
}

test('maximum thinking time changes search effort while preserving the champion', async ({ page }) => {
  const requests = await mockChampion(page);
  await openFreshSetup(page);
  await selectGame(page, 'classic', 'Even (pie)');
  await chooseChampionOpponent(page);
  await page.getByRole('button', { name: 'Maximum champion search', exact: true }).click();
  await expect(page.getByText(/Full trained model · step 185,554/)).toBeVisible();
  await page.getByRole('button', { name: 'Begin the game' }).click();
  await placeHumanOpening(page, 1);
  await expect(page.getByText('Ada to play', { exact: true })).toBeVisible();
  expect(requests).toHaveLength(1);
  expect(requests[0].search).toMatchObject({ simulations: 4096, max_considered: 32 });
});
