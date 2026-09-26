import { expect, test, type Route } from '@playwright/test';
import { DELTREL_ACTION_LAYOUT_SCHEMA_ID, DELTREL_FEATURE_SCHEMA_ID, DELTREL_RULES_HASH, DELTREL_RULES_SCHEMA_ID } from '../src/lib/deltrel/rules';
import { DELTREL_FEATURE_SCHEMA_HASH, DELTREL_FEATURE_SCHEMA_VERSION } from '../src/lib/deltrel/ai/protocol';
import type { AnalyzeRequestV3 } from '../src/lib/deltrel/ai/server-client';

const health = {
  api_schema_version: 3, status: 'ok', model: { ready: true, model_version: 'cloud-fixture', model_step: 1 },
  rules: { schema_id: DELTREL_RULES_SCHEMA_ID, hash: DELTREL_RULES_HASH },
  features: { schema_id: DELTREL_FEATURE_SCHEMA_ID, hash: DELTREL_FEATURE_SCHEMA_HASH, version: DELTREL_FEATURE_SCHEMA_VERSION },
  actions: { schema_id: DELTREL_ACTION_LAYOUT_SCHEMA_ID },
  search: {
    defaults: { simulations: 544, max_considered: 16 },
    maximums: { simulations: 16384, max_considered: 128 },
    presets: { standard: { simulations: 544, max_considered: 16 }, deep: { simulations: 4096, max_considered: 64 } },
  },
};

async function answerMove(route: Route) {
  const request = route.request();
  const position = request.postDataJSON() as AnalyzeRequestV3;
  const node = position.stones.findIndex(stone => stone === -1);
  const action = { kind: 'place', code: node, node };
  await route.fulfill({ json: {
    schema_version: 3, request_id: request.headers()['x-request-id'], action,
    root_actions: [action], root_policy: [1], root_q: [0.6], root_visits: [position.search.simulations],
    outcome: { win: 0.8, loss: 0.2 }, value: 0.6, search_value: 0.6, root_value: 0.6,
    variant: { mode: position.mode, handicap: position.handicap, pie: position.pie },
    swap_available: position.swap_available, swap_recommended: false, history_known: true,
    score_belief: { support_min: -151, support_max: 151, expected_margin: 0, probabilities: Array.from({ length: 303 }, (_, index) => index === 151 ? 1 : 0) },
    model_version: 'cloud-fixture', model_step: 1,
    timing_ms: { queue: 0, model_reload: 0, inference_search: 1, total: 1 },
  } });
}

test('cloud play preserves an interrupted position, retries explicitly, and never loads the device engine', async ({ page }) => {
  let online = true;
  const searches: AnalyzeRequestV3[] = [];
  const localAssets: string[] = [];
  page.on('request', request => {
    if (request.method() === 'GET' && (/\.onnx(?:\?|$)/.test(request.url()) || request.url().includes('/onnxruntime/'))) localAssets.push(request.url());
  });
  // Cloud play also works on browsers that cannot run the local engine.
  await page.addInitScript(() => { Object.defineProperty(window, 'Worker', { value: undefined }); });
  await page.route('**/v2/health', route => online ? route.fulfill({ json: health })
    : route.fulfill({ status: 503, json: { error: { message: 'Cloud AI is offline.' } } }));
  await page.route('**/v2/move', async route => {
    searches.push(route.request().postDataJSON());
    if (searches.length === 1) {
      online = false;
      await route.fulfill({ status: 503, json: { error: { code: 'deltrel_ai_unavailable', message: 'Cloud AI is offline.', retryable: true } } });
    } else await answerMove(route);
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Mini, 4 rings' }).click();
  const first = page.getByRole('combobox', { name: 'Player 1 controller' });
  await expect(first.locator('option[value="server"]')).toBeEnabled();
  await first.selectOption('server');
  await expect(page.getByRole('combobox', { name: 'Player 2 controller' }).locator('option[value="server"]')).toHaveCount(0);
  await page.getByText('Custom cloud search', { exact: true }).click();
  await page.getByRole('spinbutton', { name: 'Simulations' }).fill('777');
  await page.getByRole('spinbutton', { name: 'Candidate moves' }).fill('21');
  await page.getByRole('button', { name: 'Apply custom search' }).click();
  await page.getByRole('button', { name: 'Begin the game' }).click();
  await expect(page.getByRole('main').getByRole('alert')).toContainText('Cloud AI is offline.');
  expect(searches).toHaveLength(1);
  expect(searches[0].search).toMatchObject({ simulations: 777, max_considered: 21 });
  await expect(page.locator('[data-move-chip]')).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Engine estimate' })).toHaveCount(0);
  online = true;
  await page.getByRole('button', { name: 'Check cloud availability' }).click();
  await expect(page.getByText('Cloud AI is ready')).toBeVisible();
  expect(searches).toHaveLength(1);
  await page.getByRole('button', { name: 'Retry', exact: true }).click();
  await expect(page.locator('[data-move-chip]')).toHaveCount(1);
  expect(searches).toHaveLength(2);
  await page.reload();
  await expect(page.locator('[data-move-chip]')).toHaveCount(1);
  await expect(page.getByText('Cloud AI is ready')).toBeVisible();
  await expect(page.getByLabel('Selected strength: Custom')).toBeVisible();
  await expect(page.getByRole('region', { name: 'Engine estimate' })).toHaveCount(0);
  expect(localAssets).toEqual([]);
  expect(searches).toHaveLength(2);
});

test('offline cloud availability leaves human play available', async ({ page }) => {
  await page.route('**/v2/health', route => route.fulfill({ status: 503, json: { error: { message: 'Offline' } } }));
  await page.goto('/');
  await expect(page.getByText('Cloud AI is currently unavailable')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Play against Cloud AI' })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Begin the game' })).toBeEnabled();
  await page.getByRole('button', { name: 'Begin the game' }).click();
  await expect(page.getByRole('main')).toContainText('Player 1');
  await expect(page.getByRole('main').getByRole('alert')).toHaveCount(0);
});
