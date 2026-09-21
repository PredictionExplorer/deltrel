import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';
import type { AnalyzeRequestV3 } from '../src/lib/deltrel/ai/server-client';
import { DELTREL_RULES_HASH, DELTREL_RULES_SCHEMA_ID, DELTREL_FEATURE_SCHEMA_ID, DELTREL_ACTION_LAYOUT_SCHEMA_ID } from '../src/lib/deltrel/rules';
import { DELTREL_FEATURE_SCHEMA_HASH } from '../src/lib/deltrel/ai/protocol';

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => { resolve = done; });
  return { promise, resolve };
}

function head(shape: number[], mask: boolean[], sigmoid = false, probabilities?: number[]) {
  const width = shape.at(-1)!;
  const values = probabilities ?? mask.map((active, i) => active
    ? sigmoid ? 0.5 : 1 / mask.slice(Math.floor(i / width) * width, (Math.floor(i / width) + 1) * width).filter(Boolean).length
    : 0);
  return {
    shape, mask, activation: sigmoid ? 'sigmoid' : 'softmax', applicable: true,
    probabilities: values,
    logits: mask.map((active, i) => active ? probabilities ? Math.log(values[i]) : 0 : null),
  };
}

function response(body: AnalyzeRequestV3, requestId: string) {
  const n = body.stones.length;
  const empty = body.stones.map((stone) => stone === -1);
  const all = (length: number) => Array<boolean>(length).fill(true);
  const action = body.stones.indexOf(-1);
  const outcome = { loss: 0.2, win: 0.8 };
  const heads = {
    policy: head([n], empty), outcome: head([2], all(2), false, [0.2, 0.8]),
    score_margin: head([303], all(303)), ownership: head([n, 3], all(n * 3)),
    alive: head([n], all(n), true), soft_policy: head([n], empty),
    opponent_reply: { ...head([n + 1], [...empty, true]), applicable: empty.filter(Boolean).length > body.moves_left },
    second_stone: { ...head([n], empty), applicable: body.mode === 'double' && !body.opening && body.moves_left === 2 },
    final_shores: head([2, 51], Array.from({ length: 102 }, (_, i) => i % 51 <= body.rings * 5)),
    final_networks: head([2, 26], Array.from({ length: 52 }, (_, i) => i % 26 <= Math.floor(body.rings * 5 / 2))),
    final_capes: head([2, 6], all(12)),
  };
  return {
    schema_version: 3, request_id: requestId,
    action: { code: action, kind: 'place', node: action },
    root_actions: [{ code: action, kind: 'place', node: action }],
    root_policy: [1], root_q: [0.2], root_visits: [body.search.simulations],
    outcome, value: 0.6, search_value: 0.2, root_value: 0.2,
    variant: { mode: body.mode, handicap: body.handicap, pie: body.pie },
    swap_available: body.swap_available, swap_recommended: false, history_known: true,
    score_belief: { support_min: -151, support_max: 151, expected_margin: 0, probabilities: heads.score_margin.probabilities },
    model_version: 'inspection-fixture', model_step: 42,
    timing_ms: { queue: 0, model_reload: 0, inference_search: 1, total: 1 },
    network_output: { schema_version: 1, perspective: body.to_move, node_count: n, auxiliary_status: 'ready', heads },
  };
}

async function setup(page: Page) {
  await page.route('**/v2/health', (route) => route.fulfill({ json: {
    status: 'ok', api_schema_version: 3,
    model: { ready: true, model_version: 'inspection-fixture', model_step: 42 },
    rules: { schema_id: DELTREL_RULES_SCHEMA_ID, hash: DELTREL_RULES_HASH },
    features: { schema_id: DELTREL_FEATURE_SCHEMA_ID, version: 4, hash: DELTREL_FEATURE_SCHEMA_HASH },
    actions: { schema_id: DELTREL_ACTION_LAYOUT_SCHEMA_ID },
    network_output_schema_version: 1,
  } }));
  await page.goto('/');
  await page.getByRole('button', { name: 'Mini, 4 rings' }).click();
  await page.getByRole('textbox', { name: 'Player 1 name' }).fill('Ada');
  await page.getByRole('textbox', { name: 'Player 2 name' }).fill('Grace');
}

test('all network outputs are inspectable without playing a move', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  let calls = 0;
  await page.route('**/v2/move', async (route) => {
    const body = route.request().postDataJSON() as AnalyzeRequestV3;
    expect(body.include_network_output).toBe(true);
    const requestId = route.request().headers()['x-request-id'];
    calls++;
    await route.fulfill({ json: response(body, requestId), headers: { 'X-Request-ID': requestId } });
  });
  await setup(page);
  await page.getByRole('button', { name: 'Begin the game' }).click();
  const panel = page.getByRole('region', { name: 'Engine estimate' });
  expect((await panel.boundingBox())!.height).toBeGreaterThan(200);
  await panel.getByRole('button', { name: 'Analyze position' }).click();
  await expect(panel.getByRole('article', { name: 'Ada forecast' })).toContainText('80.0%');
  await expect(panel.getByRole('article', { name: 'Grace forecast' })).toContainText('20.0%');
  await expect(panel.getByRole('article', { name: 'Ada forecast' })).toContainText('10.5');
  await expect(page.getByRole('group', { name: /0 of 50 nodes occupied/ })).toBeVisible();
  expect(calls).toBe(1);

  await panel.getByRole('button', { name: /^All network outputs/ }).click();
  const chooser = panel.getByLabel('Network output', { exact: true });
  await expect(chooser.locator('option')).toHaveCount(11);
  await chooser.selectOption('ownership');
  await expect(panel.getByRole('table', { name: 'Ownership at every point' })).toBeVisible();
  await panel.getByLabel('Find a point or value').fill('G4');
  await expect(panel.getByRole('table', { name: 'Ownership at every point' }).getByRole('row')).toHaveCount(2);
  await panel.getByLabel('Show raw logits').check();
  await expect(panel.getByText('logit +0.00000')).toHaveCount(3);
  await chooser.selectOption('scoreMargin');
  await expect(panel.getByRole('img', { name: 'Final score margin probability distribution' })).toBeVisible();
  await expect(panel.getByText('1–12 of 303')).toBeVisible();
  await panel.getByRole('button', { name: /^Raw engine output/ }).click();
  const raw = JSON.parse(await panel.getByRole('textbox', { name: 'Raw engine output' }).inputValue());
  expect(Object.keys(raw.analysis.networkOutput.heads)).toHaveLength(11);
  expect(raw.analysis.networkOutput.heads.scoreMargin.probabilities).toHaveLength(303);
  expect(raw.analysis.networkOutput.heads.ownership.probabilities).toHaveLength(150);
  expect(raw.context.applied).toBe(false);
  expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1);
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
});

test('AI-versus-AI forecasts persist while paused and manual inspection stays read-only', async ({ page }) => {
  const secondGate = deferred(), secondDone = deferred(), fourthGate = deferred(), fourthDone = deferred();
  const cancelled = new Set<number>();
  const requests: AnalyzeRequestV3[] = [];
  await page.route('**/v2/move', async (route) => {
    const body = route.request().postDataJSON() as AnalyzeRequestV3;
    requests.push(body);
    const call = requests.length;
    const requestId = route.request().headers()['x-request-id'];
    if (call === 2) await secondGate.promise;
    if (call === 4) await fourthGate.promise;
    try { await route.fulfill({ json: response(body, requestId), headers: { 'X-Request-ID': requestId } }); }
    catch (error) { if (!cancelled.has(call)) throw error; }
    finally { if (call === 2) secondDone.resolve(); if (call === 4) fourthDone.resolve(); }
  });
  await setup(page);
  await page.getByRole('combobox', { name: 'Player 1 controller' }).selectOption('server');
  await page.getByRole('combobox', { name: 'Player 2 controller' }).selectOption('server');
  await page.getByRole('button', { name: 'Begin the game' }).click();
  const panel = page.getByRole('region', { name: 'Engine estimate' });
  await expect(panel.getByRole('article', { name: 'Ada forecast' })).toContainText('80.0%');
  await expect.poll(() => requests.length).toBe(2);
  await panel.getByRole('button', { name: /^All network outputs/ }).click();
  await panel.getByRole('button', { name: 'Pause AI' }).click();
  cancelled.add(2); secondGate.resolve(); await secondDone.promise;
  await expect(page.getByRole('img', { name: /1 of 50 nodes occupied/ })).toBeVisible();
  await expect(panel.getByRole('button', { name: /^All network outputs/ })).toHaveAttribute('aria-expanded', 'true');
  await panel.getByRole('button', { name: 'Analyze position' }).click();
  await expect(panel.getByRole('article', { name: 'Grace forecast' })).toContainText('80.0%');
  await expect(panel.getByText(/After move 1 · Grace to move/)).toBeVisible();
  await expect(page.getByRole('img', { name: /1 of 50 nodes occupied/ })).toBeVisible();
  await page.getByRole('button', { name: 'Resume AI' }).click();
  await expect.poll(() => requests.length).toBe(4);
  expect(requests[3].stones).toEqual(requests[2].stones);
  await panel.getByRole('button', { name: 'Pause AI' }).click();
  cancelled.add(4); fourthGate.resolve(); await fourthDone.promise;
  await expect(page.getByRole('img', { name: /1 of 50 nodes occupied/ })).toBeVisible();
});
