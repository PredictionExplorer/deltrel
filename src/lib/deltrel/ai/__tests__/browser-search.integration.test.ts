import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { beforeAll, describe, expect, it, vi } from 'vitest';
import * as ort from 'onnxruntime-web';
import fc from 'fast-check';
import publishedManifest from '../../../../../public/models/deltrel/manifest.json';
import { parseDeltrelBrowserModelManifest } from '../manifest';
import { browserStrengthOptions } from '../browser-strength';
import { buildAiRequest, type AtomicGameAction, type DeltrelAiRequest } from '../protocol';
import { parseWorkerCommand, parseWorkerEvent, type LocalAiSearchProgress } from '../worker-protocol';
import { validateNetworkOutputState } from '../network-output';
import { getBoard } from '../../board';

let worker: typeof import('@/workers/deltrel-ai.worker');
let wasm: Parameters<typeof worker.replayAndVerify>[1];
const manifest = parseDeltrelBrowserModelManifest(publishedManifest);

beforeAll(async () => {
  vi.stubGlobal('postMessage', vi.fn());
  vi.stubGlobal('addEventListener', vi.fn());
  worker = await import('@/workers/deltrel-ai.worker');
  const moduleUrl = pathToFileURL(resolve('public', manifest.wasm.moduleUrl.slice(1))).href;
  wasm = await import(/* @vite-ignore */ moduleUrl);
  await wasm.default({ module_or_path: readFileSync(resolve('public', manifest.wasm.binaryUrl.slice(1))) });
});

/** Real exported WASM and production worker; deterministic tensors isolate search mechanics. */
function fixture() {
  const run = vi.fn(async (feeds: Record<string, ort.Tensor>) => {
    const batch = feeds.node_features.dims[0];
    const nodes = feeds.node_features.dims[1];
    const head = (dims: number[]) => new ort.Tensor('float16', new Uint16Array(dims.reduce((a, b) => a * b, 1)), dims);
    return {
      policy_logits: head([batch, nodes]), outcome_logits: head([batch, 2]),
      score_margin_logits: head([batch, 303]), ownership_logits: head([batch, nodes, 3]),
      alive_logits: head([batch, nodes]), soft_policy_logits: head([batch, nodes]),
      opponent_reply_logits: head([batch, nodes + 1]), second_stone_logits: head([batch, nodes]),
      final_shores_logits: head([batch, 2, 51]), final_networks_logits: head([batch, 2, 26]),
      final_capes_logits: head([batch, 2, 6]),
    };
  });
  const runtime = { manifest, wasm, ort, session: { run }, predictions: new worker.PredictionCache() };
  const search = async (request: DeltrelAiRequest, simulations: number, maxConsidered: number) => {
    // Exercise the worker boundary too: handicap openings have more than two moves left.
    parseWorkerCommand({ type: 'choose', taskId: request.requestId, request, search: { simulations, maxConsidered } });
    const root = worker.replayAndVerify(request, wasm);
    try {
      const progress: LocalAiSearchProgress[] = [];
      const result = await worker.runTreeSearch(runtime as never, root, request.state,
        { simulations, maxConsidered }, () => {}, async () => {}, completed => {
          parseWorkerEvent({ type: 'search-progress', taskId: request.requestId, progress: completed });
          progress.push(completed);
        });
      expect(progress).toEqual(Array.from({ length: simulations }, (_, index) => ({
        completedSimulations: index + 1, totalSimulations: simulations,
      })));
      expect(result.rootVisits.reduce((sum, count) => sum + count, 0)).toBe(simulations);
      expect(result.rootActions).toEqual(request.legalActions);
      expect(request.legalActions).toContain(result.actionCode);
      expect(result.rootPolicy.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 5);
      validateNetworkOutputState(result.networkOutput, request, result.swapRecommended);
      return result;
    } finally { root.free?.(); }
  };
  return { run, runtime, search };
}

describe('published browser WASM search', () => {
  it('performs the full Quick, Balanced and Deep budgets, including immediate cached repeats', async () => {
    expect(worker.hasExpectedWasmSearch(wasm)).toBe(true);
    const f = fixture();
    const request = buildAiRequest({ rings: 4, mode: 'double', pieRule: true, playerNames: ['A', 'B'] }, []);
    const maximum = { simulations: manifest.search.maximumSimulations, maxConsidered: manifest.search.maximumMaxConsidered };
    for (const option of browserStrengthOptions(maximum)) {
      const before = f.run.mock.calls.length;
      const first = await f.search(request, option.budget.simulations, option.budget.maxConsidered);
      const after = f.run.mock.calls.length;
      expect(after).toBeGreaterThan(before); // Higher budgets must expand further than their cached predecessors.
      expect(first.rootVisits.filter(visits => visits > 0)).toHaveLength(option.budget.maxConsidered);
      const repeat = await f.search(request, option.budget.simulations, option.budget.maxConsidered);
      expect(repeat).toEqual(first);
      expect(f.run).toHaveBeenCalledTimes(after); // Cached inference never removes search visits.
    }
  });

  it.each([4, 6, 8, 10].flatMap(rings => ['classic', 'double'].map(mode => ({ rings, mode: mode as 'classic' | 'double' }))))(
    'completes every nine-stone AI opening and hands over the turn ($rings rings, $mode)', async ({ rings, mode }) => {
      const f = fixture();
      const config = { rings, mode, handicap: 9, pieRule: false, playerNames: ['A', 'B'] as [string, string] };
      const actions: AtomicGameAction[] = [];
      for (let index = 0; index < 9; index++) {
        const request = buildAiRequest(config, actions);
        expect(request.state).toMatchObject({ toMove: 0, opening: true, movesLeft: 9 - index });
        const result = await f.search(request, 8, 4);
        expect(result.swapRecommended).toBe(false);
        actions.push({ type: 'place', node: result.actionCode });
      }
      const reply = buildAiRequest(config, actions);
      expect(reply.state).toMatchObject({ toMove: 1, opening: false, movesLeft: mode === 'classic' ? 1 : 2 });
      expect(reply.state.stones.filter(owner => owner === 0)).toHaveLength(9);
      expect(reply.state.history.handicapStones).toHaveLength(9);
      await f.search(reply, 8, 4);
    },
  );

  it('still consumes Deep search visits when every continuation is a terminal result', async () => {
    const f = fixture();
    const actions: AtomicGameAction[] = Array.from({ length: 49 }, (_, node) => ({ type: 'place', node }));
    const request = buildAiRequest({ rings: 4, mode: 'classic', pieRule: false, playerNames: ['A', 'B'] }, actions);
    const result = await f.search(request, 64, 8);
    expect(result.actionCode).toBe(49);
    expect(result.rootVisits).toEqual([64]);
    expect(f.run).toHaveBeenCalledOnce(); // A terminal continuation is scored exactly, without a neural forward.
  });

  it('agrees with sequential session execution across reproducible random legal game histories', async () => {
    await fc.assert(fc.asyncProperty(
      fc.constantFrom(4, 6, 8, 10), fc.constantFrom('classic' as const, 'double' as const),
      fc.integer({ min: 1, max: 9 }), fc.boolean(), fc.boolean(),
      fc.nat(300), fc.integer({ min: 1, max: 64 }), fc.integer({ min: 1, max: 12 }),
      fc.array(fc.nat(), { minLength: 275, maxLength: 275 }),
      async (rings, mode, handicap, pie, swap, placementCount, simulations, maxConsidered, priorities) => {
        const nodes = getBoard(rings).n;
        const config = { rings, mode, handicap, pieRule: handicap === 1 && pie,
          playerNames: ['A', 'B'] as [string, string] };
        const count = placementCount % nodes;
        const order = Array.from({ length: nodes }, (_, node) => node)
          .sort((left, right) => priorities[left] - priorities[right] || left - right);
        const actions: AtomicGameAction[] = order.slice(0, count).map(node => ({ type: 'place', node }));
        if (config.pieRule && swap && count > 0) actions.splice(1, 0, { type: 'swap' });
        const request = buildAiRequest(config, actions);
        const f = fixture();
        const standard = await f.search(request, simulations, maxConsidered);
        const root = worker.replayAndVerify(request, wasm);
        try {
          const progress: LocalAiSearchProgress[] = [];
          const session = await worker.runSessionSearch(f.runtime as never, root, request.state,
            { simulations, maxConsidered }, () => {}, async () => {}, completed => progress.push(completed));
          expect(progress.at(-1)).toEqual({ completedSimulations: simulations, totalSimulations: simulations });
          progress.forEach((completed, index) => {
            expect(completed.totalSimulations).toBe(simulations);
            expect(completed.completedSimulations).toBeGreaterThan(index ? progress[index - 1].completedSimulations : 0);
          });
          for (const field of ['actionCode', 'swapRecommended', 'rootValue', 'rootActions', 'rootQ', 'rootPolicy', 'rootVisits'] as const) {
            expect(session[field]).toEqual(standard[field]);
          }
          expect(session.rootVisits.reduce((sum, visits) => sum + visits, 0)).toBe(simulations);
        } finally { root.free?.(); }
      },
    ), { seed: 902109, numRuns: 40 });
  });
});
