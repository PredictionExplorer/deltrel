import { beforeAll, describe, expect, it, vi } from 'vitest';
import type * as Ort from 'onnxruntime-web';
import type { WasmState } from '@/workers/deltrel-ai.worker';
import {
  DELTREL_GLOBAL_FEATURE_DIM,
  DELTREL_MODEL_INPUT_NAMES,
  DELTREL_MODEL_OUTPUT_NAMES,
  DELTREL_NODE_FEATURE_DIM,
  float32ToFloat16Array,
} from '../features';
import { buildAiRequest } from '../protocol';
import publishedManifest from '../../../../../public/models/deltrel/manifest.json';
import { parseDeltrelBrowserModelManifest } from '../manifest';
import { DELTREL_AI_WORKER_PROTOCOL_VERSION } from '../worker-protocol';

let registeredHandler: ((event: MessageEvent<unknown>) => void) | undefined;
let readyEvent: unknown;
const postMessage = vi.fn((message: unknown) => {
  readyEvent = message;
});
const addEventListener = vi.fn(
  (_type: string, listener: (event: MessageEvent<unknown>) => void) => {
    registeredHandler = listener;
  },
);
let runtime: typeof import('@/workers/deltrel-ai.worker');

beforeAll(async () => {
  vi.stubGlobal('postMessage', postMessage);
  vi.stubGlobal('addEventListener', addEventListener);
  runtime = await import('@/workers/deltrel-ai.worker');
});

describe('local worker runtime contract', () => {
  it('uses published defaults while accepting explicit custom search beyond preset limits', () => {
    const manifest = parseDeltrelBrowserModelManifest(publishedManifest);
    expect(runtime.resolveBrowserSearchBudget(manifest, null)).toEqual({ simulations: 512, maxConsidered: 16 });
    expect(manifest.search).toMatchObject({ maximumSimulations: 4_096, maximumMaxConsidered: 64 });
    expect(runtime.resolveBrowserSearchBudget(manifest, { simulations: 4_096, maxConsidered: 256 }))
      .toEqual({ simulations: 4_096, maxConsidered: 256 });
    expect(() => runtime.resolveBrowserSearchBudget(manifest, { simulations: 536_870_912, maxConsidered: 8 }))
      .toThrow(/search budget/);
  });

  it('invalidates a prepared runtime when release settings change without new weights', () => {
    const manifest = parseDeltrelBrowserModelManifest(publishedManifest);
    expect(runtime.sameRuntimeManifest(manifest, structuredClone(manifest))).toBe(true);
    for (const update of [
      { search: { ...manifest.search, maximumSimulations: 32 } },
      { search: { ...manifest.search, simulations: 16 } },
      { search: { ...manifest.search, firstVisitBatchSize: 2 } },
      { search: { ...manifest.search, subtreeReuse: true } },
      { search: { ...manifest.search, swapDeadZone: 0.1 } },
      { auxiliaryStatus: 'untrained' as const },
      { modelStep: (manifest.modelStep ?? 0) + 1 },
    ]) {
      expect(runtime.sameRuntimeManifest(manifest, { ...manifest, ...update })).toBe(false);
    }
  });

  it('registers exactly one message handler and announces readiness', () => {
    expect(registeredHandler).toEqual(expect.any(Function));
    expect(readyEvent).toEqual({ type: 'ready', protocolVersion: DELTREL_AI_WORKER_PROTOCOL_VERSION });
  });

  it('rejects older WASM search behavior and versions both cached assets', () => {
    expect(runtime.hasExpectedWasmSearch({})).toBe(false);
    expect(runtime.hasExpectedWasmSearch({ search_algorithm_id: () => 'old-search' })).toBe(false);
    expect(runtime.hasExpectedWasmSearch({
      search_algorithm_id: () => { throw new Error('missing WASM export'); },
    })).toBe(false);
    expect(runtime.hasExpectedWasmSearch({
      search_algorithm_id: () => runtime.DELTREL_LOCAL_SEARCH_ALGORITHM_ID,
    })).toBe(true);
    for (const filename of ['deltrel_wasm.js', 'deltrel_wasm_bg.wasm']) {
      const url = new URL(runtime.versionedWasmUrl(`/models/deltrel/${filename}`), 'https://example.test');
      expect(url.pathname).toBe(`/models/deltrel/${filename}`);
      expect(url.searchParams.get('search')).toBe(runtime.DELTREL_LOCAL_SEARCH_ALGORITHM_ID);
      expect(url.searchParams.get('implementation')).toBe('search-session-v1');
    }
  });

  it('decodes bitboards while rejecting overlap and off-board bits', () => {
    const state = {
      zero_bits: () => new BigUint64Array([BigInt(1), ...Array(6).fill(BigInt(0))]),
      one_bits: () => new BigUint64Array([BigInt(2), ...Array(6).fill(BigInt(0))]),
    } as WasmState;
    expect(runtime.stonesFromWasm(state, 50).slice(0, 3)).toEqual([0, 1, -1]);

    const overlap = {
      ...state,
      one_bits: () => new BigUint64Array([BigInt(1), ...Array(6).fill(BigInt(0))]),
    } as WasmState;
    expect(() => runtime.stonesFromWasm(overlap, 50)).toThrow(/overlapping stones/i);

    const offBoard = {
      ...state,
      zero_bits: () => new BigUint64Array([BigInt(1) << BigInt(50), ...Array(6).fill(BigInt(0))]),
      one_bits: () => new BigUint64Array(7),
    } as WasmState;
    expect(() => runtime.stonesFromWasm(offBoard, 50)).toThrow(/off-board nodes/i);
    expect(
      runtime.nodesFromBits(
        new BigUint64Array([BigInt(5), BigInt(1), ...Array(5).fill(BigInt(0))]),
        275,
        'history',
      ),
    ).toEqual([0, 2, 64]);
  });

  it('validates ONNX names, tensor types, ranks, and fixed head dimensions', () => {
    const metadata = (
      type: Ort.Tensor.Type,
      shape: readonly number[],
    ): Ort.InferenceSession.ValueMetadata =>
      ({ isTensor: true, type, shape }) as Ort.InferenceSession.ValueMetadata;
    const session = {
      inputNames: [...DELTREL_MODEL_INPUT_NAMES],
      outputNames: [...DELTREL_MODEL_OUTPUT_NAMES],
      inputMetadata: [
        metadata('float16', [1, 50, DELTREL_NODE_FEATURE_DIM]),
        metadata('float16', [1, DELTREL_GLOBAL_FEATURE_DIM]),
        metadata('int64', [1, 50, 6]),
        metadata('bool', [1, 50, 6]),
        metadata('int64', [1, 50, 6]),
        metadata('bool', [1, 50]),
        metadata('bool', [1, 50]),
        metadata('int64', [1]),
      ],
      outputMetadata: [
        metadata('float16', [1, 50]),
        metadata('float16', [1, 2]),
        metadata('float16', [1, 303]),
        metadata('float16', [1, 50, 3]),
        metadata('float16', [1, 50]),
        metadata('float16', [1, 50]),
      ],
    } as unknown as Ort.InferenceSession;
    expect(runtime.hasExpectedOnnxSchema(session)).toBe(true);
    const fp32 = { ...session,
      inputMetadata: session.inputMetadata.map((entry) => entry.isTensor && entry.type === 'float16' ? { ...entry, type: 'float32' } : entry),
      outputMetadata: session.outputMetadata.map((entry) => ({ ...entry, type: 'float32' })),
    } as Ort.InferenceSession;
    expect(runtime.hasExpectedOnnxSchema(fp32, 'float32')).toBe(true);
    expect(runtime.hasExpectedOnnxSchema(session, 'float32')).toBe(false);
    expect(runtime.hasExpectedOnnxSchema(fp32, 'float16')).toBe(false);
    const invalid = {
      ...session,
      outputNames: ['wrong', ...session.outputNames.slice(1)],
    } as unknown as Ort.InferenceSession;
    expect(runtime.hasExpectedOnnxSchema(invalid)).toBe(false);
  });

  it('decodes finite FP16 outputs and normalizes binary outcome logits', () => {
    const encoded = float32ToFloat16Array(new Float32Array([-1, 0, 2]));
    const decoded = runtime.finiteFloatData(
      { data: encoded } as unknown as Ort.OnnxValue,
      'policy',
    );
    expect(Array.from(decoded)).toEqual([-1, 0, 2]);
    expect(runtime.outcomeValue(new Float32Array([0, 0]))).toBe(0);
    expect(runtime.outcomeValue(new Float32Array([-10, 10]))).toBeGreaterThan(0.99);
    const belief = runtime.outcomeBelief(new Float32Array([0, Math.log(3)]));
    expect(belief.loss).toBeCloseTo(0.25, 6);
    expect(belief.win).toBeCloseTo(0.75, 6);
    const scoreLogits = new Float32Array(303).fill(-100);
    scoreLogits[153] = 0;
    expect(runtime.expectedScoreMargin(scoreLogits)).toBeCloseTo(2, 8);
    expect(() => runtime.outcomeValue(new Float32Array([0, 0, 0]))).toThrow(
      /two logits/i,
    );
    expect(() => runtime.expectedScoreMargin(new Float32Array([0]))).toThrow(
      /303 logits/i,
    );

    const nonFinite = float32ToFloat16Array(new Float32Array([Number.NaN]));
    expect(() =>
      runtime.finiteFloatData(
        { data: nonFinite } as unknown as Ort.OnnxValue,
        'policy',
      ),
    ).toThrow(/non-finite/i);
  });

  it('preserves full precision outputs and rejects non-finite FP32 data', () => {
    const values = new Float32Array([0.123456789, -0.987654321]);
    expect(runtime.finiteFloatData({ data: values } as unknown as Ort.OnnxValue, 'policy')).toEqual(values);
    expect(() => runtime.finiteFloatData({ data: new Float32Array([Infinity]) } as unknown as Ort.OnnxValue, 'policy')).toThrow(/non-finite/);
    expect(runtime.scoreUtility(0.5, 151, 0.05)).toBeCloseTo(0.55);
    expect(runtime.scoreUtility(0.5, -151, 0.05)).toBeCloseTo(0.45);
    expect(runtime.scoreUtility(1, 151, 0.05)).toBe(1);
    expect(runtime.scoreUtility(-1, -151, 0.05)).toBe(-1);
  });

  it('derives the browser seed from the exact native request contract', () => {
    const hash = BigInt('0xfedcba9876543210');
    const derive = vi.fn(() => BigInt(123));
    const root = { hash64: () => hash } as WasmState;
    const current = { manifest: { search: { seedContract: 'native-search-batch-v1' } }, wasm: { derive_root_seed: derive } };
    expect(runtime.rootSearchSeed(current as never, root)).toBe(BigInt(123));
    expect(derive).toHaveBeenCalledWith(hash & BigInt(Number.MAX_SAFE_INTEGER), hash, 0);
    expect(runtime.rootSearchSeed({ manifest: { search: {} } } as never, root)).toBe(hash);
    expect(() => runtime.rootSearchSeed({ ...current, wasm: {} } as never, root)).toThrow(/seed contract/);
  });

  it('replays and verifies semantic identity before local search', () => {
    const request = buildAiRequest(
      {
        rings: 4,
        mode: 'double',
        pieRule: true,
        playerNames: ['A', 'B'],
      },
      [{ type: 'place', node: 7 }, { type: 'swap' }],
    );
    const hash = BigInt(`0x${request.stateHash.slice('zobrist64:'.length)}`);
    const applied: Array<number | 'swap'> = [];
    const stoneWords = new BigUint64Array(7);
    stoneWords[0] = BigInt(1) << BigInt(7);
    class FakeState {
      readonly to_move = request.state.toMove;
      readonly moves_left = request.state.movesLeft;
      readonly opening = request.state.opening;
      readonly terminal = request.state.terminal;
      readonly mode = request.state.mode;
      readonly handicap = request.state.handicap;
      readonly pie = request.state.pie;
      readonly swap_available = request.state.swapAvailable;
      readonly swapped = request.state.swapped;
      constructor(rings: number, mode: string, handicap: number, pie: boolean) {
        expect([rings, mode, handicap, pie]).toEqual([4, 'double', 1, true]);
      }
      apply = (node: number) => {
        applied.push(node);
      };
      swap = () => {
        applied.push('swap');
      };
      zero_bits = () => new BigUint64Array(7);
      one_bits = () => stoneWords;
      current_turn_bits = () => new BigUint64Array(7);
      previous_turn_bits = () => stoneWords;
      own_previous_turn_bits = () => new BigUint64Array(7);
      handicap_bits = () => stoneWords;
      legal_actions = () => Int32Array.from(request.legalActions);
      hash64 = () => hash;
    }
    const wasm = { WasmState: FakeState };
    expect(runtime.replayAndVerify(request, wasm as never)).toBeInstanceOf(FakeState);
    expect(applied).toEqual([7, 'swap']);
    expect(request.state.stones[7]).toBe(1);

    class BadState extends FakeState {
      legal_actions = () => Int32Array.from([-1]);
    }
    expect(() =>
      runtime.replayAndVerify(request, { WasmState: BadState } as never),
    ).toThrow(/disagrees with the AI request/i);
    class DriftedState extends FakeState {
      readonly swapped = false;
    }
    expect(() =>
      runtime.replayAndVerify(request, { WasmState: DriftedState } as never),
    ).toThrow(/disagrees with the AI request/i);
  });
});

describe('local prediction reuse', () => {
  function makeRuntime(capacity = 1_024) {
    const feedsDisposed = vi.fn();
    const outputsDisposed = vi.fn();
    class Tensor {
      dispose = feedsDisposed;
    }
    const output = (values: Float32Array) => {
      const data = float32ToFloat16Array(values);
      return {
        data,
        dispose: () => {
          data.fill(0);
          outputsDisposed();
        },
      };
    };
    const run = vi.fn(async () => ({
      policy_logits: output(Float32Array.from({ length: 50 }, (_, index) => index)),
      outcome_logits: output(new Float32Array([0, 2])),
      score_margin_logits: output(new Float32Array(303)),
    }));
    const localRuntime = {
      manifest: { model: { sha256: 'model-one', precision: 'float16' }, featureSchemaHash: 'features-one', search: {} },
      ort: { Tensor },
      session: { run },
      predictions: new runtime.PredictionCache(capacity),
    };
    const semantic = buildAiRequest(
      { rings: 4, mode: 'double', pieRule: false, playerNames: ['A', 'B'] },
      [],
    ).state;
    return { localRuntime, semantic, run, feedsDisposed, outputsDisposed };
  }

  it('reuses identical predictions after releasing tensors and isolates mutable results', async () => {
    const { localRuntime, semantic, run, feedsDisposed, outputsDisposed } = makeRuntime();
    const legal = Int32Array.from([1, 2]);
    const first = await runtime.evaluate(localRuntime as never, semantic, legal);
    const expectedWin = first.outcome.win;
    first.logits.fill(100);
    first.outcome.win = 0;
    const second = await runtime.evaluate(
      localRuntime as never,
      JSON.parse(JSON.stringify(semantic)),
      legal.slice(),
    );
    expect(run).toHaveBeenCalledTimes(1);
    expect(Array.from(second.logits)).toEqual([1, 2]);
    expect(second.outcome.win).toBe(expectedWin);
    second.logits[0] = -100;
    const third = await runtime.evaluate(localRuntime as never, semantic, legal);
    expect(Array.from(third.logits)).toEqual([1, 2]);
    expect(feedsDisposed).toHaveBeenCalledTimes(8);
    expect(outputsDisposed).toHaveBeenCalledTimes(3);
  });

  it('distinguishes every feature-history plane and ordered legal actions', async () => {
    const { localRuntime, semantic, run } = makeRuntime();
    const legal = Int32Array.from([1, 2]);
    await runtime.evaluate(localRuntime as never, semantic, legal);
    for (const plane of ['currentTurn', 'previousTurn', 'ownPreviousTurn', 'handicapStones']) {
      await runtime.evaluate(localRuntime as never, {
        ...semantic,
        history: { ...semantic.history, [plane]: [1] },
      }, legal);
    }
    const reversed = await runtime.evaluate(localRuntime as never, semantic, Int32Array.from([2, 1]));
    expect(Array.from(reversed.logits)).toEqual([2, 1]);
    expect(run).toHaveBeenCalledTimes(6);
  });

  it('starts a fresh cache on runtime reload and distinguishes model and feature identities', async () => {
    const first = makeRuntime();
    const second = makeRuntime();
    const legal = Int32Array.from([1, 2]);
    await runtime.evaluate(first.localRuntime as never, first.semantic, legal);
    await runtime.evaluate(second.localRuntime as never, second.semantic, legal);
    first.localRuntime.manifest.model.sha256 = 'model-two';
    await runtime.evaluate(first.localRuntime as never, first.semantic, legal);
    first.localRuntime.manifest.featureSchemaHash = 'features-two';
    await runtime.evaluate(first.localRuntime as never, first.semantic, legal);
    expect(first.run).toHaveBeenCalledTimes(3);
    expect(second.run).toHaveBeenCalledTimes(1);
  });

  it('evicts the least recently used prediction at its capacity', async () => {
    const { localRuntime, semantic, run } = makeRuntime(2);
    const evaluate = (actions: number[]) =>
      runtime.evaluate(localRuntime as never, semantic, Int32Array.from(actions));
    await evaluate([1]);
    await evaluate([2]);
    await evaluate([1]);
    await evaluate([3]);
    await evaluate([1]);
    expect(run).toHaveBeenCalledTimes(3);
    await evaluate([2]);
    expect(run).toHaveBeenCalledTimes(4);
  });

  it('releases failed inference feeds and retries without caching the failure', async () => {
    const { localRuntime, semantic, run, feedsDisposed } = makeRuntime();
    run.mockRejectedValueOnce(new Error('temporary inference failure'));
    const legal = Int32Array.from([1, 2]);
    await expect(runtime.evaluate(localRuntime as never, semantic, legal)).rejects.toThrow(
      'temporary inference failure',
    );
    await runtime.evaluate(localRuntime as never, semantic, legal);
    expect(run).toHaveBeenCalledTimes(2);
    expect(feedsDisposed).toHaveBeenCalledTimes(16);
  });

  it('releases partially allocated leaf inputs when a tensor cannot be created', async () => {
    const { localRuntime, semantic, run } = makeRuntime();
    const dispose = vi.fn();
    let allocations = 0;
    class FailingTensor {
      dispose = dispose;
      constructor() { if (++allocations === 3) throw new Error('allocation failed'); }
    }
    await expect(runtime.evaluate({ ...localRuntime, ort: { Tensor: FailingTensor } } as never,
      semantic, Int32Array.from([1, 2]))).rejects.toThrow('allocation failed');
    expect(dispose).toHaveBeenCalledTimes(2);
    expect(run).not.toHaveBeenCalled();
  });

  it('releases rejected model outputs without retaining an invalid prediction', async () => {
    const { localRuntime, semantic, run } = makeRuntime();
    const dispose = vi.fn();
    run.mockResolvedValueOnce({
      policy_logits: { data: new Uint16Array([0x7e00]), dispose },
      outcome_logits: { data: new Uint16Array(2), dispose },
      score_margin_logits: { data: new Uint16Array(303), dispose },
    });
    const legal = Int32Array.from([1, 2]);
    await expect(runtime.evaluate(localRuntime as never, semantic, legal)).rejects.toThrow(
      /non-finite/i,
    );
    expect(dispose).toHaveBeenCalledTimes(3);
    expect(Array.from((await runtime.evaluate(localRuntime as never, semantic, legal)).logits))
      .toEqual([1, 2]);
    expect(run).toHaveBeenCalledTimes(2);
  });
});

describe('local pie search decisions', () => {
  it('keeps a winning selected continuation despite a losing exploration average', () => {
    const tree = {
      completed_q: () => Float32Array.from([-0.8, 0.6]),
      visits: () => Uint32Array.from([8, 2]),
      actions: () => Int32Array.from([3, 7]),
      policy_target: () => Float32Array.from([0.01, 0.99]),
      root_value: () => -0.52,
    };
    const scheduler = { selected: () => 1 };
    const result = runtime.summarizeSearch(tree as never, scheduler as never, -0.9, true, 0.02);
    expect(result.actionCode).toBe(7);
    expect(result.swapRecommended).toBe(false);
    expect(result.rootValue).toBe(-0.52);
  });

  it('swaps for a losing selected continuation while respecting the dead zone', () => {
    let selectedValue = -0.6;
    const tree = {
      completed_q: () => Float32Array.from([0.8, selectedValue]),
      visits: () => Uint32Array.from([8, 2]),
      actions: () => Int32Array.from([3, 7]),
      policy_target: () => Float32Array.from([0.01, 0.99]),
      root_value: () => 0.52,
    };
    const scheduler = { selected: () => 1 };
    expect(runtime.summarizeSearch(tree as never, scheduler as never, 0, true, 0.02)
      .swapRecommended).toBe(true);
    expect(runtime.summarizeSearch(tree as never, scheduler as never, 0, false, 0.02)
      .swapRecommended).toBe(false);
    selectedValue = -0.01;
    expect(runtime.summarizeSearch(tree as never, scheduler as never, 0, true, 0.02)
      .swapRecommended).toBe(false);
  });
});
