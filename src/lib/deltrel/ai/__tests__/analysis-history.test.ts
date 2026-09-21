import { describe, expect, it, vi } from 'vitest';
import type { GameAction, GameConfig } from '../../game';
import type { DeltrelAiAnalysis } from '../decision';
import {
  analysisConfigKey,
  MAX_ENGINE_ANALYSES,
  recordEngineAnalysis,
  selectEngineAnalysis,
  type StoredEngineAnalysis,
} from '../analysis-history';

const config: GameConfig = {
  rings: 4,
  mode: 'double',
  pieRule: false,
  playerNames: ['Clay', 'Seafoam'],
};
const place = (node: number): GameAction => ({ type: 'place', node });
const hash = (node: number) => `zobrist64:${node.toString(16).padStart(16, '0')}`;

function analysis(position: number, perspective: 0 | 1 = 0): DeltrelAiAnalysis {
  return {
    perspective,
    stateHash: hash(position),
    outcome: { loss: 0.2, win: 0.8 },
    modelValue: 0.6,
    searchValue: 0.3,
    rootValue: 0.4,
    swapRecommended: false,
    expectedMargin: 3,
    rootActions: [{ type: 'place', node: position }],
    rootPolicy: [1],
    rootQ: [0.4],
    rootVisits: [1],
    modelVersion: 'test-champion',
    modelStep: 123,
    modelIdentity: null,
    simulations: 1,
    maxConsidered: 1,
    timingMs: { queue: 0, modelLoad: 0, inferenceSearch: 1, total: 1 },
  };
}

function stored(
  prefix: GameAction[] = [],
  overrides: Partial<StoredEngineAnalysis> = {},
): StoredEngineAnalysis {
  return {
    analysis: analysis(prefix.length),
    source: 'server',
    ply: prefix.length,
    configKey: analysisConfigKey(config),
    prefix,
    action: place(prefix.length),
    applied: true,
    ...overrides,
  };
}

function select(
  history: readonly StoredEngineAnalysis[],
  log: GameAction[],
  overrides: Partial<Parameters<typeof selectEngineAnalysis>[1]> = {},
) {
  return selectEngineAnalysis(history, {
    config, log, ply: log.length, positionHash: hash(log.length), allowPrevious: true,
    ...overrides,
  });
}

describe('engine analysis history', () => {
  it('retains completed AI estimates across subsequent turns without calling an engine', () => {
    const first = stored();
    let history = recordEngineAnalysis([], first);
    expect(select(history, [place(0)])).toEqual({ entry: first, isExact: false });

    const second = stored([place(0)], { analysis: analysis(1, 1) });
    history = recordEngineAnalysis(history, second);
    expect(select(history, [place(0), place(1)])).toEqual({ entry: second, isExact: false });
    // The same retained result remains available while the next request fails or retries.
    expect(select(history, [place(0), place(1)])?.entry.analysis).toBe(second.analysis);
    expect(history).toHaveLength(2);
  });

  it('prefers an exact reviewed position over a newer completed prior estimate', () => {
    const exact = stored([place(0)]);
    const newer = stored();
    const history = recordEngineAnalysis(recordEngineAnalysis([], exact), newer);
    expect(select(history, [place(0)], { allowPrevious: false })).toEqual({
      entry: exact, isExact: true,
    });
    expect(select(history, [place(0), place(1)], { allowPrevious: false })).toBeNull();
  });

  it('keeps the nearest live estimate after an opening reanalysis finishes later', () => {
    const log = [place(0), place(1), place(2)];
    const nearer = stored(log.slice(0, 2));
    const openingReanalysis = stored([], { applied: false });
    const history = recordEngineAnalysis(
      recordEngineAnalysis([], nearer), openingReanalysis,
    );
    expect(select(history, log)).toEqual({ entry: nearer, isExact: false });
    expect(select(history, [], { allowPrevious: false })).toEqual({
      entry: openingReanalysis, isExact: true,
    });
    const samePlyLater = { ...nearer, analysis: analysis(42), source: 'local' as const };
    expect(select(recordEngineAnalysis(history, samePlyLater), log)).toEqual({
      entry: samePlyLater, isExact: false,
    });
  });

  it('restores exact estimates through undo and redo without showing future positions', () => {
    const first = stored();
    const second = stored([place(0)]);
    const third = stored([place(0), place(1)]);
    const history = [first, second, third].reduce(recordEngineAnalysis, [] as StoredEngineAnalysis[]);
    expect(select(history, [], { allowPrevious: false })).toEqual({ entry: first, isExact: true });
    expect(select(history, [place(0)], { allowPrevious: false })).toEqual({ entry: second, isExact: true });
    expect(select(history, [place(0), place(1)], { allowPrevious: false })).toEqual({ entry: third, isExact: true });
    expect(select([third], [place(0)])).toBeNull();
  });

  it('rejects discarded branches even when their position hashes match', () => {
    const discarded = stored([place(8)]);
    expect(select([discarded], [place(0)])).toBeNull();
    const transposed = stored([place(1), place(0)]);
    expect(select([transposed], [place(0), place(1)])).toBeNull();
  });

  it('distinguishes swaps from placements in the branch prefix', () => {
    const swapped = stored([place(0), { type: 'swap' }]);
    expect(select([swapped], [place(0), place(1)])).toBeNull();
    expect(select([swapped], [place(0), { type: 'swap' }])).toEqual({
      entry: swapped, isExact: true,
    });
    const placed = stored([place(0), place(1)]);
    expect(select([placed], [place(0), { type: 'swap' }])).toBeNull();
  });

  it('separates rule configurations while allowing name changes and implicit handicap one', () => {
    const entry = stored();
    expect(analysisConfigKey(config)).toBe(analysisConfigKey({ ...config, handicap: 1 }));
    expect(select([entry], [], { config: { ...config, playerNames: ['Ada', 'Grace'] } })?.isExact).toBe(true);
    for (const change of [{ rings: 6 }, { mode: 'classic' as const }, { handicap: 2 }, { pieRule: true }]) {
      const other = { ...config, ...change };
      expect(analysisConfigKey(other)).not.toBe(entry.configKey);
      expect(select([entry], [], { config: other })).toBeNull();
    }
  });

  it('preserves the original player perspective and swap decision', () => {
    const entry = stored([place(0)], {
      analysis: { ...analysis(1, 1), swapRecommended: true },
      action: { type: 'swap' },
    });
    const history = recordEngineAnalysis([], entry);
    const selected = select(history, [place(0), { type: 'swap' }]);
    expect(selected?.entry.analysis.perspective).toBe(1);
    expect(selected?.entry.analysis.outcome).toEqual({ loss: 0.2, win: 0.8 });
    expect(selected?.entry.analysis.expectedMargin).toBe(3);
    expect(selected?.entry.action).toEqual({ type: 'swap' });
    expect(selected?.isExact).toBe(false);
  });

  it('replaces the same position with its latest result and source', () => {
    const original = stored();
    const replacement = stored([], { source: 'local', applied: false });
    const otherConfig = stored([], { configKey: analysisConfigKey({ ...config, rings: 6 }) });
    const history = recordEngineAnalysis([original, otherConfig], replacement);
    expect(history).toEqual([otherConfig, replacement]);
    expect(select(history, [])?.entry.source).toBe('local');
    expect(select(history, [])?.entry.applied).toBe(false);
  });

  it('caps retained results at 64, evicting the oldest completed result', () => {
    let history: StoredEngineAnalysis[] = [];
    for (let index = 0; index < MAX_ENGINE_ANALYSES + 3; index++) {
      history = recordEngineAnalysis(history, stored([], { analysis: analysis(index) }));
    }
    expect(MAX_ENGINE_ANALYSES).toBe(64);
    expect(history).toHaveLength(64);
    expect(history[0].analysis.stateHash).toBe(hash(3));
    expect(history.at(-1)?.analysis.stateHash).toBe(hash(66));
  });

  it('copies recorded action prefixes and the associated action', () => {
    const prefix = [place(0), { type: 'swap' } as GameAction];
    const entry = stored(prefix);
    const history = recordEngineAnalysis([], entry);
    expect(history[0]).not.toBe(entry);
    expect(history[0].prefix).not.toBe(prefix);
    expect(history[0].prefix[0]).not.toBe(prefix[0]);
    prefix[0] = place(7);
    prefix.push(place(8));
    if (entry.action.type === 'place') entry.action.node = 99;
    expect(history[0].prefix).toEqual([place(0), { type: 'swap' }]);
    expect(history[0].action).toEqual(place(2));
  });

  it('ignores invalid source and target plies without publishing a misleading estimate', () => {
    const good = stored();
    const mismatched = stored([], { ply: 1 });
    expect(recordEngineAnalysis([good], mismatched)).toEqual([good]);
    expect(recordEngineAnalysis([good], stored([], { ply: -1 }))).toEqual([good]);
    expect(select([mismatched], [place(0)])).toBeNull();
    for (const ply of [-1, 0.5, 1, Number.NaN]) {
      expect(select([good], [], { ply })).toBeNull();
    }
  });

  it('handles unavailable position hashes without relying on BigInt', () => {
    vi.stubGlobal('BigInt', undefined);
    const history = recordEngineAnalysis([], stored());
    expect(select(history, [place(0)], { positionHash: null })?.isExact).toBe(false);
    expect(select(history, [], { positionHash: null, allowPrevious: false })).toBeNull();
    expect(select([], [], { positionHash: null })).toBeNull();
  });
});
