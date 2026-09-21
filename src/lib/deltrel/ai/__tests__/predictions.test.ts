import { describe, expect, it } from 'vitest';
import { buildAiRequest, type DeltrelAiRequest } from '../protocol';
import type { DeltrelNetworkHead, DeltrelNetworkOutput } from '../network-output';
import { predictionsFromNetworkOutput, validatePredictionState } from '../predictions';

const config = { rings: 4, mode: 'double' as const, pieRule: false, playerNames: ['Clay', 'Sea'] as [string, string] };
const request = (placements: number[] = [], changes = {}) => buildAiRequest(
  { ...config, ...changes }, placements.map(node => ({ type: 'place', node })),
);

function distribution(shape: number[], rows: Record<number, number>[], applicable = true): DeltrelNetworkHead {
  const width = shape.at(-1)!;
  const probabilities = rows.flatMap(row => Array.from({ length: width }, (_, index) => row[index] ?? 0));
  return { shape, probabilities, logits: probabilities.map(p => p > 0 ? Math.log(p) : -1000),
    mask: probabilities.map(() => true), applicable, activation: 'softmax' };
}

function outputFor(req: DeltrelAiRequest): DeltrelNetworkOutput {
  const nodes = req.state.stones.length;
  return {
    schemaVersion: 1, perspective: req.state.toMove, nodeCount: nodes, auxiliaryStatus: 'ready',
    heads: {
      policy: null, outcome: null, scoreMargin: null, ownership: null, alive: null, softPolicy: null,
      finalShores: distribution([2, 51], [{ 10: 0.25, 12: 0.75 }, { 2: 0.5, 4: 0.5 }]),
      finalNetworks: distribution([2, 26], [{ 2: 0.5, 4: 0.5 }, { 1: 1 }]),
      finalCapes: distribution([2, 6], [{ 2: 0.25, 3: 0.5, 5: 0.25 }, { 0: 0.2, 1: 0.3, 4: 0.5 }]),
      opponentReply: distribution([nodes + 1], [{ 4: 0.6, 5: 0.3, [nodes]: 0.1 }]),
      secondStone: distribution([nodes], [{ 6: 0.9, 7: 0.1 }]),
    },
  };
}

describe('compact forecasts from browser network outputs', () => {
  it.each([0, 1] as const)('maps current/opponent count distributions into stable players for perspective %i', perspective => {
    const req = perspective === 0 ? request() : request([0]);
    const output = outputFor(req);
    const result = predictionsFromNetworkOutput(output, req.state, false)!;
    expect(result.perspective).toBe(perspective);
    expect(result.finalBasis).toBe('official_end');
    expect(result.finalCounts.map(p => p.player)).toEqual([0, 1]);
    expect(result.finalCounts[perspective]).toMatchObject({ shores: 11.5, networks: 3, corners: 3.25, cornerBonusProbability: 0.75 });
    expect(result.finalCounts[1 - perspective]).toMatchObject({ shores: 3, networks: 1, corners: 2.3, cornerBonusProbability: 0.5 });
    expect(result.opponentReply).toEqual({ player: 1 - perspective, kind: 'place', node: 4, probability: 0.6 });
    expect(() => validatePredictionState(result, req)).not.toThrow();
  });

  it('uses the highest legal reply probability without renormalizing an ineligible swap', () => {
    const req = request([0]);
    const output = outputFor(req);
    output.heads.opponentReply = distribution([51], [{ 3: 0.15, 4: 0.05, 50: 0.8 }]);
    const before = structuredClone(output);
    expect(predictionsFromNetworkOutput(output, req.state, false)?.opponentReply).toEqual({
      player: 0, kind: 'place', node: 3, probability: 0.15,
    });
    expect(output).toEqual(before);
  });

  it('allows a future pie swap only while the opening turn is still in progress', () => {
    const opening = request([], { pieRule: true });
    const output = outputFor(opening);
    output.heads.opponentReply = distribution([51], [{ 3: 0.2, 50: 0.8 }]);
    expect(predictionsFromNetworkOutput(output, opening.state, false)?.opponentReply).toEqual({
      player: 1, kind: 'swap', node: null, probability: 0.8,
    });
    const responder = request([0], { pieRule: true });
    output.perspective = responder.state.toMove;
    expect(predictionsFromNetworkOutput(output, responder.state, false)?.opponentReply).toEqual({
      player: 0, kind: 'place', node: 3, probability: 0.2,
    });
  });

  it.each([
    ['opening', [], {}, false, false],
    ['two stones left', [0], {}, false, true],
    ['one stone left', [0, 1], {}, false, false],
    ['classic', [0], { mode: 'classic' }, false, false],
    ['taking pie swap', [0], { pieRule: true }, true, false],
  ] as const)('only reports an applicable second stone: %s', (_, placements, changes, swap, expected) => {
    const req = request([...placements], changes);
    const output = outputFor(req);
    const result = predictionsFromNetworkOutput(output, req.state, swap)!;
    expect(result.secondStone).toEqual(expected
      ? { player: req.state.toMove, kind: 'place', node: 6, probability: 0.9 } : null);
    expect(() => validatePredictionState(result, req)).not.toThrow();
  });

  it('suppresses inapplicable future heads while preserving final-count forecasts', () => {
    const req = request([0]);
    const output = outputFor(req);
    output.heads.opponentReply!.applicable = false;
    output.heads.secondStone!.applicable = false;
    const result = predictionsFromNetworkOutput(output, req.state, false)!;
    expect(result.opponentReply).toBeNull();
    expect(result.secondStone).toBeNull();
    expect(result.finalCounts).toHaveLength(2);
  });

  it('breaks equal probabilities by the lowest available node before the swap slot', () => {
    const req = request([], { pieRule: true });
    const output = outputFor(req);
    output.heads.opponentReply = distribution([51], [{ 1: 0.5, 50: 0.5 }]);
    expect(predictionsFromNetworkOutput(output, req.state, false)?.opponentReply?.node).toBe(1);
  });

  it.each(['absent', 'untrained'] as const)('does not expose %s auxiliary outputs as learned forecasts', auxiliaryStatus => {
    const req = request();
    expect(predictionsFromNetworkOutput({ ...outputFor(req), auxiliaryStatus }, req.state, false)).toBeNull();
  });

  it('rejects mismatched root identities and incomplete trained reports', () => {
    const req = request();
    const output = outputFor(req);
    expect(() => predictionsFromNetworkOutput({ ...output, perspective: 1 }, req.state, false)).toThrow(/predictions/);
    expect(() => predictionsFromNetworkOutput({ ...output, nodeCount: 105 }, req.state, false)).toThrow(/predictions/);
    output.heads.finalCapes = null;
    expect(() => predictionsFromNetworkOutput(output, req.state, false)).toThrow(/predictions/);
  });

  it('rejects impossible count expectations and handles only tiny floating-point overshoot', () => {
    const req = request();
    const output = outputFor(req);
    output.heads.finalShores = distribution([2, 51], [{ 50: 1 }, { 2: 1 }]);
    expect(() => predictionsFromNetworkOutput(output, req.state, false)).toThrow(/predictions/);
    output.heads.finalShores = distribution([2, 51], [{ 19: 1e-8, 20: 1 }, { 2: 1 }]);
    expect(predictionsFromNetworkOutput(output, req.state, false)?.finalCounts[0].shores).toBe(20);
  });
});
