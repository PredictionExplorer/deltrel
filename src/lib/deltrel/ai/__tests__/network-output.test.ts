import { describe, expect, it } from 'vitest';
import { buildAiRequest, type DeltrelAiRequest } from '../protocol';
import { DELTREL_NETWORK_HEAD_NAMES, parseNetworkOutput, parseServerNetworkOutput, validateNetworkOutputState,
  type DeltrelNetworkOutput, type DeltrelNetworkHeadName } from '../network-output';

function sample(request: DeltrelAiRequest, auxiliary = true): DeltrelNetworkOutput {
  const n = request.state.stones.length;
  const shapes: Record<DeltrelNetworkHeadName, number[]> = {
    policy: [n], outcome: [2], scoreMargin: [303], ownership: [n, 3], alive: [n], softPolicy: [n],
    opponentReply: [n + 1], secondStone: [n], finalShores: [2, 51], finalNetworks: [2, 26], finalCapes: [2, 6],
  };
  const heads = {} as DeltrelNetworkOutput['heads'];
  for (const [index, name] of DELTREL_NETWORK_HEAD_NAMES.entries()) {
    if (!auxiliary && index >= 6) { heads[name] = null; continue; }
    const shape = shapes[name], width = shape.at(-1)!;
    const size = shape.reduce((a, b) => a * b, 1);
    const mask = Array.from({length: size}, (_, i) => name === 'finalShores' ? i % width <= request.state.rings * 5
      : name === 'finalNetworks' ? i % width <= Math.floor(request.state.rings * 5 / 2) : true);
    const probabilities = mask.map((active, i) => !active ? 0 : name === 'alive' ? 0.5
      : 1 / mask.slice(Math.floor(i / width) * width, (Math.floor(i / width) + 1) * width).filter(Boolean).length);
    heads[name] = {shape, mask, activation: name === 'alive' ? 'sigmoid' : 'softmax',
      logits: mask.map(active => active ? 0 : null), probabilities, applicable: name !== 'secondStone'};
  }
  return {schemaVersion: 1, perspective: request.state.toMove, nodeCount: n,
    auxiliaryStatus: auxiliary ? 'ready' : 'absent', heads};
}
const request = buildAiRequest({rings: 4, mode: 'double', pieRule: false, handicap: 1, playerNames: ['Sea', 'Clay']}, [], 'heads');

describe('complete root network output contract', () => {
  it.each([4, 6, 8, 10] as const)('validates every head on %i rings with bounded size', rings => {
    const req = buildAiRequest({rings, mode: 'double', pieRule: false, handicap: 1, playerNames: ['Sea', 'Clay']}, [], `heads-${rings}`);
    const output = sample(req);
    expect(parseNetworkOutput(output)).toEqual(output);
    expect(() => validateNetworkOutputState(output, req, false)).not.toThrow();
    expect(JSON.stringify(output).length).toBeLessThan(256 * 1024);
  });
  it('preserves absent and untrained heads explicitly', () => {
    expect(parseNetworkOutput(null)).toBeNull();
    const absent = sample(request, false);
    expect(parseNetworkOutput(absent)?.heads.finalCapes).toBeNull();
    const untrained = {...sample(request), auxiliaryStatus: 'untrained'};
    expect(parseNetworkOutput(untrained)?.auxiliaryStatus).toBe('untrained');
  });
  it('rejects malformed dimensions, values, activation, masks and unknown heads', () => {
    for (const mutate of [
      (o: DeltrelNetworkOutput) => { o.heads.policy!.shape = [49]; },
      (o: DeltrelNetworkOutput) => { o.heads.alive!.probabilities[0] = NaN; },
      (o: DeltrelNetworkOutput) => { o.heads.outcome!.probabilities = [0.7, 0.3]; },
      (o: DeltrelNetworkOutput) => { o.heads.policy!.logits[0] = Infinity; },
      (o: DeltrelNetworkOutput) => { o.heads.ownership!.mask[0] = false; },
      (o: DeltrelNetworkOutput) => { o.heads.outcome!.logits[0] = 2; },
      (o: DeltrelNetworkOutput) => { o.heads.finalCapes = null; },
    ]) {
      const output = sample(request); mutate(output);
      expect(() => parseNetworkOutput(output)).toThrow(/network outputs/i);
    }
  });
  it('binds root perspective, legal nodes and forecast applicability', () => {
    const output = sample(request);
    expect(() => validateNetworkOutputState({...output, perspective: 1}, request, false)).toThrow();
    output.heads.secondStone!.applicable = true;
    expect(() => validateNetworkOutputState(output, request, false)).toThrow();
    const wrongMask = sample(request); wrongMask.heads.policy!.mask[0] = false;
    expect(() => validateNetworkOutputState(wrongMask, request, false)).toThrow();
  });
  it('normalizes the optional snake-case wire field without discarding any head', () => {
    const output = sample(request);
    const names = ['policy', 'outcome', 'score_margin', 'ownership', 'alive', 'soft_policy', 'opponent_reply',
      'second_stone', 'final_shores', 'final_networks', 'final_capes'];
    const wire = {schema_version: 1, perspective: 0, node_count: 50, auxiliary_status: 'ready',
      heads: Object.fromEntries(names.map((name, index) => [name, output.heads[DELTREL_NETWORK_HEAD_NAMES[index]]]))};
    expect(parseServerNetworkOutput(wire)).toEqual(output);
  });
});
