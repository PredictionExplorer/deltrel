import { DeltrelAiError } from './errors';
import type { DeltrelAiRequest } from './protocol';

export const DELTREL_NETWORK_HEAD_NAMES = [
  'policy', 'outcome', 'scoreMargin', 'ownership', 'alive', 'softPolicy',
  'opponentReply', 'secondStone', 'finalShores', 'finalNetworks', 'finalCapes',
] as const;
export type DeltrelNetworkHeadName = (typeof DELTREL_NETWORK_HEAD_NAMES)[number];
export interface DeltrelNetworkHead {
  /** Row-major dimensions; softmax is normalized along the final dimension. */
  shape: number[];
  activation: 'softmax' | 'sigmoid';
  /** Null denotes a structurally masked output, never a nonfinite JSON number. */
  logits: (number | null)[];
  probabilities: number[];
  mask: boolean[];
  /** Whether this prediction applies to the evaluated position/turn. */
  applicable: boolean;
}
export interface DeltrelNetworkOutput {
  schemaVersion: 1;
  /** Player to move in the evaluated root, not necessarily the current UI turn. */
  perspective: 0 | 1;
  nodeCount: number;
  auxiliaryStatus: 'ready' | 'untrained' | 'absent';
  /** Ownership classes: current, opponent, unclaimed. Count rows: current, opponent. */
  heads: Record<DeltrelNetworkHeadName, DeltrelNetworkHead | null>;
}
const WIRE_NAMES: Record<DeltrelNetworkHeadName, string> = {
  policy: 'policy', outcome: 'outcome', scoreMargin: 'score_margin', ownership: 'ownership',
  alive: 'alive', softPolicy: 'soft_policy', opponentReply: 'opponent_reply', secondStone: 'second_stone',
  finalShores: 'final_shores', finalNetworks: 'final_networks', finalCapes: 'final_capes',
};
function record(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}
function exact(v: Record<string, unknown>, keys: readonly string[]): boolean {
  return Object.keys(v).length === keys.length && keys.every(k => Object.hasOwn(v, k));
}
function invalid(): never { throw new DeltrelAiError('protocol', 'AI network outputs are invalid.'); }
function expectedShape(name: DeltrelNetworkHeadName, nodes: number): number[] {
  switch (name) {
    case 'outcome': return [2];
    case 'scoreMargin': return [303];
    case 'ownership': return [nodes, 3];
    case 'opponentReply': return [nodes + 1];
    case 'finalShores': return [2, 51];
    case 'finalNetworks': return [2, 26];
    case 'finalCapes': return [2, 6];
    default: return [nodes];
  }
}
function parseHead(value: unknown, name: DeltrelNetworkHeadName, nodes: number): DeltrelNetworkHead {
  const shape = expectedShape(name, nodes);
  const size = shape.reduce((a, b) => a * b, 1);
  if (!record(value) || !exact(value, ['shape', 'activation', 'logits', 'probabilities', 'mask', 'applicable']) ||
      !Array.isArray(value.shape) || value.shape.length !== shape.length || value.shape.some((v, i) => v !== shape[i]) ||
      value.activation !== (name === 'alive' ? 'sigmoid' : 'softmax') || typeof value.applicable !== 'boolean' ||
      !Array.isArray(value.logits) || value.logits.length !== size ||
      !Array.isArray(value.probabilities) || value.probabilities.length !== size ||
      !Array.isArray(value.mask) || value.mask.length !== size) return invalid();
  const logits = value.logits as unknown[];
  const probabilities = value.probabilities as unknown[];
  const mask = value.mask as unknown[];
  for (let i = 0; i < size; i++) {
    const p = probabilities[i], logit = logits[i];
    if (typeof mask[i] !== 'boolean' || typeof p !== 'number' || !Number.isFinite(p) || p < 0 || p > 1 ||
        (mask[i] ? (typeof logit !== 'number' || !Number.isFinite(logit)) : (logit !== null || p !== 0))) return invalid();
  }
  const width = shape.at(-1)!;
  if (value.activation === 'softmax') {
    for (let start = 0; start < size; start += width) {
      const row = probabilities.slice(start, start + width) as number[];
      if (Math.abs(row.reduce((a, b) => a + b, 0) - 1) > 1e-5) return invalid();
      const active = (logits.slice(start, start + width) as (number | null)[]).filter((x): x is number => x !== null);
      const maximum = Math.max(...active);
      const denominator = active.reduce((sum, x) => sum + Math.exp(x - maximum), 0);
      for (let j = 0; j < width; j++) {
        const logit = logits[start + j];
        const expected = logit === null ? 0 : Math.exp((logit as number) - maximum) / denominator;
        if (Math.abs((probabilities[start + j] as number) - expected) > 1e-5) return invalid();
      }
    }
  } else {
    for (let i = 0; i < size; i++) {
      if (Math.abs((probabilities[i] as number) - 1 / (1 + Math.exp(-(logits[i] as number)))) > 1e-5) return invalid();
    }
  }
  return {shape, activation: name === 'alive' ? 'sigmoid' : 'softmax', logits: [...logits] as (number | null)[],
    probabilities: [...probabilities] as number[], mask: [...mask] as boolean[], applicable: value.applicable};
}
export function parseNetworkOutput(value: unknown): DeltrelNetworkOutput | null {
  if (value === null) return null;
  if (!record(value) || !exact(value, ['schemaVersion', 'perspective', 'nodeCount', 'auxiliaryStatus', 'heads']) ||
      value.schemaVersion !== 1 || (value.perspective !== 0 && value.perspective !== 1) ||
      ![50, 105, 180, 275].includes(value.nodeCount as number) ||
      !['ready', 'untrained', 'absent'].includes(value.auxiliaryStatus as string) ||
      !record(value.heads) || !exact(value.heads, DELTREL_NETWORK_HEAD_NAMES)) return invalid();
  const heads = {} as DeltrelNetworkOutput['heads'];
  for (const [index, name] of DELTREL_NETWORK_HEAD_NAMES.entries()) {
    const item = value.heads[name];
    const absent = index >= 6 && value.auxiliaryStatus === 'absent';
    if (absent ? item !== null : item === null) return invalid();
    heads[name] = absent ? null : parseHead(item, name, value.nodeCount as number);
  }
  return {schemaVersion: 1, perspective: value.perspective, nodeCount: value.nodeCount as number,
    auxiliaryStatus: value.auxiliaryStatus as DeltrelNetworkOutput['auxiliaryStatus'], heads};
}
export function parseServerNetworkOutput(value: unknown): DeltrelNetworkOutput | null {
  if (value === null) return null;
  if (!record(value) || !exact(value, ['schema_version', 'perspective', 'node_count', 'auxiliary_status', 'heads']) ||
      !record(value.heads) || !exact(value.heads, Object.values(WIRE_NAMES))) return invalid();
  const wireHeads = value.heads;
  return parseNetworkOutput({schemaVersion: value.schema_version, perspective: value.perspective,
    nodeCount: value.node_count, auxiliaryStatus: value.auxiliary_status,
    heads: Object.fromEntries(DELTREL_NETWORK_HEAD_NAMES.map(name => [name, wireHeads[WIRE_NAMES[name]]]))});
}
export function validateNetworkOutputState(output: DeltrelNetworkOutput, request: DeltrelAiRequest, swapped: boolean): void {
  const s = request.state, n = s.stones.length;
  if (output.perspective !== s.toMove || output.nodeCount !== n) return invalid();
  for (const name of DELTREL_NETWORK_HEAD_NAMES) {
    const head = output.heads[name];
    if (head === null) continue;
    const expectedApplicable = name === 'secondStone'
      ? s.mode === 'double' && !s.opening && s.movesLeft === 2 && !swapped
      : name === 'opponentReply' ? s.stones.filter(x => x === -1).length > s.movesLeft : true;
    if (head.applicable !== expectedApplicable) return invalid();
    for (let i = 0; i < head.mask.length; i++) {
      const expected = name === 'policy' || name === 'softPolicy' || name === 'secondStone' ? s.stones[i] === -1
        : name === 'opponentReply' ? i === n || s.stones[i] === -1
        : name === 'finalShores' ? i % 51 <= 5 * s.rings
        : name === 'finalNetworks' ? i % 26 <= Math.floor(5 * s.rings / 2) : true;
      if (head.mask[i] !== expected) return invalid();
    }
  }
}
