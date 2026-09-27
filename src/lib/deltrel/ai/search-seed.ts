import { DeltrelAiError } from './errors';

/** Seeds stay exactly representable across JSON, JavaScript, and native search. */
export function isSearchSeed(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
}

/** One game identity, mixed with each position by the shared search engine. */
export function createGameSearchSeed(): number {
  const words = new Uint32Array(2);
  try {
    if (typeof globalThis.crypto?.getRandomValues === 'function') {
      globalThis.crypto.getRandomValues(words);
      return (words[0] & 0x1f_ffff) * 0x1_0000_0000 + words[1];
    }
  } catch { /* Gameplay variety still works when browser randomness is unavailable. */ }
  return Math.floor(Math.random() * 0x20_0000) * 0x1_0000_0000 +
    Math.floor(Math.random() * 0x1_0000_0000);
}

/** Omitted seeds retain the legacy, position-determined search for older callers. */
export function resolveSearchSeed(stateHash: string, seed?: number): number {
  if (seed !== undefined) {
    if (!isSearchSeed(seed)) {
      throw new DeltrelAiError('protocol', 'AI search seed must be a nonnegative safe integer.');
    }
    return seed;
  }
  const match = /^zobrist64:([0-9a-f]{16})$/.exec(stateHash);
  if (!match) throw new DeltrelAiError('protocol', 'AI state hash cannot seed search.');
  return Number(BigInt(`0x${match[1]}`) & BigInt(Number.MAX_SAFE_INTEGER));
}

/** Exact native derive_root_seed(seed, state_hash, 0) for older browser engines. */
export function deriveRootSearchSeed(seed: number, stateHash: bigint): bigint {
  if (!isSearchSeed(seed)) {
    throw new DeltrelAiError('protocol', 'AI search seed must be a nonnegative safe integer.');
  }
  const hash = BigInt.asUintN(64, stateHash);
  const rotatedHash = BigInt.asUintN(64, (hash << BigInt(17)) | (hash >> BigInt(47)));
  let mixed = BigInt.asUintN(64, (BigInt(seed) ^ rotatedHash) + BigInt('0x9e3779b97f4a7c15'));
  mixed = BigInt.asUintN(64, (mixed ^ (mixed >> BigInt(30))) * BigInt('0xbf58476d1ce4e5b9'));
  mixed = BigInt.asUintN(64, (mixed ^ (mixed >> BigInt(27))) * BigInt('0x94d049bb133111eb'));
  return mixed ^ (mixed >> BigInt(31));
}
