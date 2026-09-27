import { afterEach, describe, expect, it, vi } from 'vitest';
import { createGameSearchSeed, deriveRootSearchSeed, isSearchSeed, resolveSearchSeed } from '../search-seed';

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('search seeds', () => {
  // Recorded independently from the published native WASM derive_root_seed.
  it.each([
    [0, '0000000000000000', 'e220a8397b1dcdaf'],
    [0, 'ffffffffffffffff', 'e4d971771b652c20'],
    [Number.MAX_SAFE_INTEGER, 'fedcba9876543210', '0b29a4596fa57cc0'],
    [123, 'abcdef1234567890', '4f38514236619424'],
  ] as const)('matches native root derivation for seed %s and hash %s', (seed, hash, expected) => {
    expect(deriveRootSearchSeed(seed, BigInt(`0x${hash}`))).toBe(BigInt(`0x${expected}`));
  });

  it.each([0, 1, Number.MAX_SAFE_INTEGER])('preserves explicit seed %s exactly', (seed) => {
    expect(isSearchSeed(seed)).toBe(true);
    expect(resolveSearchSeed('zobrist64:0000000000000042', seed)).toBe(seed);
    expect(resolveSearchSeed('zobrist64:ffffffffffffffff', seed)).toBe(seed);
    expect(JSON.parse(JSON.stringify(seed))).toBe(seed);
  });

  it.each([-1, 0.5, Number.MAX_SAFE_INTEGER + 1, NaN, Infinity, -Infinity, null, '7', true, {}, []])(
    'rejects invalid seed %j', (seed) => {
      expect(isSearchSeed(seed)).toBe(false);
      expect(() => resolveSearchSeed('zobrist64:0000000000000042', seed as number)).toThrow(/search seed/);
      expect(() => deriveRootSearchSeed(seed as number, BigInt(0))).toThrow(/search seed/);
    },
  );

  it('retains the exact legacy hash mask when no seed is supplied', () => {
    expect(isSearchSeed(undefined)).toBe(false);
    expect(resolveSearchSeed('zobrist64:0000000000000000')).toBe(0);
    expect(resolveSearchSeed('zobrist64:0000000000000042')).toBe(66);
    expect(resolveSearchSeed('zobrist64:ffffffffffffffff')).toBe(Number.MAX_SAFE_INTEGER);
    expect(resolveSearchSeed('zobrist64:ff20000000000042')).toBe(66);
    for (const stateHash of ['', '66', 'zobrist64:0', 'zobrist64:FFFFFFFFFFFFFFFF']) {
      expect(() => resolveSearchSeed(stateHash)).toThrow(/state hash/);
    }
  });

  it('uses 53 browser-random bits and preserves both endpoint seeds', () => {
    const getRandomValues = vi.fn((words: Uint32Array) => words.fill(0))
      .mockImplementationOnce((words: Uint32Array) => words.fill(0xffff_ffff));
    vi.stubGlobal('crypto', { getRandomValues });
    const fallback = vi.spyOn(Math, 'random');
    expect(createGameSearchSeed()).toBe(Number.MAX_SAFE_INTEGER);
    expect(createGameSearchSeed()).toBe(0);
    expect(getRandomValues).toHaveBeenCalledTimes(2);
    expect(fallback).not.toHaveBeenCalled();
  });

  it.each([undefined, {}, { getRandomValues: () => { throw new Error('unavailable'); } }])(
    'falls back to independent random words when browser randomness is unavailable', (crypto) => {
      vi.stubGlobal('crypto', crypto);
      vi.spyOn(Math, 'random').mockReturnValueOnce(0.5).mockReturnValueOnce(0.25);
      expect(createGameSearchSeed()).toBe(0x10_0000 * 0x1_0000_0000 + 0x4000_0000);
    },
  );
});
