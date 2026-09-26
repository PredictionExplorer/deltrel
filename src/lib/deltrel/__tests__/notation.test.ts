import { describe, expect, it } from 'vitest';
import { coordinateAt, coordinateLabel, DELTREL_NOTATION_SCHEMA_ID } from '../notation';
import { DELTREL_RULES_HASH } from '../rules';

describe('independent board notation', () => {
  it('keeps the deployed game fingerprint while versioning display notation', () => {
    expect(DELTREL_NOTATION_SCHEMA_ID).toBe('deltrel.board-notation.v3');
    expect(DELTREL_RULES_HASH).toBe('fnv1a64:46e4fbcff4e17fd3');
  });

  it('uses sector, outward ring, and clockwise offset with 0 for ring ten', () => {
    expect(coordinateAt(10, 0, 1, 0)).toBe('A10');
    expect(coordinateAt(10, 0, 3, 2)).toBe('A32');
    expect(coordinateAt(10, 3, 9, 8)).toBe('D98');
    expect(coordinateAt(10, 3, 10, 0)).toBe('D00');
    expect(coordinateAt(10, 4, 10, 9)).toBe('E09');
  });

  it.each([0, 3, 5, 11, 4.5, NaN, Infinity])('rejects unsupported board size %s', (rings) => {
    expect(() => coordinateAt(rings, 0, 1, 0)).toThrow('supported board size');
    expect(() => coordinateLabel(0, rings)).toThrow('supported board size');
  });

  it.each([
    [-1, 1, 0], [5, 1, 0], [0.5, 1, 0], [NaN, 1, 0],
    [0, 0, 0], [0, 5, 0], [0, 1.5, 0], [0, Infinity, 0],
    [0, 1, -1], [0, 1, 1], [0, 2, 0.5], [0, 2, NaN],
  ])('rejects invalid board addresses (%s,%s,%s)', (sector, ring, position) => {
    expect(() => coordinateAt(4, sector, ring, position)).toThrow('invalid board address');
  });
});
