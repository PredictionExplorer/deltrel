import { describe, expect, it } from 'vitest';
import { coordinateAt, createCoordinateGrid, DELTREL_NOTATION_SCHEMA_ID } from '../notation';
import { DELTREL_RULES_HASH } from '../rules';

describe('independent board notation', () => {
  it('keeps the deployed game fingerprint while versioning display notation', () => {
    expect(DELTREL_NOTATION_SCHEMA_ID).toBe('deltrel.board-notation.v2');
    expect(DELTREL_RULES_HASH).toBe('fnv1a64:46e4fbcff4e17fd3');
  });

  it('handles exact positive half-cell ties consistently', () => {
    const grid = createCoordinateGrid(10);
    expect(coordinateAt(grid, 3, 10, 0)).toEqual({ column: 12, rank: 24, label: 'M24' });
    expect(coordinateAt(grid, 3, 2, 0)).toEqual({ column: 12, rank: 14, label: 'M14' });
  });

  it.each([0, 3, 5, 11, 4.5, NaN, Infinity])('rejects unsupported board size %s', (rings) => {
    expect(() => createCoordinateGrid(rings)).toThrow('supported board size');
  });

  it.each([
    [-1, 1, 0], [5, 1, 0], [0.5, 1, 0],
    [0, 0, 0], [0, 5, 0], [0, 1.5, 0],
    [0, 1, -1], [0, 1, 1], [0, 2, 0.5],
  ])('rejects invalid board addresses (%s,%s,%s)', (sector, ring, position) => {
    expect(() => coordinateAt(createCoordinateGrid(4), sector, ring, position)).toThrow('invalid board address');
  });
});
