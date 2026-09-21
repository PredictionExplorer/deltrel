import { describe, expect, it } from 'vitest';
import {
  coordinateLabel,
  getBoard,
  isSupportedRings,
  parseLabel,
  perimeterCycle,
  SUPPORTED_RINGS,
} from '../board';

function edgeCount(rings: number): number {
  // cycles: sum 5x = 5r(r+1)/2; inter-ring: sum_{x=2..r} 5(2x-1) = 5(r^2-1);
  // bridge chords: K5 minus the 5 existing ring-1 cycle edges = 5.
  return (5 * rings * (rings + 1)) / 2 + 5 * (rings * rings - 1) + 5;
}

describe('board construction', () => {
  it('matches published node counts (105 / 180 / 275)', () => {
    expect(getBoard(6).n).toBe(105);
    expect(getBoard(8).n).toBe(180);
    expect(getBoard(10).n).toBe(275);
    expect(getBoard(6).shoreCount).toBe(30);
    expect(getBoard(8).shoreCount).toBe(40);
    expect(getBoard(10).shoreCount).toBe(50);
  });

  it('has correct counts, degrees and symmetry for all sizes', () => {
    for (const r of SUPPORTED_RINGS) {
      const b = getBoard(r);
      expect(b.n).toBe((5 * r * (r + 1)) / 2);
      let peris = 0;
      let capes = 0;
      for (let u = 0; u < b.n; u++) {
        peris += b.isShore[u];
        capes += b.isCape[u];
      }
      expect(peris).toBe(5 * r);
      expect(capes).toBe(5);

      // Handshake + closed-form edge count.
      expect(b.adj.length).toBe(2 * edgeCount(r));
      // Symmetry and minimum degree.
      const neighborSets: Set<number>[] = [];
      for (let u = 0; u < b.n; u++) {
        const s = new Set<number>();
        for (let e = b.adjOff[u]; e < b.adjOff[u + 1]; e++) s.add(b.adj[e]);
        expect(s.size).toBe(b.adjOff[u + 1] - b.adjOff[u]); // no duplicates
        expect(s.has(u)).toBe(false); // no self loops
        expect(s.size).toBeGreaterThanOrEqual(3);
        neighborSets.push(s);
      }
      for (let u = 0; u < b.n; u++) {
        for (const v of neighborSets[u]) expect(neighborSets[v].has(u)).toBe(true);
      }
    }
  });

  it('uses unique sequential alphabetic coordinates across every board size', () => {
    expect([0, 25, 26, 51, 52, 274].map(coordinateLabel)).toEqual([
      'A', 'Z', 'AA', 'AZ', 'BA', 'JO',
    ]);
    for (const rings of SUPPORTED_RINGS) {
      const board = getBoard(rings);
      expect(new Set(board.labels).size).toBe(board.n);
      for (let node = 0; node < board.n; node++) {
        expect(board.labels[node]).toMatch(/^[A-Z]+$/);
        expect(parseLabel(board, board.labels[node])).toBe(node);
        expect(board.labels[node]).toBe(getBoard(10).labels[node]);
      }
      expect(() => parseLabel(board, 'ZZZ')).toThrow('unknown node label');
    }
    for (const invalid of [-1, 0.5, NaN, Infinity]) {
      expect(() => coordinateLabel(invalid)).toThrow('non-negative safe integer');
    }
  });

  it('has the expected hand-derived adjacencies on the 4-ring board', () => {
    const b = getBoard(4);
    const adj = (p: string, q: string) => {
      const u = parseLabel(b, p);
      const v = parseLabel(b, q);
      for (let e = b.adjOff[u]; e < b.adjOff[u + 1]; e++) {
        if (b.adj[e] === v) return true;
      }
      return false;
    };
    // ring cycle + sector wrap
    expect(adj('AE', 'AF')).toBe(true);
    expect(adj('AH', 'AI')).toBe(true);
    expect(adj('AX', 'AE')).toBe(true);
    // radial / diagonal
    expect(adj('AE', 'P')).toBe(true);
    expect(adj('AH', 'R')).toBe(true);
    expect(adj('AJ', 'S')).toBe(true);
    // corner cross
    expect(adj('G', 'B')).toBe(true);
    expect(adj('AH', 'S')).toBe(true);
    expect(adj('AL', 'V')).toBe(true);
    // bridge K5 (non-neighboring arms too)
    expect(adj('A', 'B')).toBe(true);
    expect(adj('A', 'C')).toBe(true);
    expect(adj('B', 'D')).toBe(true);
    // non-edges
    expect(adj('AE', 'AG')).toBe(false);
    expect(adj('AE', 'AI')).toBe(false);
    expect(adj('A', 'P')).toBe(false);
  });

  it('lays out the perimeter on the unit circumcircle pentagon', () => {
    const b = getBoard(6);
    const shore = perimeterCycle(b);
    expect(shore.length).toBe(30);
    for (const u of shore) {
      const radius = Math.hypot(b.xs[u], b.ys[u]);
      expect(radius).toBeGreaterThan(0.75); // on the outer pentagon
      expect(radius).toBeLessThanOrEqual(1.000001);
    }
    expect(b.minEdge).toBeGreaterThan(0);
  });

  it('accepts exactly the four supported ring counts', () => {
    expect(SUPPORTED_RINGS).toEqual([4, 6, 8, 10]);
    for (const rings of SUPPORTED_RINGS) {
      expect(isSupportedRings(rings)).toBe(true);
      expect(getBoard(rings).rings).toBe(rings);
    }
    for (const rings of [2, 3, 5, 7, 9, 11, 12, 4.5, Number.NaN]) {
      expect(isSupportedRings(rings)).toBe(false);
      expect(() => getBoard(rings)).toThrow(/one of 4, 6, 8, 10/);
    }
  });
});
