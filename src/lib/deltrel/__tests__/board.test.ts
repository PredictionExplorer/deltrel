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

  it.each(SUPPORTED_RINGS)('uses unique polar addresses on the %i-ring board', (rings) => {
    const board = getBoard(rings);
    const ringDigit = rings % 10;
    const capes = ['A', 'B', 'C', 'D', 'E'].map((arm) => `${arm}${ringDigit}0`);
    expect(capes.map((label) => parseLabel(board, label))).toEqual(
      [0, 1, 2, 3, 4].map((sector) => board.idx(sector, rings, 0)),
    );
    expect(new Set(board.labels).size).toBe(board.n);
    for (let node = 0; node < board.n; node++) {
      expect(board.labels[node]).toMatch(/^[A-E][0-9]{2}$/);
      expect(parseLabel(board, board.labels[node])).toBe(node);
      expect(parseLabel(board, ` ${board.labels[node].toLowerCase()} `)).toBe(node);
      expect(coordinateLabel(node, rings)).toBe(board.labels[node]);
      expect(Number(board.labels[node][1]) || 10).toBe(board.ringOf[node]);
      expect(Number(board.labels[node][2])).toBe(board.posOf[node]);
    }
    for (const invalid of ['A1', 'A0', 'A11', 'A', 'F10', 'A100', 'A 10']) {
      expect(() => parseLabel(board, invalid)).toThrow('unknown node label');
    }
  });

  it('keeps each address unchanged when outer rings are added', () => {
    const full = getBoard(10);
    for (const rings of SUPPORTED_RINGS) {
      const board = getBoard(rings);
      expect(board.labels).toEqual(full.labels.slice(0, board.n));
      if (rings < 10) expect(() => parseLabel(board, 'A00')).toThrow('unknown node label');
    }
  });

  it('rejects invalid ids rather than inventing coordinates', () => {
    for (const invalid of [-1, 0.5, NaN, Infinity, 50, Number.MAX_SAFE_INTEGER]) {
      expect(() => coordinateLabel(invalid, 4)).toThrow('non-negative safe integer');
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
    expect(adj('A40', 'A41')).toBe(true);
    expect(adj('A43', 'B40')).toBe(true);
    expect(adj('E43', 'A40')).toBe(true);
    // radial / diagonal
    expect(adj('A40', 'A30')).toBe(true);
    expect(adj('A43', 'A32')).toBe(true);
    expect(adj('B41', 'B30')).toBe(true);
    // corner cross
    expect(adj('A21', 'B10')).toBe(true);
    expect(adj('A43', 'B30')).toBe(true);
    expect(adj('B43', 'C30')).toBe(true);
    // bridge K5 (non-neighboring arms too)
    expect(adj('A10', 'B10')).toBe(true);
    expect(adj('A10', 'C10')).toBe(true);
    expect(adj('B10', 'D10')).toBe(true);
    // non-edges
    expect(adj('A40', 'A42')).toBe(false);
    expect(adj('A40', 'B40')).toBe(false);
    expect(adj('A10', 'A30')).toBe(false);
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
