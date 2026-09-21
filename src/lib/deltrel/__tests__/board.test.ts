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

  it.each([
    [4, 11, 10, ['I1', 'C1', 'A7', 'F10', 'K7']],
    [6, 15, 15, ['L1', 'D1', 'A9', 'H15', 'O9']],
    [8, 21, 19, ['Q1', 'E1', 'A12', 'K19', 'U12']],
    [10, 25, 24, ['T1', 'F1', 'A15', 'M24', 'Y15']],
  ] as const)('uses spatial files/ranks on the %i-ring board', (rings, files, ranks, capes) => {
      const board = getBoard(rings);
      const grid = board.coordinateGrid;
      expect(grid.columns).toHaveLength(files);
      expect(grid.ranks).toHaveLength(ranks);
      expect(capes.map((label) => parseLabel(board, label))).toEqual(
        [0, 1, 2, 3, 4].map((sector) => board.idx(sector, rings, 0)),
      );
      expect(new Set(board.labels).size).toBe(board.n);
      for (let node = 0; node < board.n; node++) {
        expect(board.labels[node]).toMatch(/^[A-Z][1-9]\d?$/);
        expect(parseLabel(board, board.labels[node])).toBe(node);
        expect(parseLabel(board, ` ${board.labels[node].toLowerCase()} `)).toBe(node);
        expect(coordinateLabel(node, rings)).toBe(board.labels[node]);
        // Each physical node must lie inside the cell named by its two axes.
        const column = grid.columns[board.columnOf[node]];
        const rank = grid.ranks[board.rankOf[node] - 1];
        expect(Math.abs(board.xs[node] - column.x)).toBeLessThanOrEqual(grid.step / 2 + 1e-8);
        expect(Math.abs(board.ys[node] - rank.y)).toBeLessThanOrEqual(grid.step / 2 + 1e-8);
      }
      const byX = Array.from({ length: board.n }, (_, node) => node).sort((a, b) => board.xs[a] - board.xs[b]);
      const byHeight = Array.from({ length: board.n }, (_, node) => node).sort((a, b) => board.ys[b] - board.ys[a]);
      expect(byX.every((node, i) => i === 0 || board.columnOf[node] >= board.columnOf[byX[i - 1]])).toBe(true);
      expect(byHeight.every((node, i) => i === 0 || board.rankOf[node] >= board.rankOf[byHeight[i - 1]])).toBe(true);
      // A1 is outside the pentagonal playable area, even though both axes exist.
      for (const invalid of ['A1', 'A0', 'A01', 'A', 'ZZZ', 'F 10']) {
        expect(() => parseLabel(board, invalid)).toThrow('unknown node label');
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
    expect(adj('I1', 'G1')).toBe(true);
    expect(adj('E1', 'C1')).toBe(true);
    expect(adj('I2', 'I1')).toBe(true);
    // radial / diagonal
    expect(adj('I1', 'H2')).toBe(true);
    expect(adj('E1', 'E2')).toBe(true);
    expect(adj('C2', 'D2')).toBe(true);
    // corner cross
    expect(adj('F3', 'E4')).toBe(true);
    expect(adj('E1', 'D2')).toBe(true);
    expect(adj('B5', 'B6')).toBe(true);
    // bridge K5 (non-neighboring arms too)
    expect(adj('G4', 'E4')).toBe(true);
    expect(adj('G4', 'E5')).toBe(true);
    expect(adj('E4', 'F6')).toBe(true);
    // non-edges
    expect(adj('I1', 'F1')).toBe(false);
    expect(adj('I1', 'C1')).toBe(false);
    expect(adj('G4', 'H2')).toBe(false);
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
