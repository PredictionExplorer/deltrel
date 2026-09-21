import { describe, expect, it } from 'vitest';
import {
  getBoard,
  SUPPORTED_RINGS,
  type Board,
} from '../board';
import { EMPTY, scorePosition, validateTerminalWinner } from '../scoring';
import { referenceScore } from './reference';

// Mechanical fixtures use stable node IDs, independently of display notation.
function position(
  board: Board,
  blue: readonly number[],
  red: readonly number[],
): Int8Array {
  const stones = new Int8Array(board.n).fill(EMPTY);
  for (const node of blue) stones[node] = 0;
  for (const node of red) stones[node] = 1;
  return stones;
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe('scorePosition: hand-built fixtures', () => {
  it('scores bridge-connected networks, separate networks and dead stones (A)', () => {
    const b = getBoard(4);
    const stones = position(
      b,
      // Blue: sector 1 + sector 3 joined through the central bridge = ONE network.
      [1, 7, 18, 34, 3, 11, 24, 42, 43],
      // Red: sectors 0 and 2 joined through the bridge (one network), a separate
      // two-shore network 47-48 (kept clear of 30, which is ring-adjacent to
      // 49), and a dead lone stone on shore 36.
      [0, 5, 15, 30, 31, 2, 9, 21, 38, 47, 48, 36],
    );
    const r = scorePosition(b, stones);
    expect(r.players[0]).toEqual({
      shores: 3, // 34, 42, 43
      capes: 2, // 34, 42
      networks: 1,
      capeBonus: 0,
      award: 2, // 2 x (2 - 1)
      total: 5,
    });
    expect(r.players[1]).toEqual({
      shores: 5, // 30, 31, 38, 47, 48
      capes: 2, // 30, 38
      networks: 2,
      capeBonus: 0,
      award: -2,
      total: 3,
    });
    expect(r.contestedShores).toBe(12);
    expect(r.leader).toBe(0);
    // The lone red stone on 36 is dead and its region touches both colors.
    expect(r.aliveStone[36]).toBe(0);
    expect(r.nodeOwner[36]).toBe(-1);
  });

  it('gives an enclosed dead stone’s shore to the surrounding network (B)', () => {
    const b = getBoard(4);
    // Red walls off the corner shore 33 completely: 32 (ring cycle), 34
    // (cycle wrap), 17 (diagonal), 18 (corner cross). A lone blue stone on
    // 33 is dead; its shore is claimed by the surrounding red network.
    const stones = position(b, [33, 40, 41], [32, 17, 18, 34]);
    const r = scorePosition(b, stones);
    expect(r.players[1]).toEqual({
      shores: 3, // 32, 34 occupied + 33 enclosed
      capes: 1, // 34
      networks: 1,
      capeBonus: 0,
      award: 0,
      total: 3,
    });
    expect(r.players[0]).toEqual({
      shores: 2, // 40, 41
      capes: 0,
      networks: 1,
      capeBonus: 0,
      award: 0,
      total: 2,
    });
    const dead = 33;
    expect(r.aliveStone[dead]).toBe(0);
    expect(r.nodeOwner[dead]).toBe(1);
    // Same position with 33 empty instead: red still owns the shore.
    stones[dead] = EMPTY;
    const r2 = scorePosition(b, stones);
    expect(r2.players[1].shores).toBe(3);
    expect(r2.nodeOwner[dead]).toBe(1);
  });

  it('awards the cape bonus for three corners and the network-count award (C)', () => {
    const b = getBoard(4);
    const stones = position(
      b,
      // Blue: one six-shore network spanning sectors 3 and 4.
      [42, 43, 44, 45, 46, 47],
      // Red: three separate two-stone corner networks.
      [30, 31, 34, 35, 38, 39],
    );
    const r = scorePosition(b, stones);
    expect(r.players[0]).toEqual({
      shores: 6,
      capes: 2, // 42, 46
      networks: 1,
      capeBonus: 0,
      award: 4,
      total: 10,
    });
    expect(r.players[1]).toEqual({
      shores: 6,
      capes: 3, // 30, 34, 38
      networks: 3,
      capeBonus: 1,
      award: -4,
      total: 3,
    });
    expect(r.leader).toBe(0);
  });

  it('breaks ties by cape count (D)', () => {
    const b = getBoard(4);
    const r = scorePosition(b, position(b, [30, 31], [35, 36]));
    expect(r.players[0].total).toBe(2);
    expect(r.players[1].total).toBe(2);
    expect(r.players[0].capes).toBe(1);
    expect(r.players[1].capes).toBe(0);
    expect(r.leader).toBe(0);
  });

  it('lets both players bridge simultaneously through the center (E)', () => {
    const b = getBoard(4);
    const r = scorePosition(
      b,
      position(
        b,
        [0, 5, 15, 30, 3, 11, 24, 42], // non-adjacent arms
        [1, 7, 18, 34, 2, 9, 21, 38],
      ),
    );
    // Each pair of arms is one network thanks to the K5 bridge.
    expect(r.players[0].networks).toBe(1);
    expect(r.players[1].networks).toBe(1);
    expect(r.players[0].award).toBe(0);
    expect(r.players[1].award).toBe(0);
  });

  it('scores the empty board as all-contested', () => {
    const b = getBoard(6);
    const r = scorePosition(b, new Int8Array(b.n).fill(EMPTY));
    expect(r.players[0].total).toBe(0);
    expect(r.players[1].total).toBe(0);
    expect(r.contestedShores).toBe(30);
    expect(r.leader).toBe(-1);
  });

  it('does not let territory bootstrap a lone perimeter stone', () => {
    const b = getBoard(6);
    const stones = new Int8Array(b.n).fill(EMPTY);
    const lone = b.idx(0, 6, 0);
    stones[lone] = 0;

    const got = scorePosition(b, stones);
    const want = referenceScore(b, stones);
    expect(got.players).toEqual([
      { shores: 0, capes: 0, networks: 0, capeBonus: 0, award: 0, total: 0 },
      { shores: 0, capes: 0, networks: 0, capeBonus: 0, award: 0, total: 0 },
    ]);
    expect(got.aliveStone[lone]).toBe(0);
    expect(got.nodeOwner[lone]).toBe(-1);
    expect(got.contestedShores).toBe(b.shoreCount);
    expect(got.players).toEqual(want.players);
    expect(Array.from(got.aliveStone)).toEqual(want.aliveStone);
    expect(Array.from(got.nodeOwner)).toEqual(want.nodeOwner);
  });

});

describe('scorePosition: cross-validation and invariants', () => {
  it('matches the naive reference scorer on every supported board', () => {
    const rng = mulberry32(0xdecafbad);
    for (const rings of SUPPORTED_RINGS) {
      const b = getBoard(rings);
      for (const density of [0.02, 0.1, 0.25, 0.55, 0.8, 1]) {
        for (let trial = 0; trial < 30; trial++) {
          const stones = new Int8Array(b.n).fill(EMPTY);
          for (let u = 0; u < b.n; u++) {
            if (rng() < density) stones[u] = rng() < 0.5 ? 0 : 1;
          }
          const got = scorePosition(b, stones);
          const want = referenceScore(b, stones);
          expect(got.players).toEqual(want.players);
          expect(got.contestedShores).toBe(want.contestedShores);
          expect(Array.from(got.aliveStone)).toEqual(want.aliveStone);
          expect(Array.from(got.nodeOwner)).toEqual(want.nodeOwner);
        }
      }
    }
  });

  it('makes every sampled full supported board decided and decisive', () => {
    const rng = mulberry32(0x5717a5);
    for (const rings of SUPPORTED_RINGS) {
      const b = getBoard(rings);
      for (let trial = 0; trial < 120; trial++) {
        const stones = new Int8Array(b.n);
        // Random full fill, biased per-trial so both clumpy and mixed boards occur.
        const bias = 0.25 + 0.5 * rng();
        for (let u = 0; u < b.n; u++) stones[u] = rng() < bias ? 0 : 1;
        const r = scorePosition(b, stones);
        const sum = r.players[0].total + r.players[1].total;
        const margin = r.players[0].total - r.players[1].total;
        expect(r.contestedShores).toBe(0);
        expect(r.players[0].shores + r.players[1].shores).toBe(b.shoreCount);
        expect(r.players[0].capes + r.players[1].capes).toBe(5);
        expect(r.players[0].capeBonus + r.players[1].capeBonus).toBe(1);
        expect(r.players[0].award + r.players[1].award).toBe(0);
        expect(sum).toBe(5 * rings + 1);
        expect(margin).not.toBe(0);
        expect(Math.abs(margin) % 2).toBe(1);
        expect(r.leader).toBe(margin > 0 ? 0 : 1);
      }
    }
  });

  it('agrees with the simplified and Schmittberger scoring margins', () => {
    const rng = mulberry32(0xace0fba5);
    const b = getBoard(6);
    for (let trial = 0; trial < 200; trial++) {
      const stones = new Int8Array(b.n).fill(EMPTY);
      for (let u = 0; u < b.n; u++) {
        if (rng() < 0.7) stones[u] = rng() < 0.5 ? 0 : 1;
      }
      const r = scorePosition(b, stones);
      const [p0, p1] = r.players;
      // Simplified: sum over networks of (owned shores - 4), plus the cape
      // shore. Identical margin to the conventional system.
      const simplified0 = p0.shores - 4 * p0.networks + p0.capeBonus;
      const simplified1 = p1.shores - 4 * p1.networks + p1.capeBonus;
      expect(simplified0 - simplified1).toBe(p0.total - p1.total);
      // Schmittberger alternative: own shores + cape bonus minus opponent's,
      // plus (own award - opponent award). Algebraically this equals the
      // conventional score difference, so positive iff conventional winner.
      const schmitt0 =
        p0.shores + p0.capeBonus - (p1.shores + p1.capeBonus) + (p0.award - p1.award);
      expect(schmitt0).toBe(p0.total - p1.total);
    }
  });

  it('handles a sustained 1k-position full-board workload', () => {
    const rng = mulberry32(0xbe5eeed);
    const b = getBoard(10);
    const fills: Int8Array[] = [];
    for (let i = 0; i < 50; i++) {
      const stones = new Int8Array(b.n);
      for (let u = 0; u < b.n; u++) stones[u] = rng() < 0.5 ? 0 : 1;
      fills.push(stones);
    }
    let sink = 0;
    for (let i = 0; i < 1_000; i++) {
      sink += scorePosition(b, fills[i % fills.length]).players[0].total;
    }
    expect(Number.isFinite(sink)).toBe(true);
  });

  it('validates a binary terminal winner and all final invariants', () => {
    for (const rings of SUPPORTED_RINGS) {
      const board = getBoard(rings);
      const stones = Int8Array.from(
        { length: board.n },
        (_, node) => (node % 3 === 0 ? 0 : 1),
      );
      const terminal = validateTerminalWinner(board, stones);
      expect(terminal.winner).toBe(terminal.margin > 0 ? 0 : 1);
      expect(terminal.score.leader).toBe(terminal.winner);
      expect(terminal.score.contestedShores).toBe(0);
      expect(
        terminal.score.players[0].total + terminal.score.players[1].total,
      ).toBe(5 * rings + 1);
      expect(Math.abs(terminal.margin) % 2).toBe(1);
    }
  });

  it('rejects nonfull, unsupported, and synthetic contested terminals', () => {
    const board = getBoard(4);
    expect(() =>
      validateTerminalWinner(board, new Int8Array(board.n).fill(EMPTY)),
    ).toThrow(/full/);

    const unsupported = { ...board, rings: 5 } as Board;
    expect(() =>
      validateTerminalWinner(unsupported, new Int8Array(board.n)),
    ).toThrow(/unsupported/);

    const disconnected = {
      ...board,
      adjOff: new Int32Array(board.n + 1),
      adj: new Int32Array(),
    };
    const mixed = Int8Array.from(
      { length: board.n },
      (_, node) => (node % 2) as 0 | 1,
    );
    expect(() => validateTerminalWinner(disconnected, mixed)).toThrow(
      /zero contested shores/,
    );
  });
});
