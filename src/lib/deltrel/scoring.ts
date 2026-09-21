/**
 * Deltrel scoring: connected waterways establish networks by directly occupying
 * at least two shoreline nodes. Unestablished groups are removed before empty
 * regions are assigned to the single surrounding network color, if any.
 *
 * Score = owned shoreline nodes + one point for owning at least three capes
 *       + 2 × (opponent established networks − own established networks).
 * Ties are broken by cape ownership. Every placement and adjacency remains
 * part of the shared, versioned rules contract.
 *
 * A union-find pass over CSR adjacency and a flood fill evaluate the position
 * using preallocated typed arrays in O((N + E) × alpha(N)) time.
 */

import { isSupportedRings, type Board } from './board';

export const EMPTY = -1;

export interface PlayerScore {
  /** Shores owned: occupied by this player's networks + enclosed territory. */
  shores: number;
  /** Corner shores (capes) owned, 0..5. */
  capes: number;
  /** Number of alive networks. */
  networks: number;
  /** 1 if this player owns three or more capes. */
  capeBonus: 0 | 1;
  /** 2 x (opponent networks - own networks). */
  award: number;
  /** shores + capeBonus + award. */
  total: number;
}

export interface ScoreResult {
  players: [PlayerScore, PlayerScore];
  /**
   * Controller of each node: 0 or 1, or -1 for none/contested. Stones of
   * alive networks map to their color; dead stones and empty nodes map to the
   * player whose networks solely border their region, if any.
   */
  nodeOwner: Int8Array;
  /** 1 for stones that belong to an alive network, 0 otherwise. */
  aliveStone: Uint8Array;
  /** Number of shores owned by neither player at this position. */
  contestedShores: number;
  /** Player ahead if scored now (0 | 1), or -1 for a dead tie. */
  leader: 0 | 1 | -1;
}

export interface TerminalWinnerResult {
  score: ScoreResult & { leader: 0 | 1 };
  winner: 0 | 1;
  /** Player-zero total minus player-one total. */
  margin: number;
}

/** Score a position. `stones[u]` is EMPTY (-1), 0, or 1. */
export function scorePosition(board: Board, stones: ArrayLike<number>): ScoreResult {
  const { n, adjOff, adj, isShore, isCape } = board;

  // ---- 1. Union-find over same-color adjacent stones ----------------------
  const parent = new Int32Array(n);
  for (let u = 0; u < n; u++) parent[u] = u;
  const find = (u: number): number => {
    let r = u;
    while (parent[r] !== r) {
      parent[r] = parent[parent[r]];
      r = parent[r];
    }
    return r;
  };
  for (let u = 0; u < n; u++) {
    const c = stones[u];
    if (c === EMPTY) continue;
    for (let e = adjOff[u]; e < adjOff[u + 1]; e++) {
      const v = adj[e];
      if (v > u && stones[v] === c) {
        const ru = find(u);
        const rv = find(v);
        if (ru !== rv) parent[rv] = ru;
      }
    }
  }

  // ---- 2. Networks: groups occupying >= 2 shores (static) ---------------------
  const occShores = new Int32Array(n); // indexed by group root
  for (let u = 0; u < n; u++) {
    if (stones[u] !== EMPTY && isShore[u]) occShores[find(u)]++;
  }
  const alive = new Uint8Array(n); // per node: stone of an alive network
  for (let u = 0; u < n; u++) {
    if (stones[u] !== EMPTY && occShores[find(u)] >= 2) alive[u] = 1;
  }

  // ---- 3. Territory regions over empty cells + dead stones ----------------
  // regionColor: -2 no bordering network, 0/1 sole color, -1 mixed.
  const regionOf = new Int32Array(n).fill(-1);
  const stack = new Int32Array(n);
  const regionColor: number[] = [];
  for (let s = 0; s < n; s++) {
    if (alive[s] || regionOf[s] !== -1) continue;
    const id = regionColor.length;
    let color = -2;
    let top = 0;
    stack[top++] = s;
    regionOf[s] = id;
    while (top > 0) {
      const u = stack[--top];
      for (let e = adjOff[u]; e < adjOff[u + 1]; e++) {
        const v = adj[e];
        if (alive[v]) {
          const cv = stones[v] as number;
          if (color === -2) color = cv;
          else if (color !== cv) color = -1;
        } else if (regionOf[v] === -1) {
          regionOf[v] = id;
          stack[top++] = v;
        }
      }
    }
    regionColor.push(color);
  }

  // ---- 4. Aggregate to players ---------------------------------------------
  const nodeOwner = new Int8Array(n).fill(-1);
  const aliveStone = new Uint8Array(n);
  const shores = [0, 0];
  const capes = [0, 0];
  const networks = [0, 0];
  let contestedShores = 0;

  for (let u = 0; u < n; u++) {
    let owner: number;
    if (alive[u]) {
      owner = stones[u] as number;
      aliveStone[u] = 1;
      if (parent[u] === u) networks[owner]++;
    } else {
      owner = regionColor[regionOf[u]];
    }
    if (owner === 0 || owner === 1) {
      nodeOwner[u] = owner;
      if (isShore[u]) {
        shores[owner]++;
        if (isCape[u]) capes[owner]++;
      }
    } else if (isShore[u]) {
      contestedShores++;
    }
  }
  // A network's root may be a non-shore stone; roots counted above only when the
  // root node itself is alive — which holds for every alive group since
  // aliveness is a per-group property. (parent[u] === u picks each group
  // exactly once.)

  const players = [0, 1].map((p) => {
    const capeBonus: 0 | 1 = capes[p] >= 3 ? 1 : 0;
    const award = 2 * (networks[1 - p] - networks[p]);
    return {
      shores: shores[p],
      capes: capes[p],
      networks: networks[p],
      capeBonus,
      award,
      total: shores[p] + capeBonus + award,
    };
  }) as [PlayerScore, PlayerScore];

  let leader: 0 | 1 | -1;
  if (players[0].total !== players[1].total) {
    leader = players[0].total > players[1].total ? 0 : 1;
  } else if (players[0].capes !== players[1].capes) {
    leader = players[0].capes > players[1].capes ? 0 : 1;
  } else {
    leader = -1;
  }

  return { players, nodeOwner, aliveStone, contestedShores, leader };
}

/**
 * Validate the invariants that make a completed supported game binary.
 *
 * Generic live-position scoring remains tie-capable; terminal consumers must
 * cross this boundary before producing a winner or model outcome.
 */
export function validateTerminalWinner(
  board: Board,
  stones: ArrayLike<number>,
): TerminalWinnerResult {
  if (!isSupportedRings(board.rings)) {
    throw new Error(`terminal board rings are unsupported: ${String(board.rings)}`);
  }
  if (stones.length !== board.n) {
    throw new Error(`terminal stones length must be ${board.n}, got ${stones.length}`);
  }
  for (let node = 0; node < board.n; node++) {
    if (stones[node] !== 0 && stones[node] !== 1) {
      throw new Error(`terminal board must be full; invalid stone at node ${node}`);
    }
  }

  const score = scorePosition(board, stones);
  if (score.contestedShores !== 0) {
    throw new Error(
      `terminal board must have zero contested shores, got ${score.contestedShores}`,
    );
  }

  const expectedTotal = 5 * board.rings + 1;
  const combinedTotal = score.players[0].total + score.players[1].total;
  if (combinedTotal !== expectedTotal) {
    throw new Error(
      `terminal score total must be ${expectedTotal}, got ${combinedTotal}`,
    );
  }

  const margin = score.players[0].total - score.players[1].total;
  if (margin === 0 || Math.abs(margin) % 2 !== 1) {
    throw new Error(`terminal score margin must be odd and nonzero, got ${margin}`);
  }
  const winner: 0 | 1 = margin > 0 ? 0 : 1;
  if (score.leader !== winner) {
    throw new Error('terminal score leader is inconsistent with the score margin');
  }
  return {
    score: score as ScoreResult & { leader: 0 | 1 },
    winner,
    margin,
  };
}
