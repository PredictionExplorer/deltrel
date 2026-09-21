/**
 * Versioned Deltrel contract shared by TypeScript and parity ports.
 *
 * The hash covers DELTREL_RULES_CANONICAL, not implementation source. A
 * consumer can therefore reject fixtures or saved protocol data produced for
 * different semantics without depending on TypeScript formatting or builds.
 *
 * Rules v3 covers the whole variant family: standard Double Deltrel, classic
 * one-stone turns, handicap openings of one to nine stones, and the pie rule.
 */

export const DELTREL_RULES_VERSION = 3 as const;
export const DELTREL_RULES_HASH_ALGORITHM = 'fnv1a64' as const;
export const DELTREL_RULES_SCHEMA_ID = 'deltrel.rules.v3' as const;
export const DELTREL_CONFORMANCE_SCHEMA_ID =
  'deltrel.conformance.v3' as const;
/**
 * Model feature definitions live outside this TypeScript rules package.
 * Exported identifiers let training/inference artifacts pin their own schema
 * without folding model-only changes into the gameplay rules hash.
 */
export const DELTREL_FEATURE_SCHEMA_ID =
  'deltrel.model-features.external.v3' as const;
export const DELTREL_ACTION_LAYOUT_SCHEMA_ID =
  'deltrel.action-layout.nodes-only.v1' as const;

/** Largest handicap: consecutive opening placements by the first player. */
export const DELTREL_MAX_HANDICAP = 9 as const;

export const DELTREL_RULES_CONTRACT = {
  schema: DELTREL_RULES_SCHEMA_ID,
  version: DELTREL_RULES_VERSION,
  variant: 'double-deltrel',
  board: {
    supportedRings: [4, 6, 8, 10],
    minimumRings: 4,
    maximumRings: 10,
    nodeCount: '5*rings*(rings+1)/2',
    nodeOrder: 'ring-major, then sector-major, then position-major',
    ringStart: '5*x*(x-1)/2',
    nodeId: '5*x*(x-1)/2 + sector*x + position',
    sectorOrder: [0, 1, 2, 3, 4],
    sectorArithmetic: 'modulo 5',
    ringAddress: {
      sector: '0..4 clockwise',
      ring: '1..rings',
      position: '0..ring-1 clockwise from the sector arm',
    },
    perimeter: 'ring == rings',
    cape: 'ring == rings && position == 0',
    adjacency: [
      'cyclic successor on each ring',
      'radial edge to (sector,ring-1,position) when ring >= 2 and position <= ring-2',
      'diagonal edge to (sector,ring-1,position-1) when ring >= 2 and position >= 1',
      'corner-cross edge to (sector+1,ring-1,0) when ring >= 2 and position == ring-1',
      'complete graph K5 over the five ring-1 nodes',
    ],
    edgeOrder:
      'iterate nodes in node-id order; attempt cycle, radial, diagonal, corner-cross edges in that order; then ring-1 K5 pairs in lexicographic arm order; keep first undirected insertion',
    csrOrder: 'neighbors retain undirected edge insertion order',
    labels: 'bijective base-26 uppercase letters of node id + 1: A..Z, AA..AZ, BA..',
  },
  scoring: {
    emptyValue: -1,
    colors: [0, 1],
    network:
      'a same-color connected group is alive iff it directly occupies at least two perimeter nodes',
    territory:
      'after dead groups are removed, a maximal non-alive region belongs to a player iff every adjacent alive network has that color and at least one alive network is adjacent',
    shores: 'owned perimeter nodes, whether occupied by an alive network or territory',
    capes: 'owned perimeter nodes with position 0',
    capeBonus: 'one point when a player owns at least three capes',
    award: '2 * (opponent alive-network count - own alive-network count)',
    total: 'shores + capeBonus + award',
    leader: 'higher total, then higher cape count, otherwise tie',
    terminalValue:
      'from terminal toMove perspective: winner 1, loser -1; a terminal tie is invalid',
    outcomeClass: 'loss 0, win 1',
    scoreMargin: 'toMove total - opponent total',
  },
  game: {
    modes: {
      classic: 'one placement every turn',
      double: 'one opening placement, then two placements per turn',
    },
    handicap:
      'the first player places k stones consecutively before the second player moves; k in 1..9; k = 1 is the standard game',
    pieRule:
      'optional; immediately after the first turn the second player may swap: the opening stone is recolored to player 1 and player 0 moves next with a full turn; unavailable after any placement',
    handicapExcludesPie: true,
    variantInSemanticKey: ['mode', 'handicap', 'pie'],
    historyInSemanticKey: [
      'currentTurn',
      'previousTurn',
      'ownPreviousTurn',
      'handicapStones',
    ],
    actionTypes: ['place', 'swap'],
    actionWireEncoding: {
      place: 'dense node id 0..n-1',
      swap: 'node count n, one past the last node',
    },
    legalActionOrder: 'legal placements by ascending node id',
    nativeActionLayout: 'node u at slot u; the swap has no slot',
    termination: 'board full',
    fullBoardResidual:
      'the final placement decrements movesLeft and terminates before endTurn; actor and turnCount are retained; movesLeft is below the turn size; midTurn is movesLeft > 0; lastMove is the final node; currentTurnMoves retains the final partial turn',
    terminalLegalActions: 'none',
    placement: 'only an empty in-range node is legal',
    replay: 'apply the ordered action log to a fresh initial state',
    pairEquivalence:
      'the two placements AB and BA in one complete nonterminal turn have the same semantic state; lastMove is presentation metadata',
  },
  symmetry: {
    group: 'D5',
    order: ['r0', 'r1', 'r2', 'r3', 'r4', 'f0', 'f1', 'f2', 'f3', 'f4'],
    ringCoordinate: 't = sector*ring + position modulo 5*ring',
    rotation: 'r(k): t -> t + k*ring for k in 0..4',
    reflection: 'f(k): t -> k*ring - t for k in 0..4',
    action: 'place transforms by the node map; the swap is fixed',
  },
} as const;

/**
 * Compact ASCII wire contract. Rust and Python consumers mirror these exact
 * bytes so every runtime derives the same unsigned 64-bit fingerprint.
 */
export const DELTREL_RULES_CANONICAL = [
  'double-deltrel/rules-v3;',
  'rings=even:{4,6,8,10};',
  'node-count=5*r*(r+1)/2;',
  'node-order=x:1..r,s:0..4,y:0..x-1;',
  'node-id=5*x*(x-1)/2+s*x+y;',
  'sector-order=0,1,2,3,4:clockwise;',
  'sector-arithmetic=mod5;',
  'label=bijective-base26-uppercase(node-id+1);',
  'shore=x==r;',
  'cape=x==r&&y==0;',
  'edges=node-order:cycle,radial,diagonal,corner-cross;then-ring1-k5-lexicographic;',
  'edge-dedupe=first-undirected-insertion;',
  'csr-neighbor-order=edge-insertion-order;',
  'cycle=(s,x,y)-(y<x-1?(s,x,y+1):(s+1,x,0));',
  'radial=x>=2&&y<=x-2?(s,x,y)-(s,x-1,y);',
  'diagonal=x>=2&&y>=1?(s,x,y)-(s,x-1,y-1);',
  'corner-cross=x>=2&&y==x-1?(s,x,y)-(s+1,x-1,0);',
  'bridge=K5((s,1,0),s=0..4);',
  'modes={classic:turn-size-1,double:opening-1-then-2};',
  'handicap=k-consecutive-opening-placements-by-player0,k-in-1..9,k=1-is-standard;',
  'pie=optional:after-first-turn-player1-may-swap,recolor-opening-stones-to-player1,player0-moves-next-with-full-turn,swap-unavailable-after-any-placement;',
  'handicap-excludes-pie;',
  'variant-in-semantic-key=mode,handicap,pie;',
  'history-in-semantic-key=currentTurn,previousTurn,ownPreviousTurn,handicapStones;',
  'actions=atomic-place|swap;',
  'action-wire=place(node)->node,swap->node-count;',
  'legal-order=empty-node-id-ascending;',
  'native-action-layout=node-u-at-u;',
  'terminal=full;',
  'full-terminal=decrement-movesLeft,retain-actor-and-turnCount,no-endTurn,movesLeft-below-turn-size,midTurn=(movesLeft>0),lastMove=final-node,currentTurnMoves=final-partial-turn;',
  'pair-semantic=AB==BA-excluding-lastMove;',
  'stones=empty:-1,players:0,1;',
  'network=same-color-connected-group-with-at-least-two-directly-occupied-shores;',
  'territory=after-dead-removal,maximal-nonalive-component-owned-iff-adjacent-alive-color-set-is-exactly-one-player;',
  'score=shores+cape-bonus+2*(opponent-networks-own-networks);',
  'tiebreak=capes;',
  'terminal-value=toMove-perspective:win=1,loss=-1,tie=invalid;',
  'outcome-class=loss:0,win:1;',
  'score-margin=toMove-total-opponent-total;',
  'terminal-legal-actions=empty;',
  'd5-order=r0,r1,r2,r3,r4,f0,f1,f2,f3,f4;',
  'd5-coordinate=t=s*x+y(mod5*x);',
  'd5-rk=t+k*x(mod5*x);',
  'd5-fk=k*x-t(mod5*x);',
  'd5-action=map-place-node,swap-fixed',
].join('');

/** Compute the unsigned 64-bit FNV-1a hash used by the parity contract. */
export function fnv1a64(value: string): string {
  let hash = BigInt('0xcbf29ce484222325');
  const prime = BigInt('0x100000001b3');
  const mask = BigInt('0xffffffffffffffff');
  for (const byte of new TextEncoder().encode(value)) {
    hash = ((hash ^ BigInt(byte)) * prime) & mask;
  }
  return hash.toString(16).padStart(16, '0');
}

/** Stable wire fingerprint for the complete gameplay contract above. */
export const DELTREL_RULES_HASH =
  'fnv1a64:46e4fbcff4e17fd3' as const;
