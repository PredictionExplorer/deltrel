//! Authoritative native rules for the Deltrel family.
//!
//! One rules contract covers four variants of one game: standard Double Deltrel
//! (one opening stone, then two placements per turn), classic Deltrel (one
//! placement every turn), handicap openings (player 0 places `k` stones
//! consecutively, `k` in `1..=9`), and the pie rule (player 1 may swap once
//! after the opening turn). Placements are atomic actions; the swap is a
//! separate action that is never a policy output. Play ends exactly when the
//! board is full.

mod bitboard;
mod board;
mod endgame;
mod game;
mod scoring;
mod symmetry;

pub use bitboard::{BITBOARD_WORDS, BitBoard, BitIter};
pub use board::{
    BOARD_NOTATION_SCHEMA, BOARD_NOTATION_VERSION, Board, BoardError, coordinate_label,
};
pub use endgame::{ExactEndgame, solve_exact_endgame};
pub use game::{
    Action, GameError, GameState, LegalActions, MAX_HANDICAP, MAX_TURN_PLACEMENTS, Mode, Player,
    StateKey, StateParts, Transition, Undo, Variant,
};
pub use scoring::{
    CompletionBounds, CompletionScenario, PlayerScore, ScoreResult, ScoringScratch,
    score_completion_bounds, score_state, terminal_value,
};
pub use symmetry::{D5_ORDER, D5Maps, Symmetry};

/// Dense node id.
pub type NodeId = u16;

/// Smallest supported board.
pub const MIN_RINGS: u8 = 4;
/// Largest supported board.
pub const MAX_RINGS: u8 = 10;
/// Complete set of supported board sizes.
pub const SUPPORTED_RINGS: [u8; 4] = [4, 6, 8, 10];
/// Maximum playable nodes: `5 * 10 * 11 / 2`.
pub const MAX_NODES: usize = 275;
/// Semantic contract version embedded into generated training data.
pub const RULES_VERSION: u32 = 3;
/// Schema of the finalized cross-language rules contract.
pub const RULES_SCHEMA: &str = "deltrel.rules.v3";
/// Tagged FNV-1a hash of the finalized canonical rules contract.
pub const RULES_HASH: &str = "fnv1a64:46e4fbcff4e17fd3";
/// Raw finalized FNV-1a rules hash.
pub const RULES_HASH_VALUE: u64 = 0x46e4_fbcf_f4e1_7fd3;
/// Schema of the generated conformance vectors.
pub const CONFORMANCE_SCHEMA: &str = "deltrel.conformance.v3";
/// Schema of the external model feature contract.
pub const FEATURE_SCHEMA: &str = "deltrel.model-features.external.v3";
/// Schema of the native nodes-only action layout.
pub const ACTION_LAYOUT_SCHEMA: &str = "deltrel.action-layout.nodes-only.v1";

/// Exact frozen canonical bytes of the rules-v3 contract. Its label clause is
/// historical presentation metadata; BOARD_NOTATION_SCHEMA defines today's
/// notation independently, without invalidating models or changing game rules.
/// The web client
/// (`src/lib/deltrel/rules.ts`) and the Python mirror (`deltreltrain/contracts.py`)
/// carry the same bytes so every runtime derives the same fingerprint.
pub const RULES_CANONICAL: &str = concat!(
    "double-deltrel/rules-v3;",
    "rings=even:{4,6,8,10};",
    "node-count=5*r*(r+1)/2;",
    "node-order=x:1..r,s:0..4,y:0..x-1;",
    "node-id=5*x*(x-1)/2+s*x+y;",
    "sector-order=0,1,2,3,4:clockwise;",
    "sector-arithmetic=mod5;",
    "label=bijective-base26-uppercase(node-id+1);",
    "shore=x==r;",
    "cape=x==r&&y==0;",
    "edges=node-order:cycle,radial,diagonal,corner-cross;then-ring1-k5-lexicographic;",
    "edge-dedupe=first-undirected-insertion;",
    "csr-neighbor-order=edge-insertion-order;",
    "cycle=(s,x,y)-(y<x-1?(s,x,y+1):(s+1,x,0));",
    "radial=x>=2&&y<=x-2?(s,x,y)-(s,x-1,y);",
    "diagonal=x>=2&&y>=1?(s,x,y)-(s,x-1,y-1);",
    "corner-cross=x>=2&&y==x-1?(s,x,y)-(s+1,x-1,0);",
    "bridge=K5((s,1,0),s=0..4);",
    "modes={classic:turn-size-1,double:opening-1-then-2};",
    "handicap=k-consecutive-opening-placements-by-player0,k-in-1..9,k=1-is-standard;",
    "pie=optional:after-first-turn-player1-may-swap,recolor-opening-stones-to-player1,",
    "player0-moves-next-with-full-turn,swap-unavailable-after-any-placement;",
    "handicap-excludes-pie;",
    "variant-in-semantic-key=mode,handicap,pie;",
    "history-in-semantic-key=currentTurn,previousTurn,ownPreviousTurn,handicapStones;",
    "actions=atomic-place|swap;",
    "action-wire=place(node)->node,swap->node-count;",
    "legal-order=empty-node-id-ascending;",
    "native-action-layout=node-u-at-u;",
    "terminal=full;",
    "full-terminal=decrement-movesLeft,retain-actor-and-turnCount,no-endTurn,",
    "movesLeft-below-turn-size,midTurn=(movesLeft>0),lastMove=final-node,",
    "currentTurnMoves=final-partial-turn;",
    "pair-semantic=AB==BA-excluding-lastMove;",
    "stones=empty:-1,players:0,1;",
    "network=same-color-connected-group-with-at-least-two-directly-occupied-shores;",
    "territory=after-dead-removal,maximal-nonalive-component-owned-iff-adjacent-",
    "alive-color-set-is-exactly-one-player;",
    "score=shores+cape-bonus+2*(opponent-networks-own-networks);",
    "tiebreak=capes;",
    "terminal-value=toMove-perspective:win=1,loss=-1,tie=invalid;",
    "outcome-class=loss:0,win:1;",
    "score-margin=toMove-total-opponent-total;",
    "terminal-legal-actions=empty;",
    "d5-order=r0,r1,r2,r3,r4,f0,f1,f2,f3,f4;",
    "d5-coordinate=t=s*x+y(mod5*x);",
    "d5-rk=t+k*x(mod5*x);",
    "d5-fk=k*x-t(mod5*x);",
    "d5-action=map-place-node,swap-fixed",
);

/// Stable hash of the complete rules contract.
#[must_use]
pub const fn rules_hash() -> u64 {
    RULES_HASH_VALUE
}

/// FNV-1a 64-bit hash of a byte string, matching the web and Python mirrors.
#[must_use]
pub const fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut index = 0;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        index += 1;
    }
    hash
}

const _: () = assert!(
    fnv1a64(RULES_CANONICAL.as_bytes()) == RULES_HASH_VALUE,
    "RULES_HASH_VALUE must equal the FNV-1a hash of RULES_CANONICAL"
);
