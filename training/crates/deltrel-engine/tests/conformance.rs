#![allow(missing_docs)]

use std::collections::BTreeSet;
use std::sync::Arc;

use deltrel_engine::{
    Action, BITBOARD_WORDS, BitBoard, Board, D5Maps, GameState, Player, PlayerScore,
    RULES_HASH_VALUE, SUPPORTED_RINGS, ScoringScratch, Symmetry, rules_hash,
};

fn edge_count(rings: u8) -> usize {
    let rings = usize::from(rings);
    5 * rings * (rings + 1) / 2 + 5 * (rings * rings - 1) + 5
}

fn position(board: &Board, zero: &[&str], one: &[&str]) -> [BitBoard; 2] {
    let mut stones = [BitBoard::empty(); 2];
    for label in zero {
        stones[0].insert(board.parse_label(label).unwrap());
    }
    for label in one {
        stones[1].insert(board.parse_label(label).unwrap());
    }
    stones
}

#[test]
fn typescript_known_board_counts_and_topology_match() {
    assert_eq!(Board::new(4).unwrap().node_count(), 50);
    assert_eq!(Board::new(6).unwrap().node_count(), 105);
    assert_eq!(Board::new(8).unwrap().node_count(), 180);
    assert_eq!(Board::new(10).unwrap().node_count(), 275);
    assert_eq!(Board::new(6).unwrap().shore_count(), 30);
    assert_eq!(Board::new(8).unwrap().shore_count(), 40);
    assert_eq!(Board::new(10).unwrap().shore_count(), 50);

    for rings in SUPPORTED_RINGS {
        let board = Board::new(rings).unwrap();
        assert_eq!(
            board.node_count(),
            5 * u16::from(rings) * (u16::from(rings) + 1) / 2
        );
        assert_eq!(board.shore_mask().count(), 5 * u16::from(rings));
        assert_eq!(board.cape_mask().count(), 5);
        assert_eq!(board.edge_count(), edge_count(rings));
        assert_eq!(board.node_mask().words().len(), BITBOARD_WORDS);

        for node in 0..board.node_count() {
            let neighbors: BTreeSet<_> = board.neighbors(node).iter().copied().collect();
            assert_eq!(neighbors.len(), board.neighbors(node).len());
            assert!(!neighbors.contains(&node));
            assert!(neighbors.len() >= 3);
            for neighbor in neighbors {
                assert!(board.neighbors(neighbor).contains(&node));
            }
        }
    }
    for rings in [0, 1, 2, 3, 5, 7, 9, 11, 12, u8::MAX] {
        assert!(Board::new(rings).is_err());
    }
}

#[test]
fn typescript_known_labels_and_adjacencies_match() {
    let ten = Board::new(10).unwrap();
    assert_eq!(ten.label(ten.index(0, 10, 0).unwrap()), "HR");
    assert_eq!(ten.label(ten.index(4, 10, 0).unwrap()), "JF");
    assert_eq!(ten.label(ten.index(1, 3, 2).unwrap()), "U");
    for label in ["HR", "U", "AN", "HQ"] {
        assert_eq!(ten.label(ten.parse_label(label).unwrap()), label);
    }

    let board = Board::new(4).unwrap();
    let adjacent = |left: &str, right: &str| {
        let left = board.parse_label(left).unwrap();
        let right = board.parse_label(right).unwrap();
        board.neighbors(left).contains(&right)
    };
    for (left, right) in [
        ("AE", "AF"),
        ("AH", "AI"),
        ("AX", "AE"),
        ("AE", "P"),
        ("AH", "R"),
        ("AJ", "S"),
        ("G", "B"),
        ("AH", "S"),
        ("AL", "V"),
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
    ] {
        assert!(adjacent(left, right), "{left} must touch {right}");
    }
    for (left, right) in [("AE", "AG"), ("AE", "AI"), ("A", "P")] {
        assert!(!adjacent(left, right), "{left} must not touch {right}");
    }
}

#[test]
fn double_deltrel_atomic_placements_and_undo_match_oracle() {
    let board = Arc::new(Board::new(4).unwrap());
    let mut state = GameState::new(Arc::clone(&board));
    assert_eq!(state.to_move(), Player::Zero);
    assert_eq!(state.moves_left(), 1);

    state.apply(Action::Place(0)).unwrap();
    assert_eq!(state.to_move(), Player::One);
    assert_eq!(state.moves_left(), 2);
    state.apply(Action::Place(1)).unwrap();
    assert_eq!(state.to_move(), Player::One);
    assert_eq!(state.moves_left(), 1);

    let key_before_second_placement = state.key();
    let (_, undo) = state.apply_reversible(Action::Place(2)).unwrap();
    assert_eq!(state.to_move(), Player::Zero);
    assert_eq!(state.moves_left(), 2);
    state.undo(undo);
    assert_eq!(state.key(), key_before_second_placement);
    assert!(!state.is_terminal());
    assert_eq!(
        state.legal_actions().to_vec(),
        (2..board.node_count())
            .map(Action::Place)
            .collect::<Vec<_>>()
    );

    let mut full = GameState::new(board);
    for node in 0..full.board().node_count() {
        full.apply(Action::Place(node)).unwrap();
    }
    assert!(full.is_terminal());
    assert_eq!(full.stones_placed(), full.board().node_count());
    let transformed_full = D5Maps::new(full.board()).state(Symmetry::ALL[6], &full);
    assert!(transformed_full.is_terminal());
    assert_eq!(transformed_full.moves_left(), full.moves_left());
}

#[test]
fn pair_order_has_the_same_semantic_key_and_hash() {
    let board = Arc::new(Board::new(4).unwrap());
    let mut left = GameState::new(Arc::clone(&board));
    left.apply(Action::Place(0)).unwrap();
    let mut right = left.clone();

    left.apply(Action::Place(1)).unwrap();
    left.apply(Action::Place(2)).unwrap();
    right.apply(Action::Place(2)).unwrap();
    right.apply(Action::Place(1)).unwrap();

    assert_eq!(left.key(), right.key());
    assert_eq!(left.hash64(), right.hash64());
    assert_eq!(left.to_move(), Player::Zero);
}

#[test]
fn d5_maps_are_deterministic_bijections_and_graph_automorphisms() {
    for rings in SUPPORTED_RINGS {
        let board = Board::new(rings).unwrap();
        let first = D5Maps::new(&board);
        let second = D5Maps::new(&board);
        for symmetry in Symmetry::ALL {
            assert_eq!(first.map(symmetry), second.map(symmetry));
            let mapped: BTreeSet<_> = first.map(symmetry).iter().copied().collect();
            assert_eq!(mapped.len(), usize::from(board.node_count()));

            for node in 0..board.node_count() {
                let transformed = first.node(symmetry, node);
                assert_eq!(
                    first.node(symmetry.inverse(), transformed),
                    node,
                    "inverse failed for {rings} rings and symmetry {}",
                    symmetry.index()
                );
                for &neighbor in board.neighbors(node) {
                    let mapped_neighbor = first.node(symmetry, neighbor);
                    assert!(board.neighbors(transformed).contains(&mapped_neighbor));
                }
            }
        }
    }
}

#[test]
fn typescript_scoring_fixtures_match_exactly() {
    let board = Board::new(4).unwrap();
    let mut scratch = ScoringScratch::default();

    let fixture_a = position(
        &board,
        &["B", "H", "S", "AI", "D", "L", "Y", "AQ", "AR"],
        &[
            "A", "F", "P", "AE", "AF", "C", "J", "V", "AM", "AV", "AW", "AK",
        ],
    );
    let score = scratch.score(&board, fixture_a);
    assert_eq!(
        score.players[0],
        PlayerScore {
            shores: 3,
            capes: 2,
            networks: 1,
            cape_bonus: 0,
            award: 2,
            total: 5,
        }
    );
    assert_eq!(
        score.players[1],
        PlayerScore {
            shores: 5,
            capes: 2,
            networks: 2,
            cape_bonus: 0,
            award: -2,
            total: 3,
        }
    );
    assert_eq!(score.contested_shores, 12);
    let dead = board.parse_label("AK").unwrap();
    assert!(!score.alive_stones.contains(dead));
    assert_eq!(score.owner(dead), None);

    let fixture_b = position(&board, &["AH", "AO", "AP"], &["AG", "R", "S", "AI"]);
    let score = scratch.score(&board, fixture_b);
    assert_eq!(
        score.players[1],
        PlayerScore {
            shores: 3,
            capes: 1,
            networks: 1,
            cape_bonus: 0,
            award: 0,
            total: 3,
        }
    );
    assert_eq!(
        score.owner(board.parse_label("AH").unwrap()),
        Some(Player::One)
    );

    let fixture_c = position(
        &board,
        &["AQ", "AR", "AS", "AT", "AU", "AV"],
        &["AE", "AF", "AI", "AJ", "AM", "AN"],
    );
    let score = scratch.score(&board, fixture_c);
    assert_eq!(score.players[0].total, 10);
    assert_eq!(score.players[0].networks, 1);
    assert_eq!(score.players[1].total, 3);
    assert_eq!(score.players[1].networks, 3);
    assert_eq!(score.players[1].cape_bonus, 1);

    let fixture_d = position(&board, &["AE", "AF"], &["AJ", "AK"]);
    let score = scratch.score(&board, fixture_d);
    assert_eq!(score.players[0].total, 2);
    assert_eq!(score.players[1].total, 2);
    assert_eq!(score.leader, Some(Player::Zero));

    let fixture_e = position(
        &board,
        &["A", "F", "P", "AE", "D", "L", "Y", "AQ"],
        &["B", "H", "S", "AI", "C", "J", "V", "AM"],
    );
    let score = scratch.score(&board, fixture_e);
    assert_eq!(score.players[0].networks, 1);
    assert_eq!(score.players[1].networks, 1);
}

#[test]
fn scoring_is_d5_invariant_and_full_board_identity_holds() {
    let board = Arc::new(Board::new(6).unwrap());
    let maps = D5Maps::new(&board);
    let mut state = GameState::new(Arc::clone(&board));
    for node in [0, 5, 11, 30, 75, 76, 77, 80, 90, 100] {
        state.apply(Action::Place(node)).unwrap();
    }
    let mut scratch = ScoringScratch::default();
    let original = scratch.score_state(&state);
    for symmetry in Symmetry::ALL {
        let transformed = maps.state(symmetry, &state);
        let score = scratch.score_state(&transformed);
        assert_eq!(score.players, original.players);
        assert_eq!(score.contested_shores, original.contested_shores);
    }

    let mut seed = 0x0057_17a5_u64;
    for rings in SUPPORTED_RINGS {
        let board = Board::new(rings).unwrap();
        for _ in 0..40 {
            let mut stones = [BitBoard::empty(); 2];
            for node in 0..board.node_count() {
                seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15).rotate_left(17)
                    ^ 0xbf58_476d_1ce4_e5b9;
                stones[usize::from((seed & 1) != 0)].insert(node);
            }
            let score = scratch.score(&board, stones);
            let total = score.players[0].total + score.players[1].total;
            assert_eq!(
                total,
                i16::try_from(board.shore_count() - score.contested_shores).unwrap()
                    + score.players[0].cape_bonus
                    + score.players[1].cape_bonus
            );
            assert!(score.players[0].cape_bonus + score.players[1].cape_bonus <= 1);
        }
    }

    assert_eq!(rules_hash(), RULES_HASH_VALUE);
    assert_eq!(rules_hash(), 0x46e4_fbcf_f4e1_7fd3);
}
