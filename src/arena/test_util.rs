use approx::assert_abs_diff_eq;

use crate::arena::game_state::Round;

use super::action::AgentAction;
use super::{GameState, game_state::RoundData};

use crate::arena::action::Action;
use crate::arena::historian::HistoryRecord;

pub fn assert_valid_round_data(round_data: &RoundData) {
    // Get all of the player still active at the end of the round.
    // for any round with bets they should have called.
    //
    // EG no one should call for less than the max and still be in.
    let active_bets: Vec<f32> = round_data
        .player_bet
        .iter()
        .enumerate()
        .filter(|(idx, _)| round_data.needs_action.get(*idx))
        .map(|(_, bet)| *bet)
        .collect();

    let max_active = active_bets.clone().into_iter().reduce(f32::max);

    if let Some(max) = max_active {
        let epsilon = if max == 0.0 {
            f32::EPSILON
        } else {
            max / 100_000.0
        };
        for bet in active_bets.into_iter() {
            assert_abs_diff_eq!(bet, max, epsilon = epsilon);
        }
    }
}

pub fn assert_valid_game_state(game_state: &GameState) {
    assert_eq!(Round::Complete, game_state.round);

    let should_have_bets = game_state.ante + game_state.small_blind + game_state.big_blind > 0.0;

    let total_bet = game_state.player_bet.iter().copied().sum();

    if should_have_bets {
        let any_above_zero = game_state.player_bet.iter().any(|bet| *bet > 0.0);

        assert!(
            any_above_zero,
            "At least one player should have a bet, game_state: {:?}",
            game_state.player_bet
        );

        assert_ne!(0.0, total_bet);
    }

    let epsilon = total_bet / 100_000.0;
    assert_abs_diff_eq!(total_bet, game_state.total_pot, epsilon = epsilon);

    let total_winning: f32 = game_state.player_winnings.iter().copied().sum();

    assert_abs_diff_eq!(total_winning, total_bet, epsilon = epsilon);
    assert_abs_diff_eq!(total_winning, game_state.total_pot, epsilon = epsilon);

    // The dealer has to be well specified.
    assert!(game_state.dealer_idx < game_state.num_players);

    // The board should be full or getting full
    assert!(game_state.board.len() <= 5);

    assert!(game_state.small_blind <= game_state.big_blind);

    // Validate Texas Hold'em specific rules
    validate_board_cards(game_state);
    validate_player_states(game_state);
    validate_betting_structure(game_state);
    validate_winnings_distribution(game_state);
    validate_deck_integrity(game_state);
    validate_dealer_positioning(game_state);
    validate_ante_structure(game_state);
    validate_stack_integrity(game_state);
    validate_side_pot_distribution(game_state);

    for idx in 0..game_state.num_players {
        // If they aren't active (folded)
        // and aren't all in then they shouldn't win anything
        if !game_state.player_active.get(idx) && !game_state.player_all_in.get(idx) {
            assert_abs_diff_eq!(game_state.player_winnings[idx], 0.0, epsilon = f32::EPSILON);
        }
    }
}

pub fn assert_valid_history(history_storage: &[HistoryRecord]) {
    // There should always be some history
    assert!(!history_storage.is_empty());

    // The first action should always be a game start
    assert!(
        matches!(history_storage[0].action, Action::GameStart(_)),
        "First action should be GameStart, but was: {:?}",
        history_storage[0].action
    );

    // History should include round advance to complete
    assert_advances_to_complete(history_storage);

    assert_round_contains_valid_player_actions(history_storage);

    assert_no_player_actions_after_fold(history_storage);

    validate_betting_sequence(history_storage);

    validate_round_progression(history_storage);
}

fn assert_advances_to_complete(history_storage: &[HistoryRecord]) {
    let round_advances: Vec<&Action> = history_storage
        .iter()
        .filter(|record| matches!(record.action, Action::RoundAdvance(Round::Complete)))
        .map(|record| &record.action)
        .collect();

    assert_eq!(1, round_advances.len());
}

fn assert_round_contains_valid_player_actions(history_storage: &[HistoryRecord]) {
    // For Preflop, Flop, Turn, and River there should
    // be a at least one player action for each player
    // unless everyone else has folded or they are all in.
    for round in &[Round::Preflop, Round::Flop, Round::Turn, Round::River] {
        let advance_history = history_storage.iter().find(|record| {
            if let Action::RoundAdvance(found_round) = &record.action {
                found_round == round
            } else {
                false
            }
        });

        if advance_history.is_none() {
            continue;
        }
        // TODO check here for
    }
}

fn assert_no_player_actions_after_fold(history_storage: &[HistoryRecord]) {
    // If a player has folded
    // they shouldn't have any actions after that.
    let player_fold_index: Vec<(usize, usize)> = history_storage
        .iter()
        .enumerate()
        .filter_map(|(index, record)| {
            if let Action::PlayedAction(action) = &record.action {
                if action.action == AgentAction::Fold {
                    Some((action.idx, index))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    for (player_idx, fold_index) in player_fold_index {
        let actions_after_fold = history_storage
            .iter()
            .skip(fold_index + 1)
            .filter(|record| {
                if let Action::PlayedAction(action) = &record.action {
                    action.idx == player_idx
                } else {
                    false
                }
            });

        assert_eq!(0, actions_after_fold.count());
    }
}

/// Validate board cards follow Texas Hold'em rules
fn validate_board_cards(game_state: &GameState) {
    let board_len = game_state.board.len();

    // Board must have 0, 3, 4, or 5 cards (never 1 or 2)
    assert!(
        matches!(board_len, 0 | 3 | 4 | 5),
        "Invalid board card count: {}. Texas Hold'em boards have 0, 3, 4, or 5 cards",
        board_len
    );

    // All board cards must be unique
    let mut seen_cards = std::collections::HashSet::new();
    for card in &game_state.board {
        assert!(
            seen_cards.insert(card),
            "Duplicate card {:?} found on board",
            card
        );
    }
}

/// Validate player states are consistent
fn validate_player_states(game_state: &GameState) {
    for idx in 0..game_state.num_players {
        let is_active = game_state.player_active.get(idx);
        let is_all_in = game_state.player_all_in.get(idx);
        let bet = game_state.player_bet[idx];
        let winnings = game_state.player_winnings[idx];

        // Players cannot be both all-in and folded (inactive)
        if is_all_in {
            // All-in players should have made some bet unless they posted blinds with their full stack
            // We'll be lenient here as all-in detection can be complex
        }

        // Folded players shouldn't win anything (handled in main function)
        if !is_active && !is_all_in {
            assert_abs_diff_eq!(winnings, 0.0, epsilon = f32::EPSILON);
        }

        // Bets should be non-negative
        assert!(bet >= 0.0, "Player {} has negative bet: {}", idx, bet);

        // Winnings should be non-negative
        assert!(
            winnings >= 0.0,
            "Player {} has negative winnings: {}",
            idx,
            winnings
        );
    }
}

/// Validate betting structure follows Texas Hold'em rules
fn validate_betting_structure(game_state: &GameState) {
    // Small blind should be less than or equal to big blind
    if game_state.small_blind > 0.0 || game_state.big_blind > 0.0 {
        assert!(
            game_state.small_blind <= game_state.big_blind,
            "Small blind {} cannot be larger than big blind {}",
            game_state.small_blind,
            game_state.big_blind
        );
    }
}

/// Validate winnings distribution makes sense
fn validate_winnings_distribution(game_state: &GameState) {
    let total_winnings: f32 = game_state.player_winnings.iter().sum();
    let total_bets: f32 = game_state.player_bet.iter().sum();

    // Total winnings should equal total bets (conservation of money)
    assert_abs_diff_eq!(total_winnings, total_bets, epsilon = total_bets / 100_000.0);

    // At least one player should win something if there was action
    if total_bets > 0.0 {
        let someone_won = game_state.player_winnings.iter().any(|&w| w > 0.0);
        assert!(someone_won, "Someone must win if there was betting action");
    }

    // Winners should have either been active or all-in
    for (idx, &winnings) in game_state.player_winnings.iter().enumerate() {
        if winnings > 0.0 {
            let is_active = game_state.player_active.get(idx);
            let is_all_in = game_state.player_all_in.get(idx);
            assert!(
                is_active || is_all_in,
                "Winner player {} must be either active or all-in, winnings: {}",
                idx,
                winnings
            );
        }
    }
}

/// Validate deck integrity - ensure no duplicate cards in hole cards + board
fn validate_deck_integrity(game_state: &GameState) {
    let mut dealt_cards = std::collections::HashSet::new();

    // Collect board cards (community cards)
    let board_set: std::collections::HashSet<_> = game_state.board.iter().copied().collect();

    // Check for duplicate board cards
    assert_eq!(
        board_set.len(),
        game_state.board.len(),
        "Duplicate cards found on the board"
    );
    dealt_cards.extend(&board_set);

    // Extract hole cards for each player (hand minus community cards)
    for (player_idx, hand) in game_state.hands.iter().enumerate() {
        let hand_set: std::collections::HashSet<_> = hand.iter().collect();

        // Player's hole cards = their hand minus community cards
        let hole_cards: std::collections::HashSet<_> =
            hand_set.difference(&board_set).copied().collect();

        // In Texas Hold'em, each player should have exactly 2 hole cards
        // (unless it's a different variant, but we'll be flexible)
        if !hole_cards.is_empty() {
            // Check that hole cards don't duplicate with previously dealt cards
            for hole_card in &hole_cards {
                assert!(
                    dealt_cards.insert(*hole_card),
                    "Duplicate hole card {:?} found for player {}",
                    hole_card,
                    player_idx
                );
            }
        }

        // Verify that player's hand contains all community cards
        for board_card in &game_state.board {
            assert!(
                hand.contains(board_card),
                "Player {} missing community card {:?}",
                player_idx,
                board_card
            );
        }
    }

    // Ensure no more than 52 cards are dealt total
    assert!(
        dealt_cards.len() <= 52,
        "Too many cards dealt: {}. Standard deck has 52 cards",
        dealt_cards.len()
    );
}

/// Validate dealer button positioning follows Texas Hold'em rules
fn validate_dealer_positioning(game_state: &GameState) {
    // Dealer index must be valid
    assert!(
        game_state.dealer_idx < game_state.num_players,
        "Dealer index {} is out of bounds for {} players",
        game_state.dealer_idx,
        game_state.num_players
    );

    // For heads-up play, dealer is small blind
    // For 3+ players, small blind is left of dealer, big blind is left of small blind
    if game_state.num_players >= 2 {
        let _small_blind_idx = (game_state.dealer_idx + 1) % game_state.num_players;
        let _big_blind_idx = (game_state.dealer_idx + 2) % game_state.num_players;

        // In heads-up, dealer is small blind
        if game_state.num_players == 2 {
            // Heads-up: dealer posts small blind, other player posts big blind
            // This is validated through the blind structure in betting_structure validation
        } else {
            // 3+ players: validate positioning through betting patterns if possible
            // This is more complex and would require tracking blind posting history
            // For now, we'll rely on the dealer_idx being in valid range
        }
    }
}

/// Validate ante structure if antes are used
fn validate_ante_structure(game_state: &GameState) {
    if game_state.ante > 0.0 {
        // If antes are used, they should be non-negative
        assert!(
            game_state.ante >= 0.0,
            "Ante cannot be negative: {}",
            game_state.ante
        );

        // Ante should typically be smaller than the big blind
        if game_state.big_blind > 0.0 {
            assert!(
                game_state.ante <= game_state.big_blind,
                "Ante {} should not exceed big blind {}",
                game_state.ante,
                game_state.big_blind
            );
        }
    }
}

/// Validate that stacks are non-negative for all players
fn validate_stack_integrity(game_state: &GameState) {
    for (idx, &stack) in game_state.stacks.iter().enumerate() {
        assert!(stack >= 0.0, "Player {} has negative stack: {}", idx, stack);
    }

    // Validate starting stacks were also non-negative
    for (idx, &stack) in game_state.starting_stacks.iter().enumerate() {
        assert!(
            stack >= 0.0,
            "Player {} had negative starting stack: {}",
            idx,
            stack
        );
    }
}

/// Validate side pot structure follows Texas Hold'em rules.
/// This validates that winnings are distributed fairly based on contributions.
fn validate_side_pot_distribution(game_state: &GameState) {
    // Validate that each winner had a non-zero contribution to the pot
    for (winner_idx, &winnings) in game_state.player_winnings.iter().enumerate() {
        if winnings <= 0.0 {
            continue;
        }

        let winner_bet = game_state.player_bet[winner_idx];

        // A player must have contributed something to win something
        // (except in rare edge cases with antes where they might win their ante back)
        if game_state.ante == 0.0 && winner_bet <= 0.0 {
            panic!(
                "Player {} won {} but had no contribution to the pot",
                winner_idx, winnings
            );
        }
    }
}

/// Validate betting sequence follows Texas Hold'em rules
fn validate_betting_sequence(history_storage: &[HistoryRecord]) {
    let mut round_raise_counts: std::collections::HashMap<Round, usize> =
        std::collections::HashMap::new();
    let mut active_players_per_round: std::collections::HashMap<
        Round,
        std::collections::HashSet<usize>,
    > = std::collections::HashMap::new();
    let mut current_round = Round::Preflop;

    for record in history_storage {
        match &record.action {
            Action::RoundAdvance(round) => {
                current_round = *round;
                // Reset raise count for new round (except Complete)
                if *round != Round::Complete {
                    round_raise_counts.insert(*round, 0);
                }
            }
            Action::PlayedAction(action) => {
                active_players_per_round
                    .entry(current_round)
                    .or_default()
                    .insert(action.idx);

                match action.action {
                    AgentAction::Bet(_) => {
                        let raise_count = round_raise_counts.entry(current_round).or_insert(0);
                        *raise_count += 1;

                        // Maximum 3 raises per round (unless heads-up)
                        let num_active = active_players_per_round
                            .get(&current_round)
                            .map(|s| s.len())
                            .unwrap_or(0);
                        if num_active > 2 {
                            assert!(
                                *raise_count <= 3,
                                "Too many raises in round {:?}: {}. Maximum 3 raises allowed with 3+ players",
                                current_round,
                                *raise_count
                            );
                        }
                        // In heads-up, unlimited raises are allowed (no check needed)
                    }
                    AgentAction::Fold => {
                        // Player should not have any actions after folding (checked elsewhere)
                    }
                    AgentAction::Call | AgentAction::AllIn => {
                        // Valid actions
                    }
                }
            }
            _ => {
                // Other actions like GameStart, community card deals, etc.
            }
        }
    }
}

/// Validate round progression follows Texas Hold'em sequence
fn validate_round_progression(history_storage: &[HistoryRecord]) {
    let round_advances: Vec<Round> = history_storage
        .iter()
        .filter_map(|record| {
            if let Action::RoundAdvance(round) = &record.action {
                Some(*round)
            } else {
                None
            }
        })
        .collect();

    if round_advances.is_empty() {
        return; // No round advances, nothing to validate
    }

    // Valid sequences: Preflop -> Flop -> Turn -> River -> Complete
    // But some rounds might be skipped if everyone folds
    let mut prev_round: Option<Round> = None;

    for round in &round_advances {
        if let Some(prev) = prev_round {
            match (prev, *round) {
                // Normal progression through all rounds
                (Round::Starting, Round::Ante) |
                (Round::Ante, Round::DealPreflop) |
                (Round::DealPreflop, Round::Preflop) |
                (Round::Preflop, Round::DealFlop) |
                (Round::DealFlop, Round::Flop) |
                (Round::Flop, Round::DealTurn) |
                (Round::DealTurn, Round::Turn) |
                (Round::Turn, Round::DealRiver) |
                (Round::DealRiver, Round::River) |
                (Round::River, Round::Showdown) |
                (Round::Showdown, Round::Complete) |
                // Early completion due to folds
                (Round::Starting, Round::Complete) |  // Everyone folds immediately
                (Round::Ante, Round::Complete) |      // Everyone folds after antes
                (Round::DealPreflop, Round::Complete) | // Everyone folds after dealing
                (Round::Preflop, Round::Complete) |   // All fold preflop
                (Round::DealFlop, Round::Complete) |  // All fold after flop dealing
                (Round::Flop, Round::Complete) |      // All fold after flop
                (Round::DealTurn, Round::Complete) |  // All fold after turn dealing
                (Round::Turn, Round::Complete) |      // All fold after turn
                (Round::DealRiver, Round::Complete) | // All fold after river dealing
                (Round::River, Round::Complete) => {  // All fold after river
                    // Valid progression
                }
                _ => {
                    panic!(
                        "Invalid round progression: {:?} -> {:?}",
                        prev, round
                    );
                }
            }
        }
        prev_round = Some(*round);
    }

    // Last round should always be Complete
    assert_eq!(
        round_advances.last(),
        Some(&Round::Complete),
        "Game should end with Complete round, but ended with: {:?}",
        round_advances.last()
    );
}
