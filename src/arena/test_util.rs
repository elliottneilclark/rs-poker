use approx::assert_relative_eq;

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
        for bet in active_bets.into_iter() {
            assert_eq!(
                bet, max,
                "Players still active should have called the max bet, round_data: {round_data:?}"
            );
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
    assert_relative_eq!(total_bet, game_state.total_pot, epsilon = epsilon);

    let total_winning: f32 = game_state.player_winnings.iter().copied().sum();

    assert_relative_eq!(total_winning, total_bet, epsilon = epsilon);
    assert_relative_eq!(total_winning, game_state.total_pot, epsilon = epsilon);

    // The dealer has to be well specified.
    assert!(game_state.dealer_idx < game_state.num_players);

    // The board should be full or getting full
    assert!(game_state.board.len() <= 5);

    assert!(game_state.small_blind <= game_state.big_blind);

    for idx in 0..game_state.num_players {
        // If they aren't active (folded)
        // and aren't all in then they shouldn't win anything
        if !game_state.player_active.get(idx) && !game_state.player_all_in.get(idx) {
            assert_eq!(0.0, game_state.player_winnings[idx]);
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
