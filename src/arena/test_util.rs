use approx::assert_relative_eq;

use crate::arena::game_state::Round;

use super::{GameState, game_state::RoundData};

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
        active_bets.into_iter().for_each(|bet| {
            assert_eq!(max, bet);
        });
    }

    // Can't raise more than we bet
    round_data
        .bet_count
        .iter()
        .zip(round_data.raise_count.iter())
        .for_each(|(bet_count, raise_count)| {
            assert!(*bet_count >= *raise_count);
        });
}

pub fn assert_valid_game_state(game_state: &GameState) {
    assert_eq!(Round::Complete, game_state.round);

    let should_have_bets = game_state.ante + game_state.small_blind + game_state.big_blind > 0.0;

    let total_bet = game_state.player_bet.iter().cloned().sum();

    let mut minimum_number_of_actions: usize = 0;
    if game_state.ante > 0.0 {
        minimum_number_of_actions += game_state.num_players;
    }
    if game_state.small_blind > 0.0 {
        minimum_number_of_actions += 1;
    }
    if game_state.big_blind > 0.0 {
        minimum_number_of_actions += 1;
    }
    minimum_number_of_actions += 1; // Fewest actions is a heads up game where first player folds immediately.

    println!("{:?} {}", game_state, minimum_number_of_actions);
    assert!(game_state.times_acted.iter().sum::<usize>() >= minimum_number_of_actions);

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

    let total_winning: f32 = game_state.player_winnings.iter().cloned().sum();

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

        let mut minimum_player_actions = 0;

        if game_state.ante > 0.0 {
            minimum_player_actions += 1;
        }

        if game_state.num_players > 2 { // Full-ring
            let left_of_dealer_idx = (game_state.dealer_idx + 1) % game_state.num_players;

            if idx == left_of_dealer_idx && game_state.small_blind > 0.0 {
                minimum_player_actions += 1;
            }

            let two_left_of_dealer_idx = (left_of_dealer_idx + 1) % game_state.num_players;

            if idx == two_left_of_dealer_idx && game_state.big_blind > 0.0 {
                minimum_player_actions += 1;
            }
        } else if game_state.num_players == 2 { // Heads-up
            if idx == game_state.dealer_idx && game_state.small_blind > 0.0 {
                minimum_player_actions += 1;
            }

            let non_dealer_idx = (game_state.dealer_idx + 1) % game_state.num_players;

            if idx == non_dealer_idx && game_state.big_blind > 0.0 {
                minimum_player_actions += 1;
            }
        }

        if let Some(action_count) = game_state.times_acted.get(idx) {
            assert!(*action_count >= minimum_player_actions);
        }
    }
}
