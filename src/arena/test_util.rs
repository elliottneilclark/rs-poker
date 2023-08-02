use crate::arena::game_state::Round;

use super::{game_state::RoundData, GameState};

pub fn assert_valid_round_data(round_data: &RoundData) {
    // Get all of the player still active at the end of the round.
    // for any round with bets they should have called.
    //
    // EG no one should call for less than the max and still be in.
    let active_bets: Vec<i32> = round_data
        .player_bet
        .iter()
        .enumerate()
        .filter(|(idx, _)| round_data.player_active.get(*idx))
        .map(|(_, bet)| *bet)
        .collect();

    let max_active = active_bets.iter().max();

    if let Some(max) = max_active {
        active_bets.iter().for_each(|bet| {
            assert_eq!(*max, *bet);
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

    let total_bet: i32 = game_state.player_bet.iter().cloned().sum();
    assert_eq!(total_bet, game_state.total_pot);
    assert_ne!(0, total_bet);

    let total_winning: i32 = game_state.player_winnings.iter().cloned().sum();

    // Because we split winnings on ties and these are all integers.
    // It's possible that there's some winnings that aren't awarded
    assert!(total_winning <= total_bet);
    assert!(total_winning <= game_state.total_pot);

    assert!(
        total_winning + game_state.num_players as i32 >= total_bet,
        "invalid game state total winnings = {} total bet = {} game state = {:?}",
        total_winning,
        total_bet,
        game_state
    );
    assert!(total_winning + game_state.num_players as i32 >= game_state.total_pot);

    // The dealer has to be well specified.
    assert!(game_state.dealer_idx < game_state.num_players);

    // The board should be full or getting full
    assert!(game_state.board.len() <= 5);

    assert!(game_state.small_blind <= game_state.big_blind);

    for idx in 0..game_state.num_players {
        // If they aren't active (folded)
        // and aren't all in then they shouldn't win anything
        if !game_state.player_active.get(idx) && !game_state.player_all_in.get(idx) {
            assert_eq!(0, game_state.player_winnings[idx]);
        }
    }
}
