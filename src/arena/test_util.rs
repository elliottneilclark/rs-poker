use super::game_state::RoundData;

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
