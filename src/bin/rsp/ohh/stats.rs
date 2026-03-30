use rs_poker::arena::historian::StatsStorage;
use rs_poker::open_hand_history::{Action, HandHistory};

use crate::tui::state::{
    GameResult, PROFIT_EPSILON, RoundLabel, SeatStats, TuiState, compute_hand_profits,
};

/// Build a TuiState from a collection of parsed hand histories.
pub fn build_state_from_hands(hands: &[HandHistory]) -> TuiState {
    let mut state = TuiState::new(Some(hands.len()));

    for hand in hands {
        let num_players = hand.players.len();
        if num_players == 0 {
            continue;
        }

        let (id_to_idx, profits) = compute_hand_profits(hand);

        let mut stats = StatsStorage::new_with_num_players(num_players);
        for (i, &profit) in profits.iter().enumerate().take(num_players) {
            stats.total_profit[i] = profit;
            // Reconstruct invested from wins - profit
            let wins: f32 = hand
                .pots
                .iter()
                .flat_map(|pot| &pot.player_wins)
                .filter(|pw| id_to_idx.get(&pw.player_id) == Some(&i))
                .map(|pw| pw.win_amount)
                .sum();
            stats.total_invested[i] = wins - profit;
            stats.hands_played[i] = 1;
            if profit > PROFIT_EPSILON {
                stats.games_won[i] = 1;
            } else if profit < -PROFIT_EPSILON {
                stats.games_lost[i] = 1;
            } else {
                stats.games_breakeven[i] = 1;
            }
        }

        let ending_round = hand
            .rounds
            .last()
            .map(|r| RoundLabel::from_street_name(&r.street))
            .unwrap_or(RoundLabel::Preflop);

        // Count basic actions per player for stats
        for round in &hand.rounds {
            let is_preflop = round.street.to_lowercase() == "preflop";
            for action in &round.actions {
                if let Some(&idx) = id_to_idx.get(&action.player_id) {
                    match action.action {
                        Action::Fold => {
                            stats.fold_count[idx] += 1;
                        }
                        Action::Call => {
                            stats.call_count[idx] += 1;
                            if is_preflop {
                                stats.hands_vpip[idx] = 1;
                            }
                        }
                        Action::Bet => {
                            stats.bet_count[idx] += 1;
                            if is_preflop {
                                stats.hands_vpip[idx] = 1;
                                stats.hands_pfr[idx] = 1;
                            }
                        }
                        Action::Raise => {
                            stats.raise_count[idx] += 1;
                            if is_preflop {
                                stats.hands_vpip[idx] = 1;
                                stats.hands_pfr[idx] = 1;
                            }
                        }
                        _ => {}
                    }
                    stats.actions_count[idx] += 1;
                }
            }
        }

        let agent_names: Vec<String> = hand.players.iter().map(|p| p.name.clone()).collect();
        let seat_stats: Vec<SeatStats> = (0..num_players)
            .map(|i| SeatStats::from_storage(&stats, i))
            .collect();

        state.update(GameResult {
            agent_names,
            profits,
            ending_round,
            seat_stats,
        });
    }

    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use rs_poker::open_hand_history::*;

    fn make_player(id: u64, name: &str) -> PlayerObj {
        PlayerObj {
            id,
            seat: id,
            name: name.to_string(),
            display: None,
            starting_stack: 1000.0,
            player_bounty: None,
            is_sitting_out: None,
        }
    }

    fn make_action(num: u64, player_id: u64, action: Action, amount: f32) -> ActionObj {
        ActionObj {
            action_number: num,
            player_id,
            action,
            amount,
            is_allin: false,
            cards: None,
        }
    }

    fn make_round(id: u64, street: &str, actions: Vec<ActionObj>) -> RoundObj {
        RoundObj {
            id,
            street: street.into(),
            cards: None,
            actions,
        }
    }

    fn make_hand(players: Vec<PlayerObj>, rounds: Vec<RoundObj>, pots: Vec<PotObj>) -> HandHistory {
        HandHistory {
            spec_version: "1.4.7".into(),
            site_name: "test".into(),
            network_name: "test".into(),
            internal_version: "1.0".into(),
            tournament: false,
            tournament_info: None,
            game_number: "1".into(),
            start_date_utc: None,
            table_name: "test".into(),
            table_handle: None,
            table_skin: None,
            game_type: GameType::Holdem,
            bet_limit: None,
            table_size: players.len() as u64,
            currency: "USD".into(),
            dealer_seat: 0,
            small_blind_amount: 5.0,
            big_blind_amount: 10.0,
            ante_amount: 0.0,
            hero_player_id: None,
            players,
            rounds,
            pots,
            tournament_bounties: None,
        }
    }

    #[test]
    fn test_empty_hands() {
        let mut state = build_state_from_hands(&[]);
        assert_eq!(state.games_completed, 0);
        assert!(state.agent_display_data().is_empty());
    }

    #[test]
    fn test_single_hand_profits() {
        let players = vec![make_player(1, "Alice"), make_player(2, "Bob")];
        let rounds = vec![make_round(
            1,
            "Preflop",
            vec![
                make_action(1, 1, Action::PostSmallBlind, 5.0),
                make_action(2, 2, Action::PostBigBlind, 10.0),
                make_action(3, 1, Action::Fold, 0.0),
            ],
        )];
        let pots = vec![PotObj {
            number: 0,
            amount: 15.0,
            rake: None,
            jackpot: None,
            player_wins: vec![PlayerWinsObj {
                player_id: 2,
                win_amount: 15.0,
                cashout_amount: None,
                cashout_fee: None,
                bonus_amount: None,
                contributed_rake: None,
            }],
        }];
        let hand = make_hand(players, rounds, pots);
        let mut state = build_state_from_hands(&[hand]);
        assert_eq!(state.games_completed, 1);

        let agents = state.agent_display_data();
        assert_eq!(agents.len(), 2);

        let bob = agents.iter().find(|a| a.name == "Bob").unwrap();
        // Bob invested 10, won 15 = profit +5
        assert!((bob.total_profit - 5.0).abs() < 0.01);

        let alice = agents.iter().find(|a| a.name == "Alice").unwrap();
        // Alice invested 5, won 0 = profit -5
        assert!((alice.total_profit - (-5.0)).abs() < 0.01);
    }

    #[test]
    fn test_preflop_only_hand() {
        let players = vec![make_player(1, "A")];
        let rounds = vec![make_round(1, "Preflop", vec![])];
        let hand = make_hand(players, rounds, vec![]);
        let state = build_state_from_hands(&[hand]);
        assert_eq!(state.street_dist.preflop, 1);
        assert_eq!(state.street_dist.total(), 1);
    }

    #[test]
    fn test_multi_round_hand() {
        let players = vec![make_player(1, "A"), make_player(2, "B")];
        let rounds = vec![
            make_round(
                1,
                "Preflop",
                vec![
                    make_action(1, 1, Action::Call, 10.0),
                    make_action(2, 2, Action::Raise, 20.0),
                ],
            ),
            make_round(2, "River", vec![make_action(3, 1, Action::Bet, 30.0)]),
        ];
        let hand = make_hand(players, rounds, vec![]);
        let state = build_state_from_hands(&[hand]);
        assert_eq!(state.street_dist.river, 1);
    }

    #[test]
    fn test_action_counting() {
        let players = vec![make_player(1, "A"), make_player(2, "B")];
        let rounds = vec![make_round(
            1,
            "Preflop",
            vec![
                make_action(1, 1, Action::Call, 10.0),
                make_action(2, 2, Action::Raise, 20.0),
                make_action(3, 1, Action::Fold, 0.0),
            ],
        )];
        let hand = make_hand(players, rounds, vec![]);
        let mut state = build_state_from_hands(&[hand]);
        let agents = state.agent_display_data();

        // Agent B raised preflop -> vpip + pfr
        let b = agents.iter().find(|a| a.name == "B").unwrap();
        assert!(b.vpip_percent > 0.0);
        assert!(b.pfr_percent > 0.0);
    }
}
