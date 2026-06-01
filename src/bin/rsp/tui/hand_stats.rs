//! Reconstruct the TUI's displayed per-agent statistics from an Open Hand
//! History record.
//!
//! This is the single canonical "stats from a hand" path, used by the static
//! OHH viewer (`ohh::stats::build_state_from_hands`) and by the filtered-view
//! recompute (`projection::build_projection`). It mirrors the definitions in
//! `StatsTrackingHistorian` (`src/arena/historian/stats_tracking.rs`) for every
//! metric surfaced in `AgentDisplayData`:
//!   profit, invested/ROI, win/loss, VPIP, PFR, 3-bet, aggression factor,
//!   c-bet, WTSD, WSD.
//!
//! Stats NOT surfaced by the TUI table (steal, per-street bet/raise/call
//! splits, per-street wins) are intentionally left at their defaults — the
//! viewer never reads them, so reconstructing them would be dead code.

use std::collections::HashMap;

use rs_poker::arena::historian::StatsStorage;
use rs_poker::open_hand_history::{Action, HandHistory};

use crate::tui::state::{GameResult, PROFIT_EPSILON, RoundLabel, SeatStats, compute_hand_profits};

/// Reconstruct a [`GameResult`] (profits, ending round, per-seat displayed
/// stats) from an OHH `HandHistory`.
pub fn game_result_from_hand(hand: &HandHistory) -> GameResult {
    let num_players = hand.players.len();
    let (id_to_idx, profits) = compute_hand_profits(hand);

    let mut storage = StatsStorage::new_with_num_players(num_players.max(1));

    // --- Financials, hands played, win/loss/breakeven ---
    for (i, &profit) in profits.iter().enumerate() {
        storage.hands_played[i] = 1;
        storage.total_profit[i] = profit;

        // invested = sum of every chip this player put in the pot, incl. blinds.
        let mut invested = 0.0_f32;
        for r in &hand.rounds {
            for a in &r.actions {
                if id_to_idx.get(&a.player_id) == Some(&i) && is_invested_action(&a.action) {
                    invested += a.amount;
                }
            }
        }
        storage.total_invested[i] = invested;

        if profit > PROFIT_EPSILON {
            storage.games_won[i] = 1;
        } else if profit < -PROFIT_EPSILON {
            storage.games_lost[i] = 1;
        } else {
            storage.games_breakeven[i] = 1;
        }
    }

    let ending_round = hand
        .rounds
        .last()
        .map(|r| RoundLabel::from_street_name(&r.street))
        .unwrap_or(RoundLabel::Preflop);

    reconstruct_action_stats(hand, &id_to_idx, &profits, &mut storage);

    let agent_names: Vec<String> = hand.players.iter().map(|p| p.name.clone()).collect();
    let seat_stats: Vec<SeatStats> = (0..num_players)
        .map(|i| SeatStats::from_storage(&storage, i))
        .collect();

    GameResult {
        agent_names,
        profits,
        ending_round,
        seat_stats,
        big_blind: hand.big_blind_amount,
    }
}

/// Actions that move chips into the pot (used for `total_invested`).
fn is_invested_action(action: &Action) -> bool {
    matches!(
        action,
        Action::Bet
            | Action::Raise
            | Action::Call
            | Action::PostSmallBlind
            | Action::PostBigBlind
            | Action::PostAnte
            | Action::Straddle
            | Action::PostDead
            | Action::PostExtraBlind
            | Action::AddedToPot
    )
}

/// Reconstruct the action-derived displayed stats: VPIP/PFR, aggression counts,
/// 3-bet (preflop), then c-bet and WTSD/WSD via dedicated helpers.
fn reconstruct_action_stats(
    hand: &HandHistory,
    id_to_idx: &HashMap<u64, usize>,
    profits: &[f32],
    storage: &mut StatsStorage,
) {
    for r in &hand.rounds {
        let is_preflop = r.street.eq_ignore_ascii_case("preflop");
        let mut pf_raises: usize = 0; // number of raises seen so far this preflop
        for a in &r.actions {
            let Some(&idx) = id_to_idx.get(&a.player_id) else {
                continue;
            };

            // 3-bet opportunity: acting while facing exactly the open raise.
            if is_preflop
                && pf_raises == 1
                && matches!(
                    a.action,
                    Action::Fold | Action::Call | Action::Raise | Action::Bet
                )
            {
                storage.three_bet_opportunities[idx] += 1;
            }

            match a.action {
                Action::Call => {
                    storage.call_count[idx] += 1;
                    if is_preflop {
                        storage.hands_vpip[idx] = 1;
                        storage.vpip_count[idx] += 1;
                        storage.vpip_total[idx] += a.amount;
                    }
                }
                Action::Bet => {
                    storage.bet_count[idx] += 1;
                    if is_preflop {
                        storage.hands_vpip[idx] = 1;
                        storage.hands_pfr[idx] = 1;
                        storage.vpip_count[idx] += 1;
                        storage.vpip_total[idx] += a.amount;
                    }
                }
                Action::Raise => {
                    storage.raise_count[idx] += 1;
                    if is_preflop {
                        storage.hands_vpip[idx] = 1;
                        storage.hands_pfr[idx] = 1;
                        storage.preflop_raise_count[idx] += 1;
                        storage.vpip_count[idx] += 1;
                        storage.vpip_total[idx] += a.amount;
                        if pf_raises == 1 {
                            storage.three_bet_count[idx] += 1;
                        }
                        pf_raises += 1;
                    }
                }
                Action::Fold => {
                    storage.fold_count[idx] += 1;
                }
                _ => {}
            }
            storage.actions_count[idx] += 1;
        }
    }

    reconstruct_cbet(hand, id_to_idx, storage);
    reconstruct_showdown(hand, id_to_idx, profits, storage);
}

/// C-bet: the preflop aggressor (last preflop raiser) gets an opportunity if,
/// on the flop, they act with no prior flop bet; taken if that action bets/raises.
fn reconstruct_cbet(
    hand: &HandHistory,
    id_to_idx: &HashMap<u64, usize>,
    storage: &mut StatsStorage,
) {
    // Preflop aggressor = player_id of the LAST preflop Raise.
    let mut aggressor: Option<u64> = None;
    for r in &hand.rounds {
        if r.street.eq_ignore_ascii_case("preflop") {
            for a in &r.actions {
                if a.action == Action::Raise {
                    aggressor = Some(a.player_id);
                }
            }
        }
    }
    let Some(aggressor_id) = aggressor else {
        return;
    };
    let Some(&aggressor_idx) = id_to_idx.get(&aggressor_id) else {
        return;
    };

    // Walk the flop: find the aggressor's first action before any flop bet.
    for r in &hand.rounds {
        if !r.street.eq_ignore_ascii_case("flop") {
            continue;
        }
        let mut flop_bet_occurred = false;
        for a in &r.actions {
            if a.player_id == aggressor_id && !flop_bet_occurred {
                storage.cbet_opportunities[aggressor_idx] += 1;
                if matches!(a.action, Action::Bet | Action::Raise) {
                    storage.cbet_count[aggressor_idx] += 1;
                }
                break; // only the first qualifying flop action matters
            }
            if matches!(a.action, Action::Bet | Action::Raise) {
                flop_bet_occurred = true;
            }
        }
        break; // only one flop round
    }
}

/// WTSD / showdown reconstruction.
fn reconstruct_showdown(
    hand: &HandHistory,
    id_to_idx: &HashMap<u64, usize>,
    profits: &[f32],
    storage: &mut StatsStorage,
) {
    let num_players = hand.players.len();
    let has_flop = hand
        .rounds
        .iter()
        .any(|r| r.street.eq_ignore_ascii_case("flop"));

    // folded[i]: player folded at any point in the hand.
    let mut folded = vec![false; num_players];
    // folded_preflop[i]: player folded during the preflop round.
    let mut folded_preflop = vec![false; num_players];
    for r in &hand.rounds {
        let is_preflop = r.street.eq_ignore_ascii_case("preflop");
        for a in &r.actions {
            if a.action == Action::Fold
                && let Some(&idx) = id_to_idx.get(&a.player_id)
            {
                folded[idx] = true;
                if is_preflop {
                    folded_preflop[idx] = true;
                }
            }
        }
    }

    let survivors = (0..num_players).filter(|&i| !folded[i]).count();
    let went_to_showdown = survivors >= 2;

    for i in 0..num_players {
        // Saw the flop: a flop round exists and the player did not fold preflop.
        if has_flop && !folded_preflop[i] {
            storage.wtsd_opportunities[i] += 1;
            if went_to_showdown && !folded[i] {
                storage.wtsd_count[i] += 1;
            }
        }
        if went_to_showdown && !folded[i] {
            storage.showdown_count[i] += 1;
            if profits[i] > PROFIT_EPSILON {
                storage.showdown_wins[i] += 1;
            }
        }
    }
}

#[cfg(test)]
pub mod test_util {
    use rs_poker::open_hand_history::*;

    /// A minimal valid heads-up hand: SB folds to BB preflop.
    pub fn simple_hand(game_number: &str) -> HandHistory {
        HandHistory {
            spec_version: "1.4.7".into(),
            site_name: "test".into(),
            network_name: "test".into(),
            internal_version: "1.0".into(),
            tournament: false,
            tournament_info: None,
            game_number: game_number.into(),
            start_date_utc: None,
            table_name: "test".into(),
            table_handle: None,
            table_skin: None,
            game_type: GameType::Holdem,
            bet_limit: None,
            table_size: 2,
            currency: "USD".into(),
            dealer_seat: 0,
            small_blind_amount: 5.0,
            big_blind_amount: 10.0,
            ante_amount: 0.0,
            hero_player_id: None,
            players: vec![
                PlayerObj {
                    id: 1,
                    seat: 0,
                    name: "A".into(),
                    display: None,
                    starting_stack: 1000.0,
                    player_bounty: None,
                    is_sitting_out: None,
                },
                PlayerObj {
                    id: 2,
                    seat: 1,
                    name: "B".into(),
                    display: None,
                    starting_stack: 1000.0,
                    player_bounty: None,
                    is_sitting_out: None,
                },
            ],
            rounds: vec![RoundObj {
                id: 0,
                street: "Preflop".into(),
                cards: None,
                actions: vec![
                    ActionObj {
                        action_number: 1,
                        player_id: 1,
                        action: Action::PostSmallBlind,
                        amount: 5.0,
                        is_allin: false,
                        cards: None,
                    },
                    ActionObj {
                        action_number: 2,
                        player_id: 2,
                        action: Action::PostBigBlind,
                        amount: 10.0,
                        is_allin: false,
                        cards: None,
                    },
                    ActionObj {
                        action_number: 3,
                        player_id: 1,
                        action: Action::Fold,
                        amount: 0.0,
                        is_allin: false,
                        cards: None,
                    },
                ],
            }],
            pots: vec![PotObj {
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
            }],
            tournament_bounties: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rs_poker::open_hand_history::*;

    pub(super) fn player(id: u64, seat: u64, name: &str) -> PlayerObj {
        PlayerObj {
            id,
            seat,
            name: name.to_string(),
            display: None,
            starting_stack: 1000.0,
            player_bounty: None,
            is_sitting_out: None,
        }
    }

    pub(super) fn act(player_id: u64, action: Action, amount: f32) -> ActionObj {
        ActionObj {
            action_number: 0,
            player_id,
            action,
            amount,
            is_allin: false,
            cards: None,
        }
    }

    pub(super) fn round(street: &str, actions: Vec<ActionObj>) -> RoundObj {
        RoundObj {
            id: 0,
            street: street.into(),
            cards: None,
            actions,
        }
    }

    pub(super) fn win(player_id: u64, amount: f32) -> PlayerWinsObj {
        PlayerWinsObj {
            player_id,
            win_amount: amount,
            cashout_amount: None,
            cashout_fee: None,
            bonus_amount: None,
            contributed_rake: None,
        }
    }

    pub(super) fn hand(
        players: Vec<PlayerObj>,
        rounds: Vec<RoundObj>,
        pots: Vec<PotObj>,
        dealer_seat: u64,
    ) -> HandHistory {
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
            dealer_seat,
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

    fn pot(amount: f32, wins: Vec<PlayerWinsObj>) -> PotObj {
        PotObj {
            number: 0,
            amount,
            rake: None,
            jackpot: None,
            player_wins: wins,
        }
    }

    #[test]
    fn test_financials_and_ending_round() {
        // Alice (SB) folds preflop, Bob (BB) wins the 15 pot.
        let h = hand(
            vec![player(1, 0, "Alice"), player(2, 1, "Bob")],
            vec![round(
                "Preflop",
                vec![
                    act(1, Action::PostSmallBlind, 5.0),
                    act(2, Action::PostBigBlind, 10.0),
                    act(1, Action::Fold, 0.0),
                ],
            )],
            vec![pot(15.0, vec![win(2, 15.0)])],
            0,
        );

        let result = game_result_from_hand(&h);
        assert_eq!(result.agent_names, vec!["Alice", "Bob"]);
        assert_eq!(result.ending_round, RoundLabel::Preflop);
        // Alice invested 5, won 0 => -5. Bob invested 10, won 15 => +5.
        assert!((result.profits[0] - (-5.0)).abs() < 0.01);
        assert!((result.profits[1] - 5.0).abs() < 0.01);

        let s = &result.seat_stats;
        assert_eq!(s[0].hands_played, 1);
        assert_eq!(s[0].games_lost, 1);
        assert_eq!(s[1].games_won, 1);
        // total_invested: Alice 5, Bob 10.
        assert!((s[0].total_invested - 5.0).abs() < 0.01);
        assert!((s[1].total_invested - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_vpip_pfr() {
        // Alice raises preflop (vpip + pfr). Bob calls (vpip only). Carol folds.
        let h = hand(
            vec![
                player(1, 0, "Alice"),
                player(2, 1, "Bob"),
                player(3, 2, "Carol"),
            ],
            vec![round(
                "Preflop",
                vec![
                    act(1, Action::PostSmallBlind, 5.0),
                    act(2, Action::PostBigBlind, 10.0),
                    act(3, Action::Raise, 30.0),
                    act(1, Action::Call, 25.0),
                    act(2, Action::Fold, 0.0),
                ],
            )],
            vec![],
            0,
        );
        // Re-map: who raised is Carol(idx2), who called is Alice(idx0).
        let r = game_result_from_hand(&h);
        let s = &r.seat_stats;
        // Carol raised => vpip + pfr.
        assert_eq!(s[2].hands_vpip, 1);
        assert_eq!(s[2].hands_pfr, 1);
        assert_eq!(s[2].preflop_raise_count, 1);
        // Alice called => vpip, not pfr.
        assert_eq!(s[0].hands_vpip, 1);
        assert_eq!(s[0].hands_pfr, 0);
        // Bob only posted BB then folded => no voluntary money => no vpip.
        assert_eq!(s[1].hands_vpip, 0);
        assert_eq!(s[1].hands_pfr, 0);
    }

    #[test]
    fn test_three_bet() {
        // Open-raise by A; B faces raise #1 and re-raises => 3-bet.
        // C faces raise #1 then a 3-bet; C only gets a 3-bet *opportunity*
        // while facing exactly the open raise.
        let h = hand(
            vec![player(1, 0, "A"), player(2, 1, "B"), player(3, 2, "C")],
            vec![round(
                "Preflop",
                vec![
                    act(1, Action::PostSmallBlind, 5.0),
                    act(2, Action::PostBigBlind, 10.0),
                    act(3, Action::Raise, 30.0), // open raise (raise #1)
                    act(1, Action::Raise, 90.0), // 3-bet (faced raise #1)
                    act(2, Action::Fold, 0.0),
                    act(3, Action::Call, 60.0),
                ],
            )],
            vec![],
            0,
        );
        let r = game_result_from_hand(&h);
        let s = &r.seat_stats;
        // A (idx0) re-raised while facing exactly raise #1 => 3-bet.
        assert_eq!(s[0].three_bet_count, 1);
        assert_eq!(s[0].three_bet_opportunities, 1);
        // C (idx2) made the open raise (facing raise #0) => not a 3-bet, no opp at that point.
        assert_eq!(s[2].three_bet_count, 0);
        // B (idx1) folded while facing raise #2 (the 3-bet) => no opportunity (only raise #1 counts).
        assert_eq!(s[1].three_bet_opportunities, 0);
    }

    #[test]
    fn test_cbet() {
        // A opens preflop (last preflop raiser => aggressor). On the flop A is
        // first to act and bets => cbet opportunity taken.
        let h = hand(
            vec![player(1, 0, "A"), player(2, 1, "B")],
            vec![
                round(
                    "Preflop",
                    vec![
                        act(1, Action::PostSmallBlind, 5.0),
                        act(2, Action::PostBigBlind, 10.0),
                        act(1, Action::Raise, 30.0),
                        act(2, Action::Call, 20.0),
                    ],
                ),
                round(
                    "Flop",
                    vec![act(1, Action::Bet, 40.0), act(2, Action::Fold, 0.0)],
                ),
            ],
            vec![],
            0,
        );
        let r = game_result_from_hand(&h);
        let s = &r.seat_stats;
        assert_eq!(s[0].cbet_opportunities, 1);
        assert_eq!(s[0].cbet_count, 1);
        // B was not the aggressor => no cbet opportunity.
        assert_eq!(s[1].cbet_opportunities, 0);
    }

    #[test]
    fn test_cbet_opportunity_not_taken_when_checked() {
        // A opens, on flop A checks (no Bet/Raise) => opportunity, not taken.
        let h = hand(
            vec![player(1, 0, "A"), player(2, 1, "B")],
            vec![
                round(
                    "Preflop",
                    vec![
                        act(1, Action::PostSmallBlind, 5.0),
                        act(2, Action::PostBigBlind, 10.0),
                        act(1, Action::Raise, 30.0),
                        act(2, Action::Call, 20.0),
                    ],
                ),
                round(
                    "Flop",
                    vec![act(1, Action::Check, 0.0), act(2, Action::Check, 0.0)],
                ),
            ],
            vec![],
            0,
        );
        let r = game_result_from_hand(&h);
        assert_eq!(r.seat_stats[0].cbet_opportunities, 1);
        assert_eq!(r.seat_stats[0].cbet_count, 0);
    }

    #[test]
    fn test_wtsd_and_wsd() {
        // A and B both reach showdown on the river; A wins.
        let h = hand(
            vec![player(1, 0, "A"), player(2, 1, "B")],
            vec![
                round(
                    "Preflop",
                    vec![
                        act(1, Action::PostSmallBlind, 5.0),
                        act(2, Action::PostBigBlind, 10.0),
                        act(1, Action::Call, 5.0),
                        act(2, Action::Check, 0.0),
                    ],
                ),
                round(
                    "Flop",
                    vec![act(2, Action::Check, 0.0), act(1, Action::Check, 0.0)],
                ),
                round(
                    "Turn",
                    vec![act(2, Action::Check, 0.0), act(1, Action::Check, 0.0)],
                ),
                round(
                    "River",
                    vec![act(2, Action::Check, 0.0), act(1, Action::Check, 0.0)],
                ),
            ],
            vec![pot(20.0, vec![win(1, 20.0)])],
            0,
        );
        let r = game_result_from_hand(&h);
        let s = &r.seat_stats;
        // Both saw the flop, both went to showdown.
        assert_eq!(s[0].wtsd_opportunities, 1);
        assert_eq!(s[0].wtsd_count, 1);
        assert_eq!(s[1].wtsd_opportunities, 1);
        assert_eq!(s[1].wtsd_count, 1);
        assert_eq!(s[0].showdown_count, 1);
        assert_eq!(s[1].showdown_count, 1);
        // A won at showdown (profit +10), B lost (-10).
        assert_eq!(s[0].showdown_wins, 1);
        assert_eq!(s[1].showdown_wins, 0);
    }

    #[test]
    fn test_no_showdown_when_one_player_left() {
        // A opens, B folds preflop => no showdown; A saw no flop opportunity.
        let h = hand(
            vec![player(1, 0, "A"), player(2, 1, "B")],
            vec![round(
                "Preflop",
                vec![
                    act(1, Action::PostSmallBlind, 5.0),
                    act(2, Action::PostBigBlind, 10.0),
                    act(1, Action::Raise, 30.0),
                    act(2, Action::Fold, 0.0),
                ],
            )],
            vec![pot(20.0, vec![win(1, 20.0)])],
            0,
        );
        let r = game_result_from_hand(&h);
        let s = &r.seat_stats;
        assert_eq!(s[0].showdown_count, 0);
        assert_eq!(s[0].wtsd_opportunities, 0); // no flop round
    }
}
