use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::arena::historian::StatsStorage;

/// Statistics for a specific position (seat)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionStats {
    /// The seat index (0-indexed)
    pub seat_index: usize,
    /// Number of games played in this position
    pub games_played: usize,
    /// Total profit in this position
    pub profit: f32,
    /// Average profit per game in this position
    pub profit_per_game: f32,
}

/// Aggregated statistics for a single agent across all permutations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStats {
    /// Name of the agent
    pub agent_name: String,

    // Financial Performance
    /// Total profit across all games
    pub total_profit: f32,
    /// Total number of games played
    pub total_games: usize,
    /// Number of games won (profit > 0)
    pub wins: usize,
    /// Number of games lost (profit < 0)
    pub losses: usize,
    /// Number of games with zero profit
    pub breakeven: usize,

    // Profitability Metrics
    /// Average profit per game
    pub profit_per_game: f32,
    /// Profit per 100 hands (profit_per_game * 100)
    pub profit_per_100_hands: f32,
    /// Return on investment percentage
    pub roi_percent: f32,

    // Position Analysis
    /// Statistics broken down by seat position
    pub position_stats: Vec<PositionStats>,

    // Poker Statistics
    /// Voluntarily Put money In Pot percentage
    pub vpip_percent: f32,
    /// Pre-Flop Raise percentage
    pub pfr_percent: f32,
    /// 3-Bet percentage
    pub three_bet_percent: f32,
    /// Aggression factor (raises + bets) / calls
    pub aggression_factor: f32,

    // Advanced Stats
    /// Continuation bet percentage
    pub cbet_percent: f32,
    /// Went To ShowDown percentage
    pub wtsd_percent: f32,
    /// Won money at ShowDown percentage
    pub wsd_percent: f32,
    /// Attempted To Steal percentage
    pub steal_percent: f32,
    /// Aggression frequency percentage
    pub aggression_frequency: f32,
    /// Flop aggression factor
    pub flop_aggression_factor: f32,
    /// Turn aggression factor
    pub turn_aggression_factor: f32,
    /// River aggression factor
    pub river_aggression_factor: f32,

    // Round-by-Round Performance
    /// Win rate for hands ending at preflop
    pub preflop_win_rate: f32,
    /// Win rate for hands ending at flop
    pub flop_win_rate: f32,
    /// Win rate for hands ending at turn
    pub turn_win_rate: f32,
    /// Win rate for hands ending at river
    pub river_win_rate: f32,
}

/// Builder for aggregating agent statistics across multiple games
pub struct AgentStatsBuilder {
    /// Accumulated stats for each agent (indexed by agent_idx)
    agent_accumulated: Vec<StatsStorage>,
    /// Position tracking: agent_idx -> seat_idx -> (games_played, total_profit)
    position_tracking: Vec<HashMap<usize, (usize, f32)>>,
    /// Agent names (indexed by agent_idx)
    agent_names: Vec<String>,
}

impl AgentStatsBuilder {
    /// Create a new builder with the given agent names
    pub fn new(agent_names: Vec<String>) -> Self {
        let num_agents = agent_names.len();
        let agent_accumulated = (0..num_agents)
            .map(|_| StatsStorage::new_with_num_players(1))
            .collect();
        let position_tracking = (0..num_agents).map(|_| HashMap::new()).collect();

        Self {
            agent_accumulated,
            position_tracking,
            agent_names,
        }
    }

    /// Merge statistics from a single permutation into the builder
    ///
    /// # Arguments
    /// * `permutation` - Vector of agent indices representing seat assignments
    /// * `stats` - StatsStorage containing the results from this game
    pub fn merge_permutation_stats(&mut self, permutation: &[usize], stats: &StatsStorage) {
        for (seat_idx, &agent_idx) in permutation.iter().enumerate() {
            let agent_stats = &mut self.agent_accumulated[agent_idx];
            let player_idx = 0; // We aggregate all stats to index 0 for each agent

            // Manually accumulate the stats from seat_idx to our single-player storage at idx 0
            agent_stats.actions_count[player_idx] += stats.actions_count[seat_idx];
            agent_stats.vpip_count[player_idx] += stats.vpip_count[seat_idx];
            agent_stats.vpip_total[player_idx] += stats.vpip_total[seat_idx];
            agent_stats.raise_count[player_idx] += stats.raise_count[seat_idx];

            // Correct per-hand tracking (VPIP and PFR - preflop only, binary per hand)
            agent_stats.hands_played[player_idx] += stats.hands_played[seat_idx];
            agent_stats.hands_vpip[player_idx] += stats.hands_vpip[seat_idx];
            agent_stats.hands_pfr[player_idx] += stats.hands_pfr[seat_idx];

            // Per-action counts
            agent_stats.preflop_raise_count[player_idx] += stats.preflop_raise_count[seat_idx];
            agent_stats.preflop_actions[player_idx] += stats.preflop_actions[seat_idx];
            agent_stats.three_bet_count[player_idx] += stats.three_bet_count[seat_idx];
            agent_stats.three_bet_opportunities[player_idx] +=
                stats.three_bet_opportunities[seat_idx];
            agent_stats.call_count[player_idx] += stats.call_count[seat_idx];
            agent_stats.bet_count[player_idx] += stats.bet_count[seat_idx];

            // Financial tracking
            agent_stats.total_profit[player_idx] += stats.total_profit[seat_idx];
            agent_stats.total_invested[player_idx] += stats.total_invested[seat_idx];
            agent_stats.games_won[player_idx] += stats.games_won[seat_idx];
            agent_stats.games_lost[player_idx] += stats.games_lost[seat_idx];
            agent_stats.games_breakeven[player_idx] += stats.games_breakeven[seat_idx];

            // Round outcomes
            agent_stats.preflop_wins[player_idx] += stats.preflop_wins[seat_idx];
            agent_stats.flop_wins[player_idx] += stats.flop_wins[seat_idx];
            agent_stats.turn_wins[player_idx] += stats.turn_wins[seat_idx];
            agent_stats.river_wins[player_idx] += stats.river_wins[seat_idx];
            agent_stats.preflop_completes[player_idx] += stats.preflop_completes[seat_idx];
            agent_stats.flop_completes[player_idx] += stats.flop_completes[seat_idx];
            agent_stats.turn_completes[player_idx] += stats.turn_completes[seat_idx];
            agent_stats.river_completes[player_idx] += stats.river_completes[seat_idx];

            // Advanced Stats
            agent_stats.cbet_opportunities[player_idx] += stats.cbet_opportunities[seat_idx];
            agent_stats.cbet_count[player_idx] += stats.cbet_count[seat_idx];
            agent_stats.wtsd_opportunities[player_idx] += stats.wtsd_opportunities[seat_idx];
            agent_stats.wtsd_count[player_idx] += stats.wtsd_count[seat_idx];
            agent_stats.showdown_count[player_idx] += stats.showdown_count[seat_idx];
            agent_stats.showdown_wins[player_idx] += stats.showdown_wins[seat_idx];
            agent_stats.fold_count[player_idx] += stats.fold_count[seat_idx];
            agent_stats.flop_bets[player_idx] += stats.flop_bets[seat_idx];
            agent_stats.flop_raises[player_idx] += stats.flop_raises[seat_idx];
            agent_stats.flop_calls[player_idx] += stats.flop_calls[seat_idx];
            agent_stats.turn_bets[player_idx] += stats.turn_bets[seat_idx];
            agent_stats.turn_raises[player_idx] += stats.turn_raises[seat_idx];
            agent_stats.turn_calls[player_idx] += stats.turn_calls[seat_idx];
            agent_stats.river_bets[player_idx] += stats.river_bets[seat_idx];
            agent_stats.river_raises[player_idx] += stats.river_raises[seat_idx];
            agent_stats.river_calls[player_idx] += stats.river_calls[seat_idx];
            agent_stats.steal_opportunities[player_idx] += stats.steal_opportunities[seat_idx];
            agent_stats.steal_count[player_idx] += stats.steal_count[seat_idx];

            // Track position-specific stats
            let pos_map = &mut self.position_tracking[agent_idx];
            let (games, profit) = pos_map.entry(seat_idx).or_insert((0, 0.0));
            *games += 1;
            *profit += stats.total_profit[seat_idx];
        }
    }

    /// Build and consume the builder, returning a map of agent stats
    pub fn build(self) -> HashMap<String, AgentStats> {
        let mut agent_stats = HashMap::new();

        for (agent_idx, agent_name) in self.agent_names.iter().enumerate() {
            let stats = &self.agent_accumulated[agent_idx];
            let player_idx = 0;

            // Calculate position stats
            let mut position_stats = Vec::new();
            if let Some(pos_map) = self.position_tracking.get(agent_idx) {
                for (seat_idx, (games_played, total_profit)) in pos_map {
                    let profit_per_game = if *games_played > 0 {
                        total_profit / *games_played as f32
                    } else {
                        0.0
                    };

                    position_stats.push(PositionStats {
                        seat_index: *seat_idx,
                        games_played: *games_played,
                        profit: *total_profit,
                        profit_per_game,
                    });
                }
            }

            // Sort position stats by seat index
            position_stats.sort_by_key(|ps| ps.seat_index);

            let total_games = stats.games_won[player_idx]
                + stats.games_lost[player_idx]
                + stats.games_breakeven[player_idx];

            // Calculate all derived metrics
            let agent_stat = AgentStats {
                agent_name: agent_name.clone(),
                total_profit: stats.total_profit[player_idx],
                total_games,
                wins: stats.games_won[player_idx],
                losses: stats.games_lost[player_idx],
                breakeven: stats.games_breakeven[player_idx],
                profit_per_game: stats.profit_per_game(player_idx),
                profit_per_100_hands: stats.profit_per_game(player_idx) * 100.0,
                roi_percent: stats.roi_percent(player_idx),
                position_stats,
                vpip_percent: stats.vpip_percent(player_idx),
                pfr_percent: stats.pfr_percent(player_idx),
                three_bet_percent: stats.three_bet_percent(player_idx),
                aggression_factor: stats.aggression_factor(player_idx),
                cbet_percent: stats.cbet_percent(player_idx),
                wtsd_percent: stats.wtsd_percent(player_idx),
                wsd_percent: stats.wsd_percent(player_idx),
                steal_percent: stats.steal_percent(player_idx),
                aggression_frequency: stats.aggression_frequency(player_idx),
                flop_aggression_factor: stats.flop_aggression_factor(player_idx),
                turn_aggression_factor: stats.turn_aggression_factor(player_idx),
                river_aggression_factor: stats.river_aggression_factor(player_idx),
                preflop_win_rate: stats.preflop_win_rate(player_idx),
                flop_win_rate: stats.flop_win_rate(player_idx),
                turn_win_rate: stats.turn_win_rate(player_idx),
                river_win_rate: stats.river_win_rate(player_idx),
            };

            agent_stats.insert(agent_name.clone(), agent_stat);
        }

        agent_stats
    }

    /// Get the agent names
    pub fn agent_names(&self) -> &[String] {
        &self.agent_names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_stats_builder_new() {
        let names = vec!["Agent1".to_string(), "Agent2".to_string()];
        let builder = AgentStatsBuilder::new(names.clone());

        assert_eq!(builder.agent_names(), &names);
        assert_eq!(builder.agent_accumulated.len(), 2);
        assert_eq!(builder.position_tracking.len(), 2);
    }

    #[test]
    fn test_agent_stats_builder_build_empty() {
        let names = vec!["Agent1".to_string()];
        let builder = AgentStatsBuilder::new(names);
        let stats = builder.build();

        assert_eq!(stats.len(), 1);
        let agent_stats = stats.get("Agent1").unwrap();
        assert_eq!(agent_stats.agent_name, "Agent1");
        assert_eq!(agent_stats.total_games, 0);
        assert_eq!(agent_stats.total_profit, 0.0);
    }

    #[test]
    fn test_merge_permutation_stats() {
        let names = vec!["Agent0".to_string(), "Agent1".to_string()];
        let mut builder = AgentStatsBuilder::new(names);

        // Create mock stats for a 2-player game
        let mut stats = StatsStorage::new_with_num_players(2);
        stats.total_profit[0] = 50.0; // Seat 0 won 50
        stats.total_profit[1] = -50.0; // Seat 1 lost 50
        stats.total_invested[0] = 100.0;
        stats.total_invested[1] = 100.0;
        stats.games_won[0] = 1;
        stats.games_lost[1] = 1;
        stats.hands_played[0] = 1;
        stats.hands_played[1] = 1;

        // Permutation: Agent0 at seat 0, Agent1 at seat 1
        builder.merge_permutation_stats(&[0, 1], &stats);

        let result = builder.build();

        let agent0 = result.get("Agent0").unwrap();
        assert_eq!(agent0.total_profit, 50.0);
        assert_eq!(agent0.wins, 1);
        assert_eq!(agent0.losses, 0);

        let agent1 = result.get("Agent1").unwrap();
        assert_eq!(agent1.total_profit, -50.0);
        assert_eq!(agent1.wins, 0);
        assert_eq!(agent1.losses, 1);
    }

    #[test]
    fn test_merge_permutation_stats_multiple_games() {
        let names = vec!["Agent0".to_string(), "Agent1".to_string()];
        let mut builder = AgentStatsBuilder::new(names);

        // Game 1: Agent0 at seat 0, Agent1 at seat 1
        let mut stats1 = StatsStorage::new_with_num_players(2);
        stats1.total_profit[0] = 50.0;
        stats1.total_profit[1] = -50.0;
        stats1.total_invested[0] = 100.0;
        stats1.total_invested[1] = 100.0;
        stats1.games_won[0] = 1;
        stats1.games_lost[1] = 1;
        builder.merge_permutation_stats(&[0, 1], &stats1);

        // Game 2: Agent1 at seat 0, Agent0 at seat 1 (swapped positions)
        let mut stats2 = StatsStorage::new_with_num_players(2);
        stats2.total_profit[0] = 30.0; // Agent1 at seat 0 won
        stats2.total_profit[1] = -30.0; // Agent0 at seat 1 lost
        stats2.total_invested[0] = 80.0;
        stats2.total_invested[1] = 80.0;
        stats2.games_won[0] = 1;
        stats2.games_lost[1] = 1;
        builder.merge_permutation_stats(&[1, 0], &stats2);

        let result = builder.build();

        // Agent0: won 50 in game 1, lost 30 in game 2 = net +20
        let agent0 = result.get("Agent0").unwrap();
        assert_eq!(agent0.total_profit, 20.0);
        assert_eq!(agent0.wins, 1);
        assert_eq!(agent0.losses, 1);

        // Agent1: lost 50 in game 1, won 30 in game 2 = net -20
        let agent1 = result.get("Agent1").unwrap();
        assert_eq!(agent1.total_profit, -20.0);
        assert_eq!(agent1.wins, 1);
        assert_eq!(agent1.losses, 1);
    }

    #[test]
    fn test_position_stats_tracking() {
        let names = vec!["Agent0".to_string(), "Agent1".to_string()];
        let mut builder = AgentStatsBuilder::new(names);

        // Agent0 plays at seat 0
        let mut stats1 = StatsStorage::new_with_num_players(2);
        stats1.total_profit[0] = 100.0;
        stats1.games_won[0] = 1;
        builder.merge_permutation_stats(&[0, 1], &stats1);

        // Agent0 plays at seat 1
        let mut stats2 = StatsStorage::new_with_num_players(2);
        stats2.total_profit[1] = -20.0;
        stats2.games_lost[1] = 1;
        builder.merge_permutation_stats(&[1, 0], &stats2);

        let result = builder.build();
        let agent0 = result.get("Agent0").unwrap();

        // Check position stats
        assert_eq!(agent0.position_stats.len(), 2);

        let seat0_stats = agent0
            .position_stats
            .iter()
            .find(|p| p.seat_index == 0)
            .unwrap();
        assert_eq!(seat0_stats.games_played, 1);
        assert_eq!(seat0_stats.profit, 100.0);

        let seat1_stats = agent0
            .position_stats
            .iter()
            .find(|p| p.seat_index == 1)
            .unwrap();
        assert_eq!(seat1_stats.games_played, 1);
        assert_eq!(seat1_stats.profit, -20.0);
    }

    #[test]
    fn test_roi_calculation_in_agent_stats() {
        let names = vec!["Agent0".to_string(), "Agent1".to_string()];
        let mut builder = AgentStatsBuilder::new(names);

        let mut stats = StatsStorage::new_with_num_players(2);
        stats.total_profit[0] = 50.0;
        stats.total_invested[0] = 200.0;
        stats.games_won[0] = 1;
        builder.merge_permutation_stats(&[0, 1], &stats);

        let result = builder.build();
        let agent0 = result.get("Agent0").unwrap();

        // ROI = (50 / 200) * 100 = 25%
        assert!((agent0.roi_percent - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_agent_stats_serialization() {
        let stats = AgentStats {
            agent_name: "Test".to_string(),
            total_profit: 100.0,
            total_games: 10,
            wins: 6,
            losses: 3,
            breakeven: 1,
            profit_per_game: 10.0,
            profit_per_100_hands: 1000.0,
            roi_percent: 25.0,
            position_stats: vec![PositionStats {
                seat_index: 0,
                games_played: 10,
                profit: 100.0,
                profit_per_game: 10.0,
            }],
            vpip_percent: 30.0,
            pfr_percent: 20.0,
            three_bet_percent: 5.0,
            aggression_factor: 2.0,
            cbet_percent: 60.0,
            wtsd_percent: 25.0,
            wsd_percent: 50.0,
            steal_percent: 30.0,
            aggression_frequency: 40.0,
            flop_aggression_factor: 2.5,
            turn_aggression_factor: 2.0,
            river_aggression_factor: 1.5,
            preflop_win_rate: 20.0,
            flop_win_rate: 30.0,
            turn_win_rate: 35.0,
            river_win_rate: 40.0,
        };

        // Test that it can be serialized to JSON
        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("\"agent_name\":\"Test\""));
        assert!(json.contains("\"total_profit\":100.0"));
        assert!(json.contains("\"roi_percent\":25.0"));

        // Test that it can be deserialized back
        let deserialized: AgentStats = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.agent_name, "Test");
        assert_eq!(deserialized.total_profit, 100.0);
        assert_eq!(deserialized.roi_percent, 25.0);
    }

    /// Verifies that raise counts are correctly accumulated when merging permutation stats.
    #[test]
    fn test_merge_permutation_stats_arithmetic_raise_count() {
        let names = vec!["Agent0".to_string()];
        let mut builder = AgentStatsBuilder::new(names);

        // Pre-populate with initial values
        builder.agent_accumulated[0].raise_count[0] = 10;

        // Merge stats with raise_count = 5
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.raise_count[0] = 5;
        builder.merge_permutation_stats(&[0], &stats);

        assert_eq!(
            builder.agent_accumulated[0].raise_count[0], 15,
            "raise_count should be 10 + 5 = 15"
        );
    }

    /// Verifies that preflop action counts are correctly accumulated when merging permutation stats.
    #[test]
    fn test_merge_permutation_stats_arithmetic_preflop_actions() {
        let names = vec!["Agent0".to_string()];
        let mut builder = AgentStatsBuilder::new(names);

        builder.agent_accumulated[0].preflop_actions[0] = 8;

        let mut stats = StatsStorage::new_with_num_players(1);
        stats.preflop_actions[0] = 4;
        builder.merge_permutation_stats(&[0], &stats);

        assert_eq!(
            builder.agent_accumulated[0].preflop_actions[0], 12,
            "preflop_actions should be 8 + 4 = 12"
        );
    }

    /// Verifies that 3-bet counts and opportunities are correctly accumulated when merging permutation stats.
    #[test]
    fn test_merge_permutation_stats_arithmetic_three_bet() {
        let names = vec!["Agent0".to_string()];
        let mut builder = AgentStatsBuilder::new(names);

        builder.agent_accumulated[0].three_bet_count[0] = 3;
        builder.agent_accumulated[0].three_bet_opportunities[0] = 10;

        let mut stats = StatsStorage::new_with_num_players(1);
        stats.three_bet_count[0] = 2;
        stats.three_bet_opportunities[0] = 5;
        builder.merge_permutation_stats(&[0], &stats);

        assert_eq!(
            builder.agent_accumulated[0].three_bet_count[0], 5,
            "three_bet_count should be 3 + 2 = 5"
        );
        assert_eq!(
            builder.agent_accumulated[0].three_bet_opportunities[0], 15,
            "three_bet_opportunities should be 10 + 5 = 15"
        );
    }

    /// Verifies that call and bet counts are correctly accumulated when merging permutation stats.
    #[test]
    fn test_merge_permutation_stats_arithmetic_call_bet_count() {
        let names = vec!["Agent0".to_string()];
        let mut builder = AgentStatsBuilder::new(names);

        builder.agent_accumulated[0].call_count[0] = 7;
        builder.agent_accumulated[0].bet_count[0] = 6;

        let mut stats = StatsStorage::new_with_num_players(1);
        stats.call_count[0] = 3;
        stats.bet_count[0] = 4;
        builder.merge_permutation_stats(&[0], &stats);

        assert_eq!(
            builder.agent_accumulated[0].call_count[0], 10,
            "call_count should be 7 + 3 = 10"
        );
        assert_eq!(
            builder.agent_accumulated[0].bet_count[0], 10,
            "bet_count should be 6 + 4 = 10"
        );
    }

    /// Verifies that round win counts (preflop, flop, turn, river) are correctly accumulated
    /// when merging permutation stats.
    #[test]
    fn test_merge_permutation_stats_arithmetic_round_wins() {
        let names = vec!["Agent0".to_string()];
        let mut builder = AgentStatsBuilder::new(names);

        builder.agent_accumulated[0].preflop_wins[0] = 2;
        builder.agent_accumulated[0].flop_wins[0] = 3;
        builder.agent_accumulated[0].turn_wins[0] = 4;
        builder.agent_accumulated[0].river_wins[0] = 5;

        let mut stats = StatsStorage::new_with_num_players(1);
        stats.preflop_wins[0] = 1;
        stats.flop_wins[0] = 2;
        stats.turn_wins[0] = 3;
        stats.river_wins[0] = 4;
        builder.merge_permutation_stats(&[0], &stats);

        assert_eq!(
            builder.agent_accumulated[0].preflop_wins[0], 3,
            "preflop_wins should be 2 + 1 = 3"
        );
        assert_eq!(
            builder.agent_accumulated[0].flop_wins[0], 5,
            "flop_wins should be 3 + 2 = 5"
        );
        assert_eq!(
            builder.agent_accumulated[0].turn_wins[0], 7,
            "turn_wins should be 4 + 3 = 7"
        );
        assert_eq!(
            builder.agent_accumulated[0].river_wins[0], 9,
            "river_wins should be 5 + 4 = 9"
        );
    }

    /// Verifies that round completion counts (preflop, flop, turn, river) are correctly accumulated
    /// when merging permutation stats.
    #[test]
    fn test_merge_permutation_stats_arithmetic_round_completes() {
        let names = vec!["Agent0".to_string()];
        let mut builder = AgentStatsBuilder::new(names);

        builder.agent_accumulated[0].preflop_completes[0] = 10;
        builder.agent_accumulated[0].flop_completes[0] = 8;
        builder.agent_accumulated[0].turn_completes[0] = 6;
        builder.agent_accumulated[0].river_completes[0] = 4;

        let mut stats = StatsStorage::new_with_num_players(1);
        stats.preflop_completes[0] = 5;
        stats.flop_completes[0] = 4;
        stats.turn_completes[0] = 3;
        stats.river_completes[0] = 2;
        builder.merge_permutation_stats(&[0], &stats);

        assert_eq!(
            builder.agent_accumulated[0].preflop_completes[0], 15,
            "preflop_completes should be 10 + 5 = 15"
        );
        assert_eq!(
            builder.agent_accumulated[0].flop_completes[0], 12,
            "flop_completes should be 8 + 4 = 12"
        );
        assert_eq!(
            builder.agent_accumulated[0].turn_completes[0], 9,
            "turn_completes should be 6 + 3 = 9"
        );
        assert_eq!(
            builder.agent_accumulated[0].river_completes[0], 6,
            "river_completes should be 4 + 2 = 6"
        );
    }

    /// Verifies that advanced poker statistics (c-bet, WTSD, showdown, fold counts) are correctly
    /// accumulated when merging permutation stats.
    #[test]
    fn test_merge_permutation_stats_arithmetic_advanced_stats() {
        let names = vec!["Agent0".to_string()];
        let mut builder = AgentStatsBuilder::new(names);

        builder.agent_accumulated[0].cbet_opportunities[0] = 5;
        builder.agent_accumulated[0].cbet_count[0] = 3;
        builder.agent_accumulated[0].wtsd_opportunities[0] = 10;
        builder.agent_accumulated[0].wtsd_count[0] = 4;
        builder.agent_accumulated[0].showdown_count[0] = 8;
        builder.agent_accumulated[0].showdown_wins[0] = 5;
        builder.agent_accumulated[0].fold_count[0] = 12;

        let mut stats = StatsStorage::new_with_num_players(1);
        stats.cbet_opportunities[0] = 3;
        stats.cbet_count[0] = 2;
        stats.wtsd_opportunities[0] = 5;
        stats.wtsd_count[0] = 2;
        stats.showdown_count[0] = 3;
        stats.showdown_wins[0] = 2;
        stats.fold_count[0] = 4;
        builder.merge_permutation_stats(&[0], &stats);

        assert_eq!(
            builder.agent_accumulated[0].cbet_opportunities[0], 8,
            "cbet_opportunities should be 5 + 3 = 8"
        );
        assert_eq!(
            builder.agent_accumulated[0].cbet_count[0], 5,
            "cbet_count should be 3 + 2 = 5"
        );
        assert_eq!(
            builder.agent_accumulated[0].wtsd_opportunities[0], 15,
            "wtsd_opportunities should be 10 + 5 = 15"
        );
        assert_eq!(
            builder.agent_accumulated[0].wtsd_count[0], 6,
            "wtsd_count should be 4 + 2 = 6"
        );
        assert_eq!(
            builder.agent_accumulated[0].showdown_count[0], 11,
            "showdown_count should be 8 + 3 = 11"
        );
        assert_eq!(
            builder.agent_accumulated[0].showdown_wins[0], 7,
            "showdown_wins should be 5 + 2 = 7"
        );
        assert_eq!(
            builder.agent_accumulated[0].fold_count[0], 16,
            "fold_count should be 12 + 4 = 16"
        );
    }

    /// Verifies that per-street action counts (bets, raises, calls for flop, turn, river) are
    /// correctly accumulated when merging permutation stats.
    #[test]
    fn test_merge_permutation_stats_arithmetic_per_street_actions() {
        let names = vec!["Agent0".to_string()];
        let mut builder = AgentStatsBuilder::new(names);

        builder.agent_accumulated[0].flop_bets[0] = 3;
        builder.agent_accumulated[0].flop_raises[0] = 2;
        builder.agent_accumulated[0].flop_calls[0] = 5;
        builder.agent_accumulated[0].turn_bets[0] = 4;
        builder.agent_accumulated[0].turn_raises[0] = 3;
        builder.agent_accumulated[0].turn_calls[0] = 6;
        builder.agent_accumulated[0].river_bets[0] = 2;
        builder.agent_accumulated[0].river_raises[0] = 1;
        builder.agent_accumulated[0].river_calls[0] = 4;

        let mut stats = StatsStorage::new_with_num_players(1);
        stats.flop_bets[0] = 2;
        stats.flop_raises[0] = 1;
        stats.flop_calls[0] = 3;
        stats.turn_bets[0] = 2;
        stats.turn_raises[0] = 2;
        stats.turn_calls[0] = 4;
        stats.river_bets[0] = 1;
        stats.river_raises[0] = 1;
        stats.river_calls[0] = 2;
        builder.merge_permutation_stats(&[0], &stats);

        assert_eq!(
            builder.agent_accumulated[0].flop_bets[0], 5,
            "flop_bets should be 3 + 2 = 5"
        );
        assert_eq!(
            builder.agent_accumulated[0].flop_raises[0], 3,
            "flop_raises should be 2 + 1 = 3"
        );
        assert_eq!(
            builder.agent_accumulated[0].flop_calls[0], 8,
            "flop_calls should be 5 + 3 = 8"
        );
        assert_eq!(
            builder.agent_accumulated[0].turn_bets[0], 6,
            "turn_bets should be 4 + 2 = 6"
        );
        assert_eq!(
            builder.agent_accumulated[0].turn_raises[0], 5,
            "turn_raises should be 3 + 2 = 5"
        );
        assert_eq!(
            builder.agent_accumulated[0].turn_calls[0], 10,
            "turn_calls should be 6 + 4 = 10"
        );
        assert_eq!(
            builder.agent_accumulated[0].river_bets[0], 3,
            "river_bets should be 2 + 1 = 3"
        );
        assert_eq!(
            builder.agent_accumulated[0].river_raises[0], 2,
            "river_raises should be 1 + 1 = 2"
        );
        assert_eq!(
            builder.agent_accumulated[0].river_calls[0], 6,
            "river_calls should be 4 + 2 = 6"
        );
    }

    /// Verifies that steal opportunity and steal count statistics are correctly accumulated
    /// when merging permutation stats.
    #[test]
    fn test_merge_permutation_stats_arithmetic_steal_stats() {
        let names = vec!["Agent0".to_string()];
        let mut builder = AgentStatsBuilder::new(names);

        builder.agent_accumulated[0].steal_opportunities[0] = 10;
        builder.agent_accumulated[0].steal_count[0] = 6;

        let mut stats = StatsStorage::new_with_num_players(1);
        stats.steal_opportunities[0] = 5;
        stats.steal_count[0] = 3;
        builder.merge_permutation_stats(&[0], &stats);

        assert_eq!(
            builder.agent_accumulated[0].steal_opportunities[0], 15,
            "steal_opportunities should be 10 + 5 = 15"
        );
        assert_eq!(
            builder.agent_accumulated[0].steal_count[0], 9,
            "steal_count should be 6 + 3 = 9"
        );
    }

    /// Verifies that total_games is correctly calculated as the sum of wins, losses, and breakeven games.
    #[test]
    fn test_build_total_games_addition() {
        let names = vec!["Agent0".to_string()];
        let mut builder = AgentStatsBuilder::new(names);

        let mut stats = StatsStorage::new_with_num_players(1);
        stats.games_won[0] = 5;
        stats.games_lost[0] = 3;
        stats.games_breakeven[0] = 2;
        builder.merge_permutation_stats(&[0], &stats);

        let result = builder.build();
        let agent0 = result.get("Agent0").unwrap();

        assert_eq!(
            agent0.total_games, 10,
            "total_games should be 5 + 3 + 2 = 10"
        );
    }

    /// Verifies that profit_per_100_hands is correctly calculated as profit_per_game multiplied by 100.
    #[test]
    fn test_build_profit_per_100_hands_multiplication() {
        let names = vec!["Agent0".to_string()];
        let mut builder = AgentStatsBuilder::new(names);

        let mut stats = StatsStorage::new_with_num_players(1);
        stats.total_profit[0] = 50.0;
        stats.games_won[0] = 3;
        stats.games_lost[0] = 2;
        stats.games_breakeven[0] = 0;
        builder.merge_permutation_stats(&[0], &stats);

        let result = builder.build();
        let agent0 = result.get("Agent0").unwrap();

        assert!(
            (agent0.profit_per_100_hands - 1000.0).abs() < 0.01,
            "profit_per_100_hands should be 10.0 * 100 = 1000.0, got {}",
            agent0.profit_per_100_hands
        );
    }

    /// Verifies that position profit_per_game is correctly calculated as total_profit divided by games_played.
    #[test]
    fn test_position_stats_profit_per_game_division() {
        let names = vec!["Agent0".to_string()];
        let mut builder = AgentStatsBuilder::new(names);

        // Play multiple games at seat 0
        let mut stats1 = StatsStorage::new_with_num_players(1);
        stats1.total_profit[0] = 30.0;
        stats1.games_won[0] = 1;
        builder.merge_permutation_stats(&[0], &stats1);

        let mut stats2 = StatsStorage::new_with_num_players(1);
        stats2.total_profit[0] = 20.0;
        stats2.games_won[0] = 1;
        builder.merge_permutation_stats(&[0], &stats2);

        let result = builder.build();
        let agent0 = result.get("Agent0").unwrap();

        // 2 games at seat 0 with total profit 50.0
        // profit_per_game = 50.0 / 2 = 25.0
        let seat0 = agent0
            .position_stats
            .iter()
            .find(|p| p.seat_index == 0)
            .unwrap();
        assert_eq!(seat0.games_played, 2);
        assert!((seat0.profit - 50.0).abs() < 0.01);
        assert!(
            (seat0.profit_per_game - 25.0).abs() < 0.01,
            "profit_per_game should be 50.0 / 2 = 25.0, got {}",
            seat0.profit_per_game
        );
    }

    /// Verifies that position_stats handles the edge case of zero games played correctly.
    #[test]
    fn test_position_stats_games_greater_than_zero_check() {
        let names = vec!["Agent0".to_string()];
        let builder = AgentStatsBuilder::new(names);

        // Build without any games played
        let result = builder.build();
        let agent0 = result.get("Agent0").unwrap();

        // With no position tracking, position_stats should be empty
        assert!(agent0.position_stats.is_empty());
    }
}
