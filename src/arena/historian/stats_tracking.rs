use std::{cell::RefCell, collections::HashMap, rc::Rc};

use super::Historian;

use crate::arena::GameState;
use crate::arena::action::{Action, AgentAction, AwardPayload, PlayedActionPayload};
use crate::arena::game_state::Round;

/// Storage for tracking various poker player statistics
///
/// # Fields
///
/// * `actions_count` - Vector storing the count of total actions performed by
///   each player
/// * `vpip_count` - Vector storing the count of voluntary put in pot (VPIP)
///   actions for each player
/// * `vpip_total` - Vector storing the running total of VPIP percentage for
///   each player
/// * `raise_count` - Vector storing the count of raise actions performed by
///   each player
#[derive(Clone)]
pub struct StatsStorage {
    num_players: usize,
    // The total number of actions each player has taken
    pub actions_count: Vec<usize>,
    // How many times each player has voluntarily put money in the pot
    pub vpip_count: Vec<usize>,
    // The total amount of money each player has voluntarily put in the pot
    pub vpip_total: Vec<f32>,
    // How many times they raised
    pub raise_count: Vec<usize>,

    // CRITICAL: For percentage stats, we store COUNTS (opportunities + occurrences)
    // This allows proper merging/aggregation across multiple games
    pub preflop_raise_count: Vec<usize>,     // PFR occurrences
    pub preflop_actions: Vec<usize>,         // Times player had action preflop (PFR opportunities)
    pub three_bet_count: Vec<usize>,         // 3-bet occurrences
    pub three_bet_opportunities: Vec<usize>, // Times player faced a raise and could 3-bet
    pub call_count: Vec<usize>,              // Call occurrences (for aggression factor)
    pub bet_count: Vec<usize>,               // Bet occurrences (for aggression factor)

    // Financial tracking
    pub total_profit: Vec<f32>,      // Cumulative profit/loss
    pub games_won: Vec<usize>,       // Win count
    pub games_lost: Vec<usize>,      // Loss count
    pub games_breakeven: Vec<usize>, // Breakeven count

    // Position tracking
    pub position_games: Vec<HashMap<usize, usize>>, // Games played per position
    pub position_profit: Vec<HashMap<usize, f32>>,  // Profit per position

    // Round outcomes (counts for win rate calculation)
    pub preflop_wins: Vec<usize>,      // Hands won that ended preflop
    pub flop_wins: Vec<usize>,         // Hands won that ended on flop
    pub turn_wins: Vec<usize>,         // Hands won that ended on turn
    pub river_wins: Vec<usize>,        // Hands won that ended on river
    pub preflop_completes: Vec<usize>, // Hands that reached preflop completion
    pub flop_completes: Vec<usize>,    // Hands that reached flop
    pub turn_completes: Vec<usize>,    // Hands that reached turn
    pub river_completes: Vec<usize>,   // Hands that reached river
}

impl StatsStorage {
    pub fn new_with_num_players(num_players: usize) -> Self {
        Self {
            num_players,

            // Existing fields
            actions_count: vec![0; num_players],
            vpip_count: vec![0; num_players],
            vpip_total: vec![0.0; num_players],
            raise_count: vec![0; num_players],

            // New statistics for Phase 1
            preflop_raise_count: vec![0; num_players],
            preflop_actions: vec![0; num_players],
            three_bet_count: vec![0; num_players],
            three_bet_opportunities: vec![0; num_players],
            call_count: vec![0; num_players],
            bet_count: vec![0; num_players],

            // Financial tracking
            total_profit: vec![0.0; num_players],
            games_won: vec![0; num_players],
            games_lost: vec![0; num_players],
            games_breakeven: vec![0; num_players],

            // Position tracking
            position_games: vec![HashMap::new(); num_players],
            position_profit: vec![HashMap::new(); num_players],

            // Round outcomes
            preflop_wins: vec![0; num_players],
            flop_wins: vec![0; num_players],
            turn_wins: vec![0; num_players],
            river_wins: vec![0; num_players],
            preflop_completes: vec![0; num_players],
            flop_completes: vec![0; num_players],
            turn_completes: vec![0; num_players],
            river_completes: vec![0; num_players],
        }
    }

    // Percentage calculations from counts
    /// Calculate VPIP percentage for a player
    pub fn vpip_percent(&self, player_idx: usize) -> f32 {
        let actions = self.actions_count[player_idx];
        if actions == 0 {
            0.0
        } else {
            (self.vpip_count[player_idx] as f32 / actions as f32) * 100.0
        }
    }

    /// Calculate PFR (Pre-Flop Raise) percentage for a player
    pub fn pfr_percent(&self, player_idx: usize) -> f32 {
        let opportunities = self.preflop_actions[player_idx];
        if opportunities == 0 {
            0.0
        } else {
            (self.preflop_raise_count[player_idx] as f32 / opportunities as f32) * 100.0
        }
    }

    /// Calculate 3-bet percentage for a player
    pub fn three_bet_percent(&self, player_idx: usize) -> f32 {
        let opportunities = self.three_bet_opportunities[player_idx];
        if opportunities == 0 {
            0.0
        } else {
            (self.three_bet_count[player_idx] as f32 / opportunities as f32) * 100.0
        }
    }

    /// Calculate aggression factor for a player
    /// (raises + bets) / calls
    pub fn aggression_factor(&self, player_idx: usize) -> f32 {
        let calls = self.call_count[player_idx];
        if calls == 0 {
            // If no calls, return infinite aggression (or a large number)
            // Return 0.0 if also no aggressive actions
            let aggressive_actions = self.raise_count[player_idx] + self.bet_count[player_idx];
            if aggressive_actions == 0 {
                0.0
            } else {
                f32::INFINITY
            }
        } else {
            let aggressive_actions =
                (self.raise_count[player_idx] + self.bet_count[player_idx]) as f32;
            aggressive_actions / calls as f32
        }
    }

    // Financial metrics
    /// Calculate profit per game for a player
    pub fn profit_per_game(&self, player_idx: usize) -> f32 {
        let total_games = self.games_won[player_idx]
            + self.games_lost[player_idx]
            + self.games_breakeven[player_idx];
        if total_games == 0 {
            0.0
        } else {
            self.total_profit[player_idx] / total_games as f32
        }
    }

    /// Calculate ROI percentage for a player
    pub fn roi_percent(&self, player_idx: usize) -> f32 {
        // ROI is typically calculated as: (profit / investment) * 100
        // For poker, we can use starting stack as investment
        // This is a simplified version - may need adjustment based on actual game structure
        let total_games = self.games_won[player_idx]
            + self.games_lost[player_idx]
            + self.games_breakeven[player_idx];
        if total_games == 0 {
            0.0
        } else {
            // Assuming profit_per_game is in BB, ROI is profit per game as percentage
            self.profit_per_game(player_idx) // Already represents profit per game; actual ROI calculation may vary
        }
    }

    /// Calculate win rate for a player
    pub fn win_rate(&self, player_idx: usize) -> f32 {
        let total_games = self.games_won[player_idx]
            + self.games_lost[player_idx]
            + self.games_breakeven[player_idx];
        if total_games == 0 {
            0.0
        } else {
            (self.games_won[player_idx] as f32 / total_games as f32) * 100.0
        }
    }

    // Position accessors
    /// Get position statistics for a player
    pub fn position_stats(&self, player_idx: usize) -> &HashMap<usize, usize> {
        &self.position_games[player_idx]
    }

    /// Get position profit for a player
    pub fn position_profit(&self, player_idx: usize) -> &HashMap<usize, f32> {
        &self.position_profit[player_idx]
    }

    // Round win rates (calculated from counts)
    /// Calculate preflop win rate for a player
    pub fn preflop_win_rate(&self, player_idx: usize) -> f32 {
        let completes = self.preflop_completes[player_idx];
        if completes == 0 {
            0.0
        } else {
            (self.preflop_wins[player_idx] as f32 / completes as f32) * 100.0
        }
    }

    /// Calculate flop win rate for a player
    pub fn flop_win_rate(&self, player_idx: usize) -> f32 {
        let completes = self.flop_completes[player_idx];
        if completes == 0 {
            0.0
        } else {
            (self.flop_wins[player_idx] as f32 / completes as f32) * 100.0
        }
    }

    /// Calculate turn win rate for a player
    pub fn turn_win_rate(&self, player_idx: usize) -> f32 {
        let completes = self.turn_completes[player_idx];
        if completes == 0 {
            0.0
        } else {
            (self.turn_wins[player_idx] as f32 / completes as f32) * 100.0
        }
    }

    /// Calculate river win rate for a player
    pub fn river_win_rate(&self, player_idx: usize) -> f32 {
        let completes = self.river_completes[player_idx];
        if completes == 0 {
            0.0
        } else {
            (self.river_wins[player_idx] as f32 / completes as f32) * 100.0
        }
    }

    /// Merge counts from another StatsStorage
    /// This is critical for aggregating stats across multiple games
    pub fn merge(&mut self, other: &StatsStorage) {
        assert_eq!(
            self.num_players, other.num_players,
            "Cannot merge stats with different number of players"
        );

        for i in 0..self.num_players {
            // Existing fields
            self.actions_count[i] += other.actions_count[i];
            self.vpip_count[i] += other.vpip_count[i];
            self.vpip_total[i] += other.vpip_total[i];
            self.raise_count[i] += other.raise_count[i];

            // New fields
            self.preflop_raise_count[i] += other.preflop_raise_count[i];
            self.preflop_actions[i] += other.preflop_actions[i];
            self.three_bet_count[i] += other.three_bet_count[i];
            self.three_bet_opportunities[i] += other.three_bet_opportunities[i];
            self.call_count[i] += other.call_count[i];
            self.bet_count[i] += other.bet_count[i];

            // Financial tracking
            self.total_profit[i] += other.total_profit[i];
            self.games_won[i] += other.games_won[i];
            self.games_lost[i] += other.games_lost[i];
            self.games_breakeven[i] += other.games_breakeven[i];

            // Position tracking - merge hashmaps
            for (position, count) in &other.position_games[i] {
                *self.position_games[i].entry(*position).or_insert(0) += count;
            }
            for (position, profit) in &other.position_profit[i] {
                *self.position_profit[i].entry(*position).or_insert(0.0) += profit;
            }

            // Round outcomes
            self.preflop_wins[i] += other.preflop_wins[i];
            self.flop_wins[i] += other.flop_wins[i];
            self.turn_wins[i] += other.turn_wins[i];
            self.river_wins[i] += other.river_wins[i];
            self.preflop_completes[i] += other.preflop_completes[i];
            self.flop_completes[i] += other.flop_completes[i];
            self.turn_completes[i] += other.turn_completes[i];
            self.river_completes[i] += other.river_completes[i];
        }
    }
}

impl Default for StatsStorage {
    fn default() -> Self {
        StatsStorage::new_with_num_players(9)
    }
}

/// A historian implementation that tracks and stores poker game statistics
///
/// # Fields
/// * `storage` - A reference-counted, mutable reference to the statistics
///   storage
/// * `dealer_idx` - The dealer position for tracking seat positions
/// * `starting_stacks` - Starting stacks for profit calculation
/// * `current_round` - Track the current round for round-based statistics
/// * `recorded_profit` - Track profit we've already recorded for this game to avoid double-counting
pub struct StatsTrackingHistorian {
    storage: Rc<RefCell<StatsStorage>>,
    dealer_idx: usize,
    starting_stacks: Vec<f32>,
    current_round: Round,
    recorded_profit: Vec<f32>, // Track what profit we've already recorded for this game
}

impl StatsTrackingHistorian {
    pub fn get_storage(&self) -> Rc<RefCell<StatsStorage>> {
        self.storage.clone()
    }

    fn record_played_action(
        &mut self,
        _games_state: &GameState,
        payload: PlayedActionPayload,
    ) -> Result<(), super::HistorianError> {
        let mut storage = self.storage.try_borrow_mut()?;
        storage.actions_count[payload.idx] += 1;

        // Track preflop opportunities
        if payload.round == Round::Preflop {
            storage.preflop_actions[payload.idx] += 1;
        }

        match payload.action {
            AgentAction::Bet(bet_amount) => {
                let put_into_pot = bet_amount - payload.starting_player_bet;

                if put_into_pot > 0.0 {
                    // Played Action Payloads can't come from a forced bet
                    // so if there's a bet amount, it's a voluntary action
                    storage.vpip_count[payload.idx] += 1;
                    // They put in the bet amount minus what they already had in the pot
                    storage.vpip_total[payload.idx] += put_into_pot;

                    // Track bet count (for aggression factor)
                    storage.bet_count[payload.idx] += 1;
                }

                // They raised
                if payload.final_bet > payload.starting_bet {
                    storage.raise_count[payload.idx] += 1;

                    // Track preflop raise
                    if payload.round == Round::Preflop {
                        storage.preflop_raise_count[payload.idx] += 1;

                        // Check if this is a 3-bet (raise after a raise)
                        // total_raise_count in round_data tracks raises
                        // If there was already a raise, this is a 3-bet opportunity
                        // We need to check the starting state to see if there was already a raise
                    }
                }

                // Check for 3-bet opportunity
                // If starting bet > big blind and this is a raise, it's a 3-bet opportunity
                if payload.round == Round::Preflop && payload.starting_bet > 0.0 {
                    storage.three_bet_opportunities[payload.idx] += 1;

                    if payload.final_bet > payload.starting_bet {
                        storage.three_bet_count[payload.idx] += 1;
                    }
                }
            }
            AgentAction::Call => {
                // Track call count (for aggression factor)
                storage.call_count[payload.idx] += 1;

                // Calling is also VPIP
                let put_into_pot = payload.final_player_bet - payload.starting_player_bet;
                if put_into_pot > 0.0 {
                    storage.vpip_count[payload.idx] += 1;
                    storage.vpip_total[payload.idx] += put_into_pot;
                }
            }
            AgentAction::Fold => {
                // Fold doesn't contribute to any stats beyond actions_count
            }
            AgentAction::AllIn => {
                // All-in counts as a bet/raise for aggression purposes
                storage.bet_count[payload.idx] += 1;

                let put_into_pot = payload.final_player_bet - payload.starting_player_bet;
                if put_into_pot > 0.0 {
                    storage.vpip_count[payload.idx] += 1;
                    storage.vpip_total[payload.idx] += put_into_pot;
                }

                // Check if this is a raise
                if payload.final_bet > payload.starting_bet {
                    storage.raise_count[payload.idx] += 1;

                    if payload.round == Round::Preflop {
                        storage.preflop_raise_count[payload.idx] += 1;
                    }
                }
            }
        }

        Ok(())
    }

    fn record_round_advance(
        &mut self,
        round: Round,
        _game_state: &GameState,
    ) -> Result<(), super::HistorianError> {
        self.current_round = round;

        // Track round completions
        let mut storage = self.storage.try_borrow_mut()?;
        let num_players = storage.num_players;

        match round {
            Round::Preflop => {
                for i in 0..num_players {
                    storage.preflop_completes[i] += 1;
                }
            }
            Round::Flop => {
                for i in 0..num_players {
                    storage.flop_completes[i] += 1;
                }
            }
            Round::Turn => {
                for i in 0..num_players {
                    storage.turn_completes[i] += 1;
                }
            }
            Round::River => {
                for i in 0..num_players {
                    storage.river_completes[i] += 1;
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Record final profits for all players when the game completes
    fn record_game_complete(
        &mut self,
        game_state: &GameState,
    ) -> Result<(), super::HistorianError> {
        let mut storage = self.storage.try_borrow_mut()?;

        // Record final profit for each player
        for player_idx in 0..game_state.num_players {
            let final_profit = game_state.player_reward(player_idx);

            // Add this game's profit to total profit
            storage.total_profit[player_idx] += final_profit;

            // Update win/loss/breakeven counts
            if final_profit > 0.01 {
                storage.games_won[player_idx] += 1;

                // Track round-based wins
                match game_state.round_before {
                    Round::Preflop => storage.preflop_wins[player_idx] += 1,
                    Round::Flop => storage.flop_wins[player_idx] += 1,
                    Round::Turn => storage.turn_wins[player_idx] += 1,
                    Round::River => storage.river_wins[player_idx] += 1,
                    _ => {}
                }
            } else if final_profit < -0.01 {
                storage.games_lost[player_idx] += 1;
            } else {
                storage.games_breakeven[player_idx] += 1;
            }

            // Track position statistics
            *storage.position_games[player_idx]
                .entry(player_idx)
                .or_insert(0) += 1;
            *storage.position_profit[player_idx]
                .entry(player_idx)
                .or_insert(0.0) += final_profit;
        }

        Ok(())
    }

    /// Record award information for statistics like round wins, but don't track profit here
    fn record_award_without_profit(
        &mut self,
        _game_state: &GameState,
        _payload: AwardPayload,
    ) -> Result<(), super::HistorianError> {
        // We no longer track profit from awards since we do it at game complete
        // This method exists to preserve any other award-related statistics if needed
        Ok(())
    }

    fn record_game_start(&mut self, game_state: &GameState) -> Result<(), super::HistorianError> {
        // Store starting stacks and dealer position
        self.starting_stacks = game_state.starting_stacks.clone();
        self.dealer_idx = game_state.dealer_idx;
        self.current_round = Round::Starting;

        // Reset recorded profit for this new game
        self.recorded_profit = vec![0.0; game_state.num_players];

        Ok(())
    }

    pub fn new_with_num_players(num_players: usize) -> Self {
        Self {
            storage: Rc::new(RefCell::new(StatsStorage::new_with_num_players(
                num_players,
            ))),
            dealer_idx: 0,
            starting_stacks: vec![0.0; num_players],
            current_round: Round::Starting,
            recorded_profit: vec![0.0; num_players],
        }
    }
}

impl Default for StatsTrackingHistorian {
    fn default() -> Self {
        Self {
            storage: Rc::new(RefCell::new(StatsStorage::default())),
            dealer_idx: 0,
            starting_stacks: vec![0.0; 9],
            current_round: Round::Starting,
            recorded_profit: vec![0.0; 9],
        }
    }
}

impl Historian for StatsTrackingHistorian {
    fn record_action(
        &mut self,
        _id: u128,
        game_state: &GameState,
        action: Action,
    ) -> Result<(), super::HistorianError> {
        match action {
            Action::GameStart(_) => self.record_game_start(game_state),
            Action::PlayedAction(payload) => self.record_played_action(game_state, payload),
            Action::FailedAction(failed_action_payload) => {
                self.record_played_action(game_state, failed_action_payload.result)
            }
            Action::RoundAdvance(round) => {
                if round == Round::Complete {
                    // Record final profits for all players when the game completes
                    self.record_game_complete(game_state)?;
                }
                self.record_round_advance(round, game_state)
            }
            Action::Award(payload) => {
                // Still record awards for other statistics (round wins, etc.)
                // but don't use them for profit calculation
                self.record_award_without_profit(game_state, payload)
            }
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::{
        Agent, HoldemSimulationBuilder,
        agent::{AllInAgent, CallingAgent, FoldingAgent, VecReplayAgent},
    };

    use super::*;

    #[test]
    fn test_all_in_agents_had_actions_counted() {
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();

        let stacks = vec![100.0; 2];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<AllInAgent>::default() as Box<dyn Agent>,
            Box::<AllInAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        assert!(
            storage
                .borrow()
                .actions_count
                .iter()
                .all(|&count| count == 1)
        );
    }

    #[test]
    fn test_calling_agents_had_actions_counted() {
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();

        let stacks = vec![100.0; 2];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<CallingAgent>::default() as Box<dyn Agent>,
            Box::<CallingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);

        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        assert!(
            storage
                .borrow()
                .actions_count
                .iter()
                .all(|&count| count == 4)
        );
    }

    #[test]
    fn test_folding_agents_had_actions_counted() {
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();

        let stacks = vec![100.0; 2];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);

        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let actions_count = &storage.borrow().actions_count;

        // Player 0 folded before player 1 could even act.
        assert_eq!(actions_count.first(), Some(&1));
        assert_eq!(actions_count.get(1), Some(&0));
    }

    #[test]
    fn test_replay_agents_had_raises_counted() {
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();
        let stacks = vec![100.0; 2];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "replay-agent-0",
                vec![AgentAction::Bet(10.0), AgentAction::Bet(40.0)],
                AgentAction::Bet(0.0),
            )) as Box<dyn Agent>,
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "replay-agent-1",
                vec![
                    AgentAction::Bet(10.0),
                    AgentAction::Bet(20.0),
                    AgentAction::Bet(40.0),
                ],
                AgentAction::Bet(0.0),
            )) as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);

        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        assert_eq!(storage.borrow().raise_count, vec![1, 1]);
    }

    #[test]
    fn test_pfr_tracking() {
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();
        let stacks = vec![100.0; 2];

        // Agent 0 raises preflop, Agent 1 calls
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "raiser",
                vec![AgentAction::Bet(20.0)], // Raise preflop
                AgentAction::Call,
            )) as Box<dyn Agent>,
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "caller",
                vec![AgentAction::Call], // Call preflop
                AgentAction::Call,
            )) as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let borrowed = storage.borrow();

        // Agent 0 should have 1 preflop raise
        assert_eq!(borrowed.preflop_raise_count[0], 1);
        assert_eq!(borrowed.preflop_raise_count[1], 0);

        // Both should have preflop opportunities
        assert!(borrowed.preflop_actions[0] > 0);
        assert!(borrowed.preflop_actions[1] > 0);

        // Calculate PFR percentage
        let pfr_0 = borrowed.pfr_percent(0);
        assert!(pfr_0 > 0.0);
    }

    #[test]
    fn test_call_tracking() {
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(3));
        let storage = hist.get_storage();
        let stacks = vec![100.0; 3];

        // Use 3 players with VecReplayAgent to ensure Call actions
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "raiser",
                vec![AgentAction::Bet(20.0)], // Raise
                AgentAction::Call,
            )) as Box<dyn Agent>,
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "caller-1",
                vec![AgentAction::Call], // Call the raise
                AgentAction::Call,
            )) as Box<dyn Agent>,
            Box::<VecReplayAgent>::new(VecReplayAgent::new_with_default(
                "caller-2",
                vec![AgentAction::Call], // Call the raise
                AgentAction::Call,
            )) as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let borrowed = storage.borrow();

        // The callers should have calls recorded
        assert!(
            borrowed.call_count[1] > 0 || borrowed.call_count[2] > 0,
            "Expected at least one caller to have calls tracked, got: {:?}",
            borrowed.call_count
        );
    }

    #[test]
    fn test_vpip_calculation() {
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();
        let stacks = vec![100.0; 2];

        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<CallingAgent>::default() as Box<dyn Agent>,
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let borrowed = storage.borrow();

        // Calling agent should have high VPIP
        let vpip_0 = borrowed.vpip_percent(0);
        assert!(vpip_0 > 0.0);

        // Folding agent should have low VPIP (only forced bets)
        let vpip_1 = borrowed.vpip_percent(1);
        assert_eq!(vpip_1, 0.0); // Folding agent never voluntarily puts money in
    }

    #[test]
    fn test_merge_stats() {
        let mut stats1 = StatsStorage::new_with_num_players(2);
        stats1.actions_count[0] = 5;
        stats1.vpip_count[0] = 3;
        stats1.total_profit[0] = 100.0;
        stats1.games_won[0] = 2;

        let mut stats2 = StatsStorage::new_with_num_players(2);
        stats2.actions_count[0] = 3;
        stats2.vpip_count[0] = 2;
        stats2.total_profit[0] = 50.0;
        stats2.games_won[0] = 1;

        stats1.merge(&stats2);

        assert_eq!(stats1.actions_count[0], 8);
        assert_eq!(stats1.vpip_count[0], 5);
        assert_eq!(stats1.total_profit[0], 150.0);
        assert_eq!(stats1.games_won[0], 3);
    }

    #[test]
    fn test_merge_position_stats() {
        let mut stats1 = StatsStorage::new_with_num_players(2);
        stats1.position_games[0].insert(0, 5);
        stats1.position_profit[0].insert(0, 100.0);

        let mut stats2 = StatsStorage::new_with_num_players(2);
        stats2.position_games[0].insert(0, 3);
        stats2.position_profit[0].insert(0, 50.0);

        stats1.merge(&stats2);

        assert_eq!(stats1.position_games[0].get(&0), Some(&8));
        assert_eq!(stats1.position_profit[0].get(&0), Some(&150.0));
    }

    #[test]
    fn test_aggression_factor_no_calls() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.raise_count[0] = 5;
        stats.bet_count[0] = 3;
        stats.call_count[0] = 0;

        let af = stats.aggression_factor(0);
        assert_eq!(af, f32::INFINITY);
    }

    #[test]
    fn test_aggression_factor_with_calls() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.raise_count[0] = 4;
        stats.bet_count[0] = 2;
        stats.call_count[0] = 2;

        let af = stats.aggression_factor(0);
        assert_eq!(af, 3.0); // (4 + 2) / 2 = 3.0
    }

    #[test]
    fn test_profit_per_game() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.total_profit[0] = 300.0;
        stats.games_won[0] = 5;
        stats.games_lost[0] = 3;
        stats.games_breakeven[0] = 2;

        let ppg = stats.profit_per_game(0);
        assert_eq!(ppg, 30.0); // 300.0 / 10 games = 30.0
    }

    #[test]
    fn test_win_rate() {
        let mut stats = StatsStorage::new_with_num_players(1);
        stats.games_won[0] = 7;
        stats.games_lost[0] = 2;
        stats.games_breakeven[0] = 1;

        let wr = stats.win_rate(0);
        assert_eq!(wr, 70.0); // 7 / 10 * 100 = 70%
    }

    #[test]
    fn test_edge_case_empty_stats() {
        let stats = StatsStorage::new_with_num_players(2);

        // All calculations should return 0 for empty stats
        assert_eq!(stats.vpip_percent(0), 0.0);
        assert_eq!(stats.pfr_percent(0), 0.0);
        assert_eq!(stats.three_bet_percent(0), 0.0);
        assert_eq!(stats.aggression_factor(0), 0.0);
        assert_eq!(stats.profit_per_game(0), 0.0);
        assert_eq!(stats.win_rate(0), 0.0);
    }

    #[test]
    fn test_round_win_rates() {
        let mut stats = StatsStorage::new_with_num_players(1);

        stats.preflop_wins[0] = 2;
        stats.preflop_completes[0] = 10;

        stats.flop_wins[0] = 3;
        stats.flop_completes[0] = 8;

        stats.turn_wins[0] = 1;
        stats.turn_completes[0] = 5;

        stats.river_wins[0] = 2;
        stats.river_completes[0] = 4;

        assert_eq!(stats.preflop_win_rate(0), 20.0);
        assert_eq!(stats.flop_win_rate(0), 37.5);
        assert_eq!(stats.turn_win_rate(0), 20.0);
        assert_eq!(stats.river_win_rate(0), 50.0);
    }

    #[test]
    fn test_zero_sum_property_simple() {
        // Test that profits sum to zero in a simple heads-up game
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();

        let stacks = vec![100.0, 100.0];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<AllInAgent>::default() as Box<dyn Agent>,
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let borrowed = storage.borrow();
        let total_profit = borrowed.total_profit[0] + borrowed.total_profit[1];

        // In a zero-sum game, total profit should be very close to zero
        // Allow for small floating point errors
        assert!(
            total_profit.abs() < 0.01,
            "Total profit should be zero (zero-sum), but got: {}. Player 0: {}, Player 1: {}",
            total_profit,
            borrowed.total_profit[0],
            borrowed.total_profit[1]
        );

        // One player should have positive profit, one should have negative (or both zero)
        let player0_profit = borrowed.total_profit[0];
        let player1_profit = borrowed.total_profit[1];

        // At least one player should have participated (won/lost/breakeven > 0)
        let total_games_0 =
            borrowed.games_won[0] + borrowed.games_lost[0] + borrowed.games_breakeven[0];
        let total_games_1 =
            borrowed.games_won[1] + borrowed.games_lost[1] + borrowed.games_breakeven[1];

        assert!(
            total_games_0 > 0 || total_games_1 > 0,
            "At least one player should have game results"
        );

        println!(
            "Player 0 profit: {}, games: {}",
            player0_profit, total_games_0
        );
        println!(
            "Player 1 profit: {}, games: {}",
            player1_profit, total_games_1
        );
        println!("Total profit: {}", total_profit);
    }

    #[test]
    fn test_zero_sum_property_three_players() {
        // Test zero-sum property with three players
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(3));
        let storage = hist.get_storage();

        let stacks = vec![100.0, 100.0, 100.0];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<AllInAgent>::default() as Box<dyn Agent>,
            Box::<CallingAgent>::default() as Box<dyn Agent>,
            Box::<FoldingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let borrowed = storage.borrow();
        let total_profit =
            borrowed.total_profit[0] + borrowed.total_profit[1] + borrowed.total_profit[2];

        // In a zero-sum game, total profit should be very close to zero
        assert!(
            total_profit.abs() < 0.01,
            "Total profit should be zero (zero-sum), but got: {}. Player profits: [{}, {}, {}]",
            total_profit,
            borrowed.total_profit[0],
            borrowed.total_profit[1],
            borrowed.total_profit[2]
        );

        println!(
            "Player profits: [{}, {}, {}], Total: {}",
            borrowed.total_profit[0],
            borrowed.total_profit[1],
            borrowed.total_profit[2],
            total_profit
        );
    }

    #[test]
    fn test_profit_calculation_matches_game_state() {
        // Test that our profit calculation matches what GameState.player_reward() would return
        let hist = Box::new(StatsTrackingHistorian::new_with_num_players(2));
        let storage = hist.get_storage();

        let stacks = vec![100.0, 100.0];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<AllInAgent>::default() as Box<dyn Agent>,
            Box::<CallingAgent>::default() as Box<dyn Agent>,
        ];

        let game_state = GameState::new_starting(stacks.clone(), 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![hist])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let final_game_state = &sim.game_state;
        let borrowed = storage.borrow();

        // Check that our tracked profit matches GameState.player_reward()
        for i in 0..2 {
            let tracked_profit = borrowed.total_profit[i];
            let actual_reward = final_game_state.player_reward(i);

            assert!(
                (tracked_profit - actual_reward).abs() < 0.01,
                "Player {} tracked profit ({}) should match actual reward ({})",
                i,
                tracked_profit,
                actual_reward
            );
        }
    }
}
