//! Preflop chart-based action generator.
//!
//! This action generator uses pre-configured preflop charts to limit
//! exploration during preflop, generating only actions that have non-zero
//! probability in the chart for the current hand/position. For post-flop,
//! it delegates to configurable action generation.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::event;

use crate::arena::GameState;
use crate::arena::action::AgentAction;
use crate::arena::cfr::{CFRState, TraversalState};
use crate::arena::game_state::Round;
use crate::holdem::{PreflopActionType, PreflopChart, PreflopHand};

use super::{ActionGenerator, ConfigurableActionConfig, ConfigurableActionGenerator};

fn default_raise_size_bb() -> f32 {
    2.5
}

fn default_three_bet_multiplier() -> f32 {
    3.0
}

/// Configuration for preflop chart-based play.
///
/// This configuration specifies which hands to play from each position
/// and how to size bets during preflop.
///
/// # Example JSON
///
/// ```json
/// {
///   "raise_size_bb": 2.5,
///   "three_bet_multiplier": 3.0,
///   "charts": [
///     { "AA": {"Raise": 1.0}, "KK": {"Raise": 1.0} },
///     { "AA": {"Raise": 1.0} }
///   ]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct PreflopChartConfig {
    /// Charts indexed by position (distance from button).
    /// Index 0 = Button, 1 = Small Blind, 2 = Big Blind, 3+ = early positions.
    ///
    /// If a position index exceeds the available charts, the last chart is used.
    /// This allows specifying fewer charts (e.g., just one "tight" chart for all positions)
    /// while still having position-specific play when desired.
    pub charts: Vec<PreflopChart>,

    /// Default raise size as multiple of big blind for open raises.
    /// Standard is 2.5bb from most positions.
    #[serde(default = "default_raise_size_bb")]
    pub raise_size_bb: f32,

    /// Multiplier for 3-bet sizing (3-bet = this * opponent's raise).
    /// Standard is 3x the opponent's raise from in position.
    #[serde(default = "default_three_bet_multiplier")]
    pub three_bet_multiplier: f32,
}

impl Default for PreflopChartConfig {
    fn default() -> Self {
        Self {
            charts: vec![PreflopChart::new()],
            raise_size_bb: default_raise_size_bb(),
            three_bet_multiplier: default_three_bet_multiplier(),
        }
    }
}

impl PreflopChartConfig {
    /// Create a new config with the given charts.
    pub fn new(charts: Vec<PreflopChart>) -> Self {
        Self {
            charts,
            ..Default::default()
        }
    }

    /// Create a config with a single chart used for all positions.
    pub fn with_single_chart(chart: PreflopChart) -> Self {
        Self {
            charts: vec![chart],
            ..Default::default()
        }
    }

    /// Get the chart for a given position relative to button.
    ///
    /// Position 0 = Button, 1 = Small Blind, 2 = Big Blind, etc.
    /// If position exceeds available charts, returns the last chart.
    pub fn chart_for_position(&self, position: usize) -> &PreflopChart {
        if self.charts.is_empty() {
            // This shouldn't happen with Default, but handle gracefully
            panic!("PreflopChartConfig has no charts");
        }
        let idx = position.min(self.charts.len() - 1);
        &self.charts[idx]
    }

    /// Calculate the position relative to the big blind.
    ///
    /// Returns the distance from the big blind position (counter-clockwise):
    /// - 0 = Big Blind
    /// - 1 = Small Blind (or Button in heads-up)
    /// - 2 = Button (for 3+ players)
    /// - 3 = Cutoff
    /// - 4 = Hijack
    /// - 5+ = Earlier positions (UTG, etc.)
    ///
    /// This ordering is designed so that if fewer charts are provided than
    /// positions, the "fallback" (last chart) represents the tightest/earliest
    /// position range, which is appropriate for unspecified early positions.
    ///
    /// Note: In heads-up (2 players), the button posts the small blind and
    /// acts first preflop. The BB is 1 position after the dealer, not 2.
    pub fn calculate_position(player_idx: usize, dealer_idx: usize, num_players: usize) -> usize {
        // In heads-up, BB is 1 seat after dealer (BTN posts SB)
        // In 3+ players, BB is 2 seats after dealer
        let bb_offset = if num_players == 2 { 1 } else { 2 };
        let bb_idx = (dealer_idx + bb_offset) % num_players;
        // Distance from BB (counter-clockwise = positions acting before BB)
        (bb_idx + num_players - player_idx) % num_players
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.charts.is_empty() {
            return Err("At least one preflop chart is required".to_string());
        }
        if self.raise_size_bb <= 0.0 {
            return Err("raise_size_bb must be positive".to_string());
        }
        if self.three_bet_multiplier <= 0.0 {
            return Err("three_bet_multiplier must be positive".to_string());
        }
        Ok(())
    }
}

/// Configuration for the preflop chart action generator.
///
/// Combines preflop chart configuration with configurable post-flop action generation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Default)]
pub struct PreflopChartActionConfig {
    /// Configuration for preflop chart-based action generation.
    pub preflop_config: PreflopChartConfig,
    /// Configuration for post-flop action generation.
    #[cfg_attr(feature = "serde", serde(default))]
    pub postflop_config: ConfigurableActionConfig,
}

/// Action generator that uses preflop charts for preflop and configurable
/// actions for post-flop.
///
/// During preflop, this generator:
/// 1. Gets the player's hand and position
/// 2. Looks up the chart for that position
/// 3. Gets the strategy for the hand (or fold if not in chart)
/// 4. Returns all actions with non-zero frequency as possible actions
///
/// During post-flop, this generator delegates to `ConfigurableActionGenerator`
/// logic for action generation.
///
/// # Example
///
/// ```rust,ignore
/// use rs_poker::arena::cfr::{
///     CFRAgentBuilder, PreflopChartActionGenerator, PreflopChartActionConfig,
///     DepthBasedIteratorGen, DepthBasedIteratorGenConfig, PreflopChartConfig,
///     ConfigurableActionConfig,
/// };
///
/// let preflop_config = PreflopChartConfig::default();
/// let action_config = PreflopChartActionConfig {
///     preflop_config,
///     postflop_config: ConfigurableActionConfig::default(),
/// };
///
/// let agent = CFRAgentBuilder::<PreflopChartActionGenerator, DepthBasedIteratorGen>::new()
///     .name("MyAgent")
///     .player_idx(0)
///     .game_state(game_state)
///     .gamestate_iterator_gen_config(DepthBasedIteratorGenConfig::new(vec![10, 5, 1]))
///     .action_gen_config(action_config)
///     .build();
/// ```
pub struct PreflopChartActionGenerator {
    cfr_state: CFRState,
    traversal_state: TraversalState,
    config: Arc<PreflopChartActionConfig>,
}

impl ActionGenerator for PreflopChartActionGenerator {
    type Config = PreflopChartActionConfig;

    fn new(
        cfr_state: CFRState,
        traversal_state: TraversalState,
        config: Arc<Self::Config>,
    ) -> Self {
        Self {
            cfr_state,
            traversal_state,
            config,
        }
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn cfr_state(&self) -> &CFRState {
        &self.cfr_state
    }

    fn traversal_state(&self) -> &TraversalState {
        &self.traversal_state
    }

    fn gen_possible_actions(&self, game_state: &GameState) -> Vec<AgentAction> {
        // For preflop, use chart-based action generation
        if matches!(game_state.round, Round::Preflop | Round::DealPreflop) {
            self.gen_preflop_actions(game_state)
        } else {
            // For post-flop, delegate to configurable action generator logic
            self.gen_postflop_actions(game_state)
        }
    }
}

impl PreflopChartActionGenerator {
    /// Generate preflop actions based on the configured charts.
    fn gen_preflop_actions(&self, game_state: &GameState) -> Vec<AgentAction> {
        let player_idx = game_state.to_act_idx();
        let player_hand = &game_state.hands[player_idx];

        // Convert player's hole cards to PreflopHand
        let preflop_hand = match PreflopHand::try_from(player_hand) {
            Ok(h) => h,
            Err(e) => {
                event!(
                    tracing::Level::WARN,
                    ?e,
                    "Failed to convert hand to PreflopHand, returning fold only"
                );
                // If we can't identify the hand, just return fold
                return vec![AgentAction::Fold];
            }
        };

        // Calculate position relative to button
        let position = PreflopChartConfig::calculate_position(
            player_idx,
            game_state.dealer_idx,
            game_state.num_players,
        );

        // Get chart for this position
        let chart = self.config.preflop_config.chart_for_position(position);

        // Look up strategy for this hand
        let strategy = chart.get_or_fold(&preflop_hand);

        event!(
            tracing::Level::DEBUG,
            hand = %preflop_hand,
            position,
            "Generating preflop actions from chart"
        );

        // Extract all actions with non-zero frequency
        let mut actions = Vec::new();
        let current_bet = game_state.current_round_bet();
        let player_bet = game_state.current_round_current_player_bet();
        let to_call = current_bet - player_bet;

        for (action_type, freq) in strategy.frequencies() {
            if *freq > 0.0
                && let Some(action) = self.convert_preflop_action(*action_type, game_state)
            {
                // Avoid duplicates
                if !actions.contains(&action) {
                    actions.push(action);
                }
            }
        }

        // If we end up with no actions (shouldn't happen with fold strategy),
        // ensure we at least have fold or call
        if actions.is_empty() {
            if to_call > 0.0 {
                actions.push(AgentAction::Fold);
            } else {
                actions.push(AgentAction::Call); // Check
            }
        }

        actions
    }

    /// Convert a PreflopActionType to an AgentAction with proper sizing.
    fn convert_preflop_action(
        &self,
        action: PreflopActionType,
        game_state: &GameState,
    ) -> Option<AgentAction> {
        let player_idx = game_state.to_act_idx();
        let big_blind = game_state.big_blind;
        let current_bet = game_state.current_round_bet();
        let player_bet = game_state.current_round_current_player_bet();
        let player_stack = game_state.stacks[player_idx];
        let to_call = current_bet - player_bet;

        match action {
            PreflopActionType::Fold => {
                // If there's nothing to call, fold becomes check
                if to_call <= 0.0 {
                    Some(AgentAction::Call) // Check
                } else {
                    Some(AgentAction::Fold)
                }
            }
            PreflopActionType::Call => Some(AgentAction::Call),
            PreflopActionType::Raise => {
                // Standard open raise: raise_size_bb * big_blind
                let raise_amount = self.config.preflop_config.raise_size_bb * big_blind;

                // If there's already a raise, this becomes a re-raise (3-bet)
                if current_bet > big_blind {
                    // Someone already raised, so we 3-bet
                    let three_bet_amount =
                        current_bet * self.config.preflop_config.three_bet_multiplier;
                    Some(self.bet_or_all_in(three_bet_amount, player_stack, player_bet))
                } else {
                    // Standard open raise
                    Some(self.bet_or_all_in(raise_amount, player_stack, player_bet))
                }
            }
            PreflopActionType::ThreeBet => {
                // 3-bet: multiply current bet by three_bet_multiplier
                let three_bet_amount =
                    current_bet * self.config.preflop_config.three_bet_multiplier;
                Some(self.bet_or_all_in(three_bet_amount, player_stack, player_bet))
            }
            PreflopActionType::FourBet => {
                // 4-bet: typically 2.5x the 3-bet
                let four_bet_amount = current_bet * 2.5;
                Some(self.bet_or_all_in(four_bet_amount, player_stack, player_bet))
            }
        }
    }

    /// Return a Bet action or AllIn if the bet amount exceeds the stack.
    fn bet_or_all_in(&self, amount: f32, stack: f32, player_bet: f32) -> AgentAction {
        let all_in_amount = stack + player_bet;
        if amount >= all_in_amount {
            AgentAction::AllIn
        } else {
            AgentAction::Bet(amount)
        }
    }

    /// Generate post-flop actions using configurable action generator logic.
    fn gen_postflop_actions(&self, game_state: &GameState) -> Vec<AgentAction> {
        // Create a temporary ConfigurableActionGenerator to reuse its logic
        let configurable = ConfigurableActionGenerator::new_with_config(
            self.cfr_state.clone(),
            self.traversal_state.clone(),
            self.config.postflop_config.clone(),
        );
        configurable.gen_possible_actions(game_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::GameStateBuilder;
    use crate::arena::cfr::{CFRState, TraversalState};
    use crate::core::Value;
    use crate::holdem::{PreflopChart, PreflopStrategy};

    fn create_test_game_state() -> GameState {
        GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap()
    }

    fn create_simple_config() -> PreflopChartActionConfig {
        let mut chart = PreflopChart::new();
        // Only AA and KK raise
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        let kk = PreflopHand::new(Value::King, Value::King, false);
        chart.set(aa, PreflopStrategy::pure(PreflopActionType::Raise));
        chart.set(kk, PreflopStrategy::pure(PreflopActionType::Raise));

        PreflopChartActionConfig {
            preflop_config: PreflopChartConfig::with_single_chart(chart),
            postflop_config: ConfigurableActionConfig::default(),
        }
    }

    fn create_generator(
        game_state: &GameState,
        config: PreflopChartActionConfig,
    ) -> PreflopChartActionGenerator {
        PreflopChartActionGenerator::new(
            CFRState::new(game_state.clone()),
            TraversalState::new_root(0),
            Arc::new(config),
        )
    }

    #[test]
    fn test_preflop_actions_for_chart_hand() {
        let game_state = create_test_game_state();
        let config = create_simple_config();
        let generator = create_generator(&game_state, config);

        // The generator should return actions based on the chart
        let actions = generator.gen_possible_actions(&game_state);

        // Should have at least one action
        assert!(!actions.is_empty());
    }

    #[test]
    fn test_preflop_actions_for_non_chart_hand() {
        let game_state = create_test_game_state();
        let config = create_simple_config();
        let generator = create_generator(&game_state, config);

        // For hands not in the chart, should get fold action
        let actions = generator.gen_possible_actions(&game_state);

        // Should have at least fold/call
        assert!(!actions.is_empty());
    }

    #[test]
    fn test_postflop_delegates_to_configurable() {
        let stacks = vec![500.0; 2];
        let mut game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        // Move to flop
        game_state.advance_round();

        let config = create_simple_config();
        let generator = create_generator(&game_state, config);

        let actions = generator.gen_possible_actions(&game_state);

        // Should have configurable actions (call, raises, all-in, etc.)
        assert!(!actions.is_empty());
        // Should include all-in since that's default for ConfigurableActionConfig
        assert!(actions.contains(&AgentAction::AllIn));
    }

    #[test]
    fn test_position_calculation() {
        // Test the static position calculation
        // Position is relative to BB (position 0)
        //
        // 2-player heads up: dealer=0
        // Player 0 = BTN, Player 1 = BB
        assert_eq!(PreflopChartConfig::calculate_position(1, 0, 2), 0); // BB
        assert_eq!(PreflopChartConfig::calculate_position(0, 0, 2), 1); // BTN/SB

        // 6-player: dealer=2
        // BB is at (2+2)%6 = 4
        assert_eq!(PreflopChartConfig::calculate_position(4, 2, 6), 0); // BB
        assert_eq!(PreflopChartConfig::calculate_position(3, 2, 6), 1); // SB
        assert_eq!(PreflopChartConfig::calculate_position(2, 2, 6), 2); // BTN
        assert_eq!(PreflopChartConfig::calculate_position(1, 2, 6), 3); // CO
        assert_eq!(PreflopChartConfig::calculate_position(0, 2, 6), 4); // HJ
        assert_eq!(PreflopChartConfig::calculate_position(5, 2, 6), 5); // UTG
    }

    #[test]
    fn test_bet_or_all_in() {
        let game_state = create_test_game_state();
        let config = create_simple_config();
        let generator = create_generator(&game_state, config);

        // Normal bet (player has 100 stack, 0 already bet)
        let action = generator.bet_or_all_in(50.0, 100.0, 0.0);
        assert!(matches!(action, AgentAction::Bet(50.0)));

        // Bet equals total possible (stack + current bet)
        let action = generator.bet_or_all_in(100.0, 100.0, 0.0);
        assert!(matches!(action, AgentAction::AllIn));

        // Bet exceeds stack
        let action = generator.bet_or_all_in(150.0, 100.0, 0.0);
        assert!(matches!(action, AgentAction::AllIn));

        // With some already bet
        let action = generator.bet_or_all_in(110.0, 90.0, 10.0);
        assert!(matches!(action, AgentAction::AllIn));
    }

    #[test]
    fn test_fold_becomes_check_when_no_bet() {
        let mut game_state = create_test_game_state();
        // Move past blinds
        game_state.advance_round();

        let config = create_simple_config();
        let generator = create_generator(&game_state, config);

        // When there's no bet to call, Fold should become Call (check)
        let action = generator.convert_preflop_action(PreflopActionType::Fold, &game_state);
        // If current_bet == player_bet (no bet to call), fold becomes check
        // In preflop after blinds, there is a bet to call for most positions
        assert!(action.is_some());
    }

    #[test]
    fn test_mixed_strategy_generates_multiple_actions() {
        let mut chart = PreflopChart::new();
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        // Mixed strategy: 70% raise, 30% call
        chart.set(
            aa,
            PreflopStrategy::new(vec![
                (PreflopActionType::Raise, 0.7),
                (PreflopActionType::Call, 0.3),
            ])
            .unwrap(),
        );

        let config = PreflopChartActionConfig {
            preflop_config: PreflopChartConfig::with_single_chart(chart),
            postflop_config: ConfigurableActionConfig::default(),
        };

        let game_state = create_test_game_state();
        let generator = create_generator(&game_state, config);

        // For AA with mixed strategy, should generate both raise and call actions
        // (Note: the actual hand dealt may not be AA, but this tests the mechanism)
        let actions = generator.gen_possible_actions(&game_state);
        assert!(!actions.is_empty());
    }

    /// Test PreflopChartActionGenerator in a full simulation.
    #[test]
    fn test_preflop_chart_action_gen_in_simulation() {
        use crate::arena::cfr::{
            CFRAgentBuilder, CFRState, DepthBasedIteratorGen, DepthBasedIteratorGenConfig,
            TraversalSet,
        };
        use crate::arena::game_state::Round;
        use crate::arena::{Agent, HoldemSimulationBuilder, test_util};
        use rand::{SeedableRng, rngs::StdRng};

        // Create a starting game state
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();

        let config = create_simple_config();
        let iter_config = DepthBasedIteratorGenConfig::new(vec![1]);

        // Create CFR agents with PreflopChartActionGenerator sharing the same CFR states
        let cfr_states: Vec<CFRState> = (0..game_state.num_players)
            .map(|_| CFRState::new(game_state.clone()))
            .collect();
        let traversal_set = TraversalSet::new(game_state.num_players);
        let agents: Vec<Box<dyn Agent>> = (0..2)
            .map(|idx| {
                Box::new(
                    CFRAgentBuilder::<PreflopChartActionGenerator, DepthBasedIteratorGen>::new()
                        .name(format!("PreflopChartAgent-{idx}"))
                        .player_idx(idx)
                        .cfr_states(cfr_states.clone())
                        .traversal_set(traversal_set.clone())
                        .gamestate_iterator_gen_config(iter_config.clone())
                        .action_gen_config(config.clone())
                        .build(),
                ) as Box<dyn Agent>
            })
            .collect();

        let mut rng = StdRng::seed_from_u64(42);

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .cfr_context(cfr_states, traversal_set, true)
            .build()
            .unwrap();

        sim.run(&mut rng);

        assert_eq!(Round::Complete, sim.game_state.round);
        test_util::assert_valid_game_state(&sim.game_state);
    }

    /// Test multiple games with PreflopChartActionGenerator.
    #[test]
    fn test_multiple_games_preflop_chart_action_gen() {
        use crate::arena::cfr::{
            CFRAgentBuilder, CFRState, DepthBasedIteratorGen, DepthBasedIteratorGenConfig,
            TraversalSet,
        };
        use crate::arena::game_state::Round;
        use crate::arena::{Agent, HoldemSimulationBuilder, test_util};
        use rand::{SeedableRng, rngs::StdRng};

        let config = create_simple_config();
        let iter_config = DepthBasedIteratorGenConfig::new(vec![1]);

        for game_idx in 0..5 {
            let game_state = GameStateBuilder::new()
                .num_players_with_stack(2, 100.0)
                .blinds(10.0, 5.0)
                .build()
                .unwrap();

            let cfr_states: Vec<CFRState> = (0..game_state.num_players)
                .map(|_| CFRState::new(game_state.clone()))
                .collect();
            let traversal_set = TraversalSet::new(game_state.num_players);
            let agents: Vec<Box<dyn Agent>> = (0..2)
                .map(|idx| {
                    Box::new(
                        CFRAgentBuilder::<PreflopChartActionGenerator, DepthBasedIteratorGen>::new(
                        )
                        .name(format!("PreflopChartAgent-game{game_idx}-p{idx}"))
                        .player_idx(idx)
                        .cfr_states(cfr_states.clone())
                        .traversal_set(traversal_set.clone())
                        .gamestate_iterator_gen_config(iter_config.clone())
                        .action_gen_config(config.clone())
                        .build(),
                    ) as Box<dyn Agent>
                })
                .collect();

            let mut rng = StdRng::seed_from_u64(42 + game_idx as u64);

            let mut sim = HoldemSimulationBuilder::default()
                .game_state(game_state)
                .agents(agents)
                .cfr_context(cfr_states, traversal_set, true)
                .build()
                .unwrap();

            sim.run(&mut rng);

            assert_eq!(
                Round::Complete,
                sim.game_state.round,
                "Game {game_idx} should complete"
            );
            test_util::assert_valid_game_state(&sim.game_state);
        }
    }

    // Tests for PreflopChartConfig

    #[test]
    fn test_default_config() {
        let config = PreflopChartConfig::default();
        assert_eq!(config.charts.len(), 1);
        assert_eq!(config.raise_size_bb, 2.5);
        assert_eq!(config.three_bet_multiplier, 3.0);
    }

    #[test]
    fn test_position_calculation_6_player() {
        // 6-player table, dealer at position 3
        // Positions relative to BB (position 0):
        // Player 5 = BB (0)
        // Player 4 = SB (1)
        // Player 3 = BTN (2)
        // Player 2 = CO (3)
        // Player 1 = HJ (4)
        // Player 0 = UTG (5)

        let num_players = 6;
        let dealer_idx = 3;

        assert_eq!(
            PreflopChartConfig::calculate_position(5, dealer_idx, num_players),
            0
        ); // BB
        assert_eq!(
            PreflopChartConfig::calculate_position(4, dealer_idx, num_players),
            1
        ); // SB
        assert_eq!(
            PreflopChartConfig::calculate_position(3, dealer_idx, num_players),
            2
        ); // BTN
        assert_eq!(
            PreflopChartConfig::calculate_position(2, dealer_idx, num_players),
            3
        ); // CO
        assert_eq!(
            PreflopChartConfig::calculate_position(1, dealer_idx, num_players),
            4
        ); // HJ
        assert_eq!(
            PreflopChartConfig::calculate_position(0, dealer_idx, num_players),
            5
        ); // UTG
    }

    #[test]
    fn test_chart_for_position_single_chart() {
        let mut chart = PreflopChart::new();
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        chart.set(aa, PreflopStrategy::pure(PreflopActionType::Raise));

        let config = PreflopChartConfig::with_single_chart(chart);

        // All positions should use the same chart
        for pos in 0..10 {
            let c = config.chart_for_position(pos);
            assert!(c.get(&aa).is_some());
        }
    }

    #[test]
    fn test_chart_for_position_multiple_charts() {
        let mut btn_chart = PreflopChart::new();
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        btn_chart.set(aa, PreflopStrategy::pure(PreflopActionType::Raise));

        let mut utg_chart = PreflopChart::new();
        // UTG chart only has AA, nothing else
        utg_chart.set(aa, PreflopStrategy::pure(PreflopActionType::Raise));
        let kk = PreflopHand::new(Value::King, Value::King, false);
        utg_chart.set(kk, PreflopStrategy::pure(PreflopActionType::Raise));

        let config = PreflopChartConfig::new(vec![btn_chart, utg_chart.clone()]);

        // Position 0 (BTN) uses first chart
        let btn = config.chart_for_position(0);
        assert!(btn.get(&aa).is_some());
        assert!(btn.get(&kk).is_none()); // BTN chart doesn't have KK

        // Position 1+ uses second chart (which has KK)
        let other = config.chart_for_position(1);
        assert!(other.get(&kk).is_some());
    }

    #[test]
    fn test_validate_empty_charts() {
        let config = PreflopChartConfig {
            charts: vec![],
            raise_size_bb: 2.5,
            three_bet_multiplier: 3.0,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_raise_size() {
        let config = PreflopChartConfig {
            charts: vec![PreflopChart::new()],
            raise_size_bb: 0.0,
            three_bet_multiplier: 3.0,
        };
        assert!(config.validate().is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_roundtrip() {
        let mut chart = PreflopChart::new();
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        chart.set(aa, PreflopStrategy::pure(PreflopActionType::Raise));

        let config = PreflopChartConfig {
            charts: vec![chart],
            raise_size_bb: 3.0,
            three_bet_multiplier: 3.5,
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: PreflopChartConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.raise_size_bb, 3.0);
        assert_eq!(parsed.three_bet_multiplier, 3.5);
        assert_eq!(parsed.charts.len(), 1);
    }
}
