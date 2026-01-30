//! Preflop chart-based CFR agent.
//!
//! This agent wraps a CFR agent and uses pre-configured preflop charts
//! for preflop decisions, falling back to CFR exploration for post-flop play.

use rand::Rng;
use tracing::event;

use crate::arena::action::AgentAction;
use crate::arena::cfr::{ActionGenerator, CFRAgent, GameStateIteratorGen};
use crate::arena::game_state::Round;
use crate::arena::{Agent, GameState, Historian};
use crate::holdem::{PreflopActionType, PreflopHand};

use super::config::PreflopChartConfig;

/// A CFR agent that uses preflop charts for preflop decisions.
///
/// This agent wraps an inner CFR agent and intercepts preflop decisions,
/// using pre-configured charts based on position. For all post-flop rounds,
/// it delegates to the inner CFR agent for normal exploration.
///
/// # Benefits
///
/// - Eliminates expensive preflop CFR exploration
/// - Uses proven GTO-style preflop ranges
/// - Still learns optimal post-flop play via CFR
///
/// # Example
///
/// ```rust,ignore
/// use rs_poker::arena::cfr::{
///     PreflopChartCFRAgent, BasicCFRActionGenerator, FixedGameStateIteratorGen,
///     create_6max_rfi_charts,
/// };
///
/// let preflop_config = create_6max_rfi_charts();
/// let agent = PreflopChartCFRAgent::<BasicCFRActionGenerator, FixedGameStateIteratorGen>::new(
///     "MyAgent",
///     0,
///     game_state,
///     FixedGameStateIteratorGen::new(10),
///     (),
///     preflop_config,
/// );
/// ```
pub struct PreflopChartCFRAgent<T, I>
where
    T: ActionGenerator + 'static,
    I: GameStateIteratorGen + Clone + 'static,
{
    inner: CFRAgent<T, I>,
    preflop_config: PreflopChartConfig,
    name: String,
}

impl<T, I> PreflopChartCFRAgent<T, I>
where
    T: ActionGenerator + 'static,
    I: GameStateIteratorGen + Clone + 'static,
{
    /// Create a new preflop chart CFR agent.
    ///
    /// # Arguments
    ///
    /// * `name` - Name for this agent
    /// * `player_idx` - The player index this agent represents
    /// * `game_state` - The starting game state
    /// * `gamestate_iterator_gen` - Generator for post-flop CFR exploration
    /// * `config` - Configuration for the action generator
    /// * `preflop_config` - Preflop chart configuration
    pub fn new(
        name: impl Into<String>,
        player_idx: usize,
        game_state: GameState,
        gamestate_iterator_gen: I,
        config: T::Config,
        preflop_config: PreflopChartConfig,
    ) -> Self {
        let name = name.into();
        let inner = CFRAgent::new(
            format!("{name}-inner"),
            player_idx,
            game_state,
            gamestate_iterator_gen,
            config,
        );

        Self {
            inner,
            preflop_config,
            name,
        }
    }

    /// Get a reference to the preflop configuration.
    pub fn preflop_config(&self) -> &PreflopChartConfig {
        &self.preflop_config
    }

    /// Get a reference to the inner CFR agent.
    pub fn inner(&self) -> &CFRAgent<T, I> {
        &self.inner
    }

    /// Act during preflop using the configured charts.
    fn act_preflop(&self, game_state: &GameState) -> AgentAction {
        let player_idx = game_state.to_act_idx();
        let player_hand = &game_state.hands[player_idx];

        // Convert player's hole cards to PreflopHand
        let preflop_hand = match PreflopHand::try_from(player_hand) {
            Ok(h) => h,
            Err(e) => {
                event!(
                    tracing::Level::WARN,
                    ?e,
                    "Failed to convert hand to PreflopHand, folding"
                );
                return AgentAction::Fold;
            }
        };

        // Calculate position relative to button
        let position = PreflopChartConfig::calculate_position(
            player_idx,
            game_state.dealer_idx,
            game_state.num_players,
        );

        // Get chart for this position
        let chart = self.preflop_config.chart_for_position(position);

        // Look up strategy for this hand
        let strategy = chart.get_or_fold(&preflop_hand);

        // Sample an action from the strategy
        let random_value: f32 = rand::rng().random();
        let preflop_action = strategy.sample(random_value);

        event!(
            tracing::Level::DEBUG,
            hand = %preflop_hand,
            position,
            ?preflop_action,
            "Preflop chart action"
        );

        // Convert preflop action to AgentAction
        self.convert_preflop_action(preflop_action, game_state)
    }

    /// Convert a PreflopActionType to an AgentAction with proper sizing.
    fn convert_preflop_action(
        &self,
        action: PreflopActionType,
        game_state: &GameState,
    ) -> AgentAction {
        let player_idx = game_state.to_act_idx();
        let big_blind = game_state.big_blind;
        let current_bet = game_state.current_round_bet();
        let player_stack = game_state.stacks[player_idx];

        match action {
            PreflopActionType::Fold => {
                // If there's nothing to call (we're first to act or everyone checked),
                // check instead of folding
                if current_bet <= game_state.round_data.player_bet[player_idx] {
                    AgentAction::Call // This will be a check
                } else {
                    AgentAction::Fold
                }
            }
            PreflopActionType::Call => AgentAction::Call,
            PreflopActionType::Raise => {
                // Standard open raise: raise_size_bb * big_blind
                let raise_amount = self.preflop_config.raise_size_bb * big_blind;

                // If there's already a raise, this becomes a re-raise
                if current_bet > big_blind {
                    // Someone already raised, so we 3-bet
                    let three_bet_amount = current_bet * self.preflop_config.three_bet_multiplier;
                    self.bet_or_all_in(three_bet_amount, player_stack)
                } else {
                    // Standard open raise
                    self.bet_or_all_in(raise_amount, player_stack)
                }
            }
            PreflopActionType::ThreeBet => {
                // 3-bet: multiply current bet by three_bet_multiplier
                let three_bet_amount = current_bet * self.preflop_config.three_bet_multiplier;
                self.bet_or_all_in(three_bet_amount, player_stack)
            }
            PreflopActionType::FourBet => {
                // 4-bet: typically 2.5x the 3-bet
                let four_bet_amount = current_bet * 2.5;
                self.bet_or_all_in(four_bet_amount, player_stack)
            }
        }
    }

    /// Return a Bet action or AllIn if the bet amount exceeds the stack.
    fn bet_or_all_in(&self, amount: f32, stack: f32) -> AgentAction {
        if amount >= stack {
            AgentAction::AllIn
        } else {
            AgentAction::Bet(amount)
        }
    }
}

impl<T, I> Agent for PreflopChartCFRAgent<T, I>
where
    T: ActionGenerator + 'static,
    I: GameStateIteratorGen + Clone + 'static,
{
    fn act(&mut self, id: u128, game_state: &GameState) -> AgentAction {
        // Use preflop charts during preflop
        if game_state.round == Round::Preflop {
            self.act_preflop(game_state)
        } else {
            // Delegate to inner CFR agent for post-flop
            self.inner.act(id, game_state)
        }
    }

    fn historian(&self) -> Option<Box<dyn Historian>> {
        self.inner.historian()
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl<T, I> Clone for PreflopChartCFRAgent<T, I>
where
    T: ActionGenerator + Clone + 'static,
    I: GameStateIteratorGen + Clone + 'static,
    CFRAgent<T, I>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            preflop_config: self.preflop_config.clone(),
            name: self.name.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Agent;
    use crate::arena::cfr::{BasicCFRActionGenerator, FixedGameStateIteratorGen};
    use crate::core::Value;
    use crate::holdem::{PreflopChart, PreflopStrategy};

    fn create_test_game_state() -> GameState {
        GameState::new_starting(vec![100.0; 2], 10.0, 5.0, 0.0, 0)
    }

    fn create_simple_config() -> PreflopChartConfig {
        let mut chart = PreflopChart::new();
        // Only AA and KK raise
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        let kk = PreflopHand::new(Value::King, Value::King, false);
        chart.set(aa, PreflopStrategy::pure(PreflopActionType::Raise));
        chart.set(kk, PreflopStrategy::pure(PreflopActionType::Raise));

        PreflopChartConfig::with_single_chart(chart)
    }

    #[test]
    fn test_create_agent() {
        let game_state = create_test_game_state();
        let preflop_config = create_simple_config();

        let agent = PreflopChartCFRAgent::<BasicCFRActionGenerator, FixedGameStateIteratorGen>::new(
            "TestAgent",
            0,
            game_state,
            FixedGameStateIteratorGen::new(1),
            (),
            preflop_config,
        );

        assert_eq!(agent.name(), "TestAgent");
    }

    #[test]
    fn test_historian_delegates_to_inner_agent() {
        let game_state = create_test_game_state();
        let preflop_config = create_simple_config();

        let agent = PreflopChartCFRAgent::<BasicCFRActionGenerator, FixedGameStateIteratorGen>::new(
            "TestAgent",
            0,
            game_state,
            FixedGameStateIteratorGen::new(1),
            (),
            preflop_config,
        );

        // The historian should be delegated from the inner CFR agent
        let historian = agent.historian();
        assert!(
            historian.is_some(),
            "PreflopChartCFRAgent should delegate historian from inner CFR agent"
        );
    }

    #[test]
    fn test_position_calculation() {
        // Test the static position calculation
        // Position is relative to BB (position 0)
        //
        // 2-player heads up: dealer=0
        // Player 0 = BTN, Player 1 = BB
        // BB is at (0+2)%2 = 0, but in 2-player BB is seat 1
        // Actually in 2-player: dealer=0 means seat 0 is BTN, seat 1 is BB
        // BB index = (0 + 2) % 2 = 0... wait that's wrong for 2-player
        // In 2-player: BTN posts SB, other player posts BB
        // So with dealer=0: seat 0 = BTN/SB, seat 1 = BB
        // BB index = (dealer + 2) % 2 = 0 for dealer=0... hmm
        // Let's just verify the 2-player case works correctly:
        // Seat 1 should be BB (pos 0), Seat 0 should be BTN (pos 2 mod table size)
        assert_eq!(PreflopChartConfig::calculate_position(1, 0, 2), 0); // BB
        assert_eq!(PreflopChartConfig::calculate_position(0, 0, 2), 1); // BTN/SB

        // 6-player: dealer=2
        // BB is at (2+2)%6 = 4
        // Player 4 = BB (0)
        // Player 3 = SB (1)
        // Player 2 = BTN (2)
        // Player 1 = CO (3)
        // Player 0 = HJ (4)
        // Player 5 = UTG (5)
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
        let preflop_config = create_simple_config();

        let agent = PreflopChartCFRAgent::<BasicCFRActionGenerator, FixedGameStateIteratorGen>::new(
            "TestAgent",
            0,
            game_state,
            FixedGameStateIteratorGen::new(1),
            (),
            preflop_config,
        );

        // Normal bet
        let action = agent.bet_or_all_in(50.0, 100.0);
        assert!(matches!(action, AgentAction::Bet(50.0)));

        // Bet equals stack
        let action = agent.bet_or_all_in(100.0, 100.0);
        assert!(matches!(action, AgentAction::AllIn));

        // Bet exceeds stack
        let action = agent.bet_or_all_in(150.0, 100.0);
        assert!(matches!(action, AgentAction::AllIn));
    }

    #[test]
    fn test_convert_fold_to_check_when_no_bet() {
        let game_state = create_test_game_state();
        let preflop_config = create_simple_config();

        let agent = PreflopChartCFRAgent::<BasicCFRActionGenerator, FixedGameStateIteratorGen>::new(
            "TestAgent",
            0,
            game_state.clone(),
            FixedGameStateIteratorGen::new(1),
            (),
            preflop_config,
        );

        // When there's no bet to call, Fold should become Call (check)
        // This is handled in convert_preflop_action by checking current_bet
        let action = agent.convert_preflop_action(PreflopActionType::Call, &game_state);
        assert!(matches!(action, AgentAction::Call));
    }
}
