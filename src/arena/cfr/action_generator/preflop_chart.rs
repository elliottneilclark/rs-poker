//! Preflop chart-based action generator.
//!
//! This action generator uses pre-configured preflop charts to limit
//! exploration during preflop, generating only actions that have non-zero
//! probability in the chart for the current (hand, position, scenario).
//! For post-flop, it delegates to configurable action generation.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use smallvec::smallvec;
use thiserror::Error;
use tracing::event;

use crate::arena::GameState;
use crate::arena::action::AgentAction;
use crate::arena::cfr::{CFRState, TraversalState};
use crate::arena::game_state::Round;
use crate::holdem::{PreflopChart, PreflopHand, PreflopScenario};

use super::{ActionGenerator, ActionVec, ConfigurableActionConfig, ConfigurableActionGenerator};

/// Errors produced when validating a [`PreflopChartConfig`].
#[derive(Debug, Error, PartialEq)]
pub enum PreflopChartConfigError {
    /// At least one position chart must be supplied.
    #[error("at least one position chart is required")]
    NoCharts,

    /// `raise_size_bb` must be strictly positive.
    #[error("raise_size_bb must be positive, got {0}")]
    NonPositiveRaiseSizeBb(f32),

    /// `three_bet_multiplier` must be strictly positive.
    #[error("three_bet_multiplier must be positive, got {0}")]
    NonPositiveThreeBetMultiplier(f32),

    /// `four_bet_plus_multiplier` must be strictly positive.
    #[error("four_bet_plus_multiplier must be positive, got {0}")]
    NonPositiveFourBetPlusMultiplier(f32),

    /// The RFI chart at `position` contains a hand with `call > 0`. Limping
    /// is not representable — the pot is unopened in RFI.
    #[error(
        "position {position}: RFI strategy for {hand} has call={call:.3}; limping not supported"
    )]
    RfiCallNotAllowed {
        position: usize,
        hand: String,
        call: f32,
    },

    /// The Vs4Bet chart at `position` contains a hand with `raise > 0`. The
    /// raise cap blocks 5-bets, so the action isn't representable.
    #[error(
        "position {position}: Vs4Bet strategy for {hand} has raise={raise:.3}; 5-betting is capped"
    )]
    Vs4BetRaiseNotAllowed {
        position: usize,
        hand: String,
        raise: f32,
    },
}

fn default_raise_size_bb() -> f32 {
    2.5
}

fn default_three_bet_multiplier() -> f32 {
    3.0
}

fn default_four_bet_plus_multiplier() -> f32 {
    2.5
}

/// Charts for each preflop scenario at a single position.
///
/// All four scenarios are optional; an empty chart is equivalent to
/// "everyone folds for this scenario". In JSON, unknown/omitted fields
/// default to an empty chart.
///
/// # Examples
///
/// ```json
/// {
///   "rfi":     { "AA": {"raise": 1.0}, "KK": {"raise": 1.0} },
///   "vs_open": { "AA": {"raise": 0.5, "call": 0.5} }
/// }
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct PositionCharts {
    /// Unopened pot — raise (open) or fold. Limping (call > 0) is rejected.
    #[serde(default, skip_serializing_if = "PreflopChart::is_empty")]
    pub rfi: PreflopChart,
    /// Facing one raise — raise (3-bet), call, or fold.
    #[serde(default, skip_serializing_if = "PreflopChart::is_empty")]
    pub vs_open: PreflopChart,
    /// Facing two raises — raise (4-bet), call, or fold.
    #[serde(default, skip_serializing_if = "PreflopChart::is_empty")]
    pub vs_3bet: PreflopChart,
    /// Facing 3+ raises — call or fold only (raise cap blocks further raises).
    #[serde(default, skip_serializing_if = "PreflopChart::is_empty")]
    pub vs_4bet: PreflopChart,
}

impl PositionCharts {
    /// Borrow the chart for a specific scenario.
    pub fn chart_for(&self, scenario: PreflopScenario) -> &PreflopChart {
        match scenario {
            PreflopScenario::Rfi => &self.rfi,
            PreflopScenario::VsOpen => &self.vs_open,
            PreflopScenario::Vs3Bet => &self.vs_3bet,
            PreflopScenario::Vs4Bet => &self.vs_4bet,
        }
    }
}

/// Configuration for preflop chart-based play.
///
/// `positions` is indexed by BB-relative distance: 0 = Big Blind, 1 = Small
/// Blind, 2 = Button, 3 = Cutoff, 4 = Hijack, 5 = UTG, 6+ = earlier
/// positions. If a position index exceeds the available charts, the last
/// chart is used — this fallback makes the tightest early-position range
/// the default for unspecified seats at bigger tables.
///
/// # Example JSON
///
/// ```json
/// {
///   "raise_size_bb": 2.5,
///   "three_bet_multiplier": 3.0,
///   "four_bet_plus_multiplier": 2.5,
///   "positions": [
///     {},  // BB: all-fold everywhere
///     { "vs_open": { "AA": {"raise": 1.0} } },
///     {
///       "rfi":     { "AA": {"raise": 1.0}, "KK": {"raise": 1.0} },
///       "vs_open": { "AA": {"raise": 0.5, "call": 0.5} }
///     }
///   ]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct PreflopChartConfig {
    /// Charts indexed by BB-relative position (0 = BB, 1 = SB, 2 = BTN, ...).
    pub positions: Vec<PositionCharts>,

    /// Open-raise size as a multiple of the big blind (used in RFI).
    #[serde(default = "default_raise_size_bb")]
    pub raise_size_bb: f32,

    /// 3-bet sizing = this * current_bet (used in VsOpen).
    #[serde(default = "default_three_bet_multiplier")]
    pub three_bet_multiplier: f32,

    /// 4-bet and subsequent raise sizing = this * current_bet (used in
    /// Vs3Bet).
    #[serde(default = "default_four_bet_plus_multiplier")]
    pub four_bet_plus_multiplier: f32,
}

impl Default for PreflopChartConfig {
    fn default() -> Self {
        Self {
            positions: vec![PositionCharts::default()],
            raise_size_bb: default_raise_size_bb(),
            three_bet_multiplier: default_three_bet_multiplier(),
            four_bet_plus_multiplier: default_four_bet_plus_multiplier(),
        }
    }
}

impl PreflopChartConfig {
    /// Create a new config with the given position charts.
    pub fn new(positions: Vec<PositionCharts>) -> Self {
        Self {
            positions,
            ..Default::default()
        }
    }

    /// Create a config with a single position chart used for all positions
    /// (via the fallback).
    pub fn with_single_position(charts: PositionCharts) -> Self {
        Self {
            positions: vec![charts],
            ..Default::default()
        }
    }

    /// Get the [`PositionCharts`] for a given BB-relative position.
    ///
    /// If `position` exceeds `positions.len()`, returns the last entry
    /// (tightest-seat fallback).
    pub fn charts_for_position(&self, position: usize) -> &PositionCharts {
        if self.positions.is_empty() {
            panic!("PreflopChartConfig has no positions");
        }
        let idx = position.min(self.positions.len() - 1);
        &self.positions[idx]
    }

    /// Get the chart for a specific (position, scenario). Missing entries
    /// return an empty (all-fold) chart.
    pub fn chart_for(&self, position: usize, scenario: PreflopScenario) -> &PreflopChart {
        self.charts_for_position(position).chart_for(scenario)
    }

    /// Calculate the BB-relative position from player/dealer indexes.
    ///
    /// - 0 = Big Blind
    /// - 1 = Small Blind (or Button in heads-up, which posts SB)
    /// - 2 = Button (for 3+ players)
    /// - 3 = Cutoff
    /// - 4 = Hijack
    /// - 5+ = Earlier positions (UTG, UTG+1, ...)
    pub fn calculate_position(player_idx: usize, dealer_idx: usize, num_players: usize) -> usize {
        // In heads-up, BB is 1 seat after dealer (BTN posts SB)
        // In 3+ players, BB is 2 seats after dealer
        let bb_offset = if num_players == 2 { 1 } else { 2 };
        let bb_idx = (dealer_idx + bb_offset) % num_players;
        (bb_idx + num_players - player_idx) % num_players
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), PreflopChartConfigError> {
        if self.positions.is_empty() {
            return Err(PreflopChartConfigError::NoCharts);
        }
        if self.raise_size_bb <= 0.0 {
            return Err(PreflopChartConfigError::NonPositiveRaiseSizeBb(
                self.raise_size_bb,
            ));
        }
        if self.three_bet_multiplier <= 0.0 {
            return Err(PreflopChartConfigError::NonPositiveThreeBetMultiplier(
                self.three_bet_multiplier,
            ));
        }
        if self.four_bet_plus_multiplier <= 0.0 {
            return Err(PreflopChartConfigError::NonPositiveFourBetPlusMultiplier(
                self.four_bet_plus_multiplier,
            ));
        }
        for (position, charts) in self.positions.iter().enumerate() {
            for (hand, strategy) in charts.rfi.iter() {
                if strategy.call() > 0.0 {
                    return Err(PreflopChartConfigError::RfiCallNotAllowed {
                        position,
                        hand: hand.to_notation(),
                        call: strategy.call(),
                    });
                }
            }
            for (hand, strategy) in charts.vs_4bet.iter() {
                if strategy.raise() > 0.0 {
                    return Err(PreflopChartConfigError::Vs4BetRaiseNotAllowed {
                        position,
                        hand: hand.to_notation(),
                        raise: strategy.raise(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Configuration for the preflop chart action generator.
///
/// Combines preflop chart configuration with configurable post-flop action
/// generation.
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

/// Action generator that uses preflop charts for preflop decisions and a
/// configurable generator for post-flop streets.
///
/// During preflop, the generator:
/// 1. Derives the player's scenario from `round_data.total_raise_count`.
/// 2. Looks up the chart for `(bb_relative_position, scenario)`.
/// 3. Reads the strategy for the player's hand (defaulting to pure-fold).
/// 4. Emits `AgentAction`s for every non-zero frequency (raise/call/fold).
///
/// # Example
///
/// ```
/// use std::sync::Arc;
///
/// use rs_poker::arena::GameStateBuilder;
/// use rs_poker::arena::cfr::{
///     ActionGenerator, CFRState, ConfigurableActionConfig, PreflopChartActionConfig,
///     PreflopChartActionGenerator, PreflopChartConfig, TraversalState,
/// };
///
/// let game_state = GameStateBuilder::new()
///     .num_players_with_stack(2, 100.0)
///     .blinds(10.0, 5.0)
///     .build()
///     .unwrap();
///
/// let action_config = PreflopChartActionConfig {
///     preflop_config: PreflopChartConfig::default(),
///     postflop_config: ConfigurableActionConfig::default(),
/// };
///
/// let generator = PreflopChartActionGenerator::new(
///     CFRState::new(game_state.clone()),
///     TraversalState::new_root(0),
///     Arc::new(action_config),
/// );
///
/// let actions = generator.gen_possible_actions(&game_state);
/// assert!(!actions.is_empty());
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

    fn gen_possible_actions(&self, game_state: &GameState) -> ActionVec {
        if matches!(game_state.round, Round::Preflop | Round::DealPreflop) {
            self.gen_preflop_actions(game_state)
        } else {
            self.gen_postflop_actions(game_state)
        }
    }
}

impl PreflopChartActionGenerator {
    /// Generate preflop actions from the (position, scenario) chart.
    fn gen_preflop_actions(&self, game_state: &GameState) -> ActionVec {
        let player_idx = game_state.to_act_idx();
        let player_hand = &game_state.hands[player_idx];

        let preflop_hand = match PreflopHand::try_from(player_hand) {
            Ok(h) => h,
            Err(e) => {
                event!(
                    tracing::Level::WARN,
                    ?e,
                    "Failed to convert hand to PreflopHand, returning fold only"
                );
                return smallvec![AgentAction::Fold];
            }
        };

        let position = PreflopChartConfig::calculate_position(
            player_idx,
            game_state.dealer_idx,
            game_state.num_players,
        );
        let scenario = PreflopScenario::from_raise_count(game_state.round_data.total_raise_count);
        let chart = self.config.preflop_config.chart_for(position, scenario);
        let strategy = chart.get_or_fold(&preflop_hand);

        event!(
            tracing::Level::DEBUG,
            hand = %preflop_hand,
            position,
            scenario = scenario.label(),
            "Generating preflop actions from chart"
        );

        let current_bet = game_state.current_round_bet();
        let player_bet = game_state.current_round_current_player_bet();
        let to_call = current_bet - player_bet;

        let mut actions = ActionVec::new();
        if strategy.raise() > 0.0
            && let Some(action) = self.raise_action(scenario, game_state)
            && !actions.contains(&action)
        {
            actions.push(action);
        }
        if strategy.call() > 0.0 {
            let action = AgentAction::Call;
            if !actions.contains(&action) {
                actions.push(action);
            }
        }
        if strategy.fold_freq() > 0.0 {
            let action = if to_call > 0.0 {
                AgentAction::Fold
            } else {
                AgentAction::Call // Check when no bet to face.
            };
            if !actions.contains(&action) {
                actions.push(action);
            }
        }

        // Never leave the picker with an empty action set (it panics on
        // empty-range sampling). Fall back to Check or Fold.
        if actions.is_empty() {
            if to_call > 0.0 {
                actions.push(AgentAction::Fold);
            } else {
                actions.push(AgentAction::Call); // Check
            }
        }

        actions
    }

    /// Build an `AgentAction::Bet` (or `AllIn`) sized for `scenario`.
    ///
    /// Returns `None` for `Vs4Bet` (no 5-bet sizing; the raise cap blocks it
    /// anyway). If the raise cap is already reached at earlier scenarios,
    /// falls back to a Call so the agent can still see the flop.
    fn raise_action(
        &self,
        scenario: PreflopScenario,
        game_state: &GameState,
    ) -> Option<AgentAction> {
        let player_idx = game_state.to_act_idx();
        let big_blind = game_state.big_blind;
        let current_bet = game_state.current_round_bet();
        let player_bet = game_state.current_round_current_player_bet();
        let player_stack = game_state.stacks[player_idx];
        let raise_capped = game_state.is_raise_capped();

        if raise_capped {
            return Some(AgentAction::Call);
        }

        let amount = match scenario {
            PreflopScenario::Rfi => self.config.preflop_config.raise_size_bb * big_blind,
            PreflopScenario::VsOpen => {
                current_bet * self.config.preflop_config.three_bet_multiplier
            }
            PreflopScenario::Vs3Bet => {
                current_bet * self.config.preflop_config.four_bet_plus_multiplier
            }
            PreflopScenario::Vs4Bet => return None,
        };
        Some(bet_or_all_in(amount, player_stack, player_bet))
    }

    /// Generate post-flop actions using configurable action generator logic.
    ///
    /// Calls the borrowed-config helper directly rather than constructing a
    /// `ConfigurableActionGenerator` per node: the postflop logic reads only the
    /// config and game state, so reconstructing the generator (cloning
    /// `cfr_state`/`traversal_state`) and deep-cloning `postflop_config` (whose
    /// `RoundActionConfig`s own `Vec<f32>`s) on every call was pure overhead.
    fn gen_postflop_actions(&self, game_state: &GameState) -> ActionVec {
        ConfigurableActionGenerator::gen_actions_from_config(
            &self.config.postflop_config,
            game_state,
        )
    }
}

/// Return a Bet action, or AllIn if the bet amount would consume the stack.
fn bet_or_all_in(amount: f32, stack: f32, player_bet: f32) -> AgentAction {
    let all_in_amount = stack + player_bet;
    if amount >= all_in_amount {
        AgentAction::AllIn
    } else {
        AgentAction::Bet(amount)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::GameStateBuilder;
    use crate::arena::cfr::{CFRState, TraversalState};
    use crate::core::Value;
    use crate::holdem::PreflopStrategy;

    #[test]
    fn test_validate_no_positions() {
        let cfg = PreflopChartConfig {
            positions: vec![],
            raise_size_bb: 2.5,
            three_bet_multiplier: 3.0,
            four_bet_plus_multiplier: 2.5,
        };
        assert_eq!(
            cfg.validate().unwrap_err(),
            PreflopChartConfigError::NoCharts
        );
    }

    #[test]
    fn test_validate_non_positive_raise_size() {
        let cfg = PreflopChartConfig {
            positions: vec![PositionCharts::default()],
            raise_size_bb: 0.0,
            three_bet_multiplier: 3.0,
            four_bet_plus_multiplier: 2.5,
        };
        assert_eq!(
            cfg.validate().unwrap_err(),
            PreflopChartConfigError::NonPositiveRaiseSizeBb(0.0)
        );
    }

    #[test]
    fn test_validate_non_positive_three_bet_multiplier() {
        let cfg = PreflopChartConfig {
            positions: vec![PositionCharts::default()],
            raise_size_bb: 2.5,
            three_bet_multiplier: -1.0,
            four_bet_plus_multiplier: 2.5,
        };
        assert_eq!(
            cfg.validate().unwrap_err(),
            PreflopChartConfigError::NonPositiveThreeBetMultiplier(-1.0)
        );
    }

    #[test]
    fn test_validate_non_positive_four_bet_plus_multiplier() {
        let cfg = PreflopChartConfig {
            positions: vec![PositionCharts::default()],
            raise_size_bb: 2.5,
            three_bet_multiplier: 3.0,
            four_bet_plus_multiplier: 0.0,
        };
        assert_eq!(
            cfg.validate().unwrap_err(),
            PreflopChartConfigError::NonPositiveFourBetPlusMultiplier(0.0)
        );
    }

    #[test]
    fn test_validate_rfi_rejects_call() {
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        let mut charts = PositionCharts::default();
        charts.rfi.set(aa, PreflopStrategy::new(0.5, 0.5).unwrap());
        let cfg = PreflopChartConfig::new(vec![charts]);
        let err = cfg.validate().unwrap_err();
        assert!(
            matches!(err, PreflopChartConfigError::RfiCallNotAllowed { .. }),
            "expected RfiCallNotAllowed, got {err:?}"
        );
    }

    #[test]
    fn test_validate_vs4bet_rejects_raise() {
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        let mut charts = PositionCharts::default();
        charts.vs_4bet.set(aa, PreflopStrategy::pure_raise());
        let cfg = PreflopChartConfig::new(vec![charts]);
        let err = cfg.validate().unwrap_err();
        assert!(
            matches!(err, PreflopChartConfigError::Vs4BetRaiseNotAllowed { .. }),
            "expected Vs4BetRaiseNotAllowed, got {err:?}"
        );
    }

    fn create_test_game_state() -> GameState {
        GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap()
    }

    fn create_simple_config() -> PreflopChartActionConfig {
        let mut charts = PositionCharts::default();
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        let kk = PreflopHand::new(Value::King, Value::King, false);
        charts.rfi.set(aa, PreflopStrategy::pure_raise());
        charts.rfi.set(kk, PreflopStrategy::pure_raise());

        PreflopChartActionConfig {
            preflop_config: PreflopChartConfig::with_single_position(charts),
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

    /// Regression: when the raise cap is reached preflop and the chart says
    /// Raise, the generator must fall back to a non-raise action so the
    /// validated action set is never empty.
    #[test]
    fn test_chart_raise_when_raise_capped_falls_back_to_call() {
        use crate::arena::cfr::validate_actions;
        use crate::arena::game_state::{Round, RoundData};
        use crate::core::{Hand, PlayerBitSet};

        // 6-handed preflop, three raises already in the pot.
        let num_players = 6;
        let big_blind = 5.0;
        let small_blind = 2.5;

        let round_player_bet = vec![0.0, 12.5, 0.0, 37.5, 0.0, 112.5];
        let stacks: Vec<f32> = vec![500.0; num_players];
        let player_bet = round_player_bet.clone();

        let mut round_data = RoundData::new_with_bets(
            big_blind,
            PlayerBitSet::new(num_players),
            0,
            round_player_bet,
        );
        round_data.total_raise_count = 3;

        let mut hands = vec![Hand::default(); num_players];
        hands[0] = Hand::new_from_str("AsAh").unwrap();

        let game_state = GameStateBuilder::new()
            .round(Round::Preflop)
            .round_data(round_data)
            .stacks(stacks)
            .player_bet(player_bet)
            .big_blind(big_blind)
            .small_blind(small_blind)
            .hands(hands)
            .build()
            .unwrap();

        assert!(
            game_state.is_raise_capped(),
            "test setup: expected raise cap reached"
        );

        // Player faces 3 raises → Vs4Bet. Pure-call chart entry so we exercise
        // the call path; raise is rejected in Vs4Bet anyway.
        let mut charts = PositionCharts::default();
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        charts.vs_4bet.set(aa, PreflopStrategy::pure_call());
        let config = PreflopChartActionConfig {
            preflop_config: PreflopChartConfig::with_single_position(charts),
            postflop_config: ConfigurableActionConfig::default(),
        };

        let generator = create_generator(&game_state, config);
        let raw = generator.gen_possible_actions(&game_state);
        assert!(
            !raw.is_empty(),
            "generator must always produce at least one action"
        );

        let validated = validate_actions(raw.clone(), &game_state);
        assert!(
            !validated.is_empty(),
            "validated action set must not be empty (raw was {raw:?})"
        );

        assert!(
            validated
                .iter()
                .any(|a| matches!(a, AgentAction::Call | AgentAction::Fold)),
            "expected Call or Fold in validated action set, got {validated:?}"
        );
    }

    #[test]
    fn test_preflop_actions_for_chart_hand() {
        let game_state = create_test_game_state();
        let config = create_simple_config();
        let generator = create_generator(&game_state, config);
        let actions = generator.gen_possible_actions(&game_state);
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
        game_state.advance_round();

        let config = create_simple_config();
        let generator = create_generator(&game_state, config);

        let actions = generator.gen_possible_actions(&game_state);
        assert!(!actions.is_empty());
        assert!(actions.contains(&AgentAction::AllIn));
    }

    #[test]
    fn test_position_calculation() {
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
        assert!(matches!(
            bet_or_all_in(50.0, 100.0, 0.0),
            AgentAction::Bet(_)
        ));
        assert!(matches!(
            bet_or_all_in(100.0, 100.0, 0.0),
            AgentAction::AllIn
        ));
        assert!(matches!(
            bet_or_all_in(150.0, 100.0, 0.0),
            AgentAction::AllIn
        ));
        assert!(matches!(
            bet_or_all_in(110.0, 90.0, 10.0),
            AgentAction::AllIn
        ));
    }

    #[test]
    fn test_mixed_strategy_generates_multiple_actions() {
        let mut charts = PositionCharts::default();
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        charts.rfi.set(aa, PreflopStrategy::new(0.7, 0.0).unwrap());

        let config = PreflopChartActionConfig {
            preflop_config: PreflopChartConfig::with_single_position(charts),
            postflop_config: ConfigurableActionConfig::default(),
        };

        let game_state = create_test_game_state();
        let generator = create_generator(&game_state, config);
        let actions = generator.gen_possible_actions(&game_state);
        assert!(!actions.is_empty());
    }

    /// Test PreflopChartActionGenerator in a full simulation.
    #[tokio::test(flavor = "current_thread")]
    async fn test_preflop_chart_action_gen_in_simulation() {
        use crate::arena::cfr::{
            Budget, CFRAgentBuilder, CFRState, IterationCount, MaxWidth, MostRestrictive, PerDepth,
            TraversalSet,
        };
        use crate::arena::game_state::Round;
        use crate::arena::{Agent, HoldemSimulationBuilder, test_util};
        use rand::{SeedableRng, rngs::StdRng};
        use std::sync::Arc;

        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();

        let config = create_simple_config();
        // Old `[1]` schedule: recurse one level, one wave per node.
        let budget: Arc<dyn Budget> = Arc::new(MostRestrictive::new(vec![
            Arc::new(PerDepth::new(
                vec![Arc::new(IterationCount::new(1)) as Arc<dyn Budget>],
                Arc::new(IterationCount::new(1)),
            )),
            Arc::new(MaxWidth::new(vec![1])),
        ]));

        let cfr_state = CFRState::new(game_state.clone());
        let traversal_set = TraversalSet::new(game_state.num_players);
        let agents: Vec<Box<dyn Agent>> = (0..2)
            .map(|idx| {
                Box::new(
                    CFRAgentBuilder::<PreflopChartActionGenerator>::new()
                        .name(format!("PreflopChartAgent-{idx}"))
                        .player_idx(idx)
                        .cfr_state(cfr_state.clone())
                        .traversal_set(traversal_set.clone())
                        .budget(budget.clone())
                        .action_gen_config(config.clone())
                        .build(),
                ) as Box<dyn Agent>
            })
            .collect();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .cfr_context(cfr_state, traversal_set, true)
            .build_with_rng(StdRng::seed_from_u64(42))
            .unwrap();

        sim.run().await;

        assert_eq!(Round::Complete, sim.game_state.round);
        test_util::assert_valid_game_state(&sim.game_state);
    }

    #[test]
    fn test_charts_for_position_fallback() {
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        let mut loose = PositionCharts::default();
        loose.rfi.set(aa, PreflopStrategy::pure_raise());

        let mut tight = PositionCharts::default();
        let kk = PreflopHand::new(Value::King, Value::King, false);
        tight.rfi.set(kk, PreflopStrategy::pure_raise());

        let config = PreflopChartConfig::new(vec![loose, tight.clone()]);

        // Position 0 uses first charts.
        assert!(config.chart_for(0, PreflopScenario::Rfi).get(&aa).is_some());
        // Position 1 uses second charts.
        assert!(config.chart_for(1, PreflopScenario::Rfi).get(&kk).is_some());
        // Position 5 (past end) falls back to the last (tightest).
        assert!(config.chart_for(5, PreflopScenario::Rfi).get(&kk).is_some());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_roundtrip() {
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        let mut charts = PositionCharts::default();
        charts.rfi.set(aa, PreflopStrategy::pure_raise());

        let config = PreflopChartConfig {
            positions: vec![charts],
            raise_size_bb: 3.0,
            three_bet_multiplier: 3.5,
            four_bet_plus_multiplier: 2.2,
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: PreflopChartConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.raise_size_bb, 3.0);
        assert_eq!(parsed.three_bet_multiplier, 3.5);
        assert_eq!(parsed.four_bet_plus_multiplier, 2.2);
        assert_eq!(parsed.positions.len(), 1);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_minimal_config() {
        // Only positions required; multipliers all default.
        let json = r#"{"positions": [{}]}"#;
        let cfg: PreflopChartConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.raise_size_bb, 2.5);
        assert_eq!(cfg.three_bet_multiplier, 3.0);
        assert_eq!(cfg.four_bet_plus_multiplier, 2.5);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_strategy_minimal_in_chart() {
        // Per-hand strategy with only "raise" set; call defaults to 0.
        let json = r#"{
            "positions": [
                {
                    "rfi": {"AA": {"raise": 1.0}}
                }
            ]
        }"#;
        let cfg: PreflopChartConfig = serde_json::from_str(json).unwrap();
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        let strategy = cfg.chart_for(0, PreflopScenario::Rfi).get(&aa).unwrap();
        assert_eq!(strategy.raise(), 1.0);
        assert_eq!(strategy.call(), 0.0);
    }
}
