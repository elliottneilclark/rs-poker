use std::sync::Arc;

use crate::arena::{GameState, action::AgentAction, game_state::Round};

use super::super::{CFRState, TraversalState};
use super::ActionGenerator;

/// Configuration for per-round bet sizing options.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct RoundActionConfig {
    /// Whether check/call is enabled for this round
    pub call_enabled: bool,
    /// Raise multipliers (e.g., [1.0, 2.0, 3.0] for 1x, 2x, 3x min raise)
    pub raise_mult: Vec<f32>,
    /// Pot multipliers (e.g., [0.33, 0.5, 1.0] for 33%, 50%, 100% pot)
    pub pot_mult: Vec<f32>,
    /// Whether to include "setup shove" action (bet so pot + call = remaining stack)
    pub setup_shove: bool,
    /// Whether to include all-in action
    pub all_in: bool,
}

impl Default for RoundActionConfig {
    fn default() -> Self {
        Self {
            call_enabled: true,
            raise_mult: vec![1.0, 2.0],
            pot_mult: vec![0.5, 1.0],
            setup_shove: false,
            all_in: true,
        }
    }
}

impl RoundActionConfig {
    /// Validate the round configuration
    pub fn validate(&self) -> Result<(), String> {
        for &mult in &self.raise_mult {
            if mult < 1.0 {
                return Err(format!(
                    "raise_mult must be >= 1.0 (cannot raise less than min raise), got {}",
                    mult
                ));
            }
        }
        for &mult in &self.pot_mult {
            if mult < 0.0 {
                return Err(format!("pot_mult must be non-negative, got {}", mult));
            }
        }
        Ok(())
    }
}

/// Configuration for the ConfigurableActionGenerator.
///
/// Allows per-round customization of bet sizing options.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Default)]
pub struct ConfigurableActionConfig {
    /// Default configuration used for rounds without specific overrides
    #[cfg_attr(feature = "serde", serde(default))]
    pub default: RoundActionConfig,
    /// Optional override for preflop
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub preflop: Option<RoundActionConfig>,
    /// Optional override for flop
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub flop: Option<RoundActionConfig>,
    /// Optional override for turn
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub turn: Option<RoundActionConfig>,
    /// Optional override for river
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub river: Option<RoundActionConfig>,
}

impl ConfigurableActionConfig {
    /// Get the configuration for a specific round
    pub fn round_config(&self, round: Round) -> &RoundActionConfig {
        match round {
            Round::DealPreflop | Round::Preflop => self.preflop.as_ref().unwrap_or(&self.default),
            Round::DealFlop | Round::Flop => self.flop.as_ref().unwrap_or(&self.default),
            Round::DealTurn | Round::Turn => self.turn.as_ref().unwrap_or(&self.default),
            Round::DealRiver | Round::River => self.river.as_ref().unwrap_or(&self.default),
            Round::Starting | Round::Ante | Round::Showdown | Round::Complete => &self.default,
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        self.default.validate()?;
        if let Some(ref cfg) = self.preflop {
            cfg.validate()?;
        }
        if let Some(ref cfg) = self.flop {
            cfg.validate()?;
        }
        if let Some(ref cfg) = self.turn {
            cfg.validate()?;
        }
        if let Some(ref cfg) = self.river {
            cfg.validate()?;
        }
        Ok(())
    }
}

/// Configurable action generator with per-round bet sizing options.
///
/// This generator allows users to specify:
/// - Per-round betting options (raise multiples, pot multiples)
/// - Enable/disable check/call and all-in
/// - "Setup shove" action (bet so pot + call = remaining stack)
///
/// Actions generated are mapped to indices using `ActionIndexMapper`
/// with a fixed 52-action space for consistent tree traversal.
pub struct ConfigurableActionGenerator {
    cfr_state: CFRState,
    traversal_state: TraversalState,
    config: Arc<ConfigurableActionConfig>,
}

impl ConfigurableActionGenerator {
    /// Create a new configurable action generator with the given configuration.
    pub fn new_with_config(
        cfr_state: CFRState,
        traversal_state: TraversalState,
        config: ConfigurableActionConfig,
    ) -> Self {
        ConfigurableActionGenerator {
            cfr_state,
            traversal_state,
            config: Arc::new(config),
        }
    }
}

impl ActionGenerator for ConfigurableActionGenerator {
    type Config = ConfigurableActionConfig;

    fn new(
        cfr_state: CFRState,
        traversal_state: TraversalState,
        config: Arc<Self::Config>,
    ) -> Self {
        ConfigurableActionGenerator {
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
        let mut actions: Vec<AgentAction> = Vec::new();
        // Track used amounts to avoid duplicates (within epsilon)
        let mut used_amounts: Vec<f32> = Vec::new();
        let epsilon = 0.01;

        let current_bet = game_state.current_round_bet();
        let player_bet = game_state.current_round_current_player_bet();
        let stack = game_state.current_player_stack();
        let pot = game_state.total_pot;
        let min_raise = game_state.current_round_min_raise();
        let to_call = current_bet - player_bet;
        let all_in_amount = player_bet + stack;

        let round_config = self.config.round_config(game_state.round);

        // Helper to check if an amount is already used
        let is_amount_used = |amount: f32, used: &[f32]| -> bool {
            used.iter().any(|&a| (a - amount).abs() < epsilon)
        };

        // Fold - only if there's something to call
        if to_call > 0.0 {
            actions.push(AgentAction::Fold);
        }

        // Call/Check - if enabled
        if round_config.call_enabled {
            actions.push(AgentAction::Bet(current_bet));
            used_amounts.push(current_bet);
        }

        // Track the minimum valid raise for ordering
        let min_valid_raise = current_bet + min_raise;

        // Raise multipliers
        for &mult in &round_config.raise_mult {
            let raise_amount = current_bet + min_raise * mult;
            // Must be at least min raise and less than all-in
            if raise_amount >= min_valid_raise
                && raise_amount < all_in_amount
                && !is_amount_used(raise_amount, &used_amounts)
            {
                actions.push(AgentAction::Bet(raise_amount));
                used_amounts.push(raise_amount);
            }
        }

        // Pot multipliers
        for &mult in &round_config.pot_mult {
            let pot_amount = current_bet + pot * mult;
            // Must be at least min raise and less than all-in
            if pot_amount >= min_valid_raise
                && pot_amount < all_in_amount
                && !is_amount_used(pot_amount, &used_amounts)
            {
                actions.push(AgentAction::Bet(pot_amount));
                used_amounts.push(pot_amount);
            }
        }

        // Setup shove (bet so pot + call = remaining stack)
        if round_config.setup_shove {
            let setup_bet = (stack + player_bet + current_bet - pot) / 2.0;
            if setup_bet >= min_valid_raise
                && setup_bet < all_in_amount
                && !is_amount_used(setup_bet, &used_amounts)
            {
                actions.push(AgentAction::Bet(setup_bet));
                used_amounts.push(setup_bet);
            }
        }

        // All-in - if enabled and we can bet more than current bet
        if round_config.all_in && all_in_amount > current_bet {
            actions.push(AgentAction::AllIn);
        }

        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::GameStateBuilder;

    fn create_configurable_generator(
        game_state: &GameState,
        config: ConfigurableActionConfig,
    ) -> ConfigurableActionGenerator {
        ConfigurableActionGenerator::new_with_config(
            CFRState::new(game_state.clone()),
            TraversalState::new_root(0),
            config,
        )
    }

    fn default_configurable_config() -> ConfigurableActionConfig {
        ConfigurableActionConfig {
            default: RoundActionConfig {
                call_enabled: true,
                raise_mult: vec![1.0, 2.0],
                pot_mult: vec![0.5, 1.0],
                setup_shove: false,
                all_in: true,
            },
            preflop: None,
            flop: None,
            turn: None,
            river: None,
        }
    }

    #[test]
    fn test_configurable_gen_actions_basic() {
        let stacks = vec![500.0; 2];
        let mut game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        game_state.advance_round();
        game_state.do_bet(30.0, false).unwrap();

        let action_gen = create_configurable_generator(&game_state, default_configurable_config());
        let actions = action_gen.gen_possible_actions(&game_state);

        // Should have: Fold, Call, raises, All-in
        assert!(actions.contains(&AgentAction::Fold));
        assert!(actions.iter().any(|a| matches!(a, AgentAction::Bet(_))));
        assert!(actions.contains(&AgentAction::AllIn));
    }

    #[test]
    fn test_configurable_no_fold_when_checking() {
        let stacks = vec![100.0; 2];
        let mut game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        game_state.advance_round();

        let action_gen = create_configurable_generator(&game_state, default_configurable_config());
        let actions = action_gen.gen_possible_actions(&game_state);

        // No fold option when there's nothing to call
        assert!(!actions.contains(&AgentAction::Fold));
    }

    #[test]
    fn test_configurable_with_setup_shove() {
        let stacks = vec![500.0; 2];
        let mut game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        game_state.advance_round();

        let config = ConfigurableActionConfig {
            default: RoundActionConfig {
                call_enabled: true,
                raise_mult: vec![1.0],
                pot_mult: vec![0.5],
                setup_shove: true,
                all_in: true,
            },
            preflop: None,
            flop: None,
            turn: None,
            river: None,
        };

        let action_gen = create_configurable_generator(&game_state, config);
        let actions = action_gen.gen_possible_actions(&game_state);

        // Should have actions including setup shove
        assert!(!actions.is_empty());
    }

    #[test]
    fn test_configurable_per_round_config() {
        let stacks = vec![500.0; 2];
        let game_state_preflop = GameStateBuilder::new()
            .stacks(stacks.clone())
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        // Preflop is the starting round for new games

        let config = ConfigurableActionConfig {
            default: RoundActionConfig {
                call_enabled: true,
                raise_mult: vec![1.0],
                pot_mult: vec![0.5, 1.0],
                setup_shove: false,
                all_in: true,
            },
            preflop: Some(RoundActionConfig {
                call_enabled: true,
                raise_mult: vec![2.0, 2.5, 3.0], // More raise options preflop
                pot_mult: vec![],                // No pot-based raises preflop
                setup_shove: false,
                all_in: true,
            }),
            flop: None,
            turn: None,
            river: None,
        };

        let action_gen = create_configurable_generator(&game_state_preflop, config.clone());

        // Actions on preflop should use preflop config (3 raise_mult, 0 pot_mult)
        let preflop_actions = action_gen.gen_possible_actions(&game_state_preflop);

        // Verify that preflop generates actions
        assert!(!preflop_actions.is_empty());
    }

    /// Helper to verify all generated actions are valid for configurable generator
    fn verify_configurable_actions_valid(game_state: &GameState, config: ConfigurableActionConfig) {
        let action_gen = create_configurable_generator(game_state, config);
        let actions = action_gen.gen_possible_actions(game_state);

        for action in &actions {
            let mut gs_copy = game_state.clone();
            let result = match action {
                AgentAction::Fold => {
                    gs_copy.fold();
                    Ok(())
                }
                AgentAction::Bet(amount) => gs_copy.do_bet(*amount, false).map(|_| ()),
                AgentAction::Call => {
                    let call_amount = gs_copy.current_round_bet();
                    gs_copy.do_bet(call_amount, false).map(|_| ())
                }
                AgentAction::AllIn => {
                    let all_in_amount =
                        gs_copy.current_round_current_player_bet() + gs_copy.current_player_stack();
                    gs_copy.do_bet(all_in_amount, false).map(|_| ())
                }
            };

            assert!(
                result.is_ok(),
                "Action {:?} should be valid but got error: {:?}\n\
                 Game state: current_bet={}, min_raise={}, player_bet={}, stack={}, pot={}",
                action,
                result.err(),
                game_state.current_round_bet(),
                game_state.current_round_min_raise(),
                game_state.current_round_current_player_bet(),
                game_state.current_player_stack(),
                game_state.total_pot
            );
        }
    }

    #[test]
    fn test_configurable_all_actions_valid_preflop() {
        let stacks = vec![100.0; 2];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        verify_configurable_actions_valid(&game_state, default_configurable_config());
    }

    #[test]
    fn test_configurable_all_actions_valid_flop() {
        let stacks = vec![100.0; 2];
        let mut game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        game_state.advance_round();
        verify_configurable_actions_valid(&game_state, default_configurable_config());
    }

    #[test]
    fn test_configurable_all_actions_valid_facing_bet() {
        let stacks = vec![500.0; 2];
        let mut game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        game_state.advance_round();
        game_state.do_bet(30.0, false).unwrap();
        verify_configurable_actions_valid(&game_state, default_configurable_config());
    }

    #[test]
    fn test_configurable_call_disabled() {
        let stacks = vec![500.0; 2];
        let mut game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        game_state.advance_round();
        game_state.do_bet(30.0, false).unwrap();

        let config = ConfigurableActionConfig {
            default: RoundActionConfig {
                call_enabled: false,
                raise_mult: vec![1.0],
                pot_mult: vec![],
                setup_shove: false,
                all_in: true,
            },
            preflop: None,
            flop: None,
            turn: None,
            river: None,
        };

        let action_gen = create_configurable_generator(&game_state, config);
        let actions = action_gen.gen_possible_actions(&game_state);

        // With call_enabled = false, we should still have actions (raise_mult and all_in)
        assert!(!actions.is_empty());
        // Verify we have at least a raise action
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, AgentAction::Bet(_) | AgentAction::AllIn))
        );
    }

    // NOTE: Max Raises Per Round tests have been moved to action_validator.rs
    // The raise cap logic is now in ValidatorConfig, not in action generators.
}
