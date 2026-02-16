use approx::abs_diff_eq;

use crate::arena::{GameState, action::AgentAction};

/// Total number of action indices in the fixed mapping scheme.
/// - Index 0: Fold
/// - Index 1: Call/Check
/// - Indices 2-50: Raises (spread from min to max using logarithmic distribution)
/// - Index 51: All-in
pub const NUM_ACTION_INDICES: usize = 52;

/// Index for fold action.
pub const ACTION_IDX_FOLD: usize = 0;

/// Index for call/check action.
pub const ACTION_IDX_CALL: usize = 1;

/// First index for raise actions.
pub const ACTION_IDX_RAISE_MIN: usize = 2;

/// Last index for raise actions.
pub const ACTION_IDX_RAISE_MAX: usize = 50;

/// Index for all-in action.
pub const ACTION_IDX_ALL_IN: usize = 51;

/// Configuration for the ActionIndexMapper.
///
/// Defines the effective range of bet amounts that will be mapped to indices 2-50.
/// Amounts outside this range will be clamped to the boundary indices.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ActionIndexMapperConfig {
    /// Minimum bet amount for the mapping range (typically big blind).
    pub min_bet: f32,
    /// Maximum bet amount for the mapping range (typically second largest stack).
    pub max_bet: f32,
}

impl ActionIndexMapperConfig {
    /// Create a new configuration with the specified bet range.
    pub fn new(min_bet: f32, max_bet: f32) -> Self {
        Self { min_bet, max_bet }
    }

    /// Compute the effective range from a game state.
    ///
    /// Returns `(min_bet, max_bet)` where:
    /// - `min_bet` is the big blind
    /// - `max_bet` is the second largest stack (or largest if only one player)
    ///
    /// The second largest stack is used because that's the maximum effective bet
    /// that can be called in a heads-up or multi-way pot.
    pub fn from_game_state(game_state: &GameState) -> Self {
        let (min_bet, max_bet) = compute_effective_range(game_state);
        Self::new(min_bet, max_bet)
    }
}

impl Default for ActionIndexMapperConfig {
    fn default() -> Self {
        // Default to typical online micro-stakes: 1BB min, 100BB max
        Self::new(1.0, 100.0)
    }
}

/// Compute the effective bet range from a game state.
///
/// Returns `(min_bet, max_bet)` where:
/// - `min_bet` is the big blind
/// - `max_bet` is the second largest stack (or largest stack if only one player)
pub fn compute_effective_range(game_state: &GameState) -> (f32, f32) {
    let min_bet = game_state.big_blind;

    // Find the two largest starting stacks to determine effective stack
    let mut stacks: Vec<f32> = game_state.starting_stacks.clone();
    stacks.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let max_bet = if stacks.len() >= 2 {
        // Second largest stack is the effective stack
        stacks[1]
    } else if !stacks.is_empty() {
        // Single player - use their stack
        stacks[0]
    } else {
        // Fallback - use big blind * 100
        min_bet * 100.0
    };

    // Ensure max_bet is at least min_bet to avoid division issues
    let max_bet = max_bet.max(min_bet * 2.0);

    (min_bet, max_bet)
}

/// Maps poker actions to fixed indices for the CFR tree.
///
/// This mapper uses a fixed absolute-amount mapping that spreads raises
/// across indices 2-50 based on the actual bet amount using logarithmic
/// distribution.
///
/// ## Index Layout (52 total)
///
/// | Index | Action |
/// |-------|--------|
/// | 0 | Fold |
/// | 1 | Call/Check |
/// | 2-50 | Raises (spread from min to max using log scale) |
/// | 51 | All-in |
///
/// The logarithmic distribution is used because poker bet sizing often
/// grows exponentially (e.g., 3x, 9x, 27x patterns).
#[derive(Debug, Clone)]
pub struct ActionIndexMapper {
    config: ActionIndexMapperConfig,
}

impl ActionIndexMapper {
    /// Create a new mapper with the given configuration.
    pub fn new(config: ActionIndexMapperConfig) -> Self {
        Self { config }
    }

    /// Create a new mapper from a game state.
    ///
    /// The effective range is computed from the big blind and stacks.
    pub fn from_game_state(game_state: &GameState) -> Self {
        Self::new(ActionIndexMapperConfig::from_game_state(game_state))
    }

    /// Get the configuration for this mapper.
    pub fn config(&self) -> &ActionIndexMapperConfig {
        &self.config
    }

    /// Map an action to its index in the children array.
    ///
    /// # Arguments
    ///
    /// * `action` - The action to map
    /// * `game_state` - The current game state (used to determine all-in amount)
    ///
    /// # Returns
    ///
    /// The index (0-51) for this action
    pub fn action_to_idx(&self, action: &AgentAction, game_state: &GameState) -> usize {
        self.action_to_idx_raw(
            action,
            game_state.current_round_bet(),
            game_state.current_round_current_player_bet(),
            game_state.current_player_stack(),
        )
    }

    /// Map an action to its index using raw state values.
    ///
    /// This is useful when you have the relevant state values but not a full GameState,
    /// such as when reconstructing pre-action state from a payload.
    ///
    /// # Arguments
    ///
    /// * `action` - The action to map
    /// * `current_round_bet` - The current bet to call in this round
    /// * `current_player_bet` - How much the current player has already bet this round
    /// * `current_player_stack` - The current player's remaining stack
    ///
    /// # Returns
    ///
    /// The index (0-51) for this action
    pub fn action_to_idx_raw(
        &self,
        action: &AgentAction,
        current_round_bet: f32,
        current_player_bet: f32,
        current_player_stack: f32,
    ) -> usize {
        match action {
            AgentAction::Fold => ACTION_IDX_FOLD,
            AgentAction::Call => ACTION_IDX_CALL,
            AgentAction::AllIn => ACTION_IDX_ALL_IN,
            AgentAction::Bet(amount) => {
                // Check if this is a call (matches current bet) FIRST.
                // This is important because when call amount == all-in amount
                // (e.g., calling uses entire stack), we want to treat it as a call
                // not an all-in raise.
                if abs_diff_eq!(*amount, current_round_bet) {
                    return ACTION_IDX_CALL;
                }

                // Check if this bet is actually an all-in
                let all_in_amount = current_player_bet + current_player_stack;

                // If the bet is approximately equal to all-in, treat it as all-in
                if abs_diff_eq!(*amount, all_in_amount) {
                    return ACTION_IDX_ALL_IN;
                }

                // Otherwise, map the bet amount to an index using log scale
                self.bet_to_index(*amount)
            }
        }
    }

    /// Map a bet amount to an index using logarithmic distribution.
    ///
    /// Bets <= min_bet map to index 2 (ACTION_IDX_RAISE_MIN).
    /// Bets >= max_bet map to index 50 (ACTION_IDX_RAISE_MAX).
    /// Bets in between are distributed logarithmically across indices 2-50.
    fn bet_to_index(&self, bet: f32) -> usize {
        let min_bet = self.config.min_bet;
        let max_bet = self.config.max_bet;

        if bet <= min_bet {
            return ACTION_IDX_RAISE_MIN;
        }
        if bet >= max_bet {
            return ACTION_IDX_RAISE_MAX;
        }

        // Use logarithmic interpolation
        let log_min = min_bet.ln();
        let log_max = max_bet.ln();
        let log_bet = bet.ln();

        // Compute fraction in log space
        let fraction = (log_bet - log_min) / (log_max - log_min);

        // Map to indices 2-50 (49 slots)
        let num_slots = (ACTION_IDX_RAISE_MAX - ACTION_IDX_RAISE_MIN) as f32;
        let index = ACTION_IDX_RAISE_MIN + (fraction * num_slots).round() as usize;

        // Clamp to valid range
        index.clamp(ACTION_IDX_RAISE_MIN, ACTION_IDX_RAISE_MAX)
    }

    /// Map an index back to an approximate bet amount.
    ///
    /// This is the inverse of `bet_to_index` and is useful for debugging
    /// or for generating representative bet amounts for each index.
    pub fn index_to_bet(&self, index: usize) -> Option<f32> {
        match index {
            ACTION_IDX_FOLD | ACTION_IDX_CALL | ACTION_IDX_ALL_IN => None,
            idx if (ACTION_IDX_RAISE_MIN..=ACTION_IDX_RAISE_MAX).contains(&idx) => {
                let min_bet = self.config.min_bet;
                let max_bet = self.config.max_bet;

                let log_min = min_bet.ln();
                let log_max = max_bet.ln();

                // Compute fraction from index
                let num_slots = (ACTION_IDX_RAISE_MAX - ACTION_IDX_RAISE_MIN) as f32;
                let fraction = (idx - ACTION_IDX_RAISE_MIN) as f32 / num_slots;

                // Convert back from log space
                let log_bet = log_min + fraction * (log_max - log_min);
                Some(log_bet.exp())
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::GameStateBuilder;

    fn create_test_game_state() -> GameState {
        GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap()
    }

    fn create_mapper() -> ActionIndexMapper {
        ActionIndexMapper::new(ActionIndexMapperConfig::new(10.0, 100.0))
    }

    // === Basic action mapping tests ===

    #[test]
    fn test_fold_always_maps_to_zero() {
        let mapper = create_mapper();
        let game_state = create_test_game_state();

        assert_eq!(
            mapper.action_to_idx(&AgentAction::Fold, &game_state),
            ACTION_IDX_FOLD
        );
    }

    #[test]
    fn test_call_always_maps_to_one() {
        let mapper = create_mapper();
        let game_state = create_test_game_state();

        assert_eq!(
            mapper.action_to_idx(&AgentAction::Call, &game_state),
            ACTION_IDX_CALL
        );
    }

    #[test]
    fn test_all_in_always_maps_to_51() {
        let mapper = create_mapper();
        let game_state = create_test_game_state();

        assert_eq!(
            mapper.action_to_idx(&AgentAction::AllIn, &game_state),
            ACTION_IDX_ALL_IN
        );
    }

    #[test]
    fn test_bet_matching_current_bet_maps_to_call() {
        let mapper = create_mapper();
        let game_state = create_test_game_state();

        let current_bet = game_state.current_round_bet();
        assert_eq!(
            mapper.action_to_idx(&AgentAction::Bet(current_bet), &game_state),
            ACTION_IDX_CALL
        );
    }

    #[test]
    fn test_bet_matching_all_in_maps_to_51() {
        let mapper = create_mapper();
        let game_state = create_test_game_state();

        let all_in_amount =
            game_state.current_round_current_player_bet() + game_state.current_player_stack();
        assert_eq!(
            mapper.action_to_idx(&AgentAction::Bet(all_in_amount), &game_state),
            ACTION_IDX_ALL_IN
        );
    }

    // === Raise mapping tests ===

    #[test]
    fn test_raises_spread_across_indices() {
        let mapper = ActionIndexMapper::new(ActionIndexMapperConfig::new(10.0, 1000.0));
        let game_state = create_test_game_state();

        // Small raise should map to lower index
        let small_raise_idx = mapper.action_to_idx(&AgentAction::Bet(20.0), &game_state);
        assert!((ACTION_IDX_RAISE_MIN..=ACTION_IDX_RAISE_MAX).contains(&small_raise_idx));

        // Large raise should map to higher index
        let large_raise_idx = mapper.action_to_idx(&AgentAction::Bet(500.0), &game_state);
        assert!((ACTION_IDX_RAISE_MIN..=ACTION_IDX_RAISE_MAX).contains(&large_raise_idx));

        // Large raise should have higher index than small raise
        assert!(large_raise_idx > small_raise_idx);
    }

    #[test]
    fn test_min_bet_maps_to_raise_min() {
        let mapper = ActionIndexMapper::new(ActionIndexMapperConfig::new(10.0, 100.0));
        let game_state = create_test_game_state();

        let idx = mapper.action_to_idx(&AgentAction::Bet(10.0), &game_state);
        assert_eq!(idx, ACTION_IDX_RAISE_MIN);
    }

    #[test]
    fn test_bet_below_min_maps_to_raise_min() {
        let mapper = ActionIndexMapper::new(ActionIndexMapperConfig::new(10.0, 100.0));
        let game_state = create_test_game_state();

        let idx = mapper.action_to_idx(&AgentAction::Bet(5.0), &game_state);
        assert_eq!(idx, ACTION_IDX_RAISE_MIN);
    }

    #[test]
    fn test_bet_at_max_maps_to_raise_max() {
        // Use a custom game state where 100 is not the all-in amount
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 200.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();

        let mapper = ActionIndexMapper::new(ActionIndexMapperConfig::new(10.0, 100.0));

        let idx = mapper.action_to_idx(&AgentAction::Bet(100.0), &game_state);
        assert_eq!(idx, ACTION_IDX_RAISE_MAX);
    }

    #[test]
    fn test_bet_above_max_maps_to_raise_max() {
        // Use a custom game state where 150 is not the all-in amount
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 200.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();

        let mapper = ActionIndexMapper::new(ActionIndexMapperConfig::new(10.0, 100.0));

        let idx = mapper.action_to_idx(&AgentAction::Bet(150.0), &game_state);
        // This should map to raise_max since 150 > 100 but < all-in (200)
        assert_eq!(idx, ACTION_IDX_RAISE_MAX);
    }

    // === Logarithmic distribution tests ===

    #[test]
    fn test_log_distribution_midpoint() {
        let mapper = ActionIndexMapper::new(ActionIndexMapperConfig::new(10.0, 1000.0));

        // Geometric mean of 10 and 1000 is sqrt(10 * 1000) = 100
        let midpoint_idx = mapper.bet_to_index(100.0);

        // Should be roughly in the middle of the range (index 26)
        let mid_idx = (ACTION_IDX_RAISE_MIN + ACTION_IDX_RAISE_MAX) / 2;
        assert!(
            (midpoint_idx as i32 - mid_idx as i32).abs() <= 1,
            "Geometric mean should map to middle index, got {} expected ~{}",
            midpoint_idx,
            mid_idx
        );
    }

    #[test]
    fn test_index_to_bet_roundtrip() {
        let mapper = ActionIndexMapper::new(ActionIndexMapperConfig::new(10.0, 1000.0));

        // Test that index_to_bet is a reasonable inverse of bet_to_index
        for idx in ACTION_IDX_RAISE_MIN..=ACTION_IDX_RAISE_MAX {
            let bet = mapper.index_to_bet(idx).unwrap();
            let recovered_idx = mapper.bet_to_index(bet);

            // Should round-trip to same index
            assert_eq!(
                idx, recovered_idx,
                "Index {} -> bet {} -> index {}",
                idx, bet, recovered_idx
            );
        }
    }

    // === Configuration tests ===

    #[test]
    fn test_compute_effective_range() {
        let game_state = GameStateBuilder::new()
            .stacks(vec![100.0, 200.0, 150.0])
            .blinds(10.0, 5.0)
            .build()
            .unwrap();

        let (min_bet, max_bet) = compute_effective_range(&game_state);

        // min_bet should be big blind
        assert_eq!(min_bet, 10.0);

        // max_bet should be second largest stack (150.0)
        assert_eq!(max_bet, 150.0);
    }

    #[test]
    fn test_compute_effective_range_two_players() {
        let game_state = GameStateBuilder::new()
            .stacks(vec![100.0, 200.0])
            .blinds(10.0, 5.0)
            .build()
            .unwrap();

        let (min_bet, max_bet) = compute_effective_range(&game_state);

        assert_eq!(min_bet, 10.0);
        // Second largest is 100.0
        assert_eq!(max_bet, 100.0);
    }

    #[test]
    fn test_config_from_game_state() {
        let game_state = GameStateBuilder::new()
            .stacks(vec![100.0, 200.0])
            .blinds(10.0, 5.0)
            .build()
            .unwrap();

        let config = ActionIndexMapperConfig::from_game_state(&game_state);

        assert_eq!(config.min_bet, 10.0);
        assert_eq!(config.max_bet, 100.0);
    }

    #[test]
    fn test_mapper_from_game_state() {
        let game_state = GameStateBuilder::new()
            .stacks(vec![100.0, 200.0])
            .blinds(10.0, 5.0)
            .build()
            .unwrap();

        let mapper = ActionIndexMapper::from_game_state(&game_state);

        assert_eq!(mapper.config().min_bet, 10.0);
        assert_eq!(mapper.config().max_bet, 100.0);
    }

    // === Edge case tests ===

    #[test]
    fn test_small_pot_edge_case() {
        // Very small pot where all bets might cluster at low indices
        let game_state = GameStateBuilder::new()
            .stacks(vec![10.0, 10.0])
            .blinds(1.0, 0.5)
            .build()
            .unwrap();

        let mapper = ActionIndexMapper::from_game_state(&game_state);

        // Should still produce valid indices
        let idx = mapper.action_to_idx(&AgentAction::Bet(5.0), &game_state);
        assert!((ACTION_IDX_RAISE_MIN..=ACTION_IDX_RAISE_MAX).contains(&idx));
    }

    #[test]
    fn test_all_in_close_to_min_raise() {
        // Edge case where all-in is very close to min raise
        let game_state = GameStateBuilder::new()
            .stacks(vec![12.0, 100.0])
            .blinds(10.0, 5.0)
            .build()
            .unwrap();

        let mapper = ActionIndexMapper::from_game_state(&game_state);

        // All-in should still map to all-in index
        assert_eq!(
            mapper.action_to_idx(&AgentAction::AllIn, &game_state),
            ACTION_IDX_ALL_IN
        );

        // Bet exactly equal to all-in amount should also map to all-in
        let all_in_amount =
            game_state.current_round_current_player_bet() + game_state.current_player_stack();
        assert_eq!(
            mapper.action_to_idx(&AgentAction::Bet(all_in_amount), &game_state),
            ACTION_IDX_ALL_IN
        );
    }

    #[test]
    fn test_index_to_bet_returns_none_for_special_indices() {
        let mapper = create_mapper();

        assert!(mapper.index_to_bet(ACTION_IDX_FOLD).is_none());
        assert!(mapper.index_to_bet(ACTION_IDX_CALL).is_none());
        assert!(mapper.index_to_bet(ACTION_IDX_ALL_IN).is_none());
    }

    #[test]
    fn test_index_to_bet_returns_none_for_out_of_range() {
        let mapper = create_mapper();

        assert!(mapper.index_to_bet(52).is_none());
        assert!(mapper.index_to_bet(100).is_none());
    }

    #[test]
    fn test_num_action_indices_constant() {
        assert_eq!(NUM_ACTION_INDICES, 52);
    }
}
