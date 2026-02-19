use tracing::event;

use crate::arena::{GameState, action::AgentAction};

use super::{ActionIndexMapper, NodeData};
use little_sorry::{DcfrPlusRegretMatcher, RegretMinimizer};

/// Picks an action from a set of possible actions using regret matching.
///
/// This struct encapsulates the logic for selecting an action based on:
/// 1. A set of valid actions for the current game state
/// 2. An optional regret matcher containing learned strategy weights
/// 3. The action index mapper for consistent action-to-index mapping
///
/// If no regret matcher is provided, actions are chosen uniformly at random.
pub struct ActionPicker<'a> {
    mapper: &'a ActionIndexMapper,
    possible_actions: &'a [AgentAction],
    regret_matcher: Option<&'a DcfrPlusRegretMatcher>,
    game_state: &'a GameState,
}

impl<'a> ActionPicker<'a> {
    /// Create a new ActionPicker.
    ///
    /// # Arguments
    ///
    /// * `mapper` - The action index mapper for consistent action-to-index mapping
    /// * `possible_actions` - The valid actions for the current game state
    /// * `regret_matcher` - Optional regret matcher containing learned strategy weights
    /// * `game_state` - The current game state (used for action-to-index mapping)
    pub fn new(
        mapper: &'a ActionIndexMapper,
        possible_actions: &'a [AgentAction],
        regret_matcher: Option<&'a DcfrPlusRegretMatcher>,
        game_state: &'a GameState,
    ) -> Self {
        debug_assert!(
            !possible_actions.is_empty(),
            "possible_actions should always contain at least one action"
        );

        Self {
            mapper,
            possible_actions,
            regret_matcher,
            game_state,
        }
    }

    /// Pick an action using the regret matcher's strategy weights.
    ///
    /// If no regret matcher is provided or all weights are zero, picks uniformly
    /// at random from the valid actions.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator for sampling
    ///
    /// # Returns
    ///
    /// The selected action
    pub fn pick_action<R: rand::Rng>(&self, rng: &mut R) -> AgentAction {
        // Build a mapping from action index to action for valid actions only
        let valid_actions: Vec<(usize, &AgentAction)> = self
            .possible_actions
            .iter()
            .map(|action| (self.mapper.action_to_idx(action, self.game_state), action))
            .collect();

        // If there's no regret matcher, use uniform random over valid actions
        let Some(matcher) = self.regret_matcher else {
            let chosen_idx = rng.random_range(0..valid_actions.len());
            event!(
                tracing::Level::DEBUG,
                chosen_idx = chosen_idx,
                "No regret matcher, using uniform random"
            );
            return valid_actions[chosen_idx].1.clone();
        };

        // Get the weights from the regret matcher (average strategy)
        let weights = matcher.best_weight();

        // Extract weights for only the valid action indices
        let valid_weights: Vec<f32> = valid_actions
            .iter()
            .map(|(idx, _)| weights.get(*idx).copied().unwrap_or(0.0).max(0.0))
            .collect();

        // Calculate total weight
        let total_weight: f32 = valid_weights.iter().sum();

        // If all weights are zero (or very close), use uniform distribution
        if total_weight < 1e-10 {
            let chosen_idx = rng.random_range(0..valid_actions.len());
            event!(
                tracing::Level::DEBUG,
                chosen_idx = chosen_idx,
                "All weights zero, using uniform random"
            );
            return valid_actions[chosen_idx].1.clone();
        }

        // Sample from the weighted distribution over valid actions
        let random_value: f32 = rng.random::<f32>() * total_weight;
        let mut cumulative = 0.0;
        for (i, weight) in valid_weights.iter().enumerate() {
            cumulative += weight;
            if random_value <= cumulative {
                event!(
                    tracing::Level::DEBUG,
                    action_idx = valid_actions[i].0,
                    weight = weight,
                    total_weight = total_weight,
                    "Selected action from regret matcher"
                );
                return valid_actions[i].1.clone();
            }
        }

        // Fallback to last action (shouldn't reach here due to floating point)
        valid_actions.last().unwrap().1.clone()
    }

    /// Pick an action deterministically based on the highest weight.
    ///
    /// This is useful for exploitation mode where we want to always play
    /// the best action rather than sampling.
    ///
    /// # Returns
    ///
    /// The action with the highest weight, or the first action if no regret matcher
    pub fn pick_best_action(&self) -> AgentAction {
        // Build a mapping from action index to action for valid actions only
        let valid_actions: Vec<(usize, &AgentAction)> = self
            .possible_actions
            .iter()
            .map(|action| (self.mapper.action_to_idx(action, self.game_state), action))
            .collect();

        // If there's no regret matcher, return first action
        let Some(matcher) = self.regret_matcher else {
            return valid_actions[0].1.clone();
        };

        // Get the weights from the regret matcher
        let weights = matcher.best_weight();

        // Find the action with the highest weight
        let mut best_idx = 0;
        let mut best_weight = f32::NEG_INFINITY;

        for (i, (action_idx, _)) in valid_actions.iter().enumerate() {
            let weight = weights.get(*action_idx).copied().unwrap_or(0.0);
            if weight > best_weight {
                best_weight = weight;
                best_idx = i;
            }
        }

        valid_actions[best_idx].1.clone()
    }
}

/// Get the regret matcher from player node data if available.
pub fn get_regret_matcher_from_node(node_data: &NodeData) -> Option<&DcfrPlusRegretMatcher> {
    if let NodeData::Player(pd) = node_data {
        pd.regret_matcher.as_ref().map(|rm| rm.as_ref())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::GameStateBuilder;
    use crate::arena::cfr::ActionIndexMapperConfig;

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

    fn create_seeded_rng() -> rand::rngs::StdRng {
        use rand::SeedableRng;
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_pick_action_uniform_no_regret_matcher() {
        let game_state = create_test_game_state();
        let mapper = create_mapper();
        let actions = vec![
            AgentAction::Fold,
            AgentAction::Bet(10.0),
            AgentAction::AllIn,
        ];
        let mut rng = create_seeded_rng();

        let picker = ActionPicker::new(&mapper, &actions, None, &game_state);

        // Should return one of the valid actions
        let picked = picker.pick_action(&mut rng);
        assert!(
            actions.contains(&picked),
            "Picked action {:?} should be in valid actions",
            picked
        );
    }

    #[test]
    fn test_pick_action_with_regret_matcher() {
        let game_state = create_test_game_state();
        let mapper = create_mapper();
        let actions = vec![
            AgentAction::Fold,
            AgentAction::Bet(10.0),
            AgentAction::AllIn,
        ];
        let mut rng = create_seeded_rng();

        // Create a regret matcher with 52 experts (our standard action space)
        let mut matcher = DcfrPlusRegretMatcher::new(52);

        // Update with rewards that heavily favor fold (index 0)
        let mut rewards = vec![0.0; 52];
        rewards[0] = 100.0; // Fold
        rewards[1] = 0.0; // Call
        rewards[51] = 0.0; // All-in
        matcher.update_regret(&rewards);

        let picker = ActionPicker::new(&mapper, &actions, Some(&matcher), &game_state);

        // Run multiple times - fold should be chosen most often
        let mut fold_count = 0;
        for _ in 0..100 {
            let picked = picker.pick_action(&mut rng);
            if picked == AgentAction::Fold {
                fold_count += 1;
            }
        }

        // Fold should be chosen significantly more often than random (33%)
        assert!(
            fold_count > 50,
            "Fold should be chosen more often when it has high weight, got {}%",
            fold_count
        );
    }

    #[test]
    fn test_pick_best_action_no_regret_matcher() {
        let game_state = create_test_game_state();
        let mapper = create_mapper();
        let actions = vec![
            AgentAction::Fold,
            AgentAction::Bet(10.0),
            AgentAction::AllIn,
        ];

        let picker = ActionPicker::new(&mapper, &actions, None, &game_state);

        // Should return the first action when no regret matcher
        let picked = picker.pick_best_action();
        assert_eq!(picked, AgentAction::Fold);
    }

    #[test]
    fn test_pick_best_action_with_regret_matcher() {
        let game_state = create_test_game_state();
        let mapper = create_mapper();
        let actions = vec![
            AgentAction::Fold,
            AgentAction::Bet(10.0),
            AgentAction::AllIn,
        ];

        // Create a regret matcher that favors all-in
        let mut matcher = DcfrPlusRegretMatcher::new(52);
        let mut rewards = vec![0.0; 52];
        rewards[0] = 10.0; // Fold
        rewards[1] = 20.0; // Call
        rewards[51] = 100.0; // All-in
        matcher.update_regret(&rewards);

        let picker = ActionPicker::new(&mapper, &actions, Some(&matcher), &game_state);

        // Should deterministically return all-in
        let picked = picker.pick_best_action();
        assert_eq!(picked, AgentAction::AllIn);
    }

    #[test]
    fn test_filters_to_valid_actions_only() {
        let game_state = create_test_game_state();
        let mapper = create_mapper();

        // Only one valid action
        let actions = vec![AgentAction::Bet(10.0)];

        // Create a regret matcher that would favor other actions
        let mut matcher = DcfrPlusRegretMatcher::new(52);
        let mut rewards = vec![0.0; 52];
        rewards[0] = 1000.0; // Fold has high weight
        rewards[1] = 1.0; // Call has low weight
        rewards[51] = 1000.0; // All-in has high weight
        matcher.update_regret(&rewards);

        let picker = ActionPicker::new(&mapper, &actions, Some(&matcher), &game_state);
        let mut rng = create_seeded_rng();

        // Must return the only valid action
        let picked = picker.pick_action(&mut rng);
        assert_eq!(picked, AgentAction::Bet(10.0));
    }

    #[test]
    fn test_handles_zero_weights() {
        let game_state = create_test_game_state();
        let mapper = create_mapper();
        let actions = vec![
            AgentAction::Fold,
            AgentAction::Bet(10.0),
            AgentAction::AllIn,
        ];
        let mut rng = create_seeded_rng();

        // Create a regret matcher with all zero weights
        let matcher = DcfrPlusRegretMatcher::new(52);

        let picker = ActionPicker::new(&mapper, &actions, Some(&matcher), &game_state);

        // Should still return a valid action (uniform random)
        let picked = picker.pick_action(&mut rng);
        assert!(
            actions.contains(&picked),
            "Picked action {:?} should be in valid actions",
            picked
        );
    }

    #[test]
    fn test_pick_best_action_deterministic() {
        let game_state = create_test_game_state();
        let mapper = create_mapper();
        let actions = vec![
            AgentAction::Fold,
            AgentAction::Bet(50.0),
            AgentAction::AllIn,
        ];

        // Create a regret matcher that strongly favors fold
        let mut matcher = DcfrPlusRegretMatcher::new(52);
        let mut rewards = vec![0.0; 52];
        rewards[0] = 1000.0; // Fold
        matcher.update_regret(&rewards);

        let picker = ActionPicker::new(&mapper, &actions, Some(&matcher), &game_state);

        // pick_best_action should always return fold
        for _ in 0..10 {
            let picked = picker.pick_best_action();
            assert_eq!(
                picked,
                AgentAction::Fold,
                "Best action should always be fold"
            );
        }
    }

    #[test]
    fn test_different_bet_amounts_map_correctly() {
        let game_state = create_test_game_state();
        let mapper = create_mapper();

        // Different bet amounts should map to different indices
        let small_bet = AgentAction::Bet(15.0);
        let medium_bet = AgentAction::Bet(50.0);
        let large_bet = AgentAction::Bet(90.0);

        let small_idx = mapper.action_to_idx(&small_bet, &game_state);
        let medium_idx = mapper.action_to_idx(&medium_bet, &game_state);
        let large_idx = mapper.action_to_idx(&large_bet, &game_state);

        // All should be in the raise range (2-50)
        assert!(
            (2..=50).contains(&small_idx),
            "Small bet index {} should be in range 2-50",
            small_idx
        );
        assert!(
            (2..=50).contains(&medium_idx),
            "Medium bet index {} should be in range 2-50",
            medium_idx
        );
        assert!(
            (2..=50).contains(&large_idx),
            "Large bet index {} should be in range 2-50",
            large_idx
        );

        // Should be ordered (logarithmic distribution)
        assert!(
            small_idx < medium_idx,
            "Small bet index {} should be less than medium {}",
            small_idx,
            medium_idx
        );
        assert!(
            medium_idx < large_idx,
            "Medium bet index {} should be less than large {}",
            medium_idx,
            large_idx
        );
    }

    #[test]
    fn test_fold_and_allin_always_same_index() {
        use crate::arena::cfr::{ACTION_IDX_ALL_IN, ACTION_IDX_FOLD};

        let game_state = create_test_game_state();
        let mapper = create_mapper();

        // Fold always maps to 0
        let fold_idx = mapper.action_to_idx(&AgentAction::Fold, &game_state);
        assert_eq!(
            fold_idx, ACTION_IDX_FOLD,
            "Fold should always map to index 0"
        );

        // AllIn always maps to 51
        let allin_idx = mapper.action_to_idx(&AgentAction::AllIn, &game_state);
        assert_eq!(
            allin_idx, ACTION_IDX_ALL_IN,
            "AllIn should always map to index 51"
        );
    }

    #[test]
    fn test_pick_action_with_only_two_valid_actions() {
        let game_state = create_test_game_state();
        let mapper = create_mapper();

        // Scenario: only fold and call are valid (common after an all-in)
        let actions = vec![AgentAction::Fold, AgentAction::Bet(10.0)];

        // Matcher strongly prefers call (index 1 = check/call)
        let mut matcher = DcfrPlusRegretMatcher::new(52);
        let mut rewards = vec![0.0; 52];
        rewards[0] = -50.0; // Fold is bad
        rewards[1] = 50.0; // Call is good

        // Note: Bet(10.0) maps to a raise index, not call index.
        // Let's update for the actual index the bet maps to.
        let bet_idx = mapper.action_to_idx(&AgentAction::Bet(10.0), &game_state);
        rewards[bet_idx] = 50.0;

        matcher.update_regret(&rewards);

        let picker = ActionPicker::new(&mapper, &actions, Some(&matcher), &game_state);
        let mut rng = create_seeded_rng();

        // Bet should be chosen more often than fold
        let mut bet_count = 0;
        for _ in 0..100 {
            let picked = picker.pick_action(&mut rng);
            if matches!(picked, AgentAction::Bet(_)) {
                bet_count += 1;
            }
        }

        assert!(
            bet_count > 60,
            "Bet should be chosen more often when it has higher weight, got {}%",
            bet_count
        );
    }

    #[test]
    fn test_pick_best_handles_ties() {
        let game_state = create_test_game_state();
        let mapper = create_mapper();
        let actions = vec![
            AgentAction::Fold,
            AgentAction::Bet(50.0),
            AgentAction::AllIn,
        ];

        // All actions have equal weight
        let mut matcher = DcfrPlusRegretMatcher::new(52);
        let mut rewards = vec![0.0; 52];
        rewards[0] = 10.0; // Fold
        let bet_idx = mapper.action_to_idx(&AgentAction::Bet(50.0), &game_state);
        rewards[bet_idx] = 10.0; // Same weight
        rewards[51] = 10.0; // All-in same weight
        matcher.update_regret(&rewards);

        let picker = ActionPicker::new(&mapper, &actions, Some(&matcher), &game_state);

        // With ties, should return first encountered (fold at index 0)
        let picked = picker.pick_best_action();
        assert_eq!(
            picked,
            AgentAction::Fold,
            "On ties, should return first action with highest weight"
        );
    }

    #[test]
    fn test_single_action_always_picked() {
        let game_state = create_test_game_state();
        let mapper = create_mapper();
        let mut rng = create_seeded_rng();

        // Only one action available
        let actions = vec![AgentAction::AllIn];

        // Even with a matcher that dislikes this action
        let mut matcher = DcfrPlusRegretMatcher::new(52);
        let mut rewards = vec![0.0; 52];
        rewards[0] = 1000.0; // Fold is great (but not available)
        rewards[51] = -1000.0; // All-in is terrible
        matcher.update_regret(&rewards);

        let picker = ActionPicker::new(&mapper, &actions, Some(&matcher), &game_state);

        // Must pick the only available action
        for _ in 0..10 {
            let picked = picker.pick_action(&mut rng);
            assert_eq!(
                picked,
                AgentAction::AllIn,
                "Must pick the only available action"
            );
        }
    }

    /// Test that CFR converges quickly when one action is clearly better.
    /// This simulates the scenario from test_should_go_all_in where calling
    /// with the nuts should always be chosen over folding.
    #[test]
    fn test_convergence_with_clear_winner() {
        let game_state = create_test_game_state();
        let mapper = create_mapper();

        // Scenario: Fold (idx 0) or Call (idx 1), only these are valid
        let actions = vec![AgentAction::Fold, AgentAction::Bet(10.0)]; // Bet(current_bet) = Call

        let mut matcher = DcfrPlusRegretMatcher::new(52);

        // Simulate multiple CFR iterations where Call wins +900 and Fold wins 0
        // Invalid actions get -100 penalty
        let call_reward = 900.0;
        let fold_reward = 0.0;
        let invalid_penalty = -100.0;

        let bet_idx = mapper.action_to_idx(&AgentAction::Bet(10.0), &game_state);

        for _ in 0..100 {
            let mut rewards = vec![invalid_penalty; 52];
            rewards[0] = fold_reward;
            rewards[bet_idx] = call_reward;
            matcher.update_regret(&rewards);
        }

        // After 100 iterations, the strategy should heavily favor Call
        let picker = ActionPicker::new(&mapper, &actions, Some(&matcher), &game_state);
        let mut rng = create_seeded_rng();

        // Count how often Call is picked (should be >95%)
        let mut call_count = 0;
        for _ in 0..1000 {
            let picked = picker.pick_action(&mut rng);
            if matches!(picked, AgentAction::Bet(_)) {
                call_count += 1;
            }
        }

        assert!(
            call_count > 900,
            "Call should be chosen >90% of the time with clear reward advantage, got {}%",
            call_count / 10
        );
    }

    /// Test convergence when both actions have equal reward.
    /// Should converge to roughly 50/50.
    #[test]
    fn test_convergence_equal_rewards() {
        let game_state = create_test_game_state();
        let mapper = create_mapper();

        let actions = vec![AgentAction::Fold, AgentAction::Bet(10.0)];

        let mut matcher = DcfrPlusRegretMatcher::new(52);

        let bet_idx = mapper.action_to_idx(&AgentAction::Bet(10.0), &game_state);

        // Both valid actions have equal reward
        for _ in 0..100 {
            let mut rewards = vec![-100.0; 52];
            rewards[0] = 50.0;
            rewards[bet_idx] = 50.0;
            matcher.update_regret(&rewards);
        }

        let picker = ActionPicker::new(&mapper, &actions, Some(&matcher), &game_state);
        let mut rng = create_seeded_rng();

        let mut call_count = 0;
        for _ in 0..1000 {
            let picked = picker.pick_action(&mut rng);
            if matches!(picked, AgentAction::Bet(_)) {
                call_count += 1;
            }
        }

        // Should be roughly 50/50, allow 30-70 range
        assert!(
            (300..=700).contains(&call_count),
            "With equal rewards, should be roughly 50/50, got {}%",
            call_count / 10
        );
    }

    /// Debug test to understand the weights after CFR iterations.
    #[test]
    fn test_debug_weights_after_iterations() {
        let game_state = create_test_game_state();
        let mapper = create_mapper();

        let mut matcher = DcfrPlusRegretMatcher::new(52);

        // Get the actual call index (should be 1 for ACTION_IDX_CALL)
        let bet_idx = mapper.action_to_idx(&AgentAction::Bet(10.0), &game_state);
        println!("bet_idx for Bet(10.0) = {}", bet_idx);

        // Clear winner: Call gets +900, Fold gets 0
        for i in 0..10 {
            let mut rewards = vec![-100.0; 52];
            rewards[0] = 0.0; // Fold (idx 0)
            rewards[1] = 900.0; // Call (idx 1)

            matcher.update_regret(&rewards);

            let weights = matcher.best_weight();
            let fold_weight = weights[0];
            let call_weight = weights[1];
            // Sum only the truly invalid indices (2-50, excluding 51 for all-in)
            let invalid_weight: f32 = weights[2..51].iter().sum();
            let allin_weight = weights[51];

            println!(
                "Iteration {}: fold={:.4}, call={:.4}, invalid_sum={:.4}, allin={:.4}",
                i + 1,
                fold_weight,
                call_weight,
                invalid_weight,
                allin_weight
            );
        }

        let final_weights = matcher.best_weight();
        let fold_weight = final_weights[0];
        let call_weight = final_weights[1];

        // Call should have much higher weight
        assert!(
            call_weight > fold_weight * 2.0,
            "Call weight ({}) should be much higher than fold weight ({})",
            call_weight,
            fold_weight
        );

        // Invalid actions should have ~0 weight
        let invalid_weight: f32 = final_weights[2..51].iter().sum();
        assert!(
            invalid_weight < 0.01,
            "Invalid actions should have near-zero total weight, got {}",
            invalid_weight
        );
    }

    /// Test the actual scenario from test_should_go_all_in.
    /// Player 0 is all-in with 1000, Player 1 has 900 in stack and needs to decide.
    #[test]
    fn test_all_in_scenario_action_mapping() {
        use crate::arena::GameStateBuilder;
        use crate::arena::cfr::{ACTION_IDX_ALL_IN, ACTION_IDX_CALL, ACTION_IDX_FOLD};
        use crate::arena::game_state::{Round, RoundData};
        use crate::core::PlayerBitSet;

        // Recreate the game state from test_should_go_all_in
        let stacks: Vec<f32> = vec![0.0, 900.0];
        let player_bet = vec![1000.0, 100.0];
        let player_bet_round = vec![900.0, 0.0];
        let round_data = RoundData::new_with_bets(100.0, PlayerBitSet::new(2), 1, player_bet_round);
        let game_state = GameStateBuilder::new()
            .round(Round::River)
            .round_data(round_data)
            .stacks(stacks)
            .player_bet(player_bet)
            .big_blind(5.0)
            .small_blind(0.0)
            .build()
            .unwrap();

        let mapper = ActionIndexMapper::from_game_state(&game_state);

        // Verify current_round_bet
        let current_bet = game_state.current_round_bet();
        println!("current_round_bet = {}", current_bet);

        // Fold should map to 0
        let fold_idx = mapper.action_to_idx(&AgentAction::Fold, &game_state);
        assert_eq!(fold_idx, ACTION_IDX_FOLD, "Fold should map to index 0");

        // Call (matching current bet of 900) should map to 1
        let call_idx = mapper.action_to_idx(&AgentAction::Bet(900.0), &game_state);
        println!("Bet(900.0) maps to index {}", call_idx);
        assert_eq!(
            call_idx, ACTION_IDX_CALL,
            "Bet(900.0) matching current_round_bet should map to CALL (index 1)"
        );

        // All-in should map to 51
        let allin_idx = mapper.action_to_idx(&AgentAction::AllIn, &game_state);
        assert_eq!(allin_idx, ACTION_IDX_ALL_IN, "AllIn should map to index 51");

        // Player 1's all-in amount (bet = player_bet + stack = 100 + 900 = 1000)
        // But wait, Bet(1000) should map to AllIn since that's the player's all-in amount
        let player_allin =
            game_state.current_round_current_player_bet() + game_state.current_player_stack();
        println!("Player 1 all-in amount = {}", player_allin);

        // Bet(1000) should map to AllIn since it equals player's all-in
        let bet_1000_idx = mapper.action_to_idx(&AgentAction::Bet(1000.0), &game_state);
        println!("Bet(1000.0) maps to index {}", bet_1000_idx);
    }
}
