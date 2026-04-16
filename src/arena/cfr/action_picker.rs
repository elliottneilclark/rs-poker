use rand::RngExt;
use tracing::event;

use crate::arena::{GameState, action::AgentAction};

use super::{ActionIndexMapper, NodeData, action_bit_set::ActionBitSet};
use little_sorry::{PcfrPlusRegretMatcher, RegretMinimizer};

/// Stack-capacity cap for deduped action sets. In practice CFR action
/// generators produce ≤ 10 actions (fold, call, a handful of bet sizes,
/// all-in), so 16 leaves generous headroom without a 52-slot buffer.
const MAX_DEDUPED_ACTIONS: usize = 16;

/// `'static` sentinel used to pre-populate the unused tail of
/// [`DedupedActions::entries`]. Never read (iteration is capped by `len`),
/// but lets us avoid `MaybeUninit` / `unsafe`.
static DEDUP_FALLBACK_ACTION: AgentAction = AgentAction::Fold;

/// Stack-allocated buffer of deduped `(index, action)` pairs.
///
/// Avoids heap allocations on the CFR hot path. Unused tail slots hold a
/// `'static` fallback so the whole array can be safely indexed without
/// `MaybeUninit` or `unsafe`; iteration is capped by `len`.
struct DedupedActions<'a> {
    entries: [(u8, &'a AgentAction); MAX_DEDUPED_ACTIONS],
    len: usize,
}

impl<'a> DedupedActions<'a> {
    fn from_slice(
        actions: &'a [AgentAction],
        mapper: &ActionIndexMapper,
        game_state: &GameState,
    ) -> Self {
        let mut seen = ActionBitSet::new();
        let mut entries: [(u8, &'a AgentAction); MAX_DEDUPED_ACTIONS] =
            [(0, &DEDUP_FALLBACK_ACTION); MAX_DEDUPED_ACTIONS];
        let mut len = 0usize;
        for action in actions {
            let idx = mapper.action_to_idx(action, game_state);
            if seen.insert(idx) {
                debug_assert!(
                    len < MAX_DEDUPED_ACTIONS,
                    "action generator produced more than {MAX_DEDUPED_ACTIONS} distinct actions"
                );
                entries[len] = (idx as u8, action);
                len += 1;
            }
        }
        Self { entries, len }
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn get(&self, i: usize) -> (usize, &'a AgentAction) {
        debug_assert!(i < self.len);
        let (idx, action) = self.entries[i];
        (idx as usize, action)
    }
}

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
    regret_matcher: Option<&'a PcfrPlusRegretMatcher>,
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
        regret_matcher: Option<&'a PcfrPlusRegretMatcher>,
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
        // Different bet sizes can quantise to the same slot via the
        // logarithmic `ActionIndexMapper` (there are only 49 raise slots),
        // and `explore_all_actions` already dedupes by index before
        // training. Without the same dedupe here, two actions mapping to
        // the same index would each read `weights[idx]` and both
        // contribute that weight to the cumulative distribution — biasing
        // the sampler toward collided indices proportional to the
        // collision count.

        // No-matcher fast path: one inline pass with reservoir sampling
        // over deduped actions. Skips the dedupe buffer entirely.
        let Some(matcher) = self.regret_matcher else {
            return self.pick_uniform_reservoir(rng);
        };

        // Weighted path: materialise the deduped actions into a stack
        // buffer so we can do the first (total-weight) and second
        // (cumulative-sample) passes without recomputing indices, which
        // involves `ln()` calls for `Bet` variants.
        let valid = DedupedActions::from_slice(self.possible_actions, self.mapper, self.game_state);
        let len = valid.len();
        debug_assert!(len > 0, "ActionPicker must have at least one valid action");

        let weights = matcher.best_weight();
        let mut total_weight: f32 = 0.0;
        for i in 0..len {
            let (idx, _) = valid.get(i);
            total_weight += weights.get(idx).copied().unwrap_or(0.0).max(0.0);
        }

        // If all weights are zero (or very close), fall back to uniform over
        // the same deduped set (cheaper than re-walking `possible_actions`).
        if total_weight < 1e-10 {
            let chosen_idx = rng.random_range(0..len);
            event!(
                tracing::Level::DEBUG,
                chosen_idx = chosen_idx,
                "All weights zero, using uniform random"
            );
            return valid.get(chosen_idx).1.clone();
        }

        // Sample from the weighted distribution in a second pass.
        let random_value: f32 = rng.random::<f32>() * total_weight;
        let mut cumulative = 0.0f32;
        for i in 0..len {
            let (action_idx, action) = valid.get(i);
            cumulative += weights.get(action_idx).copied().unwrap_or(0.0).max(0.0);
            if random_value <= cumulative {
                event!(
                    tracing::Level::DEBUG,
                    action_idx = action_idx,
                    total_weight = total_weight,
                    "Selected action from regret matcher"
                );
                return action.clone();
            }
        }

        // Fallback to last action (shouldn't reach here due to floating point)
        valid.get(len - 1).1.clone()
    }

    /// Reservoir-sample a uniformly random deduped action in a single pass.
    ///
    /// Avoids the dedupe buffer entirely for the no-matcher hot path: each
    /// newly-seen index has a `1/n` chance of replacing the current pick,
    /// which produces a uniform distribution over deduped actions.
    fn pick_uniform_reservoir<R: rand::Rng>(&self, rng: &mut R) -> AgentAction {
        let mut seen = ActionBitSet::new();
        let mut count: u32 = 0;
        let mut chosen: Option<&AgentAction> = None;
        for action in self.possible_actions {
            let idx = self.mapper.action_to_idx(action, self.game_state);
            if seen.insert(idx) {
                count += 1;
                if rng.random_range(0..count) == 0 {
                    chosen = Some(action);
                }
            }
        }
        event!(
            tracing::Level::DEBUG,
            count = count,
            "No regret matcher, using uniform random"
        );
        chosen
            .expect("possible_actions must contain at least one action")
            .clone()
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
        // Walk `possible_actions` once, deduping with `ActionBitSet` and
        // tracking the highest-weighted action inline. This skips the
        // stack-buffer dedupe used by `pick_action` — we only need the
        // winner, not the full distribution.
        let mut seen = ActionBitSet::new();

        let Some(matcher) = self.regret_matcher else {
            // Return the first deduped action.
            for action in self.possible_actions {
                let idx = self.mapper.action_to_idx(action, self.game_state);
                if seen.insert(idx) {
                    return action.clone();
                }
            }
            unreachable!("possible_actions must contain at least one action");
        };

        let weights = matcher.best_weight();
        let mut best_weight = f32::NEG_INFINITY;
        let mut best: Option<&AgentAction> = None;
        for action in self.possible_actions {
            let idx = self.mapper.action_to_idx(action, self.game_state);
            if seen.insert(idx) {
                let w = weights.get(idx).copied().unwrap_or(0.0);
                if w > best_weight {
                    best_weight = w;
                    best = Some(action);
                }
            }
        }

        best.expect("possible_actions must contain at least one action")
            .clone()
    }
}

/// Get the regret matcher from player node data if available.
pub fn get_regret_matcher_from_node(node_data: &NodeData) -> Option<&PcfrPlusRegretMatcher> {
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
    use crate::arena::cfr::{
        ACTION_IDX_ALL_IN, ACTION_IDX_RAISE_MIN, ActionIndexMapperConfig, NUM_ACTION_INDICES,
    };

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
        let mut matcher = PcfrPlusRegretMatcher::new(NUM_ACTION_INDICES);

        // Update with rewards that heavily favor fold (index 0)
        let mut rewards = vec![0.0; NUM_ACTION_INDICES];
        rewards[0] = 100.0; // Fold
        rewards[1] = 0.0; // Call
        rewards[ACTION_IDX_ALL_IN] = 0.0; // All-in
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
        let mut matcher = PcfrPlusRegretMatcher::new(NUM_ACTION_INDICES);
        let mut rewards = vec![0.0; NUM_ACTION_INDICES];
        rewards[0] = 10.0; // Fold
        rewards[1] = 20.0; // Call
        rewards[ACTION_IDX_ALL_IN] = 100.0; // All-in
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
        let mut matcher = PcfrPlusRegretMatcher::new(NUM_ACTION_INDICES);
        let mut rewards = vec![0.0; NUM_ACTION_INDICES];
        rewards[0] = 1000.0; // Fold has high weight
        rewards[1] = 1.0; // Call has low weight
        rewards[ACTION_IDX_ALL_IN] = 1000.0; // All-in has high weight
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
        let matcher = PcfrPlusRegretMatcher::new(NUM_ACTION_INDICES);

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
        let mut matcher = PcfrPlusRegretMatcher::new(NUM_ACTION_INDICES);
        let mut rewards = vec![0.0; NUM_ACTION_INDICES];
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
            "AllIn should always map to ACTION_IDX_ALL_IN"
        );
    }

    #[test]
    fn test_pick_action_with_only_two_valid_actions() {
        let game_state = create_test_game_state();
        let mapper = create_mapper();

        // Scenario: only fold and call are valid (common after an all-in)
        let actions = vec![AgentAction::Fold, AgentAction::Bet(10.0)];

        // Matcher strongly prefers call (index 1 = check/call)
        let mut matcher = PcfrPlusRegretMatcher::new(NUM_ACTION_INDICES);
        let mut rewards = vec![0.0; NUM_ACTION_INDICES];
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
        let mut matcher = PcfrPlusRegretMatcher::new(NUM_ACTION_INDICES);
        let mut rewards = vec![0.0; NUM_ACTION_INDICES];
        rewards[0] = 10.0; // Fold
        let bet_idx = mapper.action_to_idx(&AgentAction::Bet(50.0), &game_state);
        rewards[bet_idx] = 10.0; // Same weight
        rewards[ACTION_IDX_ALL_IN] = 10.0; // All-in same weight
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

    /// Regression test for M1: if two actions in `possible_actions` map to
    /// the same index (e.g. two near-identical bet sizes that quantise to
    /// the same raise slot), the picker must treat them as a single
    /// distribution entry. Otherwise the same weight is added to the
    /// cumulative distribution twice and the sampler is biased toward
    /// the collided index.
    #[test]
    fn test_pick_action_dedupes_index_collisions() {
        let game_state = create_test_game_state();
        let mapper = create_mapper();

        // Deliberately construct two bet sizes close enough to share a
        // raise slot. We build the distribution so the raise index has
        // the same weight as the call index; if the picker double-counts
        // the raise it will be sampled with probability 2/3 instead of
        // 1/2.
        let bet_a = AgentAction::Bet(60.0);
        let bet_b = AgentAction::Bet(63.0);
        let bet_a_idx = mapper.action_to_idx(&bet_a, &game_state);
        let bet_b_idx = mapper.action_to_idx(&bet_b, &game_state);
        assert_eq!(
            bet_a_idx, bet_b_idx,
            "test setup: pick two bet sizes that collide on a raise slot"
        );

        let call_idx = mapper.action_to_idx(&AgentAction::Bet(10.0), &game_state);
        let actions = vec![AgentAction::Bet(10.0), bet_a.clone(), bet_b.clone()];

        let mut matcher = PcfrPlusRegretMatcher::new(NUM_ACTION_INDICES);
        let mut rewards = vec![0.0; NUM_ACTION_INDICES];
        rewards[call_idx] = 50.0;
        rewards[bet_a_idx] = 50.0;
        matcher.update_regret(&rewards);

        let picker = ActionPicker::new(&mapper, &actions, Some(&matcher), &game_state);
        let mut rng = create_seeded_rng();

        let mut raise_count = 0;
        let iterations = 2000;
        for _ in 0..iterations {
            if matches!(picker.pick_action(&mut rng), AgentAction::Bet(x) if x > 50.0) {
                raise_count += 1;
            }
        }

        // With dedupe, call and raise have equal cumulative weight, so
        // the raise should be picked ~50% of the time. Without dedupe
        // the raise gets 2x weight and is picked ~67% of the time. Use
        // a wide band so the test is stable under RNG seed changes.
        let raise_pct = (raise_count * 100) / iterations;
        assert!(
            (40..=60).contains(&raise_pct),
            "raise picked {raise_pct}% of the time; expected ~50% after dedupe (2000 samples)"
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
        let mut matcher = PcfrPlusRegretMatcher::new(NUM_ACTION_INDICES);
        let mut rewards = vec![0.0; NUM_ACTION_INDICES];
        rewards[0] = 1000.0; // Fold is great (but not available)
        rewards[ACTION_IDX_ALL_IN] = -1000.0; // All-in is terrible
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

        let mut matcher = PcfrPlusRegretMatcher::new(NUM_ACTION_INDICES);

        // Simulate multiple CFR iterations where Call wins +900 and Fold wins 0
        // Invalid actions get -100 penalty
        let call_reward = 900.0;
        let fold_reward = 0.0;
        let invalid_penalty = -100.0;

        let bet_idx = mapper.action_to_idx(&AgentAction::Bet(10.0), &game_state);

        for _ in 0..100 {
            let mut rewards = vec![invalid_penalty; NUM_ACTION_INDICES];
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

        let mut matcher = PcfrPlusRegretMatcher::new(NUM_ACTION_INDICES);

        let bet_idx = mapper.action_to_idx(&AgentAction::Bet(10.0), &game_state);

        // Both valid actions have equal reward
        for _ in 0..100 {
            let mut rewards = vec![-100.0; NUM_ACTION_INDICES];
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

        let mut matcher = PcfrPlusRegretMatcher::new(NUM_ACTION_INDICES);

        // Get the actual call index (should be 1 for ACTION_IDX_CALL)
        let bet_idx = mapper.action_to_idx(&AgentAction::Bet(10.0), &game_state);
        println!("bet_idx for Bet(10.0) = {}", bet_idx);

        // Clear winner: Call gets +900, Fold gets 0
        for i in 0..10 {
            let mut rewards = vec![-100.0; NUM_ACTION_INDICES];
            rewards[0] = 0.0; // Fold (idx 0)
            rewards[1] = 900.0; // Call (idx 1)

            matcher.update_regret(&rewards);

            let weights = matcher.best_weight();
            let fold_weight = weights[0];
            let call_weight = weights[1];
            // Sum only the truly invalid indices (2-50, excluding 51 for all-in)
            let invalid_weight: f32 = weights[ACTION_IDX_RAISE_MIN..ACTION_IDX_ALL_IN]
                .iter()
                .sum();
            let allin_weight = weights[ACTION_IDX_ALL_IN];

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
        let invalid_weight: f32 = final_weights[ACTION_IDX_RAISE_MIN..ACTION_IDX_ALL_IN]
            .iter()
            .sum();
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
        assert_eq!(
            allin_idx, ACTION_IDX_ALL_IN,
            "AllIn should map to ACTION_IDX_ALL_IN"
        );

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
