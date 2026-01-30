use rand::Rng;
use tracing::event;

use crate::arena::{GameState, action::AgentAction, game_state::Round};

use super::{CFRState, NodeData, TraversalState};

pub trait ActionGenerator {
    /// The configuration type for this action generator.
    /// Use `()` for generators that don't need configuration.
    type Config: Clone;

    /// Create a new action generator
    ///
    /// This is used by the Agent to create identical
    /// action generators for the historians it uses.
    fn new(cfr_state: CFRState, traversal_state: TraversalState, config: Self::Config) -> Self;

    /// Get a reference to the configuration
    fn config(&self) -> &Self::Config;

    /// Get a reference to the CFR state
    fn cfr_state(&self) -> &CFRState;

    /// Get a reference to the traversal state
    fn traversal_state(&self) -> &TraversalState;

    /// Given an action return the index of the action in the children array.
    ///
    /// # Arguments
    ///
    /// * `game_state` - The current game state
    /// * `action` - The action to convert to an index
    ///
    /// # Returns
    ///
    /// The index of the action in the children array. The 0 index is the fold
    /// action. All other are defined by the implentation
    fn action_to_idx(&self, game_state: &GameState, action: &AgentAction) -> usize;

    /// How many potential actions in total might be generated.
    ///
    /// At a given node there might be fewere that will be
    /// possible, but the regret matcher doesn't keep track of that.
    ///
    /// At all time the number of potential actions is
    /// larger than or equal to the number of possible actions
    ///
    /// # Returns
    ///
    /// The number of potential actions
    fn num_potential_actions(&self, game_state: &GameState) -> usize;

    // Generate all possible actions for the current game state
    //
    // This returns a vector so that the actions can be chosen from randomly
    fn gen_possible_actions(&self, game_state: &GameState) -> Vec<AgentAction>;

    /// Using the current and the CFR's tree's regret state choose a single action to
    /// play.
    ///
    /// This default implementation uses the regret matcher from the CFR state to
    /// select an action based on the current strategy. It filters the regret matcher's
    /// probability distribution to only include valid actions for the current game state.
    fn gen_action(&self, game_state: &GameState) -> AgentAction {
        let possible = self.gen_possible_actions(game_state);

        debug_assert!(
            !possible.is_empty(),
            "gen_possible_actions should always return at least one action"
        );

        // For now always use the thread rng.
        // At some point we will want to be able to pass seeded or deterministic action
        // choices.
        let mut rng = rand::rng();

        // Get target node index
        let from_node_idx = self.traversal_state().node_idx();
        let from_child_idx = self.traversal_state().chosen_child_idx();
        let target_node_idx = self
            .cfr_state()
            .get_child(from_node_idx, from_child_idx)
            .expect("Expected target node");

        // Get the node data
        let node_data = self
            .cfr_state()
            .get_node_data(target_node_idx)
            .expect("Expected target node data");

        if let NodeData::Player(pd) = &node_data {
            // Build a mapping from action index to action for valid actions only
            let valid_actions: Vec<(usize, AgentAction)> = possible
                .iter()
                .map(|action| (self.action_to_idx(game_state, action), action.clone()))
                .collect();

            // If there's no regret matcher yet, use uniform random over valid actions
            let Some(matcher) = pd.regret_matcher.as_ref() else {
                let chosen_idx = rng.random_range(0..valid_actions.len());
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
        } else {
            panic!("Expected player node");
        }
    }
}

/// Basic CFR action generator with fold, call/check, and all-in actions.
/// Also unsed by SimpleActionGenerator as a base.
const ACTION_FOLD: usize = 0;
const ACTION_CALL: usize = 1;

pub struct BasicCFRActionGenerator {
    cfr_state: CFRState,
    traversal_state: TraversalState,
}

impl BasicCFRActionGenerator {
    pub fn new(cfr_state: CFRState, traversal_state: TraversalState) -> Self {
        BasicCFRActionGenerator {
            cfr_state,
            traversal_state,
        }
    }
}

impl ActionGenerator for BasicCFRActionGenerator {
    type Config = ();

    fn new(cfr_state: CFRState, traversal_state: TraversalState, _config: ()) -> Self {
        BasicCFRActionGenerator {
            cfr_state,
            traversal_state,
        }
    }

    fn config(&self) -> &() {
        &()
    }

    fn cfr_state(&self) -> &CFRState {
        &self.cfr_state
    }

    fn traversal_state(&self) -> &TraversalState {
        &self.traversal_state
    }

    fn gen_possible_actions(&self, game_state: &GameState) -> Vec<AgentAction> {
        let mut res: Vec<AgentAction> = Vec::with_capacity(3);
        let to_call =
            game_state.current_round_bet() - game_state.current_round_current_player_bet();
        if to_call > 0.0 {
            res.push(AgentAction::Fold);
        }
        // Call, Match the current bet (if the bet is 0 this is a check)
        res.push(AgentAction::Bet(game_state.current_round_bet()));

        let all_in_ammount =
            game_state.current_round_current_player_bet() + game_state.current_player_stack();

        if all_in_ammount > game_state.current_round_bet() {
            // All-in, Bet all the money
            // Bet everything we have bet so far plus the remaining stack
            res.push(AgentAction::AllIn);
        }
        res
    }

    fn action_to_idx(&self, _game_state: &GameState, action: &AgentAction) -> usize {
        match action {
            AgentAction::Fold => ACTION_FOLD,
            AgentAction::Call => ACTION_CALL,
            AgentAction::Bet(_) => ACTION_CALL,
            AgentAction::AllIn => 2,
        }
    }

    fn num_potential_actions(&self, _game_state: &GameState) -> usize {
        3
    }
}

/// Number of potential actions for SimpleActionGenerator
const SIMPLE_NUM_ACTIONS: usize = 6;

// Action indices for SimpleActionGenerator
//
// ACTION_FOLD = 0
// ACTION_CALL = 1
// These are from basic generator and are by convention
const SIMPLE_ACTION_MIN_RAISE: usize = 2;
const SIMPLE_ACTION_POT_33: usize = 3;
const SIMPLE_ACTION_POT_66: usize = 4;
const SIMPLE_ACTION_ALL_IN: usize = 5; // By convention ALL IN is last. So it will be different for different generators.

/// Action generator with more betting options: fold, check/call, min raise,
/// 33% pot, 66% pot, and all-in.
///
/// This provides a richer action space for CFR exploration compared to
/// BasicCFRActionGenerator which only has fold, call, and all-in.
pub struct SimpleActionGenerator {
    cfr_state: CFRState,
    traversal_state: TraversalState,
}

impl SimpleActionGenerator {
    pub fn new(cfr_state: CFRState, traversal_state: TraversalState) -> Self {
        SimpleActionGenerator {
            cfr_state,
            traversal_state,
        }
    }
}

impl ActionGenerator for SimpleActionGenerator {
    type Config = ();

    fn new(cfr_state: CFRState, traversal_state: TraversalState, _config: ()) -> Self {
        SimpleActionGenerator {
            cfr_state,
            traversal_state,
        }
    }

    fn config(&self) -> &() {
        &()
    }

    fn cfr_state(&self) -> &CFRState {
        &self.cfr_state
    }

    fn traversal_state(&self) -> &TraversalState {
        &self.traversal_state
    }

    fn gen_possible_actions(&self, game_state: &GameState) -> Vec<AgentAction> {
        let mut actions: Vec<AgentAction> = Vec::with_capacity(SIMPLE_NUM_ACTIONS);

        let current_bet = game_state.current_round_bet();
        let player_bet = game_state.current_round_current_player_bet();
        let stack = game_state.current_player_stack();
        let pot = game_state.total_pot;
        let min_raise = game_state.current_round_min_raise();
        let to_call = current_bet - player_bet;

        // All-in amount
        let all_in_amount = player_bet + stack;
        // Minimum valid raise amount
        let min_raise_amount = current_bet + min_raise;

        // Fold - only if there's something to call
        if to_call > 0.0 {
            actions.push(AgentAction::Fold);
        }

        // Call/Check - always available
        actions.push(AgentAction::Bet(current_bet));

        // Min raise = current_bet + min_raise
        if min_raise_amount > current_bet && min_raise_amount < all_in_amount {
            actions.push(AgentAction::Bet(min_raise_amount));
        }

        // 33% pot raise = current_bet + pot * 0.33
        // Must be at least min raise and greater than the min raise bet
        let pot_33_amount = current_bet + pot * 0.33;
        if pot_33_amount >= min_raise_amount
            && pot_33_amount > min_raise_amount
            && pot_33_amount < all_in_amount
        {
            actions.push(AgentAction::Bet(pot_33_amount));
        }

        // 66% pot raise = current_bet + pot * 0.66
        // Must be at least min raise and greater than 33% pot
        let pot_66_amount = current_bet + pot * 0.66;
        if pot_66_amount >= min_raise_amount
            && pot_66_amount > pot_33_amount
            && pot_66_amount < all_in_amount
        {
            actions.push(AgentAction::Bet(pot_66_amount));
        }

        // All-in - only if we can bet more than the current bet
        if all_in_amount > current_bet {
            actions.push(AgentAction::AllIn);
        }

        actions
    }

    fn action_to_idx(&self, game_state: &GameState, action: &AgentAction) -> usize {
        let current_bet = game_state.current_round_bet();
        let player_bet = game_state.current_round_current_player_bet();
        let stack = game_state.current_player_stack();
        let pot = game_state.total_pot;
        let min_raise = game_state.current_round_min_raise();

        let min_raise_amount = current_bet + min_raise;
        let pot_33_amount = current_bet + pot * 0.33;
        let pot_66_amount = current_bet + pot * 0.66;
        let all_in_amount = player_bet + stack;

        match action {
            AgentAction::Fold => ACTION_FOLD,
            AgentAction::Call => ACTION_CALL,
            AgentAction::AllIn => SIMPLE_ACTION_ALL_IN,
            AgentAction::Bet(amount) => {
                // Use small epsilon for floating point comparison
                let epsilon = 0.01;

                if (amount - current_bet).abs() < epsilon {
                    ACTION_CALL
                } else if (amount - min_raise_amount).abs() < epsilon {
                    SIMPLE_ACTION_MIN_RAISE
                } else if (amount - pot_33_amount).abs() < epsilon {
                    SIMPLE_ACTION_POT_33
                } else if (amount - pot_66_amount).abs() < epsilon {
                    SIMPLE_ACTION_POT_66
                } else if (amount - all_in_amount).abs() < epsilon {
                    SIMPLE_ACTION_ALL_IN
                } else {
                    // Fallback: find closest match
                    let amounts = [
                        (ACTION_CALL, current_bet),
                        (SIMPLE_ACTION_MIN_RAISE, min_raise_amount),
                        (SIMPLE_ACTION_POT_33, pot_33_amount),
                        (SIMPLE_ACTION_POT_66, pot_66_amount),
                        (SIMPLE_ACTION_ALL_IN, all_in_amount),
                    ];
                    amounts
                        .iter()
                        .min_by(|(_, a), (_, b)| {
                            (amount - a).abs().partial_cmp(&(amount - b).abs()).unwrap()
                        })
                        .map(|(idx, _)| *idx)
                        .unwrap_or(ACTION_CALL)
                }
            }
        }
    }

    fn num_potential_actions(&self, _game_state: &GameState) -> usize {
        SIMPLE_NUM_ACTIONS
    }
}

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

    /// Get the maximum number of raise multipliers across all rounds
    fn max_raise_mult_len(&self) -> usize {
        let mut max = self.default.raise_mult.len();
        if let Some(ref cfg) = self.preflop {
            max = max.max(cfg.raise_mult.len());
        }
        if let Some(ref cfg) = self.flop {
            max = max.max(cfg.raise_mult.len());
        }
        if let Some(ref cfg) = self.turn {
            max = max.max(cfg.raise_mult.len());
        }
        if let Some(ref cfg) = self.river {
            max = max.max(cfg.raise_mult.len());
        }
        max
    }

    /// Get the maximum number of pot multipliers across all rounds
    fn max_pot_mult_len(&self) -> usize {
        let mut max = self.default.pot_mult.len();
        if let Some(ref cfg) = self.preflop {
            max = max.max(cfg.pot_mult.len());
        }
        if let Some(ref cfg) = self.flop {
            max = max.max(cfg.pot_mult.len());
        }
        if let Some(ref cfg) = self.turn {
            max = max.max(cfg.pot_mult.len());
        }
        if let Some(ref cfg) = self.river {
            max = max.max(cfg.pot_mult.len());
        }
        max
    }

    /// Check if any round has call enabled
    fn any_call_enabled(&self) -> bool {
        if self.default.call_enabled {
            return true;
        }
        if let Some(ref cfg) = self.preflop
            && cfg.call_enabled
        {
            return true;
        }
        if let Some(ref cfg) = self.flop
            && cfg.call_enabled
        {
            return true;
        }
        if let Some(ref cfg) = self.turn
            && cfg.call_enabled
        {
            return true;
        }
        if let Some(ref cfg) = self.river
            && cfg.call_enabled
        {
            return true;
        }
        false
    }

    /// Check if any round has setup shove enabled
    fn any_setup_shove(&self) -> bool {
        if self.default.setup_shove {
            return true;
        }
        if let Some(ref cfg) = self.preflop
            && cfg.setup_shove
        {
            return true;
        }
        if let Some(ref cfg) = self.flop
            && cfg.setup_shove
        {
            return true;
        }
        if let Some(ref cfg) = self.turn
            && cfg.setup_shove
        {
            return true;
        }
        if let Some(ref cfg) = self.river
            && cfg.setup_shove
        {
            return true;
        }
        false
    }

    /// Check if any round has all-in enabled
    fn any_all_in(&self) -> bool {
        if self.default.all_in {
            return true;
        }
        if let Some(ref cfg) = self.preflop
            && cfg.all_in
        {
            return true;
        }
        if let Some(ref cfg) = self.flop
            && cfg.all_in
        {
            return true;
        }
        if let Some(ref cfg) = self.turn
            && cfg.all_in
        {
            return true;
        }
        if let Some(ref cfg) = self.river
            && cfg.all_in
        {
            return true;
        }
        false
    }
}

/// Configurable action generator with per-round bet sizing options.
///
/// This generator allows users to specify:
/// - Per-round betting options (raise multiples, pot multiples)
/// - Enable/disable check/call and all-in
/// - "Setup shove" action (bet so pot + call = remaining stack)
///
/// ## Index Layout
///
/// Indices are deterministic based on max possible actions from config:
///
/// | Index | Action Type |
/// |-------|-------------|
/// | 0 | Fold (always) |
/// | 1 | Call (if any round has call_enabled) |
/// | 2..2+R | Raise mult slots (R = max raise_mult.len()) |
/// | 2+R..2+R+P | Pot mult slots (P = max pot_mult.len()) |
/// | 2+R+P | Setup shove (if any round has setup_shove) |
/// | 2+R+P+1 | All-in (if any round has all_in) |
pub struct ConfigurableActionGenerator {
    cfr_state: CFRState,
    traversal_state: TraversalState,
    config: ConfigurableActionConfig,
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
            config,
        }
    }

    /// Index layout helpers - returns the base index for each action type
    fn call_index(&self) -> usize {
        1 // Call is always at index 1 (after fold)
    }

    fn raise_mult_base_index(&self) -> usize {
        2 // Raise mults start at index 2
    }

    fn pot_mult_base_index(&self) -> usize {
        self.raise_mult_base_index() + self.config.max_raise_mult_len()
    }

    fn setup_shove_index(&self) -> usize {
        self.pot_mult_base_index() + self.config.max_pot_mult_len()
    }

    fn all_in_index(&self) -> usize {
        let base = self.setup_shove_index();
        if self.config.any_setup_shove() {
            base + 1
        } else {
            base
        }
    }
}

impl ActionGenerator for ConfigurableActionGenerator {
    type Config = ConfigurableActionConfig;

    fn new(cfr_state: CFRState, traversal_state: TraversalState, config: Self::Config) -> Self {
        ConfigurableActionGenerator::new_with_config(cfr_state, traversal_state, config)
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

    fn action_to_idx(&self, game_state: &GameState, action: &AgentAction) -> usize {
        let current_bet = game_state.current_round_bet();
        let player_bet = game_state.current_round_current_player_bet();
        let stack = game_state.current_player_stack();
        let pot = game_state.total_pot;
        let min_raise = game_state.current_round_min_raise();
        let all_in_amount = player_bet + stack;

        let round_config = self.config.round_config(game_state.round);
        let epsilon = 0.01;

        match action {
            AgentAction::Fold => ACTION_FOLD,
            AgentAction::Call => self.call_index(),
            AgentAction::AllIn => self.all_in_index(),
            AgentAction::Bet(amount) => {
                // Check if it's a call
                if (amount - current_bet).abs() < epsilon {
                    return self.call_index();
                }

                // Check if it's all-in
                if (amount - all_in_amount).abs() < epsilon {
                    return self.all_in_index();
                }

                // Check raise multipliers
                for (i, &mult) in round_config.raise_mult.iter().enumerate() {
                    let raise_amount = current_bet + min_raise * mult;
                    if (amount - raise_amount).abs() < epsilon {
                        return self.raise_mult_base_index() + i;
                    }
                }

                // Check pot multipliers
                for (i, &mult) in round_config.pot_mult.iter().enumerate() {
                    let pot_amount = current_bet + pot * mult;
                    if (amount - pot_amount).abs() < epsilon {
                        return self.pot_mult_base_index() + i;
                    }
                }

                // Check setup shove
                if round_config.setup_shove {
                    // Setup shove: bet so that pot + call = remaining stack after bet
                    // This means: pot + (bet - current_bet) = stack - (bet - player_bet)
                    // Simplifying: pot + bet - current_bet = stack - bet + player_bet
                    // 2*bet = stack + player_bet + current_bet - pot
                    // bet = (stack + player_bet + current_bet - pot) / 2
                    let setup_bet = (stack + player_bet + current_bet - pot) / 2.0;
                    if setup_bet > current_bet && (amount - setup_bet).abs() < epsilon {
                        return self.setup_shove_index();
                    }
                }

                // Fallback: find closest match among all possible amounts
                let mut best_idx = self.call_index();
                let mut best_diff = (amount - current_bet).abs();

                // Check raise multipliers
                for (i, &mult) in round_config.raise_mult.iter().enumerate() {
                    let raise_amount = current_bet + min_raise * mult;
                    let diff = (amount - raise_amount).abs();
                    if diff < best_diff {
                        best_diff = diff;
                        best_idx = self.raise_mult_base_index() + i;
                    }
                }

                // Check pot multipliers
                for (i, &mult) in round_config.pot_mult.iter().enumerate() {
                    let pot_amount = current_bet + pot * mult;
                    let diff = (amount - pot_amount).abs();
                    if diff < best_diff {
                        best_diff = diff;
                        best_idx = self.pot_mult_base_index() + i;
                    }
                }

                // Check all-in
                let diff = (amount - all_in_amount).abs();
                if diff < best_diff {
                    best_idx = self.all_in_index();
                }

                best_idx
            }
        }
    }

    fn num_potential_actions(&self, _game_state: &GameState) -> usize {
        // fold + call + raise_mults + pot_mults + setup_shove? + all_in?
        let mut count = 1; // fold
        if self.config.any_call_enabled() {
            count += 1;
        }
        count += self.config.max_raise_mult_len();
        count += self.config.max_pot_mult_len();
        if self.config.any_setup_shove() {
            count += 1;
        }
        if self.config.any_all_in() {
            count += 1;
        }
        count
    }

    fn gen_possible_actions(&self, game_state: &GameState) -> Vec<AgentAction> {
        let mut actions: Vec<AgentAction> = Vec::new();
        let mut used_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();

        let current_bet = game_state.current_round_bet();
        let player_bet = game_state.current_round_current_player_bet();
        let stack = game_state.current_player_stack();
        let pot = game_state.total_pot;
        let min_raise = game_state.current_round_min_raise();
        let to_call = current_bet - player_bet;
        let all_in_amount = player_bet + stack;

        let round_config = self.config.round_config(game_state.round);

        // Helper to add a bet action if its index hasn't been used yet
        let try_add_bet = |action: AgentAction,
                           actions: &mut Vec<AgentAction>,
                           used: &mut std::collections::HashSet<usize>| {
            let idx = self.action_to_idx(game_state, &action);
            if !used.contains(&idx) {
                actions.push(action);
                used.insert(idx);
            }
        };

        // Fold - only if there's something to call
        if to_call > 0.0 {
            try_add_bet(AgentAction::Fold, &mut actions, &mut used_indices);
        }

        // Call/Check - if enabled
        if round_config.call_enabled {
            try_add_bet(
                AgentAction::Bet(current_bet),
                &mut actions,
                &mut used_indices,
            );
        }

        // Track the minimum valid raise for ordering
        let min_valid_raise = current_bet + min_raise;

        // Raise multipliers
        for &mult in &round_config.raise_mult {
            let raise_amount = current_bet + min_raise * mult;
            // Must be at least min raise and less than all-in
            if raise_amount >= min_valid_raise && raise_amount < all_in_amount {
                try_add_bet(
                    AgentAction::Bet(raise_amount),
                    &mut actions,
                    &mut used_indices,
                );
            }
        }

        // Pot multipliers
        for &mult in &round_config.pot_mult {
            let pot_amount = current_bet + pot * mult;
            // Must be at least min raise and less than all-in
            if pot_amount >= min_valid_raise && pot_amount < all_in_amount {
                try_add_bet(
                    AgentAction::Bet(pot_amount),
                    &mut actions,
                    &mut used_indices,
                );
            }
        }

        // Setup shove (bet so pot + call = remaining stack)
        if round_config.setup_shove {
            let setup_bet = (stack + player_bet + current_bet - pot) / 2.0;
            if setup_bet >= min_valid_raise && setup_bet < all_in_amount {
                try_add_bet(AgentAction::Bet(setup_bet), &mut actions, &mut used_indices);
            }
        }

        // All-in - if enabled and we can bet more than current bet
        if round_config.all_in && all_in_amount > current_bet {
            try_add_bet(AgentAction::AllIn, &mut actions, &mut used_indices);
        }

        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::arena::GameState;

    use std::vec;

    // === BasicCFRActionGenerator tests ===

    #[test]
    fn test_should_gen_2_actions() {
        let stacks = vec![50.0; 2];
        let game_state = GameState::new_starting(stacks, 2.0, 1.0, 0.0, 0);
        let action_generator = BasicCFRActionGenerator::new(
            CFRState::new(game_state.clone()),
            TraversalState::new_root(0),
        );
        let actions = action_generator.gen_possible_actions(&game_state);
        // We should have 2 actions: Call or All-in since 0 is the dealer when starting
        assert_eq!(actions.len(), 2);

        // None of the ations should have a child idx of 0
        for action in actions {
            assert_ne!(action_generator.action_to_idx(&game_state, &action), 0);
        }
    }

    #[test]
    fn test_should_gen_3_actions() {
        let stacks = vec![50.0; 2];
        let mut game_state = GameState::new_starting(stacks, 2.0, 1.0, 0.0, 0);
        game_state.advance_round();
        game_state.advance_round();

        game_state.do_bet(10.0, false).unwrap();
        let action_generator = BasicCFRActionGenerator::new(
            CFRState::new(game_state.clone()),
            TraversalState::new_root(0),
        );
        let actions = action_generator.gen_possible_actions(&game_state);
        // We should have 3 actions: Fold, Call, or All-in
        assert_eq!(actions.len(), 3);

        // Check the indices of the actions
        assert_eq!(
            action_generator.action_to_idx(&game_state, &AgentAction::Fold),
            0
        );
        assert_eq!(
            action_generator.action_to_idx(&game_state, &AgentAction::Bet(10.0)),
            1
        );
        assert_eq!(
            action_generator.action_to_idx(&game_state, &AgentAction::AllIn),
            2
        );
    }

    // === SimpleActionGenerator tests ===

    fn create_simple_generator(game_state: &GameState) -> SimpleActionGenerator {
        SimpleActionGenerator::new(
            CFRState::new(game_state.clone()),
            TraversalState::new_root(0),
        )
    }

    #[test]
    fn test_simple_action_indices() {
        // Verify the action index constants
        assert_eq!(ACTION_FOLD, 0);
        assert_eq!(ACTION_CALL, 1);
        assert_eq!(SIMPLE_ACTION_MIN_RAISE, 2);
        assert_eq!(SIMPLE_ACTION_POT_33, 3);
        assert_eq!(SIMPLE_ACTION_POT_66, 4);
        assert_eq!(SIMPLE_ACTION_ALL_IN, 5);
        assert_eq!(SIMPLE_NUM_ACTIONS, 6);
    }

    #[test]
    fn test_simple_fold_maps_to_zero() {
        let stacks = vec![100.0; 2];
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let action_gen = create_simple_generator(&game_state);

        assert_eq!(
            action_gen.action_to_idx(&game_state, &AgentAction::Fold),
            ACTION_FOLD
        );
    }

    #[test]
    fn test_simple_all_in_maps_to_five() {
        let stacks = vec![100.0; 2];
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let action_gen = create_simple_generator(&game_state);

        assert_eq!(
            action_gen.action_to_idx(&game_state, &AgentAction::AllIn),
            SIMPLE_ACTION_ALL_IN
        );
    }

    #[test]
    fn test_simple_call_maps_to_one() {
        let stacks = vec![100.0; 2];
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let action_gen = create_simple_generator(&game_state);

        // Call action
        assert_eq!(
            action_gen.action_to_idx(&game_state, &AgentAction::Call),
            ACTION_CALL
        );

        // Bet that matches current bet (same as call)
        let call_amount = game_state.current_round_bet();
        assert_eq!(
            action_gen.action_to_idx(&game_state, &AgentAction::Bet(call_amount)),
            ACTION_CALL
        );
    }

    #[test]
    fn test_simple_min_raise_mapping() {
        let stacks = vec![100.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        // Advance to flop so we have a clean state
        game_state.advance_round();

        // Player 0 bets 10
        game_state.do_bet(10.0, false).unwrap();

        let action_gen = create_simple_generator(&game_state);

        // Min raise = current_bet (10) + min_raise (10) = 20
        let min_raise_amount =
            game_state.current_round_bet() + game_state.current_round_min_raise();
        assert_eq!(
            action_gen.action_to_idx(&game_state, &AgentAction::Bet(min_raise_amount)),
            SIMPLE_ACTION_MIN_RAISE
        );
    }

    #[test]
    fn test_simple_pot_33_mapping() {
        let stacks = vec![500.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();

        // Player 0 bets 20
        game_state.do_bet(20.0, false).unwrap();

        let action_gen = create_simple_generator(&game_state);

        // Pot is now 35 (15 from blinds + 20 from bet)
        // 33% pot = current_bet (20) + pot (35) * 0.33 = 20 + 11.55 = 31.55
        let pot_33_amount = game_state.current_round_bet() + game_state.total_pot * 0.33;
        assert_eq!(
            action_gen.action_to_idx(&game_state, &AgentAction::Bet(pot_33_amount)),
            SIMPLE_ACTION_POT_33
        );
    }

    #[test]
    fn test_simple_pot_66_mapping() {
        let stacks = vec![500.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();

        // Player 0 bets 20
        game_state.do_bet(20.0, false).unwrap();

        let action_gen = create_simple_generator(&game_state);

        // 66% pot = current_bet (20) + pot (35) * 0.66
        let pot_66_amount = game_state.current_round_bet() + game_state.total_pot * 0.66;
        assert_eq!(
            action_gen.action_to_idx(&game_state, &AgentAction::Bet(pot_66_amount)),
            SIMPLE_ACTION_POT_66
        );
    }

    #[test]
    fn test_simple_gen_actions_with_bet_facing() {
        let stacks = vec![500.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();

        // Player 0 bets 30
        game_state.do_bet(30.0, false).unwrap();

        let action_gen = create_simple_generator(&game_state);
        let actions = action_gen.gen_possible_actions(&game_state);

        // Should have: Fold, Call, Min Raise, 33% pot, 66% pot, All-in
        // (assuming all bet sizes are distinct and valid)
        assert!(actions.contains(&AgentAction::Fold));
        assert!(actions.iter().any(|a| matches!(a, AgentAction::Bet(_))));
        assert!(actions.contains(&AgentAction::AllIn));

        // Verify each action maps to a unique index
        let mut indices: Vec<usize> = actions
            .iter()
            .map(|a| action_gen.action_to_idx(&game_state, a))
            .collect();
        indices.sort();
        indices.dedup();
        assert_eq!(
            indices.len(),
            actions.len(),
            "All actions should have unique indices"
        );
    }

    #[test]
    fn test_simple_no_fold_when_checking() {
        let stacks = vec![100.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        // Advance to flop where no one has bet yet
        game_state.advance_round();

        let action_gen = create_simple_generator(&game_state);
        let actions = action_gen.gen_possible_actions(&game_state);

        // No fold option when there's nothing to call
        assert!(!actions.contains(&AgentAction::Fold));
    }

    #[test]
    fn test_simple_num_potential_actions() {
        let stacks = vec![100.0; 2];
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let action_gen = create_simple_generator(&game_state);

        assert_eq!(
            action_gen.num_potential_actions(&game_state),
            SIMPLE_NUM_ACTIONS
        );
    }

    #[test]
    fn test_simple_all_actions_roundtrip() {
        // Test that all generated actions map back correctly
        let stacks = vec![1000.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();
        game_state.do_bet(50.0, false).unwrap();

        let action_gen = create_simple_generator(&game_state);
        let actions = action_gen.gen_possible_actions(&game_state);

        for action in &actions {
            let idx = action_gen.action_to_idx(&game_state, action);
            // Find action with this index
            let found = actions
                .iter()
                .find(|a| action_gen.action_to_idx(&game_state, a) == idx);
            assert!(
                found.is_some(),
                "Action {:?} mapped to idx {} but no action maps back",
                action,
                idx
            );
        }
    }

    // === SimpleActionGenerator validation tests ===
    // These tests verify that all generated actions are valid when applied to the game state

    /// Helper to verify all generated actions are valid by applying them to game state
    fn verify_all_actions_valid(game_state: &GameState) {
        let action_gen = create_simple_generator(game_state);
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
    fn test_simple_all_actions_valid_preflop_sb() {
        // Small blind facing big blind
        let stacks = vec![100.0; 2];
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_preflop_bb() {
        // Big blind after small blind completes
        let stacks = vec![100.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.do_bet(10.0, false).unwrap(); // SB completes to BB
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_flop_first_to_act() {
        // First to act on flop (can check)
        let stacks = vec![100.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round(); // To flop
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_facing_bet() {
        // Facing a bet on flop
        let stacks = vec![100.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();
        game_state.do_bet(20.0, false).unwrap(); // Opponent bets
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_facing_raise() {
        // Facing a raise
        let stacks = vec![200.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();
        game_state.do_bet(20.0, false).unwrap(); // Bet
        game_state.do_bet(50.0, false).unwrap(); // Raise
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_small_stack() {
        // Player with small stack
        let stacks = vec![30.0, 100.0];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();
        game_state.do_bet(15.0, false).unwrap();
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_large_pot() {
        // Large pot scenario where pot bets might exceed stack
        let stacks = vec![100.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        // Build up pot through betting rounds
        game_state.do_bet(10.0, false).unwrap(); // SB completes
        game_state.do_bet(30.0, false).unwrap(); // BB raises
        game_state.do_bet(30.0, false).unwrap(); // SB calls
        game_state.advance_round(); // To flop
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_tiny_pot() {
        // Tiny pot where pot-based bets might be smaller than min raise
        let stacks = vec![1000.0; 2];
        let mut game_state = GameState::new_starting(stacks, 2.0, 1.0, 0.0, 0);
        game_state.advance_round();
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_after_multiple_raises() {
        // After multiple raises (min raise increases)
        let stacks = vec![500.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();
        game_state.do_bet(20.0, false).unwrap(); // Bet 20
        game_state.do_bet(50.0, false).unwrap(); // Raise to 50 (raise of 30)
        game_state.do_bet(110.0, false).unwrap(); // Re-raise to 110 (raise of 60)
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_three_players() {
        // Three player scenario
        let stacks = vec![100.0; 3];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();
        game_state.do_bet(15.0, false).unwrap();
        game_state.do_bet(15.0, false).unwrap();
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_river() {
        // River scenario
        let stacks = vec![100.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round(); // Flop
        game_state.advance_round(); // Turn
        game_state.advance_round(); // River
        verify_all_actions_valid(&game_state);
    }

    // === ConfigurableActionGenerator tests ===

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
    fn test_configurable_fold_maps_to_zero() {
        let stacks = vec![100.0; 2];
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let action_gen = create_configurable_generator(&game_state, default_configurable_config());

        assert_eq!(action_gen.action_to_idx(&game_state, &AgentAction::Fold), 0);
    }

    #[test]
    fn test_configurable_gen_actions_basic() {
        let stacks = vec![500.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
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
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();

        let action_gen = create_configurable_generator(&game_state, default_configurable_config());
        let actions = action_gen.gen_possible_actions(&game_state);

        // No fold option when there's nothing to call
        assert!(!actions.contains(&AgentAction::Fold));
    }

    #[test]
    fn test_configurable_num_potential_actions() {
        let stacks = vec![100.0; 2];
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let config = default_configurable_config();
        let action_gen = create_configurable_generator(&game_state, config);

        // fold + call + 2 raise_mult + 2 pot_mult + all_in = 7
        assert_eq!(action_gen.num_potential_actions(&game_state), 7);
    }

    #[test]
    fn test_configurable_with_setup_shove() {
        let stacks = vec![500.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
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
        // fold + call + 1 raise_mult + 1 pot_mult + setup_shove + all_in = 6
        assert_eq!(action_gen.num_potential_actions(&game_state), 6);
    }

    #[test]
    fn test_configurable_per_round_config() {
        let stacks = vec![500.0; 2];
        let game_state_preflop = GameState::new_starting(stacks.clone(), 10.0, 5.0, 0.0, 0);
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

        // num_potential should be based on max across all rounds
        // fold + call + max(3, 1) raise_mult + max(0, 2) pot_mult + all_in = 1 + 1 + 3 + 2 + 1 = 8
        assert_eq!(action_gen.num_potential_actions(&game_state_preflop), 8);

        // Verify that preflop generates raise actions (not pot-based)
        // Since we're in preflop with the override, we should have raise_mult actions
        assert!(!preflop_actions.is_empty());
    }

    #[test]
    fn test_configurable_action_indices_stable() {
        let stacks = vec![500.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();
        game_state.do_bet(30.0, false).unwrap();

        let action_gen = create_configurable_generator(&game_state, default_configurable_config());
        let actions = action_gen.gen_possible_actions(&game_state);

        // Verify each action maps to a unique index
        let mut indices: Vec<usize> = actions
            .iter()
            .map(|a| action_gen.action_to_idx(&game_state, a))
            .collect();
        indices.sort();
        indices.dedup();
        assert_eq!(
            indices.len(),
            actions.len(),
            "All actions should have unique indices"
        );
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
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        verify_configurable_actions_valid(&game_state, default_configurable_config());
    }

    #[test]
    fn test_configurable_all_actions_valid_flop() {
        let stacks = vec![100.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();
        verify_configurable_actions_valid(&game_state, default_configurable_config());
    }

    #[test]
    fn test_configurable_all_actions_valid_facing_bet() {
        let stacks = vec![500.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        game_state.advance_round();
        game_state.do_bet(30.0, false).unwrap();
        verify_configurable_actions_valid(&game_state, default_configurable_config());
    }

    #[test]
    fn test_configurable_call_disabled() {
        let stacks = vec![500.0; 2];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
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
}
