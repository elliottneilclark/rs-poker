use std::marker::PhantomData;

use little_sorry::RegretMatcher;
use ndarray::ArrayView1;
use rand::{Rng, SeedableRng, rngs::StdRng};
use tracing::event;

use crate::arena::{Agent, GameState, Historian, HoldemSimulationBuilder, action::AgentAction};

use super::{
    ActionIndexMapper, ActionPicker, CFRHistorian, GameStateIteratorGen, NUM_ACTION_INDICES,
    NodeData,
    action_bit_set::ActionBitSet,
    action_generator::ActionGenerator,
    action_validator::{ValidatorMode, validate_actions},
    get_regret_matcher_from_node,
    state::{CFRState, TraversalState},
    state_store::StateStore,
};

/// A CFR (Counterfactual Regret Minimization) agent for poker.
///
/// This agent uses CFR to compute optimal strategies by exploring the game tree
/// and learning from regret. It maintains state across simulations via a shared
/// StateStore.
///
/// # Type Parameters
/// * `T` - The action generator type (implements `ActionGenerator`)
/// * `I` - The game state iterator generator type
/// * `R` - The random number generator type (defaults to `StdRng`)
pub struct CFRAgent<T, I, R = StdRng>
where
    T: ActionGenerator,
    I: GameStateIteratorGen,
    R: Rng + SeedableRng,
{
    name: String,
    state_store: StateStore,
    traversal_state: TraversalState,
    cfr_state: CFRState,
    action_generator: T,
    gamestate_iterator_gen: I,
    /// The action index mapper for consistent action-to-index mapping.
    action_index_mapper: ActionIndexMapper,

    // This will be the next action to play
    // This allows us to start exploration
    // from a specific action.
    forced_action: Option<AgentAction>,

    /// Recursion depth in CFR tree (0 = root agent, 1+ = sub-agent).
    /// Sub-agents (depth > 0) don't have historians because the parent historian handles
    /// tree updates for all players.
    depth: usize,

    /// Depth at which to switch to limited action exploration.
    /// When depth >= this value, only Fold, Call, AllIn actions are explored.
    /// None means no limit (always use full action set).
    limited_exploration_depth: Option<usize>,

    /// Whether to allow mutating node types when a mismatch is found.
    /// When true, if the agent expects a Player node but finds a different type
    /// (e.g., Terminal), it will convert the node to the expected type.
    /// This is necessary for action generators that produce bet amounts mapping
    /// to the same index but resulting in different outcomes (all-in vs not).
    /// Default is true for ConfigurableActionGenerator, false for BasicCFRActionGenerator.
    allow_node_mutation: bool,

    /// Random number generator for action selection.
    rng: R,
}

/// Builder for creating CFR agents with a fluent API.
///
/// All CFR agents in a simulation should share the same `StateStore` to enable
/// shared learning. Use `StateStore::new_for_game()` to create the store, then
/// pass it to all agents.
///
/// # Example
///
/// ```rust,ignore
/// use rs_poker::arena::cfr::{
///     CFRAgentBuilder, BasicCFRActionGenerator,
///     DepthBasedIteratorGen, DepthBasedIteratorGenConfig, StateStore,
/// };
///
/// let state_store = StateStore::new_for_game(game_state.clone());
/// let iter_config = DepthBasedIteratorGenConfig::new(vec![10, 5, 1]);
/// let agent = CFRAgentBuilder::<BasicCFRActionGenerator, DepthBasedIteratorGen>::new()
///     .name("MyAgent")
///     .player_idx(0)
///     .state_store(state_store)
///     .gamestate_iterator_gen_config(iter_config)
///     .action_gen_config(())
///     .build();
/// ```
///
/// # Type Parameters
/// * `T` - The action generator type (implements `ActionGenerator`)
/// * `I` - The game state iterator generator type
/// * `R` - The random number generator type (defaults to `StdRng`)
pub struct CFRAgentBuilder<T, I, R = StdRng>
where
    T: ActionGenerator,
    I: GameStateIteratorGen,
    R: Rng + SeedableRng,
{
    name: Option<String>,
    player_idx: Option<usize>,
    gamestate_iterator_gen_config: Option<I::Config>,
    action_gen_config: Option<T::Config>,
    state_store: Option<StateStore>,
    forced_action: Option<AgentAction>,
    /// Recursion depth in CFR tree (0 = root, 1+ = sub-agent).
    depth: usize,
    /// Depth at which to switch to limited action exploration.
    limited_exploration_depth: Option<usize>,
    /// Whether to allow mutating node types when a mismatch is found.
    allow_node_mutation: bool,
    /// Optional RNG instance. If None, creates one from system entropy.
    rng: Option<R>,
    /// Phantom data to satisfy the compiler.
    _marker: PhantomData<R>,
}

impl<T, I, R> Default for CFRAgentBuilder<T, I, R>
where
    T: ActionGenerator,
    I: GameStateIteratorGen,
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        Self {
            name: None,
            player_idx: None,
            gamestate_iterator_gen_config: None,
            action_gen_config: None,
            state_store: None,
            forced_action: None,
            depth: 0,
            limited_exploration_depth: Some(4),
            allow_node_mutation: true,
            rng: None,
            _marker: PhantomData,
        }
    }
}

impl<T, I, R> CFRAgentBuilder<T, I, R>
where
    T: ActionGenerator,
    I: GameStateIteratorGen,
    R: Rng + SeedableRng,
{
    /// Create a new CFRAgentBuilder with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the name for the agent.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the player index for the agent.
    pub fn player_idx(mut self, player_idx: usize) -> Self {
        self.player_idx = Some(player_idx);
        self
    }

    /// Set the game state iterator generator configuration.
    pub fn gamestate_iterator_gen_config(mut self, config: I::Config) -> Self {
        self.gamestate_iterator_gen_config = Some(config);
        self
    }

    /// Set the action generator configuration.
    pub fn action_gen_config(mut self, config: T::Config) -> Self {
        self.action_gen_config = Some(config);
        self
    }

    /// Set the shared state store for this agent.
    ///
    /// All CFR agents in a simulation should share the same StateStore.
    /// Use `StateStore::new_for_game()` to create an initialized store.
    pub fn state_store(mut self, state_store: StateStore) -> Self {
        self.state_store = Some(state_store);
        self
    }

    /// Set the recursion depth for this agent.
    ///
    /// Depth 0 is the root agent, depth 1+ are sub-agents.
    /// Sub-agents (depth > 0) don't create historians.
    pub fn depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    /// Set a forced action for the first act call.
    ///
    /// When set, the agent will play this action on its first act() call
    /// instead of computing one. Used for exploration in sub-simulations.
    pub fn forced_action(mut self, action: AgentAction) -> Self {
        self.forced_action = Some(action);
        self
    }

    /// Set the depth at which to switch to limited action exploration.
    ///
    /// When depth >= this value, only Fold, Call, AllIn actions are explored.
    /// This reduces the branching factor deep in the CFR tree.
    pub fn limited_exploration_depth(mut self, depth: usize) -> Self {
        self.limited_exploration_depth = Some(depth);
        self
    }

    /// Set the random number generator instance.
    ///
    /// When set, the agent will use this RNG for all random decisions.
    /// This allows passing in a pre-seeded or custom RNG.
    /// If not set, the agent creates an RNG from system entropy.
    pub fn rng(mut self, rng: R) -> Self {
        self.rng = Some(rng);
        self
    }

    /// Build the CFRAgent.
    ///
    /// # Panics
    ///
    /// Panics if any required fields are not set:
    /// - name
    /// - player_idx
    /// - state_store
    /// - gamestate_iterator_gen_config
    /// - action_gen_config
    pub fn build(self) -> CFRAgent<T, I, R> {
        let name = self.name.expect("name is required");
        let player_idx = self.player_idx.expect("player_idx is required");
        let state_store = self.state_store.expect("state_store is required");
        let gamestate_iterator_gen_config = self
            .gamestate_iterator_gen_config
            .expect("gamestate_iterator_gen_config is required");
        let action_gen_config = self
            .action_gen_config
            .expect("action_gen_config is required");

        let (cfr_state, traversal_state) = state_store.clone().push_traversal(player_idx);
        let action_generator = T::new(
            cfr_state.clone(),
            traversal_state.clone(),
            action_gen_config,
        );
        let gamestate_iterator_gen = I::new(&gamestate_iterator_gen_config, self.depth);

        // Create the action index mapper from the CFR state's mapper config
        let mapper_config = cfr_state.mapper_config();
        let action_index_mapper = ActionIndexMapper::new(mapper_config);

        // Create the RNG - use provided RNG or generate from system entropy
        let rng = self.rng.unwrap_or_else(|| R::from_rng(&mut rand::rng()));

        CFRAgent {
            name,
            state_store,
            cfr_state,
            traversal_state,
            action_generator,
            gamestate_iterator_gen,
            action_index_mapper,
            forced_action: self.forced_action,
            depth: self.depth,
            limited_exploration_depth: self.limited_exploration_depth,
            allow_node_mutation: self.allow_node_mutation,
            rng,
        }
    }

    /// Set whether to allow mutating node types when a mismatch is found.
    ///
    /// When true (default), if the agent expects a Player node but finds a different
    /// type (e.g., Terminal), it will convert the node to the expected type.
    /// This is necessary for action generators that produce bet amounts mapping
    /// to the same index but resulting in different outcomes (all-in vs not).
    ///
    /// Set to false when testing with BasicCFRActionGenerator to catch bugs.
    pub fn allow_node_mutation(mut self, allow: bool) -> Self {
        self.allow_node_mutation = allow;
        self
    }
}

impl<T, I, R> CFRAgent<T, I, R>
where
    T: ActionGenerator + 'static,
    I: GameStateIteratorGen + 'static,
    R: Rng + SeedableRng + 'static,
{
    /// Returns a reference to this agent's CFR state.
    ///
    /// The CFR state contains the game tree with regret information learned
    /// during simulations. This can be used for visualization or analysis.
    pub fn cfr_state(&self) -> &CFRState {
        &self.cfr_state
    }

    fn build_historian(&self) -> CFRHistorian<T> {
        // The historian uses the full StateStore and updates all players'
        // CFR states and traversal states when recording actions.
        CFRHistorian::new(
            self.state_store.clone(),
            self.action_generator.config().clone(),
            self.allow_node_mutation,
        )
    }

    /// Compute the expected reward for taking a specific action.
    ///
    /// This function simulates a game from the current state, where all players
    /// play optimally using CFR. The reward is the expected payout for the
    /// calling agent if they take the specified action.
    fn reward(&mut self, game_state: &GameState, action: AgentAction) -> f32 {
        let num_agents = game_state.num_players;

        // Get all traversal state fields in a single lock acquisition
        let (before_node_idx, before_child_idx, player_idx) = self.traversal_state.get_all();

        event!(
            tracing::Level::TRACE,
            num_agents,
            ?action,
            player_idx = player_idx,
            "Computing reward via sub-simulation"
        );

        // Create sub-agents for all players using the shared state store.
        // Each sub-agent will push their own traversal state onto the stack.
        // Sub-agents use depth + 1 to track recursion and get appropriate iteration counts.
        // Create new RNGs for sub-agents seeded from the parent RNG.
        let sub_depth = self.depth + 1;

        // Extract configs before the loop to avoid borrow conflicts with mutable RNG access
        let iter_config = self.gamestate_iterator_gen.config().clone();
        let action_config = self.action_generator.config().clone();
        let state_store = self.state_store.clone();
        let limited_exploration_depth = self.limited_exploration_depth;

        let agents: Vec<_> = (0..num_agents)
            .map(|i| {
                // Create a new RNG for each sub-agent seeded from the parent RNG
                let sub_rng = R::from_rng(&mut self.rng);
                let mut builder = CFRAgentBuilder::<T, I, R>::new()
                    .name("CFRAgent-sub")
                    .player_idx(i)
                    .gamestate_iterator_gen_config(iter_config.clone())
                    .action_gen_config(action_config.clone())
                    .state_store(state_store.clone())
                    .depth(sub_depth)
                    .rng(sub_rng);

                // Propagate limited_exploration_depth to sub-agents
                if let Some(limited_depth) = limited_exploration_depth {
                    builder = builder.limited_exploration_depth(limited_depth);
                }

                if i == player_idx as usize {
                    builder = builder.forced_action(action.clone());
                }

                Box::new(builder.build())
            })
            .collect();

        let dyn_agents = agents.into_iter().map(|a| a as Box<dyn Agent>).collect();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state.clone())
            .agents(dyn_agents)
            .build()
            .unwrap();

        sim.run(&mut self.rng);

        // After each agent explores we need to return the traversal state
        for p_idx in 0..num_agents {
            self.state_store.pop_traversal(p_idx);
        }

        // Debug assertions use get_position() to batch the reads
        #[cfg(debug_assertions)]
        {
            let (after_node_idx, after_child_idx) = self.traversal_state.get_position();
            debug_assert_eq!(
                before_node_idx, after_node_idx,
                "Node index should be the same after exploration"
            );
            debug_assert_eq!(
                before_child_idx, after_child_idx,
                "Child index should be the same after exploration"
            );
        }
        // Suppress unused variable warnings in release builds
        #[cfg(not(debug_assertions))]
        {
            let _ = before_node_idx;
            let _ = before_child_idx;
        }

        sim.game_state.player_reward(player_idx as usize)
    }

    fn target_node_idx(&self) -> Option<usize> {
        let (from_node_idx, from_child_idx) = self.traversal_state.get_position();
        self.cfr_state.get_child(from_node_idx, from_child_idx)
    }

    /// Ensure that the target node is created and that it is a player node.
    ///
    /// Uses `CFRState::ensure_child` which handles the case where different bet
    /// amounts map to the same index but lead to different outcomes. If a node
    /// exists with a different type and `allow_node_mutation` is true, it will
    /// be updated to a Player node.
    fn ensure_target_node(&mut self, _game_state: &GameState) -> usize {
        // Get all traversal state fields in a single lock acquisition
        let (node_idx, chosen_child_idx, player_idx) = self.traversal_state.get_all();

        let expected_data = super::NodeData::Player(super::PlayerData {
            regret_matcher: None,
            player_idx,
        });

        self.cfr_state.ensure_child(
            node_idx,
            chosen_child_idx,
            expected_data,
            self.allow_node_mutation,
        )
    }

    fn ensure_regret_matcher(&mut self, game_state: &GameState) {
        let target_node_idx = self.ensure_target_node(game_state);

        self.cfr_state
            .update_node(target_node_idx, |node| {
                if let NodeData::Player(ref mut player_data) = node.data
                    && player_data.regret_matcher.is_none()
                {
                    // Use the fixed constant for number of action indices (52)
                    let regret_matcher = Box::new(RegretMatcher::new(NUM_ACTION_INDICES).unwrap());
                    player_data.regret_matcher = Some(regret_matcher);
                }
            })
            .unwrap();
    }

    pub fn explore_all_actions(&mut self, game_state: &GameState) {
        // Determine validation mode based on depth
        let mode = if self.depth >= self.limited_exploration_depth.unwrap_or(usize::MAX) {
            ValidatorMode::Limited
        } else {
            ValidatorMode::Standard
        };

        // Generate actions and apply validation filters
        let raw_actions = self.action_generator.gen_possible_actions(game_state);
        let validated_actions = validate_actions(raw_actions, game_state, mode);

        // Filter actions to ensure each maps to a unique index.
        // Different bet amounts can map to the same index due to the logarithmic
        // mapping (only 49 slots for raises). We keep the first action for each index.
        // Using ActionBitSet for O(1) operations with no heap allocation.
        let mut seen_indices = ActionBitSet::new();
        let actions: Vec<_> = validated_actions
            .into_iter()
            .filter(|a| {
                let idx = self.action_index_mapper.action_to_idx(a, game_state);
                seen_indices.insert(idx)
            })
            .collect();

        debug_assert!(!actions.is_empty(), "Must have at least one valid action");

        // Penalty for invalid actions - using player's starting stack since
        // losing your whole stack is the worst outcome.
        let invalid_action_penalty =
            -(game_state.starting_stacks[self.traversal_state.player_idx() as usize]);

        // TODO: generate should consume the gamestate to avoid needing to capture self in the
        // iterator. This will allow us to avoid collecting all game states into a vec.
        let game_states: Vec<_> = self.gamestate_iterator_gen.generate(game_state).collect();
        let target_node_idx = self.target_node_idx().unwrap();

        // Process each game state independently and update regret immediately.
        // This helps the regret matcher converge better than averaging rewards
        // across all game states before updating.
        for starting_gamestate in game_states {
            // Start with penalty for all actions, valid ones get overwritten
            let mut rewards: Vec<f32> = vec![invalid_action_penalty; NUM_ACTION_INDICES];

            for action in actions.clone() {
                let reward_idx = self.action_index_mapper.action_to_idx(&action, game_state);

                debug_assert!(
                    reward_idx < rewards.len(),
                    "Action index {} should be less than number of potential actions {}",
                    reward_idx,
                    rewards.len()
                );
                debug_assert!(
                    seen_indices.contains(reward_idx),
                    "Action {:?} mapped to index {} which should be in seen_indices",
                    action,
                    reward_idx
                );

                rewards[reward_idx] = self.reward(&starting_gamestate, action);
            }

            // Update regret immediately for this game state
            self.cfr_state
                .update_node(target_node_idx, |node| {
                    if let NodeData::Player(player_data) = &mut node.data {
                        let regret_matcher = player_data.regret_matcher.as_mut().unwrap();
                        regret_matcher
                            .update_regret(ArrayView1::from(&rewards))
                            .unwrap();
                    } else {
                        panic!("Expected player data");
                    }
                })
                .unwrap();
        }
    }
}

impl<T, I, R> Agent for CFRAgent<T, I, R>
where
    T: ActionGenerator + 'static,
    I: GameStateIteratorGen + 'static,
    R: Rng + SeedableRng + 'static,
{
    fn act(&mut self, id: u128, game_state: &GameState) -> crate::arena::action::AgentAction {
        event!(tracing::Level::TRACE, ?id, "Agent acting");
        assert!(
            game_state.round_data.to_act_idx == self.traversal_state.player_idx() as usize,
            "Agent should only be called when it's the player's turn"
        );

        // make sure that we have at least 2 cards
        let player_idx = self.traversal_state.player_idx() as usize;
        assert!(
            game_state.hands[player_idx].count() == 2 || game_state.hands[player_idx].count() >= 5,
            "Agent should only be called when it has at least 2 cards"
        );

        // Make sure that the CFR state has a regret matcher for this node
        self.ensure_target_node(game_state);

        if let Some(force_action) = self.forced_action.take() {
            event!(
                tracing::Level::DEBUG,
                ?force_action,
                "Playing forced action"
            );

            // Validate that the forced_action is still valid for this game state.
            // If not, we need to find a similar valid action.
            let valid_actions = self.action_generator.gen_possible_actions(game_state);

            // Check if the forced_action is in the valid actions (or close to it for Bet)
            match &force_action {
                AgentAction::Fold => {
                    if valid_actions.contains(&AgentAction::Fold) {
                        force_action
                    } else {
                        // Can't fold when there's nothing to call - this shouldn't happen
                        // but if it does, just call/check instead
                        event!(
                            tracing::Level::WARN,
                            "Forced Fold action invalid, using first valid action"
                        );
                        valid_actions.first().cloned().unwrap_or(AgentAction::Fold)
                    }
                }
                AgentAction::AllIn => {
                    // All-in should always be valid if we have chips
                    force_action
                }
                AgentAction::Call => {
                    // Call should always be valid
                    force_action
                }
                AgentAction::Bet(amount) => {
                    // For Bet, we need to verify the amount is still valid.
                    // The forced_action was generated for a specific game state, and while
                    // we expect the game state to be the same, let's validate to be safe.
                    // Use the ActionIndexMapper for consistent action-to-index mapping.
                    let forced_idx = self
                        .action_index_mapper
                        .action_to_idx(&force_action, game_state);

                    // Find a valid action with the same index
                    if let Some(valid_action) = valid_actions.iter().find(|a| {
                        self.action_index_mapper.action_to_idx(a, game_state) == forced_idx
                    }) {
                        // Found a valid action with the same index - use it
                        // (it might have a slightly different amount due to game state changes)
                        valid_action.clone()
                    } else {
                        // The forced action's index doesn't correspond to a valid action.
                        // This indicates a game state mismatch. Log and use a fallback.
                        event!(
                            tracing::Level::WARN,
                            ?force_action,
                            forced_idx = forced_idx,
                            amount = amount,
                            current_bet = game_state.current_round_bet(),
                            min_raise = game_state.current_round_min_raise(),
                            "Forced Bet action index not valid, using first valid action"
                        );
                        valid_actions.first().cloned().unwrap_or(AgentAction::Fold)
                    }
                }
            }
        } else {
            self.ensure_regret_matcher(game_state);
            // Explore all the potential actions
            self.explore_all_actions(game_state);

            // Use ActionPicker to select an action based on the regret matcher
            // We use with_node_data to avoid cloning the entire NodeData
            let possible_actions = self.action_generator.gen_possible_actions(game_state);
            let target_node_idx = self.target_node_idx().unwrap();

            self.cfr_state.with_node_data(target_node_idx, |node_data| {
                let regret_matcher = node_data.and_then(get_regret_matcher_from_node);

                let picker = ActionPicker::new(
                    &self.action_index_mapper,
                    &possible_actions,
                    regret_matcher,
                    game_state,
                );
                picker.pick_action(&mut self.rng)
            })
        }
    }

    /// CFRAgent has a historian in two cases:
    /// 1. Root agents (not sub-agents) - one historian per CFR agent
    /// 2. Sub-agents with a forced action - one historian per sub-simulation
    ///
    /// The sub-agent with the forced action is the one driving the exploration
    /// and provides the single historian for the sub-simulation. Other sub-agents
    /// don't have historians to avoid duplicate updates.
    fn historian(&self) -> Option<Box<dyn Historian>> {
        // Root agents (depth 0) always have historians
        // Sub-agents (depth > 0) only have historians if they have a forced action
        if self.depth == 0 || self.forced_action.is_some() {
            Some(Box::new(self.build_historian()) as Box<dyn Historian>)
        } else {
            None
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {

    use crate::arena::GameStateBuilder;
    use crate::arena::agent::CallingAgent;
    use crate::arena::cfr::{
        BasicCFRActionGenerator, DepthBasedIteratorGen, DepthBasedIteratorGenConfig,
    };

    use super::*;

    /// Test that a CFR agent can play against a non-CFR agent.
    /// This is a regression test for a bug where the CFR agent's reward()
    /// function assumed all players had CFR state initialized.
    ///
    /// The scenario: Player 0 is a CallingAgent (non-CFR), Player 1 is a CFR agent.
    /// The CFR agent uses a shared StateStore with states for ALL players.
    #[test]
    fn test_cfr_vs_non_cfr_agent() {
        let stacks: Vec<f32> = vec![50.0, 50.0];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(5.0, 2.5)
            .build()
            .unwrap();

        // Create a shared StateStore with states for all players
        let state_store = StateStore::new(game_state.clone());

        let cfr_agent = Box::new(
            CFRAgentBuilder::<BasicCFRActionGenerator, DepthBasedIteratorGen>::new()
                .name("CFRAgent-player1")
                .player_idx(1)
                .state_store(state_store)
                .gamestate_iterator_gen_config(DepthBasedIteratorGenConfig::new(vec![1]))
                .action_gen_config(())
                .build(),
        );

        // Player 0 is a simple calling agent (non-CFR)
        let calling_agent = Box::new(CallingAgent::new("CallingAgent-player0"));

        let agents: Vec<Box<dyn Agent>> = vec![calling_agent, cfr_agent];

        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();

        // This should not panic - the CFR agent properly handles
        // mixed-agent simulations
        sim.run(&mut rng);
    }

    #[test]
    fn test_create_agent() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(3, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let state_store = StateStore::new(game_state);
        let _ = CFRAgentBuilder::<BasicCFRActionGenerator, DepthBasedIteratorGen>::new()
            .name("CFRAgent-test")
            .player_idx(0)
            .state_store(state_store)
            .gamestate_iterator_gen_config(DepthBasedIteratorGenConfig::new(vec![1]))
            .action_gen_config(())
            .build();
    }

    #[test]
    fn test_run_heads_up() {
        let num_agents = 2;
        let stacks: Vec<f32> = vec![50.0, 50.0];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(5.0, 2.5)
            .build()
            .unwrap();

        // All CFR agents share the same StateStore for shared learning.
        let state_store = StateStore::new(game_state.clone());
        let agents: Vec<Box<dyn Agent>> = (0..num_agents)
            .map(|i| {
                Box::new(
                    CFRAgentBuilder::<BasicCFRActionGenerator, DepthBasedIteratorGen>::new()
                        .name(format!("CFRAgent-test-{i}"))
                        .player_idx(i)
                        .state_store(state_store.clone())
                        .gamestate_iterator_gen_config(DepthBasedIteratorGenConfig::new(vec![2, 1]))
                        .action_gen_config(())
                        .build(),
                ) as Box<dyn Agent>
            })
            .collect();

        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();

        sim.run(&mut rng);
    }

    #[test]
    fn test_sub_agent_without_forced_action_returns_none_historian() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let state_store = StateStore::new(game_state);

        // Create a sub-agent WITHOUT forced_action (depth > 0 means sub-agent)
        let iter_config = DepthBasedIteratorGenConfig::new(vec![1]);
        let sub_agent = CFRAgentBuilder::<BasicCFRActionGenerator, DepthBasedIteratorGen>::new()
            .name("SubAgent")
            .player_idx(0)
            .state_store(state_store)
            .depth(1) // depth > 0 makes it a sub-agent
            .gamestate_iterator_gen_config(iter_config)
            .action_gen_config(())
            .build();

        // Sub-agents (depth > 0) without forced_action should return None for historian
        assert!(sub_agent.historian().is_none());
    }

    #[test]
    fn test_sub_agent_with_forced_action_returns_some_historian() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let state_store = StateStore::new(game_state);

        // Create a sub-agent WITH forced_action
        let iter_config = DepthBasedIteratorGenConfig::new(vec![1]);
        let sub_agent = CFRAgentBuilder::<BasicCFRActionGenerator, DepthBasedIteratorGen>::new()
            .name("SubAgent")
            .player_idx(0)
            .state_store(state_store)
            .depth(1) // depth > 0 makes it a sub-agent
            .forced_action(AgentAction::Call)
            .gamestate_iterator_gen_config(iter_config)
            .action_gen_config(())
            .build();

        // Sub-agents with forced_action should return Some for historian
        assert!(sub_agent.historian().is_some());
    }

    /// Test that agents sharing a StateStore actually share the same CFR tree.
    /// This verifies the new shared state pattern works correctly.
    #[test]
    fn test_shared_state_store_between_agents() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();

        // Create a shared state store
        let state_store = StateStore::new(game_state.clone());

        let iter_config = DepthBasedIteratorGenConfig::new(vec![1]);

        // Create two agents with the same shared store
        let agent0 = CFRAgentBuilder::<BasicCFRActionGenerator, DepthBasedIteratorGen>::new()
            .name("Agent0")
            .player_idx(0)
            .state_store(state_store.clone())
            .gamestate_iterator_gen_config(iter_config.clone())
            .action_gen_config(())
            .build();

        let agent1 = CFRAgentBuilder::<BasicCFRActionGenerator, DepthBasedIteratorGen>::new()
            .name("Agent1")
            .player_idx(1)
            .state_store(state_store.clone())
            .gamestate_iterator_gen_config(iter_config)
            .action_gen_config(())
            .build();

        // Both agents should have access to CFR state
        let state0 = agent0.cfr_state();
        let state1 = agent1.cfr_state();

        // Since they share the store, modifying one should be visible from the other
        // (not directly, but they access the same underlying data via Arc)
        // The cfr_state() returns a clone, but internally they share via Arc<RwLock<>>
        // Verify both see the same starting state (root node exists)
        assert!(state0.get_node_data(0).is_some());
        assert!(state1.get_node_data(0).is_some());

        // Both agents should be at valid positions
        assert!(state0.get_child(0, 0).is_none()); // No children yet at root
        assert!(state1.get_child(0, 0).is_none());
    }

    #[test]
    fn test_root_agent_returns_some_historian() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let state_store = StateStore::new(game_state);

        // Create a root agent (depth = 0, the default)
        let root_agent = CFRAgentBuilder::<BasicCFRActionGenerator, DepthBasedIteratorGen>::new()
            .name("RootAgent")
            .player_idx(0)
            .state_store(state_store)
            .gamestate_iterator_gen_config(DepthBasedIteratorGenConfig::new(vec![1]))
            .action_gen_config(())
            .build();

        // Root agents should return Some for historian
        assert!(root_agent.historian().is_some());
    }

    /// Test that sub-simulations correctly use pushed traversal states
    /// and don't interfere with the parent's traversal state.
    #[test]
    fn test_sub_simulation_traversal_isolation() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let mut state_store = StateStore::new(game_state.clone());

        // Get the initial traversal state position for player 0
        let initial_traversal = state_store.peek_traversal(0).unwrap();
        let initial_node = initial_traversal.node_idx();
        let initial_child = initial_traversal.chosen_child_idx();

        // Push a new traversal state (simulating sub-simulation start)
        let (_, sub_traversal) = state_store.push_traversal(0);

        // The sub-traversal should start at the same position
        assert_eq!(sub_traversal.node_idx(), initial_node);
        assert_eq!(sub_traversal.chosen_child_idx(), initial_child);

        // Move the sub-traversal
        let mut sub_traversal_clone = sub_traversal.clone();
        sub_traversal_clone.move_to(5, 3);

        // The sub-traversal should have moved
        assert_eq!(sub_traversal.node_idx(), 5);
        assert_eq!(sub_traversal.chosen_child_idx(), 3);

        // Pop the sub-traversal
        state_store.pop_traversal(0);

        // The original traversal should be unchanged
        let restored_traversal = state_store.peek_traversal(0).unwrap();
        assert_eq!(restored_traversal.node_idx(), initial_node);
        assert_eq!(restored_traversal.chosen_child_idx(), initial_child);
    }
}
