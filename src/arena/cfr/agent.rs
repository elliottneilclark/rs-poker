use little_sorry::RegretMatcher;
use ndarray::ArrayView1;
use tracing::event;

use crate::arena::{Agent, GameState, Historian, HoldemSimulationBuilder, action::AgentAction};

use super::{
    CFRHistorian, GameStateIteratorGen, NodeData,
    action_generator::ActionGenerator,
    state::{CFRState, TraversalState},
    state_store::StateStore,
};

/// A CFR (Counterfactual Regret Minimization) agent for poker.
///
/// This agent uses CFR to compute optimal strategies by exploring the game tree
/// and learning from regret. It maintains state across simulations via a shared
/// StateStore.
pub struct CFRAgent<T, I>
where
    T: ActionGenerator + 'static,
    I: GameStateIteratorGen + Clone + 'static,
{
    name: String,
    state_store: StateStore,
    traversal_state: TraversalState,
    cfr_state: CFRState,
    action_generator: T,
    gamestate_iterator_gen: I,
    force_recompute: bool,

    // This will be the next action to play
    // This allows us to start exploration
    // from a specific action.
    forced_action: Option<AgentAction>,

    // Sub-agents (created in reward()) don't have historians because they
    // share the parent's tree but have different card perspectives. Recording
    // from different perspectives would corrupt the tree.
    is_sub_agent: bool,
}

impl<T, I> CFRAgent<T, I>
where
    T: ActionGenerator + 'static,
    I: GameStateIteratorGen + Clone + 'static,
{
    /// Create a new CFR agent that initializes states for ALL players.
    ///
    /// This is the primary constructor for CFR agents. It creates its own
    /// StateStore and ensures all players have properly initialized state,
    /// enabling correct sub-simulations even against non-CFR agents.
    ///
    /// # Arguments
    /// * `name` - Name for this agent
    /// * `player_idx` - The player index this agent represents
    /// * `game_state` - The starting game state (used to initialize all player states)
    /// * `gamestate_iterator_gen` - Generator for game state iteration during exploration
    pub fn new(
        name: impl Into<String>,
        player_idx: usize,
        game_state: GameState,
        gamestate_iterator_gen: I,
    ) -> Self {
        let mut state_store = StateStore::new();

        // Initialize CFR state for ALL players, not just this one.
        // This ensures sub-simulations have correct state for all players.
        state_store.ensure_all_players(game_state.clone(), game_state.num_players);

        let (cfr_state, traversal_state) = state_store.push_traversal(player_idx);

        event!(
            tracing::Level::DEBUG,
            player_idx,
            num_players = game_state.num_players,
            "Created CFR agent"
        );

        let action_generator = T::new(cfr_state.clone(), traversal_state.clone());
        CFRAgent {
            name: name.into(),
            state_store,
            cfr_state,
            traversal_state,
            action_generator,
            gamestate_iterator_gen,
            force_recompute: false,
            forced_action: None,
            is_sub_agent: false,
        }
    }

    /// Create a new CFR agent for sub-simulations with a shared state store.
    ///
    /// Used by `reward()` to create agents that share the same state store
    /// as the parent agent. Each sub-agent gets its own traversal state
    /// pushed onto the stack. This allows learning across simulations.
    ///
    /// Sub-agents don't return historians, preventing tree corruption from
    /// different card perspectives during reward computation.
    ///
    /// # Arguments
    /// * `name` - Name for this agent
    /// * `state_store` - Shared state store from the parent agent
    /// * `player_idx` - The player index this agent represents
    /// * `gamestate_iterator_gen` - Generator for game state iteration during exploration
    /// * `forced_action` - Optional action to force on first act (for exploration)
    pub fn new_sub_agent(
        name: impl Into<String>,
        state_store: StateStore,
        player_idx: usize,
        gamestate_iterator_gen: I,
        forced_action: Option<AgentAction>,
    ) -> Self {
        let (cfr_state, traversal_state) = state_store.clone().push_traversal(player_idx);

        event!(
            tracing::Level::TRACE,
            player_idx,
            ?forced_action,
            "Created sub-agent with shared state store"
        );

        let action_generator = T::new(cfr_state.clone(), traversal_state.clone());
        CFRAgent {
            name: name.into(),
            state_store,
            cfr_state,
            traversal_state,
            action_generator,
            gamestate_iterator_gen,
            force_recompute: false,
            forced_action,
            is_sub_agent: true,
        }
    }

    /// Returns a reference to this agent's CFR state.
    ///
    /// The CFR state contains the game tree with regret information learned
    /// during simulations. This can be used for visualization or analysis.
    pub fn cfr_state(&self) -> &CFRState {
        &self.cfr_state
    }

    fn build_historian(&self) -> CFRHistorian<T> {
        // Use single-player historian - each player tracks only their own
        // traversal state. This is correct because the CFR tree branches
        // based on private cards, so each player's view is different.
        CFRHistorian::new(self.traversal_state.clone(), self.cfr_state.clone())
    }

    /// Compute the expected reward for taking a specific action.
    ///
    /// This function simulates a game from the current state, where all players
    /// play optimally using CFR. The reward is the expected payout for the
    /// calling agent if they take the specified action.
    fn reward(&mut self, game_state: &GameState, action: AgentAction) -> f32 {
        let num_agents = game_state.num_players;
        let mut rand = rand::rng();

        // Debug assertions to show that checking for rewards doesn't move us through
        // the tree
        //
        // These are only used in debug build so this shouldn't be a performance concern
        let before_node_idx = self.traversal_state.node_idx();
        let before_child_idx = self.traversal_state.chosen_child_idx();

        event!(
            tracing::Level::TRACE,
            num_agents,
            ?action,
            player_idx = self.traversal_state.player_idx(),
            "Computing reward via sub-simulation"
        );

        // Create sub-agents for all players using the shared state store.
        // Each sub-agent will push their own traversal state onto the stack.
        let agents: Vec<_> = (0..num_agents)
            .map(|i| {
                let agent_name = format!("CFRAgent-sub-{i}");
                let forced_action = if i == self.traversal_state.player_idx() {
                    Some(action.clone())
                } else {
                    None
                };

                Box::new(CFRAgent::<T, I>::new_sub_agent(
                    agent_name,
                    self.state_store.clone(),
                    i,
                    self.gamestate_iterator_gen.clone(),
                    forced_action,
                ))
            })
            .collect();

        let dyn_agents = agents.into_iter().map(|a| a as Box<dyn Agent>).collect();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state.clone())
            .agents(dyn_agents)
            .build()
            .unwrap();

        sim.run(&mut rand);

        // After each agent explores we need to return the traversal state
        for player_idx in 0..num_agents {
            self.state_store.pop_traversal(player_idx);
        }

        debug_assert_eq!(
            before_node_idx,
            self.traversal_state.node_idx(),
            "Node index should be the same after exploration"
        );
        debug_assert_eq!(
            before_child_idx,
            self.traversal_state.chosen_child_idx(),
            "Child index should be the same after exploration"
        );

        sim.game_state
            .player_reward(self.traversal_state.player_idx())
    }

    fn target_node_idx(&self) -> Option<usize> {
        let from_node_idx = self.traversal_state.node_idx();
        let from_child_idx = self.traversal_state.chosen_child_idx();
        self.cfr_state.get_child(from_node_idx, from_child_idx)
    }

    /// Ensure that the target node is created and that it is a player node with
    /// a regret matcher. Agent should always know the node is a player node
    /// before the historian this will eagarly create the node.
    fn ensure_target_node(&mut self, game_state: &GameState) -> usize {
        match self.target_node_idx() {
            Some(t) => {
                let target_node_data = self.cfr_state.get_node_data(t).unwrap();
                if let NodeData::Player(ref player_data) = target_node_data {
                    assert_eq!(
                        player_data.player_idx,
                        self.traversal_state.player_idx(),
                        "Player node should have the same player index as the agent"
                    );
                } else {
                    // This should never happen
                    // The agent should only be called when it's the player's turn
                    // and some agent should create this node.
                    panic!(
                        "Expected player data at index {t}, found {target_node_data:?}. Game state {game_state:?}"
                    );
                }
                t
            }
            None => self.cfr_state.add(
                self.traversal_state.node_idx(),
                self.traversal_state.chosen_child_idx(),
                super::NodeData::Player(super::PlayerData {
                    regret_matcher: None,
                    player_idx: self.traversal_state.player_idx(),
                }),
            ),
        }
    }

    fn ensure_regret_matcher(&mut self, game_state: &GameState) {
        let target_node_idx = self.ensure_target_node(game_state);
        let num_experts = self.action_generator.num_potential_actions(game_state);

        self.cfr_state
            .update_node(target_node_idx, |node| {
                if let NodeData::Player(ref mut player_data) = node.data
                    && player_data.regret_matcher.is_none()
                {
                    let regret_matcher = Box::new(RegretMatcher::new(num_experts).unwrap());
                    player_data.regret_matcher = Some(regret_matcher);
                }
            })
            .unwrap();
    }

    fn needs_to_explore(&mut self) -> bool {
        self.force_recompute || !self.has_regret_matcher()
    }

    fn has_regret_matcher(&mut self) -> bool {
        self.target_node_idx()
            .and_then(|t| self.cfr_state.get_node_data(t))
            .map(|node_data| {
                if let NodeData::Player(ref player_data) = node_data {
                    player_data.regret_matcher.is_some()
                } else {
                    false
                }
            })
            .unwrap_or(false)
    }

    pub fn explore_all_actions(&mut self, game_state: &GameState) {
        let actions = self.action_generator.gen_possible_actions(game_state);
        let num_potential_actions = self.action_generator.num_potential_actions(game_state);

        // Build a set of valid action indices - these are the indices that correspond
        // to actions we can actually take in this game state.
        let valid_indices: std::collections::HashSet<usize> = actions
            .iter()
            .map(|a| self.action_generator.action_to_idx(game_state, a))
            .collect();

        debug_assert!(
            !valid_indices.is_empty(),
            "Must have at least one valid action"
        );
        debug_assert!(
            valid_indices.len() == actions.len(),
            "All actions should map to unique indices, got {} actions but {} unique indices",
            actions.len(),
            valid_indices.len()
        );

        // Initialize rewards for invalid actions to a very negative value.
        // This ensures the regret matcher learns to avoid actions that aren't
        // valid in this game state. Using the player's starting stack as a
        // penalty since losing your whole stack is the worst outcome.
        let invalid_action_penalty =
            -(game_state.starting_stacks[self.traversal_state.player_idx()]);
        let mut rewards: Vec<f32> = (0..num_potential_actions)
            .map(|idx| {
                if valid_indices.contains(&idx) {
                    0.0 // Will be populated with actual reward
                } else {
                    invalid_action_penalty
                }
            })
            .collect();

        let game_states: Vec<_> = self.gamestate_iterator_gen.generate(game_state).collect();
        let num_game_states = game_states.len();

        debug_assert!(num_game_states > 0, "Must have at least one game state");

        for starting_gamestate in game_states {
            // For every action try it and see what the result is
            for action in actions.clone() {
                // Use game_state (not starting_gamestate) to map action to index.
                // The actions were generated from game_state, so the index mapping
                // must use the same state to be consistent.
                let reward_idx = self.action_generator.action_to_idx(game_state, &action);

                debug_assert!(
                    reward_idx < rewards.len(),
                    "Action index {} should be less than number of potential actions {}",
                    reward_idx,
                    rewards.len()
                );
                debug_assert!(
                    valid_indices.contains(&reward_idx),
                    "Action {:?} mapped to index {} which should be in valid_indices",
                    action,
                    reward_idx
                );

                rewards[reward_idx] += self.reward(&starting_gamestate, action);
            }
        }

        // Normalize rewards by the number of game states explored
        // (only for valid action indices - invalid ones keep their penalty)
        if num_game_states > 0 {
            for idx in &valid_indices {
                rewards[*idx] /= num_game_states as f32;
            }
        }

        // Update the regret matcher with the rewards
        let target_node_idx = self.target_node_idx().unwrap();
        self.cfr_state
            .update_node(target_node_idx, |node| {
                if let NodeData::Player(player_data) = &mut node.data {
                    let regret_matcher = player_data.regret_matcher.as_mut().unwrap();
                    regret_matcher
                        .update_regret(ArrayView1::from(&rewards))
                        .unwrap();
                } else {
                    // This should never happen since ensure_target_node
                    // has been called before this.
                    panic!("Expected player data");
                }
            })
            .unwrap();
    }
}

impl<T, I> Agent for CFRAgent<T, I>
where
    T: ActionGenerator + 'static,
    I: GameStateIteratorGen + Clone + 'static,
{
    fn act(&mut self, id: u128, game_state: &GameState) -> crate::arena::action::AgentAction {
        event!(tracing::Level::TRACE, ?id, "Agent acting");
        assert!(
            game_state.round_data.to_act_idx == self.traversal_state.player_idx(),
            "Agent should only be called when it's the player's turn"
        );

        // make sure that we have at least 2 cards
        assert!(
            game_state.hands[self.traversal_state.player_idx()].count() == 2
                || game_state.hands[self.traversal_state.player_idx()].count() >= 5,
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
                    let forced_idx = self
                        .action_generator
                        .action_to_idx(game_state, &force_action);

                    // Find a valid action with the same index
                    if let Some(valid_action) = valid_actions
                        .iter()
                        .find(|a| self.action_generator.action_to_idx(game_state, a) == forced_idx)
                    {
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
            // If there's no regret matcher, we need to explore the actions
            if self.needs_to_explore() {
                self.ensure_regret_matcher(game_state);
                // Explore all the potential actions
                self.explore_all_actions(game_state);
            }
            // Now the regret matcher should have all the needed data
            // to choose an action.
            self.action_generator.gen_action(game_state)
        }
    }

    /// CFRAgent has a historian unless it's a sub-agent.
    ///
    /// Sub-agents (created during reward computation) don't have historians
    /// because they share the parent's CFR tree but have different private
    /// card perspectives. Recording from different perspectives would corrupt
    /// the tree structure.
    fn historian(&self) -> Option<Box<dyn Historian>> {
        if self.is_sub_agent {
            None
        } else {
            Some(Box::new(self.build_historian()) as Box<dyn Historian>)
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {

    use crate::arena::GameState;
    use crate::arena::agent::CallingAgent;
    use crate::arena::cfr::{BasicCFRActionGenerator, FixedGameStateIteratorGen};

    use super::*;

    /// Test that a CFR agent can play against a non-CFR agent.
    /// This is a regression test for a bug where the CFR agent's reward()
    /// function assumed all players had CFR state initialized.
    ///
    /// The scenario: Player 0 is a CallingAgent (non-CFR), Player 1 is a CFR agent.
    /// The CFR agent creates its own StateStore with states for ALL players.
    #[test]
    fn test_cfr_vs_non_cfr_agent() {
        let stacks: Vec<f32> = vec![50.0, 50.0];
        let game_state = GameState::new_starting(stacks, 5.0, 2.5, 0.0, 0);

        // CFR agent creates its own StateStore with states for all players
        // This enables proper sub-simulation with CFR agents for all seats
        let cfr_agent = Box::new(
            CFRAgent::<BasicCFRActionGenerator, FixedGameStateIteratorGen>::new(
                "CFRAgent-player1",
                1,
                game_state.clone(),
                FixedGameStateIteratorGen::new(1),
            ),
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
        let game_state = GameState::new_starting(vec![100.0; 3], 10.0, 5.0, 0.0, 0);
        let _ = CFRAgent::<BasicCFRActionGenerator, FixedGameStateIteratorGen>::new(
            "CFRAgent-test",
            0,
            game_state,
            FixedGameStateIteratorGen::new(1),
        );
    }

    #[test]
    fn test_run_heads_up() {
        let num_agents = 2;
        let stacks: Vec<f32> = vec![50.0, 50.0];
        let game_state = GameState::new_starting(stacks, 5.0, 2.5, 0.0, 0);

        // Each CFR agent creates its own StateStore with states for all players.
        // This is the correct pattern - agents don't share StateStores.
        let agents: Vec<Box<dyn Agent>> = (0..num_agents)
            .map(|i| {
                Box::new(
                    CFRAgent::<BasicCFRActionGenerator, FixedGameStateIteratorGen>::new(
                        format!("CFRAgent-test-{i}"),
                        i,
                        game_state.clone(),
                        FixedGameStateIteratorGen::new(2),
                    ),
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
}
