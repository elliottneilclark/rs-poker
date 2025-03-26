use std::cell::RefMut;

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

pub struct CFRAgent<T, I>
where
    T: ActionGenerator + 'static,
    I: GameStateIteratorGen + Clone + 'static,
{
    state_store: StateStore,
    traversal_state: TraversalState,
    cfr_state: CFRState,
    action_generator: T,
    gamestate_iterator_gen: I,

    // This will be the next action to play
    // This allows us to start exploration
    // from a specific action.
    forced_action: Option<AgentAction>,
}

impl<T, I> CFRAgent<T, I>
where
    T: ActionGenerator + 'static,
    I: GameStateIteratorGen + Clone + 'static,
{
    pub fn new(
        state_store: StateStore,
        cfr_state: CFRState,
        traversal_state: TraversalState,
        gamestate_iterator_gen: I,
    ) -> Self {
        debug_assert!(
            state_store.len() > traversal_state.player_idx(),
            "State store should have a state for the player"
        );
        let action_generator = T::new(cfr_state.clone(), traversal_state.clone());
        CFRAgent {
            state_store,
            cfr_state,
            traversal_state,
            action_generator,
            gamestate_iterator_gen,
            forced_action: None,
        }
    }

    fn new_with_forced_action(
        state_store: StateStore,
        cfr_state: CFRState,
        traversal_state: TraversalState,
        gamestate_iterator_gen: I,
        forced_action: AgentAction,
    ) -> Self {
        let action_generator = T::new(cfr_state.clone(), traversal_state.clone());
        CFRAgent {
            state_store,
            cfr_state,
            traversal_state,
            action_generator,
            gamestate_iterator_gen,
            forced_action: Some(forced_action),
        }
    }

    pub fn historian(&self) -> CFRHistorian<T> {
        CFRHistorian::new(self.traversal_state.clone(), self.cfr_state.clone())
    }

    fn reward(&mut self, game_state: &GameState, action: AgentAction) -> f32 {
        let num_agents = game_state.num_players;
        let mut rand = rand::rng();

        // Debug assertions to show that checking for rewards doesn't move us through
        // the tree
        //
        // These are only used in debug build so this shouldn't be a performance concern
        let before_node_idx = self.traversal_state.node_idx();
        let before_child_idx = self.traversal_state.chosen_child_idx();

        let agents: Vec<_> = (0..num_agents)
            .map(|i| {
                let (cfr_state, traversal_state) = self.state_store.push_traversal(i);

                if i == self.traversal_state.player_idx() {
                    Box::new(CFRAgent::<T, I>::new_with_forced_action(
                        self.state_store.clone(),
                        cfr_state,
                        traversal_state,
                        self.gamestate_iterator_gen.clone(),
                        action.clone(),
                    ))
                } else {
                    Box::new(CFRAgent::<T, I>::new(
                        self.state_store.clone(),
                        cfr_state,
                        traversal_state,
                        self.gamestate_iterator_gen.clone(),
                    ))
                }
            })
            .collect();

        let historians: Vec<Box<dyn Historian>> = agents
            .iter()
            .map(|a| Box::new(a.historian()) as Box<dyn Historian>)
            .collect();

        let dyn_agents = agents.into_iter().map(|a| a as Box<dyn Agent>).collect();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state.clone())
            .agents(dyn_agents)
            .historians(historians)
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
        self.cfr_state
            .get(from_node_idx)
            .unwrap()
            .get_child(from_child_idx)
    }

    fn get_mut_target_node(&mut self) -> RefMut<super::Node> {
        let target_node_idx = self.target_node_idx().unwrap();
        self.cfr_state.get_mut(target_node_idx).unwrap()
    }

    /// Ensure that the target node is created and that it is a player node with
    /// a regret matcher. Agent should always know the node is a player node
    /// before the historian this will eagarly create the node.
    fn ensure_target_node(&mut self, game_state: &GameState) -> usize {
        match self.target_node_idx() {
            Some(t) => {
                let target_node = self.cfr_state.get(t).unwrap();
                if let NodeData::Player(ref player_data) = target_node.data {
                    assert!(
                        player_data.regret_matcher.is_some(),
                        "Player node should have regret matcher"
                    );

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
                        "Expected player data at index {}, found {:?}. Game state {:?}",
                        t, target_node, game_state
                    );
                }
                t
            }
            None => {
                let num_experts = self.action_generator.num_potential_actions(game_state);
                let regret_matcher = Box::new(RegretMatcher::new(num_experts).unwrap());
                self.cfr_state.add(
                    self.traversal_state.node_idx(),
                    self.traversal_state.chosen_child_idx(),
                    super::NodeData::Player(super::PlayerData {
                        regret_matcher: Some(regret_matcher),
                        player_idx: self.traversal_state.player_idx(),
                    }),
                )
            }
        }
    }

    pub fn explore_all_actions(&mut self, game_state: &GameState) {
        let actions = self.action_generator.gen_possible_actions(game_state);

        // We assume that any non-explored action would be bad for the player, so we
        // assign them a reward of losing our entire stack.
        let mut rewards: Vec<f32> =
            vec![0.0; self.action_generator.num_potential_actions(game_state)];
        let mut explored_game_states = 0;

        let game_states: Vec<_> = self.gamestate_iterator_gen.generate(game_state).collect();
        for starting_gamestate in game_states {
            // Keep track of the number of game states we have explored
            explored_game_states += 1;

            // For every action try it and see what the result is
            for action in actions.clone() {
                let reward_idx = self
                    .action_generator
                    .action_to_idx(&starting_gamestate, &action);

                // We pre-allocated the rewards vector for each possble action as the
                // action_generator told us So make sure that holds true here.
                assert!(
                    reward_idx < rewards.len(),
                    "Action index {} should be less than number of possible action {}",
                    reward_idx,
                    rewards.len()
                );

                rewards[reward_idx] += self.reward(&starting_gamestate, action);
            }

            // normalize the rewards by the number of game states we have explored
            if explored_game_states > 0 {
                for reward in &mut rewards {
                    *reward /= explored_game_states as f32;
                }
            }

            // Update the regret matcher with the rewards
            let mut target_node = self.get_mut_target_node();
            if let NodeData::Player(player_data) = &mut target_node.data {
                let regret_matcher = player_data.regret_matcher.as_mut().unwrap();
                regret_matcher
                    .update_regret(ArrayView1::from(&rewards))
                    .unwrap();
            } else {
                // This should never happen since ensure_target_node
                // has been called before this.
                panic!("Expected player data");
            }
        }
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
            force_action.clone()
        } else {
            // Explore all the potential actions
            self.explore_all_actions(game_state);

            // Now the regret matcher should have all the needed data
            // to choose an action.
            self.action_generator.gen_action(game_state)
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::arena::GameState;
    use crate::arena::cfr::{BasicCFRActionGenerator, FixedGameStateIteratorGen};

    use super::*;

    #[test]
    fn test_create_agent() {
        let game_state = GameState::new_starting(vec![100.0; 3], 10.0, 5.0, 0.0, 0);
        let mut state_store = StateStore::new();
        let (cfr_state, traversal_state) = state_store.new_state(game_state.clone(), 0);
        let _ = CFRAgent::<BasicCFRActionGenerator, FixedGameStateIteratorGen>::new(
            state_store.clone(),
            cfr_state,
            traversal_state,
            FixedGameStateIteratorGen::new(1),
        );
    }

    #[test]
    fn test_run_heads_up() {
        let num_agents = 2;
        // Zero is all in.
        let stacks: Vec<f32> = vec![50.0, 50.0];
        let game_state = GameState::new_starting(stacks, 5.0, 2.5, 0.0, 0);
        let mut state_store = StateStore::new();

        let agents: Vec<_> = (0..num_agents)
            .map(|i| {
                assert_eq!(i, state_store.len());
                let (cfr_state, traversal_state) = state_store.new_state(game_state.clone(), i);
                assert_eq!(i + 1, state_store.len());
                Box::new(
                    CFRAgent::<BasicCFRActionGenerator, FixedGameStateIteratorGen>::new(
                        state_store.clone(),
                        cfr_state,
                        traversal_state,
                        FixedGameStateIteratorGen::new(2),
                    ),
                )
            })
            .collect();

        assert_eq!(num_agents, state_store.len());

        for (i, agent) in agents.iter().enumerate() {
            assert_eq!(i, agent.traversal_state.player_idx());

            // There's always a root + the current exploration
            assert_eq!(2, state_store.traversal_len(i));

            assert_eq!(
                TraversalState::new_root(i),
                agents[i].traversal_state.clone()
            );
        }

        let historians: Vec<Box<dyn Historian>> = agents
            .iter()
            .map(|a| Box::new(a.historian()) as Box<dyn Historian>)
            .collect();

        let dyn_agents = agents.into_iter().map(|a| a as Box<dyn Agent>).collect();

        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(dyn_agents)
            .historians(historians)
            .build()
            .unwrap();

        sim.run(&mut rng);
    }
}
