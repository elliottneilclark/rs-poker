use std::{cell::RefMut, path::Path};

use little_sorry::RegretMatcher;
use ndarray::ArrayView1;
use tracing::event;

use crate::arena::{action::AgentAction, cfr::export_to_png, Agent, GameState, Historian, HoldemSimulationBuilder};

use super::{
    CFRHistorian, NodeData,
    action_generator::ActionGenerator,
    state::{CFRState, TraversalState},
};

pub struct CFRAgent<T>
where
    T: ActionGenerator + 'static,
{
    pub traversal_state: TraversalState,
    pub cfr_state: CFRState,
    pub action_generator: T,
    forced_action: Option<AgentAction>,
    num_iterations: usize,
}

impl<T> CFRAgent<T>
where
    T: ActionGenerator + 'static,
{
    pub fn new(cfr_state: CFRState, player_idx: usize) -> Self {
        let traversal_state = TraversalState::new_root(player_idx);
        let action_generator = T::new(cfr_state.clone(), traversal_state.clone());
        CFRAgent {
            cfr_state,
            traversal_state,
            action_generator,
            forced_action: None,
            num_iterations: 4,
        }
    }

    fn new_with_forced_action(
        cfr_state: CFRState,
        traversal_state: TraversalState,
        forced_action: AgentAction,
    ) -> Self {
        let action_generator = T::new(cfr_state.clone(), traversal_state.clone());
        CFRAgent {
            cfr_state,
            traversal_state,
            action_generator,
            forced_action: Some(forced_action),
            num_iterations: 4,
        }
    }

    pub fn historian(&self) -> CFRHistorian<T> {
        CFRHistorian::new(self.traversal_state.clone(), self.cfr_state.clone())
    }

    fn reward(&self, game_state: &GameState, action: AgentAction) -> f32 {
        let num_agents = game_state.num_players;

        let states: Vec<_> = (0..num_agents)
            .map(|i| {
                if i == self.traversal_state.player_idx() {
                    CFRState::deep_copy(&self.cfr_state)
                } else {
                    CFRState::new(game_state.clone())
                }
            })
            .collect();

        let agents: Vec<_> = states
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                if i == self.traversal_state.player_idx() {
                    Box::new(CFRAgent::<T>::new_with_forced_action(
                        s,
                        TraversalState::new(
                            self.traversal_state.node_idx(),
                            self.traversal_state.chosen_child_idx(),
                            i,
                        ),
                        action.clone(),
                    ))
                } else {
                    Box::new(CFRAgent::<T>::new(s, i))
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

        sim.run();

        sim.game_state.stacks[self.traversal_state.player_idx()]
            - sim.game_state.starting_stacks[self.traversal_state.player_idx()]
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
					export_to_png(&self.cfr_state, Path::new("output.png"), false).unwrap();
                    panic!("Expected player data, found {:?}", target_node);
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
        let mut rewards: Vec<f32> = vec![
            -game_state.current_player_starting_stack();
            self.action_generator.num_potential_actions(game_state)
        ];

        for _i in 0..self.num_iterations {
            // For every action try it and see what the result is
            for action in actions.clone() {
                let reward_idx = self.action_generator.action_to_idx(game_state, &action);

                // We pre-allocated the rewards vector for each possble action as the
                // action_generator told us So make sure that holds true here.
                assert!(
                    reward_idx < rewards.len(),
                    "Action index {} should be less than number of possible action {}",
                    reward_idx,
                    rewards.len()
                );

                rewards[reward_idx] = self.reward(game_state, action);
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

impl<T> Agent for CFRAgent<T>
where
    T: ActionGenerator + 'static,
{
    fn act(
        &mut self,
        id: &uuid::Uuid,
        game_state: &GameState,
    ) -> crate::arena::action::AgentAction {
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

        if let Some(force_action) = &self.forced_action {
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
    use crate::arena::cfr::BasicCFRActionGenerator;

    use crate::arena::game_state;

    use super::*;

    #[test]
    fn test_create_agent() {
        let game_state = game_state::GameState::new_starting(vec![100.0; 3], 10.0, 5.0, 0.0, 0);
        let cfr_state = CFRState::new(game_state);
        let _ = CFRAgent::<BasicCFRActionGenerator>::new(cfr_state.clone(), 0);
    }

    #[test]
    fn test_run_heads_up() {
        let num_agents = 2;
        // Zero is all in.
        let stacks: Vec<f32> = vec![50.0, 50.0];
        let game_state = game_state::GameState::new_starting(stacks, 5.0, 2.5, 0.0, 0);

        let states: Vec<_> = (0..num_agents)
            .map(|_| CFRState::new(game_state.clone()))
            .collect();

        let agents: Vec<_> = states
            .iter()
            .enumerate()
            .map(|(i, s)| Box::new(CFRAgent::<BasicCFRActionGenerator>::new(s.clone(), i)))
            .collect();

        let historians: Vec<Box<dyn Historian>> = agents
            .iter()
            .map(|a| Box::new(a.historian()) as Box<dyn Historian>)
            .collect();

        let dyn_agents = agents.into_iter().map(|a| a as Box<dyn Agent>).collect();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(dyn_agents)
            .historians(historians)
            .build()
            .unwrap();

        sim.run();

		println!("GAME STATE {:?}", sim.game_state);
    }
}
