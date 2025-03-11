use std::cell::RefMut;

use little_sorry::RegretMatcher;
use ndarray::ArrayView1;
use tracing::event;

use crate::arena::{Agent, GameState, Historian, HoldemSimulationBuilder, action::AgentAction};

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
            num_iterations: 100,
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
            num_iterations: 10,
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
                    self.cfr_state.clone()
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
                        self.cfr_state.clone(),
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
                // Create a block scope to limit the lifetime of the immutable borrow
                let is_player_with_regret_matcher;
                let is_chance_node;
                let node_data_for_error;

                {
                    // Scope for the immutable borrow
                    let target_node = self.cfr_state.get(t).unwrap();

                    match &target_node.data {
                        NodeData::Player(player_data) => {
                            is_player_with_regret_matcher = player_data.regret_matcher.is_some();
                            is_chance_node = false;
                            node_data_for_error = None;
                        }
                        NodeData::Chance => {
                            is_player_with_regret_matcher = false;
                            is_chance_node = true;
                            node_data_for_error = None;
                        }
                        NodeData::Terminal(terminal_data) => {
                            // Found a terminal node - this could happen at the end of a hand
                            println!(
                                "Found Terminal node with utility: {}",
                                terminal_data.total_utility
                            );
                            is_player_with_regret_matcher = false;
                            is_chance_node = false;
                            // We'll treat this specially below
                            node_data_for_error = Some("Terminal".to_string());
                        }
                        _ => {
                            is_player_with_regret_matcher = false;
                            is_chance_node = false;
                            node_data_for_error = Some(format!("{:?}", target_node.data));
                        }
                    }
                } // Immutable borrow ends here

                // Now we can check the conditions and potentially mutate self.cfr_state
                if is_player_with_regret_matcher {
                    // All good, the node is already a Player with a regret
                    // matcher
                } else if is_chance_node {
                    // Transform the Chance node to a Player node
                    println!("Converting Chance node to Player node at index {}", t);

                    // Create a new Player node to replace the Chance node
                    let num_experts = self.action_generator.num_potential_actions(game_state);
                    let regret_matcher = Box::new(RegretMatcher::new(num_experts).unwrap());

                    // Create new player node data
                    let player_node_data = super::NodeData::Player(super::PlayerData {
                        regret_matcher: Some(regret_matcher),
                    });

                    // Now we can safely mutate the CFR state
                    self.cfr_state.replace_node_data(t, player_node_data);
                } else if let Some(data_str) = node_data_for_error {
                    if data_str == "Terminal" {
                        // When finding a terminal node during agent processing:
                        // 1. We've reached the end of a betting round or hand
                        // 2. Just return the current node index without modifying it
                        // 3. The caller will need to handle this appropriately
                        println!("Returning terminal node index without modification: {}", t);
                    } else {
                        // This should never happen
                        panic!(
                            "Expected player data, chance data, or terminal data, found {}",
                            data_str
                        );
                    }
                } else {
                    // This case shouldn't be reachable based on our match, but added for safety
                    panic!(
                        "Expected player data, chance data, or terminal data, found unexpected node type"
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

            // Store the node index before the mutable borrow to avoid borrow checker issues
            let current_node_idx = self.traversal_state.node_idx();

            // Update the regret matcher with the rewards only for Player nodes
            let mut target_node = self.get_mut_target_node();
            match &mut target_node.data {
                NodeData::Player(player_data) => {
                    if let Some(regret_matcher) = player_data.regret_matcher.as_mut() {
                        regret_matcher
                            .update_regret(ArrayView1::from(&rewards))
                            .unwrap();
                    } else {
                        // This can happen during initial setup or when we've converted a node
                        println!(
                            "Warning: Player node does not have a regret matcher at node {}",
                            current_node_idx
                        );
                    }
                }
                NodeData::Terminal(terminal_data) => {
                    // Terminal nodes don't need regret updates
                    println!(
                        "Skipping regret update for Terminal node with utility {} at node {}",
                        terminal_data.total_utility, current_node_idx
                    );
                }
                NodeData::Chance => {
                    // Skip Chance nodes - this might happen during specific states of the game
                    println!(
                        "Skipping regret update for Chance node at node {}",
                        current_node_idx
                    );
                }
                _ => {
                    // Handle other node types (like Root) appropriately
                    println!(
                        "Skipping regret update for node of type {:?} at node {}",
                        target_node.data, current_node_idx
                    );
                }
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

    // This test runs a full Texas Holdem game with CFR agents
    #[test]
    fn test_run_heads_up() {
        use std::path::Path;
        use std::fs::create_dir_all;
        use crate::arena::cfr::export::{export_to_svg, export_to_png, export_cfr_state, ExportFormat};
        // Import only what we need

        let num_agents = 2;
        // Each player starts with 50 chips
        let stacks: Vec<f32> = vec![50.0, 50.0];
        let game_state = game_state::GameState::new_starting(stacks, 5.0, 2.5, 0.0, 0);

        println!("============ Test Step: Creating CFR states ============");
        // Create CFR states for each agent
        // Modify the CFRState creation to include debugging info
        let states: Vec<_> = (0..num_agents)
            .map(|i| {
                println!("Creating CFR state for player {}", i);
                CFRState::new(game_state.clone())
            })
            .collect();

        // Optional visualization - wrap in a function that we only execute if not in CI
        fn try_visualization<F, E>(operation: &str, f: F) 
        where 
            F: FnOnce() -> Result<(), E>,
            E: std::fmt::Display
        {
            // Check if we're running in CI (GitHub Actions sets this environment variable)
            if std::env::var("CI").is_ok() {
                println!("Skipping visualization in CI environment: {}", operation);
                return;
            }
            
            match f() {
                Ok(_) => println!("Visualization successful: {}", operation),
                Err(e) => println!("Visualization failed (non-critical): {} - {}", operation, e),
            }
        }
        
        // Prepare visualization directory - only if not in CI
        let viz_dir = Path::new("target/cfr_visualization");
        try_visualization("Create visualization directory", || create_dir_all(viz_dir));

        // Export initial CFR state for player 0 (SVG and PNG) - only if not in CI
        let state0_before = &states[0];
        
        // SVG export
        let state0_before_svg_path = viz_dir.join("state0_before.svg");
        try_visualization("Export initial state to SVG", || {
            export_to_svg(state0_before, &state0_before_svg_path, true)
        });
        
        // PNG export
        let state0_before_png_path = viz_dir.join("state0_before.png");
        try_visualization("Export initial state to PNG", || {
            export_to_png(state0_before, &state0_before_png_path, true)
        });
            
        println!("Exported or skipped initial CFR state visualization");

        // Dump debug info about the initial CFR state
        println!("Initial CFR state for player 0");
        // Try to access the root node (index 0) if it exists
        if let Some(root_node) = state0_before.get(0) {
            println!("Root node data: {:?}", root_node.data);
        } else {
            println!("No root node found in CFR state");
        }

        println!("============ Test Step: Creating agents ============");
        // Create agents from the CFR states
        let agents: Vec<_> = states
            .iter()
            .enumerate()
            .map(|(i, s)| {
                println!("Creating agent for player {}", i);
                Box::new(CFRAgent::<BasicCFRActionGenerator>::new(s.clone(), i))
            })
            .collect();

        println!("============ Test Step: Creating historians ============");
        // Create historians for each agent
        let historians: Vec<Box<dyn Historian>> = agents
            .iter()
            .enumerate()
            .map(|(i, a)| {
                println!("Creating historian for player {}", i);
                Box::new(a.historian()) as Box<dyn Historian>
            })
            .collect();

        // Convert to dynamic trait objects for the simulation
        let dyn_agents = agents.into_iter().map(|a| a as Box<dyn Agent>).collect();

        println!("============ Test Step: Building simulation ============");
        // Build the simulation but don't run it yet
        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(dyn_agents)
            .historians(historians)
            .build()
            .unwrap();

        // Export intermediate CFR state for player 0 before running
        let state0_before_run = &states[0];
        
        // SVG export
        let state0_before_run_svg_path = viz_dir.join("state0_before_run.svg");
        try_visualization("Export state before run to SVG", || {
            export_to_svg(state0_before_run, &state0_before_run_svg_path, true)
        });
            
        // PNG export
        let state0_before_run_png_path = viz_dir.join("state0_before_run.png");
        try_visualization("Export state before run to PNG", || {
            export_to_png(state0_before_run, &state0_before_run_png_path, true)
        });
            
        println!("Exported or skipped CFR state visualization before running simulation");

        println!("============ Test Step: Running simulation ============");
        // Use a try-catch block to prevent the test from failing and allow export of final state
        let run_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Run the game
            sim.run();
        }));

        // Check if we had an error - use clone to avoid moving the value
        let had_error = run_result.is_err();
        if had_error {
            println!("Simulation panicked with an error");
        } else {
            println!("Simulation completed successfully");
        }

        println!("============ Test Step: Exporting final state ============");
        // Export the final state regardless of whether simulation succeeded
        let state0_after = &states[0];
        
        // SVG export
        let state0_after_svg_path = viz_dir.join("state0_after.svg");
        try_visualization("Export final state to SVG", || {
            export_to_svg(state0_after, &state0_after_svg_path, true)
        });
            
        // PNG export
        let state0_after_png_path = viz_dir.join("state0_after.png");
        try_visualization("Export final state to PNG", || {
            export_to_png(state0_after, &state0_after_png_path, true)
        });
            
        println!("Exported or skipped CFR state visualization after simulation");
        
        // Also export a DOT file for more detailed analysis
        let state0_after_dot_path = viz_dir.join("state0_after.dot");
        try_visualization("Export final state to DOT", || {
            export_cfr_state(state0_after, &state0_after_dot_path, ExportFormat::Dot)
        });

        // Check CFR state structure after simulation
        println!("Final CFR state for player 0");
        // Try to access the root node (index 0) if it exists
        if let Some(root_node) = state0_after.get(0) {
            println!("Root node data: {:?}", root_node.data);
            
            // Print a few additional nodes for debugging (if they exist)
            for i in 1..5 {
                if let Some(node) = state0_after.get(i) {
                    println!("Node {}: {:?}", i, node.data);
                }
            }
        } else {
            println!("No root node found in final CFR state");
        }

        // If we caught a panic, now re-throw it to make the test fail
        if had_error {
            panic!("Simulation failed, but we've exported the CFR state for debugging");
        }

        // Success criteria: The simulation completed without panicking
        // No assert needed - test passes if it doesn't panic
    }
}
