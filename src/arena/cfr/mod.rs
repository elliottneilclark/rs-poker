mod agent;
mod historian;
mod node;
mod state;

pub const PREFLOP_EXPERTS: [usize; 6] = [0, 1, 2, 3, 4, 5];
pub const EXPERTS: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

pub const MAX_RAISE_PREFLOP_EXPERTS: [usize; 2] = [0, 5];
pub const MAX_RAISE_EXPERTS: [usize; 2] = [0, 7];

// The ranges that we consider for random bet sizes
pub const LOWER_MULT: f32 = 0.9;
pub const UPPER_MULT: f32 = 1.1;

pub use agent::*;
pub use historian::*;
pub use state::*;

#[cfg(test)]
mod tests {
    
    use std::{cell::RefCell, rc::Rc};

    use crate::arena::{historian::Historian, Agent, GameState, HoldemSimulationBuilder};

    use super::*;

    #[test]
    fn test_should_fold_all_in() {
        let num_agents = 2;
        assert_eq!(2, num_agents);
    }

    #[test]
    fn test_crf() {
        let num_agents = 2;
        let game_state = GameState::new_starting(vec![100.0; num_agents], 10.0, 5.0, 0.0, 0);
        // CFR states for each seat
        let cfr_states: Vec<_> = (0..num_agents)
            .map(|_| Rc::new(RefCell::new(PlayerCFRState::new(game_state.clone()))))
            .collect();

        let save_points: Vec<CFRSavePoint> = cfr_states
            .iter()
            .map(|state| state.borrow().save_point())
            .collect();

        // Test a lot of simulations to show that we can add on state that overlaps
        // however we don't need this to be a fuzz test
        for _ in 0..100 {
            let agents: Vec<Box<dyn Agent>> = cfr_states
                .iter()
                .enumerate()
                .map(|(idx, _state)| {
                    // pass in all the states and the index of the current agent
                    // This allows each agent to run simulations with how the other
                    // agents would play
                    Box::new(ArenaCFRAgent::new(cfr_states.clone(), idx)) as Box<dyn Agent>
                })
                .collect();

            // The historians to watch
            let historians = cfr_states
                .iter()
                .enumerate()
                .map(|(idx, state)| {
                    Box::new(ArenaCFRHistorian::new(state.clone(), idx)) as Box<dyn Historian>
                })
                .collect();

            // Build the simulation
            let mut sim = HoldemSimulationBuilder::default()
                .agents(agents)
                .historians(historians)
                .game_state(game_state.clone())
                .build()
                .unwrap();

            sim.run();

            for (idx, state) in cfr_states.iter().enumerate() {
                state.borrow_mut().restore_save_point(save_points[idx]);
            }
        }
    }
}
