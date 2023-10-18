use dyn_clone::DynClone;
use game_state::GameState;
use rand::{thread_rng, Rng};

use crate::arena::{game_state, Agent, HoldemSimulation, HoldemSimulationBuilder};

pub trait HoldemSimulationGenerator: DynClone + IntoIterator<Item = HoldemSimulation> {
    fn num_agents(&self) -> usize;
}

#[derive(Clone)]
pub struct StaticHandGenerator {
    /// The simulation that is being
    /// run including agents, gamestate, and decks
    pub sim: HoldemSimulation,
}

impl Iterator for StaticHandGenerator {
    type Item = HoldemSimulation;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.sim.clone())
    }
}

impl HoldemSimulationGenerator for StaticHandGenerator {
    fn num_agents(&self) -> usize {
        self.sim.num_agents()
    }
}

#[derive(Clone)]
pub struct RandomHandSimulationGenerator {
    pub agents: Vec<Box<dyn Agent>>,
    pub min_stack: i32,
    pub max_stack: i32,
    pub big_blind: i32,
    pub small_blind: i32,
}

impl Iterator for RandomHandSimulationGenerator {
    type Item = HoldemSimulation;

    fn next(&mut self) -> Option<Self::Item> {
        // Get a hold of the thread local rng
        let mut rng = thread_rng();
        let agents: Vec<Box<dyn Agent>> = self.agents.clone();
        // Choose some random numbers in the range for the hand's starting values
        let stacks = (0..agents.len())
            .map(|_idx| rng.gen_range(self.min_stack..self.max_stack))
            .collect();
        // Create a new game state with nothing started
        let game_state = GameState::new(
            stacks,
            self.big_blind,
            self.small_blind,
            rng.gen_range(0..agents.len()),
        );
        // Build the simulation passing along a clone
        // of the agents and the rng that we used.
        HoldemSimulationBuilder::default()
            .agents(agents)
            .game_state(game_state)
            .rng(rng)
            .build()
            .ok()
    }
}

impl HoldemSimulationGenerator for RandomHandSimulationGenerator {
    fn num_agents(&self) -> usize {
        self.agents.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::agent::FoldingAgent;

    use super::*;

    #[test]
    fn test_static_simulation_generator() {
        let rng = thread_rng();
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<FoldingAgent>::default(),
            Box::<FoldingAgent>::default(),
        ];
        let stacks: Vec<i32> = vec![];
        let game_state = GameState::new(stacks, 2, 1, 0);
        let sim = HoldemSimulationBuilder::default()
            .agents(agents)
            .game_state(game_state)
            .rng(rng)
            .build()
            .expect("Wut");

        let mut sim_gen = StaticHandGenerator { sim };

        let first = sim_gen
            .next()
            .expect("There should always be a first simulation");

        for (_idx, sim) in (0..10).zip(sim_gen) {
            assert_eq!(first.game_state, sim.game_state);
            assert_eq!(first.deck, sim.deck);
        }
    }
}
