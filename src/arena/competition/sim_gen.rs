use rand::thread_rng;

use crate::arena::{HoldemSimulation, HoldemSimulationBuilder};

use super::{AgentsGenerator, GameStateGenerator, HistorianGenerator};

pub trait HoldemSimulationGenerator: Iterator<Item = HoldemSimulation> {}

pub struct StandardSimulationGenerator<A, G, H>
where
    A: AgentsGenerator,
    G: GameStateGenerator,
    H: HistorianGenerator,
{
    agent_vec_generator: A,
    game_state_generator: G,
    historian_generator: H,
}

impl<A, G, H> StandardSimulationGenerator<A, G, H>
where
    A: AgentsGenerator,
    G: GameStateGenerator,
    H: HistorianGenerator,
{
    pub fn new(
        agent_vec_generator: A,
        game_state_generator: G,
        historian_generator: H,
    ) -> StandardSimulationGenerator<A, G, H> {
        StandardSimulationGenerator {
            agent_vec_generator,
            game_state_generator,
            historian_generator,
        }
    }
}

impl<A, G, H> Iterator for StandardSimulationGenerator<A, G, H>
where
    A: AgentsGenerator,
    G: GameStateGenerator,
    H: HistorianGenerator,
{
    type Item = HoldemSimulation;

    fn next(&mut self) -> Option<Self::Item> {
        // Get a hold of the thread local rng
        let rng = thread_rng();

        let game_state = self.game_state_generator.generate();
        let agents = self.agent_vec_generator.generate(&game_state);
        let historians = self.historian_generator.generate(&game_state, &agents);

        HoldemSimulationBuilder::default()
            .agents(agents)
            .historians(historians)
            .game_state(game_state)
            .rng(rng)
            .build()
            .ok()
    }
}

impl<A, G, H> HoldemSimulationGenerator for StandardSimulationGenerator<A, G, H>
where
    A: AgentsGenerator,
    G: GameStateGenerator,
    H: HistorianGenerator,
{
}

#[cfg(test)]
mod tests {
    use crate::arena::{
        agent::FoldingAgent,
        competition::{
            CloneAgent, CloneGameStateGenerator, CloningAgentsGenerator, EmptyHistorianGenerator,
        },
        GameState,
    };

    use super::*;

    #[test]
    fn test_static_simulation_generator() {
        let agents: Vec<Box<dyn CloneAgent>> = vec![
            Box::<FoldingAgent>::default(),
            Box::<FoldingAgent>::default(),
        ];

        let stacks = vec![100.0, 100.0];
        let game_state = GameState::new(stacks, 10.0, 5.0, 0);
        let mut sim_gen = StandardSimulationGenerator::new(
            CloningAgentsGenerator::new(agents),
            CloneGameStateGenerator::new(game_state),
            EmptyHistorianGenerator,
        );

        let first = sim_gen
            .next()
            .expect("There should always be a first simulation");

        for (_idx, sim) in (0..10).zip(sim_gen) {
            assert_ne!(first.deck, sim.deck);
        }
    }
}
