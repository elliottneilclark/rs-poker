use rand::rng;

use crate::arena::{
    historian::HistorianGenerator, AgentGenerator, GameStateGenerator, HoldemSimulation,
    HoldemSimulationBuilder,
};

pub trait HoldemSimulationGenerator: Iterator<Item = HoldemSimulation> {}

pub struct StandardSimulationGenerator<G>
where
    G: GameStateGenerator,
{
    agent_generators: Vec<Box<dyn AgentGenerator>>,
    historian_generators: Vec<Box<dyn HistorianGenerator>>,
    game_state_generator: G,
}

impl<G> StandardSimulationGenerator<G>
where
    G: GameStateGenerator,
{
    pub fn new(
        agent_generators: Vec<Box<dyn AgentGenerator>>,
        historian_generators: Vec<Box<dyn HistorianGenerator>>,
        game_state_generator: G,
    ) -> StandardSimulationGenerator<G> {
        StandardSimulationGenerator {
            agent_generators,
            historian_generators,
            game_state_generator,
        }
    }
}

impl<G> Iterator for StandardSimulationGenerator<G>
where
    G: GameStateGenerator,
{
    type Item = HoldemSimulation;

    fn next(&mut self) -> Option<Self::Item> {
        // Get a hold of the thread local rng
        let rng = rng();

        let game_state = self.game_state_generator.generate();
        let agents = self
            .agent_generators
            .iter()
            .map(|g| g.generate(&game_state))
            .collect();
        let historians = self
            .historian_generators
            .iter()
            .map(|g| g.generate(&game_state))
            .collect();

        HoldemSimulationBuilder::default()
            .agents(agents)
            .historians(historians)
            .game_state(game_state)
            .rng(rng)
            .build()
            .ok()
    }
}

impl<G> HoldemSimulationGenerator for StandardSimulationGenerator<G> where G: GameStateGenerator {}

#[cfg(test)]
mod tests {
    use crate::arena::{
        agent::FoldingAgentGenerator, game_state::CloneGameStateGenerator, GameState,
    };

    use super::*;

    #[test]
    fn test_static_simulation_generator() {
        let generators: Vec<Box<dyn AgentGenerator>> = vec![
            Box::<FoldingAgentGenerator>::default(),
            Box::<FoldingAgentGenerator>::default(),
            Box::<FoldingAgentGenerator>::default(),
        ];
        let stacks = vec![100.0; 3];
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut sim_gen = StandardSimulationGenerator::new(
            generators,
            vec![],
            CloneGameStateGenerator::new(game_state),
        );

        let first = sim_gen
            .next()
            .expect("There should always be a first simulation");

        for (_idx, sim) in (0..10).zip(sim_gen) {
            assert_ne!(first.deck, sim.deck);
        }
    }
}
