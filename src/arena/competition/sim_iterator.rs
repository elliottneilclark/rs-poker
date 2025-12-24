use crate::arena::{
    AgentGenerator, GameState, HoldemSimulation, HoldemSimulationBuilder,
    historian::HistorianGenerator,
};

pub struct StandardSimulationIterator<G>
where
    G: Iterator<Item = GameState>,
{
    agent_generators: Vec<Box<dyn AgentGenerator>>,
    historian_generators: Vec<Box<dyn HistorianGenerator>>,
    game_state_iterator: G,
}

impl<G> StandardSimulationIterator<G>
where
    G: Iterator<Item = GameState>,
{
    pub fn new(
        agent_generators: Vec<Box<dyn AgentGenerator>>,
        historian_generators: Vec<Box<dyn HistorianGenerator>>,
        game_state_iterator: G,
    ) -> StandardSimulationIterator<G> {
        StandardSimulationIterator {
            agent_generators,
            historian_generators,
            game_state_iterator,
        }
    }
}

impl<G> StandardSimulationIterator<G>
where
    G: Iterator<Item = GameState>,
{
    fn generate(&mut self, game_state: GameState) -> Option<HoldemSimulation> {
        let agents = self
            .agent_generators
            .iter()
            .enumerate()
            .map(|(idx, g)| g.generate(idx, &game_state))
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
            .build()
            .ok()
    }
}

impl<G> Iterator for StandardSimulationIterator<G>
where
    G: Iterator<Item = GameState>,
{
    type Item = HoldemSimulation;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(game_state) = self.game_state_iterator.next() {
            self.generate(game_state)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::{
        GameState, agent::FoldingAgentGenerator, game_state::CloneGameStateGenerator,
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
        let mut sim_gen = StandardSimulationIterator::new(
            generators,
            vec![],
            CloneGameStateGenerator::new(game_state),
        );

        let _first = sim_gen
            .next()
            .expect("There should always be a first simulation");
    }
}
