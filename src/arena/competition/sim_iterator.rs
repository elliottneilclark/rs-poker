use rand::{SeedableRng, rng as os_rng, rngs::StdRng};

use crate::arena::{
    AgentGenerator, GameState, HoldemSimulation, HoldemSimulationBuilder,
    historian::HistorianGenerator,
};

/// Iterator that materialises one [`HoldemSimulation`] per item pulled from
/// the underlying game-state iterator.
///
/// The iterator owns an RNG used exclusively to generate per-simulation
/// IDs (see [`HoldemSimulationBuilder::build_with_rng`]). By default the
/// RNG is seeded from OS entropy, so repeated runs produce distinct IDs.
/// For deterministic runs (e.g., reproducing a seeded competition), build
/// with [`StandardSimulationIterator::with_rng`] and pass an RNG forked
/// from the competition's own RNG.
pub struct StandardSimulationIterator<G>
where
    G: Iterator<Item = GameState>,
{
    agent_generators: Vec<Box<dyn AgentGenerator>>,
    historian_generators: Vec<Box<dyn HistorianGenerator>>,
    game_state_iterator: G,
    rng: StdRng,
}

impl<G> StandardSimulationIterator<G>
where
    G: Iterator<Item = GameState>,
{
    /// Create a new iterator seeded from OS entropy.
    ///
    /// Equivalent to [`Self::with_rng`] seeded from `rand::rng()`.
    pub fn new(
        agent_generators: Vec<Box<dyn AgentGenerator>>,
        historian_generators: Vec<Box<dyn HistorianGenerator>>,
        game_state_iterator: G,
    ) -> StandardSimulationIterator<G> {
        Self::with_rng(
            agent_generators,
            historian_generators,
            game_state_iterator,
            StdRng::from_rng(&mut os_rng()),
        )
    }

    /// Create a new iterator that draws simulation IDs from the provided
    /// RNG. Use this when you need repeated runs of the same competition
    /// to produce identical simulation IDs (and therefore identical hand
    /// histories and CFR keys).
    pub fn with_rng(
        agent_generators: Vec<Box<dyn AgentGenerator>>,
        historian_generators: Vec<Box<dyn HistorianGenerator>>,
        game_state_iterator: G,
        rng: StdRng,
    ) -> StandardSimulationIterator<G> {
        StandardSimulationIterator {
            agent_generators,
            historian_generators,
            game_state_iterator,
            rng,
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
            .build_with_rng(&mut self.rng)
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
    use crate::arena::{agent::FoldingAgentGenerator, game_state::CloneGameStateGenerator};

    use super::*;
    use crate::arena::GameStateBuilder;

    #[test]
    fn test_static_simulation_generator() {
        let generators: Vec<Box<dyn AgentGenerator>> = vec![
            Box::<FoldingAgentGenerator>::default(),
            Box::<FoldingAgentGenerator>::default(),
            Box::<FoldingAgentGenerator>::default(),
        ];
        let stacks = vec![100.0; 3];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let mut sim_gen = StandardSimulationIterator::new(
            generators,
            vec![],
            CloneGameStateGenerator::new(game_state),
        );

        let _first = sim_gen
            .next()
            .expect("There should always be a first simulation");
    }

    /// Regression test for M2: two iterators seeded from the same RNG
    /// must produce identical simulation IDs. Previously the builder
    /// called `rand::rng()` unconditionally, so repeated runs diverged
    /// even with identical inputs.
    #[test]
    fn test_with_rng_is_deterministic() {
        let build = || {
            let generators: Vec<Box<dyn AgentGenerator>> = vec![
                Box::<FoldingAgentGenerator>::default(),
                Box::<FoldingAgentGenerator>::default(),
            ];
            let game_state = GameStateBuilder::new()
                .stacks(vec![100.0; 2])
                .blinds(10.0, 5.0)
                .build()
                .unwrap();
            StandardSimulationIterator::with_rng(
                generators,
                vec![],
                CloneGameStateGenerator::new(game_state),
                StdRng::seed_from_u64(42),
            )
        };

        let mut a = build();
        let mut b = build();
        for _ in 0..5 {
            let sa = a.next().unwrap();
            let sb = b.next().unwrap();
            assert_eq!(sa.id, sb.id);
        }
    }
}
