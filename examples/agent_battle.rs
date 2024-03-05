extern crate rs_poker;

use rs_poker::arena::{
    agent::{RandomAgent, RandomPotControlAgent},
    competition::{
        CloneAgent, CloningAgentsGenerator, EmptyHistorianGenerator, HoldemCompetition,
        RandomGameStateGenerator, StandardSimulationGenerator,
    },
};

const ROUNDS_BATCH: usize = 500;
fn main() {
    // Start with very random dumb agents.
    let agents: Vec<Box<dyn CloneAgent>> = vec![
        Box::<RandomAgent>::default(),
        Box::new(RandomPotControlAgent::new(vec![0.7, 0.3])),
        Box::new(RandomPotControlAgent::new(vec![0.5, 0.3])),
        Box::new(RandomPotControlAgent::new(vec![0.3, 0.1])),
        Box::<RandomAgent>::default(),
    ];
    // Run the games with completely random hands.
    // Starting stack of at least 10 big blinds (10x10=100 chips)
    // Starting stack of no more than 1000 big blinds (10x1000=10000 chips)
    // This isn't deep stack poker at it's finest.
    let game_state_gen =
        RandomGameStateGenerator::new(agents.len(), 100.0, 10000.0, 10.0, 5.0, 0.0);
    let gen = StandardSimulationGenerator::new(
        CloningAgentsGenerator::new(agents),
        game_state_gen,
        EmptyHistorianGenerator,
    );
    let mut comp = HoldemCompetition::new(gen);
    for _i in 0..5000 {
        let _res = comp.run(ROUNDS_BATCH).expect("competition failed");
        println!("Current Competition Stats: {:?}", comp);
    }
}
