extern crate rs_poker;

use rs_poker::arena::{
    agent::{RandomAgent, RandomPotControlAgent},
    competition::{holdem_competition::HoldemCompetition, sim_gen::RandomHandSimulationGenerator},
    Agent,
};

const ROUNDS_BATCH: usize = 500;
fn main() {
    // Start with very random dumb agents.
    let agents: Vec<Box<dyn Agent>> = vec![
        Box::<RandomAgent>::default(),
        Box::new(RandomPotControlAgent::new(vec![0.7, 0.3])),
        Box::new(RandomPotControlAgent::new(vec![0.5, 0.3])),
        Box::new(RandomPotControlAgent::new(vec![0.3, 0.1])),
        Box::<RandomAgent>::default(),
    ];
    // Run the games with completely random hands.
    // Starting stack of at least 10 big blinds
    // Starting stack of no more than 100 big blinds
    // This isn't deep stack poker at it's finest.
    let gen = RandomHandSimulationGenerator {
        agents,
        min_stack: 100,
        max_stack: 10000,
        big_blind: 10,
        small_blind: 5,
    };
    let mut comp = HoldemCompetition::new(gen);
    for _i in 0..5000 {
        let _res = comp.run(ROUNDS_BATCH).expect("competition failed");
        println!("{:?}", comp);
    }
}
