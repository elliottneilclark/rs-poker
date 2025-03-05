extern crate rs_poker;

use rs_poker::arena::{
    AgentGenerator, CloneHistorianGenerator, HistorianGenerator,
    agent::{CloneAgentGenerator, RandomAgentGenerator, RandomPotControlAgent},
    competition::{HoldemCompetition, StandardSimulationGenerator},
    game_state::RandomGameStateGenerator,
    historian::DirectoryHistorian,
};

const ROUNDS_BATCH: usize = 500;
fn main() {
    // Start with very random dumb agents.
    let agent_gens: Vec<Box<dyn AgentGenerator>> = vec![
        Box::<RandomAgentGenerator>::default(),
        Box::<RandomAgentGenerator>::default(),
        Box::new(CloneAgentGenerator::new(RandomPotControlAgent::new(vec![
            0.5, 0.3,
        ]))),
        Box::new(CloneAgentGenerator::new(RandomPotControlAgent::new(vec![
            0.3, 0.3,
        ]))),
    ];

    // Show how to use the historian to record the games.
    let path = std::env::current_dir().unwrap();
    let dir = path.join("historian_out");
    let hist_gens: Vec<Box<dyn HistorianGenerator>> = vec![Box::new(CloneHistorianGenerator::new(
        DirectoryHistorian::new(dir),
    ))];

    // Run the games with completely random hands.
    // Starting stack of at least 10 big blinds (10x10=100 chips)
    // Starting stack of no more than 1000 big blinds (10x1000=10000 chips)
    // This isn't deep stack poker at it's finest.
    let game_state_gen =
        RandomGameStateGenerator::new(agent_gens.len(), 100.0, 10000.0, 10.0, 5.0, 0.0);
    let simulation_gen = StandardSimulationGenerator::new(agent_gens, hist_gens, game_state_gen);
    let mut comp = HoldemCompetition::new(simulation_gen);
    for _i in 0..5000 {
        let _res = comp.run(ROUNDS_BATCH).expect("competition failed");
        println!("Current Competition Stats: {:?}", comp);
    }
}
