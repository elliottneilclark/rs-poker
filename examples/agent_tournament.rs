use std::vec;

use rs_poker::arena::{
    agent::{CallingAgentGenerator, RandomAgentGenerator},
    competition::SingleTableTournamentBuilder,
    AgentGenerator,
};

fn main() {
    let stacks = vec![100.0, 100.0, 50.0];

    let agent_builders: Vec<Box<dyn AgentGenerator>> = vec![
        Box::new(CallingAgentGenerator),
        Box::<RandomAgentGenerator>::default(),
        Box::<RandomAgentGenerator>::default(),
    ];

    let game_state = rs_poker::arena::game_state::GameState::new(stacks, 10.0, 5.0, 0.0, 0);

    let tournament = SingleTableTournamentBuilder::default()
        .agent_generators(agent_builders)
        .starting_game_state(game_state)
        .build()
        .unwrap();

    let results = tournament.run().unwrap();

    println!("Agent Results: {:?}", results);
}
