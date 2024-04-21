use std::vec;

use rs_poker::arena::{
    agent::{CallingAgentBuilder, RandomAgentBuilder},
    tournament, AgentBuilder,
};

fn main() {
    let stacks = vec![100.0, 100.0, 50.0];

    let agent_builders: Vec<Box<dyn AgentBuilder>> = vec![
        Box::new(CallingAgentBuilder),
        Box::<RandomAgentBuilder>::default(),
        Box::<RandomAgentBuilder>::default(),
    ];

    let game_state = rs_poker::arena::game_state::GameState::new(stacks, 10.0, 5.0, 0.0, 0);

    let tournament = tournament::SingleTableTournamentBuilder::default()
        .agent_builders(agent_builders)
        .starting_game_state(game_state)
        .build()
        .unwrap();

    let results = tournament.run().unwrap();

    println!("Agent Results: {:?}", results);
}
