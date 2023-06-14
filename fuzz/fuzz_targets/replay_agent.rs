#![no_main]

extern crate arbitrary;
extern crate libfuzzer_sys;
extern crate rs_poker;

use rs_poker::arena::{
    action::AgentAction, agent::VecReplayAgent, game_state::Round,
    test_util::assert_valid_round_data, Agent, GameState, HoldemSimulation,
};

use libfuzzer_sys::fuzz_target;

#[derive(Debug, Clone, arbitrary::Arbitrary)]
struct Input {
    pub dealer_actions: Vec<AgentAction>,
    pub sb_actions: Vec<AgentAction>,
}

fuzz_target!(|input: Input| {
    let stacks = vec![1000; 2];
    let game_state = GameState::new(stacks, 2, 1, 0);
    let agents: Vec<Box<dyn Agent>> = vec![
        Box::<VecReplayAgent>::new(VecReplayAgent::new(input.dealer_actions)),
        Box::<VecReplayAgent>::new(VecReplayAgent::new(input.sb_actions)),
    ];
    let mut sim = HoldemSimulation::new_with_agents(game_state, agents);
    sim.run();

    assert_eq!(Round::Complete, sim.game_state.round);
    assert_ne!(0, sim.game_state.player_bet.iter().sum());

    // For every round that we saw
    // Check that it's valid
    sim.game_state
        .round_data
        .iter()
        .for_each(assert_valid_round_data);
});
