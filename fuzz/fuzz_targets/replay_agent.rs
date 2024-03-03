#![no_main]

extern crate approx;
extern crate arbitrary;
extern crate libfuzzer_sys;
extern crate rand;
extern crate rs_poker;

use approx::assert_relative_ne;
use rand::{rngs::StdRng, SeedableRng};

use rs_poker::arena::{
    action::AgentAction, agent::VecReplayAgent, game_state::Round,
    test_util::assert_valid_round_data, Agent, GameState, HoldemSimulation,
    RngHoldemSimulationBuilder,
};

use libfuzzer_sys::fuzz_target;

#[derive(Debug, Clone, arbitrary::Arbitrary)]
struct Input {
    pub dealer_actions: Vec<AgentAction>,
    pub sb_actions: Vec<AgentAction>,
    pub seed: u64,
}

fuzz_target!(|input: Input| {
    let stacks = vec![50.0; 2];
    let game_state = GameState::new(stacks, 2.0, 1.0, 0.0, 0);
    let agents: Vec<Box<dyn Agent>> = vec![
        Box::<VecReplayAgent>::new(VecReplayAgent::new(input.dealer_actions)),
        Box::<VecReplayAgent>::new(VecReplayAgent::new(input.sb_actions)),
    ];
    let rng = StdRng::seed_from_u64(input.seed);
    let mut sim: HoldemSimulation = RngHoldemSimulationBuilder::default()
        .rng(rng)
        .game_state(game_state)
        .agents(agents)
        .build()
        .unwrap();
    sim.run();

    assert_eq!(Round::Complete, sim.game_state.round);
    assert_relative_ne!(0.0_f32, sim.game_state.player_bet.iter().sum());

    assert_valid_round_data(&sim.game_state.round_data);
});
