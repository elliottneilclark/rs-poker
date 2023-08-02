#![no_main]

extern crate arbitrary;
extern crate libfuzzer_sys;
extern crate rand;
extern crate rs_poker;

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
    let stacks = vec![50; 2];
    let game_state = GameState::new(stacks, 2, 1, 0);
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
    assert_ne!(0, sim.game_state.player_bet.iter().sum());

    // For every round that we saw
    // Check that it's valid
    sim.game_state
        .round_data
        .iter()
        .for_each(assert_valid_round_data);

    assert!(sim.actions.len() > 10, "We expected there to be a lot of actions but only found {}", sim.actions.len());
});
