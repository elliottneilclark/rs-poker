#![no_main]

extern crate arbitrary;
extern crate libfuzzer_sys;
extern crate rand;
extern crate rs_poker;

use rand::{rngs::StdRng, SeedableRng};

use rs_poker::arena::{
    action::AgentAction, agent::VecReplayAgent, test_util::assert_valid_game_state,
    test_util::assert_valid_round_data, Agent, GameState, HoldemSimulation,
    RngHoldemSimulationBuilder,
};

use libfuzzer_sys::fuzz_target;

#[derive(Debug, Clone, arbitrary::Arbitrary)]
struct PlayerInput {
    pub stack: u16,
    pub actions: Vec<AgentAction>,
}

#[derive(Debug, Clone, arbitrary::Arbitrary)]
struct MultiInput {
    pub players: Vec<PlayerInput>,
    pub sb: i16,
    pub bb: i16,
    pub dealer_idx: usize,
    pub seed: u64,
}

fn build_agent(actions: Vec<AgentAction>) -> Box<dyn Agent> {
    Box::<VecReplayAgent>::new(VecReplayAgent::new(actions))
}

fuzz_target!(|input: MultiInput| {
    let num_players = input.players.len();
    let sb = input.sb as i32;
    let bb = input.bb as i32;

    // Extract the stacks adding the big blind to make sure that every agent can at
    // least post the blinds (which would be required at any tale in the world)
    let stacks: Vec<i32> = input
        .players
        .iter()
        .map(|pi| pi.stack as i32)
        .collect();

    // The Safety Valves for input
    if num_players < 2 {
        return;
    }
    if num_players > 9 {
        return;
    }
    if sb < 1 {
        return;
    }
    if bb < sb || bb < 2 {
        return;
    }
    if bb > *stacks.iter().min().unwrap_or(&0) {
        return;
    }

    let agents: Vec<Box<dyn Agent>> = input
        .players
        .into_iter()
        .map(|pi| build_agent(pi.actions))
        .collect();

    // Create the game state
    // Notice that dealer_idx is sanitized to ensure it's in the proper range here
    // rather than with the rest of the safety checks.
    let game_state = GameState::new(stacks, bb, sb, input.dealer_idx % agents.len());
    let rng = StdRng::seed_from_u64(input.seed);

    // Do the thing
    let mut sim: HoldemSimulation = RngHoldemSimulationBuilder::default()
        .rng(rng)
        .game_state(game_state)
        .agents(agents)
        .build()
        .unwrap();
    sim.run();

    // For every round that we saw
    // Check that it's valid
    sim.game_state
        .round_data
        .iter()
        .for_each(assert_valid_round_data);

    assert_valid_game_state(&sim.game_state);

    assert!(
        sim.actions.len() >= 10,
        "We expected there to be a lot of actions but only found {}",
        sim.actions.len()
    );
});
