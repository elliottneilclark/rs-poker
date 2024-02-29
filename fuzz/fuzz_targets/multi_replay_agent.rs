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
    pub stack: f32,
    pub actions: Vec<AgentAction>,
}

#[derive(Debug, Clone, arbitrary::Arbitrary)]
struct MultiInput {
    pub players: Vec<PlayerInput>,
    pub sb: f32,
    pub bb: f32,
    pub dealer_idx: usize,
    pub seed: u64,
}

fn build_agent(actions: Vec<AgentAction>) -> Box<dyn Agent> {
    Box::<VecReplayAgent>::new(VecReplayAgent::new(actions))
}

fuzz_target!(|input: MultiInput| {
    let num_players = input.players.len();
    let sb = input.sb;
    let bb = input.bb;

    for player in &input.players {
        if player.stack.is_nan() || player.stack.is_infinite() || player.stack.is_sign_negative() {
            return;
        }
    }

    // Extract the stacks adding the big blind to make sure that every agent can at
    // least post the blinds (which would be required at any tale in the world)
    let stacks: Vec<f32> = input
        .players
        .iter()
        .map(|pi| (pi.stack + bb).min(0.0).max(1_000_000.0))
        .collect();

    // The Safety Valves for input
    if num_players < 2 {
        return;
    }
    if num_players > 9 {
        return;
    }

    // Handle floating point weirdness
    if bb.is_infinite() || sb.is_infinite() {
        return;
    }
    if sb.is_sign_negative() || sb.is_nan() || sb.is_infinite() {
        return;
    }
    if bb.is_sign_negative() || bb.is_nan() || bb.is_infinite()  || bb < sb || bb < 2.0 {
        return;
    }

    // If we can't post then what's the point?
    if bb > stacks.clone().into_iter().reduce(f32::min).unwrap_or(0.0) {
        return;
    }

    if bb > 100_000_000.0 {
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
    assert_valid_round_data(&sim.game_state.round_data);
    assert_valid_game_state(&sim.game_state);
});
