#![no_main]

extern crate arbitrary;
extern crate libfuzzer_sys;
extern crate rand;
extern crate rs_poker;

use rand::{rngs::StdRng, SeedableRng};

use rs_poker::arena::{
    action::AgentAction, agent::VecReplayAgent, historian::VecHistorian,
    test_util::assert_valid_game_state, test_util::assert_valid_round_data, Agent, GameState,
    HoldemSimulation, RngHoldemSimulationBuilder,
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
    pub ante: f32,
    pub dealer_idx: usize,
    pub seed: u64,
}

fn build_agent(actions: Vec<AgentAction>) -> Box<dyn Agent> {
    Box::<VecReplayAgent>::new(VecReplayAgent::new(actions))
}

fn input_good(input: &MultiInput) -> bool {
    for player in &input.players {
        if player.stack.is_nan() || player.stack.is_infinite() || player.stack.is_sign_negative() {
            return false;
        }
    }

    if input.players.len() <= 1 {
        return false;
    }

    if input.players.len() > 9 {
        return false;
    }

    // Handle floating point weirdness
    if input.ante.is_sign_negative()
        || input.ante.is_nan()
        || input.ante.is_infinite()
        || input.ante < 0.00
    {
        return false;
    }
    if input.sb.is_sign_negative()
        || input.sb.is_nan()
        || input.sb.is_infinite()
        || input.sb < input.ante
        || input.sb < 0.00
    {
        return false;
    }
    if input.bb.is_sign_negative()
        || input.bb.is_nan()
        || input.bb.is_infinite()
        || input.bb < input.sb
        || input.bb < 1.0
    {
        return false;
    }

    // If we can't post then what's the point?
    let min_stack = input
        .players
        .iter()
        .map(|p| p.stack)
        .clone()
        .into_iter()
        .reduce(f32::min)
        .unwrap_or(0.0);

    if input.bb + input.ante > min_stack {
        return false;
    }

    if input.bb > 100_000_000.0 {
        return false;
    }

    return true;
}

fuzz_target!(|input: MultiInput| {
    let sb = input.sb;
    let bb = input.bb;
    let ante = input.ante;

    if !input_good(&input) {
        return;
    }

    let stacks: Vec<f32> = input
        .players
        .iter()
        .map(|pi| (pi.stack).clamp(0.0, 1_000_000.0))
        .collect();

    let agents: Vec<Box<dyn Agent>> = input
        .players
        .into_iter()
        .map(|pi| build_agent(pi.actions))
        .collect();

    // Create the game state
    // Notice that dealer_idx is sanitized to ensure it's in the proper range here
    // rather than with the rest of the safety checks.
    let game_state = GameState::new(stacks, bb, sb, ante, input.dealer_idx % agents.len());
    let rng = StdRng::seed_from_u64(input.seed);

    let records = VecHistorian::new_storage();
    let hist = Box::new(VecHistorian::new(records.clone()));

    // Do the thing
    let mut sim: HoldemSimulation = RngHoldemSimulationBuilder::default()
        .rng(rng)
        .game_state(game_state)
        .agents(agents)
        .historians(vec![hist])
        .build()
        .unwrap();
    sim.run();

    for _record in records.borrow().iter() {
        // println!("{:?}", record.action);
    }
    assert_valid_round_data(&sim.game_state.round_data);
    assert_valid_game_state(&sim.game_state);
});
