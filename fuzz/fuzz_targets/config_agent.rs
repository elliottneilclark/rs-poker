#![no_main]

extern crate arbitrary;
extern crate libfuzzer_sys;
extern crate rand;
extern crate rs_poker;

use rand::{rngs::StdRng, SeedableRng};

use rs_poker::arena::{
    agent::{AgentConfig, ConfigAgentGenerator},
    historian::{self, OpenHandHistoryVecHistorian},
    test_util::assert_valid_game_state,
    test_util::assert_valid_round_data,
    AgentGenerator, GameState, HoldemSimulation, HoldemSimulationBuilder,
};
use rs_poker::open_hand_history::{
    assert_open_hand_history_matches_game_state, assert_valid_open_hand_history,
};

use libfuzzer_sys::fuzz_target;

const MIN_BLIND: f32 = 1e-15;

#[derive(Debug, Clone, arbitrary::Arbitrary)]
struct ConfigAgentInput {
    pub player_configs: Vec<AgentConfig>,
    pub stacks: Vec<f32>,
    pub sb: f32,
    pub bb: f32,
    pub ante: f32,
    pub dealer_idx: usize,
    pub seed: u64,
}

/// Validate probability vectors are non-empty and in [0.0, 1.0]
fn valid_probabilities(probs: &[f64]) -> bool {
    !probs.is_empty() && probs.iter().all(|&p| (0.0..=1.0).contains(&p))
}

/// Validate CFR iteration limits to keep fuzzing fast
/// Require at least 1 iteration to avoid uninitialized state issues
fn valid_cfr_limits(config: &AgentConfig) -> bool {
    /// Validate depth_hands: non-empty, all values >= 1 and <= 5
    fn valid_depth_hands(depth_hands: &[usize]) -> bool {
        !depth_hands.is_empty() && depth_hands.iter().all(|&h| h >= 1 && h <= 5)
    }

    match config {
        AgentConfig::CfrBasic { depth_hands, .. }
        | AgentConfig::CfrSimple { depth_hands, .. }
        | AgentConfig::CfrConfigurable { depth_hands, .. }
        | AgentConfig::CfrPreflopChart { depth_hands, .. } => valid_depth_hands(depth_hands),
        _ => true,
    }
}

fn input_good(input: &ConfigAgentInput) -> bool {
    // Player count 2-9
    if input.player_configs.len() < 2 || input.player_configs.len() > 9 {
        return false;
    }

    // Stacks length must match player count
    if input.stacks.len() != input.player_configs.len() {
        return false;
    }

    // Validate stacks - no NaN/infinite/negative
    for stack in &input.stacks {
        if stack.is_nan() || stack.is_infinite() || stack.is_sign_negative() {
            return false;
        }
    }

    // Validate blinds and ante
    if input.ante.is_sign_negative()
        || input.ante.is_nan()
        || input.ante.is_infinite()
        || input.ante < 0.0
    {
        return false;
    }
    if input.sb.is_sign_negative()
        || input.sb.is_nan()
        || input.sb.is_infinite()
        || input.sb < input.ante
        || input.sb < 0.0
        || (input.sb > 0.0 && input.sb < MIN_BLIND)
    {
        return false;
    }
    if input.bb.is_sign_negative()
        || input.bb.is_nan()
        || input.bb.is_infinite()
        || input.bb < input.sb
        || input.bb < 1.0
        || (input.bb > 0.0 && input.bb < MIN_BLIND)
    {
        return false;
    }

    // Check that min stack covers bb + ante
    let min_stack = input
        .stacks
        .iter()
        .map(|s| s.clamp(0.0, 100_000_000.0))
        .reduce(f32::min)
        .unwrap_or(0.0);

    if input.bb + input.ante > min_stack {
        return false;
    }

    if input.bb > 100_000_000.0 {
        return false;
    }

    // Validate each agent config
    for config in &input.player_configs {
        // Check probability vectors for Random and RandomPotControl
        match config {
            AgentConfig::Random {
                percent_fold,
                percent_call,
                ..
            } => {
                if !valid_probabilities(percent_fold) || !valid_probabilities(percent_call) {
                    return false;
                }
            }
            AgentConfig::RandomPotControl { percent_call, .. } => {
                if !valid_probabilities(percent_call) {
                    return false;
                }
            }
            _ => {}
        }

        // Validate CFR iteration limits
        if !valid_cfr_limits(config) {
            return false;
        }
    }

    true
}

fuzz_target!(|input: ConfigAgentInput| {
    if !input_good(&input) {
        return;
    }

    let stacks: Vec<f32> = input
        .stacks
        .iter()
        .map(|s| s.clamp(0.0, 100_000_000.0))
        .collect();

    let dealer_idx = input.dealer_idx % input.player_configs.len();

    // Create game state using the builder
    let game_state = match GameStateBuilder::new()
        .stacks(stacks)
        .blinds(input.bb, input.sb)
        .ante(input.ante)
        .dealer_idx(dealer_idx)
        .build()
    {
        Ok(gs) => gs,
        Err(_) => return, // Invalid input, skip
    };

    // Generate agents from configs
    // CFR agents now create their own StateStore internally
    // Skip configs that fail validation (e.g., preset preflop charts not supported)
    let generators: Result<Vec<_>, _> = input
        .player_configs
        .iter()
        .map(|config| ConfigAgentGenerator::new(config.clone()))
        .collect();

    let generators = match generators {
        Ok(g) => g,
        Err(_) => return, // Skip invalid configs (e.g., preset charts)
    };

    let agents: Vec<Box<dyn rs_poker::arena::Agent>> = generators
        .iter()
        .enumerate()
        .map(|(idx, generator)| generator.generate(idx, &game_state))
        .collect();

    // Set up historian
    let open_hand_hist = Box::new(OpenHandHistoryVecHistorian::new());
    let hand_storage = open_hand_hist.get_storage();
    let historians: Vec<Box<dyn historian::Historian>> = vec![open_hand_hist];

    // Create RNG
    let mut rng = StdRng::seed_from_u64(input.seed);

    // Run the simulation
    let mut sim: HoldemSimulation = HoldemSimulationBuilder::default()
        .game_state(game_state)
        .agents(agents)
        .historians(historians)
        .max_raises_per_round(Some(3))
        .build()
        .unwrap();
    sim.run(&mut rng);

    // Validate results
    assert_valid_round_data(&sim.game_state.round_data);
    assert_valid_game_state(&sim.game_state);

    let hands = hand_storage.borrow();
    assert!(!hands.is_empty());
    for hand in hands.iter() {
        if std::env::var_os("DUMP_HAND").is_some() {
            println!("{hand:#?}");
        }
        assert_valid_open_hand_history(hand);
        assert_open_hand_history_matches_game_state(hand, &sim.game_state);
    }
});
