#![no_main]

extern crate arbitrary;
extern crate libfuzzer_sys;
extern crate rand;
extern crate rs_poker;

use rand::{rngs::StdRng, SeedableRng};

use rs_poker::arena::{
    agent::{AgentConfig, CfrExploration, ConfigAgentBuilder},
    cfr::{BudgetItem, CFRState, TraversalSet},
    historian::{self, OpenHandHistoryVecHistorian},
    test_util::assert_valid_game_state,
    test_util::assert_valid_round_data,
    GameStateBuilder, HoldemSimulation, HoldemSimulationBuilder,
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

/// Validate CFR exploration limits to keep fuzzing fast and terminating.
///
/// Only accept budgets whose items are all small count-bound
/// `IterationCount`/`PerDepthIterations` entries (every count in `1..=5`,
/// non-empty). Time-based deadlines, regret-floor budgets, and other
/// open-ended shapes are rejected — they could hang the fuzzer. An
/// unset budget falls through to the library's small safe default,
/// which is also acceptable.
fn valid_cfr_limits(config: &AgentConfig) -> bool {
    fn valid_item(item: &BudgetItem) -> bool {
        match item {
            BudgetItem::IterationCount { max } => (1..=5).contains(max),
            BudgetItem::PerDepthIterations { counts, fallback } => {
                !counts.is_empty()
                    && counts.iter().all(|c| (1..=5).contains(c))
                    && (1..=5).contains(fallback)
            }
            _ => false,
        }
    }

    fn valid_exploration(exploration: &CfrExploration) -> bool {
        match exploration.budget.as_ref() {
            None => true,
            Some(cfg) => !cfg.0.is_empty() && cfg.0.iter().all(valid_item),
        }
    }

    match config {
        AgentConfig::CfrBasic { exploration, .. }
        | AgentConfig::CfrSimple { exploration, .. }
        | AgentConfig::CfrConfigurable { exploration, .. }
        | AgentConfig::CfrPreflopChart { exploration, .. } => valid_exploration(exploration),
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

    // Check if any agent is CFR-based; if so create shared context
    let has_cfr = input.player_configs.iter().any(|c| c.is_cfr());
    let cfr_context = if has_cfr {
        let cfr_state = CFRState::new(game_state.clone());
        let traversal_set = TraversalSet::new(game_state.num_players);
        Some((cfr_state, traversal_set))
    } else {
        None
    };

    // Build agents from configs
    // Skip configs that fail validation (e.g., preset preflop charts not supported)
    let agents: Result<Vec<Box<dyn rs_poker::arena::Agent>>, _> = input
        .player_configs
        .iter()
        .enumerate()
        .map(|(idx, config)| {
            let mut builder = ConfigAgentBuilder::new(config.clone())?
                .player_idx(idx)
                .game_state(game_state.clone());
            if let Some((ref cfr_state, ref ts)) = cfr_context {
                builder = builder.cfr_context(cfr_state.clone(), ts.clone());
            }
            Ok(builder.build())
        })
        .collect();

    let agents = match agents {
        Ok(a) => a,
        Err::<_, rs_poker::arena::agent::AgentConfigError>(_) => return,
    };

    // Set up historian
    let open_hand_hist = Box::new(OpenHandHistoryVecHistorian::new());
    let hand_storage = open_hand_hist.get_storage();
    let historians: Vec<Box<dyn historian::Historian>> = vec![open_hand_hist];

    // Run the simulation
    let mut sim_builder = HoldemSimulationBuilder::default()
        .game_state(game_state)
        .agents(agents)
        .historians(historians);
    if let Some((cfr_state, traversal_set)) = cfr_context {
        sim_builder = sim_builder.cfr_context(cfr_state, traversal_set, true);
    }
    let mut sim: HoldemSimulation = sim_builder
        .build_with_rng(StdRng::seed_from_u64(input.seed))
        .unwrap();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(sim.run());

    // Validate results
    assert_valid_round_data(&sim.game_state.round_data);
    assert_valid_game_state(&sim.game_state);

    let hands = hand_storage.lock().unwrap();
    assert!(!hands.is_empty());
    for hand in hands.iter() {
        if std::env::var_os("DUMP_HAND").is_some() {
            println!("{hand:#?}");
        }
        assert_valid_open_hand_history(hand);
        assert_open_hand_history_matches_game_state(hand, &sim.game_state);
    }
});
