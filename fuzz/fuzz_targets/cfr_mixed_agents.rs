#![no_main]

extern crate arbitrary;
extern crate libfuzzer_sys;
extern crate rand;
extern crate rs_poker;

use libfuzzer_sys::fuzz_target;
use rand::{rngs::StdRng, SeedableRng};
use std::sync::Arc;

use rs_poker::arena::{
    action::AgentAction,
    agent::{CallingAgent, FoldingAgent, VecReplayAgent},
    cfr::{
        Budget, CFRAgentBuilder, CFRState, ConfigurableActionConfig, ConfigurableActionGenerator,
        IterationCount, PerDepth, RoundActionConfig, SimpleActionGenerator, TraversalSet,
    },
    game_state::Round,
    test_util::assert_valid_game_state,
    Agent, GameStateBuilder, HoldemSimulationBuilder,
};

/// Which CFR action generator variant to use
#[derive(Debug, Clone, Copy, arbitrary::Arbitrary)]
enum CfrVariant {
    Simple,
    Configurable,
}

/// Input for fuzzing mixed CFR and non-CFR agent scenarios.
#[derive(Debug, Clone, arbitrary::Arbitrary)]
struct MixedAgentInput {
    /// Which players are CFR agents (true = CFR, false = non-CFR)
    pub cfr_indices: [bool; 2],
    /// Which CFR variant to use for each player
    pub cfr_variants: [CfrVariant; 2],
    /// Actions for replay agents (used when cfr_indices[i] is false)
    pub player0_actions: Vec<AgentAction>,
    pub player1_actions: Vec<AgentAction>,
    /// RNG seed
    pub seed: u64,
    /// Per-depth wave (iteration) counts (capped to avoid slow tests)
    pub depth_0_iters: u8,
    pub depth_1_iters: u8,
}

fuzz_target!(|input: MixedAgentInput| {
    // Ensure at least one CFR agent
    if !input.cfr_indices[0] && !input.cfr_indices[1] {
        return;
    }

    // Cap CFR iterations to avoid extremely slow fuzz runs
    // depth 0: 1-3 waves, depth 1: 1-2 waves, depth 2+: 1 wave
    let depth_0 = (input.depth_0_iters % 3) as usize + 1;
    let depth_1 = (input.depth_1_iters % 2) as usize + 1;
    let iters_per_depth = vec![depth_0, depth_1, 1];

    let game_state = GameStateBuilder::new()
        .num_players_with_stack(2, 50.0)
        .blinds(2.0, 1.0)
        .build()
        .unwrap();

    // Create shared CFR context (single tree for all players)
    let cfr_state = CFRState::new(game_state.clone());
    let traversal_set = TraversalSet::new(game_state.num_players);

    let agents: Vec<Box<dyn Agent>> = vec![
        create_agent(
            0,
            input.cfr_indices[0],
            input.cfr_variants[0],
            &input.player0_actions,
            &cfr_state,
            &traversal_set,
            &iters_per_depth,
        ),
        create_agent(
            1,
            input.cfr_indices[1],
            input.cfr_variants[1],
            &input.player1_actions,
            &cfr_state,
            &traversal_set,
            &iters_per_depth,
        ),
    ];

    let mut sim = HoldemSimulationBuilder::default()
        .game_state(game_state)
        .agents(agents)
        .cfr_context(cfr_state, traversal_set, true)
        .build_with_rng(StdRng::seed_from_u64(input.seed))
        .unwrap();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(sim.run());

    // Verify the game completed successfully
    assert_eq!(Round::Complete, sim.game_state.round);
    assert_valid_game_state(&sim.game_state);
});

/// Create an agent based on whether it should be CFR or non-CFR.
/// Uses either SimpleActionGenerator or ConfigurableActionGenerator based on variant.
fn create_agent(
    player_idx: usize,
    is_cfr: bool,
    cfr_variant: CfrVariant,
    actions: &[AgentAction],
    cfr_state: &CFRState,
    traversal_set: &TraversalSet,
    iters_per_depth: &[usize],
) -> Box<dyn Agent> {
    if is_cfr {
        // Reproduce the per-depth wave counts via a `PerDepth` of
        // `IterationCount` budgets. Depths past the vec fall back to a
        // single-iteration budget, which stops recursion past the schedule.
        let by_depth: Vec<Arc<dyn Budget>> = iters_per_depth
            .iter()
            .map(|&h| Arc::new(IterationCount::new(h as u64)) as Arc<dyn Budget>)
            .collect();
        let budget: Arc<dyn Budget> =
            Arc::new(PerDepth::new(by_depth, Arc::new(IterationCount::new(1))));
        match cfr_variant {
            CfrVariant::Simple => Box::new(
                CFRAgentBuilder::<SimpleActionGenerator>::new()
                    .name(format!("CFRAgent-{player_idx}"))
                    .player_idx(player_idx)
                    .cfr_state(cfr_state.clone())
                    .traversal_set(traversal_set.clone())
                    .budget(budget)
                    .action_gen_config(())
                    .build(),
            ),
            CfrVariant::Configurable => {
                // Use a reasonable default config for fuzzing
                let config = ConfigurableActionConfig {
                    default: RoundActionConfig {
                        call_enabled: true,
                        raise_mult: vec![1.0, 2.0],
                        pot_mult: vec![0.5, 1.0],
                        setup_shove: false,
                        all_in: true,
                    },
                    preflop: None,
                    flop: None,
                    turn: None,
                    river: None,
                };
                Box::new(
                    CFRAgentBuilder::<ConfigurableActionGenerator>::new()
                        .name(format!("CFRConfigurableAgent-{player_idx}"))
                        .player_idx(player_idx)
                        .cfr_state(cfr_state.clone())
                        .traversal_set(traversal_set.clone())
                        .budget(budget)
                        .action_gen_config(config)
                        .build(),
                )
            }
        }
    } else if actions.is_empty() {
        // Use a deterministic agent when no replay actions provided
        if player_idx.is_multiple_of(2) {
            Box::new(CallingAgent::new(format!("CallingAgent-{player_idx}")))
        } else {
            Box::new(FoldingAgent::new(format!("FoldingAgent-{player_idx}")))
        }
    } else {
        Box::new(VecReplayAgent::new(
            format!("ReplayAgent-{player_idx}"),
            actions.to_vec(),
        ))
    }
}
