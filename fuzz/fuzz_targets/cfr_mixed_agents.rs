#![no_main]

extern crate arbitrary;
extern crate libfuzzer_sys;
extern crate rand;
extern crate rs_poker;

use libfuzzer_sys::fuzz_target;
use rand::{rngs::StdRng, SeedableRng};
use rs_poker::arena::{
    action::AgentAction,
    agent::{CallingAgent, FoldingAgent, VecReplayAgent},
    cfr::{
        CFRAgentBuilder, ConfigurableActionConfig, ConfigurableActionGenerator,
        DepthBasedIteratorGen, DepthBasedIteratorGenConfig, RoundActionConfig, SimpleActionConfig,
        SimpleActionGenerator, StateStore,
    },
    game_state::Round,
    test_util::assert_valid_game_state,
    Agent, GameState, HoldemSimulationBuilder,
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
    /// Depth-based iteration counts (capped to avoid slow tests)
    pub depth_0_hands: u8,
    pub depth_1_hands: u8,
}

fuzz_target!(|input: MixedAgentInput| {
    // Ensure at least one CFR agent
    if !input.cfr_indices[0] && !input.cfr_indices[1] {
        return;
    }

    // Cap CFR iterations to avoid extremely slow fuzz runs
    // depth 0: 1-3 hands, depth 1+: 1 hand
    let depth_0 = (input.depth_0_hands % 3) as usize + 1;
    let depth_1 = (input.depth_1_hands % 2) as usize + 1;
    let depth_hands = vec![depth_0, depth_1, 1];

    let game_state = GameStateBuilder::new()
        .num_players_with_stack(2, 50.0)
        .blinds(2.0, 1.0)
        .build()
        .unwrap();

    let agents: Vec<Box<dyn Agent>> = vec![
        create_agent(
            0,
            input.cfr_indices[0],
            input.cfr_variants[0],
            &input.player0_actions,
            &game_state,
            &depth_hands,
        ),
        create_agent(
            1,
            input.cfr_indices[1],
            input.cfr_variants[1],
            &input.player1_actions,
            &game_state,
            &depth_hands,
        ),
    ];

    let mut rng = StdRng::seed_from_u64(input.seed);
    let mut sim = HoldemSimulationBuilder::default()
        .game_state(game_state)
        .agents(agents)
        .build()
        .unwrap();

    sim.run(&mut rng);

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
    game_state: &GameState,
    depth_hands: &[usize],
) -> Box<dyn Agent> {
    if is_cfr {
        let state_store = StateStore::new_for_game(game_state.clone());
        let iter_config = DepthBasedIteratorGenConfig::new(depth_hands.to_vec());
        match cfr_variant {
            CfrVariant::Simple => Box::new(
                CFRAgentBuilder::<SimpleActionGenerator, DepthBasedIteratorGen>::new()
                    .name(format!("CFRAgent-{player_idx}"))
                    .player_idx(player_idx)
                    .state_store(state_store)
                    .gamestate_iterator_gen_config(iter_config)
                    .action_gen_config(SimpleActionConfig::default())
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
                    CFRAgentBuilder::<ConfigurableActionGenerator, DepthBasedIteratorGen>::new()
                        .name(format!("CFRConfigurableAgent-{player_idx}"))
                        .player_idx(player_idx)
                        .state_store(state_store)
                        .gamestate_iterator_gen_config(iter_config)
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
