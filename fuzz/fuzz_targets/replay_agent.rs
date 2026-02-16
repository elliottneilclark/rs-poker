#![no_main]

extern crate approx;
extern crate arbitrary;
extern crate libfuzzer_sys;
extern crate rand;
extern crate rs_poker;

use approx::assert_relative_ne;
use rand::{rngs::StdRng, SeedableRng};

use rs_poker::arena::{
    action::AgentAction,
    agent::VecReplayAgent,
    game_state::Round,
    historian::{self, OpenHandHistoryVecHistorian},
    test_util::{assert_valid_game_state, assert_valid_history, assert_valid_round_data},
    Agent, GameState, HoldemSimulation, HoldemSimulationBuilder,
};
use rs_poker::open_hand_history::{
    assert_open_hand_history_matches_game_state,
    assert_valid_open_hand_history,
};

use libfuzzer_sys::fuzz_target;

#[derive(Debug, Clone, arbitrary::Arbitrary)]
struct Input {
    pub dealer_actions: Vec<AgentAction>,
    pub sb_actions: Vec<AgentAction>,
    pub seed: u64,
}

fuzz_target!(|input: Input| {
    let game_state = GameStateBuilder::new()
        .num_players_with_stack(2, 50.0)
        .blinds(2.0, 1.0)
        .build()
        .unwrap();
    let agents: Vec<Box<dyn Agent>> = vec![
        Box::<VecReplayAgent>::new(VecReplayAgent::new("replay-agent-dealer", input.dealer_actions)),
        Box::<VecReplayAgent>::new(VecReplayAgent::new("replay-agent-sb", input.sb_actions)),
    ];

    let vec_historian = Box::<historian::VecHistorian>::new(historian::VecHistorian::new());
    let open_hand_hist = Box::new(OpenHandHistoryVecHistorian::new());

    let storage = vec_historian.get_storage();
    let hand_storage = open_hand_hist.get_storage();

    let historians: Vec<Box<dyn historian::Historian>> = vec![vec_historian, open_hand_hist];
    let mut rng = StdRng::seed_from_u64(input.seed);
    let mut sim: HoldemSimulation = HoldemSimulationBuilder::default()
        .game_state(game_state)
        .agents(agents)
        .historians(historians)
        .build()
        .unwrap();
    sim.run(&mut rng);

    assert_eq!(Round::Complete, sim.game_state.round);
    assert_relative_ne!(0.0_f32, sim.game_state.player_bet.iter().sum());

    assert_valid_round_data(&sim.game_state.round_data);
    assert_valid_game_state(&sim.game_state);

    assert_valid_history(&storage.borrow());

    let hands = hand_storage.borrow();
    assert!(!hands.is_empty());
    for hand in hands.iter() {
        assert_valid_open_hand_history(hand);
        assert_open_hand_history_matches_game_state(hand, &sim.game_state);
    }
});
