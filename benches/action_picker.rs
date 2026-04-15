use criterion::{Criterion, criterion_group, criterion_main};
use little_sorry::{PcfrPlusRegretMatcher, RegretMinimizer};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rs_poker::arena::GameStateBuilder;
use rs_poker::arena::action::AgentAction;
use rs_poker::arena::cfr::{ActionIndexMapper, ActionIndexMapperConfig, ActionPicker};

fn make_state_and_mapper() -> (rs_poker::arena::GameState, ActionIndexMapper) {
    let gs = GameStateBuilder::new()
        .num_players_with_stack(2, 10_000.0)
        .blinds(100.0, 50.0)
        .build()
        .unwrap();
    let mapper = ActionIndexMapper::new(ActionIndexMapperConfig::new(100.0, 10_000.0));
    (gs, mapper)
}

fn make_trained_matcher(
    mapper: &ActionIndexMapper,
    gs: &rs_poker::arena::GameState,
) -> PcfrPlusRegretMatcher {
    let mut m = PcfrPlusRegretMatcher::new(52);
    let mut rewards = vec![0.0f32; 52];
    rewards[0] = 10.0;
    rewards[1] = 30.0;
    rewards[mapper.action_to_idx(&AgentAction::Bet(300.0), gs)] = 20.0;
    rewards[mapper.action_to_idx(&AgentAction::Bet(600.0), gs)] = 15.0;
    rewards[mapper.action_to_idx(&AgentAction::Bet(1200.0), gs)] = 5.0;
    rewards[51] = 2.0;
    for _ in 0..16 {
        m.update_regret(&rewards);
    }
    m
}

fn bench_pick_action(c: &mut Criterion) {
    let (gs, mapper) = make_state_and_mapper();
    let matcher = make_trained_matcher(&mapper, &gs);

    // Typical CFR action set: fold, call, several bets, all-in
    let actions = vec![
        AgentAction::Fold,
        AgentAction::Bet(100.0), // call
        AgentAction::Bet(300.0),
        AgentAction::Bet(600.0),
        AgentAction::Bet(1200.0),
        AgentAction::AllIn,
    ];

    c.bench_function("pick_action_typical", |b| {
        let mut rng = StdRng::seed_from_u64(42);
        b.iter(|| {
            let picker = ActionPicker::new(&mapper, &actions, Some(&matcher), &gs);
            std::hint::black_box(picker.pick_action(&mut rng))
        })
    });

    // Collision-heavy set: many bet sizes that will quantise together
    let collision_actions = vec![
        AgentAction::Fold,
        AgentAction::Bet(100.0),
        AgentAction::Bet(200.0),
        AgentAction::Bet(205.0),
        AgentAction::Bet(210.0),
        AgentAction::Bet(500.0),
        AgentAction::Bet(510.0),
        AgentAction::Bet(520.0),
        AgentAction::AllIn,
    ];

    c.bench_function("pick_action_collisions", |b| {
        let mut rng = StdRng::seed_from_u64(42);
        b.iter(|| {
            let picker = ActionPicker::new(&mapper, &collision_actions, Some(&matcher), &gs);
            std::hint::black_box(picker.pick_action(&mut rng))
        })
    });

    c.bench_function("pick_best_action_typical", |b| {
        b.iter(|| {
            let picker = ActionPicker::new(&mapper, &actions, Some(&matcher), &gs);
            std::hint::black_box(picker.pick_best_action())
        })
    });

    // No regret matcher (uniform random path)
    c.bench_function("pick_action_uniform", |b| {
        let mut rng = StdRng::seed_from_u64(42);
        b.iter(|| {
            let picker = ActionPicker::new(&mapper, &actions, None, &gs);
            std::hint::black_box(picker.pick_action(&mut rng))
        })
    });
}

criterion_group!(benches, bench_pick_action);
criterion_main!(benches);
