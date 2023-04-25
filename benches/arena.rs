use criterion::BenchmarkId;
use criterion::Criterion;

use criterion::criterion_group;
use criterion::criterion_main;
use rs_poker::arena::agent::RandomAgent;
use rs_poker::arena::Agent;
use rs_poker::arena::GameState;
use rs_poker::arena::HoldemSimulation;
use rs_poker::core::{Deck, FlatDeck};

const STARTING_STACK: i32 = 100_000;
const SMALL_BLIND: i32 = 250;
const BIG_BLIND: i32 = 500;

fn run_one_arena(num_players: usize) -> GameState {
    let mut deck: FlatDeck = Deck::default().into();
    deck.shuffle();

    let stacks = vec![STARTING_STACK; num_players];
    let game_state = GameState::new(stacks, BIG_BLIND, SMALL_BLIND, 0);
    let agents: Vec<Box<dyn Agent>> = (0..num_players)
        .map(|_| -> Box<dyn Agent> { Box::<RandomAgent>::default() })
        .collect();
    let mut sim = HoldemSimulation::new_with_agents_and_deck(game_state, deck, agents);
    sim.run();
    sim.game_state
}

fn bench_num_random_agent_players(c: &mut Criterion) {
    let mut group = c.benchmark_group("arena_random_agents");
    for ref num_players in 2..7 {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_players),
            num_players,
            |b, &num_players| {
                b.iter(|| run_one_arena(num_players));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_num_random_agent_players);
criterion_main!(benches);
