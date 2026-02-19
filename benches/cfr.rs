use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::rng;
use rs_poker::arena::agent::ConfigAgentBuilder;
use rs_poker::arena::cfr::{StateStore, TraversalSet};
use rs_poker::arena::{Agent, GameState, GameStateBuilder, HoldemSimulationBuilder};

const STARTING_STACK: f32 = 100_000.0;
const ANTE: f32 = 50.0;
const SMALL_BLIND: f32 = 250.0;
const BIG_BLIND: f32 = 500.0;

/// CFR configurable agent config matching examples/configs/cfr_configurable.json
const CFR_CONFIGURABLE_JSON: &str = r#"{
  "type": "cfr_configurable",
  "name": "CFR-Configurable",
  "num_hands": 15,
  "action_config": {
    "default": {
      "call_enabled": true,
      "raise_mult": [1.0, 3.0],
      "pot_mult": [0.5, 0.75, 1.0],
      "setup_shove": true,
      "all_in": true
    },
    "preflop": {
      "call_enabled": true,
      "raise_mult": [3.0, 4.0],
      "pot_mult": [],
      "setup_shove": false,
      "all_in": true
    }
  }
}"#;

fn run_cfr_configurable_arena(num_hands: usize) -> GameState {
    let json = format!(
        r#"{{
      "type": "cfr_configurable",
      "name": "CFR-Configurable",
      "num_hands": {},
      "action_config": {{
        "default": {{
          "call_enabled": true,
          "raise_mult": [1.0, 3.0],
          "pot_mult": [0.5, 0.75, 1.0],
          "setup_shove": true,
          "all_in": true
        }},
        "preflop": {{
          "call_enabled": true,
          "raise_mult": [3.0, 4.0],
          "pot_mult": [],
          "setup_shove": false,
          "all_in": true
        }}
      }}
    }}"#,
        num_hands
    );

    let game_state = GameStateBuilder::new()
        .num_players_with_stack(2, STARTING_STACK)
        .blinds(BIG_BLIND, SMALL_BLIND)
        .ante(ANTE)
        .build()
        .unwrap();

    let state_store = StateStore::new(game_state.clone());
    let traversal_set = TraversalSet::new(2);
    let builder = ConfigAgentBuilder::from_json(&json).expect("Failed to parse CFR config");

    let agents: Vec<Box<dyn Agent>> = (0..2)
        .map(|idx| {
            builder
                .clone()
                .player_idx(idx)
                .game_state(game_state.clone())
                .cfr_context(state_store.clone(), traversal_set.clone())
                .build()
        })
        .collect();

    let mut sim = HoldemSimulationBuilder::default()
        .game_state(game_state)
        .agents(agents)
        .cfr_context(state_store, traversal_set, true)
        .build()
        .unwrap();

    let mut rand = rng();
    sim.run(&mut rand);
    sim.game_state
}

fn run_cfr_configurable_arena_default() -> GameState {
    let game_state = GameStateBuilder::new()
        .num_players_with_stack(2, STARTING_STACK)
        .blinds(BIG_BLIND, SMALL_BLIND)
        .ante(ANTE)
        .build()
        .unwrap();

    let state_store = StateStore::new(game_state.clone());
    let traversal_set = TraversalSet::new(2);
    let builder =
        ConfigAgentBuilder::from_json(CFR_CONFIGURABLE_JSON).expect("Failed to parse CFR config");

    let agents: Vec<Box<dyn Agent>> = (0..2)
        .map(|idx| {
            builder
                .clone()
                .player_idx(idx)
                .game_state(game_state.clone())
                .cfr_context(state_store.clone(), traversal_set.clone())
                .build()
        })
        .collect();

    let mut sim = HoldemSimulationBuilder::default()
        .game_state(game_state)
        .agents(agents)
        .cfr_context(state_store, traversal_set, true)
        .build()
        .unwrap();

    let mut rand = rng();
    sim.run(&mut rand);
    sim.game_state
}

fn bench_cfr_configurable_agents(c: &mut Criterion) {
    let mut group = c.benchmark_group("cfr_configurable_agents");

    // Benchmark with varying num_hands
    for num_hands in [1, 5, 10, 15, 20] {
        group.bench_with_input(
            BenchmarkId::new("num_hands", num_hands),
            &num_hands,
            |b, &num_hands| {
                b.iter(|| run_cfr_configurable_arena(num_hands));
            },
        );
    }

    group.finish();
}

fn bench_cfr_configurable_default(c: &mut Criterion) {
    c.bench_function("cfr_configurable_default", |b| {
        b.iter(run_cfr_configurable_arena_default)
    });
}

criterion_group!(
    benches,
    bench_cfr_configurable_agents,
    bench_cfr_configurable_default
);
criterion_main!(benches);
