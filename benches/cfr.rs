use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rs_poker::arena::agent::ConfigAgentBuilder;
use rs_poker::arena::cfr::{
    BudgetConfig, BudgetItem, CFRAgentBuilder, CFRState, ConfigurableActionConfig,
    ConfigurableActionGenerator, TraversalSet,
};
use rs_poker::arena::hand_estimator::KnownHandsEstimator;
use rs_poker::arena::{
    Agent, GameLog, GameState, GameStateBuilder, HandDistributionEstimator,
    HoldemSimulationBuilder, OpponentRanges,
};

const STARTING_STACK: f32 = 100_000.0;
const ANTE: f32 = 50.0;
const SMALL_BLIND: f32 = 250.0;
const BIG_BLIND: f32 = 500.0;
const BENCH_SEED: u64 = 0xDEAD_BEEF;

/// CFR configurable agent config matching examples/configs/cfr_configurable.json.
fn cfr_configurable_json(num_hands: usize) -> String {
    format!(
        r#"{{
      "type": "cfr_configurable",
      "name": "CFR-Configurable",
      "exploration": {{
        "budget": [
          {{ "type": "per_depth_iterations", "counts": [{num_hands}, 5, 1] }}
        ]
      }},
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
    }}"#
    )
}

/// Build a 2-player CFR simulation.
///
/// The simulation's own dealing RNG is seeded via `build_with_rng`. CFR agents
/// draw from a thread-local RNG (no per-agent seed), so the exploration tree is
/// not byte-identical across runs; the count-bound `per_depth_iterations`
/// budget (with `act_deadline_ms: null`) keeps the workload size stable so
/// wall-clock deltas reflect code changes rather than deadline truncation.
fn build_sim(num_hands: usize) -> rs_poker::arena::HoldemSimulation {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let game_state = GameStateBuilder::new()
        .num_players_with_stack(2, STARTING_STACK)
        .blinds(BIG_BLIND, SMALL_BLIND)
        .ante(ANTE)
        .build()
        .unwrap();

    let cfr_state = CFRState::new(game_state.clone());
    let traversal_set = TraversalSet::new(2);
    let builder = ConfigAgentBuilder::from_json(&cfr_configurable_json(num_hands))
        .expect("Failed to parse CFR config");

    let agents: Vec<Box<dyn Agent>> = (0..2)
        .map(|idx| {
            builder
                .clone()
                .player_idx(idx)
                .game_state(game_state.clone())
                .cfr_context(cfr_state.clone(), traversal_set.clone())
                .build()
        })
        .collect();

    HoldemSimulationBuilder::default()
        .game_state(game_state)
        .agents(agents)
        .cfr_context(cfr_state, traversal_set, true)
        .build_with_rng(StdRng::seed_from_u64(BENCH_SEED))
        .unwrap()
}

/// Run a built sim to completion, returning the (now-finished) sim so criterion
/// drops it — and the large shared CFR tree it owns — *outside* the timed
/// region (`iter_batched` drops outputs untimed).
async fn run_sim(mut sim: rs_poker::arena::HoldemSimulation) -> rs_poker::arena::HoldemSimulation {
    sim.run().await;
    sim
}

/// Measure node count once (outside timing) for a given workload so throughput
/// can be reported. CFR agents use a thread-local RNG, so the count is
/// approximate (used only to normalize Criterion's throughput estimate).
fn node_count_for(num_hands: usize) -> u64 {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let game_state = GameStateBuilder::new()
        .num_players_with_stack(2, STARTING_STACK)
        .blinds(BIG_BLIND, SMALL_BLIND)
        .ante(ANTE)
        .build()
        .unwrap();
    let cfr_state = CFRState::new(game_state.clone());
    let traversal_set = TraversalSet::new(2);
    let builder = ConfigAgentBuilder::from_json(&cfr_configurable_json(num_hands))
        .expect("Failed to parse CFR config");
    let agents: Vec<Box<dyn Agent>> = (0..2)
        .map(|idx| {
            builder
                .clone()
                .player_idx(idx)
                .game_state(game_state.clone())
                .cfr_context(cfr_state.clone(), traversal_set.clone())
                .build()
        })
        .collect();
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    let mut sim = HoldemSimulationBuilder::default()
        .game_state(game_state)
        .agents(agents)
        .cfr_context(cfr_state.clone(), traversal_set, true)
        .build_with_rng(StdRng::seed_from_u64(BENCH_SEED))
        .unwrap();
    rt.block_on(async { sim.run().await });
    cfr_state.node_count() as u64
}

/// Single-threaded throughput. Isolates the per-node cost of the exploration
/// engine (async state-machine overhead, allocation, hashing) with zero
/// cross-thread lock contention or scheduler noise. This is the cleanest
/// signal for "how expensive is one node of work".
fn bench_cfr_single_thread(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("cfr_single_thread");

    for num_hands in [5, 15] {
        let nodes = node_count_for(num_hands);
        group.throughput(Throughput::Elements(nodes));
        group.bench_with_input(
            BenchmarkId::new("num_hands", num_hands),
            &num_hands,
            |b, &num_hands| {
                b.iter_batched(
                    || build_sim(num_hands),
                    |sim| rt.block_on(run_sim(sim)),
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

/// Multi-threaded throughput. Measures parallel scaling of the exploration
/// engine. Node counts can vary slightly run-to-run because spawn/update
/// ordering affects regret-based pruning, so this group is noisier than the
/// single-threaded one; use it for scaling, not fine-grained deltas.
fn bench_cfr_multi_thread(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .thread_stack_size(48 * 1024 * 1024)
        .enable_all()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("cfr_multi_thread");

    for num_hands in [15, 20] {
        let nodes = node_count_for(num_hands);
        group.throughput(Throughput::Elements(nodes));
        group.bench_with_input(
            BenchmarkId::new("num_hands", num_hands),
            &num_hands,
            |b, &num_hands| {
                b.iter_batched(
                    || build_sim(num_hands),
                    |sim| rt.block_on(run_sim(sim)),
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

/// A `needs_history` stub estimator. Sampling semantics are identical to
/// `KnownHandsEstimator`, but `needs_history()` returns `true`, which activates
/// the copy-on-write `HandLog` path (freeze at depth 0, copy-tail descent,
/// per-sim `HandLogHistorian`). Because sampling is unchanged, this isolates the
/// logging overhead of the history path versus the no-history default groups.
#[derive(Default)]
struct HistoryNeedingKnown;

#[async_trait::async_trait]
impl HandDistributionEstimator for HistoryNeedingKnown {
    async fn estimate(
        &self,
        game_state: &GameState,
        _history: Option<&GameLog<'_>>,
    ) -> OpponentRanges {
        KnownHandsEstimator.estimate(game_state, None).await
    }

    fn needs_history(&self) -> bool {
        true
    }
}

/// Build a 2-player CFR simulation whose agents use the `HistoryNeedingKnown`
/// estimator, exercising the `HandLog` history path.
///
/// Agents are built DIRECTLY (not via JSON) so a custom estimator can be
/// injected. The budget matches `build_sim`'s `per_depth_iterations` workload
/// (`[num_hands, 5, 1]`) so old-vs-new and history-vs-default comparisons use
/// the same exploration size.
fn build_sim_with_history(num_hands: usize) -> rs_poker::arena::HoldemSimulation {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let game_state = GameStateBuilder::new()
        .num_players_with_stack(2, STARTING_STACK)
        .blinds(BIG_BLIND, SMALL_BLIND)
        .ante(ANTE)
        .build()
        .unwrap();

    let cfr_state = CFRState::new(game_state.clone());
    let traversal_set = TraversalSet::new(2);

    let budget = BudgetConfig(vec![BudgetItem::PerDepthIterations {
        counts: vec![num_hands as u64, 5, 1],
        fallback: 1,
    }])
    .build();

    let agents: Vec<Box<dyn Agent>> = (0..2)
        .map(|idx| {
            Box::new(
                CFRAgentBuilder::<ConfigurableActionGenerator>::new()
                    .name(format!("CFR-History-{idx}"))
                    .player_idx(idx)
                    .cfr_state(cfr_state.clone())
                    .traversal_set(traversal_set.clone())
                    .budget(budget.clone())
                    .estimator(Arc::new(HistoryNeedingKnown))
                    .action_gen_config(ConfigurableActionConfig::default())
                    .build(),
            ) as Box<dyn Agent>
        })
        .collect();

    HoldemSimulationBuilder::default()
        .game_state(game_state)
        .agents(agents)
        .cfr_context(cfr_state, traversal_set, true)
        .build_with_rng(StdRng::seed_from_u64(BENCH_SEED))
        .unwrap()
}

/// History-path latency. Same single-threaded setup as
/// `bench_cfr_single_thread`, but every agent's estimator returns
/// `needs_history() == true`, so the `HandLog` recording path runs. Unlike the
/// default groups this measures raw per-game wall time (no `Throughput`
/// normalization), since the point is the absolute logging overhead. Compare
/// against the default groups (same budget) to read that overhead.
fn bench_cfr_history(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("cfr_history");

    for num_hands in [5, 15] {
        group.bench_with_input(
            BenchmarkId::new("num_hands", num_hands),
            &num_hands,
            |b, &num_hands| {
                b.iter_batched(
                    || build_sim_with_history(num_hands),
                    |sim| rt.block_on(run_sim(sim)),
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_cfr_single_thread,
    bench_cfr_multi_thread,
    bench_cfr_history
);
criterion_main!(benches);
