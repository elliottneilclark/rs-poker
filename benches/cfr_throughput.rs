//! CFR throughput benchmark — the production-representative companion to
//! `benches/cfr.rs`.
//!
//! Both are criterion benches over the same deterministic, seeded 2-player CFR
//! simulation (a given seed fully determines the exploration tree, since every
//! sub-agent's RNG derives from the parent via `R::from_rng`). Throughput is
//! reported in **nodes/sec** via `Throughput::Elements(node_count)`.
//!
//! What this one adds over `cfr.rs`:
//! - **jemalloc** global allocator, matching the production `rsp` binary
//!   (`src/bin/rsp/main.rs`). Measuring under the system allocator overstates
//!   allocation cost relative to production.
//! - **Env-configurable depth schedule** via `CFR_DEPTH` (e.g.
//!   `CFR_DEPTH=15,5,3,1`), to measure the cost of going deeper before
//!   fast-forward. Defaults to `15,5,1`.
//! - **A thread-count sweep** via `CFR_THREADS` (e.g. `CFR_THREADS=1,2,4,8,16`),
//!   so multi-thread scaling is itself a criterion benchmark. Defaults to the
//!   machine's available parallelism. NB: this is an 8-physical-core box; the
//!   16th "core" is a hyperthread and CPU-bound hand-eval barely uses it.
//!
//! A/B:        `cargo bench --bench cfr_throughput -- --save-baseline before`
//!             `cargo bench --bench cfr_throughput -- --baseline before`
//! Profiling:  `cargo bench --bench cfr_throughput -- --profile-time 20`
//!             (then `perf record` the bench binary, or run under perf directly)

// Match the production `rsp` binary's allocator so throughput numbers reflect
// what production sees (the system allocator overstates allocation cost).
#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rs_poker::arena::agent::ConfigAgentBuilder;
use rs_poker::arena::cfr::{CFRState, TraversalSet};
use rs_poker::arena::{Agent, GameStateBuilder, HoldemSimulation, HoldemSimulationBuilder};

const STARTING_STACK: f32 = 100_000.0;
const ANTE: f32 = 50.0;
const SMALL_BLIND: f32 = 250.0;
const BIG_BLIND: f32 = 500.0;
const BENCH_SEED: u64 = 0xDEAD_BEEF;

fn cfr_configurable_json(counts: &str) -> String {
    // The recursion depth is implied by the number of per-depth iteration counts,
    // so the workload matches the historical `[N, 5, 1]` schedule (recurse to that
    // depth). Supplying only a `per_depth_iterations` budget (no `deadline`
    // component) keeps the bench count-bound — no wall-clock truncation of the
    // explored tree.
    format!(
        r#"{{
      "type": "cfr_configurable",
      "name": "CFR-Configurable",
      "exploration": {{
        "budget": [
          {{ "type": "per_depth_iterations", "counts": [{counts}] }}
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

/// Build a 2-player CFR sim. The dealing RNG is seeded; CFR agents draw from a
/// thread-local RNG (no per-agent seed), so the tree is not byte-identical
/// across runs. The count-bound `per_depth_iterations` budget keeps the
/// workload size stable so wall-clock deltas reflect code changes.
fn build_sim(counts: &str, players: usize) -> (HoldemSimulation, CFRState) {
    let game_state = GameStateBuilder::new()
        .num_players_with_stack(players, STARTING_STACK)
        .blinds(BIG_BLIND, SMALL_BLIND)
        .ante(ANTE)
        .build()
        .unwrap();

    let cfr_state = CFRState::new(game_state.clone());
    let traversal_set = TraversalSet::new(players);
    let builder = ConfigAgentBuilder::from_json(&cfr_configurable_json(counts))
        .expect("Failed to parse CFR config");

    let agents: Vec<Box<dyn Agent>> = (0..players)
        .map(|idx: usize| {
            builder
                .clone()
                .player_idx(idx)
                .game_state(game_state.clone())
                .cfr_context(cfr_state.clone(), traversal_set.clone())
                .build()
        })
        .collect();

    let sim = HoldemSimulationBuilder::default()
        .game_state(game_state)
        .agents(agents)
        .cfr_context(cfr_state.clone(), traversal_set, true)
        .build_with_rng(StdRng::seed_from_u64(BENCH_SEED))
        .unwrap();
    (sim, cfr_state)
}

/// Run a built sim, returning it so criterion drops the (large) shared tree
/// *outside* the timed region (`iter_batched` drops outputs untimed).
async fn run_sim(mut sim: HoldemSimulation) -> HoldemSimulation {
    sim.run().await;
    sim
}

/// Node count for the workload, measured once (untimed) so throughput is
/// reported in nodes/sec. CFR agents use a thread-local RNG, so the count is
/// approximate (used only to normalize Criterion's throughput estimate).
fn node_count_for(counts: &str, players: usize) -> u64 {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let (mut sim, cfr_state) = build_sim(counts, players);
    rt.block_on(async { sim.run().await });
    cfr_state.node_count() as u64
}

/// Parse a comma-separated env list of positive usizes (e.g.
/// `CFR_THREADS=1,2,4,8`), falling back to `default` when unset.
fn env_usize_list(key: &str, default: Vec<usize>) -> Vec<usize> {
    let parsed: Vec<usize> = match std::env::var(key) {
        Ok(v) => v.split(',').filter_map(|s| s.trim().parse().ok()).collect(),
        Err(_) => default,
    };
    parsed.into_iter().filter(|&n| n > 0).collect()
}

fn bench_cfr_throughput(c: &mut Criterion) {
    // Per-depth iteration counts; default mirrors the historical `[N, 5, 1]`.
    let counts = std::env::var("CFR_DEPTH").unwrap_or_else(|_| "15, 5, 1".to_string());
    // Thread counts to sweep on the multi-thread runtime.
    let default_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let thread_counts = env_usize_list("CFR_THREADS", vec![default_threads]);

    let nodes = node_count_for(&counts, 2);

    let mut group = c.benchmark_group("cfr_throughput");
    group.throughput(Throughput::Elements(nodes));
    // Deep schedules make each iteration expensive; keep the sample count low
    // so a run stays tractable. Override measurement/warmup via the CLI
    // (`--measurement-time`, `--warm-up-time`).
    group.sample_size(10);

    // Single-threaded: cleanest per-node signal (no scheduler/contention noise).
    {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        group.bench_function("single_thread", |b| {
            b.iter_batched(
                || build_sim(&counts, 2),
                |(sim, _state)| rt.block_on(run_sim(sim)),
                criterion::BatchSize::SmallInput,
            );
        });
    }

    // Multi-threaded scaling sweep.
    for &threads in &thread_counts {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(threads)
            .thread_stack_size(48 * 1024 * 1024)
            .enable_all()
            .build()
            .unwrap();
        group.bench_with_input(
            BenchmarkId::new("multi_thread", threads),
            &threads,
            |b, _| {
                b.iter_batched(
                    || build_sim(&counts, 2),
                    |(sim, _state)| rt.block_on(run_sim(sim)),
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

/// Middle-ground multi-threaded throughput: a **4-handed**, count-bound CFR sim.
///
/// Heads-up (the `bench_cfr_throughput` workload) barely fans out, so it
/// under-exercises the scheduler that dominates production cost. Four-handed
/// fans out enough to actually load the worker threads, without a full table's
/// memory. The worker-thread counts are a fixed, hardcoded sweep (no env vars)
/// so multi-thread scaling is a first-class, reproducible target. Count-bound
/// (`act_deadline_ms: null`, via `cfr_configurable_json`) keeps the workload
/// fixed so wall deltas reflect code changes rather than a wall-clock budget.
fn bench_cfr_4handed_threads(c: &mut Criterion) {
    // A 4-handed tree fans out much faster than heads-up, so use a shallower
    // per-depth schedule than the heads-up bench to keep the count-bound
    // workload (and its peak memory) tractable while still building a
    // multi-thousand-node tree.
    const COUNTS: &str = "8, 3, 1";
    const PLAYERS: usize = 4;

    let nodes = node_count_for(COUNTS, PLAYERS);

    let mut group = c.benchmark_group("cfr_4handed_threads");
    group.throughput(Throughput::Elements(nodes));
    group.sample_size(10);

    for &threads in &[4usize, 8, 12, 16] {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(threads)
            .thread_stack_size(48 * 1024 * 1024)
            .enable_all()
            .build()
            .unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(threads), &threads, |b, _| {
            b.iter_batched(
                || build_sim(COUNTS, PLAYERS),
                |(sim, _state)| rt.block_on(run_sim(sim)),
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_cfr_throughput, bench_cfr_4handed_threads);
criterion_main!(benches);
