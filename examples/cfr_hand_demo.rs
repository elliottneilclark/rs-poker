mod common;

use std::sync::Arc;

use clap::Parser;
use rand::rngs::SmallRng;
use rs_poker::arena::{
    Agent, Historian, HoldemSimulationBuilder,
    cfr::{
        CFRAgentBuilder, CFRState, ConfigurableActionConfig, ConfigurableActionGenerator,
        DepthBasedIteratorGen, DepthBasedIteratorGenConfig, ExportFormat, RoundActionConfig,
        TraversalSet, export_cfr_state,
    },
    historian::DirectoryHistorian,
};

#[derive(Parser)]
struct Args {
    /// Number of agents in the simulation
    num_agents: usize,

    /// Optional path to export game history and CFR state tree diagrams
    export_path: Option<std::path::PathBuf>,

    /// Number of threads for parallel action exploration.
    /// When omitted, agents run sequentially (existing behavior).
    #[arg(long)]
    parallel: Option<usize>,
}

fn run_simulation(
    num_agents: usize,
    export_path: Option<std::path::PathBuf>,
    thread_pool: Option<Arc<rayon::ThreadPool>>,
) {
    // Create a game state with the specified number of agents
    let game_state = rs_poker::arena::game_state::GameStateBuilder::new()
        .num_players_with_stack(num_agents, 500.0)
        .blinds(10.0, 5.0)
        .build()
        .unwrap();

    // All CFR agents share the same CFR states and traversal set for shared learning
    let cfr_states: Vec<CFRState> = (0..num_agents)
        .map(|_| CFRState::new(game_state.clone()))
        .collect();
    let traversal_set = TraversalSet::new(num_agents);
    let iter_config = DepthBasedIteratorGenConfig::default();

    // Configure action generator with 4x raise, half pot, and full pot
    let action_config = ConfigurableActionConfig {
        default: RoundActionConfig {
            call_enabled: true,
            raise_mult: vec![4.0],    // 4x min raise
            pot_mult: vec![0.5, 1.0], // Half pot and full pot
            setup_shove: false,
            all_in: true,
        },
        preflop: Some(RoundActionConfig {
            call_enabled: true,
            raise_mult: vec![3.0], // 3x min raise
            pot_mult: vec![],
            setup_shove: false,
            all_in: false,
        }),
        flop: None,
        turn: None,
        river: None,
    };

    let agents: Vec<_> = (0..num_agents)
        .map(|idx| {
            let mut builder = CFRAgentBuilder::<
                ConfigurableActionGenerator,
                DepthBasedIteratorGen,
                SmallRng,
            >::new()
            .name(format!("CFRAgent-demo-{idx}"))
            .player_idx(idx)
            .cfr_states(cfr_states.clone())
            .traversal_set(traversal_set.clone())
            .gamestate_iterator_gen_config(iter_config.clone())
            .action_gen_config(action_config.clone());

            if let Some(pool) = &thread_pool {
                builder = builder.thread_pool(pool.clone());
            }

            Box::new(builder.build())
        })
        .collect();

    // Clone CFR states before moving agents into the simulation.
    // CFRState uses Arc internally, so clones share the same underlying data.
    let export_cfr_states: Vec<_> = agents.iter().map(|a| a.cfr_state().clone()).collect();

    let mut historians: Vec<Box<dyn Historian>> = Vec::new();

    if let Some(path) = export_path.clone() {
        // If a path is provided, we create a directory historian
        // to store the game history
        let dir_hist = DirectoryHistorian::new(path);

        // We don't need to create the dir_hist because the
        // DirectoryHistorian already does that on first action to record
        historians.push(Box::new(dir_hist));
    }

    let dyn_agents = agents.into_iter().map(|a| a as Box<dyn Agent>).collect();

    let mut sim = HoldemSimulationBuilder::default()
        .game_state(game_state)
        .agents(dyn_agents)
        .historians(historians)
        .cfr_context(cfr_states, traversal_set, true)
        .build()
        .unwrap();

    let mut rand = rand::rng();
    let start = std::time::Instant::now();
    sim.run(&mut rand);
    let elapsed = start.elapsed();
    println!("Simulation completed in {elapsed:.2?}");

    // Export CFR states if an export path was provided
    if let Some(path) = export_path {
        for (i, cfr_state) in export_cfr_states.iter().enumerate() {
            export_cfr_state(
                cfr_state,
                path.join(format!("cfr_state_{i}.svg")).as_path(),
                ExportFormat::Svg,
            )
            .expect("failed to export cfr state");
        }
    }
}

// Since simulation runs hot and heavy anything we can do to reduce the
// Allocation overhead is a good thing.
//
#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() {
    // Only initialize the tracing subscriber when RUST_LOG is explicitly set.
    // When no subscriber exists, tracing macros are a complete no-op, avoiding
    // span creation/filtering overhead in the millions of CFR sub-simulations.
    if std::env::var_os("RUST_LOG").is_some() {
        common::init_tracing_from_env();
    }

    let args = Args::parse();

    let thread_pool = args.parallel.map(|num_threads| {
        Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .expect("failed to create thread pool"),
        )
    });

    run_simulation(args.num_agents, args.export_path, thread_pool);
}
