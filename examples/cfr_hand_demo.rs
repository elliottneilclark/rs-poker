mod common;

use rs_poker::arena::{
    Agent, Historian, HoldemSimulationBuilder,
    cfr::{
        BasicCFRActionGenerator, CFRAgent, ExportFormat, PerRoundFixedGameStateIteratorGen,
        export_cfr_state,
    },
    historian::DirectoryHistorian,
};

fn run_simulation(num_agents: usize, export_path: Option<std::path::PathBuf>) {
    // Create a game state with the specified number of agents
    let stacks = vec![500.0; num_agents];
    let game_state =
        rs_poker::arena::game_state::GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);

    // Each CFR agent creates its own state store with states for all players
    let agents: Vec<_> = (0..num_agents)
        .map(|idx| {
            Box::new(
                // Create a CFR Agent for each player
                // Each agent has its own state store that tracks all players
                // Please note that the default iteration count is way too small
                // for a real CFR simulation, but it is enough to demonstrate
                // the CFR state tree and the export of the game history
                CFRAgent::<BasicCFRActionGenerator, PerRoundFixedGameStateIteratorGen>::new(
                    format!("CFRAgent-demo-{idx}"),
                    idx,
                    game_state.clone(),
                    PerRoundFixedGameStateIteratorGen::default(),
                    (),
                ),
            )
        })
        .collect();

    // Clone CFR states before moving agents into the simulation.
    // CFRState uses Arc internally, so clones share the same underlying data.
    let cfr_states: Vec<_> = agents.iter().map(|a| a.cfr_state().clone()).collect();

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
        .build()
        .unwrap();

    let mut rand = rand::rng();
    sim.run(&mut rand);

    // Export CFR states if an export path was provided
    if let Some(path) = export_path {
        for (i, cfr_state) in cfr_states.iter().enumerate() {
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
    common::init_tracing_from_env();

    // The first argument is the number of agents
    let num_agents = std::env::args()
        .nth(1)
        .expect("number of agents")
        .parse::<usize>()
        .expect("invalid number of agents");

    // The second argument is an optional path to where we should store
    // The JSON game history and the CFR state tree diagram
    // If no path is provided, no files will be created
    let export_path = std::env::args().nth(2).map(std::path::PathBuf::from);

    run_simulation(num_agents, export_path.clone());
}
