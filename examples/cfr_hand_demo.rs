use rs_poker::arena::{
    Agent, Historian, HoldemSimulationBuilder,
    cfr::{
        BasicCFRActionGenerator, CFRAgent, ExportFormat, PerRoundFixedGameStateIteratorGen,
        StateStore, export_cfr_state,
    },
    historian::DirectoryHistorian,
};

fn run_simulation(num_agents: usize, export_path: Option<std::path::PathBuf>) {
    // Create a game state with the specified number of agents
    let stacks = vec![500.0; num_agents];
    let game_state =
        rs_poker::arena::game_state::GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);

    let mut state_store = StateStore::new();

    let states = (0..num_agents)
        .map(|player_idx| state_store.new_state(game_state.clone(), player_idx))
        .collect::<Vec<_>>();

    let agents: Vec<_> = states
        .iter()
        .map(|(cfr_state, traversal_state)| {
            Box::new(
                // Create a CFR Agent for each player
                // They have their own CFR state and
                // and for now a fixed game state iterator
                // that will try a very few hands
                CFRAgent::<BasicCFRActionGenerator, PerRoundFixedGameStateIteratorGen>::new(
                    state_store.clone(),
                    cfr_state.clone(),
                    traversal_state.clone(),
                    // please note that this is way too small
                    // for a real CFR simulation, but it is
                    // enough to demonstrate the CFR state tree
                    // and the export of the game history
                    PerRoundFixedGameStateIteratorGen::default(),
                ),
            )
        })
        .collect();

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

    // If there's an export path then we want to export each of the states
    if let Some(path) = export_path.clone() {
        for (i, (cfr_state, _)) in states.iter().enumerate() {
            // Export the CFR state to JSON
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
