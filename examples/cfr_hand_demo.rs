use rs_poker::arena::{
    Agent, Historian, HoldemSimulationBuilder,
    cfr::{
        BasicCFRActionGenerator, CFRAgent, ExportFormat, FixedGameStateIteratorGen,
        export_cfr_state,
    },
    historian::DirectoryHistorian,
};

fn run_simulation(num_agents: usize, export_path: Option<std::path::PathBuf>) {
    // Create a game state with the specified number of agents
    let stacks = vec![500.0; num_agents];
    let game_state =
        rs_poker::arena::game_state::GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);

    let cfr_states = (0..num_agents)
        .map(|_| rs_poker::arena::cfr::CFRState::new(game_state.clone()))
        .collect::<Vec<_>>();

    let agents: Vec<_> = cfr_states
        .iter()
        .enumerate()
        .map(|(i, s)| {
            Box::new(
                // Create a CFR Agent for each player
                // They have their own CFR state and
                // and for now a fixed game state iterator
                // that will try a very few hands
                CFRAgent::<BasicCFRActionGenerator, FixedGameStateIteratorGen>::new(
                    s.clone(),
                    // please note that this is way too small
                    // for a real CFR simulation, but it is
                    // enough to demonstrate the CFR state tree
                    // and the export of the game history
                    FixedGameStateIteratorGen::new(3),
                    i,
                ),
            )
        })
        .collect();

    let mut historians: Vec<Box<dyn Historian>> = agents
        .iter()
        .map(|a| Box::new(a.historian()) as Box<dyn Historian>)
        .collect();

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
        for (i, state) in cfr_states.iter().enumerate() {
            // Export the CFR state to JSON
            export_cfr_state(
                state,
                path.join(format!("cfr_state_{}.png", i)).as_path(),
                ExportFormat::Png,
            )
            .expect("failed to export cfr state");
        }
    }
}

#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;

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
