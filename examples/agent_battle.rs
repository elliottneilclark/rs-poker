extern crate rs_poker;

use clap::Parser;
use rs_poker::arena::{
    AgentGenerator, CloneHistorianGenerator, HistorianGenerator,
    agent::{AgentConfig, AgentConfigError, ConfigAgentGenerator},
    cfr::StateStore,
    competition::{HoldemCompetition, StandardSimulationIterator},
    errors::HoldemSimulationError,
    game_state::RandomGameStateGenerator,
    historian::DirectoryHistorian,
};
use std::{fs, sync::Arc};
use thiserror::Error;

#[derive(Parser, Debug)]
#[command(
    name = "agent_battle",
    about = "Run a poker agent battle simulation",
    long_about = "Simulate poker games with agents configured via JSON files or inline JSON.\n\
                  Use --agent-config multiple times to add multiple agents."
)]
struct Args {
    /// Agent specification (JSON file path or inline JSON)
    /// Can be repeated to add multiple agents
    #[arg(short = 'a', long = "agent-config", required = true)]
    agent_configs: Vec<String>,

    /// Number of rounds to simulate per batch
    #[arg(short = 'b', long, default_value_t = 500)]
    rounds_per_batch: usize,

    /// Total number of batches to run
    #[arg(short = 'n', long, default_value_t = 5000)]
    num_batches: usize,

    /// Directory to save game history
    #[arg(short = 'd', long, default_value = "historian_out")]
    output_dir: String,

    /// Minimum starting stack (in big blinds)
    #[arg(long, default_value_t = 10.0)]
    min_stack_bb: f32,

    /// Maximum starting stack (in big blinds)
    #[arg(long, default_value_t = 1000.0)]
    max_stack_bb: f32,

    /// Big blind amount
    #[arg(long, default_value_t = 10.0)]
    big_blind: f32,

    /// Small blind amount
    #[arg(long, default_value_t = 5.0)]
    small_blind: f32,
}

type Result<T> = std::result::Result<T, AgentBattleError>;

#[derive(Debug, Error)]
enum AgentBattleError {
    #[error("failed to read agent spec {path}")]
    ReadSpec {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse agent spec {label}")]
    ParseSpec {
        label: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to build generator for spec #{index}")]
    BuildAgent {
        index: usize,
        #[source]
        source: AgentConfigError,
    },
    #[error(transparent)]
    Holdem(#[from] HoldemSimulationError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Agent Battle Configuration:");
    println!("===========================");
    println!("Number of agents: {}", args.agent_configs.len());
    println!("Rounds per batch: {}", args.rounds_per_batch);
    println!("Total batches: {}", args.num_batches);
    println!("Output directory: {}", args.output_dir);
    println!(
        "Stack range: {}-{} BB",
        args.min_stack_bb, args.max_stack_bb
    );
    println!("Blinds: {}/{}", args.small_blind, args.big_blind);
    println!();

    // Load agent configurations
    println!("Loading agent configurations:");
    let mut agent_gens: Vec<Box<dyn AgentGenerator>> = Vec::new();

    // Create a shared StateStore for CFR agents
    let state_store = Arc::new(StateStore::new());

    for (idx, spec) in args.agent_configs.iter().enumerate() {
        let generator = build_generator(idx, spec, &state_store)?;
        println!("  Agent {}: {:?}", idx, generator.config());
        agent_gens.push(Box::new(generator));
    }
    println!();

    // Show how to use the historian to record the games.
    let path = std::env::current_dir()?;
    let dir = path.join(&args.output_dir);
    let hist_gens: Vec<Box<dyn HistorianGenerator>> = vec![Box::new(CloneHistorianGenerator::new(
        DirectoryHistorian::new(dir),
    ))];

    // Convert BB to chip stacks
    let min_stack = args.min_stack_bb * args.big_blind;
    let max_stack = args.max_stack_bb * args.big_blind;

    // Run the games with completely random hands.
    let game_state_gen = RandomGameStateGenerator::new(
        agent_gens.len(),
        min_stack,
        max_stack,
        args.big_blind,
        args.small_blind,
        0.0,
    );
    let simulation_gen = StandardSimulationIterator::new(agent_gens, hist_gens, game_state_gen);
    let mut comp = HoldemCompetition::new(simulation_gen);

    println!("Starting simulation...");
    println!();

    let progress_interval = args.num_batches / 10;
    let progress_interval = if progress_interval < 1 {
        1
    } else {
        progress_interval
    };

    for i in 0..args.num_batches {
        let _recent_sims = comp.run(args.rounds_per_batch)?;

        // Show progress stats throughout execution
        if (i + 1) % progress_interval == 0 || i == 0 || i == args.num_batches - 1 {
            let progress_pct = ((i + 1) as f64 / args.num_batches as f64) * 100.0;
            println!(
                "Progress: {}/{} batches ({:.1}%)",
                i + 1,
                args.num_batches,
                progress_pct
            );
            println!("Current Stats: {comp:?}");
            println!();
        }
    }

    println!("Simulation complete!");
    println!("Final Competition Stats: {comp:?}");

    Ok(())
}

fn build_generator(
    idx: usize,
    spec: &str,
    state_store: &Arc<StateStore>,
) -> Result<ConfigAgentGenerator> {
    let config = parse_agent_config(spec, idx)?;
    let needs_state_store = matches!(
        &config,
        AgentConfig::CfrBasic { .. } | AgentConfig::CfrPerRound { .. }
    );

    if needs_state_store {
        ConfigAgentGenerator::with_state_store(config, Some(state_store.clone()))
    } else {
        ConfigAgentGenerator::new(config)
    }
    .map_err(|source| AgentBattleError::BuildAgent { index: idx, source })
}

fn parse_agent_config(spec: &str, idx: usize) -> Result<AgentConfig> {
    if spec.trim_start().starts_with('{') {
        serde_json::from_str(spec).map_err(|source| AgentBattleError::ParseSpec {
            label: format!("inline spec #{idx}"),
            source,
        })
    } else {
        let contents = fs::read_to_string(spec).map_err(|source| AgentBattleError::ReadSpec {
            path: spec.to_string(),
            source,
        })?;
        serde_json::from_str(&contents).map_err(|source| AgentBattleError::ParseSpec {
            label: spec.to_string(),
            source,
        })
    }
}
