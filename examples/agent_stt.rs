use clap::Parser;
use rs_poker::arena::{
    AgentGenerator,
    agent::{AgentConfig, AgentConfigError, ConfigAgentGenerator},
    cfr::StateStore,
    competition::SingleTableTournamentBuilder,
    errors::HoldemSimulationError,
    game_state::GameState,
};
use std::{fs, sync::Arc};
use thiserror::Error;

#[derive(Parser, Debug)]
#[command(
    name = "agent_stt",
    about = "Run a single table poker tournament",
    long_about = "Simulate a single table tournament with agents configured via JSON.\n\
                  Use --agent-config multiple times to add multiple agents."
)]
struct Args {
    /// Agent specification (JSON file path or inline JSON)
    /// Can be repeated to add multiple agents
    #[arg(short = 'a', long = "agent-config", required = true)]
    agent_configs: Vec<String>,

    /// Big blind amount
    #[arg(long, default_value_t = 10.0)]
    big_blind: f32,

    /// Small blind amount
    #[arg(long, default_value_t = 5.0)]
    small_blind: f32,

    /// Starting stack for each player
    #[arg(long, default_value_t = 100.0)]
    starting_stack: f32,
}

type Result<T> = std::result::Result<T, AgentSttError>;

#[derive(Debug, Error)]
enum AgentSttError {
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

    println!("Tournament Configuration:");
    println!("========================");
    println!("Number of players: {}", args.agent_configs.len());
    println!("Starting stack: {}", args.starting_stack);
    println!("Blinds: {}/{}", args.small_blind, args.big_blind);
    println!();

    // Load agent configurations
    println!("Loading agent configurations:");
    let mut agent_builders: Vec<Box<dyn AgentGenerator>> = Vec::new();

    // Create a shared StateStore for CFR agents
    let state_store = Arc::new(StateStore::new());

    for (idx, spec) in args.agent_configs.iter().enumerate() {
        let generator = build_generator(idx, spec, &state_store)?;
        println!("  Player {}: {:?}", idx, generator.config());
        agent_builders.push(Box::new(generator));
    }
    println!();

    // Create starting stacks for all players
    let stacks = vec![args.starting_stack; agent_builders.len()];

    let game_state = GameState::new_starting(stacks, args.big_blind, args.small_blind, 0.0, 0);

    let tournament = SingleTableTournamentBuilder::default()
        .agent_generators(agent_builders)
        .starting_game_state(game_state)
        .build()?;

    println!("Starting tournament...");
    println!();

    let results = tournament.run()?;

    println!("Tournament complete!");
    println!("Agent Results: {results:?}");

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
        ConfigAgentGenerator::with_state_store(config, Some(Arc::clone(state_store)))
    } else {
        ConfigAgentGenerator::new(config)
    }
    .map_err(|source| AgentSttError::BuildAgent { index: idx, source })
}

fn parse_agent_config(spec: &str, idx: usize) -> Result<AgentConfig> {
    if spec.trim_start().starts_with('{') {
        serde_json::from_str(spec).map_err(|source| AgentSttError::ParseSpec {
            label: format!("inline spec #{idx}"),
            source,
        })
    } else {
        let contents = fs::read_to_string(spec).map_err(|source| AgentSttError::ReadSpec {
            path: spec.to_string(),
            source,
        })?;
        serde_json::from_str(&contents).map_err(|source| AgentSttError::ParseSpec {
            label: spec.to_string(),
            source,
        })
    }
}
