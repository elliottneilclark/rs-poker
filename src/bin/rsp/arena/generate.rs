use std::fs;
use std::path::Path;
use std::sync::Arc;

use clap::Args;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use tracing::{info, warn};

use rs_poker::arena::agent::{AgentConfig, ConfigAgentBuilder};
use rs_poker::arena::cfr::{CFRState, TraversalSet};
use rs_poker::arena::historian::OpenHandHistoryHistorian;
use rs_poker::arena::{GameStateBuilder, HoldemSimulationBuilder};

/// Maximum consecutive failures before aborting generation
const MAX_CONSECUTIVE_FAILURES: usize = 100;

#[derive(Debug, thiserror::Error)]
pub enum GenerateError {
    #[error("--min-players must be >= 2")]
    MinPlayersTooFew,
    #[error("--min-players must be <= --max-players")]
    MinPlayersExceedsMax,
    #[error("--max-players ({max}) must be <= number of loaded agent configs ({configs})")]
    MaxPlayersExceedsConfigs { max: usize, configs: usize },
    #[error("stack sizes must be > 0")]
    InvalidStackSize,
    #[error("--min-stack-bb must be <= --max-stack-bb")]
    MinStackExceedsMax,
    #[error("blinds must be > 0")]
    InvalidBlinds,
    #[error("--small-blind must be < --big-blind")]
    SmallBlindExceedsBig,
    #[error("failed to read agents directory '{path}': {source}")]
    ReadAgentsDir {
        path: String,
        source: std::io::Error,
    },
    #[error("no valid agent configs found in '{0}'")]
    NoConfigs(String),
    #[error("failed to create thread pool: {0}")]
    ThreadPool(#[from] rayon::ThreadPoolBuildError),
    #[error("too many consecutive failures ({0}), aborting generation")]
    TooManyFailures(usize),
}

/// Generate Open Hand History files from poker simulations
#[derive(Args, Debug)]
pub struct GenerateArgs {
    /// Directory containing agent JSON config files
    agents_dir: std::path::PathBuf,

    /// Path to output .ohh file (appends if exists)
    #[arg(short = 'o', long = "output")]
    output: std::path::PathBuf,

    /// Number of games to generate (0 = run forever)
    #[arg(short = 'n', long = "num-games", default_value_t = 0)]
    num_games: usize,

    /// Minimum number of players per game
    #[arg(long = "min-players", default_value_t = 2)]
    min_players: usize,

    /// Maximum number of players per game
    #[arg(long = "max-players", default_value_t = 3)]
    max_players: usize,

    /// Big blind amount
    #[arg(long = "big-blind", default_value_t = 10.0)]
    big_blind: f32,

    /// Small blind amount
    #[arg(long = "small-blind", default_value_t = 5.0)]
    small_blind: f32,

    /// Minimum starting stack in big blinds
    #[arg(long = "min-stack-bb", default_value_t = 100.0)]
    min_stack_bb: f32,

    /// Maximum starting stack in big blinds
    #[arg(long = "max-stack-bb", default_value_t = 300.0)]
    max_stack_bb: f32,

    /// Number of threads for parallel CFR action exploration
    #[arg(long)]
    parallel: Option<usize>,

    /// Optional random seed for reproducibility
    #[arg(short = 's', long = "seed")]
    seed: Option<u64>,
}

fn load_configs(dir: &Path) -> Result<Vec<AgentConfig>, GenerateError> {
    let mut configs = Vec::new();
    let entries = fs::read_dir(dir).map_err(|e| GenerateError::ReadAgentsDir {
        path: dir.display().to_string(),
        source: e,
    })?;

    let mut dir_entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
    dir_entries.sort_by_key(|e| e.path());

    for entry in dir_entries {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        match ConfigAgentBuilder::from_file(&path) {
            Ok(builder) => {
                info!("Loaded config: {}", path.display());
                configs.push(builder.config().clone());
            }
            Err(e) => {
                warn!("Skipping '{}': {}", path.display(), e);
            }
        }
    }
    Ok(configs)
}

fn validate_args(args: &GenerateArgs, num_configs: usize) -> Result<(), GenerateError> {
    if args.min_players < 2 {
        return Err(GenerateError::MinPlayersTooFew);
    }
    if args.min_players > args.max_players {
        return Err(GenerateError::MinPlayersExceedsMax);
    }
    if args.max_players > num_configs {
        return Err(GenerateError::MaxPlayersExceedsConfigs {
            max: args.max_players,
            configs: num_configs,
        });
    }
    if args.min_stack_bb <= 0.0 || args.max_stack_bb <= 0.0 {
        return Err(GenerateError::InvalidStackSize);
    }
    if args.min_stack_bb > args.max_stack_bb {
        return Err(GenerateError::MinStackExceedsMax);
    }
    if args.small_blind <= 0.0 || args.big_blind <= 0.0 {
        return Err(GenerateError::InvalidBlinds);
    }
    if args.small_blind >= args.big_blind {
        return Err(GenerateError::SmallBlindExceedsBig);
    }
    Ok(())
}

fn run_generation(args: &GenerateArgs, configs: &[AgentConfig]) -> Result<(), GenerateError> {
    let mut rng: StdRng = match args.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_rng(&mut rand::rng()),
    };

    let thread_pool = match args.parallel {
        Some(num_threads) => Some(Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()?,
        )),
        None => None,
    };

    let mut games_completed: usize = 0;
    let mut consecutive_failures: usize = 0;
    let run_forever = args.num_games == 0;

    // Dynamic progress reporting interval
    let report_interval = if run_forever {
        1000
    } else {
        (args.num_games / 10).max(1)
    };

    loop {
        if !run_forever && games_completed >= args.num_games {
            break;
        }

        // Random number of players
        let num_players = rng.random_range(args.min_players..=args.max_players);

        // Random stacks
        let stacks: Vec<f32> = (0..num_players)
            .map(|_| rng.random_range(args.min_stack_bb..=args.max_stack_bb) * args.big_blind)
            .collect();

        // Random dealer
        let dealer_idx = rng.random_range(0..num_players);

        // Random agent configs (with replacement)
        let selected_configs: Vec<&AgentConfig> = (0..num_players)
            .map(|_| &configs[rng.random_range(0..configs.len())])
            .collect();

        // Build game state
        let game_state = match GameStateBuilder::new()
            .stacks(stacks)
            .big_blind(args.big_blind)
            .small_blind(args.small_blind)
            .dealer_idx(dealer_idx)
            .build()
        {
            Ok(gs) => gs,
            Err(e) => {
                consecutive_failures += 1;
                warn!("Failed to build game state: {}", e);
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                    return Err(GenerateError::TooManyFailures(consecutive_failures));
                }
                continue;
            }
        };

        // Check if any selected config is CFR
        let has_cfr = selected_configs.iter().any(|c| c.is_cfr());

        // Create shared CFR context if needed
        let cfr_context = if has_cfr {
            let cfr_states: Vec<CFRState> = (0..num_players)
                .map(|_| CFRState::new(game_state.clone()))
                .collect();
            let traversal_set = TraversalSet::new(num_players);
            Some((cfr_states, traversal_set))
        } else {
            None
        };

        // Build agents
        let agents_result: Result<Vec<_>, String> = selected_configs
            .iter()
            .enumerate()
            .map(|(idx, config)| {
                let mut builder = ConfigAgentBuilder::new((*config).clone())
                    .map_err(|e| format!("config error: {}", e))?
                    .player_idx(idx);
                // Inject shared CFR context BEFORE game_state to avoid
                // wasted eager allocation in game_state()
                if let Some((ref cfr_states, ref ts)) = cfr_context {
                    builder = builder.cfr_context(cfr_states.clone(), ts.clone());
                }
                builder = builder.game_state(game_state.clone());
                if let Some(ref pool) = thread_pool {
                    builder = builder.thread_pool(pool.clone());
                }
                builder = builder.rng_seed(rng.random::<u64>());
                Ok(builder.build())
            })
            .collect();

        let agents = match agents_result {
            Ok(a) => a,
            Err(e) => {
                consecutive_failures += 1;
                warn!("Failed to build agents: {}", e);
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                    return Err(GenerateError::TooManyFailures(consecutive_failures));
                }
                continue;
            }
        };

        // Create historian
        let historian = OpenHandHistoryHistorian::new(args.output.clone());

        // Build simulation
        let sim_result = {
            let mut builder = HoldemSimulationBuilder::default()
                .game_state(game_state)
                .agents(agents)
                .historians(vec![Box::new(historian)]);
            if let Some((cfr_states, traversal_set)) = cfr_context {
                builder = builder.cfr_context(cfr_states, traversal_set, true);
            }
            builder.build()
        };

        let mut sim = match sim_result {
            Ok(s) => s,
            Err(e) => {
                consecutive_failures += 1;
                warn!("Failed to build simulation: {}", e);
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                    return Err(GenerateError::TooManyFailures(consecutive_failures));
                }
                continue;
            }
        };

        // Run the game
        sim.run(&mut rng);
        games_completed += 1;
        consecutive_failures = 0;

        if games_completed.is_multiple_of(report_interval) {
            info!("Generated {} hands...", games_completed);
        }
    }

    info!("Done. Generated {} hands total.", games_completed);
    Ok(())
}

pub fn run(args: GenerateArgs) -> Result<(), GenerateError> {
    let configs = load_configs(&args.agents_dir)?;
    if configs.is_empty() {
        return Err(GenerateError::NoConfigs(
            args.agents_dir.display().to_string(),
        ));
    }
    info!("Loaded {} agent config(s)", configs.len());

    validate_args(&args, configs.len())?;
    run_generation(&args, &configs)
}
