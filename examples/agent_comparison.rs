extern crate rs_poker;

use clap::Parser;
use itertools::Itertools;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rs_poker::arena::agent::{AgentConfig, AgentConfigError, ConfigAgentGenerator};
use rs_poker::arena::cfr::StateStore;
use rs_poker::arena::game_state::{GameState, RandomGameStateGenerator};
use rs_poker::arena::historian::{StatsStorage, StatsTrackingHistorian};
use rs_poker::arena::{AgentGenerator, HoldemSimulationBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

#[derive(Parser, Debug)]
#[command(
    name = "agent_comparison",
    about = "Compare poker agents across all possible matchups and positions",
    long_about = "Evaluates poker agents by running all permutations of seat arrangements,\n\
                  tracking detailed per-agent statistics to determine which agents perform best."
)]
struct Args {
    /// Directory containing agent JSON config files
    agents_dir: PathBuf,

    /// Number of unique game states to test
    #[arg(short = 'n', long = "num-games", default_value_t = 1000)]
    num_games: usize,

    /// Number of players per table (must be >= 2 and <= number of agents)
    #[arg(short = 'p', long = "players", default_value_t = 3)]
    players_per_table: usize,

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
    #[arg(long = "max-stack-bb", default_value_t = 100.0)]
    max_stack_bb: f32,

    /// Optional directory to save game history and results
    #[arg(short = 'o', long = "output-dir")]
    output_dir: Option<PathBuf>,

    /// Optional random seed for reproducibility
    #[arg(short = 's', long = "seed")]
    seed: Option<u64>,
}

#[derive(Debug)]
struct AgentComparisonConfig {
    agents_dir: PathBuf,
    num_games: usize,
    players_per_table: usize,
    big_blind: f32,
    small_blind: f32,
    min_stack_bb: f32,
    max_stack_bb: f32,
    output_dir: Option<PathBuf>,
    seed: Option<u64>,
}

impl From<Args> for AgentComparisonConfig {
    fn from(args: Args) -> Self {
        Self {
            agents_dir: args.agents_dir,
            num_games: args.num_games,
            players_per_table: args.players_per_table,
            big_blind: args.big_blind,
            small_blind: args.small_blind,
            min_stack_bb: args.min_stack_bb,
            max_stack_bb: args.max_stack_bb,
            output_dir: args.output_dir,
            seed: args.seed,
        }
    }
}

impl AgentComparisonConfig {
    /// Validate the comparison configuration
    fn validate(&self, num_agents: usize) -> Result<()> {
        // Verify players_per_table >= 2
        if self.players_per_table < 2 {
            return Err(AgentComparisonError::ValidationError(
                "players_per_table must be at least 2".to_string(),
            ));
        }

        // Verify players_per_table <= num_agents
        if self.players_per_table > num_agents {
            return Err(AgentComparisonError::ValidationError(format!(
                "players_per_table ({}) cannot exceed number of agents ({})",
                self.players_per_table, num_agents
            )));
        }

        // Verify num_games > 0
        if self.num_games == 0 {
            return Err(AgentComparisonError::ValidationError(
                "num_games must be greater than 0".to_string(),
            ));
        }

        // Verify blinds are positive
        if self.big_blind <= 0.0 {
            return Err(AgentComparisonError::ValidationError(
                "big_blind must be positive".to_string(),
            ));
        }

        if self.small_blind <= 0.0 {
            return Err(AgentComparisonError::ValidationError(
                "small_blind must be positive".to_string(),
            ));
        }

        // Verify small blind is less than big blind
        if self.small_blind >= self.big_blind {
            return Err(AgentComparisonError::ValidationError(
                "small_blind must be less than big_blind".to_string(),
            ));
        }

        // Verify stack sizes are positive
        if self.min_stack_bb <= 0.0 {
            return Err(AgentComparisonError::ValidationError(
                "min_stack_bb must be positive".to_string(),
            ));
        }

        if self.max_stack_bb <= 0.0 {
            return Err(AgentComparisonError::ValidationError(
                "max_stack_bb must be positive".to_string(),
            ));
        }

        // Verify min <= max
        if self.min_stack_bb > self.max_stack_bb {
            return Err(AgentComparisonError::ValidationError(
                "min_stack_bb cannot exceed max_stack_bb".to_string(),
            ));
        }

        Ok(())
    }
}

/// Statistics for a specific position
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PositionStats {
    seat_index: usize,
    games_played: usize,
    profit: f32,
    profit_per_game: f32,
}

/// Aggregated statistics for a single agent across all permutations
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentStats {
    agent_name: String,

    // Financial Performance
    total_profit: f32,
    total_games: usize,
    wins: usize,
    losses: usize,
    breakeven: usize,

    // Profitability Metrics (calculated from counts)
    profit_per_game: f32,
    profit_per_100_hands: f32,
    roi_percent: f32,

    // Position Analysis
    position_stats: Vec<PositionStats>,

    // Poker Statistics (calculated from counts in StatsStorage)
    vpip_percent: f32,
    pfr_percent: f32,
    three_bet_percent: f32,
    aggression_factor: f32,

    // Round-by-Round Performance (calculated from counts)
    preflop_win_rate: f32,
    flop_win_rate: f32,
    turn_win_rate: f32,
    river_win_rate: f32,
}

/// Builder for aggregating agent statistics across multiple games
struct AgentStatsBuilder {
    // Accumulated stats for each agent (indexed by agent_idx)
    agent_accumulated: Vec<StatsStorage>,
    // Position tracking: agent_idx -> seat_idx -> (games_played, total_profit)
    position_tracking: Vec<HashMap<usize, (usize, f32)>>,
    // Agent names (indexed by agent_idx)
    agent_names: Vec<String>,
}

impl AgentStatsBuilder {
    /// Create a new builder with the given agent names
    fn new(agent_names: Vec<String>) -> Self {
        let num_agents = agent_names.len();
        let agent_accumulated = (0..num_agents)
            .map(|_| StatsStorage::new_with_num_players(1))
            .collect();
        let position_tracking = (0..num_agents).map(|_| HashMap::new()).collect();

        Self {
            agent_accumulated,
            position_tracking,
            agent_names,
        }
    }

    /// Merge statistics from a single permutation into the builder
    ///
    /// # Arguments
    /// * `permutation` - Vector of agent indices representing seat assignments
    /// * `stats` - StatsStorage containing the results from this game
    fn merge_permutation_stats(&mut self, permutation: &[usize], stats: &StatsStorage) {
        for (seat_idx, &agent_idx) in permutation.iter().enumerate() {
            let agent_stats = &mut self.agent_accumulated[agent_idx];
            let player_idx = 0; // We aggregate all stats to index 0 for each agent

            // Manually accumulate the stats from seat_idx to our single-player storage at idx 0
            agent_stats.actions_count[player_idx] += stats.actions_count[seat_idx];
            agent_stats.vpip_count[player_idx] += stats.vpip_count[seat_idx];
            agent_stats.vpip_total[player_idx] += stats.vpip_total[seat_idx];
            agent_stats.raise_count[player_idx] += stats.raise_count[seat_idx];

            // New Phase 1 fields (now public)
            agent_stats.preflop_raise_count[player_idx] += stats.preflop_raise_count[seat_idx];
            agent_stats.preflop_actions[player_idx] += stats.preflop_actions[seat_idx];
            agent_stats.three_bet_count[player_idx] += stats.three_bet_count[seat_idx];
            agent_stats.three_bet_opportunities[player_idx] +=
                stats.three_bet_opportunities[seat_idx];
            agent_stats.call_count[player_idx] += stats.call_count[seat_idx];
            agent_stats.bet_count[player_idx] += stats.bet_count[seat_idx];

            // Financial tracking
            agent_stats.total_profit[player_idx] += stats.total_profit[seat_idx];
            agent_stats.games_won[player_idx] += stats.games_won[seat_idx];
            agent_stats.games_lost[player_idx] += stats.games_lost[seat_idx];
            agent_stats.games_breakeven[player_idx] += stats.games_breakeven[seat_idx];

            // Round outcomes
            agent_stats.preflop_wins[player_idx] += stats.preflop_wins[seat_idx];
            agent_stats.flop_wins[player_idx] += stats.flop_wins[seat_idx];
            agent_stats.turn_wins[player_idx] += stats.turn_wins[seat_idx];
            agent_stats.river_wins[player_idx] += stats.river_wins[seat_idx];
            agent_stats.preflop_completes[player_idx] += stats.preflop_completes[seat_idx];
            agent_stats.flop_completes[player_idx] += stats.flop_completes[seat_idx];
            agent_stats.turn_completes[player_idx] += stats.turn_completes[seat_idx];
            agent_stats.river_completes[player_idx] += stats.river_completes[seat_idx];

            // Track position-specific stats
            let pos_map = &mut self.position_tracking[agent_idx];
            let (games, profit) = pos_map.entry(seat_idx).or_insert((0, 0.0));
            *games += 1;
            *profit += stats.total_profit[seat_idx];
        }
    }

    /// Build and consume the builder, returning aggregated stats
    fn build(self) -> AgentStatsAggregator {
        let mut agent_stats = HashMap::new();

        // Finalize each agent's stats
        for (agent_idx, agent_name) in self.agent_names.iter().enumerate() {
            let stats = &self.agent_accumulated[agent_idx];
            let player_idx = 0;

            // Calculate position stats
            let mut position_stats = Vec::new();
            if let Some(pos_map) = self.position_tracking.get(agent_idx) {
                for (seat_idx, (games_played, total_profit)) in pos_map {
                    let profit_per_game = if *games_played > 0 {
                        total_profit / *games_played as f32
                    } else {
                        0.0
                    };

                    position_stats.push(PositionStats {
                        seat_index: *seat_idx,
                        games_played: *games_played,
                        profit: *total_profit,
                        profit_per_game,
                    });
                }
            }

            // Sort position stats by seat index
            position_stats.sort_by_key(|ps| ps.seat_index);

            let total_games = stats.games_won[player_idx]
                + stats.games_lost[player_idx]
                + stats.games_breakeven[player_idx];

            // Calculate all derived metrics
            let agent_stat = AgentStats {
                agent_name: agent_name.clone(),
                total_profit: stats.total_profit[player_idx],
                total_games,
                wins: stats.games_won[player_idx],
                losses: stats.games_lost[player_idx],
                breakeven: stats.games_breakeven[player_idx],
                profit_per_game: stats.profit_per_game(player_idx),
                profit_per_100_hands: stats.profit_per_game(player_idx) * 100.0,
                roi_percent: stats.roi_percent(player_idx),
                position_stats,
                vpip_percent: stats.vpip_percent(player_idx),
                pfr_percent: stats.pfr_percent(player_idx),
                three_bet_percent: stats.three_bet_percent(player_idx),
                aggression_factor: stats.aggression_factor(player_idx),
                preflop_win_rate: stats.preflop_win_rate(player_idx),
                flop_win_rate: stats.flop_win_rate(player_idx),
                turn_win_rate: stats.turn_win_rate(player_idx),
                river_win_rate: stats.river_win_rate(player_idx),
            };

            agent_stats.insert(agent_name.clone(), agent_stat);
        }

        AgentStatsAggregator {
            agent_names: self.agent_names,
            agent_stats,
        }
    }
}

/// Aggregated statistics for all agents
struct AgentStatsAggregator {
    agent_names: Vec<String>,
    agent_stats: HashMap<String, AgentStats>,
}

impl AgentStatsAggregator {
    /// Get rankings sorted by profit per game (descending)
    fn get_rankings(&self) -> Vec<(&String, &AgentStats)> {
        let mut rankings: Vec<_> = self.agent_stats.iter().collect();
        rankings.sort_by(|a, b| {
            b.1.profit_per_game
                .partial_cmp(&a.1.profit_per_game)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rankings
    }
}

type Result<T> = std::result::Result<T, AgentComparisonError>;

#[derive(Debug, Error)]
enum AgentComparisonError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to parse agent config from {path}: {source}")]
    ParseConfig {
        path: String,
        source: serde_json::Error,
    },

    #[error("Failed to validate agent config: {0}")]
    InvalidConfig(#[from] AgentConfigError),

    #[error("Configuration validation error: {0}")]
    ValidationError(String),

    #[error("No agent config files found in directory: {0}")]
    NoAgentsFound(String),

    #[error("Simulation error: {0}")]
    SimulationError(String),

    #[error("Failed to serialize JSON: {0}")]
    JsonSerialize(#[from] serde_json::Error),
}

/// Run all permutations for a single game state
///
/// Generates all permutations of agents and runs a game for each permutation
fn run_single_game_state(
    game_state: GameState,
    players_per_table: usize,
    agent_configs: &[AgentConfig],
    builder: &mut AgentStatsBuilder,
    rng: &mut impl Rng,
) -> Result<()> {
    // Generate all permutations of size players_per_table from num_agents
    for permutation in (0..agent_configs.len()).permutations(players_per_table) {
        // Create a fresh StateStore for this specific permutation
        // (only used if there are CFR agents, otherwise has no cost)
        let mut state_store = StateStore::new();

        // Make sure that the store has states for each player in this game
        let _states = (0..players_per_table)
            .map(|player_idx| state_store.new_state(game_state.clone(), player_idx))
            .collect::<Vec<_>>();

        let arc_state_store = Arc::new(state_store);
        // Create ConfigAgentGenerator for each agent in this permutation
        let agent_generators: Vec<ConfigAgentGenerator> = permutation
            .iter()
            .map(|&agent_idx| {
                ConfigAgentGenerator::with_state_store(
                    agent_configs[agent_idx].clone(),
                    Some(arc_state_store.clone()),
                )
                .expect("Failed to create agent generator")
            })
            .collect();

        // Create agents for this permutation
        let boxed_agents: Vec<Box<dyn rs_poker::arena::Agent>> = agent_generators
            .iter()
            .enumerate()
            .map(|(idx, generator)| generator.generate(idx, &game_state))
            .collect();

        // Create stats historian and get a clone of its storage
        let historian = StatsTrackingHistorian::new_with_num_players(players_per_table);
        let stats_storage = historian.get_storage();

        // Run simulation with the cloned game state
        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state.clone())
            .agents(boxed_agents)
            .historians(vec![Box::new(historian)])
            .build()
            .map_err(|e| AgentComparisonError::SimulationError(e.to_string()))?;

        sim.run(rng);

        // Extract statistics from the historian via the shared storage
        let stats = stats_storage
            .try_borrow()
            .map_err(|e| AgentComparisonError::SimulationError(e.to_string()))?;

        // Merge the stats into the builder using agent indices
        builder.merge_permutation_stats(&permutation, &stats);
    }

    Ok(())
}

/// Run all game states with all permutations
///
/// Generates num_games unique game states, and for each one runs all permutations
/// of agent combinations.
fn run_all_game_states(
    agents: Vec<(String, AgentConfig)>,
    config: &AgentComparisonConfig,
) -> Result<AgentStatsAggregator> {
    // Create RNG
    let mut rng = if let Some(seed) = config.seed {
        StdRng::seed_from_u64(seed)
    } else {
        // Use a random seed from the OS
        let seed = rand::random::<u64>();
        StdRng::seed_from_u64(seed)
    };

    // Extract names and configs separately
    let agent_names: Vec<String> = agents.iter().map(|(name, _)| name.clone()).collect();
    let agent_configs: Vec<AgentConfig> = agents.into_iter().map(|(_, config)| config).collect();

    // Create stats builder
    let mut builder = AgentStatsBuilder::new(agent_names);

    // Create game state generator
    let min_stack = config.min_stack_bb * config.big_blind;
    let max_stack = config.max_stack_bb * config.big_blind;
    let mut game_state_gen = RandomGameStateGenerator::new(
        config.players_per_table,
        min_stack,
        max_stack,
        config.big_blind,
        config.small_blind,
        0.0, // no ante
    );

    // Run simulations for each game state
    for _game_idx in 0..config.num_games {
        // Generate a game state
        let game_state = game_state_gen.next().ok_or_else(|| {
            AgentComparisonError::SimulationError("Failed to generate game state".to_string())
        })?;

        // Run all permutations with this game state
        run_single_game_state(
            game_state,
            config.players_per_table,
            &agent_configs,
            &mut builder,
            &mut rng,
        )?;
    }

    println!("\nCompleted all {} game states!", config.num_games);

    // Build and return the final aggregated statistics
    Ok(builder.build())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = AgentComparisonConfig::from(args);

    // Load agent configurations from directory
    let agents = load_agents_from_dir(&config.agents_dir)?;

    // Validate configuration
    config.validate(agents.len())?;

    // Print configuration summary
    print_configuration_summary(&config, &agents);

    // Run simulations and aggregate statistics
    println!("\nStarting simulations...");
    let aggregator = run_all_game_states(agents, &config)?;

    // Generate output
    let num_agents = aggregator.agent_names.len();
    let total_permutations = (0..num_agents)
        .permutations(config.players_per_table)
        .count();
    let markdown_output = format_markdown_output(&aggregator, &config, total_permutations);

    // Print to console
    println!("{}", markdown_output);

    // Save to files if output directory specified
    if let Some(ref output_dir) = config.output_dir {
        save_results(&aggregator, &markdown_output, output_dir)?;
        println!("\nResults saved to:");
        println!("  - {}", output_dir.join("results.json").display());
        println!("  - {}", output_dir.join("results.md").display());
    }

    Ok(())
}

/// Extract the name from an AgentConfig, using the name field if present
fn get_agent_name(config: &AgentConfig, fallback_name: &str) -> String {
    match config {
        AgentConfig::AllIn { name } => name.clone().unwrap_or_else(|| fallback_name.to_string()),
        AgentConfig::Calling { name } => name.clone().unwrap_or_else(|| fallback_name.to_string()),
        AgentConfig::Folding { name } => name.clone().unwrap_or_else(|| fallback_name.to_string()),
        AgentConfig::Random { name, .. } => {
            name.clone().unwrap_or_else(|| fallback_name.to_string())
        }
        AgentConfig::RandomPotControl { name, .. } => {
            name.clone().unwrap_or_else(|| fallback_name.to_string())
        }
        AgentConfig::CfrBasic { name, .. } => {
            name.clone().unwrap_or_else(|| fallback_name.to_string())
        }
        AgentConfig::CfrPerRound { name, .. } => {
            name.clone().unwrap_or_else(|| fallback_name.to_string())
        }
    }
}

/// Load agent configurations from a directory of JSON files
fn load_agents_from_dir(dir: &Path) -> Result<Vec<(String, AgentConfig)>> {
    if !dir.exists() {
        return Err(AgentComparisonError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Directory does not exist: {}", dir.display()),
        )));
    }

    if !dir.is_dir() {
        return Err(AgentComparisonError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Path is not a directory: {}", dir.display()),
        )));
    }

    let mut agents = Vec::new();
    let entries = std::fs::read_dir(dir)?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        // Only process .json files
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }

        let fallback_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        match load_agent_config(&path) {
            Ok(config) => {
                let agent_name = get_agent_name(&config, &fallback_name);
                agents.push((agent_name, config));
            }
            Err(e) => {
                eprintln!(
                    "Warning: Skipping invalid config file {}: {}",
                    path.display(),
                    e
                );
            }
        }
    }

    if agents.is_empty() {
        return Err(AgentComparisonError::NoAgentsFound(
            dir.display().to_string(),
        ));
    }

    // Sort by agent name for consistent ordering
    agents.sort_by(|a, b| a.0.cmp(&b.0));

    Ok(agents)
}

/// Load and validate a single agent configuration from a file
fn load_agent_config(path: &Path) -> Result<AgentConfig> {
    let contents = std::fs::read_to_string(path)?;
    let config: AgentConfig =
        serde_json::from_str(&contents).map_err(|source| AgentComparisonError::ParseConfig {
            path: path.display().to_string(),
            source,
        })?;

    // Validate the configuration
    config.validate()?;

    Ok(config)
}

/// Print a summary of the configuration
fn print_configuration_summary(config: &AgentComparisonConfig, agents: &[(String, AgentConfig)]) {
    println!("Agent Comparison Configuration");
    println!("==============================");
    println!();
    println!("Agents Directory: {}", config.agents_dir.display());
    println!("Number of Agents: {}", agents.len());
    println!("Players per Table: {}", config.players_per_table);
    println!("Number of Games: {}", config.num_games);
    println!();
    println!("Game Settings:");
    println!("  Big Blind: {}", config.big_blind);
    println!("  Small Blind: {}", config.small_blind);
    println!(
        "  Stack Range: {}-{} BB",
        config.min_stack_bb, config.max_stack_bb
    );
    if let Some(seed) = config.seed {
        println!("  Random Seed: {}", seed);
    }
    if let Some(ref output_dir) = config.output_dir {
        println!("  Output Directory: {}", output_dir.display());
    }
    println!();
    println!("Loaded Agents:");
    for (name, agent_config) in agents {
        println!("  - {}: {:?}", name, agent_config);
    }
    println!();

    // Calculate and display permutation count using itertools
    let total_permutations = (0..agents.len())
        .permutations(config.players_per_table)
        .count();
    let total_games = total_permutations * config.num_games;

    println!("Simulation Scale:");
    println!("  Total permutations: {}", total_permutations);
    println!(
        "  Total games to simulate: {} × {} = {}",
        total_permutations, config.num_games, total_games
    );
    println!();
}

/// Format results as Markdown output
fn format_markdown_output(
    aggregator: &AgentStatsAggregator,
    config: &AgentComparisonConfig,
    total_permutations: usize,
) -> String {
    let mut output = String::new();

    // Header
    output.push_str(&format!("{}\n", "=".repeat(80)));
    output.push_str("# Agent Comparison Results\n");
    output.push_str(&format!("{}\n\n", "=".repeat(80)));

    // Configuration section
    output.push_str("## Configuration\n\n");
    output.push_str(&format!(
        "- **Agents Tested**: {}\n",
        aggregator.agent_names.len()
    ));
    output.push_str(&format!(
        "- **Players per Table**: {}\n",
        config.players_per_table
    ));
    output.push_str(&format!("- **Unique Game States**: {}\n", config.num_games));
    output.push_str(&format!(
        "- **Total Permutations**: {}\n",
        total_permutations
    ));
    output.push_str(&format!(
        "- **Total Games Simulated**: {}\n",
        total_permutations * config.num_games
    ));
    output.push_str(&format!("- **Big Blind**: {}\n", config.big_blind));
    output.push_str(&format!("- **Small Blind**: {}\n", config.small_blind));
    output.push_str(&format!(
        "- **Stack Range**: {}-{} BB\n",
        config.min_stack_bb, config.max_stack_bb
    ));
    if let Some(seed) = config.seed {
        output.push_str(&format!("- **Random Seed**: {}\n", seed));
    }
    output.push('\n');

    // Rankings
    output.push_str("## Rankings (by Profit per Game)\n\n");
    let rankings = aggregator.get_rankings();
    for (rank, (agent_name, stats)) in rankings.iter().enumerate() {
        let profit_bb = stats.profit_per_game / config.big_blind;
        output.push_str(&format!(
            "{}. **{}**: {:+.2} bb/game (ROI: {:+.1}%)\n",
            rank + 1,
            agent_name,
            profit_bb,
            stats.roi_percent
        ));
    }
    output.push('\n');

    // Detailed statistics for each agent
    output.push_str("## Detailed Statistics\n\n");
    for (agent_name, stats) in rankings.iter() {
        output.push_str(&format!("### {}\n\n", agent_name));

        // Financial Performance
        output.push_str("#### Financial Performance\n\n");
        output.push_str("| Metric | Value |\n");
        output.push_str("|--------|-------|\n");
        output.push_str(&format!(
            "| Total Profit | {:+.2} chips ({:+.2} bb) |\n",
            stats.total_profit,
            stats.total_profit / config.big_blind
        ));
        output.push_str(&format!("| Games Played | {} |\n", stats.total_games));
        output.push_str(&format!(
            "| Wins | {} ({:.1}%) |\n",
            stats.wins,
            if stats.total_games > 0 {
                100.0 * stats.wins as f32 / stats.total_games as f32
            } else {
                0.0
            }
        ));
        output.push_str(&format!(
            "| Losses | {} ({:.1}%) |\n",
            stats.losses,
            if stats.total_games > 0 {
                100.0 * stats.losses as f32 / stats.total_games as f32
            } else {
                0.0
            }
        ));
        output.push_str(&format!(
            "| Breakeven | {} ({:.1}%) |\n",
            stats.breakeven,
            if stats.total_games > 0 {
                100.0 * stats.breakeven as f32 / stats.total_games as f32
            } else {
                0.0
            }
        ));
        output.push_str(&format!(
            "| Profit/Game | {:+.2} bb |\n",
            stats.profit_per_game / config.big_blind
        ));
        output.push_str(&format!(
            "| Profit/100 Hands | {:+.2} bb |\n",
            stats.profit_per_100_hands / config.big_blind
        ));
        output.push_str(&format!("| ROI | {:+.1}% |\n", stats.roi_percent));
        output.push('\n');

        // Playing Style
        output.push_str("#### Playing Style\n\n");
        output.push_str("| Metric | Value |\n");
        output.push_str("|--------|-------|\n");
        output.push_str(&format!("| VPIP | {:.1}% |\n", stats.vpip_percent));
        output.push_str(&format!("| PFR | {:.1}% |\n", stats.pfr_percent));
        output.push_str(&format!("| 3-Bet % | {:.1}% |\n", stats.three_bet_percent));
        output.push_str(&format!(
            "| Aggression Factor | {:.2} |\n",
            stats.aggression_factor
        ));
        output.push('\n');

        // Round-by-Round Win Rates
        output.push_str("#### Round-by-Round Win Rates\n\n");
        output.push_str("| Round | Win Rate |\n");
        output.push_str("|-------|----------|\n");
        output.push_str(&format!("| Preflop | {:.1}% |\n", stats.preflop_win_rate));
        output.push_str(&format!("| Flop | {:.1}% |\n", stats.flop_win_rate));
        output.push_str(&format!("| Turn | {:.1}% |\n", stats.turn_win_rate));
        output.push_str(&format!("| River | {:.1}% |\n", stats.river_win_rate));
        output.push('\n');

        // Position Performance
        if !stats.position_stats.is_empty() {
            output.push_str("#### Position Performance\n\n");
            output.push_str("| Position (Seat) | Profit/Game | Games Played |\n");
            output.push_str("|-----------------|-------------|-------------|\n");
            for pos_stat in &stats.position_stats {
                output.push_str(&format!(
                    "| Seat {} | {:+.2} bb | {} |\n",
                    pos_stat.seat_index,
                    pos_stat.profit_per_game / config.big_blind,
                    pos_stat.games_played
                ));
            }
            output.push('\n');
        }

        output.push_str("---\n\n");
    }

    output
}

/// Save results to JSON and Markdown files
fn save_results(
    aggregator: &AgentStatsAggregator,
    markdown_output: &str,
    output_dir: &Path,
) -> Result<()> {
    // Create output directory if it doesn't exist
    std::fs::create_dir_all(output_dir)?;

    // Save JSON
    let json_path = output_dir.join("results.json");
    let json_output = serde_json::to_string_pretty(&aggregator.agent_stats)?;
    std::fs::write(&json_path, json_output)?;

    // Save Markdown
    let md_path = output_dir.join("results.md");
    std::fs::write(&md_path, markdown_output)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_config_valid() {
        let config = AgentComparisonConfig {
            agents_dir: PathBuf::from("."),
            num_games: 100,
            players_per_table: 3,
            big_blind: 10.0,
            small_blind: 5.0,
            min_stack_bb: 100.0,
            max_stack_bb: 100.0,
            output_dir: None,
            seed: None,
        };

        assert!(config.validate(5).is_ok());
    }

    #[test]
    fn test_validate_config_too_few_players() {
        let config = AgentComparisonConfig {
            agents_dir: PathBuf::from("."),
            num_games: 100,
            players_per_table: 1,
            big_blind: 10.0,
            small_blind: 5.0,
            min_stack_bb: 100.0,
            max_stack_bb: 100.0,
            output_dir: None,
            seed: None,
        };

        assert!(config.validate(5).is_err());
    }

    #[test]
    fn test_validate_config_too_many_players() {
        let config = AgentComparisonConfig {
            agents_dir: PathBuf::from("."),
            num_games: 100,
            players_per_table: 6,
            big_blind: 10.0,
            small_blind: 5.0,
            min_stack_bb: 100.0,
            max_stack_bb: 100.0,
            output_dir: None,
            seed: None,
        };

        assert!(config.validate(5).is_err());
    }

    #[test]
    fn test_validate_config_zero_games() {
        let config = AgentComparisonConfig {
            agents_dir: PathBuf::from("."),
            num_games: 0,
            players_per_table: 3,
            big_blind: 10.0,
            small_blind: 5.0,
            min_stack_bb: 100.0,
            max_stack_bb: 100.0,
            output_dir: None,
            seed: None,
        };

        assert!(config.validate(5).is_err());
    }

    #[test]
    fn test_validate_config_invalid_blinds() {
        let config = AgentComparisonConfig {
            agents_dir: PathBuf::from("."),
            num_games: 100,
            players_per_table: 3,
            big_blind: 5.0,
            small_blind: 10.0,
            min_stack_bb: 100.0,
            max_stack_bb: 100.0,
            output_dir: None,
            seed: None,
        };

        assert!(config.validate(5).is_err());
    }

    #[test]
    fn test_validate_config_invalid_stack_range() {
        let config = AgentComparisonConfig {
            agents_dir: PathBuf::from("."),
            num_games: 100,
            players_per_table: 3,
            big_blind: 10.0,
            small_blind: 5.0,
            min_stack_bb: 200.0,
            max_stack_bb: 100.0,
            output_dir: None,
            seed: None,
        };

        assert!(config.validate(5).is_err());
    }

    #[test]
    fn test_permutations_count() {
        // Test that itertools generates the correct number of permutations
        let num_agents = 5;
        let players_per_table = 3;

        let perms_count = (0..num_agents).permutations(players_per_table).count();

        // P(5, 3) = 5! / (5-3)! = 5 * 4 * 3 = 60
        assert_eq!(perms_count, 60);
    }

    #[test]
    fn test_permutations_unique() {
        // Test that all permutations are unique
        let perms: Vec<Vec<usize>> = (0..4).permutations(2).collect();

        // P(4, 2) = 4! / (4-2)! = 4 * 3 = 12
        assert_eq!(perms.len(), 12);

        // Verify all permutations are unique
        use std::collections::HashSet;
        let unique_perms: HashSet<Vec<usize>> = perms.iter().cloned().collect();
        assert_eq!(unique_perms.len(), perms.len());
    }

    #[test]
    fn test_get_agent_name() {
        let config_with_name = AgentConfig::AllIn {
            name: Some("MyAllIn".to_string()),
        };
        assert_eq!(get_agent_name(&config_with_name, "fallback"), "MyAllIn");

        let config_without_name = AgentConfig::AllIn { name: None };
        assert_eq!(get_agent_name(&config_without_name, "fallback"), "fallback");
    }
}
