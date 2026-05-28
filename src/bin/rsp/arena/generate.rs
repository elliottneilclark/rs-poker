use std::fs;
use std::path::Path;

use clap::Args;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use tracing::{info, warn};

use rs_poker::arena::agent::{Agent, AgentConfig, AgentConfigError, ConfigAgentBuilder};
use rs_poker::arena::cfr::{CFRState, TraversalSet};
use rs_poker::arena::game_state::GameState;
use rs_poker::arena::historian::{OpenHandHistoryHistorian, SharedStatsStorage};
use rs_poker::arena::{GameStateBuilder, Historian, HoldemSimulationBuilder, seeded_rng};

use crate::tui::app::{self, App};
use crate::tui::event::{EventHandler, SimError, SimMessage};
use crate::tui::hand_store::HandStore;
use crate::tui::state::{GameResult, SeatStats, ending_round_from_stats};
use crate::tui::{TuiFlags, run_blocking_tui_loop};

/// Maximum consecutive failures before aborting generation
const MAX_CONSECUTIVE_FAILURES: usize = 100;

#[derive(Debug, thiserror::Error)]
pub enum GenerateError {
    #[error("--min-players must be >= 2")]
    MinPlayersTooFew,
    #[error("--min-players must be <= --max-players")]
    MinPlayersExceedsMax,
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
    #[error("too many consecutive failures ({0}), aborting generation")]
    TooManyFailures(usize),
    #[error("TUI error: {0}")]
    TuiError(#[from] std::io::Error),
}

/// Generate Open Hand History files from poker simulations
#[derive(Args, Debug, Clone)]
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

    /// Optional random seed for reproducibility
    #[arg(short = 's', long = "seed")]
    seed: Option<u64>,

    #[command(flatten)]
    tui: TuiFlags,
}

impl GenerateArgs {
    fn validate(&self) -> Result<(), GenerateError> {
        if self.min_players < 2 {
            return Err(GenerateError::MinPlayersTooFew);
        }
        if self.min_players > self.max_players {
            return Err(GenerateError::MinPlayersExceedsMax);
        }
        // Allow max_players to exceed num_configs; configs can be reused.
        if self.min_stack_bb <= 0.0 || self.max_stack_bb <= 0.0 {
            return Err(GenerateError::InvalidStackSize);
        }
        if self.min_stack_bb > self.max_stack_bb {
            return Err(GenerateError::MinStackExceedsMax);
        }
        if self.small_blind <= 0.0 || self.big_blind <= 0.0 {
            return Err(GenerateError::InvalidBlinds);
        }
        if self.small_blind >= self.big_blind {
            return Err(GenerateError::SmallBlindExceedsBig);
        }
        Ok(())
    }
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

/// A successfully set up game, ready to be run.
struct GameSetup {
    game_state: GameState,
    agents: Vec<Box<dyn Agent>>,
    cfr_context: Option<(CFRState, TraversalSet)>,
    num_players: usize,
}

/// Shared generation context that handles game setup, RNG, and failure tracking.
struct GenerationContext<'a> {
    rng: StdRng,
    args: &'a GenerateArgs,
    configs: &'a [AgentConfig],
    consecutive_failures: usize,
    games_completed: usize,
}

impl<'a> GenerationContext<'a> {
    fn new(args: &'a GenerateArgs, configs: &'a [AgentConfig]) -> Self {
        let rng = seeded_rng(args.seed);
        Self {
            rng,
            args,
            configs,
            consecutive_failures: 0,
            games_completed: 0,
        }
    }

    fn is_done(&self) -> bool {
        self.args.num_games > 0 && self.games_completed >= self.args.num_games
    }

    /// Attempt to set up the next game. Returns `Ok(Some(setup))` on success,
    /// `Ok(None)` if setup failed but can retry, or `Err` if too many failures.
    fn next_game(&mut self) -> Result<Option<GameSetup>, GenerateError> {
        let num_players = self
            .rng
            .random_range(self.args.min_players..=self.args.max_players);
        let stacks: Vec<f32> = (0..num_players)
            .map(|_| {
                self.rng
                    .random_range(self.args.min_stack_bb..=self.args.max_stack_bb)
                    * self.args.big_blind
            })
            .collect();
        let dealer_idx = self.rng.random_range(0..num_players);
        let selected_configs: Vec<&AgentConfig> = (0..num_players)
            .map(|_| &self.configs[self.rng.random_range(0..self.configs.len())])
            .collect();

        let game_state = match GameStateBuilder::new()
            .stacks(stacks)
            .big_blind(self.args.big_blind)
            .small_blind(self.args.small_blind)
            .dealer_idx(dealer_idx)
            .build()
        {
            Ok(gs) => gs,
            Err(e) => {
                warn!("Failed to build game state: {}", e);
                return self.record_failure("game state build");
            }
        };

        let cfr_context = AgentConfig::maybe_shared_cfr_context(
            selected_configs.iter().copied(),
            &game_state,
            num_players,
        );

        let agents_result: Result<Vec<_>, AgentConfigError> = selected_configs
            .iter()
            .enumerate()
            .map(|(idx, config)| {
                let mut builder = ConfigAgentBuilder::new((*config).clone())?.player_idx(idx);
                // Inject shared CFR context BEFORE game_state to avoid
                // wasted eager allocation in game_state()
                if let Some((ref cfr_state, ref ts)) = cfr_context {
                    builder = builder.cfr_context(cfr_state.clone(), ts.clone());
                }
                builder = builder.game_state(game_state.clone());
                builder = builder.rng_seed(self.rng.random::<u64>());
                Ok(builder.build())
            })
            .collect();

        let agents = match agents_result {
            Ok(a) => a,
            Err(e) => {
                warn!("Failed to build agents: {}", e);
                return self.record_failure("agent build");
            }
        };

        Ok(Some(GameSetup {
            game_state,
            agents,
            cfr_context,
            num_players,
        }))
    }

    fn record_success(&mut self) {
        self.games_completed += 1;
        self.consecutive_failures = 0;
    }

    fn record_failure(&mut self, _phase: &str) -> Result<Option<GameSetup>, GenerateError> {
        self.consecutive_failures += 1;
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
            return Err(GenerateError::TooManyFailures(self.consecutive_failures));
        }
        Ok(None)
    }
}

/// Per-game work produced by a generation loop's setup hook.
///
/// Splitting the work in two lets the shared loop own the simulation lifecycle
/// (build, run, drop the CFR tree) while each caller controls the historians
/// it wants and what happens once the game has completed.
struct GamePlan {
    /// Historians to attach to the simulation for this game.
    historians: Vec<Box<dyn Historian>>,
    /// Runs after the simulation has been dropped and the success has been
    /// recorded; receives the running count of completed games. Returns
    /// `Break` to stop the loop cleanly (e.g. a downstream consumer has gone
    /// away).
    on_complete: Box<dyn FnOnce(usize) -> std::ops::ControlFlow<()> + Send>,
}

/// Shared generation loop driving setup → build → run → completion.
///
/// `setup_game` is invoked for each game to produce its historians and a
/// completion hook; the loop owns the simulation lifecycle and failure
/// tracking so callers stay focused on their own per-game concerns. When
/// `warn_on_build_failure` is set, simulation build errors are logged (the TUI
/// path suppresses these to avoid corrupting the terminal). Returns the number
/// of games completed.
async fn run_generation_loop<F>(
    args: &GenerateArgs,
    configs: &[AgentConfig],
    warn_on_build_failure: bool,
    mut setup_game: F,
) -> Result<usize, GenerateError>
where
    F: FnMut(&GameSetup) -> GamePlan,
{
    let mut ctx = GenerationContext::new(args, configs);

    loop {
        if ctx.is_done() {
            break;
        }

        let setup = match ctx.next_game()? {
            Some(s) => s,
            None => continue,
        };

        let GamePlan {
            historians,
            on_complete,
        } = setup_game(&setup);

        let sub_rng = StdRng::from_rng(&mut ctx.rng);
        let sim_result = {
            let mut builder = HoldemSimulationBuilder::default()
                .game_state(setup.game_state)
                .agents(setup.agents)
                .historians(historians);
            if let Some((cfr_state, traversal_set)) = setup.cfr_context {
                builder = builder.cfr_context(cfr_state, traversal_set, true);
            }
            builder.build_with_rng(sub_rng)
        };

        let mut sim = match sim_result {
            Ok(s) => s,
            Err(e) => {
                if warn_on_build_failure {
                    warn!("Failed to build simulation: {}", e);
                }
                ctx.record_failure("sim build")?;
                continue;
            }
        };

        sim.run().await;

        // Drop the simulation immediately to free the CFR tree (~19GB for 3
        // players) before the completion hook snapshots stats or blocks on a
        // channel send. Without this, the tree stays alive through any
        // potentially-blocking work, keeping peak RSS ~2x higher than needed.
        drop(sim);

        ctx.record_success();

        if on_complete(ctx.games_completed).is_break() {
            return Ok(ctx.games_completed);
        }
    }

    Ok(ctx.games_completed)
}

async fn run_generation(args: &GenerateArgs, configs: &[AgentConfig]) -> Result<(), GenerateError> {
    let report_interval = if args.num_games == 0 {
        10
    } else {
        (args.num_games / 10).max(1)
    };

    let total = run_generation_loop(args, configs, true, |_setup| GamePlan {
        historians: vec![Box::new(OpenHandHistoryHistorian::new(args.output.clone()))],
        on_complete: Box::new(move |games_completed| {
            if games_completed.is_multiple_of(report_interval) {
                info!("Generated {} hands...", games_completed);
            }
            std::ops::ControlFlow::Continue(())
        }),
    })
    .await?;

    info!("Done. Generated {} hands total.", total);
    Ok(())
}

/// Run the generation loop as a background tokio task, sending GameResults
/// over a channel.
///
/// The generation work itself is spawned as a nested task so that a panic in
/// CFR exploration surfaces as a `JoinError` here (reported as `SimError::Panic`)
/// rather than aborting the process.
async fn run_generation_background(
    args: GenerateArgs,
    configs: Vec<AgentConfig>,
    tx: std::sync::mpsc::SyncSender<SimMessage<GameResult>>,
    hand_store: HandStore,
) {
    let inner_tx = tx.clone();
    let join =
        tokio::spawn(
            async move { run_generation_inner(&args, &configs, &inner_tx, &hand_store).await },
        )
        .await;

    match join {
        Ok(Ok(())) => {
            let _ = tx.send(SimMessage::Completed);
        }
        Ok(Err(GenerateError::TooManyFailures(n))) => {
            let _ = tx.send(SimMessage::Error(SimError::TooManyFailures {
                consecutive_failures: n,
            }));
        }
        Ok(Err(_)) | Err(_) => {
            let _ = tx.send(SimMessage::Error(SimError::Panic));
        }
    }
}

async fn run_generation_inner(
    args: &GenerateArgs,
    configs: &[AgentConfig],
    tx: &std::sync::mpsc::SyncSender<SimMessage<GameResult>>,
    hand_store: &HandStore,
) -> Result<(), GenerateError> {
    run_generation_loop(args, configs, false, |setup| {
        let num_players = setup.num_players;
        let agent_names: Vec<String> = setup.agents.iter().map(|a| a.name().to_string()).collect();

        // Stat the OHH file to get the byte offset before writing
        let pre_offset = std::fs::metadata(&args.output)
            .map(|m| m.len())
            .unwrap_or(0);

        let ohh_historian = OpenHandHistoryHistorian::new(args.output.clone());
        let stats_storage = SharedStatsStorage::new(num_players);
        let stats_historian = stats_storage.historian();

        let big_blind = args.big_blind;
        let hand_store = hand_store.clone();
        let tx = tx.clone();

        GamePlan {
            historians: vec![Box::new(ohh_historian), Box::new(stats_historian)],
            on_complete: Box::new(move |_games_completed| {
                // Record the byte offset so the TUI can fetch this hand on demand
                hand_store.push_offset(pre_offset);

                let stats_snap = stats_storage.snapshot();
                let ending_round = ending_round_from_stats(&stats_snap, num_players);
                let profits: Vec<f32> = (0..num_players)
                    .map(|i| stats_snap.total_profit[i])
                    .collect();
                let seat_stats: Vec<SeatStats> = (0..num_players)
                    .map(|i| SeatStats::from_storage(&stats_snap, i))
                    .collect();
                drop(stats_snap);

                let game_result = GameResult {
                    agent_names,
                    profits,
                    ending_round,
                    seat_stats,
                    big_blind,
                };

                // If send fails, the TUI has quit - exit the loop cleanly
                if tx.send(SimMessage::GameResult(game_result)).is_err() {
                    std::ops::ControlFlow::Break(())
                } else {
                    std::ops::ControlFlow::Continue(())
                }
            }),
        }
    })
    .await?;

    Ok(())
}

/// Run generation with the TUI dashboard.
///
/// The generation work runs as a tokio task on the multi-thread runtime (whose
/// worker threads carry a large stack for deep CFR recursion), sending results
/// over a sync channel. The blocking ratatui render loop runs concurrently on a
/// dedicated blocking thread via `spawn_blocking`.
async fn run_generation_with_tui(
    args: GenerateArgs,
    configs: Vec<AgentConfig>,
) -> Result<(), GenerateError> {
    let (tx, rx) = std::sync::mpsc::sync_channel::<SimMessage<GameResult>>(1024);

    let games_target = if args.num_games > 0 {
        Some(args.num_games)
    } else {
        None
    };

    let hand_store = HandStore::new(args.output.clone());

    // Spawn the generation as a background tokio task. CFR exploration recurses
    // deeply; the runtime's worker threads are built with a large stack (see
    // `main`) so this no longer needs a hand-rolled OS thread.
    let bg_args = args.clone();
    let bg_configs = configs.clone();
    let bg_hand_store = hand_store.clone();
    let bg_handle = tokio::spawn(async move {
        run_generation_background(bg_args, bg_configs, tx, bg_hand_store).await;
    });

    // The TUI render loop is blocking (crossterm poll + terminal draw); the
    // shared helper runs it on a blocking thread and joins the background task.
    run_blocking_tui_loop(
        move || {
            let handler = EventHandler::new(rx, std::time::Duration::from_millis(33));
            let mut tui_app = App::new(games_target);
            tui_app.hand_store = hand_store;
            app::run_app(&mut tui_app, &handler)
        },
        bg_handle,
        || {},
    )
    .await?;

    Ok(())
}

pub async fn run(
    args: GenerateArgs,
    default_budget: &rs_poker::arena::cfr::BudgetConfig,
) -> Result<(), GenerateError> {
    let mut configs = load_configs(&args.agents_dir)?;
    if configs.is_empty() {
        return Err(GenerateError::NoConfigs(
            args.agents_dir.display().to_string(),
        ));
    }
    info!("Loaded {} agent config(s)", configs.len());

    // Fill the binary's default budget on every CFR config that doesn't
    // already have one. Explicit per-config budgets are preserved.
    crate::budget::override_budgets(&mut configs, default_budget);

    args.validate()?;

    if args.tui.should_use_tui() {
        run_generation_with_tui(args, configs).await
    } else {
        run_generation(&args, &configs).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::state::RoundLabel;
    use rs_poker::arena::historian::StatsStorage;

    fn make_stats(num_players: usize) -> StatsStorage {
        StatsStorage::new_with_num_players(num_players)
    }

    #[test]
    fn test_ending_round_preflop() {
        let stats = make_stats(2);
        // No completion counters set => Preflop
        assert_eq!(ending_round_from_stats(&stats, 2), RoundLabel::Preflop);
    }

    #[test]
    fn test_ending_round_flop() {
        let mut stats = make_stats(2);
        stats.flop_completes[0] = 1;
        assert_eq!(ending_round_from_stats(&stats, 2), RoundLabel::Flop);
    }

    #[test]
    fn test_ending_round_turn() {
        let mut stats = make_stats(2);
        stats.flop_completes[0] = 1;
        stats.turn_completes[1] = 1;
        assert_eq!(ending_round_from_stats(&stats, 2), RoundLabel::Turn);
    }

    #[test]
    fn test_ending_round_river() {
        let mut stats = make_stats(2);
        stats.flop_completes[0] = 1;
        stats.turn_completes[0] = 1;
        stats.river_completes[0] = 1;
        assert_eq!(ending_round_from_stats(&stats, 2), RoundLabel::River);
    }

    #[test]
    fn test_ending_round_showdown() {
        let mut stats = make_stats(2);
        stats.showdown_count[0] = 1;
        assert_eq!(ending_round_from_stats(&stats, 2), RoundLabel::Showdown);
    }

    #[test]
    fn test_ending_round_ignores_players_beyond_num() {
        let mut stats = make_stats(3);
        // Only player at index 2 has showdown, but we only check 2 players
        stats.showdown_count[2] = 1;
        assert_eq!(ending_round_from_stats(&stats, 2), RoundLabel::Preflop);
    }
}
