use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use clap::Args;
use rs_poker::arena::comparison::{ArenaComparison, ComparisonBuilder, PermutationResult};

use crate::tui::TuiFlags;
use crate::tui::app::{self, App};
use crate::tui::event::{EventHandler, SimError, SimMessage};
use crate::tui::hand_store::HandStore;
use crate::tui::state::{GameResult, SeatStats, ending_round_from_stats};

#[derive(Debug, thiserror::Error)]
pub enum CompareError {
    #[error(transparent)]
    Comparison(#[from] rs_poker::arena::comparison::ComparisonError),
    #[error("failed to create thread pool: {0}")]
    ThreadPool(#[from] rayon::ThreadPoolBuildError),
    #[error("TUI error: {0}")]
    TuiError(#[from] std::io::Error),
}

#[derive(Args, Debug)]
#[command(
    about = "Compare poker agents across all possible matchups and positions",
    long_about = "Evaluates poker agents by running all permutations of seat arrangements,\n\
                  tracking detailed per-agent statistics to determine which agents perform best."
)]
pub struct CompareArgs {
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

    /// Number of threads for parallel CFR action exploration.
    /// When omitted, CFR agents run sequentially.
    #[arg(long)]
    parallel: Option<usize>,

    #[command(flatten)]
    tui: TuiFlags,
}

fn build_comparison(args: &CompareArgs) -> Result<ArenaComparison, CompareError> {
    let mut builder = ComparisonBuilder::new()
        .num_games(args.num_games)
        .players_per_table(args.players_per_table)
        .big_blind(args.big_blind)
        .small_blind(args.small_blind)
        .min_stack_bb(args.min_stack_bb)
        .max_stack_bb(args.max_stack_bb)
        .load_agents_from_dir(&args.agents_dir)?;

    if let Some(seed) = args.seed {
        builder = builder.seed(seed);
    }

    if let Some(ref output_dir) = args.output_dir {
        builder = builder.output_dir(output_dir);
    }

    if let Some(num_threads) = args.parallel {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()?,
        );
        builder = builder.thread_pool(pool);
    }

    Ok(builder.build()?)
}

/// Convert a PermutationResult into a GameResult for the TUI.
fn perm_to_game_result(perm: PermutationResult, big_blind: f32) -> GameResult {
    let num_players = perm.agent_names.len();
    let ending_round = ending_round_from_stats(&perm.stats, num_players);
    let profits: Vec<f32> = (0..num_players)
        .map(|i| perm.stats.total_profit[i])
        .collect();
    let seat_stats: Vec<SeatStats> = (0..num_players)
        .map(|i| SeatStats::from_storage(&perm.stats, i))
        .collect();
    // perm.stats (with its 40+ Vecs) is dropped here, on the comparison thread

    GameResult {
        agent_names: perm.agent_names,
        profits,
        ending_round,
        seat_stats,
        big_blind,
    }
}

/// Run the comparison in a background thread, sending results over a channel.
///
/// `cancel` is a shared cancellation flag. The callback checks it after
/// each permutation and returns `Break` to stop the comparison early,
/// so the worker stops allocating CFR trees as soon as the TUI quits.
fn run_comparison_background(
    comparison: ArenaComparison,
    tx: std::sync::mpsc::SyncSender<SimMessage<GameResult>>,
    hand_store: HandStore,
    ohh_path: Option<PathBuf>,
    big_blind: f32,
    cancel: Arc<AtomicBool>,
) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut prev_file_size: u64 = 0;
        comparison.run_with_cancellable_callback(|perm| {
            // Track byte offsets for on-demand hand loading
            if let Some(ref path) = ohh_path {
                let current_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                if current_size > prev_file_size {
                    hand_store.push_offset(prev_file_size);
                }
                prev_file_size = current_size;
            }

            let game_result = perm_to_game_result(perm, big_blind);
            // If send fails the TUI has quit — mark cancelled so we stop
            // as soon as the current permutation ends rather than
            // allocating the next CFR tree.
            if tx.send(SimMessage::GameResult(game_result)).is_err() {
                cancel.store(true, Ordering::Release);
            }

            if cancel.load(Ordering::Acquire) {
                std::ops::ControlFlow::Break(())
            } else {
                std::ops::ControlFlow::Continue(())
            }
        })
    }));

    match result {
        Ok(Ok(_)) => {
            let _ = tx.send(SimMessage::Completed);
        }
        Ok(Err(e)) => {
            let _ = tx.send(SimMessage::Error(SimError::ComparisonFailed { source: e }));
        }
        Err(_) => {
            let _ = tx.send(SimMessage::Error(SimError::Panic));
        }
    }
}

/// Run comparison with the TUI dashboard.
fn run_comparison_with_tui(
    comparison: ArenaComparison,
    big_blind: f32,
) -> Result<(), CompareError> {
    let total_games = comparison.total_games();

    // Extract OHH path before moving comparison into background thread
    let ohh_path = comparison
        .config()
        .output_dir
        .as_ref()
        .map(|dir| dir.join("hands.jsonl"));
    let hand_store = match ohh_path {
        Some(ref p) => HandStore::new(p.clone()),
        None => HandStore::none(),
    };

    let (tx, rx) = std::sync::mpsc::sync_channel::<SimMessage<GameResult>>(1024);

    // Shared cancellation flag: set when the TUI exits so the background
    // comparison stops allocating new CFR trees.
    let cancel = Arc::new(AtomicBool::new(false));

    // Must use a large stack to match the main binary's linker-configured stack
    // (47 MB via -Wl,-zstack-size), since CFR traversal with 3+ players recurses
    // deeply and overflows the default 8 MB thread stack.
    let bg_hand_store = hand_store.clone();
    let bg_cancel = Arc::clone(&cancel);
    const STACK_SIZE: usize = 47 * 1024 * 1024;
    let handle = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(move || {
            run_comparison_background(
                comparison,
                tx,
                bg_hand_store,
                ohh_path,
                big_blind,
                bg_cancel,
            );
        })
        .expect("failed to spawn comparison thread");

    let handler = EventHandler::new(rx, Duration::from_millis(33));
    let mut tui_app = App::new(Some(total_games));
    tui_app.hand_store = hand_store;

    let tui_result = app::run_app(&mut tui_app, &handler);

    // Whatever happened in the TUI, signal the worker to stop and wait
    // for it to exit before returning. This keeps any caller-owned temp
    // directory alive until the OHH historian is done writing to it.
    cancel.store(true, Ordering::Release);
    let _ = handle.join();

    tui_result?;
    Ok(())
}

pub fn run(mut args: CompareArgs) -> Result<(), CompareError> {
    let use_tui = args.tui.should_use_tui();

    // When using the TUI without an explicit output dir, use a temp dir
    // so OHH hands are always written and game detail view works.
    let _temp_dir;
    if args.output_dir.is_none() && use_tui {
        let tmp = tempfile::TempDir::new()?;
        args.output_dir = Some(tmp.path().to_path_buf());
        _temp_dir = Some(tmp);
    } else {
        _temp_dir = None;
    }

    let comparison = build_comparison(&args)?;

    if use_tui {
        comparison.print_configuration_summary();
        run_comparison_with_tui(comparison, args.big_blind)
    } else {
        // Print configuration summary
        comparison.print_configuration_summary();

        // Run simulations
        println!("Starting simulations...");
        let result = comparison.run()?;
        println!("\nCompleted all {} game states!", result.config().num_games);

        // Print results
        println!("{}", result.to_markdown());

        // Save to files if output directory specified
        if let Some(ref output_dir) = args.output_dir {
            result.save_to_dir(output_dir)?;
            println!("Results saved to:");
            println!("  - {}", output_dir.join("results.json").display());
            println!("  - {}", output_dir.join("results.md").display());
            println!("  - {}", output_dir.join("hands.jsonl").display());
        }

        Ok(())
    }
}
