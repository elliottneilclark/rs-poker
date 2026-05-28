#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Configure jemalloc to purge freed pages after 1 second instead of the
/// default 10 seconds.
///
/// CFR solvers allocate ~17GB per game. With the default 10-second dirty page
/// decay, the old tree's pages can overlap with the new allocation, spiking
/// peak RSS to ~34GB. A 1-second decay is fast enough to reclaim pages between
/// games (3-player CFR games take several seconds) while avoiding the syscall
/// overhead of immediate purging (decay_ms:0).
#[cfg(not(target_env = "msvc"))]
#[allow(non_upper_case_globals)]
#[unsafe(no_mangle)]
pub static malloc_conf: &[u8; 40] = b"dirty_decay_ms:1000,muzzy_decay_ms:1000\0";

mod arena;
mod budget;
mod common;
mod holdem;
mod icm;
mod ohh;
mod omaha;
mod tui;

use clap::{Parser, Subcommand};
use common::TracingArgs;

#[derive(Parser)]
#[command(name = "rsp", about = "A poker toolkit")]
struct Cli {
    #[command(flatten)]
    tracing: TracingArgs,

    /// Budget for CFR agents — path to a BudgetConfig JSON file, or
    /// inline JSON (starting with `[` or `{`). Falls back to RSP_BUDGET
    /// env var, then to the binary's operational default. Per-config
    /// budgets explicitly set in JSON are NOT overwritten.
    #[arg(long, global = true)]
    budget: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Texas Hold'em tools
    Holdem(holdem::HoldemArgs),
    /// Multi-agent simulation tools
    Arena(arena::ArenaArgs),
    /// Omaha poker tools
    Omaha(omaha::OmahaArgs),
    /// ICM calculators
    Icm(icm::IcmArgs),
    /// Open Hand History tools
    Ohh(ohh::OhhArgs),
}

#[derive(Debug, thiserror::Error)]
enum CliError {
    #[error(transparent)]
    Holdem(#[from] holdem::HoldemError),
    #[error(transparent)]
    Arena(#[from] arena::ArenaError),
    #[error(transparent)]
    Omaha(#[from] omaha::OmahaError),
    #[error(transparent)]
    Icm(#[from] icm::MatusowMeltdown),
    #[error(transparent)]
    Ohh(#[from] ohh::OhhError),
    #[error(transparent)]
    Budget(#[from] budget::BudgetError),
}

/// Worker-thread stack size for the tokio runtime.
///
/// CFR exploration spawns subtree walkers as tokio tasks (and recurses inline
/// when the in-flight limiter is saturated). With 3+ players this recursion is
/// deep enough to overflow tokio's default 2 MB worker stack, so we match the
/// 47 MB stack the binary's main thread already gets from the linker
/// (`-Wl,-zstack-size`, see `.cargo/config.toml`).
const WORKER_STACK_SIZE: usize = 47 * 1024 * 1024;

fn main() -> Result<(), CliError> {
    // Build the multi-thread runtime by hand rather than via `#[tokio::main]`
    // so we can give the worker threads a large stack for deep CFR recursion.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(WORKER_STACK_SIZE)
        .build()
        .expect("failed to build tokio runtime");

    runtime.block_on(async_main())
}

async fn async_main() -> Result<(), CliError> {
    let cli = Cli::parse();
    cli.tracing.init_tracing();

    match cli.command {
        Commands::Holdem(args) => holdem::run(args)?,
        Commands::Arena(args) => {
            let default_budget = budget::effective_budget(cli.budget.as_deref())?;
            arena::run(args, default_budget).await?
        }
        Commands::Omaha(args) => omaha::run(args)?,
        Commands::Icm(args) => icm::run(args).await?,
        Commands::Ohh(args) => ohh::run(args)?,
    }
    Ok(())
}
