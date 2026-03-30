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
mod common;
mod holdem;
mod icm;
mod ohh;
mod omaha;
mod tui;

use clap::{Parser, Subcommand};
use common::TracingArgs;
use tui::TuiFlags;

#[derive(Parser)]
#[command(name = "rsp", about = "A poker toolkit")]
struct Cli {
    #[command(flatten)]
    tracing: TracingArgs,

    #[command(flatten)]
    tui: TuiFlags,

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
}

fn main() -> Result<(), CliError> {
    let cli = Cli::parse();
    cli.tracing.init_tracing();

    match cli.command {
        Commands::Holdem(args) => holdem::run(args)?,
        Commands::Arena(args) => arena::run(args, &cli.tui)?,
        Commands::Omaha(args) => omaha::run(args)?,
        Commands::Icm(args) => icm::run(args)?,
        Commands::Ohh(args) => ohh::run(args, &cli.tui)?,
    }
    Ok(())
}
