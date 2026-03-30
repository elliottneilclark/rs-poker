#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

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
