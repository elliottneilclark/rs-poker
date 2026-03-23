mod arena;
mod common;
mod holdem;
mod omaha;

use clap::{Parser, Subcommand};
use common::TracingArgs;

#[derive(Parser)]
#[command(name = "rsp", about = "A poker toolkit")]
struct Cli {
    #[command(flatten)]
    tracing: TracingArgs,

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
}

#[derive(Debug, thiserror::Error)]
enum CliError {
    #[error(transparent)]
    Holdem(#[from] holdem::HoldemError),
    #[error(transparent)]
    Arena(#[from] arena::ArenaError),
    #[error(transparent)]
    Omaha(#[from] omaha::OmahaError),
}

fn main() -> Result<(), CliError> {
    let cli = Cli::parse();
    cli.tracing.init_tracing();

    match cli.command {
        Commands::Holdem(args) => holdem::run(args)?,
        Commands::Arena(args) => arena::run(args)?,
        Commands::Omaha(args) => omaha::run(args)?,
    }
    Ok(())
}
