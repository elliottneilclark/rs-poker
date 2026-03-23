use clap::{Args, Subcommand};

pub mod rank;

#[derive(Args)]
pub struct OmahaArgs {
    #[command(subcommand)]
    command: OmahaCommand,
}

#[derive(Subcommand)]
enum OmahaCommand {
    /// Evaluate Omaha hand rank (uses best 2 hole + 3 board)
    Rank(rank::RankArgs),
}

#[derive(Debug, thiserror::Error)]
pub enum OmahaError {
    #[error(transparent)]
    Rank(#[from] rank::OmahaRankError),
}

pub fn run(args: OmahaArgs) -> Result<(), OmahaError> {
    match args.command {
        OmahaCommand::Rank(a) => rank::run(a)?,
    }
    Ok(())
}
