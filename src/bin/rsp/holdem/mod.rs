use clap::{Args, Subcommand};

pub mod outs;
pub mod rank;
pub mod simulate;

#[derive(Args)]
pub struct HoldemArgs {
    #[command(subcommand)]
    command: HoldemCommand,
}

#[derive(Subcommand)]
enum HoldemCommand {
    /// Evaluate hand rank for 5-7 cards
    Rank(rank::RankArgs),
    /// Run Monte Carlo equity simulation
    Simulate(simulate::SimulateArgs),
    /// Calculate outs and equity for Texas Hold'em hands
    Outs(outs::OutsArgs),
}

#[derive(Debug, thiserror::Error)]
pub enum HoldemError {
    #[error(transparent)]
    Rank(#[from] rank::RankError),
    #[error(transparent)]
    Simulate(#[from] simulate::SimulateError),
    #[error(transparent)]
    Outs(#[from] outs::OutsError),
}

pub fn run(args: HoldemArgs) -> Result<(), HoldemError> {
    match args.command {
        HoldemCommand::Rank(a) => rank::run(a)?,
        HoldemCommand::Simulate(a) => simulate::run(a)?,
        HoldemCommand::Outs(a) => outs::run(a)?,
    }
    Ok(())
}
