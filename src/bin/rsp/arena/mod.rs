use clap::{Args, Subcommand};

use crate::tui::TuiFlags;

pub mod compare;
pub mod generate;

#[derive(Args)]
pub struct ArenaArgs {
    #[command(subcommand)]
    command: ArenaCommand,
}

#[derive(Subcommand)]
enum ArenaCommand {
    /// Compare poker agents across all possible matchups
    Compare(compare::CompareArgs),
    /// Generate Open Hand History files from poker simulations
    Generate(generate::GenerateArgs),
}

#[derive(Debug, thiserror::Error)]
pub enum ArenaError {
    #[error(transparent)]
    Compare(#[from] compare::CompareError),
    #[error(transparent)]
    Generate(#[from] generate::GenerateError),
}

pub fn run(args: ArenaArgs, tui_flags: &TuiFlags) -> Result<(), ArenaError> {
    match args.command {
        ArenaCommand::Compare(a) => compare::run(a, tui_flags)?,
        ArenaCommand::Generate(a) => generate::run(a, tui_flags)?,
    }
    Ok(())
}
