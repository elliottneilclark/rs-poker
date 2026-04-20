use clap::{Args, Subcommand};

use crate::tui::TuiFlags;

pub mod charts;
pub mod compare;
pub mod generate;
pub mod verify;

#[derive(Args)]
pub struct ArenaArgs {
    #[command(subcommand)]
    command: ArenaCommand,
}

#[derive(Subcommand)]
enum ArenaCommand {
    /// View preflop charts for an agent config in an interactive TUI
    Charts(charts::ChartsArgs),
    /// Compare poker agents across all possible matchups
    Compare(compare::CompareArgs),
    /// Generate Open Hand History files from poker simulations
    Generate(generate::GenerateArgs),
    /// Verify agent config files load correctly
    Verify(verify::VerifyArgs),
}

#[derive(Debug, thiserror::Error)]
pub enum ArenaError {
    #[error(transparent)]
    Charts(#[from] charts::ChartsError),
    #[error(transparent)]
    Compare(#[from] compare::CompareError),
    #[error(transparent)]
    Generate(#[from] generate::GenerateError),
    #[error(transparent)]
    Verify(#[from] verify::VerifyError),
}

pub fn run(args: ArenaArgs, tui_flags: &TuiFlags) -> Result<(), ArenaError> {
    match args.command {
        ArenaCommand::Charts(a) => charts::run(a)?,
        ArenaCommand::Compare(a) => compare::run(a, tui_flags)?,
        ArenaCommand::Generate(a) => generate::run(a, tui_flags)?,
        ArenaCommand::Verify(a) => verify::run(a)?,
    }
    Ok(())
}
