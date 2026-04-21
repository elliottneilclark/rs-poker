pub mod anonymize;
pub mod stats;
pub mod view;

use clap::{Args, Subcommand};

#[derive(Args)]
pub struct OhhArgs {
    #[command(subcommand)]
    command: OhhCommand,
}

#[derive(Subcommand)]
enum OhhCommand {
    /// View hand history file or directory with interactive TUI
    View(view::ViewArgs),
    /// Anonymize an OHH hand history stream
    Anonymize(anonymize::AnonymizeArgs),
}

#[derive(Debug, thiserror::Error)]
pub enum OhhError {
    #[error(transparent)]
    View(#[from] view::ViewError),
    #[error(transparent)]
    Anonymize(#[from] anonymize::AnonymizeError),
}

pub fn run(args: OhhArgs) -> Result<(), OhhError> {
    match args.command {
        OhhCommand::View(a) => view::run(a)?,
        OhhCommand::Anonymize(a) => anonymize::run(a)?,
    }
    Ok(())
}
