pub mod reader;
pub mod stats;
pub mod view;

use clap::{Args, Subcommand};

use crate::tui::TuiFlags;

#[derive(Args)]
pub struct OhhArgs {
    #[command(subcommand)]
    command: OhhCommand,
}

#[derive(Subcommand)]
enum OhhCommand {
    /// View hand history file with interactive TUI
    View(view::ViewArgs),
}

#[derive(Debug, thiserror::Error)]
pub enum OhhError {
    #[error(transparent)]
    View(#[from] view::ViewError),
}

pub fn run(args: OhhArgs, tui_flags: &TuiFlags) -> Result<(), OhhError> {
    match args.command {
        OhhCommand::View(a) => view::run(a, tui_flags)?,
    }
    Ok(())
}
