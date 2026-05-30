use clap::{Args, Subcommand};

pub mod cfr;

/// Diagnostic tools for inspecting captured arena/CFR data.
///
/// Each subcommand consumes a different captured stream (currently only
/// `cfr_diag` JSONL) and prints a plain-text summary. Add new diagnostic
/// targets here as siblings of `cfr`.
#[derive(Args)]
pub struct DiagArgs {
    #[command(subcommand)]
    command: DiagCommand,
}

#[derive(Subcommand)]
enum DiagCommand {
    /// Summarize a captured `cfr_diag` JSONL stream (stop_cause × depth,
    /// deadline utilization, regret quantiles).
    Cfr(cfr::CfrArgs),
}

#[derive(Debug, thiserror::Error)]
pub enum DiagError {
    #[error(transparent)]
    Cfr(#[from] cfr::CfrDiagError),
}

pub fn run(args: DiagArgs) -> Result<(), DiagError> {
    match args.command {
        DiagCommand::Cfr(a) => cfr::run(a)?,
    }
    Ok(())
}
