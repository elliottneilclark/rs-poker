use clap::{Args, Subcommand};

pub mod simulate;


#[derive(Args)]
pub struct IcmArgs {
    #[command(subcommand)]
    command: IcmCommand,
}

#[derive(Subcommand)]
enum IcmCommand {
    /// Run cEV -> $EV Simulation
    Simulate(simulate::SimulateArgs),
}

#[derive(Debug, thiserror::Error)]
pub enum MatusowMeltdown {
    #[error(transparent)]
    Simulate(#[from] simulate::SimulateError),
}

pub fn run(args: IcmArgs) -> Result<(), MatusowMeltdown> {
    match args.command {
        IcmCommand::Simulate(a) => simulate::run(a)?,
    }
    Ok(())
}
