extern crate rs_poker;

mod common;

use clap::Parser;
use rs_poker::arena::comparison::{ComparisonBuilder, ComparisonError};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "agent_comparison",
    about = "Compare poker agents across all possible matchups and positions",
    long_about = "Evaluates poker agents by running all permutations of seat arrangements,\n\
                  tracking detailed per-agent statistics to determine which agents perform best."
)]
struct Args {
    /// Tracing/logging options
    #[command(flatten)]
    tracing: common::TracingArgs,

    /// Directory containing agent JSON config files
    agents_dir: PathBuf,

    /// Number of unique game states to test
    #[arg(short = 'n', long = "num-games", default_value_t = 1000)]
    num_games: usize,

    /// Number of players per table (must be >= 2 and <= number of agents)
    #[arg(short = 'p', long = "players", default_value_t = 3)]
    players_per_table: usize,

    /// Big blind amount
    #[arg(long = "big-blind", default_value_t = 10.0)]
    big_blind: f32,

    /// Small blind amount
    #[arg(long = "small-blind", default_value_t = 5.0)]
    small_blind: f32,

    /// Minimum starting stack in big blinds
    #[arg(long = "min-stack-bb", default_value_t = 100.0)]
    min_stack_bb: f32,

    /// Maximum starting stack in big blinds
    #[arg(long = "max-stack-bb", default_value_t = 100.0)]
    max_stack_bb: f32,

    /// Optional directory to save game history and results
    #[arg(short = 'o', long = "output-dir")]
    output_dir: Option<PathBuf>,

    /// Optional random seed for reproducibility
    #[arg(short = 's', long = "seed")]
    seed: Option<u64>,
}

fn main() -> Result<(), ComparisonError> {
    let args = Args::parse();
    args.tracing.init_tracing();

    // Build the comparison using the builder pattern
    let mut builder = ComparisonBuilder::new()
        .num_games(args.num_games)
        .players_per_table(args.players_per_table)
        .big_blind(args.big_blind)
        .small_blind(args.small_blind)
        .min_stack_bb(args.min_stack_bb)
        .max_stack_bb(args.max_stack_bb)
        .load_agents_from_dir(&args.agents_dir)?;

    // Add optional configuration
    if let Some(seed) = args.seed {
        builder = builder.seed(seed);
    }

    if let Some(ref output_dir) = args.output_dir {
        builder = builder.output_dir(output_dir);
    }

    let comparison = builder.build()?;

    // Print configuration summary
    comparison.print_configuration_summary();

    // Run simulations
    println!("Starting simulations...");
    let result = comparison.run()?;
    println!("\nCompleted all {} game states!", result.config().num_games);

    // Print results
    println!("{}", result.to_markdown());

    // Save to files if output directory specified
    if let Some(ref output_dir) = args.output_dir {
        result.save_to_dir(output_dir)?;
        println!("Results saved to:");
        println!("  - {}", output_dir.join("results.json").display());
        println!("  - {}", output_dir.join("results.md").display());
        println!("  - {}", output_dir.join("hands.jsonl").display());
    }

    Ok(())
}
