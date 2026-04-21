//! `rsp ohh anonymize` — stream-anonymize an OHH hand-history file.
//!
//! This module is intentionally thin: it maps CLI flags onto an
//! [`AnonymizeConfig`], opens I/O streams, and delegates the actual
//! work to [`rs_poker::open_hand_history::anonymize::anonymize_stream`].
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Duration;

use clap::{Args, ValueEnum};
use rs_poker::open_hand_history::anonymize::{
    AnonymizeConfig, Anonymizer, NameStrategy, StreamError, TimeFuzzConfig, anonymize_stream,
};

/// Name-strategy choices exposed on the command line.
///
/// Mirrors [`NameStrategy`] but has its own `clap::ValueEnum` so the
/// library type doesn't take on a CLI dependency.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum NameStrategyArg {
    /// Preserve original player names.
    Keep,
    /// Random names per-hand; cross-hand identity is lost.
    PerHand,
    /// Stable names across the whole stream (default).
    Stable,
}

impl From<NameStrategyArg> for NameStrategy {
    fn from(v: NameStrategyArg) -> Self {
        match v {
            NameStrategyArg::Keep => NameStrategy::Keep,
            NameStrategyArg::PerHand => NameStrategy::PerHand,
            NameStrategyArg::Stable => NameStrategy::Stable,
        }
    }
}

/// Anonymize an Open Hand History file.
///
/// Reads a JSONL `.ohh` file (or stdin with `-`) and writes an
/// anonymized copy (or stdout with `-`). Memory usage stays O(one
/// hand) regardless of input size.
#[derive(Args, Debug)]
pub struct AnonymizeArgs {
    /// Input `.ohh` file, or `-` for stdin.
    input: PathBuf,

    /// Output `.ohh` file, or `-` for stdout.
    #[arg(short, long, default_value = "-")]
    output: PathBuf,

    /// How to replace player names.
    #[arg(long, value_enum, default_value_t = NameStrategyArg::Stable)]
    names: NameStrategyArg,

    /// Disable site-name rotation.
    #[arg(long)]
    keep_site: bool,

    /// Disable network-name rotation.
    #[arg(long)]
    keep_network: bool,

    /// Disable internal-version rotation.
    #[arg(long)]
    keep_version: bool,

    /// Disable table-name rotation.
    #[arg(long)]
    keep_tables: bool,

    /// Disable game-number / tournament-number / tournament-name
    /// rotation.
    #[arg(long)]
    keep_game_numbers: bool,

    /// Disable timestamp fuzzing entirely.
    #[arg(long)]
    keep_times: bool,

    /// Maximum absolute global time shift, in minutes.
    #[arg(long, default_value_t = 30)]
    shift_minutes: u64,

    /// Maximum absolute per-hand jitter, in seconds.
    #[arg(long, default_value_t = 5)]
    jitter_seconds: u64,

    /// Optional seed for reproducible output.
    #[arg(long)]
    seed: Option<u64>,
}

/// Errors surfaced by `rsp ohh anonymize`.
#[derive(Debug, thiserror::Error)]
pub enum AnonymizeError {
    /// Opening the input or output file failed.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    /// The underlying [`anonymize_stream`] driver returned an error.
    #[error(transparent)]
    Stream(#[from] StreamError),
}

/// Entry point invoked from [`crate::ohh::run`].
pub fn run(args: AnonymizeArgs) -> Result<(), AnonymizeError> {
    let config = build_config(&args);
    let mut anonymizer = Anonymizer::new(config);

    let input: Box<dyn BufRead> = open_input(&args.input)?;
    let mut output: Box<dyn Write> = open_output(&args.output)?;

    let count = anonymize_stream(input, &mut output, &mut anonymizer)?;
    output.flush()?;

    eprintln!("anonymized {count} hand(s)");
    Ok(())
}

/// Translate CLI flags into an [`AnonymizeConfig`].
fn build_config(args: &AnonymizeArgs) -> AnonymizeConfig {
    let time_fuzz = if args.keep_times {
        None
    } else {
        Some(TimeFuzzConfig {
            max_global_shift: Duration::from_secs(args.shift_minutes * 60),
            max_per_hand_jitter: Duration::from_secs(args.jitter_seconds),
        })
    };

    AnonymizeConfig {
        name_strategy: args.names.into(),
        name_pool: None,
        rotate_site: !args.keep_site,
        rotate_network: !args.keep_network,
        rotate_internal_version: !args.keep_version,
        rotate_table_name: !args.keep_tables,
        rotate_game_numbers: !args.keep_game_numbers,
        time_fuzz,
        seed: args.seed,
    }
}

/// Open an input path, treating `-` as stdin.
fn open_input(path: &PathBuf) -> io::Result<Box<dyn BufRead>> {
    if path.as_os_str() == "-" {
        Ok(Box::new(BufReader::new(io::stdin().lock())))
    } else {
        Ok(Box::new(BufReader::new(File::open(path)?)))
    }
}

/// Open an output path, treating `-` as stdout.
fn open_output(path: &PathBuf) -> io::Result<Box<dyn Write>> {
    if path.as_os_str() == "-" {
        Ok(Box::new(BufWriter::new(io::stdout().lock())))
    } else {
        Ok(Box::new(BufWriter::new(File::create(path)?)))
    }
}
