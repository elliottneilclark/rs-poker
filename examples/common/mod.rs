//! Shared utilities for rs-poker examples.
//!
//! This module provides common functionality used across example programs,
//! including tracing/logging setup with CLI argument support.

use tracing_subscriber::{EnvFilter, fmt, prelude::*};

/// CLI arguments for controlling tracing/logging output.
///
/// These can be embedded into any example's CLI using `#[command(flatten)]`.
///
/// # Examples
///
/// ```ignore
/// use clap::Parser;
/// use common::TracingArgs;
///
/// #[derive(Parser)]
/// struct Args {
///     #[command(flatten)]
///     tracing: TracingArgs,
///     // ... other args
/// }
/// ```
#[derive(clap::Args, Debug, Clone)]
pub struct TracingArgs {
    /// Increase logging verbosity (can be repeated: -v, -vv, -vvv)
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count, global = true)]
    pub verbosity: u8,

    /// Suppress all output except warnings and errors
    #[arg(short = 'q', long = "quiet", global = true)]
    pub quiet: bool,

    /// Log output format: compact, pretty, or json
    #[arg(long = "log-format", default_value = "compact", global = true)]
    pub log_format: LogFormat,
}

/// Available log output formats.
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum LogFormat {
    /// Compact single-line format (default)
    #[default]
    Compact,
    /// Pretty multi-line format with colors
    Pretty,
    /// JSON format for machine parsing
    Json,
}

impl TracingArgs {
    /// Initialize the tracing subscriber based on CLI arguments.
    ///
    /// Priority:
    /// 1. If `RUST_LOG` environment variable is set, use it
    /// 2. Otherwise, use verbosity flags to determine level:
    ///    - `-q` (quiet): warn
    ///    - (default): info
    ///    - `-v`: debug
    ///    - `-vv`: trace
    ///    - `-vvv`: trace with all spans
    ///
    /// # Panics
    ///
    /// Panics if the subscriber has already been set.
    pub fn init_tracing(&self) {
        // Check if RUST_LOG is set - if so, use it directly
        let filter = if std::env::var("RUST_LOG").is_ok() {
            EnvFilter::from_default_env()
        } else {
            // Build filter from CLI args
            let level = if self.quiet {
                "warn"
            } else {
                match self.verbosity {
                    0 => "info",
                    1 => "debug",
                    _ => "trace",
                }
            };

            // Build a filter that applies the level to rs_poker and the example
            EnvFilter::new(format!("{level},rs_poker={level}"))
        };

        // Build the subscriber with the appropriate format
        match self.log_format {
            LogFormat::Compact => {
                tracing_subscriber::registry()
                    .with(filter)
                    .with(fmt::layer().compact())
                    .init();
            }
            LogFormat::Pretty => {
                tracing_subscriber::registry()
                    .with(filter)
                    .with(fmt::layer().pretty())
                    .init();
            }
            LogFormat::Json => {
                tracing_subscriber::registry()
                    .with(filter)
                    .with(fmt::layer().json())
                    .init();
            }
        }
    }
}

/// Initialize tracing with default settings (info level, compact format).
///
/// This is a convenience function for examples that don't use clap or
/// don't need CLI control over logging.
///
/// Respects `RUST_LOG` environment variable if set.
#[allow(dead_code)]
pub fn init_tracing_default() {
    let args = TracingArgs {
        verbosity: 0,
        quiet: false,
        log_format: LogFormat::Compact,
    };
    args.init_tracing();
}

/// Initialize tracing from environment only.
///
/// Uses `RUST_LOG` if set, otherwise defaults to info level.
/// Uses compact format.
#[allow(dead_code)]
pub fn init_tracing_from_env() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().compact())
        .init();
}
