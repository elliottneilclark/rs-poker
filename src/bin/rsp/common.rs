use tracing_subscriber::{EnvFilter, fmt, prelude::*};

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
    ///    - `-vv` (and above): trace
    ///
    /// # Panics
    ///
    /// Panics if the subscriber has already been set.
    pub fn init_tracing(&self) {
        // Derive the human filter as a string first so we can rebuild a
        // fresh EnvFilter inside each match arm (EnvFilter isn't Clone,
        // and each fmt layer takes ownership of its own filter).
        let filter_string = if let Ok(rust_log) = std::env::var("RUST_LOG") {
            rust_log
        } else {
            let level = if self.quiet {
                "warn"
            } else {
                match self.verbosity {
                    0 => "info",
                    1 => "debug",
                    _ => "trace",
                }
            };
            format!("{level},rs_poker={level},rsp={level}")
        };
        let make_human_filter = || {
            EnvFilter::new(&filter_string)
                .add_directive("cfr_diag=off".parse().expect("static directive parses"))
        };

        // Diagnostic JSON layer: off unless RSP_DIAG_LOG is set. Writes to
        // stderr as JSON. Independently controllable from RUST_LOG so turning
        // the diag sink on doesn't accidentally enable trace-level chatter
        // from unrelated targets.
        //
        // The layer is constructed inside each match arm because tracing-
        // subscriber's Layered types are not type-erased — the concrete
        // subscriber type S must be consistent within each arm.
        let make_diag_filter = || match EnvFilter::try_from_env("RSP_DIAG_LOG") {
            Ok(f) => f,
            Err(e) if std::env::var("RSP_DIAG_LOG").is_ok() => {
                // Var is set but unparseable — warn so the user catches the typo
                // instead of silently getting an empty JSONL file.
                eprintln!("rsp: ignoring invalid RSP_DIAG_LOG: {e}");
                EnvFilter::new("off")
            }
            Err(_) => EnvFilter::new("off"),
        };

        match self.log_format {
            LogFormat::Compact => {
                tracing_subscriber::registry()
                    .with(fmt::layer().compact().with_filter(make_human_filter()))
                    .with(
                        fmt::layer()
                            .json()
                            .with_writer(std::io::stderr)
                            .with_filter(make_diag_filter()),
                    )
                    .init();
            }
            LogFormat::Pretty => {
                tracing_subscriber::registry()
                    .with(fmt::layer().pretty().with_filter(make_human_filter()))
                    .with(
                        fmt::layer()
                            .json()
                            .with_writer(std::io::stderr)
                            .with_filter(make_diag_filter()),
                    )
                    .init();
            }
            LogFormat::Json => {
                tracing_subscriber::registry()
                    .with(fmt::layer().json().with_filter(make_human_filter()))
                    .with(
                        fmt::layer()
                            .json()
                            .with_writer(std::io::stderr)
                            .with_filter(make_diag_filter()),
                    )
                    .init();
            }
        }
    }
}
