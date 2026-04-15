use thiserror::Error;

use crate::arena::agent::AgentConfigError;
use crate::arena::errors::HoldemSimulationError;
use crate::arena::historian::HistorianError;

/// Errors produced while validating a [`ComparisonConfig`].
///
/// [`ComparisonConfig`]: super::config::ComparisonConfig
#[derive(Debug, Error, PartialEq)]
pub enum ComparisonConfigError {
    #[error("players_per_table must be at least 2, got {0}")]
    PlayersPerTableTooSmall(usize),

    #[error("players_per_table ({players}) cannot exceed number of agents ({num_agents})")]
    PlayersPerTableExceedsAgents { players: usize, num_agents: usize },

    #[error("num_games must be greater than 0")]
    NumGamesZero,

    #[error("big_blind must be positive, got {0}")]
    NonPositiveBigBlind(f32),

    #[error("small_blind must be positive, got {0}")]
    NonPositiveSmallBlind(f32),

    #[error("small_blind ({small}) must be less than big_blind ({big})")]
    SmallBlindNotLessThanBigBlind { small: f32, big: f32 },

    #[error("min_stack_bb must be positive, got {0}")]
    NonPositiveMinStack(f32),

    #[error("max_stack_bb must be positive, got {0}")]
    NonPositiveMaxStack(f32),

    #[error("min_stack_bb ({min}) cannot exceed max_stack_bb ({max})")]
    MinStackExceedsMax { min: f32, max: f32 },

    #[error("ante must be non-negative, got {0}")]
    NegativeAnte(f32),
}

/// Errors that can occur during agent comparison
#[derive(Debug, Error)]
pub enum ComparisonError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to parse agent config from {path}: {source}")]
    ParseConfig {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("Failed to validate agent config: {0}")]
    InvalidAgentConfig(#[from] AgentConfigError),

    #[error("Invalid comparison configuration: {0}")]
    InvalidConfig(#[from] ComparisonConfigError),

    #[error("No agent config files found in directory: {0}")]
    NoAgentsFound(String),

    #[error("Simulation error: {0}")]
    Simulation(#[from] HoldemSimulationError),

    #[error("Historian error: {0}")]
    Historian(#[from] HistorianError),

    /// The random game-state generator ran out of game states before the
    /// comparison completed.
    #[error("Random game-state generator exhausted before comparison finished")]
    GameStateGeneratorExhausted,

    /// Reading stats from the historian's shared storage failed.
    #[error("Failed to read stats from historian storage: {reason}")]
    StatsUnavailable { reason: String },

    #[error("Failed to serialize JSON: {0}")]
    JsonSerialize(#[from] serde_json::Error),

    #[error("Comparison not yet run - call run() first")]
    NotRun,

    #[error("Missing required configuration: {0}")]
    MissingConfig(String),
}

/// Result type for comparison operations
pub type Result<T> = std::result::Result<T, ComparisonError>;
