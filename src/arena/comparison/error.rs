use thiserror::Error;

use crate::arena::agent::AgentConfigError;

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
    InvalidConfig(#[from] AgentConfigError),

    #[error("Configuration validation error: {0}")]
    ValidationError(String),

    #[error("No agent config files found in directory: {0}")]
    NoAgentsFound(String),

    #[error("Simulation error: {0}")]
    SimulationError(String),

    #[error("Failed to serialize JSON: {0}")]
    JsonSerialize(#[from] serde_json::Error),

    #[error("Comparison not yet run - call run() first")]
    NotRun,

    #[error("Missing required configuration: {0}")]
    MissingConfig(String),
}

/// Result type for comparison operations
pub type Result<T> = std::result::Result<T, ComparisonError>;
