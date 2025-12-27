use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum GameStateError {
    #[error("Invalid number for a bet")]
    BetInvalidSize,
    #[error("The amount bet doesn't call the previous bet")]
    BetSizeDoesntCall,
    #[error("The amount bet doesn't call our own previous bet")]
    BetSizeDoesntCallSelf,
    #[error("The raise is below the minimum raise size")]
    RaiseSizeTooSmall,
    #[error("Can't advance after showdown")]
    CantAdvanceRound,
}

#[derive(Error, Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum HoldemSimulationError {
    #[error("Builder needs a game state")]
    NeedGameState,

    #[error("Builder needs agents")]
    NeedAgents,

    #[error("Expected GameState to contain a winner (agent with all the money)")]
    NoWinner,
}

#[derive(Error, Debug)]
pub enum ExportError {
    #[error("Error exporting caused by IO error")]
    Io(#[from] std::io::Error),

    #[error("Invalid export format")]
    InvalidExportFormat(String),

    #[error("Failed to run dot")]
    FailedToRunDot(std::process::ExitStatus),
}

#[derive(Error, Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum CFRStateError {
    #[error("Node not found at the specified index")]
    NodeNotFound,
}

#[cfg(all(feature = "open-hand-history", feature = "arena"))]
#[derive(Error, Debug, PartialEq, Eq, Clone, Hash)]
pub enum OHHConversionError {
    #[error("Invalid street transition from {from} to {to}")]
    InvalidStreetTransition { from: String, to: String },

    #[error("Missing required data: {0}")]
    MissingData(String),

    #[error("Inconsistent state: {0}")]
    InconsistentState(String),

    #[error("Action occurred before game initialization")]
    NotInitialized,
}
