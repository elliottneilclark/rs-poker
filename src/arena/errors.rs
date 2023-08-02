use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum GameStateError {
    #[error("Invalid number for a bet")]
    BetInvalidSize,
    #[error("The ammount bet doesn't call the previous bet")]
    BetSizeDoesntCall,
    #[error("The ammount bet doesn't call our own previous bet")]
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
}
