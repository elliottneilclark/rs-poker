use super::{action::Action, GameState};

/// HistorianError is the error type for historian implementations.
#[derive(Error, Debug)]
pub enum HistorianError {
    #[error("Unable to record action")]
    UnableToRecordAction,
    #[error("IO Error: {0}")]
    IOError(#[from] std::io::Error),
    #[error("Borrow Mut Error: {0}")]
    BorrowMutError(#[from] std::cell::BorrowMutError),
    #[error("Borrow Error: {0}")]
    BorrowError(#[from] std::cell::BorrowError),
}

/// Historians are a way for the simulation to record or notify of
/// actions while the game is progressing. This is useful for
/// logging, debugging, or even for implementing a replay system.
/// However it's also useful for CFR+ as each action
/// moves the game along the nodes.
pub trait Historian {
    /// This method is called by the simulation when an action is received.
    ///
    /// # Arguments
    /// - `id` - The id of the simulation that the action was received on.
    /// - `game_state` - The game state after the action was played
    /// - `action` - The action that was played
    ///
    /// # Returns
    /// - `Ok(())` if the action was recorded successfully
    /// - `Err(HistorianError)` if there was an error recording the action.
    ///
    /// Returning an error will cause the historian to be dropped from the
    /// `Simulation`.
    fn record_action(
        &mut self,
        id: &uuid::Uuid,
        game_state: &GameState,
        action: Action,
    ) -> Result<(), HistorianError>;
}

mod fn_historian;
mod vec;

pub use fn_historian::FnHistorian;
use thiserror::Error;
pub use vec::VecHistorian;
