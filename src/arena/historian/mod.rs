use super::{action::Action, GameState};
use thiserror::Error;

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

/// `HistorianGenerator` is a trait that is used to build historians
/// for tournaments where each simulation needs a new historian.
pub trait HistorianGenerator {
    /// This method is called before each game to build a new historian.
    fn generate(&self, game_state: &GameState) -> Box<dyn Historian>;
}

pub trait CloneHistorian: Historian {
    fn clone_box(&self) -> Box<dyn Historian>;
}

impl<T> CloneHistorian for T
where
    T: 'static + Historian + Clone,
{
    fn clone_box(&self) -> Box<dyn Historian> {
        Box::new(self.clone())
    }
}

pub struct CloneHistorianGenerator<T> {
    historian: T,
}

impl<T> CloneHistorianGenerator<T>
where
    T: CloneHistorian,
{
    pub fn new(historian: T) -> Self {
        CloneHistorianGenerator { historian }
    }
}

impl<T> HistorianGenerator for CloneHistorianGenerator<T>
where
    T: CloneHistorian,
{
    fn generate(&self, _game_state: &GameState) -> Box<dyn Historian> {
        self.historian.clone_box()
    }
}

mod failing;
mod fn_historian;
mod null;
mod vec;

pub use failing::FailingHistorian;
pub use fn_historian::FnHistorian;
pub use null::NullHistorian;
pub use vec::VecHistorian;
