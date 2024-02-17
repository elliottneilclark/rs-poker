use super::{action::Action, GameState};

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
    fn record_action(&mut self, id: &uuid::Uuid, game_state: &GameState, action: Action);
}

mod fn_historian;
mod vec;

pub use fn_historian::FnHistorian;
pub use vec::VecHistorian;
