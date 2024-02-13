use dyn_clone::DynClone;

use super::{action::Action, GameState};

//
pub trait Historian: DynClone {
    fn record_action(&mut self, id: &uuid::Uuid, game_state: &GameState, action: Action);
    fn get_history(&self) -> Option<Vec<Action>> {
        None
    }
}
dyn_clone::clone_trait_object!(Historian);

pub mod vec;
