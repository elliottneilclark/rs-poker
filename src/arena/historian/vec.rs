use crate::arena::action::Action;

use super::Historian;

#[derive(Clone)]
struct VecHistorian {
    actions: Vec<Action>,
}

impl Historian for VecHistorian {
    fn get_history(&self) -> Option<Vec<Action>> {
        Some(self.actions.clone())
    }

    fn record_action(
        &mut self,
        _id: &uuid::Uuid,
        _game_state: &crate::arena::GameState,
        action: Action,
    ) {
        self.actions.push(action)
    }
}
