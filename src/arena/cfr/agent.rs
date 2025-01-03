use crate::arena::{action::Action, Agent, GameState, Historian, HistorianError};

use super::{
    action_generator::ActionGenerator,
    state::{CFRState, TraversalState},
};

pub struct CFRAgent<T>
where
    T: ActionGenerator,
{
    pub traversal_state: TraversalState,
    pub cfr_state: CFRState,
    pub action_generator: T,
}

pub struct CFRHistorian<T>
where
    T: ActionGenerator,
{
    pub traversal_state: TraversalState,
    pub cfr_state: CFRState,
    pub action_generator: T,
}

impl<T> CFRHistorian<T>
where
    T: ActionGenerator,
{
    fn new(traversal_state: TraversalState, cfr_state: CFRState) -> Self {
        let action_generator = T::new(cfr_state.clone(), traversal_state.clone());
        CFRHistorian {
            traversal_state,
            cfr_state,
            action_generator,
        }
    }
}

impl<T> Historian for CFRHistorian<T>
where
    T: ActionGenerator,
{
    fn record_action(
        &mut self,
        _id: &uuid::Uuid,
        _game_state: &GameState,
        action: Action,
    ) -> Result<(), HistorianError> {
        match action {
            // These are all assumed from game start and encoded in the root node.
            Action::GameStart(_) | Action::ForcedBet(_) | Action::PlayerSit(_) => Ok(()),
            // We don't encode round advance in the tree because it never changes the outcome.
            Action::RoundAdvance(_) => Ok(()),
            // Rather than use award since it can be for a side pot we use the final award ammount
            // in the terminal node.
            Action::Award(_) => Ok(()),
            Action::DealStartingHand(_deal_starting_hand_payload) => todo!(),
            Action::PlayedAction(_played_action_payload) => todo!(),
            Action::FailedAction(_failed_action_payload) => todo!(),
            Action::DealCommunity(_card) => todo!(),
        }
    }
}

impl<T> CFRAgent<T>
where
    T: ActionGenerator,
{
    pub fn new(
        cfr_state: CFRState,
        node_idx: usize,
        chosen_child: usize,
        player_idx: usize,
    ) -> Self {
        let traversal_state = TraversalState::new(node_idx, chosen_child, player_idx);
        let action_generator = T::new(cfr_state.clone(), traversal_state.clone());
        CFRAgent {
            cfr_state,
            traversal_state,
            action_generator,
        }
    }

    pub fn historian(&self) -> CFRHistorian<T> {
        CFRHistorian::new(self.traversal_state.clone(), self.cfr_state.clone())
    }
}

impl<T> Agent for CFRAgent<T>
where
    T: ActionGenerator,
{
    fn act(
        &mut self,
        _id: &uuid::Uuid,
        game_state: &GameState,
    ) -> crate::arena::action::AgentAction {
        self.action_generator.gen_action(game_state)
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::cfr::CFRActionGenerator;

    use crate::arena::game_state;

    use super::*;

    #[test]
    fn test_create_agent() {
        let game_state = game_state::GameState::new_starting(vec![100.0; 3], 10.0, 5.0, 0.0, 0);
        let cfr_state = CFRState::new(game_state);
        let _ = CFRAgent::<CFRActionGenerator>::new(
            cfr_state.clone(),
            // we are still at root so 0
            0,
            0,
            0,
        );
    }
}
