use tracing::{event, Level};

use crate::arena::{
    action::{Action, AgentAction},
    GameState,
};

use super::{CFRState, TraversalState};

pub trait ActionGenerator {
    fn new(cfr_state: CFRState, traversal_state: TraversalState) -> Self;

    fn action_to_idx(&self, action: &Action) -> usize;

    fn gen_action(&self, game_state: &GameState) -> AgentAction;

    fn num_possible_actions(&self, game_state: &GameState) -> usize;
}

pub struct CFRActionGenerator {
    cfr_state: CFRState,
    traversal_state: TraversalState,
}

impl ActionGenerator for CFRActionGenerator {
    fn action_to_idx(&self, _action: &Action) -> usize {
        todo!()
    }

    fn gen_action(&self, _game_state: &GameState) -> AgentAction {
        event!(Level::TRACE, ?self.cfr_state, ?self.traversal_state, "Generating a new action");
        AgentAction::Fold
    }

    fn new(cfr_state: CFRState, traversal_state: TraversalState) -> Self {
        CFRActionGenerator {
            cfr_state,
            traversal_state,
        }
    }

    fn num_possible_actions(&self, _game_state: &GameState) -> usize {
        // TODO: Implement this. It has to always be less
        // than 52 since we use the same children array
        // for all nodes including chance nodes.
        8
    }
}
