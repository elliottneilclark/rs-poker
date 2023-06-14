use crate::arena::{action::AgentAction, game_state::GameState};

use super::Agent;

#[derive(Debug, Clone)]
pub struct VecReplayAgent {
    actions: Vec<AgentAction>,
    idx: usize,
    default: AgentAction,
}

impl VecReplayAgent {
    pub fn new(actions: Vec<AgentAction>) -> Self {
        Self {
            actions,
            idx: 0,
            default: AgentAction::Fold,
        }
    }
}

pub struct SliceReplayAgent<'a> {
    actions: &'a [AgentAction],
    idx: usize,
    default: AgentAction,
}

impl Agent for VecReplayAgent {
    fn act(self: &mut VecReplayAgent, _game_state: &GameState) -> AgentAction {
        let idx = self.idx;
        self.idx += 1;
        *self.actions.get(idx).unwrap_or(&self.default)
    }
}

impl<'a> Agent for SliceReplayAgent<'a> {
    fn act(self: &mut SliceReplayAgent<'a>, _game_state: &GameState) -> AgentAction {
        let idx = self.idx;
        self.idx += 1;
        *self.actions.get(idx).unwrap_or(&self.default)
    }
}
