use crate::arena::{Agent, AgentGenerator, GameState};

pub trait CloneAgent: Agent {
    fn clone_box(&self) -> Box<dyn Agent>;
}

impl<T> CloneAgent for T
where
    T: 'static + Agent + Clone,
{
    fn clone_box(&self) -> Box<dyn Agent> {
        Box::new(self.clone())
    }
}

pub struct CloneAgentGenerator<T> {
    agent: T,
}

impl<T> CloneAgentGenerator<T>
where
    T: CloneAgent,
{
    pub fn new(agent: T) -> Self {
        CloneAgentGenerator { agent }
    }
}

impl<T> AgentGenerator for CloneAgentGenerator<T>
where
    T: CloneAgent,
{
    fn generate(&self, _player_idx: usize, _game_state: &GameState) -> Box<dyn Agent> {
        self.agent.clone_box()
    }
}
