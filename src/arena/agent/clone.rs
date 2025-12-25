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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::action::AgentAction;
    use std::{cell::RefCell, rc::Rc};

    #[derive(Clone)]
    struct TestAgent {
        name: String,
        call_log: Rc<RefCell<Vec<u128>>>,
    }

    impl Agent for TestAgent {
        fn act(&mut self, id: u128, _game_state: &GameState) -> AgentAction {
            self.call_log.borrow_mut().push(id);
            AgentAction::Fold
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[test]
    fn test_clone_agent_generator_clones_template() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let template = TestAgent {
            name: "TemplateAgent".into(),
            call_log: log.clone(),
        };

        let generator = CloneAgentGenerator::new(template);
        let game_state = GameState::new_starting(vec![100.0; 2], 10.0, 5.0, 0.0, 0);

        let mut first = generator.generate(0, &game_state);
        let mut second = generator.generate(1, &game_state);

        assert_eq!(first.name(), "TemplateAgent");
        assert_eq!(second.name(), "TemplateAgent");

        first.act(11, &game_state);
        second.act(22, &game_state);

        let entries = log.borrow();
        assert_eq!(entries.as_slice(), &[11, 22]);
    }
}
