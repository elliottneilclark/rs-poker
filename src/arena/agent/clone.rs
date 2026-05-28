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
    use crate::arena::GameStateBuilder;
    use crate::arena::action::AgentAction;
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct TestAgent {
        name: String,
        call_log: Arc<Mutex<Vec<u128>>>,
    }

    #[async_trait]
    impl Agent for TestAgent {
        async fn act(&mut self, id: u128, _game_state: &GameState) -> AgentAction {
            self.call_log.lock().unwrap().push(id);
            AgentAction::Fold
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_clone_agent_generator_clones_template() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let template = TestAgent {
            name: "TemplateAgent".into(),
            call_log: log.clone(),
        };

        let generator = CloneAgentGenerator::new(template);
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();

        let mut first = generator.generate(0, &game_state);
        let mut second = generator.generate(1, &game_state);

        assert_eq!(first.name(), "TemplateAgent");
        assert_eq!(second.name(), "TemplateAgent");

        first.act(11, &game_state).await;
        second.act(22, &game_state).await;

        let entries = log.lock().unwrap();
        assert_eq!(entries.as_slice(), &[11, 22]);
    }
}
