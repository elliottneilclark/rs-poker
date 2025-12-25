use std::sync::atomic::{AtomicUsize, Ordering};

use crate::arena::{action::AgentAction, game_state::GameState};

use super::{Agent, AgentGenerator};

/// A simple agent that always calls.
#[derive(Debug, Clone)]
pub struct CallingAgent {
    name: String,
}

impl CallingAgent {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Default for CallingAgent {
    fn default() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let idx = COUNTER.fetch_add(1, Ordering::Relaxed);
        CallingAgent::new(format!("CallingAgent-{idx}"))
    }
}

impl Agent for CallingAgent {
    fn act(self: &mut CallingAgent, _id: u128, game_state: &GameState) -> AgentAction {
        AgentAction::Bet(game_state.current_round_bet())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Default `AgentGenerator` for `CallingAgent`.
#[derive(Debug, Clone, Default)]
pub struct CallingAgentGenerator {
    name: Option<String>,
}

impl CallingAgentGenerator {
    pub fn new() -> Self {
        Self { name: None }
    }

    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
        }
    }

    fn resolve_name(&self, player_idx: usize) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| format!("CallingAgent-{player_idx}"))
    }
}

impl AgentGenerator for CallingAgentGenerator {
    fn generate(&self, player_idx: usize, _game_state: &GameState) -> Box<dyn Agent> {
        Box::new(CallingAgent::new(self.resolve_name(player_idx)))
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::HoldemSimulationBuilder;

    use super::*;

    #[test]
    fn test_calling_generator_creates_named_caller() {
        let generator = CallingAgentGenerator::default();
        let game_state = GameState::new_starting(vec![100.0; 3], 10.0, 5.0, 0.0, 0);

        let mut agent = generator.generate(2, &game_state);
        assert_eq!(agent.name(), "CallingAgent-2");

        match agent.act(0, &game_state) {
            AgentAction::Bet(amount) => {
                assert_eq!(amount, game_state.current_round_bet());
            }
            action => panic!("Expected call-sized bet, got {:?}", action),
        }
    }

    #[test]
    fn test_calling_generator_uses_custom_name() {
        let generator = CallingAgentGenerator::with_name("CallerX");
        let game_state = GameState::new_starting(vec![50.0; 2], 10.0, 5.0, 0.0, 0);

        let agent = generator.generate(0, &game_state);
        assert_eq!(agent.name(), "CallerX");
    }

    #[test_log::test]
    fn test_call_agents() {
        let stacks = vec![100.0; 4];
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut rng = rand::rng();
        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(vec![
                Box::new(CallingAgent::new("CallingAgent-0")),
                Box::new(CallingAgent::new("CallingAgent-1")),
                Box::new(CallingAgent::new("CallingAgent-2")),
                Box::new(CallingAgent::new("CallingAgent-3")),
            ])
            .build()
            .unwrap();

        sim.run(&mut rng);

        assert_eq!(sim.game_state.num_active_players(), 4);

        assert_ne!(0.0, sim.game_state.player_winnings.iter().sum::<f32>());
        assert_eq!(40.0, sim.game_state.player_winnings.iter().sum::<f32>());
    }
}
