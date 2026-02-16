use std::sync::atomic::{AtomicUsize, Ordering};

use tracing::{instrument, trace};

use crate::arena::{Agent, AgentGenerator, GameState, action::AgentAction};

/// A simple agent that always goes all-in regardless of context.
#[derive(Debug, Clone)]
pub struct AllInAgent {
    name: String,
}

impl AllInAgent {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Default for AllInAgent {
    fn default() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let idx = COUNTER.fetch_add(1, Ordering::Relaxed);
        AllInAgent::new(format!("AllInAgent-{idx}"))
    }
}

impl Agent for AllInAgent {
    #[instrument(level = "trace", skip(self, game_state), fields(agent_name = %self.name))]
    fn act(self: &mut AllInAgent, _id: u128, game_state: &GameState) -> AgentAction {
        let bet = game_state.current_player_stack() + game_state.current_round_bet();
        trace!(bet, "AllInAgent going all-in");
        AgentAction::Bet(bet)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Default `AgentGenerator` for `AllInAgent`.
#[derive(Debug, Clone, Default)]
pub struct AllInAgentGenerator {
    name: Option<String>,
}

impl AllInAgentGenerator {
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
            .unwrap_or_else(|| format!("AllInAgent-{player_idx}"))
    }
}

impl AgentGenerator for AllInAgentGenerator {
    fn generate(&self, player_idx: usize, _game_state: &GameState) -> Box<dyn Agent> {
        Box::new(AllInAgent::new(self.resolve_name(player_idx)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::GameStateBuilder;

    fn test_game_state(stacks: Vec<f32>, big_blind: f32, small_blind: f32) -> GameState {
        GameStateBuilder::new()
            .stacks(stacks)
            .blinds(big_blind, small_blind)
            .build()
            .unwrap()
    }

    #[test]
    fn test_all_in_generator_produces_named_bet() {
        let generator = AllInAgentGenerator::default();
        let game_state = test_game_state(vec![100.0; 2], 10.0, 5.0);

        let mut agent = generator.generate(1, &game_state);
        assert_eq!(agent.name(), "AllInAgent-1");

        match agent.act(0, &game_state) {
            AgentAction::Bet(amount) => {
                let expected = game_state.current_player_stack() + game_state.current_round_bet();
                assert_eq!(amount, expected);
            }
            action => panic!("Expected all-in bet, got {:?}", action),
        }
    }

    #[test]
    fn test_all_in_generator_uses_custom_name() {
        let generator = AllInAgentGenerator::with_name("HeroBot");
        let game_state = test_game_state(vec![50.0; 2], 10.0, 5.0);

        let agent = generator.generate(0, &game_state);
        assert_eq!(agent.name(), "HeroBot");
    }
}
