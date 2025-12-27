use std::sync::atomic::{AtomicUsize, Ordering};

use crate::arena::{action::AgentAction, game_state::GameState};

use super::{Agent, AgentGenerator};

/// A simple agent that folds unless there is only one active player left.
#[derive(Debug, Clone)]
pub struct FoldingAgent {
    name: String,
}

impl FoldingAgent {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Default for FoldingAgent {
    fn default() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let idx = COUNTER.fetch_add(1, Ordering::Relaxed);
        FoldingAgent::new(format!("FoldingAgent-{idx}"))
    }
}

impl Agent for FoldingAgent {
    fn act(self: &mut FoldingAgent, _id: u128, game_state: &GameState) -> AgentAction {
        let count = game_state.current_round_num_active_players() + game_state.num_all_in_players();
        if count == 1 {
            AgentAction::Bet(game_state.current_round_bet())
        } else {
            AgentAction::Fold
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Default Generator for `FoldingAgent`.
#[derive(Debug, Clone, Default)]
pub struct FoldingAgentGenerator {
    name: Option<String>,
}

impl FoldingAgentGenerator {
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
            .unwrap_or_else(|| format!("FoldingAgent-{player_idx}"))
    }
}

impl AgentGenerator for FoldingAgentGenerator {
    fn generate(&self, player_idx: usize, _game_state: &GameState) -> Box<dyn Agent> {
        Box::new(FoldingAgent::new(self.resolve_name(player_idx)))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use rand::{SeedableRng, rngs::StdRng};

    use crate::arena::{HoldemSimulationBuilder, game_state::Round};

    use super::*;

    #[test]
    fn test_folding_generator_creates_named_folder() {
        let generator = FoldingAgentGenerator::default();
        let game_state = GameState::new_starting(vec![100.0; 2], 10.0, 5.0, 0.0, 0);

        let mut agent = generator.generate(0, &game_state);
        assert_eq!(agent.name(), "FoldingAgent-0");

        match agent.act(0, &game_state) {
            AgentAction::Fold => {}
            action => panic!("Expected fold action, got {:?}", action),
        }
    }

    #[test]
    fn test_folding_generator_uses_custom_name() {
        let generator = FoldingAgentGenerator::with_name("FolderZ");
        let game_state = GameState::new_starting(vec![40.0; 2], 10.0, 5.0, 0.0, 0);

        let agent = generator.generate(0, &game_state);
        assert_eq!(agent.name(), "FolderZ");
    }

    #[test_log::test]
    fn test_folding_agents() {
        let stacks = vec![100.0; 2];
        let mut rng = StdRng::seed_from_u64(420);

        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(vec![
                Box::new(FoldingAgent::new("FoldingAgent-0")),
                Box::new(FoldingAgent::new("FoldingAgent-1")),
            ])
            .build()
            .unwrap();

        sim.run(&mut rng);

        assert_eq!(sim.game_state.num_active_players(), 1);
        assert_eq!(sim.game_state.round, Round::Complete);

        assert_relative_eq!(15.0_f32, sim.game_state.player_bet.iter().sum());

        assert_relative_eq!(15.0_f32, sim.game_state.player_winnings.iter().sum());
        assert_relative_eq!(15.0_f32, sim.game_state.player_winnings[1]);
    }
}
