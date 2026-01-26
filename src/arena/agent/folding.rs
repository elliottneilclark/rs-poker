use std::sync::atomic::{AtomicUsize, Ordering};

use tracing::{instrument, trace};

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
    #[instrument(level = "trace", skip(self, game_state), fields(agent_name = %self.name))]
    fn act(self: &mut FoldingAgent, _id: u128, game_state: &GameState) -> AgentAction {
        // Count all players still in the hand (not folded), including those who are all-in
        // Note: num_active_players() counts players who haven't folded and aren't all-in
        // num_all_in_players() counts players who are all-in
        let players_in_hand = game_state.num_active_players() + game_state.num_all_in_players();
        if players_in_hand == 1 {
            // We're the only one left (everyone else folded or is all-in and we're last)
            // Just bet the minimum to claim the pot
            let bet = game_state.current_round_bet();
            trace!(
                bet,
                players_in_hand, "FoldingAgent claiming pot (last player)"
            );
            AgentAction::Bet(bet)
        } else {
            // Check if we can fold (only valid when there's something to call)
            let current_bet = game_state.current_round_bet();
            let player_bet = game_state.current_round_current_player_bet();
            let to_call = current_bet - player_bet;

            if to_call > 0.0 {
                trace!(players_in_hand, to_call, "FoldingAgent folding");
                AgentAction::Fold
            } else {
                // Can't fold when there's nothing to call - check instead
                trace!(players_in_hand, "FoldingAgent checking (nothing to call)");
                AgentAction::Bet(current_bet)
            }
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

        // In a Starting round, blinds haven't been posted yet, so there's
        // nothing to call. The FoldingAgent correctly checks instead of folding.
        match agent.act(0, &game_state) {
            AgentAction::Bet(0.0) => {} // Check (nothing to call)
            action => panic!("Expected Bet(0.0) action (check), got {:?}", action),
        }
    }

    #[test]
    fn test_folding_agent_folds_when_facing_bet() {
        use crate::arena::game_state::RoundData;
        use crate::core::PlayerBitSet;

        // Create a game state where there's a bet to call
        let mut round_data = RoundData::new(2, 10.0, PlayerBitSet::new(2), 1);
        round_data.bet = 20.0; // Current bet is 20
        round_data.player_bet[0] = 20.0; // Player 0 has bet 20
        round_data.player_bet[1] = 10.0; // Player 1 (to act) has bet 10

        let game_state = GameState::new(
            crate::arena::game_state::Round::Preflop,
            round_data,
            vec![],
            vec![crate::core::Hand::default(); 2],
            vec![100.0; 2],
            vec![0.0; 2],
            10.0,
            5.0,
            0.0,
            0,
        );

        let mut agent = FoldingAgent::new("TestFolder");

        // Now there's something to call (10 chips), so the agent should fold
        match agent.act(0, &game_state) {
            AgentAction::Fold => {}
            action => panic!("Expected Fold action, got {:?}", action),
        }
    }

    #[test]
    fn test_folding_generator_uses_custom_name() {
        let generator = FoldingAgentGenerator::with_name("FolderZ");
        let game_state = GameState::new_starting(vec![40.0; 2], 10.0, 5.0, 0.0, 0);

        let agent = generator.generate(0, &game_state);
        assert_eq!(agent.name(), "FolderZ");
    }

    #[test]
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

    /// Verifies that FoldingAgent checks (not folds) when the player
    /// has already matched the current bet and owes nothing.
    #[test]
    fn test_folding_agent_checks_when_bet_matched() {
        use crate::arena::game_state::RoundData;
        use crate::core::PlayerBitSet;

        // Create a game state where player has already matched the current bet
        let mut round_data = RoundData::new(2, 20.0, PlayerBitSet::new(2), 1);
        round_data.bet = 20.0; // Current bet is 20
        round_data.player_bet[0] = 20.0; // Player 0 has bet 20
        round_data.player_bet[1] = 20.0; // Player 1 (to act) has also bet 20

        let game_state = GameState::new(
            crate::arena::game_state::Round::Preflop,
            round_data,
            vec![],
            vec![crate::core::Hand::default(); 2],
            vec![100.0; 2],
            vec![0.0; 2],
            10.0,
            5.0,
            0.0,
            0,
        );

        let mut agent = FoldingAgent::new("TestFolder");

        // Player has matched the bet (to_call = 0), so agent should check
        match agent.act(0, &game_state) {
            AgentAction::Bet(bet) => {
                assert_eq!(
                    bet, 20.0,
                    "Should check/call at current bet level when nothing to call"
                );
            }
            action => panic!("Expected Bet action (check), got {:?}", action),
        }
    }
}
