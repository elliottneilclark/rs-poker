use std::sync::Arc;

use crate::arena::{GameState, action::AgentAction};

use super::super::{CFRState, TraversalState};
use super::ActionGenerator;

/// Basic CFR action generator with fold, call/check, and all-in actions.
///
/// This is a minimal action generator that provides basic poker actions.
/// The actual action-to-index mapping is done by `ActionIndexMapper`.
pub struct BasicCFRActionGenerator {
    cfr_state: CFRState,
    traversal_state: TraversalState,
}

impl BasicCFRActionGenerator {
    pub fn new(cfr_state: CFRState, traversal_state: TraversalState) -> Self {
        BasicCFRActionGenerator {
            cfr_state,
            traversal_state,
        }
    }
}

impl ActionGenerator for BasicCFRActionGenerator {
    type Config = ();

    fn new(cfr_state: CFRState, traversal_state: TraversalState, _config: Arc<()>) -> Self {
        BasicCFRActionGenerator {
            cfr_state,
            traversal_state,
        }
    }

    fn config(&self) -> &Self::Config {
        &()
    }

    fn cfr_state(&self) -> &CFRState {
        &self.cfr_state
    }

    fn traversal_state(&self) -> &TraversalState {
        &self.traversal_state
    }

    fn gen_possible_actions(&self, game_state: &GameState) -> Vec<AgentAction> {
        let mut res: Vec<AgentAction> = Vec::with_capacity(3);
        let to_call =
            game_state.current_round_bet() - game_state.current_round_current_player_bet();
        if to_call > 0.0 {
            res.push(AgentAction::Fold);
        }
        // Call, Match the current bet (if the bet is 0 this is a check)
        res.push(AgentAction::Bet(game_state.current_round_bet()));

        let all_in_ammount =
            game_state.current_round_current_player_bet() + game_state.current_player_stack();

        if all_in_ammount > game_state.current_round_bet() {
            // All-in, Bet all the money
            // Bet everything we have bet so far plus the remaining stack
            res.push(AgentAction::AllIn);
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::GameStateBuilder;

    #[test]
    fn test_should_gen_2_actions() {
        let stacks = vec![50.0; 2];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(2.0, 1.0)
            .build()
            .unwrap();
        let action_generator = BasicCFRActionGenerator::new(
            CFRState::new(game_state.clone()),
            TraversalState::new_root(0),
        );
        let actions = action_generator.gen_possible_actions(&game_state);
        // We should have 2 actions: Call or All-in since 0 is the dealer when starting
        assert_eq!(actions.len(), 2);

        // Neither action should be Fold (since there's nothing to fold to)
        assert!(!actions.contains(&AgentAction::Fold));
    }

    #[test]
    fn test_should_gen_3_actions() {
        let stacks = vec![50.0; 2];
        let mut game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(2.0, 1.0)
            .build()
            .unwrap();
        game_state.advance_round();
        game_state.advance_round();

        game_state.do_bet(10.0, false).unwrap();
        let action_generator = BasicCFRActionGenerator::new(
            CFRState::new(game_state.clone()),
            TraversalState::new_root(0),
        );
        let actions = action_generator.gen_possible_actions(&game_state);
        // We should have 3 actions: Fold, Call, or All-in
        assert_eq!(actions.len(), 3);

        // Check that we have fold, a bet, and all-in
        assert!(actions.contains(&AgentAction::Fold));
        assert!(actions.iter().any(|a| matches!(a, AgentAction::Bet(_))));
        assert!(actions.contains(&AgentAction::AllIn));
    }
}
