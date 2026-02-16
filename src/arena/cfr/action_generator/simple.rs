use crate::arena::{GameState, action::AgentAction};

use super::super::{CFRState, TraversalState};
use super::ActionGenerator;

/// Action generator with more betting options: fold, check/call, min raise,
/// 33% pot, 66% pot, and all-in.
///
/// This provides a richer action space for CFR exploration compared to
/// BasicCFRActionGenerator which only has fold, call, and all-in.
pub struct SimpleActionGenerator {
    cfr_state: CFRState,
    traversal_state: TraversalState,
}

impl SimpleActionGenerator {
    pub fn new(cfr_state: CFRState, traversal_state: TraversalState) -> Self {
        SimpleActionGenerator {
            cfr_state,
            traversal_state,
        }
    }
}

impl ActionGenerator for SimpleActionGenerator {
    type Config = ();

    fn new(cfr_state: CFRState, traversal_state: TraversalState, _config: ()) -> Self {
        SimpleActionGenerator::new(cfr_state, traversal_state)
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
        let mut actions: Vec<AgentAction> = Vec::with_capacity(6);

        let current_bet = game_state.current_round_bet();
        let player_bet = game_state.current_round_current_player_bet();
        let stack = game_state.current_player_stack();
        let pot = game_state.total_pot;
        let min_raise = game_state.current_round_min_raise();
        let to_call = current_bet - player_bet;

        // All-in amount
        let all_in_amount = player_bet + stack;
        // Minimum valid raise amount
        let min_raise_amount = current_bet + min_raise;

        // Fold - only if there's something to call
        if to_call > 0.0 {
            actions.push(AgentAction::Fold);
        }

        // Call/Check - always available
        actions.push(AgentAction::Bet(current_bet));

        // Min raise = current_bet + min_raise
        if min_raise_amount > current_bet && min_raise_amount < all_in_amount {
            actions.push(AgentAction::Bet(min_raise_amount));
        }

        // 33% pot raise = current_bet + pot * 0.33
        // Must be at least min raise and greater than the min raise bet
        let pot_33_amount = current_bet + pot * 0.33;
        if pot_33_amount >= min_raise_amount
            && pot_33_amount > min_raise_amount
            && pot_33_amount < all_in_amount
        {
            actions.push(AgentAction::Bet(pot_33_amount));
        }

        // 66% pot raise = current_bet + pot * 0.66
        // Must be at least min raise and greater than 33% pot
        let pot_66_amount = current_bet + pot * 0.66;
        if pot_66_amount >= min_raise_amount
            && pot_66_amount > pot_33_amount
            && pot_66_amount < all_in_amount
        {
            actions.push(AgentAction::Bet(pot_66_amount));
        }

        // All-in - only if we can bet more than the current bet
        if all_in_amount > current_bet {
            actions.push(AgentAction::AllIn);
        }

        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::GameStateBuilder;

    fn create_simple_generator(game_state: &GameState) -> SimpleActionGenerator {
        SimpleActionGenerator::new(
            CFRState::new(game_state.clone()),
            TraversalState::new_root(0),
        )
    }

    #[test]
    fn test_simple_gen_actions_with_bet_facing() {
        let stacks = vec![500.0; 2];
        let mut game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        game_state.advance_round();

        // Player 0 bets 30
        game_state.do_bet(30.0, false).unwrap();

        let action_gen = create_simple_generator(&game_state);
        let actions = action_gen.gen_possible_actions(&game_state);

        // Should have: Fold, Call, Min Raise, 33% pot, 66% pot, All-in
        // (assuming all bet sizes are distinct and valid)
        assert!(actions.contains(&AgentAction::Fold));
        assert!(actions.iter().any(|a| matches!(a, AgentAction::Bet(_))));
        assert!(actions.contains(&AgentAction::AllIn));
    }

    #[test]
    fn test_simple_no_fold_when_checking() {
        let stacks = vec![100.0; 2];
        let mut game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        // Advance to flop where no one has bet yet
        game_state.advance_round();

        let action_gen = create_simple_generator(&game_state);
        let actions = action_gen.gen_possible_actions(&game_state);

        // No fold option when there's nothing to call
        assert!(!actions.contains(&AgentAction::Fold));
    }

    /// Helper to verify all generated actions are valid by applying them to game state
    fn verify_all_actions_valid(game_state: &GameState) {
        let action_gen = create_simple_generator(game_state);
        let actions = action_gen.gen_possible_actions(game_state);

        for action in &actions {
            let mut gs_copy = game_state.clone();
            let result = match action {
                AgentAction::Fold => {
                    gs_copy.fold();
                    Ok(())
                }
                AgentAction::Bet(amount) => gs_copy.do_bet(*amount, false).map(|_| ()),
                AgentAction::Call => {
                    let call_amount = gs_copy.current_round_bet();
                    gs_copy.do_bet(call_amount, false).map(|_| ())
                }
                AgentAction::AllIn => {
                    let all_in_amount =
                        gs_copy.current_round_current_player_bet() + gs_copy.current_player_stack();
                    gs_copy.do_bet(all_in_amount, false).map(|_| ())
                }
            };

            assert!(
                result.is_ok(),
                "Action {:?} should be valid but got error: {:?}\n\
                 Game state: current_bet={}, min_raise={}, player_bet={}, stack={}, pot={}",
                action,
                result.err(),
                game_state.current_round_bet(),
                game_state.current_round_min_raise(),
                game_state.current_round_current_player_bet(),
                game_state.current_player_stack(),
                game_state.total_pot
            );
        }
    }

    #[test]
    fn test_simple_all_actions_valid_preflop_sb() {
        // Small blind facing big blind
        let stacks = vec![100.0; 2];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_preflop_bb() {
        // Big blind after small blind completes
        let stacks = vec![100.0; 2];
        let mut game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        game_state.do_bet(10.0, false).unwrap(); // SB completes to BB
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_flop_first_to_act() {
        // First to act on flop (can check)
        let stacks = vec![100.0; 2];
        let mut game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        game_state.advance_round(); // To flop
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_facing_bet() {
        // Facing a bet on flop
        let stacks = vec![100.0; 2];
        let mut game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        game_state.advance_round();
        game_state.do_bet(20.0, false).unwrap(); // Opponent bets
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_facing_raise() {
        // Facing a raise
        let stacks = vec![200.0; 2];
        let mut game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        game_state.advance_round();
        game_state.do_bet(20.0, false).unwrap(); // Bet
        game_state.do_bet(50.0, false).unwrap(); // Raise
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_small_stack() {
        // Player with small stack
        let stacks = vec![30.0, 100.0];
        let mut game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        game_state.advance_round();
        game_state.do_bet(15.0, false).unwrap();
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_large_pot() {
        // Large pot scenario where pot bets might exceed stack
        let stacks = vec![100.0; 2];
        let mut game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        // Build up pot through betting rounds
        game_state.do_bet(10.0, false).unwrap(); // SB completes
        game_state.do_bet(30.0, false).unwrap(); // BB raises
        game_state.do_bet(30.0, false).unwrap(); // SB calls
        game_state.advance_round(); // To flop
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_tiny_pot() {
        // Tiny pot where pot-based bets might be smaller than min raise
        let stacks = vec![1000.0; 2];
        let mut game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(2.0, 1.0)
            .build()
            .unwrap();
        game_state.advance_round();
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_after_multiple_raises() {
        // After multiple raises (min raise increases)
        let stacks = vec![500.0; 2];
        let mut game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        game_state.advance_round();
        game_state.do_bet(20.0, false).unwrap(); // Bet 20
        game_state.do_bet(50.0, false).unwrap(); // Raise to 50 (raise of 30)
        game_state.do_bet(110.0, false).unwrap(); // Re-raise to 110 (raise of 60)
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_three_players() {
        // Three player scenario
        let stacks = vec![100.0; 3];
        let mut game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        game_state.advance_round();
        game_state.do_bet(15.0, false).unwrap();
        game_state.do_bet(15.0, false).unwrap();
        verify_all_actions_valid(&game_state);
    }

    #[test]
    fn test_simple_all_actions_valid_river() {
        // River scenario
        let stacks = vec![100.0; 2];
        let mut game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        game_state.advance_round(); // Flop
        game_state.advance_round(); // Turn
        game_state.advance_round(); // River
        verify_all_actions_valid(&game_state);
    }
}
