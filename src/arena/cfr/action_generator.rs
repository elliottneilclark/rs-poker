use std::cell::Ref;

use tracing::event;

use crate::arena::{GameState, action::AgentAction};

use super::{CFRState, Node, NodeData, TraversalState};

pub trait ActionGenerator {
    /// Create a new action generator
    ///
    /// This is used by the Agent to create identical
    /// action generators for the historians it uses.
    fn new(cfr_state: CFRState, traversal_state: TraversalState) -> Self;

    /// Given an action return the index of the action in the children array.
    ///
    /// # Arguments
    ///
    /// * `game_state` - The current game state
    /// * `action` - The action to convert to an index
    ///
    /// # Returns
    ///
    /// The index of the action in the children array. The 0 index is the fold
    /// action. All other are defined by the implentation
    fn action_to_idx(&self, action: &AgentAction) -> usize;

    /// How many potential actions in total might be generated.
    ///
    /// At a given node there might be fewere that will be
    /// possible, but the regret matcher doesn't keep track of that.
    ///
    /// At all time the number of potential actions is
    /// larger than or equal to the number of possible actions
    ///
    /// # Returns
    ///
    /// The number of potential actions
    fn num_potential_actions(&self, game_state: &GameState) -> usize;

    // Generate all possible actions for the current game state
    //
    // This returns a vector so that the actions can be chosen from randomly
    fn gen_possible_actions(&self, game_state: &GameState) -> Vec<AgentAction>;

    // Using the current and the CFR's tree's regret state choose a single action to
    // play.
    fn gen_action(&self, game_state: &GameState) -> AgentAction;
}

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

    fn get_target_node(&self) -> Option<Ref<Node>> {
        let from_node_idx = self.traversal_state.node_idx();
        let from_child_idx = self.traversal_state.chosen_child_idx();
        self.cfr_state
            .get(from_node_idx)
            .unwrap()
            .get_child(from_child_idx)
            .and_then(|idx| self.cfr_state.get(idx))
    }
}

impl ActionGenerator for BasicCFRActionGenerator {
    fn gen_action(&self, game_state: &GameState) -> AgentAction {
        let possible = self.gen_possible_actions(game_state);
        // For now always use the thread rng.
        // At somepoint we will want to be able to pass seeded or deterministic action
        // choices.
        let mut rng = rand::rng();

        // We expect there to be a target node with a regret matcher
        match self.get_target_node() {
            Some(node) => {
                if let NodeData::Player(pd) = &node.data {
                    let next_action = pd
                        .regret_matcher
                        .as_ref()
                        .map_or(0, |matcher| matcher.next_action(&mut rng));

                    event!(
                        tracing::Level::DEBUG,
                        next_action = next_action,
                        "Next action index"
                    );

                    // Find the first action that matches the index picked from the regret matcher
                    possible
                    .iter()
                    .find_map(|action| {
                        if self.action_to_idx(action) == next_action {
                            Some(action.clone())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| {
                        // Just in case the regret matcher returns an action that is not in the possible actions
                        // choose the first possible action as a fallback or fold if there are no possible actions
                        let fallback = possible.first().unwrap_or(&AgentAction::Fold).clone();
                        event!(tracing::Level::WARN, fallback = ?fallback, "No action found for next action index");
                        fallback
                    })
                } else {
                    panic!("Expected player node");
                }
            }
            _ => {
                panic!("Expected target node");
            }
        }
    }

    fn new(cfr_state: CFRState, traversal_state: TraversalState) -> Self {
        BasicCFRActionGenerator {
            cfr_state,
            traversal_state,
        }
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

    fn action_to_idx(&self, action: &AgentAction) -> usize {
        match action {
            AgentAction::Fold => 0,
            AgentAction::Bet(_) => 1,
            AgentAction::AllIn => 2,
        }
    }

    fn num_potential_actions(&self, _game_state: &GameState) -> usize {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::arena::GameState;

    use std::vec;

    #[test]
    fn test_should_gen_2_actions() {
        let stacks = vec![50.0; 2];
        let game_state = GameState::new_starting(stacks, 2.0, 1.0, 0.0, 0);
        let action_generator = BasicCFRActionGenerator::new(
            CFRState::new(game_state.clone()),
            TraversalState::new_root(0),
        );
        let actions = action_generator.gen_possible_actions(&game_state);
        // We should have 2 actions: Call or All-in since 0 is the dealer when starting
        assert_eq!(actions.len(), 2);

        // None of the ations should have a child idx of 0
        for action in actions {
            assert_ne!(action_generator.action_to_idx(&action), 0);
        }
    }

    #[test]
    fn test_should_gen_3_actions() {
        let stacks = vec![50.0; 2];
        let mut game_state = GameState::new_starting(stacks, 2.0, 1.0, 0.0, 0);
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

        // Check the indices of the actions
        assert_eq!(
            action_generator.action_to_idx(&AgentAction::Fold),
            0
        );
        assert_eq!(
            action_generator.action_to_idx(&AgentAction::Bet(10.0)),
            1
        );
        assert_eq!(
            action_generator.action_to_idx(&AgentAction::AllIn),
            2
        );
    }
}
