use std::{cell::RefCell, rc::Rc};

use crate::arena::{
    action::{Action, AgentAction, DealStartingHandPayload, PlayedActionPayload},
    game_state::Round,
    historian::{Historian, HistorianError},
    GameState,
};

use super::{
    node::NodeData,
    state::{EnsureNodeType, PlayerCFRState},
    LOWER_MULT, UPPER_MULT,
};

pub struct ArenaCFRHistorian {
    pub state: Rc<RefCell<PlayerCFRState>>,
    pub player_idx: usize,
}

impl ArenaCFRHistorian {
    pub fn new(state: Rc<RefCell<PlayerCFRState>>, player_idx: usize) -> ArenaCFRHistorian {
        ArenaCFRHistorian { state, player_idx }
    }

    fn played_action_to_idx(
        &self,
        game_state: &GameState,
        played_action: &PlayedActionPayload,
    ) -> usize {
        match game_state.round {
            Round::Preflop => self.preflop_action_to_idx(played_action),
            _ => self.postflop_action_to_idx(played_action),
        }
    }

    fn preflop_action_to_idx(&self, played_action: &PlayedActionPayload) -> usize {
        // Fold is 0
        // The rest will have to be figured out
        match played_action.action {
            AgentAction::Fold => 0,
            AgentAction::Bet(_) => self.preflop_bet_action_to_idx(played_action),
        }
    }

    fn postflop_action_to_idx(&self, played_action: &PlayedActionPayload) -> usize {
        // Fold is 0
        // The rest will have to be figured out
        match played_action.action {
            AgentAction::Fold => 0,
            AgentAction::Bet(_) => self.postflot_bet_action_to_idx(played_action),
        }
    }

    // Guess which expert created this action
    // that is a Bet of some kind.
    //
    // 1 -> Check
    // 2 -> Min Raise
    // 3 -> 1/2 pot
    // 4 -> 2/3 pot
    // 5 -> pot size
    // 6 -> pot == player stack
    // 7 -> All In
    // 8 -> random ? huh?
    fn postflot_bet_action_to_idx(&self, played_action: &PlayedActionPayload) -> usize {
        let raise_amount = played_action.raise_amount();

        // For 1/2 pot
        let min_one_half = played_action.starting_pot * 0.5 * LOWER_MULT;
        let max_one_half = played_action.starting_pot * 0.5 * UPPER_MULT;

        // for 2/3 pot
        let min_two_thirds = played_action.starting_pot * 0.66666 * LOWER_MULT;
        let max_two_thirds = played_action.starting_pot * 0.66666 * UPPER_MULT;

        // pot
        let min_pot = played_action.starting_pot * LOWER_MULT;
        let max_pot = played_action.starting_pot * UPPER_MULT;

        // maybe geo?
        //
        // What if the player is setting up for a shove next round
        let min_per = raise_amount * LOWER_MULT;
        // If we're setting up for a pot sized shove this is the expected pot
        // The starting pot plus everyone still left calling
        let expected_geo =
            played_action.starting_pot + min_per * played_action.players_active.count() as f32;

        let min_geo_expected = expected_geo * LOWER_MULT;
        let max_geo_expected = expected_geo * UPPER_MULT;

        if played_action.starting_bet == played_action.final_bet {
            // Check
            1
        } else if raise_amount <= played_action.starting_min_raise * UPPER_MULT {
            // min raise
            2
        } else if raise_amount <= min_one_half && raise_amount >= max_one_half {
            // about 50% of pot raise
            3
        } else if raise_amount >= min_two_thirds && raise_amount <= max_two_thirds {
            4
        } else if raise_amount >= min_pot && raise_amount <= max_pot {
            // About pot sized raise
            5
        } else if played_action.player_stack * LOWER_MULT >= min_geo_expected
            && played_action.player_stack * UPPER_MULT <= max_geo_expected
        {
            // it look like the player is setting for a shove
            6
        } else if played_action.player_stack == 0.0 {
            // All In
            7
        } else {
            // Dunno random bet ?
            8
        }
    }
    fn preflop_bet_action_to_idx(&self, played_action: &PlayedActionPayload) -> usize {
        let raise_amount = played_action.raise_amount();

        // for 2/3 pot
        let min_two_thirds = played_action.starting_pot * 0.66666 * LOWER_MULT;
        let max_two_thirds = played_action.starting_pot * 0.66666 * UPPER_MULT;

        // pot
        let min_pot = played_action.starting_pot * LOWER_MULT;
        let max_pot = played_action.starting_pot * UPPER_MULT;

        if played_action.starting_bet == played_action.final_bet {
            // Check
            1
        } else if raise_amount <= played_action.starting_min_raise * UPPER_MULT {
            // min raise
            2
        } else if raise_amount >= min_two_thirds && raise_amount <= max_two_thirds {
            3
        } else if raise_amount >= min_pot && raise_amount <= max_pot {
            // About pot sized raise
            4
        } else {
            // random I guess
            5
        }
    }

    fn handle_terminal_node(&mut self, game_state: &GameState) {
        // Compute the utility for every player
        let utility: Vec<f32> = game_state
            .player_winnings
            .iter()
            .zip(game_state.player_bet.iter())
            .map(|(winnings, bet)| winnings - bet)
            .collect();

        let mut state = self.state.borrow_mut();
        // Well store that in the terminal node
        if let Some(terminal_node) = state.get_mut_current_node() {
            if let NodeData::Terminal(terminal_data) = &mut terminal_node.data {
                terminal_data.utility = utility;
            }
        }
    }

    fn handle_played_action_payload(
        &mut self,
        game_state: &GameState,
        played_action: PlayedActionPayload,
    ) -> Result<(), HistorianError> {
        if played_action.idx == self.player_idx {
            self.state
                .try_borrow_mut()?
                .ensure_current_node(EnsureNodeType::Player(played_action.idx), game_state)?;
        } else {
            self.state
                .try_borrow_mut()?
                .ensure_current_node(EnsureNodeType::Action(played_action.idx), game_state)?;
        }

        // Use the current game state and the played action to get the index of the
        // action in the current node's children. We will take that path
        // next.
        let action_idx = self.played_action_to_idx(game_state, &played_action);
        self.state.try_borrow_mut()?.set_next_node(action_idx)
    }
}

impl Historian for ArenaCFRHistorian {
    fn record_action(
        &mut self,
        _id: &uuid::Uuid,
        game_state: &GameState,
        action: Action,
    ) -> Result<(), HistorianError> {
        // Record the action in the game tree
        match action {
            Action::PlayedAction(played_action) => {
                self.handle_played_action_payload(game_state, played_action)
            }
            Action::FailedAction(failed_action) => {
                // An agent failed to take an appropriate action
                // handle the result
                self.handle_played_action_payload(game_state, failed_action.result)
            }
            Action::DealStartingHand(DealStartingHandPayload { card, idx, .. }) => {
                // Only record the action if it's the player's action
                // So we can't get information leakage
                if idx == self.player_idx {
                    // This is a chance node
                    self.state
                        .try_borrow_mut()?
                        .ensure_current_node(EnsureNodeType::Chance, game_state)?;
                    self.state
                        .try_borrow_mut()?
                        .set_next_node(u8::from(card) as usize)
                } else {
                    Ok(())
                }
            }
            Action::DealCommunity(card) => {
                // This is chance node
                self.state
                    .try_borrow_mut()?
                    .ensure_current_node(EnsureNodeType::Chance, game_state)?;
                self.state
                    .try_borrow_mut()?
                    .set_next_node(u8::from(card) as usize)?;
                Ok(())
            }
            Action::RoundAdvance(Round::Complete) => {
                // This is a terminal node
                self.state
                    .try_borrow_mut()?
                    .ensure_current_node(EnsureNodeType::Terminal, game_state)?;

                self.handle_terminal_node(game_state);
                Ok(())
            }
            // The rest of the actions are ignored (Partial payouts, or sitting down).
            Action::GameStart(_) | Action::ForcedBet(_) | Action::PlayerSit(_) => Ok(()),
            Action::RoundAdvance(_) | Action::Award(_) => Ok(()),
        }
    }
}
