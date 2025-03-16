use crate::arena::action::Action;
use crate::arena::game_state::Round;

use crate::arena::action::AgentAction;

use crate::arena::Historian;
use crate::core::Card;

use crate::arena::GameState;

use crate::arena::HistorianError;

use super::ActionGenerator;
use super::CFRState;
use super::NodeData;
use super::PlayerData;
use super::TerminalData;
use super::TraversalState;

/// The `CFRHistorian` struct is responsible for managing the state and actions
/// within the Counterfactual Regret Minimization (CFR) algorithm for poker
/// games.
///
/// # Type Parameters
/// - `T`: A type that implements the `ActionGenerator` trait, used to generate
///   actions based on the current game state.
///
/// # Fields
/// - `traversal_state`: The current state of the traversal within the game
///   tree.
/// - `cfr_state`: The current state of the CFR algorithm, including node data
///   and counts.
/// - `action_generator`: An instance of the action generator used to map
///   actions to indices.
///
/// # Trait Implementations
/// - `Historian`: Implements the `Historian` trait, allowing the `CFRHistorian`
///   to record various game actions and states.
pub struct CFRHistorian<T>
where
    T: ActionGenerator,
{
    pub traversal_state: TraversalState,
    pub cfr_state: CFRState,
    pub action_generator: T,
}

impl<T> CFRHistorian<T>
where
    T: ActionGenerator,
{
    pub(crate) fn new(traversal_state: TraversalState, cfr_state: CFRState) -> Self {
        let action_generator = T::new(cfr_state.clone(), traversal_state.clone());
        CFRHistorian {
            traversal_state,
            cfr_state,
            action_generator,
        }
    }

    /// Prepare to navigate to a child node. This will increment the count of
    /// the node we are coming from and return the index of the child node
    /// we are navigating to.
    pub(crate) fn ensure_target_node(
        &mut self,
        node_data: NodeData,
    ) -> Result<usize, HistorianError> {
        let from_node_idx = self.traversal_state.node_idx();
        let from_child_idx = self.traversal_state.chosen_child_idx();
        println!("HISTORRRRRRRRRRR {:?} {:?}", node_data, self.traversal_state);

        // Increment the count of the node we are coming from
        self.cfr_state
            .get_mut(from_node_idx)
            .ok_or(HistorianError::CFRNodeNotFound)?
            .increment_count(from_child_idx);

        let to = self
            .cfr_state
            .get(from_node_idx)
            .ok_or(HistorianError::CFRNodeNotFound)?
            .get_child(from_child_idx);

        match to {
            // The node already exists so our work is done here
            Some(t) => {
                println!("HISTORRRRRRRRRRR exists");
                Ok(t)
            },
            // The node doesn't exist so we need to create it with the provided data
            //
            // We then wrap it in an Ok so we tell the world how error free we are....
            None => {
                println!("HISTORRRRRRRRRRR doesnt exists");
                let t = self.cfr_state.add(from_node_idx, from_child_idx, node_data);
                if t == 5 {
                    println!("HEEEEEEEEEEEEEEEEY");
                    println!("{:p}", *self.cfr_state.internal_state());
                }
                Ok(t)
            },
        }
    }

    pub(crate) fn record_card(
        &mut self,
        _game_state: &GameState,
        card: Card,
    ) -> Result<(), HistorianError> {
        let card_value: u8 = card.into();
        let to_node_idx = self.ensure_target_node(NodeData::Chance)?;
        self.traversal_state
            .move_to(to_node_idx, card_value as usize);

        Ok(())
    }

    pub(crate) fn record_action(
        &mut self,
        _game_state: &GameState,
        action: AgentAction,
        player_idx: usize,
    ) -> Result<(), HistorianError> {
        let action_idx = self.action_generator.action_to_idx(&action);
        let to_node_idx = self.ensure_target_node(NodeData::Player(PlayerData {
            regret_matcher: Option::default(),
            player_idx
        }))?;
        self.traversal_state.move_to(to_node_idx, action_idx);
        Ok(())
    }

    pub(crate) fn record_terminal(&mut self, game_state: &GameState) -> Result<(), HistorianError> {
        let to_node_idx = self.ensure_target_node(NodeData::Terminal(TerminalData::default()))?;
        self.traversal_state.move_to(to_node_idx, 0);

        // Use the current player idx to get the award amount
        let award = game_state.player_winnings[self.traversal_state.player_idx()];

        let mut node = self
            .cfr_state
            .get_mut(to_node_idx)
            .ok_or(HistorianError::CFRNodeNotFound)?;

        // For terminal nodes we will never have a child so we repurpose
        // the child visited counter.
        node.increment_count(0);
        if let NodeData::Terminal(td) = &mut node.data {
            td.total_utility += award;
            Ok(())
        } else {
            Err(HistorianError::CFRUnexpectedNode(
                "Expected terminal node".to_string(),
            ))
        }
    }
}

impl<T> Historian for CFRHistorian<T>
where
    T: ActionGenerator,
{
    fn record_action(
        &mut self,
        _id: &uuid::Uuid,
        game_state: &GameState,
        action: Action,
    ) -> Result<(), HistorianError> {
        match action {
            // These are all assumed from game start and encoded in the root node.
            Action::GameStart(_) | Action::ForcedBet(_) | Action::PlayerSit(_) => Ok(()),
            // For the final round we need to use that to get the final award amount
            Action::RoundAdvance(Round::Complete) => self.record_terminal(game_state),
            // We don't encode round advance in the tree because it never changes the outcome.
            Action::RoundAdvance(_) => Ok(()),
            // Rather than use award since it can be for a side pot we use the final award ammount
            // in the terminal node.
            Action::Award(_) => Ok(()),
            Action::DealStartingHand(payload) => {
                // We only record our own hand
                // so the state can be shared between simulation runs.
                if payload.idx == self.traversal_state.player_idx() {
                    self.record_card(game_state, payload.card)
                } else {
                    Ok(())
                }
            }
            Action::PlayedAction(payload) => {
                self.record_action(game_state, payload.action, payload.idx)
            }
            Action::FailedAction(failed_action_payload) => self.record_action(
                game_state,
                failed_action_payload.result.action,
                failed_action_payload.result.idx,
            ),
            Action::DealCommunity(card) => self.record_card(game_state, card),
        }
    }
}
