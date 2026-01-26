use tracing::event;

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
/// - `traversal_states`: The traversal states for ALL players in the game.
///   This is needed because when CFR agents simulate games, they need all
///   players to start at the correct tree position.
/// - `owning_player_idx`: The player index that owns this historian. Used to
///   determine which player's starting hand cards to record (info set privacy).
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
    /// Traversal states for all players. All states move through the tree
    /// together so sub-agents start at the correct position.
    pub traversal_states: Vec<TraversalState>,
    /// The player index that owns this historian (for card privacy).
    pub owning_player_idx: usize,
    pub cfr_state: CFRState,
    pub action_generator: T,
}

impl<T> CFRHistorian<T>
where
    T: ActionGenerator,
{
    /// Create a new CFRHistorian for a single player (legacy constructor).
    ///
    /// This is kept for backwards compatibility with tests that manually
    /// create a single traversal state. For production use, prefer
    /// `new_multi_player()`.
    pub(crate) fn new(traversal_state: TraversalState, cfr_state: CFRState) -> Self {
        let owning_player_idx = traversal_state.player_idx();
        let action_generator = T::new(cfr_state.clone(), traversal_state.clone());

        // In single-player mode, we create a sparse vec where the traversal
        // state is at its actual player index position. Fill earlier slots
        // with clones (they won't be used since we only update owning player).
        let mut traversal_states = Vec::with_capacity(owning_player_idx + 1);
        for _ in 0..owning_player_idx {
            // These are placeholder states that won't be used
            traversal_states.push(traversal_state.clone());
        }
        traversal_states.push(traversal_state);

        CFRHistorian {
            traversal_states,
            owning_player_idx,
            cfr_state,
            action_generator,
        }
    }

    /// Get the owning player's traversal state (used for tree navigation).
    fn owning_traversal(&self) -> &TraversalState {
        &self.traversal_states[self.owning_player_idx]
    }

    /// Prepare to navigate to a child node. This will increment the count of
    /// the node we are coming from and return the index of the child node
    /// we are navigating to.
    ///
    /// Uses the owning player's traversal state since they own the CFR tree.
    pub(crate) fn ensure_target_node(
        &mut self,
        node_data: NodeData,
    ) -> Result<usize, HistorianError> {
        let from_node_idx = self.owning_traversal().node_idx();
        let from_child_idx = self.owning_traversal().chosen_child_idx();

        // Increment the count of the node we are coming from
        self.cfr_state
            .increment_count(from_node_idx, from_child_idx)
            .map_err(|_| HistorianError::CFRNodeNotFound)?;

        let to = self.cfr_state.get_child(from_node_idx, from_child_idx);

        match to {
            // The node already exists so our work is done here
            Some(t) => Ok(t),
            // The node doesn't exist so we need to create it with the provided data
            //
            // We then wrap it in an Ok so we tell the world how error free we are....
            None => Ok(self.cfr_state.add(from_node_idx, from_child_idx, node_data)),
        }
    }

    /// Move all traversal states to the target node.
    fn move_all_to(&mut self, to_node_idx: usize, child_idx: usize) {
        event!(
            tracing::Level::TRACE,
            to_node_idx,
            child_idx,
            num_players = self.traversal_states.len(),
            "Moving all traversal states"
        );
        for traversal_state in &mut self.traversal_states {
            traversal_state.move_to(to_node_idx, child_idx);
        }
    }

    /// Record a community card (flop, turn, river).
    /// All traversal states move to the same position since community cards
    /// are public information.
    pub(crate) fn record_community_card(&mut self, card: Card) -> Result<(), HistorianError> {
        let card_value: u8 = card.into();
        let to_node_idx = self.ensure_target_node(NodeData::Chance)?;
        self.move_all_to(to_node_idx, card_value as usize);
        Ok(())
    }

    /// Record a starting hand card for a specific player.
    /// Only the owning player's cards are recorded in their traversal state
    /// to preserve information set privacy.
    pub(crate) fn record_starting_hand_card(
        &mut self,
        card: Card,
        player_idx: usize,
    ) -> Result<(), HistorianError> {
        // Only record cards for the owning player (info set privacy)
        if player_idx == self.owning_player_idx {
            let card_value: u8 = card.into();
            let to_node_idx = self.ensure_target_node(NodeData::Chance)?;
            // Only move the owning player's traversal state
            self.traversal_states[self.owning_player_idx].move_to(to_node_idx, card_value as usize);
        }
        Ok(())
    }

    pub(crate) fn record_action(
        &mut self,
        game_state: &GameState,
        action: AgentAction,
        player_idx: usize,
    ) -> Result<(), HistorianError> {
        let action_idx = self.action_generator.action_to_idx(game_state, &action);
        let to_node_idx = self.ensure_target_node(NodeData::Player(PlayerData {
            regret_matcher: Option::default(),
            player_idx,
        }))?;

        event!(
            tracing::Level::TRACE,
            player_idx,
            ?action,
            action_idx,
            to_node_idx,
            "Recording action for all players"
        );

        // Move ALL traversal states to the same position
        // This ensures sub-agents start at the correct tree position
        self.move_all_to(to_node_idx, action_idx);
        Ok(())
    }

    pub(crate) fn record_terminal(&mut self, game_state: &GameState) -> Result<(), HistorianError> {
        let to_node_idx = self.ensure_target_node(NodeData::Terminal(TerminalData::default()))?;

        // Move all traversal states to the terminal node
        self.move_all_to(to_node_idx, 0);

        let reward = game_state.player_reward(self.owning_player_idx);

        event!(
            tracing::Level::TRACE,
            to_node_idx,
            reward,
            owning_player_idx = self.owning_player_idx,
            "Recording terminal node"
        );

        // For terminal nodes we will never have a child so we repurpose
        // the child visited counter.
        self.cfr_state
            .increment_count(to_node_idx, 0)
            .map_err(|_| HistorianError::CFRNodeNotFound)?;

        self.cfr_state
            .update_node(to_node_idx, |node| {
                if let NodeData::Terminal(td) = &mut node.data {
                    td.total_utility += reward;
                }
            })
            .map_err(|_| HistorianError::CFRNodeNotFound)?;

        // Verify the node is actually a terminal node
        let node_data = self
            .cfr_state
            .get_node_data(to_node_idx)
            .ok_or(HistorianError::CFRNodeNotFound)?;

        if !matches!(node_data, NodeData::Terminal(_)) {
            return Err(HistorianError::CFRUnexpectedNode(
                "Expected terminal node".to_string(),
            ));
        }

        Ok(())
    }
}

impl<T> Historian for CFRHistorian<T>
where
    T: ActionGenerator,
{
    fn record_action(
        &mut self,
        _id: u128,
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
                // We only record our own hand to preserve info set privacy.
                // Only the owning player's cards are recorded.
                self.record_starting_hand_card(payload.card, payload.idx)
            }
            Action::PlayedAction(payload) => {
                self.record_action(game_state, payload.action, payload.idx)
            }
            Action::FailedAction(failed_action_payload) => self.record_action(
                game_state,
                failed_action_payload.result.action,
                failed_action_payload.result.idx,
            ),
            // Community cards are public - all traversal states move together
            Action::DealCommunity(card) => self.record_community_card(card),
        }
    }
}
