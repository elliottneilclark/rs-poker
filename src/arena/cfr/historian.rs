use tracing::event;

use crate::arena::action::Action;
use crate::arena::action::PlayedActionPayload;
use crate::arena::game_state::Round;

use crate::arena::Historian;
use crate::core::Card;

use crate::arena::GameState;

use crate::arena::HistorianError;

use super::ActionGenerator;
use super::ActionIndexMapper;
use super::NodeData;
use super::PlayerData;
use super::StateStore;
use super::TerminalData;

/// The `CFRHistorian` struct is responsible for managing the state and actions
/// within the Counterfactual Regret Minimization (CFR) algorithm for poker
/// games.
///
/// This historian updates ALL players' CFR states and traversal states when
/// recording actions. Each player has their own tree, but the historian
/// ensures they all advance appropriately based on the action type:
/// - Public actions (bets, community cards): all players advance
/// - Private actions (hole cards): only the specific player advances
/// - Terminal: each player records their own reward
///
/// # Type Parameters
/// - `T`: A type that implements the `ActionGenerator` trait, used to generate
///   actions based on the current game state.
pub struct CFRHistorian<T>
where
    T: ActionGenerator,
{
    state_store: StateStore,
    #[allow(dead_code)]
    action_generator: T,
    /// The action index mapper for consistent action-to-index mapping.
    action_index_mapper: ActionIndexMapper,
    /// Whether to allow mutating node types when a mismatch is found.
    allow_node_mutation: bool,
}

impl<T> CFRHistorian<T>
where
    T: ActionGenerator,
{
    /// Create a new CFRHistorian.
    ///
    /// # Arguments
    /// * `state_store` - The shared state store containing all players' states
    /// * `config` - Configuration for the action generator
    /// * `allow_node_mutation` - Whether to allow mutating node types when a mismatch is found
    pub(crate) fn new(
        state_store: StateStore,
        config: T::Config,
        allow_node_mutation: bool,
    ) -> Self {
        // Create action generator using player 0's state.
        // The action mapping is player-independent.
        let cfr_state = state_store
            .get_cfr_state(0)
            .expect("At least one player should exist");
        let traversal_state = state_store
            .peek_traversal(0)
            .expect("At least one player should exist");
        let action_generator = T::new(cfr_state.clone(), traversal_state, config);

        // Create the action index mapper from the CFR state's mapper config
        let mapper_config = cfr_state.mapper_config();
        let action_index_mapper = ActionIndexMapper::new(mapper_config);

        CFRHistorian {
            state_store,
            action_generator,
            action_index_mapper,
            allow_node_mutation,
        }
    }

    /// Ensure target node exists for a specific player and return the node index.
    /// This increments the visit count and creates or updates the node as needed.
    ///
    /// Uses `CFRState::ensure_child` to handle the case where different bet amounts
    /// map to the same index but lead to different outcomes (e.g., all-in vs not).
    fn ensure_target_node_for_player(
        &self,
        player_idx: usize,
        node_data: NodeData,
    ) -> Result<usize, HistorianError> {
        let mut cfr_state = self
            .state_store
            .get_cfr_state(player_idx)
            .ok_or(HistorianError::CFRNodeNotFound)?;
        let traversal_state = self
            .state_store
            .peek_traversal(player_idx)
            .ok_or(HistorianError::CFRNodeNotFound)?;

        // Get both fields in a single lock acquisition
        let (from_node_idx, from_child_idx) = traversal_state.get_position();

        // Increment the count of the node we are coming from
        cfr_state
            .increment_count(from_node_idx, from_child_idx)
            .map_err(|_| HistorianError::CFRNodeNotFound)?;

        // Use ensure_child which handles node type mismatches
        Ok(cfr_state.ensure_child(
            from_node_idx,
            from_child_idx,
            node_data,
            self.allow_node_mutation,
        ))
    }

    /// Move a specific player's traversal state to the target node.
    fn move_player_to(&self, player_idx: usize, to_node_idx: usize, child_idx: usize) {
        if let Some(mut traversal) = self.state_store.peek_traversal(player_idx) {
            traversal.move_to(to_node_idx, child_idx);
        }
    }

    /// Record a community card (flop, turn, river).
    /// All players' traversal states move since community cards are public.
    pub(crate) fn record_community_card(&self, card: Card) -> Result<(), HistorianError> {
        let card_value: u8 = card.into();

        for player_idx in 0..self.state_store.num_players() {
            let to_node_idx = self.ensure_target_node_for_player(player_idx, NodeData::Chance)?;
            self.move_player_to(player_idx, to_node_idx, card_value as usize);
        }
        Ok(())
    }

    /// Record a starting hand card being dealt.
    ///
    /// All players' traversal states advance for every hole card dealt to any player.
    /// There is no information hiding - the full deal sequence is recorded in the tree.
    /// This works because agents only look forward (at available actions), never backward
    /// at the deal history, and it produces a cleaner tree for analysis.
    pub(crate) fn record_starting_hand_card(&self, card: Card) -> Result<(), HistorianError> {
        let card_value: u8 = card.into();

        for idx in 0..self.state_store.num_players() {
            let to_node_idx = self.ensure_target_node_for_player(idx, NodeData::Chance)?;
            self.move_player_to(idx, to_node_idx, card_value as usize);
        }
        Ok(())
    }

    /// Record a player action (bet, call, fold, etc.).
    /// All players' traversal states move since actions are public.
    ///
    /// Uses pre-action game state values from the payload to ensure consistent
    /// action indexing. This is critical because the simulation modifies the game
    /// state before calling record_action, but action_to_idx depends on the
    /// state WHEN the action was chosen, not AFTER it was applied.
    pub(crate) fn record_action(
        &self,
        _game_state: &GameState,
        payload: &PlayedActionPayload,
    ) -> Result<(), HistorianError> {
        // Compute pre-action values needed for action indexing.
        // We use the raw values directly instead of cloning the entire GameState.
        let pre_action_round_bet = payload.starting_bet;
        let pre_action_player_bet = payload.starting_player_bet;
        // Compute pre-action stack: post-action stack + amount bet this action
        let amount_bet = payload.final_player_bet - payload.starting_player_bet;
        let pre_action_stack = payload.player_stack + amount_bet;

        // Use the raw-value action-to-index mapping (avoids GameState clone)
        let action_idx = self.action_index_mapper.action_to_idx_raw(
            &payload.action,
            pre_action_round_bet,
            pre_action_player_bet,
            pre_action_stack,
        );

        event!(
            tracing::Level::TRACE,
            acting_player_idx = payload.idx,
            ?payload.action,
            action_idx,
            num_players = self.state_store.num_players(),
            starting_pot = payload.starting_pot,
            starting_bet = payload.starting_bet,
            "Recording action for all players"
        );

        for player_idx in 0..self.state_store.num_players() {
            let to_node_idx = self.ensure_target_node_for_player(
                player_idx,
                NodeData::Player(PlayerData {
                    regret_matcher: Option::default(),
                    player_idx: payload.idx as u8,
                }),
            )?;
            self.move_player_to(player_idx, to_node_idx, action_idx);
        }
        Ok(())
    }

    /// Record terminal state.
    /// Each player records their own reward in their own tree.
    pub(crate) fn record_terminal(&self, game_state: &GameState) -> Result<(), HistorianError> {
        for player_idx in 0..self.state_store.num_players() {
            let to_node_idx = self.ensure_target_node_for_player(
                player_idx,
                NodeData::Terminal(TerminalData::default()),
            )?;
            self.move_player_to(player_idx, to_node_idx, 0);

            let reward = game_state.player_reward(player_idx);

            event!(
                tracing::Level::TRACE,
                to_node_idx,
                reward,
                player_idx,
                "Recording terminal node"
            );

            let mut cfr_state = self
                .state_store
                .get_cfr_state(player_idx)
                .ok_or(HistorianError::CFRNodeNotFound)?;

            // For terminal nodes we repurpose the child visited counter.
            cfr_state
                .increment_count(to_node_idx, 0)
                .map_err(|_| HistorianError::CFRNodeNotFound)?;

            cfr_state
                .update_node(to_node_idx, |node| {
                    if let NodeData::Terminal(td) = &mut node.data {
                        td.total_utility += reward;
                    }
                })
                .map_err(|_| HistorianError::CFRNodeNotFound)?;
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
            // Rather than use award since it can be for a side pot we use the final award amount
            // in the terminal node.
            Action::Award(_) => Ok(()),
            Action::DealStartingHand(payload) => self.record_starting_hand_card(payload.card),
            Action::PlayedAction(payload) => {
                CFRHistorian::record_action(self, game_state, &payload)
            }
            Action::FailedAction(failed_action_payload) => {
                CFRHistorian::record_action(self, game_state, &failed_action_payload.result)
            }
            // Community cards are public - all traversal states move together
            Action::DealCommunity(card) => self.record_community_card(card),
        }
    }
}
