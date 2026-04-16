use tracing::event;

use crate::arena::action::Action;
use crate::arena::action::PlayedActionPayload;
use crate::arena::game_state::Round;

use crate::arena::Historian;

use crate::arena::GameState;

use crate::arena::HistorianError;

use super::ActionIndexMapper;
use super::CFRState;
use super::NodeData;
use super::PlayerData;
use super::TerminalData;
use super::TraversalSet;

/// The `CFRHistorian` struct is responsible for managing the state and actions
/// within the Counterfactual Regret Minimization (CFR) algorithm for poker
/// games.
///
/// This historian updates ALL players' traversal states when recording actions.
/// All players share a single CFR tree (NodeArena). The historian ensures
/// traversal states advance appropriately based on the action type:
/// - Public actions (bets, community cards): all traversal states advance
/// - Private actions (hole cards): all traversal states advance (the full deal
///   sequence is recorded in the tree; there is no information hiding)
/// - Terminal: records reward in the shared terminal node
pub struct CFRHistorian {
    /// Single shared CFR state for the entire game tree.
    /// All players share this tree; only regret data differs per player.
    cfr_state: CFRState,
    traversal_set: TraversalSet,
    /// The action index mapper for consistent action-to-index mapping.
    action_index_mapper: ActionIndexMapper,
    /// Whether to allow mutating node types when a mismatch is found.
    allow_node_mutation: bool,
}

impl CFRHistorian {
    /// Create a new CFRHistorian.
    ///
    /// # Arguments
    /// * `cfr_state` - The single shared CFR state for all players
    /// * `traversal_set` - The traversal set tracking each player's position
    /// * `allow_node_mutation` - Whether to allow mutating node types when a mismatch is found
    pub(crate) fn new(
        cfr_state: &CFRState,
        traversal_set: TraversalSet,
        allow_node_mutation: bool,
    ) -> Self {
        // Create the action index mapper from the shared state's mapper config
        let action_index_mapper = ActionIndexMapper::new(*cfr_state.mapper_config());

        CFRHistorian {
            cfr_state: cfr_state.clone(),
            traversal_set,
            action_index_mapper,
            allow_node_mutation,
        }
    }

    /// Ensure target node exists in the shared tree using the first player's
    /// traversal position, and return the node index.
    ///
    /// Since all players share one tree and their traversal states are always
    /// in sync (the historian moves them all together), we use player 0's
    /// position to create/find the node.
    fn ensure_target_node(&self, node_data: NodeData) -> Result<usize, HistorianError> {
        let traversal_state = self.traversal_set.get(0);
        let (from_node_idx, from_child_idx) = traversal_state.get_position();

        Ok(self.cfr_state.ensure_child(
            from_node_idx,
            from_child_idx,
            node_data,
            self.allow_node_mutation,
        ))
    }

    /// Move all players' traversal states to the target node.
    fn move_all_players_to(&self, to_node_idx: usize, child_idx: usize) {
        for player_idx in 0..self.traversal_set.num_players() {
            self.traversal_set
                .get(player_idx)
                .move_to(to_node_idx, child_idx);
        }
    }

    /// Record a community card (flop, turn, river).
    /// All players' traversal states move since community cards are public.
    pub(crate) fn record_community_card(
        &self,
        card: crate::core::Card,
    ) -> Result<(), HistorianError> {
        let card_value: u8 = card.into();
        let to_node_idx = self.ensure_target_node(NodeData::Chance)?;
        self.move_all_players_to(to_node_idx, card_value as usize);
        Ok(())
    }

    /// Record a starting hand card being dealt.
    ///
    /// All players' traversal states advance for every hole card dealt to any player.
    /// There is no information hiding - the full deal sequence is recorded in the tree.
    pub(crate) fn record_starting_hand_card(
        &self,
        card: crate::core::Card,
    ) -> Result<(), HistorianError> {
        let card_value: u8 = card.into();
        let to_node_idx = self.ensure_target_node(NodeData::Chance)?;
        self.move_all_players_to(to_node_idx, card_value as usize);
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
        let pre_action_round_bet = payload.starting_bet;
        let pre_action_player_bet = payload.starting_player_bet;
        let amount_bet = payload.final_player_bet - payload.starting_player_bet;
        let pre_action_stack = payload.player_stack + amount_bet;

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
            num_players = self.traversal_set.num_players(),
            starting_pot = payload.starting_pot,
            starting_bet = payload.starting_bet,
            "Recording action for all players"
        );

        let to_node_idx = self.ensure_target_node(NodeData::Player(PlayerData {
            regret_matcher: Option::default(),
            player_idx: payload.idx as u8,
        }))?;
        self.move_all_players_to(to_node_idx, action_idx);
        Ok(())
    }

    /// Record terminal state.
    /// Creates a single terminal node in the shared tree and accumulates
    /// all players' rewards into it.
    pub(crate) fn record_terminal(&self, game_state: &GameState) -> Result<(), HistorianError> {
        let to_node_idx = self.ensure_target_node(NodeData::Terminal(TerminalData::default()))?;
        self.move_all_players_to(to_node_idx, 0);

        // Accumulate all players' rewards into the shared terminal node.
        // Note: total_utility in a shared tree is the sum of all players' rewards
        // (which should net to zero in a zero-sum game). This field is only used
        // for export/visualization, not by the CFR algorithm itself.
        let total_reward: f32 = (0..self.traversal_set.num_players())
            .map(|idx| game_state.player_reward(idx))
            .sum();

        if total_reward != 0.0 {
            self.cfr_state
                .update_node(to_node_idx, |data| {
                    if let NodeData::Terminal(td) = data {
                        td.total_utility += total_reward;
                    }
                })
                .map_err(|_| HistorianError::CFRNodeNotFound)?;
        }

        Ok(())
    }
}

impl Historian for CFRHistorian {
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
