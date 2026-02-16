use std::sync::{Arc, RwLock};

use crate::arena::GameState;

use super::{CFRState, TraversalState};

#[derive(Debug, Clone)]
struct StateStoreInternal {
    // The tree structure of counter factual regret.
    pub cfr_states: Vec<CFRState>,

    // The current place in the tree that each player is at. This is used as a stack
    pub traversal_states: Vec<Vec<TraversalState>>,
}

impl StateStoreInternal {
    /// Check if a player has CFR state initialized.
    fn has_player_state(&self, player_idx: usize) -> bool {
        self.cfr_states.get(player_idx).is_some()
            && self
                .traversal_states
                .get(player_idx)
                .is_some_and(|ts| !ts.is_empty())
    }

    /// Ensure vectors are large enough to accommodate player_idx.
    /// This handles cases where CFR agents are created for non-sequential
    /// player indices (e.g., when some players are non-CFR agents).
    fn ensure_capacity(&mut self, player_idx: usize, game_state: &GameState) {
        while self.cfr_states.len() <= player_idx {
            // Add placeholder states that will be replaced when actually used
            self.cfr_states.push(CFRState::new(game_state.clone()));
        }
        while self.traversal_states.len() <= player_idx {
            // Add empty traversal state vectors as placeholders
            self.traversal_states.push(Vec::new());
        }
    }

    /// Ensure CFR state exists for all players from 0 to num_players-1.
    fn ensure_all_players(&mut self, game_state: &GameState, num_players: usize) {
        for i in 0..num_players {
            if !self.has_player_state(i) {
                self.init_player_state(game_state, i);
            }
        }
    }

    /// Initialize CFR state for a specific player.
    fn init_player_state(&mut self, game_state: &GameState, player_idx: usize) {
        self.ensure_capacity(player_idx, game_state);
        self.cfr_states[player_idx] = CFRState::new(game_state.clone());
        self.traversal_states[player_idx] = vec![TraversalState::new_root(player_idx as u8)];
    }
}

/// `StateStore` is a structure to hold all CFR states and other data needed for
/// a single game that is being solved. Since all players use the same store it
/// enables reuse of the memory and regret matchers of all players.
///
/// This state store is not thread safe so it has to be used in a single thread.
#[derive(Debug, Clone)]
pub struct StateStore {
    inner: Arc<RwLock<StateStoreInternal>>,
}

impl StateStore {
    /// Create a new StateStore initialized for all players in the game.
    ///
    /// This is the preferred way to create a StateStore for CFR simulations.
    /// The store is initialized with CFR state and traversal state for all players,
    /// making it ready to be shared across all CFR agents.
    ///
    /// # Example
    /// ```
    /// use rs_poker::arena::GameStateBuilder;
    /// use rs_poker::arena::cfr::StateStore;
    ///
    /// let game_state = GameStateBuilder::new()
    ///     .num_players_with_stack(2, 100.0)
    ///     .blinds(10.0, 5.0)
    ///     .build()
    ///     .unwrap();
    /// let state_store = StateStore::new(game_state);
    /// ```
    pub fn new(game_state: GameState) -> Self {
        let mut store = StateStore {
            inner: Arc::new(RwLock::new(StateStoreInternal {
                cfr_states: Vec::new(),
                traversal_states: Vec::new(),
            })),
        };
        store.ensure_all_players(game_state.clone(), game_state.num_players);
        store
    }

    pub fn len(&self) -> usize {
        self.inner.read().unwrap().cfr_states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the number of players with CFR state initialized.
    ///
    /// This returns the length of the cfr_states vector, which represents
    /// the number of player slots that have been allocated. Note that some
    /// slots may be placeholders if ensure_capacity was called.
    pub fn num_players(&self) -> usize {
        self.inner.read().unwrap().cfr_states.len()
    }

    /// Get a clone of the CFR state for a specific player.
    pub fn get_cfr_state(&self, player_idx: usize) -> Option<CFRState> {
        self.inner
            .read()
            .unwrap()
            .cfr_states
            .get(player_idx)
            .cloned()
    }

    /// Get the mapper configuration from player 0's CFR state.
    ///
    /// All players share the same mapper configuration since it's derived
    /// from the game state. Returns None if no players are initialized.
    pub fn mapper_config(&self) -> Option<super::ActionIndexMapperConfig> {
        self.get_cfr_state(0).map(|state| state.mapper_config())
    }

    /// Check if a player has CFR state initialized.
    /// Returns true if the player has both CFR state and traversal state.
    pub fn has_player_state(&self, player_idx: usize) -> bool {
        self.inner.read().unwrap().has_player_state(player_idx)
    }

    /// Ensure CFR state exists for all players from 0 to num_players-1.
    fn ensure_all_players(&mut self, game_state: GameState, num_players: usize) {
        self.inner
            .write()
            .unwrap()
            .ensure_all_players(&game_state, num_players);
    }

    pub fn traversal_len(&self, player_idx: usize) -> usize {
        self.inner
            .read()
            .unwrap()
            .traversal_states
            .get(player_idx)
            .map_or(0, |traversal| traversal.len())
    }

    pub fn peek_traversal(&self, player_idx: usize) -> Option<TraversalState> {
        self.inner
            .read()
            .unwrap()
            .traversal_states
            .get(player_idx)
            .and_then(|traversal| traversal.last().cloned())
    }

    /// Push a new traversal state onto the stack for a player.
    /// Returns clones of the CFR state and the new traversal state.
    pub fn push_traversal(&mut self, player_idx: usize) -> (CFRState, TraversalState) {
        let mut inner = self.inner.write().unwrap();

        let traversal_states = inner
            .traversal_states
            .get_mut(player_idx)
            .unwrap_or_else(|| panic!("Traversal state for player {player_idx} not found"));

        let last = traversal_states.last().expect("No traversal state found");

        // Use get_all() to get all fields in a single lock acquisition
        let (node_idx, chosen_child_idx, last_player_idx) = last.get_all();
        let new_traversal_state = TraversalState::new(node_idx, chosen_child_idx, last_player_idx);

        traversal_states.push(new_traversal_state.clone());

        let cfr_state = inner
            .cfr_states
            .get(player_idx)
            .unwrap_or_else(|| panic!("State for player {player_idx} not found"))
            .clone();

        (cfr_state, new_traversal_state)
    }

    pub fn pop_traversal(&mut self, player_idx: usize) {
        let mut inner = self.inner.write().unwrap();
        let traversal_states = inner
            .traversal_states
            .get_mut(player_idx)
            .expect("Traversal state for player not found");
        assert!(
            !traversal_states.is_empty(),
            "No traversal state to pop for player {player_idx}"
        );
        traversal_states.pop();
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::arena::GameStateBuilder;

    #[test]
    fn test_new_for_game() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(3, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let state_store = StateStore::new(game_state);
        assert_eq!(
            state_store.num_players(),
            3,
            "State store should have 3 players"
        );
    }

    #[test]
    fn test_push_traversal() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(3, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let mut state_store = StateStore::new(game_state.clone());

        // Initial traversal stack should have 1 entry per player
        assert_eq!(state_store.traversal_len(0), 1);

        // Push a new traversal state
        let (state, _traversal) = state_store.push_traversal(0);
        assert_eq!(state_store.traversal_len(0), 2);
        assert_eq!(
            state.starting_game_state(),
            game_state,
            "State should match the game state"
        );

        // Pop and verify
        state_store.pop_traversal(0);
        assert_eq!(state_store.traversal_len(0), 1);
    }

    #[test]
    fn test_num_players() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(3, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let state_store = StateStore::new(game_state);
        assert_eq!(state_store.num_players(), 3);
    }

    #[test]
    fn test_peeked_traversal_shares_arc() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let state_store = StateStore::new(game_state);

        // Get two references to the same player's traversal state
        let traversal1 = state_store.peek_traversal(0).unwrap();
        let traversal2 = state_store.peek_traversal(0).unwrap();

        // Both should be at the same position
        assert_eq!(traversal1.node_idx(), traversal2.node_idx());
        assert_eq!(traversal1.chosen_child_idx(), traversal2.chosen_child_idx());

        // Move the first one
        let mut traversal1_mut = traversal1;
        traversal1_mut.move_to(5, 3);

        // The second one should also have moved (Arc sharing)
        assert_eq!(traversal2.node_idx(), 5);
        assert_eq!(traversal2.chosen_child_idx(), 3);
    }
}
