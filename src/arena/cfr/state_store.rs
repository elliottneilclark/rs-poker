use std::sync::{Arc, RwLock};

use crate::arena::GameState;

use super::CFRState;

#[derive(Debug, Clone)]
struct StateStoreInternal {
    // The tree structure of counter factual regret.
    pub cfr_states: Vec<CFRState>,
}

impl StateStoreInternal {
    /// Check if a player has CFR state initialized.
    fn has_player_state(&self, player_idx: usize) -> bool {
        self.cfr_states.get(player_idx).is_some()
    }

    /// Ensure vectors are large enough to accommodate player_idx.
    /// This handles cases where CFR agents are created for non-sequential
    /// player indices (e.g., when some players are non-CFR agents).
    fn ensure_capacity(&mut self, player_idx: usize, game_state: &GameState) {
        while self.cfr_states.len() <= player_idx {
            // Add placeholder states that will be replaced when actually used
            self.cfr_states.push(CFRState::new(game_state.clone()));
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
    }
}

/// `StateStore` is a structure to hold all CFR states for a single game that
/// is being solved. Since all players use the same store it enables reuse of
/// the memory and regret matchers of all players.
///
/// Traversal state is managed separately via `TraversalSet`.
#[derive(Debug, Clone)]
pub struct StateStore {
    inner: Arc<RwLock<StateStoreInternal>>,
}

impl StateStore {
    /// Create a new StateStore initialized for all players in the game.
    ///
    /// This is the preferred way to create a StateStore for CFR simulations.
    /// The store is initialized with CFR state for all players,
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

    /// Get clones of all CFR states in a single lock acquisition.
    ///
    /// This is more efficient than calling `get_cfr_state` N times
    /// when you need states for all players (e.g., in CFR sub-agent construction).
    pub fn get_all_cfr_states(&self) -> Vec<CFRState> {
        self.inner.read().unwrap().cfr_states.clone()
    }

    /// Get the mapper configuration from player 0's CFR state.
    ///
    /// All players share the same mapper configuration since it's derived
    /// from the game state. Returns None if no players are initialized.
    pub fn mapper_config(&self) -> Option<super::ActionIndexMapperConfig> {
        self.get_cfr_state(0).map(|state| state.mapper_config())
    }

    /// Check if a player has CFR state initialized.
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
    fn test_num_players() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(3, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let state_store = StateStore::new(game_state);
        assert_eq!(state_store.num_players(), 3);
    }
}
