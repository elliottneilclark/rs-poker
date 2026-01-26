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
        self.traversal_states[player_idx] = vec![TraversalState::new_root(player_idx)];
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
    pub fn new() -> Self {
        StateStore {
            inner: Arc::new(RwLock::new(StateStoreInternal {
                cfr_states: Vec::new(),
                traversal_states: Vec::new(),
            })),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.read().unwrap().cfr_states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

    /// Check if a player has CFR state initialized.
    /// Returns true if the player has both CFR state and traversal state.
    pub fn has_player_state(&self, player_idx: usize) -> bool {
        self.inner.read().unwrap().has_player_state(player_idx)
    }

    /// Ensure CFR state exists for all players from 0 to num_players-1.
    /// This is needed when a CFR agent plays against non-CFR agents,
    /// because the CFR agent's regret computation assumes all players
    /// have CFR state to simulate optimal play.
    pub fn ensure_all_players(&mut self, game_state: GameState, num_players: usize) {
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

    /// Initialize CFR state for a player and push an initial traversal state.
    /// Returns clones of the CFR state and the new traversal state.
    pub fn new_state(
        &mut self,
        game_state: GameState,
        player_idx: usize,
    ) -> (CFRState, TraversalState) {
        {
            let mut inner = self.inner.write().unwrap();
            inner.init_player_state(&game_state, player_idx);
        }
        self.push_traversal(player_idx)
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

        let new_traversal_state =
            TraversalState::new(last.node_idx(), last.chosen_child_idx(), last.player_idx());

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

impl Default for StateStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_new() {
        let store = StateStore::new();
        assert_eq!(store.len(), 0, "New state store should have no states");
    }

    #[test]
    fn test_push() {
        let mut state_store = StateStore::new();
        let game_state = GameState::new_starting(vec![100.0; 3], 10.0, 5.0, 0.0, 0);
        let (state, _traversal) = state_store.new_state(game_state.clone(), 0);
        assert_eq!(
            state_store.len(),
            1,
            "State store should have one state after push"
        );
        assert_eq!(
            state.starting_game_state(),
            game_state,
            "State should match the game state"
        );
    }

    #[test]
    fn test_push_len() {
        let mut state_store = StateStore::new();

        let game_state = GameState::new_starting(vec![100.0; 3], 10.0, 5.0, 0.0, 0);

        let _stores = (0..2)
            .map(|i| {
                let (state, traversal) = state_store.new_state(game_state.clone(), i);
                assert_eq!(
                    state_store.len(),
                    i + 1,
                    "State store should have one state after push"
                );
                (state, traversal)
            })
            .collect::<Vec<_>>();

        assert_eq!(2, state_store.len(), "State store should have two states");

        let mut store_clones = (0..2).map(|_| state_store.clone()).collect::<Vec<_>>();

        for (player_idx, cloned_state_store) in store_clones.iter_mut().enumerate() {
            assert_eq!(
                cloned_state_store.len(),
                2,
                "Cloned state store should have two states"
            );

            let (_, _) = cloned_state_store.push_traversal(player_idx);
            assert_eq!(
                cloned_state_store.len(),
                2,
                "Cloned state store should still have two states"
            );
        }

        for i in 0..2 {
            state_store.pop_traversal(i);
        }
    }
}
