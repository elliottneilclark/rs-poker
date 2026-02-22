use std::sync::Arc;

use parking_lot::RwLock;

/// The internal state for tracking traversal through the CFR tree.
///
/// This struct holds the mutable state that gets shared between clones
/// of `TraversalState` via `Arc<RwLock<>>`. It tracks:
/// - Current position in the tree (node index)
/// - Which branch we're traversing (child index)
/// - Which player this traversal belongs to
#[derive(Debug, PartialEq)]
pub struct TraversalStateInternal {
    // What node are we at
    pub node_idx: usize,
    // Which branch of the children are we currently going down?
    //
    // After a card is dealt or a player acts this will be set to the
    // index of the child node we are going down. This allows us to
    // lazily create the next node in the tree.
    //
    // For root nodes we assume that the first child is always taken.
    // So we will go down index 0 in the children array for all root nodes.
    pub chosen_child_idx: usize,
    // What player are we
    // This allows us to ignore
    // starting hands for others.
    pub player_idx: u8,
}

/// State for tracking position during CFR tree traversal.
///
/// This struct wraps the internal state in an `Arc<RwLock<>>` so that
/// clones share the same underlying state. This is important because
/// both the agent and historian need to track the same position in
/// the tree during a simulation.
///
/// # Examples
///
/// ```
/// use rs_poker::arena::cfr::TraversalState;
///
/// // Create a new traversal starting at the root for player 0
/// let traversal = TraversalState::new_root(0);
///
/// assert_eq!(traversal.node_idx(), 0);
/// assert_eq!(traversal.chosen_child_idx(), 0);
/// assert_eq!(traversal.player_idx(), 0);
///
/// // Move to a new position
/// traversal.move_to(5, 2);
///
/// assert_eq!(traversal.node_idx(), 5);
/// assert_eq!(traversal.chosen_child_idx(), 2);
/// ```
///
/// Cloned traversal states share the same position:
///
/// ```
/// use rs_poker::arena::cfr::TraversalState;
///
/// let traversal = TraversalState::new_root(0);
/// let cloned = traversal.clone();
///
/// traversal.move_to(10, 3);
///
/// // Both see the new position
/// assert_eq!(traversal.node_idx(), 10);
/// assert_eq!(cloned.node_idx(), 10);
/// ```
#[derive(Debug, Clone)]
pub struct TraversalState {
    inner_state: Arc<RwLock<TraversalStateInternal>>,
}

impl PartialEq for TraversalState {
    fn eq(&self, other: &Self) -> bool {
        *self.inner_state.read() == *other.inner_state.read()
    }
}

impl Eq for TraversalState {}

impl TraversalState {
    /// Create a new traversal state at a specific position.
    ///
    /// # Arguments
    ///
    /// * `node_idx` - The index of the current node in the tree
    /// * `chosen_child_idx` - The index of the child branch being traversed
    /// * `player_idx` - The player this traversal belongs to
    pub fn new(node_idx: usize, chosen_child_idx: usize, player_idx: u8) -> Self {
        TraversalState {
            inner_state: Arc::new(RwLock::new(TraversalStateInternal {
                node_idx,
                chosen_child_idx,
                player_idx,
            })),
        }
    }

    /// Create a new traversal state at the root node for a specific player.
    ///
    /// This initializes the traversal at node index 0 with child index 0.
    ///
    /// # Arguments
    ///
    /// * `player_idx` - The player this traversal belongs to
    pub fn new_root(player_idx: u8) -> Self {
        TraversalState::new(0, 0, player_idx)
    }

    /// Get the current node index in the tree.
    pub fn node_idx(&self) -> usize {
        self.inner_state.read().node_idx
    }

    /// Get the player index this traversal belongs to.
    pub fn player_idx(&self) -> u8 {
        self.inner_state.read().player_idx
    }

    /// Get the index of the child branch currently being traversed.
    pub fn chosen_child_idx(&self) -> usize {
        self.inner_state.read().chosen_child_idx
    }

    /// Move the traversal to a new position in the tree.
    ///
    /// Takes `&self` because the underlying state is shared via `Arc<RwLock<>>`.
    /// Cloned handles see the mutation immediately.
    pub fn move_to(&self, node_idx: usize, chosen_child_idx: usize) {
        let mut state = self.inner_state.write();
        state.node_idx = node_idx;
        state.chosen_child_idx = chosen_child_idx;
    }

    /// Get all traversal state fields in a single lock acquisition.
    ///
    /// This is more efficient than calling `node_idx()`, `chosen_child_idx()`,
    /// and `player_idx()` separately when you need multiple values.
    ///
    /// Returns (node_idx, chosen_child_idx, player_idx).
    #[inline]
    pub fn get_all(&self) -> (usize, usize, u8) {
        let state = self.inner_state.read();
        (state.node_idx, state.chosen_child_idx, state.player_idx)
    }

    /// Get node_idx and chosen_child_idx in a single lock acquisition.
    ///
    /// This is more efficient than calling both getters separately.
    ///
    /// Returns (node_idx, chosen_child_idx).
    #[inline]
    pub fn get_position(&self) -> (usize, usize) {
        let state = self.inner_state.read();
        (state.node_idx, state.chosen_child_idx)
    }
}

/// A set of traversal states for all players in a game.
///
/// `TraversalSet` holds one `TraversalState` per player. Since `TraversalState`
/// uses `Arc<RwLock<>>` internally:
/// - **`clone()`** is shallow (Arc-clone): the clone shares the same underlying
///   state. This is used so that the agent, historian, and sim builder all
///   track the same positions.
/// - **`fork()`** creates a deep copy: new independent `TraversalState` Arcs
///   at the same positions. Used for sub-simulation isolation (replaces the
///   old push/pop traversal stack).
///
/// # Examples
///
/// ```
/// use rs_poker::arena::cfr::TraversalSet;
///
/// let set = TraversalSet::new(3);
/// assert_eq!(set.num_players(), 3);
///
/// // Each player starts at root (node 0, child 0)
/// let ts = set.get(0);
/// assert_eq!(ts.node_idx(), 0);
/// ```
///
/// Forking creates independent copies:
///
/// ```
/// use rs_poker::arena::cfr::TraversalSet;
///
/// let set = TraversalSet::new(2);
/// let forked = set.fork();
///
/// // Mutating the fork doesn't affect the original
/// forked.get(0).move_to(10, 3);
/// assert_eq!(set.get(0).node_idx(), 0);
/// assert_eq!(forked.get(0).node_idx(), 10);
/// ```
#[derive(Debug, Clone)]
pub struct TraversalSet {
    states: Vec<TraversalState>,
}

impl TraversalSet {
    /// Create a new `TraversalSet` with one root traversal state per player.
    pub fn new(num_players: usize) -> Self {
        let states = (0..num_players)
            .map(|i| TraversalState::new_root(i as u8))
            .collect();
        TraversalSet { states }
    }

    /// Get the traversal state for a specific player.
    ///
    /// This returns a clone of the `TraversalState`, which shares the same
    /// underlying `Arc<RwLock<>>` — mutations through either handle are visible
    /// to both.
    pub fn get(&self, player_idx: usize) -> TraversalState {
        self.states[player_idx].clone()
    }

    /// Create a deep copy (fork) of this traversal set.
    ///
    /// Each player gets a new independent `TraversalState` initialized at the
    /// same position as the original. Changes to the fork do not affect the
    /// original, and vice versa.
    ///
    /// This is used for sub-simulation isolation: before running a
    /// sub-simulation, fork the set so the sub-sim can freely mutate positions
    /// without affecting the parent.
    pub fn fork(&self) -> TraversalSet {
        let states = self
            .states
            .iter()
            .map(|ts| {
                let (node_idx, chosen_child_idx, player_idx) = ts.get_all();
                TraversalState::new(node_idx, chosen_child_idx, player_idx)
            })
            .collect();
        TraversalSet { states }
    }

    /// Return the number of players in this set.
    pub fn num_players(&self) -> usize {
        self.states.len()
    }

    /// Iterate over references to the traversal states.
    pub fn iter(&self) -> impl Iterator<Item = &TraversalState> {
        self.states.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::{TraversalSet, TraversalState};

    #[test]
    fn test_new_and_getters() {
        let traversal = TraversalState::new(5, 10, 2);

        assert_eq!(traversal.node_idx(), 5);
        assert_eq!(traversal.chosen_child_idx(), 10);
        assert_eq!(traversal.player_idx(), 2);
    }

    #[test]
    fn test_new_root() {
        let traversal = TraversalState::new_root(3);

        assert_eq!(traversal.node_idx(), 0);
        assert_eq!(traversal.chosen_child_idx(), 0);
        assert_eq!(traversal.player_idx(), 3);
    }

    #[test]
    fn test_move_to() {
        let traversal = TraversalState::new_root(0);

        assert_eq!(traversal.node_idx(), 0);
        assert_eq!(traversal.chosen_child_idx(), 0);

        traversal.move_to(42, 7);

        assert_eq!(traversal.node_idx(), 42);
        assert_eq!(traversal.chosen_child_idx(), 7);
        // player_idx should remain unchanged
        assert_eq!(traversal.player_idx(), 0);
    }

    #[test]
    fn test_cloned_traversal_share_loc() {
        let traversal = TraversalState::new(0, 0, 0);
        let cloned = traversal.clone();

        assert_eq!(traversal.node_idx(), 0);
        assert_eq!(traversal.player_idx(), 0);
        assert_eq!(traversal.chosen_child_idx(), 0);

        assert_eq!(cloned.node_idx(), 0);
        assert_eq!(cloned.player_idx(), 0);
        assert_eq!(cloned.chosen_child_idx(), 0);

        // Simulate traversing the tree
        traversal.move_to(2, 42);

        assert_eq!(traversal.node_idx(), 2);
        assert_eq!(traversal.chosen_child_idx(), 42);

        // Cloned should have the same values
        assert_eq!(cloned.node_idx(), 2);
        assert_eq!(cloned.chosen_child_idx(), 42);
    }

    #[test]
    fn test_get_all_after_move() {
        let traversal = TraversalState::new(0, 0, 2);
        traversal.move_to(100, 50);

        let (node_idx, chosen_child_idx, player_idx) = traversal.get_all();

        assert_eq!(node_idx, 100);
        assert_eq!(chosen_child_idx, 50);
        assert_eq!(player_idx, 2); // Unchanged
    }

    // TraversalSet tests

    #[test]
    fn test_traversal_set_new() {
        let set = TraversalSet::new(3);
        assert_eq!(set.num_players(), 3);

        for i in 0..3 {
            let ts = set.get(i);
            assert_eq!(ts.node_idx(), 0);
            assert_eq!(ts.chosen_child_idx(), 0);
            assert_eq!(ts.player_idx(), i as u8);
        }
    }

    #[test]
    fn test_clone_shares_state() {
        let set = TraversalSet::new(2);
        let cloned = set.clone();

        // Mutate through the original
        set.get(0).move_to(10, 3);

        // Clone should see the mutation (Arc sharing)
        assert_eq!(cloned.get(0).node_idx(), 10);
        assert_eq!(cloned.get(0).chosen_child_idx(), 3);
    }

    #[test]
    fn test_fork_is_independent() {
        let set = TraversalSet::new(2);
        // Move player 0 to a non-root position
        set.get(0).move_to(5, 2);

        let forked = set.fork();

        // Forked should start at the same position
        assert_eq!(forked.get(0).node_idx(), 5);
        assert_eq!(forked.get(0).chosen_child_idx(), 2);

        // Mutate the fork
        forked.get(0).move_to(20, 7);

        // Original should be unaffected
        assert_eq!(set.get(0).node_idx(), 5);
        assert_eq!(set.get(0).chosen_child_idx(), 2);

        // Fork should reflect the mutation
        assert_eq!(forked.get(0).node_idx(), 20);
        assert_eq!(forked.get(0).chosen_child_idx(), 7);
    }
}
