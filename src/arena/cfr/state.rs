use std::sync::Arc;

use crate::arena::{GameState, errors::CFRStateError};

use super::node_arena::NodeArena;
use super::{ActionIndexMapperConfig, Node, NodeData};

/// Counterfactual Regret Minimization (CFR) state tracker.
///
/// This struct manages the game tree used for CFR algorithm calculations. The
/// tree is built lazily as actions are taken in the game. Each node in the tree
/// represents a game state and stores regret values used by the CFR algorithm.
///
/// Nodes are stored in a chunked arena (`NodeArena`) that provides:
/// - **Lock-free reads**: `get_child`, `with_node_data` use only atomic loads
///   and per-node read locks (no global lock).
/// - **Per-node write locks**: `update_node` locks only the target node's data,
///   not the entire arena.
/// - **Mutex-protected appends**: New node allocation uses a mutex only when
///   a new chunk is needed (rare after warmup).
///
/// The `starting_game_state` and `mapper_config` are immutable after
/// construction and require no locking.
///
/// # Examples
///
/// ```
/// use rs_poker::arena::GameStateBuilder;
/// use rs_poker::arena::cfr::CFRState;
///
/// let game_state = GameStateBuilder::new().num_players_with_stack(2, 100.0).blinds(10.0, 5.0).build().unwrap();
/// let cfr_state = CFRState::new(game_state);
/// ```
#[derive(Debug, Clone)]
pub struct CFRState {
    arena: Arc<NodeArena>,
    starting_game_state: Arc<GameState>,
    mapper_config: ActionIndexMapperConfig,
}

impl CFRState {
    pub fn new(game_state: GameState) -> Self {
        let mapper_config = ActionIndexMapperConfig::from_game_state(&game_state);
        let arena = NodeArena::new();
        arena.push(Node::new_root());
        CFRState {
            arena: Arc::new(arena),
            starting_game_state: Arc::new(game_state),
            mapper_config,
        }
    }

    pub fn starting_game_state(&self) -> &GameState {
        &self.starting_game_state
    }

    /// Get the action index mapper configuration for this CFR state.
    ///
    /// This configuration defines the bet range for mapping actions to indices.
    pub fn mapper_config(&self) -> &ActionIndexMapperConfig {
        &self.mapper_config
    }

    /// Add a new child node at the given position.
    ///
    /// # Panics
    ///
    /// Panics if a child already exists at `(parent_idx, child_idx)`.
    /// This method is intended for single-threaded tree construction (e.g., tests).
    /// For concurrent use, prefer `ensure_child`.
    pub fn add(&self, parent_idx: usize, child_idx: usize, data: NodeData) -> usize {
        let node = Node::new(parent_idx, child_idx, data);
        let idx = self.arena.push(node);

        // The parent node needs to be updated to point to the new child.
        // Use try_set_child to get a clear error instead of a swap-then-assert.
        self.arena
            .get(parent_idx)
            .try_set_child(child_idx, idx)
            .unwrap_or_else(|existing| {
                panic!(
                    "Child already set at parent_idx={parent_idx}, child_idx={child_idx}: \
                     existing node index={existing}. Use ensure_child() for concurrent access."
                )
            });

        idx
    }

    /// Get the data for a node at the specified index.
    ///
    /// # Arguments
    ///
    /// * `idx` - The index of the node to retrieve
    ///
    /// # Returns
    ///
    /// * `Some(NodeData)` - A clone of the node's data if it exists
    /// * `None` - If the node doesn't exist at that index
    pub fn get_node_data(&self, idx: usize) -> Option<NodeData> {
        if idx < self.arena.len() {
            Some(self.arena.get(idx).read_data().clone())
        } else {
            None
        }
    }

    /// Access node data without cloning, by calling a closure with a reference.
    ///
    /// This is more efficient than `get_node_data` when you only need to read
    /// the node data (e.g., to access the regret matcher) without owning it.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is out of bounds.
    pub fn with_node_data<F, R>(&self, idx: usize, f: F) -> R
    where
        F: FnOnce(&NodeData) -> R,
    {
        let guard = self.arena.get(idx).read_data();
        f(&guard)
    }

    /// Get the child node index for a given parent node and child index.
    ///
    /// This is fully lock-free: it uses an atomic load on the parent's
    /// child slot.
    ///
    /// # Arguments
    ///
    /// * `parent_idx` - The index of the parent node
    /// * `child_idx` - The child index within the parent's children array
    ///
    /// # Returns
    ///
    /// * `Some(usize)` - The index of the child node if it exists
    /// * `None` - If the parent doesn't exist or the child slot is empty
    pub fn get_child(&self, parent_idx: usize, child_idx: usize) -> Option<usize> {
        if parent_idx < self.arena.len() {
            self.arena.get(parent_idx).get_child(child_idx)
        } else {
            None
        }
    }

    /// Update a node's data using a closure.
    ///
    /// This acquires only the per-node write lock — no global lock is needed.
    ///
    /// # Arguments
    ///
    /// * `node_idx` - The index of the node to update
    /// * `f` - A closure that takes a mutable reference to the node's data
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the update was successful
    /// * `Err(CFRStateError::NodeNotFound)` - If the node doesn't exist
    pub fn update_node<F>(&self, node_idx: usize, f: F) -> Result<(), CFRStateError>
    where
        F: FnOnce(&mut NodeData),
    {
        if node_idx < self.arena.len() {
            let mut guard = self.arena.get(node_idx).write_data();
            f(&mut guard);
            Ok(())
        } else {
            Err(CFRStateError::NodeNotFound)
        }
    }

    /// Access the underlying node arena.
    ///
    /// This method provides access to the arena for advanced
    /// operations like visualization and debugging.
    pub fn arena(&self) -> &Arc<NodeArena> {
        &self.arena
    }

    /// Verify that an existing node's type matches `expected_data`, optionally
    /// mutating the node if a mismatch is found and `allow_mutation` is true.
    ///
    /// # Returns
    ///
    /// The index of the verified (and possibly mutated) node.
    ///
    /// # Panics
    ///
    /// Panics if `allow_mutation` is false and a type mismatch is detected.
    fn verify_existing_child(
        &self,
        existing_idx: usize,
        expected_data: NodeData,
        parent_idx: usize,
        child_idx: usize,
        allow_mutation: bool,
    ) -> usize {
        // Per-node read lock to check type match.
        let data_guard = self.arena.get(existing_idx).read_data();
        let types_match =
            std::mem::discriminant(&*data_guard) == std::mem::discriminant(&expected_data);
        if types_match {
            return existing_idx;
        }
        drop(data_guard);

        // Type mismatch — need per-node write lock to mutate.
        if allow_mutation {
            tracing::debug!(
                parent_idx,
                child_idx,
                existing_idx,
                ?expected_data,
                "Node type mismatch - updating node type. This occurs when different \
                 bet amounts map to the same index but lead to different outcomes."
            );
            let mut data_guard = self.arena.get(existing_idx).write_data();
            *data_guard = expected_data;
        } else {
            let data_guard = self.arena.get(existing_idx).read_data();
            panic!(
                "Node type mismatch at parent_idx={}, child_idx={}: \
                 expected {:?}, found {:?}. This can occur when different bet \
                 amounts map to the same index. Set allow_node_mutation=true \
                 to handle this case.",
                parent_idx, child_idx, expected_data, &*data_guard
            );
        }

        existing_idx
    }

    /// Ensure a child node exists at the given position, creating if needed.
    ///
    /// Uses a lock-free fast path and per-node locking for mutations:
    /// 1. Lock-free atomic load: check if child already exists (fast path)
    /// 2. Per-node read lock: verify type match
    /// 3. Arena push + CAS: allocate new node and atomically set child pointer
    /// 4. If CAS fails, another thread won — verify the winner's node type
    ///
    /// # Arguments
    ///
    /// * `parent_idx` - The index of the parent node
    /// * `child_idx` - The child index within the parent's children array
    /// * `expected_data` - The expected node data type for this position
    /// * `allow_mutation` - If true, update the node type when a mismatch is found.
    ///   If false, panic on mismatch (useful for testing with
    ///   BasicCFRActionGenerator where mismatches indicate bugs).
    ///
    /// # Returns
    ///
    /// The index of the child node (either existing or newly created)
    ///
    /// # Panics
    ///
    /// Panics if `allow_mutation` is false and a node type mismatch is detected.
    pub fn ensure_child(
        &self,
        parent_idx: usize,
        child_idx: usize,
        expected_data: NodeData,
        allow_mutation: bool,
    ) -> usize {
        // Fast path: lock-free atomic load to check if child already exists.
        if let Some(existing_idx) = self.arena.get(parent_idx).get_child(child_idx) {
            return self.verify_existing_child(
                existing_idx,
                expected_data,
                parent_idx,
                child_idx,
                allow_mutation,
            );
        }

        // No child exists — create new node and use CAS to set child.
        let node = Node::new(parent_idx, child_idx, expected_data.clone());
        let idx = self.arena.push(node);

        // Use try_set_child (CAS) to handle concurrent creation.
        // If another thread beat us, verify the winner's node type.
        // The loser's node becomes an orphan in the arena (unreachable but harmless).
        match self.arena.get(parent_idx).try_set_child(child_idx, idx) {
            Ok(()) => idx,
            Err(existing) => self.verify_existing_child(
                existing,
                expected_data,
                parent_idx,
                child_idx,
                allow_mutation,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::cfr::{NodeData, PlayerData};

    use crate::arena::GameStateBuilder;

    use super::CFRState;

    #[test]
    fn test_add_get_node() {
        let state = CFRState::new(
            GameStateBuilder::new()
                .num_players_with_stack(3, 100.0)
                .blinds(10.0, 5.0)
                .build()
                .unwrap(),
        );
        let new_data = NodeData::Player(PlayerData {
            regret_matcher: None,
            player_idx: 0,
        });

        let player_idx: usize = state.add(0, 0, new_data);

        let node_data = state.get_node_data(player_idx).unwrap();
        match &node_data {
            NodeData::Player(pd) => assert!(pd.regret_matcher.is_none()),
            _ => panic!("Expected player data"),
        }

        // assert that the parent node has the correct child idx
        assert_eq!(state.get_child(0, 0), Some(player_idx));
    }

    #[test]
    fn test_node_get_not_exist() {
        let state = CFRState::new(
            GameStateBuilder::new()
                .num_players_with_stack(3, 100.0)
                .blinds(10.0, 5.0)
                .build()
                .unwrap(),
        );
        // root node is always at index 0
        let root = state.get_node_data(0);
        assert!(root.is_some());

        let node = state.get_node_data(100);
        assert!(node.is_none());
    }

    #[test]
    #[should_panic]
    fn test_with_node_data_panics_out_of_bounds() {
        let state = CFRState::new(
            GameStateBuilder::new()
                .num_players_with_stack(2, 100.0)
                .blinds(10.0, 5.0)
                .build()
                .unwrap(),
        );
        // Index 100 doesn't exist, should panic
        state.with_node_data(100, |_| {});
    }

    #[test]
    fn test_with_node_data_reads_correctly() {
        let state = CFRState::new(
            GameStateBuilder::new()
                .num_players_with_stack(2, 100.0)
                .blinds(10.0, 5.0)
                .build()
                .unwrap(),
        );
        // Root node should be readable
        let is_root = state.with_node_data(0, |data| data.is_root());
        assert!(is_root);

        // Add a player node and verify
        let idx = state.add(
            0,
            0,
            NodeData::Player(PlayerData {
                regret_matcher: None,
                player_idx: 1,
            }),
        );
        let is_player = state.with_node_data(idx, |data| data.is_player());
        assert!(is_player);
    }

    #[test]
    fn test_update_node() {
        let state = CFRState::new(
            GameStateBuilder::new()
                .num_players_with_stack(2, 100.0)
                .blinds(10.0, 5.0)
                .build()
                .unwrap(),
        );

        let idx = state.add(
            0,
            0,
            NodeData::Terminal(crate::arena::cfr::TerminalData::default()),
        );

        // Update the terminal utility
        state
            .update_node(idx, |data| {
                if let NodeData::Terminal(td) = data {
                    td.total_utility = 42.0;
                }
            })
            .unwrap();

        // Verify the update
        let utility = state.with_node_data(idx, |data| match data {
            NodeData::Terminal(td) => td.total_utility,
            _ => panic!("Expected terminal"),
        });
        assert_eq!(utility, 42.0);
    }

    #[test]
    fn test_update_node_not_found() {
        let state = CFRState::new(
            GameStateBuilder::new()
                .num_players_with_stack(2, 100.0)
                .blinds(10.0, 5.0)
                .build()
                .unwrap(),
        );
        let result = state.update_node(999, |_| {});
        assert!(result.is_err());
    }

    #[test]
    fn test_ensure_child_creates_new() {
        let state = CFRState::new(
            GameStateBuilder::new()
                .num_players_with_stack(2, 100.0)
                .blinds(10.0, 5.0)
                .build()
                .unwrap(),
        );

        // No child at (0, 5) yet
        assert!(state.get_child(0, 5).is_none());

        // ensure_child should create it
        let idx = state.ensure_child(0, 5, NodeData::Chance, false);
        assert!(idx > 0);
        assert_eq!(state.get_child(0, 5), Some(idx));

        // Calling again should return the same index
        let idx2 = state.ensure_child(0, 5, NodeData::Chance, false);
        assert_eq!(idx, idx2);
    }

    #[test]
    fn test_ensure_child_concurrent_race() {
        use std::sync::Arc;

        let state = Arc::new(CFRState::new(
            GameStateBuilder::new()
                .num_players_with_stack(2, 100.0)
                .blinds(10.0, 5.0)
                .build()
                .unwrap(),
        ));

        let num_threads = 8;
        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let state = state.clone();
                std::thread::spawn(move || state.ensure_child(0, 3, NodeData::Chance, false))
            })
            .collect();

        let indices: Vec<usize> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All threads should agree on the same child index
        let first = indices[0];
        for idx in &indices {
            assert_eq!(*idx, first, "All threads must see the same child node");
        }

        // The child should be at that index
        assert_eq!(state.get_child(0, 3), Some(first));
    }

    #[test]
    fn test_ensure_child_type_mismatch_with_mutation() {
        let state = CFRState::new(
            GameStateBuilder::new()
                .num_players_with_stack(2, 100.0)
                .blinds(10.0, 5.0)
                .build()
                .unwrap(),
        );

        // Create a Chance node
        let idx = state.ensure_child(0, 0, NodeData::Chance, true);

        // Now request a Player node at the same position with mutation allowed
        let idx2 = state.ensure_child(
            0,
            0,
            NodeData::Player(PlayerData {
                regret_matcher: None,
                player_idx: 0,
            }),
            true,
        );

        // Should return the same index (node was mutated in place)
        assert_eq!(idx, idx2);

        // The node should now be a Player node
        let is_player = state.with_node_data(idx, |data| data.is_player());
        assert!(is_player);
    }

    #[test]
    #[should_panic(expected = "Node type mismatch")]
    fn test_ensure_child_type_mismatch_without_mutation_panics() {
        let state = CFRState::new(
            GameStateBuilder::new()
                .num_players_with_stack(2, 100.0)
                .blinds(10.0, 5.0)
                .build()
                .unwrap(),
        );

        // Create a Chance node
        state.ensure_child(0, 0, NodeData::Chance, false);

        // Request a Player node at the same position without mutation - should panic
        state.ensure_child(
            0,
            0,
            NodeData::Player(PlayerData {
                regret_matcher: None,
                player_idx: 0,
            }),
            false,
        );
    }
}
