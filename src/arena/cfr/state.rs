use std::sync::{Arc, RwLock};

use crate::arena::{GameState, errors::CFRStateError};

use super::{ActionIndexMapperConfig, Node, NodeData};

/// The internal state for tracking CFR nodes.
///
/// This uses a vector to store all the nodes in the game tree. Each node is
/// identified by its index in this vector. This approach was chosen over a more
/// traditional tree structure with heap allocations and pointers because:
///
/// 1. It avoids complex lifetime issues with rust's borrow checker that arise
///    from nodes referencing their parent/children
/// 2. It provides better memory locality since nodes are stored contiguously
/// 3. It makes serialization/deserialization simpler since we just need to
///    store indices rather than reconstruct pointer relationships
#[derive(Debug)]
pub struct CFRStateInternal {
    /// Vector storing all nodes in the game tree. Nodes reference each other
    /// using their indices into this vector rather than direct pointers.
    pub nodes: Vec<Node>,
    pub starting_game_state: GameState,
    /// The next available index for inserting a new node
    next_node_idx: usize,
    /// Configuration for the action index mapper, derived from the game state.
    /// This ensures consistent action-to-index mapping across all agents and historians.
    pub mapper_config: ActionIndexMapperConfig,
}

/// Counterfactual Regret Minimization (CFR) state tracker.
///
/// This struct manages the game tree used for CFR algorithm calculations. The
/// tree is built lazily as actions are taken in the game. Each node in the tree
/// represents a game state and stores regret values used by the CFR algorithm.
///
/// The state is wrapped in an atomically reference-counted readers-writer lock
/// (Arc<RwLock<>>) to allow sharing between the agent and historian components:
///
/// - The agent needs mutable access to update regret values during simulations
/// - The historian needs read access to traverse the tree and record actions
/// - Both components need to be able to lazily create new nodes
///
/// Rather than using a traditional tree structure with heap allocations and
/// pointers, nodes are stored in a vector and reference each other by index.
/// See `CFRStateInternal` docs for details on this design choice.
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
    inner_state: Arc<RwLock<CFRStateInternal>>,
}

impl CFRState {
    pub fn new(game_state: GameState) -> Self {
        let mapper_config = ActionIndexMapperConfig::from_game_state(&game_state);
        CFRState {
            inner_state: Arc::new(RwLock::new(CFRStateInternal {
                nodes: vec![Node::new_root()],
                starting_game_state: game_state.clone(),
                next_node_idx: 1,
                mapper_config,
            })),
        }
    }

    /// Create a new CFRState with a specific mapper configuration.
    ///
    /// This is useful when you want to use a custom bet range mapping
    /// instead of deriving it from the game state.
    pub fn new_with_mapper_config(
        game_state: GameState,
        mapper_config: ActionIndexMapperConfig,
    ) -> Self {
        CFRState {
            inner_state: Arc::new(RwLock::new(CFRStateInternal {
                nodes: vec![Node::new_root()],
                starting_game_state: game_state.clone(),
                next_node_idx: 1,
                mapper_config,
            })),
        }
    }

    pub fn starting_game_state(&self) -> GameState {
        self.inner_state.read().unwrap().starting_game_state.clone()
    }

    /// Get the action index mapper configuration for this CFR state.
    ///
    /// This configuration defines the bet range for mapping actions to indices.
    pub fn mapper_config(&self) -> ActionIndexMapperConfig {
        self.inner_state.read().unwrap().mapper_config.clone()
    }

    pub fn add(&mut self, parent_idx: usize, child_idx: usize, data: NodeData) -> usize {
        let mut state = self.inner_state.write().unwrap();

        let idx = state.next_node_idx;
        state.next_node_idx += 1;

        let node = Node::new(idx, parent_idx, child_idx, data);
        state.nodes.push(node);

        // The parent node needs to be updated to point to the new child
        state.nodes[parent_idx].set_child(child_idx, idx);

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
        self.inner_state
            .read()
            .unwrap()
            .nodes
            .get(idx)
            .map(|node| node.data.clone())
    }

    /// Access node data without cloning, by calling a closure with a reference.
    ///
    /// This is more efficient than `get_node_data` when you only need to read
    /// the node data (e.g., to access the regret matcher) without owning it.
    ///
    /// # Arguments
    ///
    /// * `idx` - The index of the node to access
    /// * `f` - A closure that receives an `Option<&NodeData>` and returns a value
    ///
    /// # Returns
    ///
    /// The value returned by the closure
    pub fn with_node_data<F, R>(&self, idx: usize, f: F) -> R
    where
        F: FnOnce(Option<&NodeData>) -> R,
    {
        let state = self.inner_state.read().unwrap();
        let node_data = state.nodes.get(idx).map(|node| &node.data);
        f(node_data)
    }

    /// Get the child node index for a given parent node and child index.
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
        self.inner_state
            .read()
            .unwrap()
            .nodes
            .get(parent_idx)?
            .get_child(child_idx)
    }

    /// Get the count for a specific child index on a node.
    ///
    /// # Arguments
    ///
    /// * `node_idx` - The index of the node
    /// * `child_idx` - The child index to get the count for
    ///
    /// # Returns
    ///
    /// * `Some(u32)` - The count if the node exists
    /// * `None` - If the node doesn't exist
    pub fn get_count(&self, node_idx: usize, child_idx: usize) -> Option<u32> {
        self.inner_state
            .read()
            .unwrap()
            .nodes
            .get(node_idx)
            .map(|node| node.get_count(child_idx))
    }

    /// Increment the count for a specific child index on a node.
    ///
    /// # Arguments
    ///
    /// * `node_idx` - The index of the node
    /// * `child_idx` - The child index to increment the count for
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the increment was successful
    /// * `Err(CFRStateError::NodeNotFound)` - If the node doesn't exist
    pub fn increment_count(
        &mut self,
        node_idx: usize,
        child_idx: usize,
    ) -> Result<(), CFRStateError> {
        let mut state = self.inner_state.write().unwrap();
        state
            .nodes
            .get_mut(node_idx)
            .map(|node| {
                node.increment_count(child_idx);
            })
            .ok_or(CFRStateError::NodeNotFound)
    }

    /// Update a node using a closure.
    ///
    /// This method provides mutable access to a node for arbitrary updates.
    ///
    /// # Arguments
    ///
    /// * `node_idx` - The index of the node to update
    /// * `f` - A closure that takes a mutable reference to the node
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the update was successful
    /// * `Err(CFRStateError::NodeNotFound)` - If the node doesn't exist
    pub fn update_node<F>(&mut self, node_idx: usize, f: F) -> Result<(), CFRStateError>
    where
        F: FnOnce(&mut Node),
    {
        let mut state = self.inner_state.write().unwrap();
        state
            .nodes
            .get_mut(node_idx)
            .map(f)
            .ok_or(CFRStateError::NodeNotFound)
    }

    /// Access the internal state of the CFR state structure.
    ///
    /// This method provides access to the internal state for advanced
    /// operations like visualization and debugging.
    ///
    /// # Returns
    ///
    /// A reference to the internal state wrapped in Arc<RwLock<>>
    pub fn internal_state(&self) -> &Arc<RwLock<CFRStateInternal>> {
        &self.inner_state
    }

    /// Ensure a child node exists at the given position, creating or updating as needed.
    ///
    /// This method handles the case where different bet amounts map to the same index
    /// but lead to different outcomes (e.g., one is all-in, one is not).
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
        &mut self,
        parent_idx: usize,
        child_idx: usize,
        expected_data: NodeData,
        allow_mutation: bool,
    ) -> usize {
        let Some(existing_idx) = self.get_child(parent_idx, child_idx) else {
            // No child exists - create new node
            return self.add(parent_idx, child_idx, expected_data);
        };

        // Check if types match using discriminant comparison.
        // Use with_node_data to avoid cloning the NodeData (which includes Box<RegretMatcher>).
        let types_match = self.with_node_data(existing_idx, |node_data| {
            match node_data {
                Some(data) => {
                    std::mem::discriminant(data) == std::mem::discriminant(&expected_data)
                }
                None => true, // Node doesn't exist, consider it a match
            }
        });

        if types_match {
            return existing_idx;
        }

        if allow_mutation {
            // Only clone for debug logging when mutation is needed (rare path)
            tracing::debug!(
                parent_idx,
                child_idx,
                existing_idx,
                ?expected_data,
                "Node type mismatch - updating node type. This occurs when different \
                 bet amounts map to the same index but lead to different outcomes."
            );

            self.update_node(existing_idx, |node| {
                node.data = expected_data.clone();
            })
            .expect("Node should exist since we just retrieved it");
        } else {
            // For panic path, we need the data for the error message
            let data = self.get_node_data(existing_idx);
            panic!(
                "Node type mismatch at parent_idx={}, child_idx={}: \
                 expected {:?}, found {:?}. This can occur when different bet \
                 amounts map to the same index. Set allow_node_mutation=true \
                 to handle this case.",
                parent_idx, child_idx, expected_data, data
            );
        }

        existing_idx
    }
}

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

#[derive(Debug, Clone)]
pub struct TraversalState {
    inner_state: Arc<RwLock<TraversalStateInternal>>,
}

impl PartialEq for TraversalState {
    fn eq(&self, other: &Self) -> bool {
        *self.inner_state.read().unwrap() == *other.inner_state.read().unwrap()
    }
}

impl Eq for TraversalState {}

impl TraversalState {
    pub fn new(node_idx: usize, chosen_child_idx: usize, player_idx: u8) -> Self {
        TraversalState {
            inner_state: Arc::new(RwLock::new(TraversalStateInternal {
                node_idx,
                chosen_child_idx,
                player_idx,
            })),
        }
    }

    pub fn new_root(player_idx: u8) -> Self {
        TraversalState::new(0, 0, player_idx)
    }

    pub fn node_idx(&self) -> usize {
        self.inner_state.read().unwrap().node_idx
    }

    pub fn player_idx(&self) -> u8 {
        self.inner_state.read().unwrap().player_idx
    }

    pub fn chosen_child_idx(&self) -> usize {
        self.inner_state.read().unwrap().chosen_child_idx
    }

    pub fn move_to(&mut self, node_idx: usize, chosen_child_idx: usize) {
        let mut state = self.inner_state.write().unwrap();
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
        let state = self.inner_state.read().unwrap();
        (state.node_idx, state.chosen_child_idx, state.player_idx)
    }

    /// Get node_idx and chosen_child_idx in a single lock acquisition.
    ///
    /// This is more efficient than calling both getters separately.
    ///
    /// Returns (node_idx, chosen_child_idx).
    #[inline]
    pub fn get_position(&self) -> (usize, usize) {
        let state = self.inner_state.read().unwrap();
        (state.node_idx, state.chosen_child_idx)
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::cfr::{NodeData, PlayerData, TraversalState};

    use crate::arena::GameStateBuilder;

    use super::CFRState;

    #[test]
    fn test_add_get_node() {
        let mut state = CFRState::new(
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
    fn test_cloned_traversal_share_loc() {
        let mut traversal = TraversalState::new(0, 0, 0);
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

    /// Verifies get_count returns the actual count value stored for a node.
    #[test]
    fn test_get_count_returns_correct_value() {
        let mut state = CFRState::new(
            GameStateBuilder::new()
                .num_players_with_stack(2, 100.0)
                .blinds(10.0, 5.0)
                .build()
                .unwrap(),
        );

        // Initially count should be 0
        let initial_count = state.get_count(0, 0);
        assert_eq!(initial_count, Some(0));

        // Increment count several times
        state.increment_count(0, 0).unwrap();
        state.increment_count(0, 0).unwrap();
        state.increment_count(0, 0).unwrap();

        // Count should now be 3
        let count = state.get_count(0, 0);
        assert_eq!(count, Some(3));
        assert_ne!(count, Some(0));
        assert_ne!(count, Some(1));

        // Non-existent node should return None
        let none_count = state.get_count(999, 0);
        assert_eq!(none_count, None);
    }

    /// Verifies TraversalState equality - states with same values should be equal.
    #[test]
    fn test_traversal_state_equality() {
        let state1 = TraversalState::new(5, 10, 2);
        let state2 = TraversalState::new(5, 10, 2);

        // Equal states should be equal
        assert_eq!(state1, state2);
    }

    /// Verifies TraversalState inequality - states with different values should not be equal.
    #[test]
    fn test_traversal_state_inequality() {
        let state1 = TraversalState::new(5, 10, 2);
        let state_diff_node = TraversalState::new(6, 10, 2);
        let state_diff_child = TraversalState::new(5, 11, 2);
        let state_diff_player = TraversalState::new(5, 10, 3);

        // Different states should not be equal
        assert_ne!(state1, state_diff_node);
        assert_ne!(state1, state_diff_child);
        assert_ne!(state1, state_diff_player);
    }
}
