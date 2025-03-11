use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

use crate::arena::GameState;

use super::{Node, NodeData};

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
}

/// Counterfactual Regret Minimization (CFR) state tracker.
///
/// This struct manages the game tree used for CFR algorithm calculations. The
/// tree is built lazily as actions are taken in the game. Each node in the tree
/// represents a game state and stores regret values used by the CFR algorithm.
///
/// The state is wrapped in a reference-counted cell (Rc<RefCell<>>) to allow
/// sharing between the agent and historian components:
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
/// use rs_poker::arena::GameState;
/// use rs_poker::arena::cfr::CFRState;
///
/// let game_state = GameState::new_starting(vec![100.0; 2], 10.0, 5.0, 0.0, 0);
/// let cfr_state = CFRState::new(game_state);
/// ```
#[derive(Debug, Clone)]
pub struct CFRState {
    inner_state: Rc<RefCell<CFRStateInternal>>,
}

impl CFRState {
    pub fn new(game_state: GameState) -> Self {
        CFRState {
            inner_state: Rc::new(RefCell::new(CFRStateInternal {
                nodes: vec![Node::new_root()],
                starting_game_state: game_state.clone(),
                next_node_idx: 1,
            })),
        }
    }

    pub fn starting_game_state(&self) -> GameState {
        self.inner_state.borrow().starting_game_state.clone()
    }

    pub fn add(&mut self, parent_idx: usize, child_idx: usize, data: NodeData) -> usize {
        let mut state = self.inner_state.borrow_mut();

        let idx = state.next_node_idx;
        state.next_node_idx += 1;

        let node = Node::new(idx, parent_idx, child_idx, data);
        state.nodes.push(node);

        // The parent node needs to be updated to point to the new child
        state.nodes[parent_idx].set_child(child_idx, idx);

        idx
    }

    pub fn get(&self, idx: usize) -> Option<Ref<Node>> {
        let inner_ref = self.inner_state.borrow();

        Ref::filter_map(inner_ref, |state| state.nodes.get(idx)).ok()
    }

    pub fn get_mut(&mut self, idx: usize) -> Option<RefMut<Node>> {
        let inner_ref = self.inner_state.borrow_mut();

        RefMut::filter_map(inner_ref, |state| state.nodes.get_mut(idx)).ok()
    }

    /// Replace the data of an existing node without changing its structure
    ///
    /// # Arguments
    ///
    /// * `idx` - The index of the node to replace the data for
    /// * `data` - The new data to set for the node
    ///
    /// # Returns
    ///
    /// `true` if the node was found and updated, `false` otherwise
    pub fn replace_node_data(&mut self, idx: usize, data: NodeData) -> bool {
        let mut inner_ref = self.inner_state.borrow_mut();

        if let Some(node) = inner_ref.nodes.get_mut(idx) {
            // Replace just the data part, keeping all child links intact
            node.data = data;
            true
        } else {
            false
        }
    }

    /// Access the internal state of the CFR state structure.
    ///
    /// This method provides access to the internal state for advanced
    /// operations like visualization and debugging.
    ///
    /// # Returns
    ///
    /// A reference to the internal state wrapped in Rc<RefCell<>>
    pub fn internal_state(&self) -> &Rc<RefCell<CFRStateInternal>> {
        &self.inner_state
    }
}

#[derive(Debug)]
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
    pub player_idx: usize,
}

#[derive(Debug, Clone)]
pub struct TraversalState {
    inner_state: Rc<RefCell<TraversalStateInternal>>,
}

impl TraversalState {
    pub fn new(node_idx: usize, chosen_child_idx: usize, player_idx: usize) -> Self {
        TraversalState {
            inner_state: Rc::new(RefCell::new(TraversalStateInternal {
                node_idx,
                chosen_child_idx,
                player_idx,
            })),
        }
    }

    pub fn new_root(player_idx: usize) -> Self {
        TraversalState::new(0, 0, player_idx)
    }

    pub fn node_idx(&self) -> usize {
        self.inner_state.borrow().node_idx
    }

    pub fn player_idx(&self) -> usize {
        self.inner_state.borrow().player_idx
    }

    pub fn chosen_child_idx(&self) -> usize {
        self.inner_state.borrow().chosen_child_idx
    }

    pub fn move_to(&mut self, node_idx: usize, chosen_child_idx: usize) {
        self.inner_state.borrow_mut().node_idx = node_idx;
        self.inner_state.borrow_mut().chosen_child_idx = chosen_child_idx;
    }
}

#[cfg(test)]
mod tests {
    use crate::arena::cfr::{NodeData, PlayerData, TraversalState};

    use crate::arena::GameState;

    use super::CFRState;

    #[test]
    fn test_add_get_node() {
        let mut state = CFRState::new(GameState::new_starting(vec![100.0; 3], 10.0, 5.0, 0.0, 0));
        let new_data = NodeData::Player(PlayerData {
            regret_matcher: None,
        });

        let player_idx: usize = state.add(0, 0, new_data);

        let node = state.get(player_idx).unwrap();
        match &node.data {
            NodeData::Player(pd) => assert!(pd.regret_matcher.is_none()),
            _ => panic!("Expected player data"),
        }

        // assert that the parent and child idx are correct
        assert_eq!(node.parent, Some(0));
        assert_eq!(node.parent_child_idx, Some(0));

        let parent = state.get(0).unwrap();

        // assert that the parent node has the correct child idx
        assert_eq!(parent.get_child(0), Some(player_idx));
    }

    #[test]
    fn test_node_get_not_exist() {
        let state = CFRState::new(GameState::new_starting(vec![100.0; 3], 10.0, 5.0, 0.0, 0));
        // root node is always at index 0
        let root = state.get(0);
        assert!(root.is_some());

        let node = state.get(100);
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
}
