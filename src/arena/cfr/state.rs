use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

use crate::arena::GameState;

use super::{Node, NodeData};

#[derive(Debug)]
pub struct CFRStateInternal {
    pub nodes: Vec<Node>,
    pub starting_game_state: GameState,
    next_node_idx: usize,
}

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

    pub fn add(&mut self, parent_idx: usize, data: NodeData) -> usize {
        let mut state = self.inner_state.borrow_mut();

        let idx = state.next_node_idx;
        state.next_node_idx += 1;

        let node = Node::new(idx, parent_idx, data);
        state.nodes.push(node);

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
    pub chosen_child: usize,
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
    pub fn new(node_idx: usize, chosen_child: usize, player_idx: usize) -> Self {
        TraversalState {
            inner_state: Rc::new(RefCell::new(TraversalStateInternal {
                node_idx,
                chosen_child,
                player_idx,
            })),
        }
    }

    pub fn player_idx(&self) -> usize {
        self.inner_state.borrow().player_idx
    }

    pub fn node_idx(&self) -> usize {
        self.inner_state.borrow().node_idx
    }

    pub fn chosen_child(&self) -> usize {
        self.inner_state.borrow().chosen_child
    }

    pub fn move_to(&mut self, node_idx: usize, chosen_child: usize) {
        let mut state = self.inner_state.borrow_mut();
        state.node_idx = node_idx;
        state.chosen_child = chosen_child;
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
            player_idx: 0,
            regret_matcher: None,
        });

        let player_idx: usize = state.add(0, new_data);

        let node = state.get(player_idx).unwrap();
        match &node.data {
            NodeData::Player(pd) => {
                assert_eq!(pd.player_idx, 0);
            }
            _ => panic!("Expected player data"),
        }
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
        assert_eq!(traversal.chosen_child(), 0);

        assert_eq!(cloned.node_idx(), 0);
        assert_eq!(cloned.player_idx(), 0);
        assert_eq!(cloned.chosen_child(), 0);

        // Simulate traversing the tree
        traversal.move_to(2, 8);

        assert_eq!(traversal.node_idx(), 2);
        assert_eq!(traversal.chosen_child(), 8);
        assert_eq!(traversal.player_idx(), 0);

        // Cloned should have the same values
        assert_eq!(cloned.node_idx(), 2);
        assert_eq!(cloned.chosen_child(), 8);
    }

    #[test]
    fn test_new_dont_share_internal() {
        let mut traversal = TraversalState::new(0, 0, 0);
        let created = TraversalState::new(0, 0, 0);

        assert_eq!(traversal.node_idx(), 0);
        assert_eq!(traversal.player_idx(), 0);
        assert_eq!(traversal.chosen_child(), 0);

        traversal.move_to(43, 3);

        assert_eq!(traversal.node_idx(), 43);
        assert_eq!(traversal.chosen_child(), 3);

        // The newly created one doesn't share the same internal state
        assert_eq!(created.node_idx(), 0);
        assert_eq!(created.chosen_child(), 0);
    }
}
