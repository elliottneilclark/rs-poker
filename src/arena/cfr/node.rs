use std::num::{NonZeroU8, NonZeroU32};
use std::sync::atomic::{AtomicU32, Ordering};

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

#[derive(Debug, Clone)]
pub struct PlayerData {
    pub regret_matcher: Option<Box<little_sorry::DcfrPlusRegretMatcher>>,
    pub player_idx: u8,
}

#[derive(Debug, Clone)]
pub struct TerminalData {
    pub total_utility: f32,
}

impl TerminalData {
    pub fn new(total_utility: f32) -> Self {
        TerminalData { total_utility }
    }
}

impl Default for TerminalData {
    fn default() -> Self {
        TerminalData::new(0.0)
    }
}

// The base node type for Poker CFR
#[derive(Debug, Clone)]
pub enum NodeData {
    /// The root node.
    ///
    /// This node is always the first node in the tree, we don't
    /// use the GameStart action to create the node. By egarly
    /// creating the root node we can simplify the traversal.
    /// All that's required is to ignore GameStart, ForcedBet, and
    /// PlayerSit actions as they are all assumed in the root node.
    ///
    /// For all traversals we start at the root node and then follow the
    /// 0th child node for the first real action that follows from
    /// the starting game state. That could be a chance card if the player
    /// is going to get dealt starting hands, or it could be the first
    /// player action if the gamestate starts with hands already dealt.
    Root,

    /// A chance node.
    ///
    /// This node represents the dealing of a single card.
    /// Each child index in the children array represents a card.
    Chance,
    Player(PlayerData),
    Terminal(TerminalData),
}

impl NodeData {
    pub fn is_terminal(&self) -> bool {
        matches!(self, NodeData::Terminal(_))
    }

    pub fn is_chance(&self) -> bool {
        matches!(self, NodeData::Chance)
    }

    pub fn is_player(&self) -> bool {
        matches!(self, NodeData::Player(_))
    }

    pub fn is_root(&self) -> bool {
        matches!(self, NodeData::Root)
    }
}

impl std::fmt::Display for NodeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeData::Root => write!(f, "Root"),
            NodeData::Chance => write!(f, "Chance"),
            NodeData::Player(_) => write!(f, "Player"),
            NodeData::Terminal(_) => write!(f, "Terminal"),
        }
    }
}

pub struct Node {
    pub idx: u32,
    /// Per-node lock for the node's data. Readers can access node data without
    /// holding any global lock; writers only lock this specific node.
    data: RwLock<NodeData>,
    // We store `index + 1` internally to enable NonZero niche optimization.
    // This reduces Option<usize> (16 bytes) to Option<NonZeroU32> (4 bytes).
    parent: Option<NonZeroU32>,
    // Child index is 0-51, so u8 suffices. Same +1 trick for niche optimization.
    parent_child_idx: Option<NonZeroU8>,

    // Atomic child pointers: 0 = no child, n > 0 = child at node index n-1.
    // Same +1 encoding as the previous NonZeroU32 approach, but using AtomicU32
    // for lock-free concurrent reads and compare-exchange writes.
    // This limits the tree to ~4 billion nodes and 52 actions per node.
    children: [AtomicU32; 52],
}

// AtomicU32 and RwLock don't implement Clone, so implement manually.
impl Clone for Node {
    fn clone(&self) -> Self {
        let mut children = [const { AtomicU32::new(0) }; 52];
        for (i, child) in children.iter_mut().enumerate() {
            *child = AtomicU32::new(self.children[i].load(Ordering::Relaxed));
        }
        Node {
            idx: self.idx,
            data: RwLock::new(self.data.read().clone()),
            parent: self.parent,
            parent_child_idx: self.parent_child_idx,
            children,
        }
    }
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("idx", &self.idx)
            .field("data", &*self.data.read())
            .field("parent", &self.parent)
            .field("parent_child_idx", &self.parent_child_idx)
            .finish_non_exhaustive()
    }
}

impl Node {
    pub fn new_root() -> Self {
        Node {
            idx: 0,
            data: RwLock::new(NodeData::Root),
            // Root is its own parent; store 0 + 1 = 1 for niche optimization
            parent: NonZeroU32::new(1),
            parent_child_idx: None,
            children: [const { AtomicU32::new(0) }; 52],
        }
    }

    /// Create a new node with the provided parent index and data.
    ///
    /// The `idx` field is set to 0 as a placeholder. When the node is pushed
    /// into a `NodeArena`, the arena sets `idx` to the actual allocated index.
    ///
    /// # Arguments
    ///
    /// * `parent` - The index of the parent node
    /// * `parent_child_idx` - The index within the parent's children array
    /// * `data` - The data for the node
    ///
    /// # Example
    ///
    /// ```
    /// use rs_poker::arena::cfr::{Node, NodeData};
    ///
    /// let parent = 0;
    /// let parent_child_idx = 0;
    /// let data = NodeData::Chance;
    /// let node = Node::new(parent, parent_child_idx, data);
    /// ```
    pub fn new(parent: usize, parent_child_idx: usize, data: NodeData) -> Self {
        Node {
            idx: 0,
            data: RwLock::new(data),
            // Store parent + 1 for niche optimization
            parent: NonZeroU32::new((parent + 1) as u32),
            // Store parent_child_idx + 1 for niche optimization
            parent_child_idx: NonZeroU8::new((parent_child_idx + 1) as u8),
            children: [const { AtomicU32::new(0) }; 52],
        }
    }

    /// Acquire a read lock on this node's data.
    pub fn read_data(&self) -> RwLockReadGuard<'_, NodeData> {
        self.data.read()
    }

    /// Acquire a write lock on this node's data.
    pub fn write_data(&self) -> RwLockWriteGuard<'_, NodeData> {
        self.data.write()
    }

    /// Set child node at the provided index.
    /// Uses atomic store; safe to call through a shared reference.
    /// Panics if a child is already set at this index.
    pub fn set_child(&self, idx: usize, child: usize) {
        let prev = self.children[idx].swap((child + 1) as u32, Ordering::Release);
        assert_eq!(prev, 0, "Child already set at index {idx}");
    }

    /// Try to set child node at the provided index using compare-exchange.
    /// Returns Ok(()) if the child was set, or Err(existing_child) if another
    /// thread already set a child at this index.
    pub fn try_set_child(&self, idx: usize, child: usize) -> Result<(), usize> {
        match self.children[idx].compare_exchange(
            0,
            (child + 1) as u32,
            Ordering::Release,
            Ordering::Acquire,
        ) {
            Ok(_) => Ok(()),
            Err(existing) => Err((existing - 1) as usize),
        }
    }

    /// Get the child node at the provided index.
    /// Lock-free atomic load.
    pub fn get_child(&self, idx: usize) -> Option<usize> {
        let val = self.children[idx].load(Ordering::Acquire);
        if val == 0 {
            None
        } else {
            Some((val - 1) as usize)
        }
    }

    /// Get the parent node index.
    /// Returns None only if this is somehow an orphan node (shouldn't happen in practice).
    pub fn get_parent(&self) -> Option<usize> {
        self.parent.map(|v| (v.get() - 1) as usize)
    }

    /// Get the index of this node in its parent's children array.
    pub fn get_parent_child_idx(&self) -> Option<usize> {
        self.parent_child_idx.map(|v| (v.get() - 1) as usize)
    }

    /// Get an iterator over all the node's children with their indices
    ///
    /// This is useful for traversing the tree for visualization or debugging.
    ///
    /// # Returns
    ///
    /// An iterator over tuples of (child_idx, child_node_idx) where:
    /// - child_idx is the index in the children array
    /// - child_node_idx is the index of the child node in the nodes vector
    pub fn iter_children(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.children.iter().enumerate().filter_map(|(idx, child)| {
            let val = child.load(Ordering::Relaxed);
            if val == 0 {
                None
            } else {
                Some((idx, (val - 1) as usize))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terminal_data_new() {
        let terminal_data = TerminalData::new(10.0);
        assert_eq!(terminal_data.total_utility, 10.0);
    }

    #[test]
    fn test_node_data_is_terminal() {
        let node_data = NodeData::Terminal(TerminalData::new(10.0));
        assert!(node_data.is_terminal());
    }

    #[test]
    fn test_node_data_is_chance() {
        let node_data = NodeData::Chance;
        assert!(node_data.is_chance());
    }

    #[test]
    fn test_node_data_is_player() {
        let node_data = NodeData::Player(PlayerData {
            regret_matcher: None,
            player_idx: 0,
        });
        assert!(node_data.is_player());
    }

    #[test]
    fn test_node_data_is_root() {
        let node_data = NodeData::Root;
        assert!(node_data.is_root());
    }

    #[test]
    fn test_node_new_root() {
        let node = Node::new_root();
        assert_eq!(node.idx, 0);
        // Root is its own parent
        assert!(node.get_parent().is_some());
        assert_eq!(node.get_parent(), Some(0));
        assert!(matches!(*node.read_data(), NodeData::Root));
    }

    #[test]
    fn test_node_new() {
        let node = Node::new(0, 0, NodeData::Chance);
        assert_eq!(node.idx, 0); // idx is placeholder, set by arena
        assert_eq!(node.get_parent(), Some(0));
        assert!(matches!(*node.read_data(), NodeData::Chance));
    }

    #[test]
    fn test_node_set_get_child() {
        let node = Node::new(0, 0, NodeData::Chance);
        node.set_child(0, 2);
        assert_eq!(node.get_child(0), Some(2));
    }

    #[test]
    fn test_node_try_set_child_success() {
        let node = Node::new(0, 0, NodeData::Chance);
        assert!(node.try_set_child(0, 2).is_ok());
        assert_eq!(node.get_child(0), Some(2));
    }

    #[test]
    fn test_node_try_set_child_already_set() {
        let node = Node::new(0, 0, NodeData::Chance);
        node.set_child(0, 2);
        assert_eq!(node.try_set_child(0, 5), Err(2));
        // Original child should be unchanged
        assert_eq!(node.get_child(0), Some(2));
    }

    #[test]
    fn test_node_get_parent_and_child_idx() {
        let node = Node::new(5, 10, NodeData::Chance);
        assert_eq!(node.get_parent(), Some(5));
        assert_eq!(node.get_parent_child_idx(), Some(10));
    }

    #[test]
    fn test_node_iter_children() {
        let node = Node::new(0, 0, NodeData::Chance);
        node.set_child(3, 10);
        node.set_child(7, 20);
        node.set_child(51, 30);

        let children: Vec<(usize, usize)> = node.iter_children().collect();
        assert_eq!(children, vec![(3, 10), (7, 20), (51, 30)]);
    }

    #[test]
    fn test_node_write_data() {
        let node = Node::new(0, 0, NodeData::Chance);
        assert!(node.read_data().is_chance());

        {
            let mut guard = node.write_data();
            *guard = NodeData::Terminal(TerminalData::new(5.0));
        }

        assert!(node.read_data().is_terminal());
    }

    #[test]
    fn test_node_clone() {
        let node = Node::new(3, 7, NodeData::Chance);
        node.set_child(0, 42);

        let cloned = node.clone();
        assert_eq!(cloned.idx, node.idx);
        assert_eq!(cloned.get_parent(), node.get_parent());
        assert_eq!(cloned.get_parent_child_idx(), node.get_parent_child_idx());
        assert_eq!(cloned.get_child(0), Some(42));
        assert!(cloned.read_data().is_chance());
    }

    #[test]
    #[should_panic(expected = "Child already set")]
    fn test_set_child_panics_on_double_set() {
        let node = Node::new(0, 0, NodeData::Chance);
        node.set_child(0, 1);
        node.set_child(0, 2); // Should panic
    }

    /// Verify that memory optimizations keep Node size reasonable.
    /// This test documents the expected size and will fail if accidental
    /// changes increase the struct size significantly.
    #[test]
    fn test_node_size_optimization() {
        use std::mem::size_of;

        // Node should be <= 264 bytes after optimizations:
        // - idx: u32 = 4 bytes
        // - data: RwLock<NodeData> ~= 24 bytes (RwLock overhead + NodeData)
        // - parent: Option<NonZeroU32> = 4 bytes
        // - parent_child_idx: Option<NonZeroU8> = 1 byte
        // - padding for alignment ~= 7 bytes
        // - children: [AtomicU32; 52] = 208 bytes
        // Total: ~248 bytes
        let node_size = size_of::<Node>();
        assert!(
            node_size <= 264,
            "Node size {} bytes exceeds expected max of 264 bytes. \
             Check for unintentional size increases.",
            node_size
        );

        // Verify Option<NonZeroU32> is 4 bytes (niche optimization)
        assert_eq!(
            size_of::<Option<NonZeroU32>>(),
            4,
            "Option<NonZeroU32> should be 4 bytes due to niche optimization"
        );

        // Verify Option<NonZeroU8> is 1 byte (niche optimization)
        assert_eq!(
            size_of::<Option<NonZeroU8>>(),
            1,
            "Option<NonZeroU8> should be 1 byte due to niche optimization"
        );
    }
}
