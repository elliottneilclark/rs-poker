use std::num::{NonZeroU8, NonZeroU32};

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
    /// The count array is used to track the number of times a card
    /// has been dealt.
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

#[derive(Debug, Clone)]
pub struct Node {
    pub idx: u32,
    pub data: NodeData,
    // We store `index + 1` internally to enable NonZero niche optimization.
    // This reduces Option<usize> (16 bytes) to Option<NonZeroU32> (4 bytes).
    parent: Option<NonZeroU32>,
    // Child index is 0-51, so u8 suffices. Same +1 trick for niche optimization.
    parent_child_idx: Option<NonZeroU8>,

    // We use an array of Option<NonZeroU32> to represent the children of the node.
    // The index of the array is the action index or the card index for chance nodes.
    //
    // Using NonZeroU32 enables niche optimization, reducing the array size from
    // 832 bytes (Option<usize>) to 208 bytes (Option<NonZeroU32> is 4 bytes).
    //
    // We store `node_index + 1` internally and subtract 1 when retrieving to
    // handle the case where node index 0 (the root) can be a valid child.
    //
    // This limits the tree to ~4 billion nodes and 52 actions per node.
    children: [Option<NonZeroU32>; 52],
    count: [u32; 52],
}

impl Node {
    pub fn new_root() -> Self {
        Node {
            idx: 0,
            data: NodeData::Root,
            // Root is its own parent; store 0 + 1 = 1 for niche optimization
            parent: NonZeroU32::new(1),
            parent_child_idx: None,
            children: [None; 52],
            count: [0; 52],
        }
    }

    /// Create a new node with the provided index, parent index, and data.
    ///
    /// # Arguments
    ///
    /// * `idx` - The index of the node
    /// * `parent` - The index of the parent node
    /// * `data` - The data for the node
    ///
    /// # Returns
    ///
    /// A new node with the provided index, parent index, and data.
    ///
    /// # Example
    ///
    /// ```
    /// use rs_poker::arena::cfr::{Node, NodeData};
    ///
    /// let idx = 1;
    /// let parent = 0;
    /// let parent_child_idx = 0;
    /// let data = NodeData::Chance;
    /// let node = Node::new(idx, parent, parent_child_idx, data);
    /// ```
    pub fn new(idx: usize, parent: usize, parent_child_idx: usize, data: NodeData) -> Self {
        Node {
            idx: idx as u32,
            data,
            // Store parent + 1 for niche optimization
            parent: NonZeroU32::new((parent + 1) as u32),
            // Store parent_child_idx + 1 for niche optimization
            parent_child_idx: NonZeroU8::new((parent_child_idx + 1) as u8),
            children: [None; 52],
            count: [0; 52],
        }
    }

    // Set child node at the provided index.
    // Stores child + 1 internally to enable NonZeroU32 niche optimization.
    pub fn set_child(&mut self, idx: usize, child: usize) {
        assert!(self.children[idx].is_none());
        // Store child + 1 to handle index 0 (NonZeroU32 cannot be 0)
        self.children[idx] = NonZeroU32::new((child + 1) as u32);
    }

    // Get the child node at the provided index.
    // Subtracts 1 from stored value to recover original index.
    pub fn get_child(&self, idx: usize) -> Option<usize> {
        self.children[idx].map(|v| (v.get() - 1) as usize)
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

    // Increment the count for the provided index.
    // Uses saturating addition to prevent overflow panics.
    pub fn increment_count(&mut self, idx: usize) {
        assert!(idx == 0 || !self.data.is_terminal());
        self.count[idx] = self.count[idx].saturating_add(1);
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
        self.children
            .iter()
            .enumerate()
            .filter_map(|(idx, &child)| child.map(|c| (idx, (c.get() - 1) as usize)))
    }

    /// Get the count for a specific child index
    ///
    /// # Arguments
    ///
    /// * `idx` - The index of the child
    ///
    /// # Returns
    ///
    /// The count for the specified child
    pub fn get_count(&self, idx: usize) -> u32 {
        self.count[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terminal_data_default() {
        let terminal_data = TerminalData::default();
        assert_eq!(terminal_data.total_utility, 0.0);
    }

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
        assert!(matches!(node.data, NodeData::Root));
    }

    #[test]
    fn test_node_new() {
        let node = Node::new(1, 0, 0, NodeData::Chance);
        assert_eq!(node.idx, 1);
        assert_eq!(node.get_parent(), Some(0));
        assert!(matches!(node.data, NodeData::Chance));
    }

    #[test]
    fn test_node_set_get_child() {
        let mut node = Node::new(1, 0, 0, NodeData::Chance);
        node.set_child(0, 2);
        assert_eq!(node.get_child(0), Some(2));
    }

    #[test]
    fn test_node_increment_count() {
        let mut node = Node::new(1, 0, 0, NodeData::Chance);
        node.increment_count(0);
        assert_eq!(node.count[0], 1);
    }

    /// Verifies is_chance returns false for Root, Player, and Terminal nodes.
    #[test]
    fn test_node_data_is_chance_false_for_others() {
        assert!(!NodeData::Root.is_chance());
        assert!(
            !NodeData::Player(PlayerData {
                regret_matcher: None,
                player_idx: 0
            })
            .is_chance()
        );
        assert!(!NodeData::Terminal(TerminalData::default()).is_chance());
    }

    /// Verifies is_player returns false for Root, Chance, and Terminal nodes.
    #[test]
    fn test_node_data_is_player_false_for_others() {
        assert!(!NodeData::Root.is_player());
        assert!(!NodeData::Chance.is_player());
        assert!(!NodeData::Terminal(TerminalData::default()).is_player());
    }

    /// Verifies is_root returns false for Chance, Player, and Terminal nodes.
    #[test]
    fn test_node_data_is_root_false_for_others() {
        assert!(!NodeData::Chance.is_root());
        assert!(
            !NodeData::Player(PlayerData {
                regret_matcher: None,
                player_idx: 0
            })
            .is_root()
        );
        assert!(!NodeData::Terminal(TerminalData::default()).is_root());
    }

    /// Verifies Display implementation outputs meaningful strings for each node type.
    #[test]
    fn test_node_data_display() {
        let root_str = format!("{}", NodeData::Root);
        assert!(!root_str.is_empty(), "Root display should not be empty");
        assert!(root_str.contains("Root"));

        let chance_str = format!("{}", NodeData::Chance);
        assert!(!chance_str.is_empty(), "Chance display should not be empty");
        assert!(chance_str.contains("Chance"));

        let player_str = format!(
            "{}",
            NodeData::Player(PlayerData {
                regret_matcher: None,
                player_idx: 0
            })
        );
        assert!(!player_str.is_empty(), "Player display should not be empty");
        assert!(player_str.contains("Player"));

        let terminal_str = format!("{}", NodeData::Terminal(TerminalData::default()));
        assert!(
            !terminal_str.is_empty(),
            "Terminal display should not be empty"
        );
        assert!(terminal_str.contains("Terminal"));
    }

    /// Verify that memory optimizations keep Node size reasonable.
    /// This test documents the expected size and will fail if accidental
    /// changes increase the struct size significantly.
    #[test]
    fn test_node_size_optimization() {
        use std::mem::size_of;

        // Node should be <= 464 bytes after optimizations:
        // - idx: u32 = 4 bytes
        // - data: NodeData ~= 16 bytes (PlayerData with Option<Box<_>> and u8)
        // - parent: Option<NonZeroU32> = 4 bytes
        // - parent_child_idx: Option<NonZeroU8> = 1 byte
        // - padding for alignment ~= 7 bytes
        // - children: [Option<NonZeroU32>; 52] = 208 bytes
        // - count: [u32; 52] = 208 bytes
        // Total: ~456 bytes
        let node_size = size_of::<Node>();
        assert!(
            node_size <= 464,
            "Node size {} bytes exceeds expected max of 464 bytes. \
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
