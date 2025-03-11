#[derive(Debug, Clone, Default)]
pub struct PlayerData {
    pub regret_matcher: Option<Box<little_sorry::RegretMatcher>>,
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
    pub idx: usize,
    pub data: NodeData,
    pub parent: Option<usize>,
    pub parent_child_idx: Option<usize>,

    // We use an array of Option<usize> to represent the children of the node.
    // The index of the array is the action index or the card index for chance nodes.
    //
    // This limits the number of possible agent actions to 52, but in return we
    // get contiguous memory for no pointer chasing.
    children: [Option<usize>; 52],
    count: [u32; 52],
}

impl Node {
    pub fn new_root() -> Self {
        Node {
            idx: 0,
            data: NodeData::Root,
            parent: Some(0),
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
            idx,
            data,
            parent: Some(parent),
            parent_child_idx: Some(parent_child_idx),
            children: [None; 52],
            count: [0; 52],
        }
    }

    // Set child node at the provided index
    pub fn set_child(&mut self, idx: usize, child: usize) {
        assert_eq!(self.children[idx], None);
        self.children[idx] = Some(child);
    }

    // Get the child node at the provided index
    pub fn get_child(&self, idx: usize) -> Option<usize> {
        self.children[idx]
    }

    // Increment the count for the provided index
    pub fn increment_count(&mut self, idx: usize) {
        if idx == 0 || !self.data.is_terminal() {
            self.count[idx] += 1;
        } else {
            // Log this situation but don't panic - useful for debugging
            println!(
                "Warning: Attempted to increment count for terminal node at index {}",
                idx
            );
        }
    }

    // Check if this node is terminal
    pub fn is_terminal(&self) -> bool {
        self.data.is_terminal()
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
            .filter_map(|(idx, &child)| child.map(|c| (idx, c)))
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
        let node_data = NodeData::Player(PlayerData::default());
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
        // Root is it's own parent
        assert!(node.parent.is_some());
        assert_eq!(node.parent, Some(0));
        assert!(matches!(node.data, NodeData::Root));
    }

    #[test]
    fn test_node_new() {
        let node = Node::new(1, 0, 0, NodeData::Chance);
        assert_eq!(node.idx, 1);
        assert_eq!(node.parent, Some(0));
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
}
