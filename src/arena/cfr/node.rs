#[derive(Debug, Clone)]
pub struct PlayerData {
    pub player_idx: usize,
    pub regret_matcher: Option<Box<little_sorry::RegretMatcher>>,
}

#[derive(Debug, Clone)]
pub struct TerminalData {
    pub utility: f32,
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

    // We use an array of Option<usize> to represent the children of the node.
    // The index of the array is the action index or the card index for chance nodes.
    //
    // This limits the number of possible agent actions to 52, but in return we
    // get contiguous memory for no pointer chasing.
    children: [Option<usize>; 52],
    count: [usize; 52],
}

impl Node {
    pub fn new_root() -> Self {
        Self::new(0, 0, NodeData::Root)
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
    /// let data = NodeData::Chance;
    /// let node = Node::new(idx, parent, data);
    /// ```
    pub fn new(idx: usize, parent: usize, data: NodeData) -> Self {
        Node {
            idx,
            data,
            parent: Some(parent),
            children: [None; 52],
            count: [0; 52],
        }
    }

    // Set child node at the provided index
    pub fn set_child(&mut self, idx: usize, child: usize) {
        self.children[idx] = Some(child);
    }

    // Get the child node at the provided index
    pub fn get_child(&self, idx: usize) -> Option<usize> {
        self.children[idx]
    }

    // Increment the count for the provided index
    pub fn increment_count(&mut self, idx: usize) {
        self.count[idx] += 1;
    }
}
