use crate::arena::GameState;

#[derive(Debug)]
pub struct RootData {
    pub game_state: GameState,
}

#[derive(Debug)]
pub struct ActionData {
    /// The index of the play that this action is for
    pub idx: usize,
}

#[derive(Debug)]
pub struct PlayerData {
    pub idx: usize,
    pub regrets: little_sorry::RegretMatcher,
}

#[derive(Debug)]
pub struct TerminalData {
    pub utility: Vec<f32>,
}

// The base node type for Poker CFR
#[derive(Debug)]
pub enum NodeData {
    Root(RootData),
    Chance,
    Action(ActionData),
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

    pub fn is_action(&self) -> bool {
        matches!(self, NodeData::Action(_))
    }

    pub fn is_root(&self) -> bool {
        matches!(self, NodeData::Root(_))
    }
}

impl std::fmt::Display for NodeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeData::Root(_) => write!(f, "Root"),
            NodeData::Chance => write!(f, "Chance"),
            NodeData::Action(_) => write!(f, "Action"),
            NodeData::Player(_) => write!(f, "Player"),
            NodeData::Terminal(_) => write!(f, "Terminal"),
        }
    }
}

#[derive(Debug)]
pub struct Node {
    pub idx: usize,
    pub data: NodeData,
    pub parent: Option<usize>,
    pub children: Vec<Option<usize>>,
    pub count: Vec<usize>,
}
