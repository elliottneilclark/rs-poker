use uuid::Uuid;

use crate::arena::{game_state::Round, historian::HistorianError, GameState};

use super::{
    node::{ActionData, Node, NodeData, PlayerData, RootData, TerminalData},
    EXPERTS, PREFLOP_EXPERTS,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CFRSavePoint {
    current_node: Option<usize>,
    previous_node: Option<usize>,
    child_idx: Option<usize>,
}

pub struct PlayerCFRState {
    pub id: Uuid,
    current_node: Option<usize>,
    previous_node: Option<usize>,
    child_idx: Option<usize>,
    arena: Vec<Node>,
}

// These are the possible node types that we
// encounter on the way to the next node.
#[derive(Debug, PartialEq, Eq)]
pub enum EnsureNodeType {
    // Ensure that it's a player node for this index
    Player(usize),
    // Endure that it's an action node for this index
    Action(usize),
    Chance,
    Terminal,
}

impl PlayerCFRState {
    pub fn new(game_state: GameState) -> PlayerCFRState {
        let arena: Vec<Node> = vec![Node {
            idx: 0,
            data: NodeData::Root(RootData {
                game_state: game_state.clone(),
            }),
            parent: None,
            children: vec![None],

            // At some point we probably want to initialize this
            // and inline it
            count: vec![0],
        }];

        let id = uuid::Uuid::now_v7();

        PlayerCFRState {
            id,
            // We don't know what the next node will be yet
            current_node: None,
            // Root node is the previous node
            // And by convention, there's only one child at index 0
            previous_node: Some(0),
            child_idx: Some(0),
            arena,
        }
    }

    pub fn get_current_node(&self) -> Option<&Node> {
        self.current_node.map(|idx| &self.arena[idx])
    }
    pub fn get_mut_current_node(&mut self) -> Option<&mut Node> {
        self.current_node.map(|idx| &mut self.arena[idx])
    }

    pub fn get(&self, idx: usize) -> Option<&Node> {
        self.arena.get(idx)
    }

    pub fn get_mut(&mut self, idx: usize) -> Option<&mut Node> {
        self.arena.get_mut(idx)
    }

    pub fn set_next_node(&mut self, child_idx: usize) -> Result<(), HistorianError> {
        if let Some(current_idx) = self.current_node {
            // Ensure that there are children to pull from
            // And counters to keep track of it.
            if self.arena[current_idx].children.len() <= child_idx {
                self.arena[current_idx].children.resize(child_idx + 1, None);
                self.arena[current_idx].count.resize(child_idx + 1, 0)
            }

            self.current_node = self.arena[current_idx].children[child_idx];
            self.previous_node = Some(current_idx);
            self.child_idx = Some(child_idx);

            // Increment the count of times we
            // have visited the child via this path
            self.arena[current_idx].count[child_idx] += 1;
            Ok(())
        } else {
            Err(HistorianError::CFRNodeNotFound)
        }
    }

    pub fn num_children(&self, data: &NodeData) -> usize {
        match data {
            NodeData::Root(_) => 1,
            NodeData::Player(_) => 6,
            NodeData::Action(_) => 8,
            NodeData::Chance => 52,
            NodeData::Terminal(_) => 0,
        }
    }

    pub fn add_current_node(&mut self, data: NodeData) -> usize {
        let idx = self.arena.len();

        let num_children = self.num_children(&data);

        let node = Node {
            idx,
            data,
            parent: self.previous_node,
            children: vec![None; num_children],
            count: vec![0; num_children],
        };

        // Add the node to the arena
        self.arena.push(node);
        // The previous node's child at the child index is now the current node
        let previous_node = &mut self.arena[self.previous_node.unwrap()];

        // This the child index that the new node will be at
        let path_idx = self.child_idx.unwrap();
        if previous_node.children.len() <= path_idx {
            previous_node.children.resize(path_idx + 1, None);
            previous_node.count.resize(path_idx + 1, 0)
        }
        previous_node.children[self.child_idx.unwrap()] = Some(idx);
        // And the previously empty current node is now the new node
        self.current_node = Some(idx);

        idx
    }

    pub fn ensure_current_node(
        &mut self,
        node_type: EnsureNodeType,
        round: Round,
    ) -> Result<(), HistorianError> {
        if let Some(current_node) = self.get_current_node() {
            // debug assert that the current node's data matches the ensure node type
            match node_type {
                EnsureNodeType::Player(idx) => {
                    if let NodeData::Player(player_data) = &current_node.data {
                        if player_data.idx == idx {
                            Ok(())
                        } else {
                            Err(HistorianError::CFRUnexpectedNode(
                                "Expected Player idx does not match".to_string(),
                            ))
                        }
                    } else {
                        Err(HistorianError::CFRUnexpectedNode(format!(
                            "Expected Player found #{current_node:?}"
                        )))
                    }
                }
                EnsureNodeType::Action(_idx) => {
                    if let NodeData::Action(action_data) = &current_node.data {
                        if action_data.idx == _idx {
                            Ok(())
                        } else {
                            Err(HistorianError::CFRUnexpectedNode(
                                "Expected Action idx does not match".to_string(),
                            ))
                        }
                    } else {
                        Err(HistorianError::CFRUnexpectedNode(format!(
                            "Expected Action found #{current_node:?}"
                        )))
                    }
                }
                EnsureNodeType::Chance => {
                    if current_node.data.is_chance() {
                        Ok(())
                    } else {
                        Err(HistorianError::CFRUnexpectedNode(format!(
                            "Expected Chance found #{current_node:?}"
                        )))
                    }
                }
                EnsureNodeType::Terminal => {
                    if current_node.data.is_terminal() {
                        Ok(())
                    } else {
                        Err(HistorianError::CFRUnexpectedNode(format!(
                            "Expected Terminal found #{current_node:?}"
                        )))
                    }
                }
            }
        } else {
            // Based upon the node type create the default node data
            let data = match node_type {
                EnsureNodeType::Player(idx) => NodeData::Player(PlayerData {
                    idx,
                    regrets: self.build_regret_matcher(round),
                }),
                EnsureNodeType::Action(idx) => NodeData::Action(ActionData { idx }),
                EnsureNodeType::Chance => NodeData::Chance,
                EnsureNodeType::Terminal => NodeData::Terminal(TerminalData { utility: vec![] }),
            };
            // Then add that to self.state as the current node
            self.add_current_node(data);
            Ok(())
        }
    }

    pub fn build_regret_matcher(&self, round: Round) -> little_sorry::RegretMatcher {
        let num_experts = self.num_experts(round);
        little_sorry::RegretMatcher::new(num_experts).unwrap()
    }

    pub fn num_experts(&self, round: Round) -> usize {
        match round {
            Round::Preflop => PREFLOP_EXPERTS.len(),
            _ => EXPERTS.len(),
        }
    }

    pub fn reset(&mut self) {
        self.current_node = self.arena[0].children[0];
        self.previous_node = Some(0);
        self.child_idx = Some(0);
    }

    pub fn save_point(&self) -> CFRSavePoint {
        CFRSavePoint {
            current_node: self.current_node,
            previous_node: self.previous_node,
            child_idx: self.child_idx,
        }
    }

    pub fn restore_save_point(&mut self, save_state: CFRSavePoint) {
        self.current_node = save_state.current_node;
        self.previous_node = save_state.previous_node;
        self.child_idx = save_state.child_idx;
    }
}
