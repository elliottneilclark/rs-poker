use std::{cell::RefCell, rc::Rc};

use ndarray::aview1;
use rand::{thread_rng, Rng};
use uuid::Uuid;

use crate::arena::{
    action::AgentAction, cfr::ArenaCFRHistorian, game_state::Round, Agent, GameState, Historian,
    HoldemSimulationBuilder,
};

use super::{
    node::{NodeData, PlayerData},
    EnsureNodeType, PlayerCFRState, EXPERTS, LOWER_MULT, MAX_RAISE_EXPERTS,
    MAX_RAISE_PREFLOP_EXPERTS, PREFLOP_EXPERTS, UPPER_MULT,
};

pub struct ArenaCFRAgent {
    pub id: Uuid,
    pub cfr_states: Vec<Rc<RefCell<PlayerCFRState>>>,
    pub player_idx: usize,
    forced_action: Option<AgentAction>,
}

impl ArenaCFRAgent {
    pub fn new(cfr_states: Vec<Rc<RefCell<PlayerCFRState>>>, player_idx: usize) -> Self {
        let id = Uuid::now_v7();

        Self {
            id,
            cfr_states,
            player_idx,
            forced_action: None,
        }
    }

    pub fn new_with_forced_action(
        cfr_states: Vec<Rc<RefCell<PlayerCFRState>>>,
        player_idx: usize,
        forced_action: AgentAction,
    ) -> Self {
        let id = Uuid::now_v7();

        Self {
            id,
            cfr_states,
            player_idx,
            forced_action: Some(forced_action),
        }
    }

    fn state(&self) -> &Rc<RefCell<PlayerCFRState>> {
        &self.cfr_states[self.player_idx]
    }

    fn mut_state(&mut self) -> &mut Rc<RefCell<PlayerCFRState>> {
        &mut self.cfr_states[self.player_idx]
    }

    fn expert_action(&self, game_state: &GameState, expert: usize) -> AgentAction {
        match game_state.round {
            Round::Preflop => self.preflop_action(game_state, expert),
            _ => self.postflop_action(game_state, expert),
        }
    }

    fn preflop_action(&self, game_state: &GameState, expert: usize) -> AgentAction {
        let current_round_bet = game_state.current_round_bet();
        match expert {
            0 => AgentAction::Fold,
            1 => AgentAction::Bet(current_round_bet),
            2 => AgentAction::Bet(current_round_bet + game_state.current_round_min_raise()),
            3 => AgentAction::Bet(self.pot_bet(game_state, 0.5)),
            4 => AgentAction::Bet(self.pot_bet(game_state, 1.0)),
            5 => AgentAction::Bet(current_round_bet + game_state.stacks[self.player_idx]),
            _ => panic!("Un-expected expert"),
        }
    }

    fn postflop_action(&self, game_state: &GameState, expert: usize) -> AgentAction {
        let current_round_bet = game_state.current_round_bet();
        match expert {
            0 => AgentAction::Fold,
            1 => AgentAction::Bet(current_round_bet),
            2 => AgentAction::Bet(current_round_bet + game_state.current_round_min_raise()),
            3 => AgentAction::Bet(self.pot_bet(game_state, 0.5)),
            4 => AgentAction::Bet(self.pot_bet(game_state, 0.6666)),
            5 => AgentAction::Bet(self.pot_bet(game_state, 1.0)),
            6 => AgentAction::Bet(self.prepare_shove_bet(game_state)),
            7 => AgentAction::Bet(current_round_bet + game_state.stacks[self.player_idx]),
            _ => panic!("Un-expected expert"),
        }
    }

    fn pot_bet(&self, game_state: &GameState, ratio: f32) -> f32 {
        let pot = game_state.total_pot;
        let current_round_bet = game_state.current_round_bet();
        let lower_raise = LOWER_MULT * ratio * pot;
        let upper_raise = UPPER_MULT * ratio * pot;
        if lower_raise >= upper_raise {
            current_round_bet
        } else {
            let mut rng = thread_rng();
            current_round_bet + rng.gen_range(lower_raise..upper_raise)
        }
    }

    fn prepare_shove_bet(&self, game_state: &GameState) -> f32 {
        let diff = game_state.stacks[self.player_idx] - game_state.total_pot;

        if diff <= 0.0 {
            0.0
        } else {
            let current_round_bet = game_state.current_round_bet();
            let per_player_more = diff / game_state.num_active_players() as f32;
            let lower_raise = LOWER_MULT * per_player_more;
            let upper_raise = UPPER_MULT * per_player_more;

            if lower_raise >= upper_raise {
                current_round_bet
            } else {
                let mut rng = thread_rng();
                current_round_bet + rng.gen_range(lower_raise..upper_raise)
            }
        }
    }

    fn play_round(&mut self, game_state: &GameState) -> AgentAction {
        // First explore all the experts suggestions
        self.try_all_experts(game_state);
        // Then after that which updates the regrets
        // we can get the best action
        let state = self.state().borrow();
        let data = &state.get_current_node().unwrap().data;
        match &data {
            NodeData::Player(player_data) => self.final_action(player_data, game_state),
            _ => panic!("Expected a player node"),
        }
    }

    fn final_action(&self, player_data: &PlayerData, game_state: &GameState) -> AgentAction {
        // Get the expert action
        let expert = player_data.regrets.next_action();
        self.expert_action(game_state, expert)
    }

    fn try_all_experts(&mut self, game_state: &GameState) {
        let save_points = self
            .cfr_states
            .iter()
            .map(|state| state.borrow().save_point())
            .collect::<Vec<_>>();

        let mut rewards: Vec<f32> = vec![0.0; self.num_possible_experts(game_state)];
        // For every expert try to see what the reward would be
        // then update the regret matcher
        for &expert in self.possible_experts(game_state) {
            let inner_agents: Vec<Box<dyn Agent>> = self
                .cfr_states
                .iter()
                .enumerate()
                .map(|(idx, _state)| {
                    // pass in all the states and the index of the current agent
                    // This allows each agent to run simulations with how the other
                    // agents would play
                    if self.player_idx == idx {
                        let action = self.expert_action(game_state, expert);
                        Box::new(ArenaCFRAgent::new_with_forced_action(
                            self.cfr_states.clone(),
                            idx,
                            action,
                        )) as Box<dyn Agent>
                    } else {
                        Box::new(ArenaCFRAgent::new(self.cfr_states.clone(), idx)) as Box<dyn Agent>
                    }
                })
                .collect();

            // The historians to watch
            let inner_historians = self
                .cfr_states
                .iter()
                .enumerate()
                .map(|(idx, state)| {
                    Box::new(ArenaCFRHistorian::new(state.clone(), idx)) as Box<dyn Historian>
                })
                .collect();

            let mut sim = HoldemSimulationBuilder::default()
                .agents(inner_agents)
                .historians(inner_historians)
                .game_state(game_state.clone())
                .panic_on_historian_error(true)
                .build()
                .unwrap();

            sim.run();

            rewards[expert] = sim.game_state.player_winnings[self.player_idx];

            // Reset the trees
            for (state, save_point) in self.cfr_states.iter().zip(save_points.iter()) {
                state.borrow_mut().restore_save_point(*save_point);
            }
        }
        self.update_regrets(rewards)
    }

    fn update_regrets(&mut self, rewards: Vec<f32>) {
        let mut state = self.mut_state().borrow_mut();
        let current_node = state.get_mut_current_node().unwrap();

        match &mut current_node.data {
            NodeData::Player(player_data) => {
                player_data.regrets.update_regret(aview1(&rewards)).unwrap()
            }
            _ => panic!("Expected a player node"),
        };
    }

    fn possible_experts(&self, game_state: &GameState) -> impl Iterator<Item = &usize> {
        match game_state.round {
            Round::Preflop => {
                if game_state.round_data.total_raise_count > 4 {
                    MAX_RAISE_PREFLOP_EXPERTS.iter()
                } else {
                    PREFLOP_EXPERTS.iter()
                }
            }
            _ => {
                if game_state.round_data.total_raise_count > 4 {
                    MAX_RAISE_EXPERTS.iter()
                } else {
                    EXPERTS.iter()
                }
            }
        }
    }

    fn num_possible_experts(&self, game_state: &GameState) -> usize {
        match game_state.round {
            Round::Preflop => PREFLOP_EXPERTS.len(),
            _ => EXPERTS.len(),
        }
    }

    fn next_action(&mut self, game_state: &GameState) -> AgentAction {
        self.state()
            .borrow_mut()
            .ensure_current_node(EnsureNodeType::Player(self.player_idx), game_state.round)
            .unwrap();

        // If the agent has been told to explore a path then do that
        // and clear the forced action
        if let Some(forced_action) = self.forced_action.take() {
            forced_action
        } else {
            self.play_round(game_state)
        }
    }
}

impl Agent for ArenaCFRAgent {
    fn act(&mut self, _id: &uuid::Uuid, game_state: &GameState) -> AgentAction {
        self.next_action(game_state)
    }
}
