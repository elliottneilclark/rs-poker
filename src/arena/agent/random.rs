use rand::{thread_rng, Rng};

use crate::{
    arena::{
        action::AgentAction,
        game_state::{GameState, Round},
    },
    core::Hand,
    holdem::MonteCarloGame,
};

use super::Agent;

#[derive(Debug, Clone)]
pub struct RandomAgent {
    percent_fold: Vec<f64>,
    percent_call: Vec<f64>,
}

impl RandomAgent {
    pub fn new(percent_fold: Vec<f64>, percent_call: Vec<f64>) -> Self {
        Self {
            percent_call,
            percent_fold,
        }
    }
}

impl Default for RandomAgent {
    fn default() -> Self {
        Self {
            percent_fold: vec![0.25, 0.30, 0.50],
            percent_call: vec![0.5, 0.6, 0.45],
        }
    }
}

impl Agent for RandomAgent {
    fn act(self: &mut RandomAgent, _id: &uuid::Uuid, game_state: &GameState) -> AgentAction {
        let round_data = &game_state.round_data;
        let player_bet = round_data.current_player_bet();
        let player_stack = game_state.stacks[round_data.to_act_idx];
        let curr_bet = round_data.bet;
        let raise_count = round_data.total_raise_count;

        let mut rng = thread_rng();

        // The min we can bet when not calling is the current bet plus the min raise
        // However it's possible that would put the player all in.
        let min = (curr_bet + round_data.min_raise).min(player_bet + player_stack);

        // The max we can bet going all in.
        //
        // However we don't want to overbet too early
        // so cap to a value representing how much we
        // could get everyone to put into the pot by
        // calling a pot sized bet (plus a little more for spicyness)
        //
        // That could be the same as the min
        let pot_value = (round_data.num_players_need_action() as f32 + 1.0) * game_state.total_pot;
        let max = (player_bet + player_stack).min(pot_value).max(min);

        // We shouldn't fold when checking is an option.
        let can_fold = curr_bet > player_bet;

        // As there are more raises we should look deeper
        // into the fold percentaages that the user gave us
        let fold_idx = raise_count.min((self.percent_fold.len() - 1) as u8) as usize;
        let percent_fold = self.percent_fold.get(fold_idx).map_or_else(|| 1.0, |v| *v);

        // As there are more raises we should look deeper
        // into the call percentages that the user gave us
        let call_idx = raise_count.min((self.percent_call.len() - 1) as u8) as usize;
        let percent_call = self.percent_call.get(call_idx).map_or_else(|| 1.0, |v| *v);

        // Now do the action decision
        if can_fold && rng.gen_bool(percent_fold) {
            // We can fold and the rng was in favor so fold.
            AgentAction::Fold
        } else if rng.gen_bool(percent_call) {
            // We're calling, which is the same as betting the same as the current.
            // Luckily for us the simulation will take care of us if this puts us all in.
            AgentAction::Bet(curr_bet)
        } else if max > min {
            // If there's some range and the rng didn't choose another option. So bet some
            // amount.
            AgentAction::Bet(rng.gen_range(min..max))
        } else {
            AgentAction::Bet(max)
        }
    }
}

/// This is an `Agent` implementation that chooses random actions in some
/// relation to the value of the pot. It assumes that it's up against totally
/// random cards for each hand then estimates the value of the pot for what
/// range of values to bet.
///
/// The percent_call is the percent that the agent will not bet even though it
/// values the pot above the current bet or 0 if it's the first to act.
#[derive(Debug, Clone)]
pub struct RandomPotControlAgent {
    percent_call: Vec<f64>,
}

impl RandomPotControlAgent {
    fn expected_pot(&self, game_state: &GameState) -> f32 {
        if game_state.round == Round::Preflop {
            (3.0 * game_state.big_blind).max(game_state.total_pot)
        } else {
            game_state.total_pot
        }
    }

    fn clean_hands(&self, game_state: &GameState) -> Vec<Hand> {
        let default_hand = Hand::new_with_cards(game_state.board.clone());

        let to_act_idx = game_state.to_act_idx();
        game_state
            .hands
            .clone()
            .into_iter()
            .enumerate()
            .map(|(hand_idx, hand)| {
                if hand_idx == to_act_idx {
                    hand
                } else {
                    default_hand.clone()
                }
            })
            .collect()
    }

    fn monte_carlo_based_action(
        &self,
        game_state: &GameState,
        mut monte: MonteCarloGame,
    ) -> AgentAction {
        // We play some trickery to make sure that someone will call before there's
        // money in the pot
        let expected_pot = self.expected_pot(game_state);
        // run the monte carlo simulation a lot of times to see who would win with the
        // knowledge that we have. Keeping in mind that we have no information and are
        // actively guessing no hand ranges at all. So this is likely a horrible way to
        // estimate hand strength
        //
        // Then truncate the values to f32.
        let values: Vec<f32> = monte.estimate_equity(1_000).into_iter().collect();
        let to_act_idx = game_state.to_act_idx();

        // How much do I actually value the pot right now?
        let my_value = values.get(to_act_idx).unwrap_or(&0.0_f32) * expected_pot;

        // What have we already put into the pot for the round?
        let bet_already = game_state.current_round_player_bet(to_act_idx);
        // How much total is required to continue
        let to_call = game_state.current_round_bet();
        // What more is needed from us
        let needed = to_call - bet_already;

        // If we don't value the pot at what's required then just bail out.
        if my_value < needed {
            AgentAction::Fold
        } else {
            self.random_action(game_state, my_value)
        }
    }

    fn random_action(&self, game_state: &GameState, max_value: f32) -> AgentAction {
        let mut rng = thread_rng();
        // Use the number of bets to determine the call percentage
        let round_data = &game_state.round_data;
        let raise_count = round_data.total_raise_count;

        let call_idx = raise_count.min((self.percent_call.len() - 1) as u8) as usize;
        let percent_call = self.percent_call.get(call_idx).map_or_else(|| 1.0, |v| *v);

        if rng.gen_bool(percent_call) {
            AgentAction::Bet(round_data.bet)
        } else {
            // Even thoush this is a random action try not to under min raise
            let min_raise = round_data.min_raise;
            // We always give some room to bet
            let low = round_data.bet + min_raise;
            let bet_value = rng.gen_range(low..max_value.max(low + min_raise));

            // Round the chosen value to take f32 to i32
            AgentAction::Bet(bet_value)
        }
    }

    pub fn new(percent_call: Vec<f64>) -> Self {
        Self { percent_call }
    }
}

impl Agent for RandomPotControlAgent {
    fn act(&mut self, _id: &uuid::Uuid, game_state: &GameState) -> AgentAction {
        // We don't want to cheat.
        // So replace all the hands but our own
        let clean_hands = self.clean_hands(game_state);
        // Now check if we can simulate that
        if let Ok(monte) = MonteCarloGame::new(clean_hands) {
            self.monte_carlo_based_action(game_state, monte)
        } else {
            AgentAction::Fold
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        arena::{
            test_util::{assert_valid_game_state, assert_valid_round_data},
            HoldemSimulationBuilder,
        },
        core::{Deck, FlatDeck},
    };

    use super::*;

    #[test_log::test]
    fn test_random_five_nl() {
        let mut deck: FlatDeck = Deck::default().into();

        let stacks = vec![100.0; 5];
        let mut game_state = GameState::new(stacks, 10.0, 5.0, 0.0, 0);
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<RandomAgent>::default(),
            Box::<RandomAgent>::default(),
            Box::<RandomAgent>::default(),
            Box::<RandomAgent>::default(),
            Box::<RandomAgent>::default(),
        ];

        // Add two random cards to every hand.
        for hand in game_state.hands.iter_mut() {
            hand.push(deck.deal().unwrap());
            hand.push(deck.deal().unwrap());
        }

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .deck(deck)
            .build()
            .unwrap();

        sim.run();

        let min_stack = sim
            .game_state
            .stacks
            .clone()
            .into_iter()
            .reduce(f32::min)
            .unwrap();
        let max_stack = sim
            .game_state
            .stacks
            .clone()
            .into_iter()
            .reduce(f32::max)
            .unwrap();

        assert_ne!(min_stack, max_stack, "There should have been some betting.");

        assert_valid_round_data(&sim.game_state.round_data);
        assert_valid_game_state(&sim.game_state);
    }

    #[test_log::test]
    fn test_five_pot_control() {
        let stacks = vec![100.0; 5];
        let game_state = GameState::new(stacks, 10.0, 5.0, 0.0, 0);
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::new(RandomPotControlAgent::new(vec![0.3])),
            Box::new(RandomPotControlAgent::new(vec![0.3])),
            Box::new(RandomPotControlAgent::new(vec![0.3])),
            Box::new(RandomPotControlAgent::new(vec![0.3])),
            Box::new(RandomPotControlAgent::new(vec![0.3])),
        ];

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();

        sim.run();

        let min_stack = sim
            .game_state
            .stacks
            .clone()
            .into_iter()
            .reduce(f32::min)
            .unwrap();
        let max_stack = sim
            .game_state
            .stacks
            .clone()
            .into_iter()
            .reduce(f32::max)
            .unwrap();

        assert_ne!(min_stack, max_stack, "There should have been some betting.");
        assert_valid_round_data(&sim.game_state.round_data);
        assert_valid_game_state(&sim.game_state);
    }

    #[test_log::test]
    fn test_random_agents_no_fold_get_all_rounds() {
        let stacks = vec![100.0; 5];
        let game_state = GameState::new(stacks, 10.0, 5.0, 0.0, 0);
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::new(RandomAgent::new(vec![0.0], vec![0.75])),
            Box::new(RandomAgent::new(vec![0.0], vec![0.75])),
            Box::new(RandomAgent::new(vec![0.0], vec![0.75])),
            Box::new(RandomAgent::new(vec![0.0], vec![0.75])),
            Box::new(RandomAgent::new(vec![0.0], vec![0.75])),
        ];
        let mut sim = HoldemSimulationBuilder::default()
            .agents(agents)
            .game_state(game_state)
            .build()
            .unwrap();

        sim.run();
        assert!(sim.game_state.is_complete());
        assert_valid_game_state(&sim.game_state);
    }
}
