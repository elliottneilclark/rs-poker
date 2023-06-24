use rand::{thread_rng, Rng};

use crate::arena::{action::AgentAction, game_state::GameState};

use super::Agent;

#[derive(Debug, Clone)]
pub struct RandomAgent {
    percent_fold: f64,
    percent_call: f64,
}

impl RandomAgent {
    pub fn new(percent_fold: f64, percent_call: f64) -> Self {
        Self {
            percent_call,
            percent_fold,
        }
    }
}

impl Default for RandomAgent {
    fn default() -> Self {
        Self {
            percent_fold: 0.15,
            percent_call: 0.5,
        }
    }
}

impl Agent for RandomAgent {
    fn act(self: &mut RandomAgent, game_state: &GameState) -> AgentAction {
        let current_round_data = game_state.current_round_data();
        let player_bet = current_round_data.current_player_bet();
        let player_stack = game_state.stacks[current_round_data.to_act_idx];
        let curr_bet = current_round_data.bet;
        let mut rng = thread_rng();

        // The min we can bet when not calling is the current bet plus the min raise
        // However it's possible that would put the player all in.
        let min = (curr_bet + current_round_data.min_raise).min(player_bet + player_stack);

        // The max we can bet going all in.
        //
        // However we don't want to overbet too early
        // so cap to a value representing how much we
        // could get everyone to put into the pot by
        // calling a pot sized bet (plus a little more for spicyness)
        //
        // That could be the same as the min
        let pot_value = (current_round_data.num_active_players() as i32 + 1) * game_state.total_pot;
        let max = (player_bet + player_stack).min(pot_value).max(min);

        // We shouldn't fold when checking is an option.
        let can_fold = curr_bet > player_bet;

        // Now do the action decision
        if can_fold && rng.gen_bool(self.percent_fold) {
            // We can fold and the rng was in favor so fold.
            AgentAction::Fold
        } else if rng.gen_bool(self.percent_call) {
            // We're calling, which is the same as betting the same as the current.
            // Luckily for us the simulation will take care of us if this puts us all in.
            AgentAction::Bet(curr_bet)
        } else if max > min {
            // If there's some range and the rng didn't choose another option. So bet some
            // ammount.
            AgentAction::Bet(rng.gen_range(min..max))
        } else {
            AgentAction::Bet(max)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        arena::{
            game_state::GameState, simulation::HoldemSimulation, test_util::assert_valid_round_data,
        },
        core::{Deck, FlatDeck},
    };

    use super::*;

    #[test]
    fn test_random_five_nl() {
        let mut deck: FlatDeck = Deck::default().into();
        deck.shuffle();

        let stacks = vec![100; 5];
        let mut game_state = GameState::new(stacks, 10, 5, 0);
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

        let mut sim = HoldemSimulation::new_with_agents_and_deck(game_state, agents, deck);

        sim.run();

        let min_stack = sim.game_state.stacks.iter().min().unwrap();
        let max_stack = sim.game_state.stacks.iter().max().unwrap();

        assert_ne!(min_stack, max_stack, "There should have been some betting.");
        sim.game_state
            .round_data
            .iter()
            .for_each(assert_valid_round_data);
    }

    #[test]
    fn test_random_agents_no_fold_get_all_rounds() {
        let stacks = vec![100; 5];
        let game_state = GameState::new(stacks, 10, 5, 0);
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::new(RandomAgent::new(0.0, 0.75)),
            Box::new(RandomAgent::new(0.0, 0.75)),
            Box::new(RandomAgent::new(0.0, 0.75)),
            Box::new(RandomAgent::new(0.0, 0.75)),
            Box::new(RandomAgent::new(0.0, 0.75)),
        ];
        let mut sim = HoldemSimulation::new_with_agents(game_state, agents);

        sim.run();
        assert!(sim.game_state.is_complete());
        assert_eq!(7, sim.game_state.round_data.len());
    }
}
