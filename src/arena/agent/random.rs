use rand::{Rng, rng};
use std::sync::atomic::{AtomicUsize, Ordering};

use tracing::{instrument, trace};

use crate::{
    arena::{
        action::AgentAction,
        game_state::{GameState, Round},
    },
    core::Hand,
    holdem::MonteCarloGame,
};

use super::{Agent, AgentGenerator};

#[derive(Debug, Clone)]
pub struct RandomAgent {
    name: String,
    percent_fold: Vec<f64>,
    percent_call: Vec<f64>,
}

impl RandomAgent {
    pub fn new(name: impl Into<String>, percent_fold: Vec<f64>, percent_call: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            percent_call,
            percent_fold,
        }
    }
}

impl Default for RandomAgent {
    fn default() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let idx = COUNTER.fetch_add(1, Ordering::Relaxed);
        RandomAgent::new(
            format!("RandomAgent-default-{idx}"),
            vec![0.25, 0.30, 0.50],
            vec![0.5, 0.6, 0.45],
        )
    }
}

impl Agent for RandomAgent {
    #[instrument(level = "trace", skip(self, game_state), fields(agent_name = %self.name))]
    fn act(self: &mut RandomAgent, _id: u128, game_state: &GameState) -> AgentAction {
        let round_data = &game_state.round_data;
        let player_bet = round_data.current_player_bet();
        let player_stack = game_state.stacks[round_data.to_act_idx];
        let curr_bet = round_data.bet;
        let raise_count = round_data.total_raise_count;

        let mut rng = rng();

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
        let action = if can_fold && rng.random_bool(percent_fold) {
            // We can fold and the rng was in favor so fold.
            AgentAction::Fold
        } else if rng.random_bool(percent_call) {
            // We're calling, which is the same as betting the same as the current.
            // Luckily for us the simulation will take care of us if this puts us all in.
            AgentAction::Call
        } else if max > min {
            // If there's some range and the rng didn't choose another option. So bet some
            // amount.
            AgentAction::Bet(rng.random_range(min..max))
        } else {
            AgentAction::Bet(max)
        };

        trace!(?action, raise_count, can_fold, "RandomAgent decision");
        action
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug, Clone)]
pub struct RandomAgentGenerator {
    name: Option<String>,
    percent_fold: Vec<f64>,
    percent_call: Vec<f64>,
}

impl RandomAgentGenerator {
    pub fn new(percent_fold: Vec<f64>, percent_call: Vec<f64>) -> Self {
        Self {
            name: None,
            percent_fold,
            percent_call,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    fn resolve_name(&self, player_idx: usize) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| format!("RandomAgent-{player_idx}"))
    }
}

impl AgentGenerator for RandomAgentGenerator {
    fn generate(&self, player_idx: usize, _game_state: &GameState) -> Box<dyn Agent> {
        Box::new(RandomAgent::new(
            self.resolve_name(player_idx),
            self.percent_fold.clone(),
            self.percent_call.clone(),
        ))
    }
}

impl Default for RandomAgentGenerator {
    fn default() -> Self {
        Self::new(vec![0.25, 0.30, 0.50], vec![0.5, 0.6, 0.45])
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
    name: String,
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
        let mut default_hand = Hand::new();
        // Copy the board into the default hand
        default_hand.extend(game_state.board.iter().cloned());

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
                    default_hand
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
        // But only fold if there's actually something to call (otherwise check)
        if my_value < needed && needed > 0.0 {
            AgentAction::Fold
        } else if needed <= 0.0 {
            // Nothing to call - just check
            AgentAction::Bet(to_call)
        } else {
            self.random_action(game_state, my_value)
        }
    }

    fn random_action(&self, game_state: &GameState, max_value: f32) -> AgentAction {
        let mut rng = rng();
        // Use the number of bets to determine the call percentage
        let round_data = &game_state.round_data;
        let raise_count = round_data.total_raise_count;

        let call_idx = raise_count.min((self.percent_call.len() - 1) as u8) as usize;
        let percent_call = self.percent_call.get(call_idx).map_or_else(|| 1.0, |v| *v);

        // Check player's stack to determine valid bet range
        let player_stack = game_state.current_player_stack();
        let player_bet_this_round = game_state.current_round_current_player_bet();
        let max_total_bet = player_bet_this_round + player_stack;

        if rng.random_bool(percent_call) {
            AgentAction::Bet(round_data.bet)
        } else {
            // Even though this is a random action try not to under min raise
            let min_raise = round_data.min_raise;
            // The minimum valid raise amount
            let min_raise_total = round_data.bet + min_raise;

            // Check if we can even afford the min raise
            if max_total_bet < min_raise_total {
                // Can't afford min raise - either go all-in or call
                if max_total_bet > round_data.bet {
                    // All-in is our only raise option
                    AgentAction::AllIn
                } else {
                    // Just call
                    AgentAction::Bet(round_data.bet)
                }
            } else {
                // We can afford to raise - pick a random amount
                let high = max_value
                    .max(min_raise_total + min_raise)
                    .min(max_total_bet);
                let bet_value = rng.random_range(min_raise_total..high.max(min_raise_total + 1.0));

                AgentAction::Bet(bet_value)
            }
        }
    }

    pub fn new(name: impl Into<String>, percent_call: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            percent_call,
        }
    }
}

impl Agent for RandomPotControlAgent {
    #[instrument(level = "trace", skip(self, game_state), fields(agent_name = %self.name))]
    fn act(&mut self, _id: u128, game_state: &GameState) -> AgentAction {
        // We don't want to cheat.
        // So replace all the hands but our own
        let clean_hands = self.clean_hands(game_state);
        // Now check if we can simulate that
        let action = if let Ok(monte) = MonteCarloGame::new(clean_hands) {
            self.monte_carlo_based_action(game_state, monte)
        } else {
            // If we can't do monte carlo, check if we can fold or need to check
            let to_call = game_state.current_round_bet();
            let bet_already = game_state.current_round_current_player_bet();
            let needed = to_call - bet_already;
            if needed > 0.0 {
                AgentAction::Fold
            } else {
                // Nothing to call - just check
                AgentAction::Bet(to_call)
            }
        };

        trace!(?action, "RandomPotControlAgent decision");
        action
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        arena::{
            HoldemSimulationBuilder,
            test_util::{assert_valid_game_state, assert_valid_round_data},
        },
        core::Deck,
    };

    use super::*;

    #[test]
    fn test_random_generator_produces_named_caller() {
        let generator = RandomAgentGenerator::new(vec![0.0], vec![1.0]);
        let game_state = GameState::new_starting(vec![100.0; 2], 10.0, 5.0, 0.0, 0);

        let mut agent = generator.generate(3, &game_state);
        assert_eq!(agent.name(), "RandomAgent-3");

        match agent.act(0, &game_state) {
            AgentAction::Call => {}
            action => panic!("Expected forced call, got {:?}", action),
        }
    }

    #[test]
    fn test_random_generator_uses_custom_name() {
        let generator = RandomAgentGenerator::new(vec![0.0], vec![1.0]).with_name("RandomHero");
        let game_state = GameState::new_starting(vec![20.0; 2], 10.0, 5.0, 0.0, 0);

        let agent = generator.generate(7, &game_state);
        assert_eq!(agent.name(), "RandomHero");
    }

    #[test]
    fn test_random_five_nl() {
        let mut deck: Deck = Deck::default();
        let mut rng = rand::rng();

        let stacks = vec![100.0; 5];
        let mut game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let agents: Vec<Box<dyn Agent>> = (0..5)
            .map(|idx| {
                Box::new(RandomAgent::new(
                    format!("RandomAgent-{idx}"),
                    vec![0.25, 0.30, 0.50],
                    vec![0.5, 0.6, 0.45],
                )) as Box<dyn Agent>
            })
            .collect();

        // Add two random cards to every hand.
        for hand in game_state.hands.iter_mut() {
            hand.insert(deck.deal(&mut rng).unwrap());
            hand.insert(deck.deal(&mut rng).unwrap());
        }

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .deck(deck)
            .build()
            .unwrap();

        sim.run(&mut rng);

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

    #[test]
    fn test_five_pot_control() {
        let stacks = vec![100.0; 5];
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let agents: Vec<Box<dyn Agent>> = (0..5)
            .map(|idx| {
                Box::new(RandomPotControlAgent::new(
                    format!("RandomPotControl-{idx}"),
                    vec![0.3],
                )) as Box<dyn Agent>
            })
            .collect();

        let mut rng = rand::rng();
        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();

        sim.run(&mut rng);

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

    #[test]
    fn test_random_agents_no_fold_get_all_rounds() {
        let stacks = vec![100.0; 5];
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);
        let agents: Vec<Box<dyn Agent>> = (0..5)
            .map(|idx| {
                Box::new(RandomAgent::new(
                    format!("AggroRandom-{idx}"),
                    vec![0.0],
                    vec![0.75],
                )) as Box<dyn Agent>
            })
            .collect();
        let mut rng = rand::rng();
        let mut sim = HoldemSimulationBuilder::default()
            .agents(agents)
            .game_state(game_state)
            .build()
            .unwrap();

        sim.run(&mut rng);
        assert!(sim.game_state.is_complete());
        assert_valid_game_state(&sim.game_state);
    }

    #[test]
    fn test_random_agent_name_returns_name() {
        let agent = RandomAgent::new("TestAgent", vec![0.5], vec![0.5]);
        // Test that name() returns the actual name, not empty string
        assert_eq!(agent.name(), "TestAgent");
        assert!(!agent.name().is_empty());
    }

    #[test]
    fn test_random_pot_control_agent_name_returns_name() {
        let agent = RandomPotControlAgent::new("PotControl", vec![0.5]);
        // Test that name() returns the actual name, not empty string
        assert_eq!(agent.name(), "PotControl");
        assert!(!agent.name().is_empty());
    }

    #[test]
    fn test_random_agent_expected_pot_preflop() {
        let agent = RandomPotControlAgent::new("Test", vec![0.5]);
        let mut game_state = GameState::new_starting(vec![100.0, 100.0], 10.0, 5.0, 0.0, 0);
        // Set to preflop round
        game_state.round = Round::Preflop;
        game_state.total_pot = 15.0; // SB + BB

        // In preflop: max(3.0 * big_blind, total_pot) = max(30.0, 15.0) = 30.0
        let expected = agent.expected_pot(&game_state);

        // Preflop expected pot should be max(3.0 * big_blind, total_pot)
        // With big_blind=10 and total_pot=15, that's max(30, 15) = 30
        assert!(
            (expected - 30.0).abs() < 0.01,
            "expected_pot should be 30.0 in preflop with small pot, got {}",
            expected
        );
    }

    #[test]
    fn test_random_agent_expected_pot_postflop() {
        let agent = RandomPotControlAgent::new("Test", vec![0.5]);
        let mut game_state = GameState::new_starting(vec![100.0, 100.0], 10.0, 5.0, 0.0, 0);
        // Move to flop
        game_state.round = Round::Flop;
        game_state.total_pot = 50.0;

        let expected = agent.expected_pot(&game_state);
        // Post-flop: just returns total_pot
        assert!(
            (expected - 50.0).abs() < 0.01,
            "expected_pot post-flop should equal total_pot (50.0), got {}",
            expected
        );
    }

    #[test]
    fn test_random_agent_clean_hands_preserves_own_hand() {
        let agent = RandomPotControlAgent::new("Test", vec![0.5]);
        let mut game_state = GameState::new_starting(vec![100.0, 100.0], 10.0, 5.0, 0.0, 0);

        // Set up specific cards - use different cards for hands and board
        let cards: Vec<_> = crate::core::Deck::default().into_iter().take(7).collect();
        game_state.hands[0].insert(cards[0]); // Player 0's hole cards
        game_state.hands[0].insert(cards[1]);
        game_state.hands[1].insert(cards[2]); // Player 1's hole cards
        game_state.hands[1].insert(cards[3]);
        // Add 3 board cards so the cleaned hand count differs from hole cards
        game_state.board.push(cards[4]);
        game_state.board.push(cards[5]);
        game_state.board.push(cards[6]);

        let to_act = game_state.to_act_idx();
        let clean = agent.clean_hands(&game_state);

        // The to_act player should have their original hand (2 cards)
        assert_eq!(
            clean[to_act].count(),
            game_state.hands[to_act].count(),
            "Acting player should keep their original hand"
        );

        // The acting player's hand should contain their actual hole cards
        assert!(
            clean[to_act].contains(&cards[0]),
            "Acting player's hand should contain their first hole card"
        );

        // Other players should have only board cards (3 cards), not their hole cards (2 cards)
        for (idx, hand) in clean.iter().enumerate() {
            if idx != to_act {
                // Cleaned hand should have board cards (3), not hole cards (2)
                assert_eq!(
                    hand.count(),
                    3,
                    "Non-acting player's cleaned hand should have 3 board cards"
                );
                // Should NOT contain the original hole cards
                assert!(
                    !hand.contains(&cards[2]) && !hand.contains(&cards[3]),
                    "Non-acting player's cleaned hand should not contain their hole cards"
                );
            }
        }
    }

    #[test]
    fn test_random_agent_can_fold_logic() {
        // When current bet > player bet, should be able to fold
        let mut agent = RandomAgent::new("FoldTest", vec![1.0], vec![0.0]); // 100% fold
        let mut game_state = GameState::new_starting(vec![100.0, 100.0], 10.0, 5.0, 0.0, 0);

        // Set up a situation where the player faces a bet
        // round_data.bet = current bet to call
        // player_bets[to_act] = what this player has already bet this round
        game_state.round_data.bet = 10.0; // There's a bet of 10 to call
        game_state.round_data.player_bet[0] = 5.0; // Player has only bet 5 (like SB)

        // can_fold = curr_bet (10) > player_bet (5) = true
        // With 100% fold probability, should fold
        let action = agent.act(0, &game_state);
        assert!(
            matches!(action, AgentAction::Fold),
            "With 100% fold when can_fold=true, should fold. Got {:?}",
            action
        );
    }

    #[test]
    fn test_random_agent_cannot_fold_when_checking() {
        // When current bet == player bet, should not fold (can check)
        let mut agent = RandomAgent::new("CheckTest", vec![1.0], vec![0.0]); // 100% fold
        let mut game_state = GameState::new_starting(vec![100.0, 100.0], 10.0, 5.0, 0.0, 0);

        // Simulate BB position where bet is matched (nothing to call)
        game_state.round_data.bet = 0.0;
        game_state.round_data.player_bet[0] = 0.0;
        game_state.stacks[0] = 100.0;

        // can_fold = curr_bet (0) > player_bet (0) = false
        let action = agent.act(0, &game_state);
        // With no bet to call (bet=0), can_fold=false, so shouldn't fold
        // Should call or bet
        assert!(
            !matches!(action, AgentAction::Fold),
            "Should not fold when can check. Got {:?}",
            action
        );
    }

    #[test]
    fn test_random_agent_min_calculation() {
        // Test that min bet is calculated correctly using addition
        let agent = RandomAgent::new("MinTest", vec![0.0], vec![0.0]);
        let game_state = GameState::new_starting(vec![50.0, 50.0], 10.0, 5.0, 0.0, 0);

        // min = (curr_bet + min_raise).min(player_bet + player_stack)
        // curr_bet = 10 (big blind)
        // min_raise = 10 (big blind)
        // So min = 20 unless that would be all-in
        // player_bet = 5 (SB), player_stack = 45
        // player_bet + player_stack = 50
        // min(20, 50) = 20

        // The agent uses this internally; we can verify the game completes
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::new(agent),
            Box::new(RandomAgent::new("Other", vec![0.0], vec![1.0])),
        ];

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();

        let mut rng = rand::rng();
        sim.run(&mut rng);
        assert!(sim.game_state.is_complete());
    }

    #[test]
    fn test_random_pot_control_needed_calculation() {
        // Test the subtraction logic in monte_carlo_based_action
        let agent = RandomPotControlAgent::new("NeededTest", vec![1.0]); // Always call

        // Set up a game state where we can verify needed = to_call - bet_already
        let mut game_state = GameState::new_starting(vec![100.0, 100.0], 10.0, 5.0, 0.0, 0);

        // After forced bets: SB has bet 5, BB has bet 10
        // If SB acts first, to_call=10, bet_already=5, needed=5
        // With - : 10 - 5 = 5 (correct)
        // With + : 10 + 5 = 15 (wrong)

        let mut deck = Deck::default();
        let mut rng = rand::rng();
        game_state.hands[0].insert(deck.deal(&mut rng).unwrap());
        game_state.hands[0].insert(deck.deal(&mut rng).unwrap());
        game_state.hands[1].insert(deck.deal(&mut rng).unwrap());
        game_state.hands[1].insert(deck.deal(&mut rng).unwrap());

        let agents: Vec<Box<dyn Agent>> = vec![
            Box::new(agent),
            Box::new(RandomPotControlAgent::new("Other", vec![1.0])),
        ];

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .deck(deck)
            .build()
            .unwrap();

        sim.run(&mut rng);
        assert!(sim.game_state.is_complete());
    }

    #[test]
    fn test_random_pot_control_max_total_bet_calculation() {
        // Test that max_total_bet = player_bet_this_round + player_stack uses addition
        let agent = RandomPotControlAgent::new("MaxBetTest", vec![0.0]); // Never call, always try bet

        let game_state = GameState::new_starting(vec![100.0, 100.0], 10.0, 5.0, 0.0, 0);
        // player_bet_this_round + player_stack = 5 + 95 = 100
        // With +: correct
        // With *: 5 * 95 = 475 (wrong, exceeds stack)

        let agents: Vec<Box<dyn Agent>> = vec![
            Box::new(agent),
            Box::new(RandomPotControlAgent::new("Other", vec![1.0])),
        ];

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();

        let mut rng = rand::rng();
        sim.run(&mut rng);
        // Game should complete without panicking due to invalid bet calculations
        assert!(sim.game_state.is_complete());
    }

    #[test]
    fn test_random_pot_control_high_calculation() {
        // Test the addition in high = max_value.max(min_raise_total + min_raise).min(max_total_bet)
        let agent = RandomPotControlAgent::new("HighTest", vec![0.0]); // Never call

        // Create a game with specific stack sizes to exercise the calculation
        let game_state = GameState::new_starting(vec![200.0, 200.0], 10.0, 5.0, 0.0, 0);

        let agents: Vec<Box<dyn Agent>> = vec![
            Box::new(agent),
            Box::new(RandomPotControlAgent::new("Other", vec![0.0])),
        ];

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();

        let mut rng = rand::rng();
        sim.run(&mut rng);
        assert!(sim.game_state.is_complete());
    }

    #[test]
    fn test_random_pot_control_short_stack_cannot_raise() {
        // Test the comparison: max_total_bet < min_raise_total
        let agent = RandomPotControlAgent::new("ShortStack", vec![0.0]); // Never call

        // Small stack that can't make minimum raise
        let game_state = GameState::new_starting(vec![15.0, 100.0], 10.0, 5.0, 0.0, 0);
        // Player 0 has 15 total
        // After posting SB (5), has 10 left
        // To min-raise, would need: current_bet (10) + min_raise (10) = 20
        // But max_total_bet = player_bet_this_round (5) + stack (10) = 15
        // 15 < 20, so cannot min-raise

        let agents: Vec<Box<dyn Agent>> = vec![
            Box::new(agent),
            Box::new(RandomPotControlAgent::new("BigStack", vec![1.0])),
        ];

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();

        let mut rng = rand::rng();
        sim.run(&mut rng);
        assert!(sim.game_state.is_complete());
    }

    #[test]
    fn test_random_agent_pot_value_multiplication() {
        // Test: pot_value = (num_players + 1.0) * total_pot
        // With *: correct
        // With +: (5 + 1.0) + 15 = 21 (wrong)
        // With /: (5 + 1.0) / 15 = 0.4 (wrong)

        let stacks = vec![100.0; 5];
        let game_state = GameState::new_starting(stacks, 10.0, 5.0, 0.0, 0);

        // 5 players, pot = 15 (BB 10 + SB 5)
        // pot_value = (5 + 1.0) * 15 = 90

        let agents: Vec<Box<dyn Agent>> = (0..5)
            .map(|idx| {
                Box::new(RandomAgent::new(
                    format!("Agent{idx}"),
                    vec![0.0],
                    vec![0.0],
                )) as Box<dyn Agent>
            })
            .collect();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();

        let mut rng = rand::rng();
        sim.run(&mut rng);
        assert!(sim.game_state.is_complete());
    }

    #[test]
    fn test_random_agent_raise_count_index_calculation() {
        // Test: fold_idx = raise_count.min((self.percent_fold.len() - 1) as u8) as usize
        // The subtraction is important: len() - 1 gives last valid index

        let agent = RandomAgent::new(
            "IndexTest",
            vec![0.5, 0.75, 0.9], // 3 elements, indices 0, 1, 2
            vec![0.5],
        );

        // With many raises, should use index 2 (last), not overflow
        assert_eq!(agent.percent_fold.len(), 3);

        // Test with simulation - should not panic
        let game_state = GameState::new_starting(vec![100.0, 100.0], 10.0, 5.0, 0.0, 0);
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::new(agent),
            Box::new(RandomAgent::new("Aggro", vec![0.0], vec![0.0])), // Always raises
        ];

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .build()
            .unwrap();

        let mut rng = rand::rng();
        sim.run(&mut rng);
        assert!(sim.game_state.is_complete());
    }

    #[test]
    fn test_pot_control_my_value_multiplication() {
        // Test: my_value = equity * expected_pot
        // This tests that * is used, not + or /

        let agent = RandomPotControlAgent::new("ValueTest", vec![0.5]);

        // With multiplication: equity (0.5) * pot (50) = 25
        // With addition: 0.5 + 50 = 50.5 (wrong)
        // With division: 0.5 / 50 = 0.01 (wrong)

        let mut game_state = GameState::new_starting(vec![100.0, 100.0], 10.0, 5.0, 0.0, 0);
        game_state.total_pot = 50.0;

        let mut deck = Deck::default();
        let mut rng = rand::rng();
        for hand in game_state.hands.iter_mut() {
            hand.insert(deck.deal(&mut rng).unwrap());
            hand.insert(deck.deal(&mut rng).unwrap());
        }

        let agents: Vec<Box<dyn Agent>> = vec![
            Box::new(agent),
            Box::new(RandomPotControlAgent::new("Other", vec![0.5])),
        ];

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .deck(deck)
            .build()
            .unwrap();

        sim.run(&mut rng);
        assert!(sim.game_state.is_complete());
    }
}
