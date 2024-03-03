use rand::{rngs::ThreadRng, thread_rng, Rng};

use crate::core::{CardBitSet, FlatDeck};

use super::{
    agent::FoldingAgent, errors::HoldemSimulationError, historian::Historian, Agent, GameState,
    HoldemSimulation,
};

// Some builder methods to help with turning a builder struct into a ready
// simulation
fn build_flat_deck<R: Rng>(game_state: &GameState, rng: &mut R) -> FlatDeck {
    let mut d = CardBitSet::default();

    for hand in game_state.hands.iter() {
        for card in hand.iter() {
            d.remove(*card);
        }
    }
    let mut flat_deck: FlatDeck = d.into();
    flat_deck.shuffle(rng);
    flat_deck
}

fn build_agents(num_agents: usize) -> Vec<Box<dyn Agent>> {
    (0..num_agents)
        .map(|_| -> Box<dyn Agent> { Box::<FoldingAgent>::default() })
        .collect()
}

/// # HoldemSimulationBuilder
///
/// `RngHoldemSimulationBuilder` is a builder to allow for complex
/// configurations of a holdem simulation played via agents. A game state is
/// required, other fields are optional.
///
/// `HolemSimulationBuilder` is a type alias
/// for `RngHoldemSimulationBuilder<ThreadRng>` which is the default builder.
///
/// ## Setters
///
/// Each setter will set the optional value to the passed in value. Then return
/// the mutated builder.
///
/// While agents are not required the default is a full ring of folding agents.
/// So likely not that interesting a simulation.
///
/// ## Examples
///
/// ```
/// use rs_poker::arena::{GameState, HoldemSimulationBuilder};
///
/// let game_state = GameState::new(vec![100.0; 5], 2.0, 1.0, 0.0, 3);
/// let sim = HoldemSimulationBuilder::default()
///     .game_state(game_state)
///     .build()
///     .unwrap();
/// ```
/// However sometimes you want to use a known but random simulation. In that
/// case you can pass in the rng like this:
///
/// ```
/// use rand::{rngs::StdRng, SeedableRng};
/// use rs_poker::arena::{GameState, RngHoldemSimulationBuilder};
///
/// let game_state = GameState::new(vec![100.0; 5], 2.0, 1.0, 0.0, 3);
/// let rng = StdRng::seed_from_u64(420);
/// let sim = RngHoldemSimulationBuilder::default()
///     .game_state(game_state)
///     .rng(rng)
///     .build()
///     .unwrap();
/// ```
pub struct RngHoldemSimulationBuilder<R: Rng> {
    agents: Option<Vec<Box<dyn Agent>>>,
    historians: Vec<Box<dyn Historian>>,
    game_state: Option<GameState>,
    deck: Option<FlatDeck>,
    rng: Option<R>,
}

/// # Examples
/// ```
/// use rand::{rngs::StdRng, SeedableRng};
/// use rs_poker::arena::{agent::FoldingAgent, Agent};
/// use rs_poker::arena::{GameState, RngHoldemSimulationBuilder};
///
/// let game_state = GameState::new(vec![100.0; 5], 2.0, 1.0, 0.0, 3);
/// let agents: Vec<Box<dyn Agent>> = (0..5)
///     .map(|_| Box::<FoldingAgent>::default() as Box<dyn Agent>)
///     .collect();
/// let rng = StdRng::seed_from_u64(420);
/// let sim = RngHoldemSimulationBuilder::default()
///     .game_state(game_state)
///     .agents(agents)
///     .rng(rng)
///     .build();
/// ```
impl<R: Rng> RngHoldemSimulationBuilder<R> {
    /// Set the agents for the simulation created by this builder.
    pub fn agents(mut self, agents: Vec<Box<dyn Agent>>) -> Self {
        self.agents = Some(agents);
        self
    }

    /// Set the game state for ths simulation created by this bu
    pub fn game_state(mut self, game_state: GameState) -> Self {
        self.game_state = Some(game_state);
        self
    }

    /// Set the deck. If not set a deck will be
    /// created from the game state and shuffled.
    pub fn deck(mut self, deck: FlatDeck) -> Self {
        self.deck = Some(deck);
        self
    }

    pub fn rng(mut self, rng: R) -> Self {
        self.rng = Some(rng);
        self
    }

    /// Set the historians for the simulation created by this builder.
    pub fn historians(mut self, historians: Vec<Box<dyn Historian>>) -> Self {
        self.historians = historians;
        self
    }

    /// Given the fields already specified build any that are not specified and
    /// create a new HoldemSimulation.
    ///
    /// @returns HoldemSimulationError if no game_state was given.
    pub fn build(self) -> Result<HoldemSimulation, HoldemSimulationError> {
        let game_state = self
            .game_state
            .ok_or(HoldemSimulationError::NeedGameState)?;

        let agents = self
            .agents
            .unwrap_or_else(|| build_agents(game_state.hands.len()));

        // If the deck was passed in use that with no shuffling to allow for
        // this to be a determinitic simulation
        let deck = self.deck.unwrap_or_else(|| {
            if let Some(mut rng) = self.rng {
                build_flat_deck(&game_state, &mut rng)
            } else {
                let mut rng = thread_rng();
                build_flat_deck(&game_state, &mut rng)
            }
        });

        // Create a new simulation id.
        // This will be used to track
        // this exact run of a simulation.
        let id = uuid::Uuid::now_v7();

        Ok(HoldemSimulation {
            agents,
            game_state,
            deck,
            id,
            historians: self.historians,
        })
    }
}

impl<R: Rng> Default for RngHoldemSimulationBuilder<R> {
    fn default() -> Self {
        Self {
            agents: None,
            historians: vec![],
            game_state: None,
            deck: None,
            rng: None,
        }
    }
}

/// The rng is ThreadRng.
pub type HoldemSimulationBuilder = RngHoldemSimulationBuilder<ThreadRng>;

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};

    use crate::{arena::game_state::Round, core::Card};

    use super::*;

    #[test_log::test]
    fn test_single_step_agent() {
        let stacks = vec![100.0; 9];
        let game_state = GameState::new(stacks, 10.0, 5.0, 1.0, 0);
        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .build()
            .unwrap();

        assert_eq!(100.0, sim.game_state.stacks[1]);
        assert_eq!(100.0, sim.game_state.stacks[2]);
        // We are starting out.
        sim.run_round();
        assert_eq!(100.0, sim.game_state.stacks[1]);
        assert_eq!(100.0, sim.game_state.stacks[2]);

        // Post the ante and check the results.
        sim.run_round();
        for i in 0..9 {
            assert_eq!(99.0, sim.game_state.stacks[i]);
        }

        // Post the blinds and check the results.
        sim.run_round();
        assert_eq!(6.0, sim.game_state.player_bet[1]);
        assert_eq!(11.0, sim.game_state.player_bet[2]);
    }

    #[test_log::test]
    fn test_flatdeck_order() {
        let stacks = vec![100.0; 2];
        let game_state = GameState::new(stacks, 10.0, 5.0, 0.0, 0);

        let rng_one = StdRng::seed_from_u64(420);
        let sim_one = RngHoldemSimulationBuilder::default()
            .rng(rng_one)
            .game_state(game_state.clone())
            .build()
            .unwrap();

        let rng_two = StdRng::seed_from_u64(420);
        let sim_two = RngHoldemSimulationBuilder::default()
            .rng(rng_two)
            .game_state(game_state)
            .build()
            .unwrap();

        assert_eq!(sim_two.deck[..], sim_one.deck[..]);
    }

    #[test_log::test]
    fn test_simulation_complex_showdown() {
        let stacks = vec![102.0, 7.0, 12.0, 102.0, 202.0];
        let mut game_state = GameState::new(stacks, 10.0, 5.0, 2.0, 0);
        let mut deck = CardBitSet::default();

        deal_hand_card(0, "Ks", &mut deck, &mut game_state);
        deal_hand_card(0, "Kh", &mut deck, &mut game_state);

        deal_hand_card(1, "As", &mut deck, &mut game_state);
        deal_hand_card(1, "Ac", &mut deck, &mut game_state);

        deal_hand_card(2, "Ad", &mut deck, &mut game_state);
        deal_hand_card(2, "Ah", &mut deck, &mut game_state);

        deal_hand_card(3, "6d", &mut deck, &mut game_state);
        deal_hand_card(3, "4d", &mut deck, &mut game_state);

        deal_hand_card(4, "9d", &mut deck, &mut game_state);
        deal_hand_card(4, "9s", &mut deck, &mut game_state);

        // Start
        game_state.advance_round();

        // Ante
        game_state.do_bet(2.0, true).unwrap(); // ante@idx 1
        game_state.do_bet(2.0, true).unwrap(); // ante@idx 2
        game_state.do_bet(2.0, true).unwrap(); // ante@idx 3
        game_state.do_bet(2.0, true).unwrap(); // ante@idx 4
        game_state.do_bet(2.0, true).unwrap(); // ante@idx 0
        game_state.advance_round();

        // Preflop
        game_state.do_bet(5.0, true).unwrap(); // blinds@idx 1
        game_state.do_bet(10.0, true).unwrap(); // blinds@idx 2
        game_state.fold(); // idx 3
        game_state.do_bet(10.0, false).unwrap(); // idx 4
        game_state.do_bet(10.0, false).unwrap(); // idx 0

        game_state.advance_round();
        assert_eq!(game_state.num_active_players(), 2);

        deal_community_card("6c", &mut deck, &mut game_state);
        deal_community_card("2d", &mut deck, &mut game_state);
        deal_community_card("3d", &mut deck, &mut game_state);
        // Flop
        game_state.do_bet(90.0, false).unwrap(); // idx 4
        game_state.do_bet(90.0, false).unwrap(); // idx 0
        game_state.advance_round();
        assert_eq!(game_state.num_active_players(), 1);

        deal_community_card("8h", &mut deck, &mut game_state);
        // Turn
        game_state.do_bet(0.0, false).unwrap(); // idx 4
        game_state.advance_round();
        assert_eq!(game_state.num_active_players(), 1);

        // River
        deal_community_card("8s", &mut deck, &mut game_state);
        game_state.do_bet(100.0, false).unwrap(); // idx 4
        game_state.advance_round();
        assert_eq!(game_state.num_active_players(), 0);

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .build()
            .unwrap();
        sim.run();

        assert_eq!(Round::Complete, sim.game_state.round);

        assert_eq!(180.0, sim.game_state.player_winnings[0]);
        assert_eq!(15.0, sim.game_state.player_winnings[1]);
        assert_eq!(30.0, sim.game_state.player_winnings[2]);
        assert_eq!(0.0, sim.game_state.player_winnings[3]);
        assert_eq!(100.0, sim.game_state.player_winnings[4]);

        assert_eq!(180.0, sim.game_state.stacks[0]);
        assert_eq!(15.0, sim.game_state.stacks[1]);
        assert_eq!(30.0, sim.game_state.stacks[2]);
        assert_eq!(100.0, sim.game_state.stacks[3]);
        assert_eq!(100.0, sim.game_state.stacks[4]);
    }

    fn deal_hand_card(
        idx: usize,
        card_str: &str,
        deck: &mut CardBitSet,
        game_state: &mut GameState,
    ) {
        let c = Card::try_from(card_str).unwrap();
        assert!(deck.contains(c));
        deck.remove(c);
        game_state.hands[idx].push(c);
    }

    fn deal_community_card(card_str: &str, deck: &mut CardBitSet, game_state: &mut GameState) {
        let c = Card::try_from(card_str).unwrap();
        assert!(deck.contains(c));
        deck.remove(c);
        for h in &mut game_state.hands {
            h.push(c);
        }

        game_state.board.push(c);
    }
}
