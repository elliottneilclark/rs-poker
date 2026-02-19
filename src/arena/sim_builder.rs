use rand::Rng;

use crate::core::{CardBitSet, Deck};

use super::{
    Agent, GameState, HoldemSimulation,
    agent::FoldingAgent,
    cfr::{CFRHistorian, StateStore, TraversalSet},
    errors::HoldemSimulationError,
    historian::Historian,
};

// Some builder methods to help with turning a builder struct into a ready
// simulation
fn build_deck(game_state: &GameState) -> Deck {
    let mut d = CardBitSet::default();

    for hand in game_state.hands.iter() {
        let bitset: CardBitSet = (*hand).into();

        d &= !bitset; // remove the cards in the hand from the deck
    }
    for card in game_state.board.iter() {
        d.remove(*card); // remove the cards on the board from the deck
    }

    d.into() // convert the bitset into a deck
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
/// use rs_poker::arena::{GameStateBuilder, HoldemSimulationBuilder};
///
/// let game_state = GameStateBuilder::new()
///     .num_players_with_stack(5, 100.0)
///     .blinds(2.0, 1.0)
///     .dealer_idx(3)
///     .build()
///     .unwrap();
/// let sim = HoldemSimulationBuilder::default()
///     .game_state(game_state)
///     .build()
///     .unwrap();
/// ```
/// However sometimes you want to use a known but random simulation. In that
/// case you can pass in the rng like this:
///
/// ```
/// use rand::{SeedableRng, rngs::StdRng};
/// use rs_poker::arena::{GameStateBuilder, HoldemSimulationBuilder};
///
/// let game_state = GameStateBuilder::new()
///     .num_players_with_stack(5, 100.0)
///     .blinds(2.0, 1.0)
///     .dealer_idx(3)
///     .build()
///     .unwrap();
/// let sim = HoldemSimulationBuilder::default()
///     .game_state(game_state)
///     .build()
///     .unwrap();
/// ```
pub struct HoldemSimulationBuilder {
    agents: Option<Vec<Box<dyn Agent>>>,
    historians: Vec<Box<dyn Historian>>,
    game_state: Option<GameState>,
    deck: Option<Deck>,
    panic_on_historian_error: bool,
    /// Optional CFR context for automatic historian creation.
    cfr_state_store: Option<StateStore>,
    cfr_traversal_set: Option<TraversalSet>,
    cfr_allow_node_mutation: bool,
}

/// # Examples
/// ```
/// use rand::{SeedableRng, rngs::StdRng};
/// use rs_poker::arena::{Agent, agent::FoldingAgent};
/// use rs_poker::arena::{GameStateBuilder, HoldemSimulationBuilder};
///
/// let game_state = GameStateBuilder::new()
///     .num_players_with_stack(5, 100.0)
///     .blinds(2.0, 1.0)
///     .dealer_idx(3)
///     .build()
///     .unwrap();
/// let agents: Vec<Box<dyn Agent>> = (0..5)
///     .map(|_| Box::<FoldingAgent>::default() as Box<dyn Agent>)
///     .collect();
/// let sim = HoldemSimulationBuilder::default()
///     .game_state(game_state)
///     .agents(agents)
///     .build();
/// ```
impl HoldemSimulationBuilder {
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
    pub fn deck(mut self, deck: Deck) -> Self {
        self.deck = Some(deck);
        self
    }

    /// Set the historians for the simulation created by this builder.
    pub fn historians(mut self, historians: Vec<Box<dyn Historian>>) -> Self {
        self.historians = historians;
        self
    }

    /// Should the simulation panic if a historian errors.
    /// Default is false and allows the simulation to continue if a historian
    /// errors. It will be removed from the simulation and recorded in the logs.
    pub fn panic_on_historian_error(mut self, panic_on_historian_error: bool) -> Self {
        self.panic_on_historian_error = panic_on_historian_error;
        self
    }

    /// Provide CFR context for this simulation.
    ///
    /// When set, the builder will automatically create a `CFRHistorian` and
    /// add it to the simulation's historians. This replaces the old pattern
    /// where each CFR agent returned its own historian via `Agent::historian()`.
    ///
    /// # Arguments
    /// * `state_store` - The shared state store containing all players' CFR states
    /// * `traversal_set` - The traversal set tracking each player's position
    /// * `allow_node_mutation` - Whether to allow mutating node types on mismatch
    pub fn cfr_context(
        mut self,
        state_store: StateStore,
        traversal_set: TraversalSet,
        allow_node_mutation: bool,
    ) -> Self {
        self.cfr_state_store = Some(state_store);
        self.cfr_traversal_set = Some(traversal_set);
        self.cfr_allow_node_mutation = allow_node_mutation;
        self
    }

    /// Given the fields already specified build any that are not specified and
    /// create a new HoldemSimulation.
    ///
    /// Uses the OS entropy source for simulation ID generation. For hot paths
    /// where many simulations are created (e.g., CFR sub-simulations), prefer
    /// `build_with_rng` to avoid repeated entropy syscalls.
    ///
    /// @returns HoldemSimulationError if no game_state was given.
    pub fn build(self) -> Result<HoldemSimulation, HoldemSimulationError> {
        let mut rand = rand::rng();
        self.build_with_rng(&mut rand)
    }

    /// Build the simulation using the provided RNG for ID generation.
    ///
    /// This avoids creating a new OS RNG (and its associated syscall) for each
    /// simulation, which is significant when creating millions of sub-simulations
    /// in CFR.
    pub fn build_with_rng<R: Rng>(
        self,
        rng: &mut R,
    ) -> Result<HoldemSimulation, HoldemSimulationError> {
        let game_state = self
            .game_state
            .ok_or(HoldemSimulationError::NeedGameState)?;

        let agents = self
            .agents
            .unwrap_or_else(|| build_agents(game_state.hands.len()));

        let has_cfr_context = self.cfr_state_store.is_some();

        // Skip agent historian collection when CFR context is set,
        // since CFR agents don't provide historians — the CFRHistorian
        // is provided via cfr_context() instead.
        let mut historians: Vec<_> = if has_cfr_context {
            self.historians.into_iter().collect()
        } else {
            let agent_historians = agents.iter().filter_map(|a| a.historian());
            self.historians
                .into_iter()
                .chain(agent_historians)
                .collect()
        };

        // If CFR context was provided, create and add a CFRHistorian.
        if let (Some(state_store), Some(traversal_set)) =
            (self.cfr_state_store, self.cfr_traversal_set)
        {
            let cfr_historian =
                CFRHistorian::new(state_store, traversal_set, self.cfr_allow_node_mutation);
            historians.push(Box::new(cfr_historian));
        }

        let deck = self.deck.unwrap_or_else(|| build_deck(&game_state));

        let id = rng.random::<u128>();

        Ok(HoldemSimulation {
            agents,
            game_state,
            deck,
            id,
            historians,
            panic_on_historian_error: self.panic_on_historian_error,
        })
    }
}

impl Default for HoldemSimulationBuilder {
    fn default() -> Self {
        Self {
            agents: None,
            historians: vec![],
            game_state: None,
            deck: None,
            panic_on_historian_error: true,
            cfr_state_store: None,
            cfr_traversal_set: None,
            cfr_allow_node_mutation: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};

    use crate::{arena::action::AgentAction, arena::game_state::Round, core::Card};

    use super::*;
    use crate::arena::GameStateBuilder;

    /// Test helper to create a game state with standard defaults
    fn test_game_state(
        stacks: Vec<f32>,
        big_blind: f32,
        small_blind: f32,
        ante: f32,
        dealer_idx: usize,
    ) -> GameState {
        GameStateBuilder::new()
            .stacks(stacks)
            .big_blind(big_blind)
            .small_blind(small_blind)
            .ante(ante)
            .dealer_idx(dealer_idx)
            .build()
            .unwrap()
    }

    #[test]
    fn test_single_step_agent() {
        let mut rng = StdRng::seed_from_u64(420);
        let stacks = vec![100.0; 9];
        let game_state = test_game_state(stacks, 10.0, 5.0, 1.0, 0);
        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .build()
            .unwrap();

        assert_eq!(100.0, sim.game_state.stacks[1]);
        assert_eq!(100.0, sim.game_state.stacks[2]);
        // We are starting out.
        sim.run_round(&mut rng);
        assert_eq!(100.0, sim.game_state.stacks[1]);
        assert_eq!(100.0, sim.game_state.stacks[2]);

        // Post the ante and check the results.
        sim.run_round(&mut rng);
        for i in 0..9 {
            assert_eq!(99.0, sim.game_state.stacks[i]);
        }

        // Deal Pre-Flop
        sim.run_round(&mut rng);

        // Post the blinds and check the results.
        sim.run_round(&mut rng);
        assert_eq!(6.0, sim.game_state.player_bet[1]);
        assert_eq!(11.0, sim.game_state.player_bet[2]);
    }

    #[test]
    fn test_simulation_complex_showdown() {
        let stacks = vec![102.0, 7.0, 12.0, 102.0, 202.0];
        let mut game_state = test_game_state(stacks, 10.0, 5.0, 2.0, 0);
        let mut deck = CardBitSet::default();
        let mut rng = rand::rng();

        // Start
        game_state.advance_round();

        // Ante
        game_state.do_bet(2.0, true).unwrap(); // ante@idx 1
        game_state.do_bet(2.0, true).unwrap(); // ante@idx 2
        game_state.do_bet(2.0, true).unwrap(); // ante@idx 3
        game_state.do_bet(2.0, true).unwrap(); // ante@idx 4
        game_state.do_bet(2.0, true).unwrap(); // ante@idx 0
        game_state.advance_round();

        // Deal Preflop
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
        game_state.advance_round();

        // Preflop
        game_state.do_bet(5.0, true).unwrap(); // blinds@idx 1
        game_state.do_bet(10.0, true).unwrap(); // blinds@idx 2
        game_state.fold(); // idx 3
        game_state.do_bet(10.0, false).unwrap(); // idx 4
        game_state.do_bet(10.0, false).unwrap(); // idx 0
        game_state.advance_round();

        // Deal Flop
        deal_community_card("6c", &mut deck, &mut game_state);
        deal_community_card("2d", &mut deck, &mut game_state);
        deal_community_card("3d", &mut deck, &mut game_state);
        game_state.advance_round();

        // Flop
        assert_eq!(game_state.num_active_players(), 2);
        game_state.do_bet(90.0, false).unwrap(); // idx 4
        game_state.do_bet(90.0, false).unwrap(); // idx 0
        game_state.advance_round();
        assert_eq!(game_state.num_active_players(), 1);

        // Deal Turn
        deal_community_card("8h", &mut deck, &mut game_state);
        game_state.advance_round();

        // Turn
        game_state.do_bet(0.0, false).unwrap(); // idx 4
        game_state.advance_round();
        assert_eq!(game_state.num_active_players(), 1);

        // Deal River
        deal_community_card("8s", &mut deck, &mut game_state);
        game_state.advance_round();

        // River
        game_state.do_bet(100.0, false).unwrap(); // idx 4
        game_state.advance_round();
        assert_eq!(game_state.num_active_players(), 0);

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .build()
            .unwrap();
        sim.run(&mut rng);

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
        let card = Card::try_from(card_str).unwrap();
        assert!(deck.contains(card));
        deck.remove(card);
        game_state.hands[idx].insert(card);
    }

    fn deal_community_card(card_str: &str, deck: &mut CardBitSet, game_state: &mut GameState) {
        let card = Card::try_from(card_str).unwrap();
        assert!(deck.contains(card));
        deck.remove(card);
        for h in &mut game_state.hands {
            h.insert(card);
        }

        game_state.board.push(card);
    }

    /// An agent that returns an invalid bet amount to trigger error handling
    #[derive(Clone)]
    struct InvalidBetAgent {
        name: String,
        bet_amount: f32,
    }

    impl crate::arena::Agent for InvalidBetAgent {
        fn act(&mut self, _id: u128, _game_state: &GameState) -> AgentAction {
            AgentAction::Bet(self.bet_amount)
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[test]
    fn test_invalid_bet_triggers_fold() {
        let mut rng = StdRng::seed_from_u64(42);
        let stacks = vec![100.0; 3];
        let game_state = test_game_state(stacks, 10.0, 5.0, 0.0, 0);

        // Create an agent that bets 1.0 - less than the big blind, which is invalid
        let invalid_agent = InvalidBetAgent {
            name: "InvalidBetAgent".to_string(),
            bet_amount: 1.0, // Too small to call the big blind
        };

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .panic_on_historian_error(false)
            .agents(vec![
                Box::new(invalid_agent.clone()),
                Box::new(invalid_agent.clone()),
                Box::new(invalid_agent.clone()),
            ])
            .build()
            .unwrap();

        // Run the simulation - agents with invalid bets should be force-folded
        sim.run(&mut rng);

        // Game should complete
        assert_eq!(Round::Complete, sim.game_state.round);
    }

    #[test]
    fn test_num_agents() {
        let stacks = vec![100.0; 5];
        let game_state = test_game_state(stacks, 10.0, 5.0, 0.0, 0);
        let sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .build()
            .unwrap();

        assert_eq!(5, sim.num_agents());
    }

    #[test]
    fn test_max_raises_default_is_three() {
        let stacks = vec![100.0; 2];
        let game_state = test_game_state(stacks, 10.0, 5.0, 0.0, 0);
        let sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .build()
            .unwrap();

        assert_eq!(Some(3), sim.game_state.max_raises_per_round);
    }

    #[test]
    fn test_max_raises_none_allows_unlimited() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .max_raises_per_round(None)
            .build()
            .unwrap();
        let sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .build()
            .unwrap();

        assert_eq!(None, sim.game_state.max_raises_per_round);
    }

    /// Agent that always raises by min-raise amount
    #[derive(Clone)]
    struct RaisingAgent {
        name: String,
    }

    impl crate::arena::Agent for RaisingAgent {
        fn act(&mut self, _id: u128, game_state: &GameState) -> AgentAction {
            // Always try to raise by min-raise
            let current_bet = game_state.current_round_bet();
            let min_raise = game_state.current_round_min_raise();
            AgentAction::Bet(current_bet + min_raise)
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[test]
    fn test_max_raises_converts_raise_to_call() {
        use crate::arena::action::Action;
        use crate::arena::historian::VecHistorian;

        let mut rng = StdRng::seed_from_u64(42);
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 1000.0)
            .blinds(10.0, 5.0)
            .max_raises_per_round(Some(2)) // Only 2 raises allowed
            .build()
            .unwrap();

        let hist = Box::new(VecHistorian::default());
        let records = hist.get_storage();

        let raiser = RaisingAgent {
            name: "Raiser".to_string(),
        };

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(vec![Box::new(raiser.clone()), Box::new(raiser.clone())])
            .historians(vec![hist])
            .build()
            .unwrap();

        // Run just the preflop betting round
        // Starting -> Ante -> DealPreflop -> Preflop
        sim.run_round(&mut rng); // Starting
        sim.run_round(&mut rng); // Ante
        sim.run_round(&mut rng); // DealPreflop
        sim.run_round(&mut rng); // Preflop (runs betting)

        // Count failed actions (raises converted to calls)
        let failed_actions: Vec<_> = records
            .borrow()
            .iter()
            .filter(|r| matches!(r.action, Action::FailedAction(_)))
            .cloned()
            .collect();

        // With max_raises=2, after 2 raises, subsequent raise attempts should fail
        assert!(
            !failed_actions.is_empty(),
            "Expected some raises to be converted to calls"
        );
    }

    #[test]
    fn test_max_raises_all_in_always_allowed() {
        use crate::arena::action::Action;
        use crate::arena::historian::VecHistorian;

        let mut game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .max_raises_per_round(Some(3)) // Max 3 raises
            .build()
            .unwrap();

        // Advance to preflop and post blinds
        game_state.advance_round(); // Starting
        game_state.advance_round(); // Ante
        game_state.advance_round(); // DealPreflop
        game_state.advance_round(); // Preflop
        game_state.do_bet(5.0, true).unwrap(); // SB posts
        game_state.do_bet(10.0, true).unwrap(); // BB posts

        // Manually set raise count to max to simulate raises already occurred
        game_state.round_data.total_raise_count = 3;

        // Current state: SB to act, has bet 5, needs to match 10 (BB)
        // SB stack is 95.0, BB stack is 90.0

        let hist = Box::new(VecHistorian::default());
        let records = hist.get_storage();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(vec![
                Box::<crate::arena::agent::FoldingAgent>::default(),
                Box::<crate::arena::agent::FoldingAgent>::default(),
            ])
            .historians(vec![hist])
            .build()
            .unwrap();

        // SB is to act (idx 0 after blinds in 2-player)
        // Current bet is 10.0, player bet is 5.0, stack is 95.0
        // All-in bet would be 5.0 (current) + 95.0 (stack) = 100.0
        let all_in_bet = sim.game_state.current_round_current_player_bet()
            + sim.game_state.current_player_stack();
        assert!(
            all_in_bet > sim.game_state.current_round_bet(),
            "All-in should be a raise"
        );

        // Manually call run_agent_action with an all-in bet
        sim.run_agent_action(AgentAction::Bet(all_in_bet));

        // The all-in should be recorded as PlayedAction, not FailedAction
        let played: Vec<_> = records
            .borrow()
            .iter()
            .filter(|r| matches!(r.action, Action::PlayedAction(_)))
            .cloned()
            .collect();

        let failed: Vec<_> = records
            .borrow()
            .iter()
            .filter(|r| matches!(r.action, Action::FailedAction(_)))
            .cloned()
            .collect();

        assert_eq!(played.len(), 1, "All-in should be recorded as PlayedAction");
        assert!(
            failed.is_empty(),
            "All-in should NOT be recorded as FailedAction"
        );
    }

    #[test]
    fn test_max_raises_resets_each_round() {
        let mut rng = StdRng::seed_from_u64(42);
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 500.0)
            .blinds(10.0, 5.0)
            .max_raises_per_round(Some(2))
            .build()
            .unwrap();

        let raiser = RaisingAgent {
            name: "Raiser".to_string(),
        };

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(vec![Box::new(raiser.clone()), Box::new(raiser.clone())])
            .build()
            .unwrap();

        // Run through preflop
        sim.run_round(&mut rng); // Starting
        sim.run_round(&mut rng); // Ante
        sim.run_round(&mut rng); // DealPreflop
        sim.run_round(&mut rng); // Preflop betting

        // After preflop, advance to flop
        sim.run_round(&mut rng); // DealFlop

        // The raise count should be 0 at the start of flop
        assert_eq!(
            sim.game_state.round_data.total_raise_count, 0,
            "Raise count should reset at the start of each betting round"
        );
    }

    #[test]
    fn test_max_raises_records_failed_action() {
        use crate::arena::action::Action;
        use crate::arena::historian::VecHistorian;

        let mut game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 1000.0)
            .blinds(10.0, 5.0)
            .max_raises_per_round(Some(2))
            .build()
            .unwrap();

        // Advance to preflop and post blinds
        game_state.advance_round(); // Starting
        game_state.advance_round(); // Ante
        game_state.advance_round(); // DealPreflop
        game_state.advance_round(); // Preflop
        game_state.do_bet(5.0, true).unwrap(); // SB
        game_state.do_bet(10.0, true).unwrap(); // BB

        // Set raise count to max
        game_state.round_data.total_raise_count = 2;

        let hist = Box::new(VecHistorian::default());
        let records = hist.get_storage();

        let raiser = RaisingAgent {
            name: "Raiser".to_string(),
        };

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(vec![Box::new(raiser.clone()), Box::new(raiser.clone())])
            .historians(vec![hist])
            .build()
            .unwrap();

        // Directly call run_agent_action with a raise attempt
        // Current bet is 10 (BB), min raise is 10, so a raise to 20 should be capped
        sim.run_agent_action(AgentAction::Bet(20.0));

        // The raise should be recorded as a FailedAction
        let failed_actions: Vec<_> = records
            .borrow()
            .iter()
            .filter_map(|r| {
                if let Action::FailedAction(payload) = &r.action {
                    Some(payload.clone())
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(
            failed_actions.len(),
            1,
            "Should have exactly one failed action"
        );

        // Verify the original action was a Bet (raise attempt)
        assert!(
            matches!(failed_actions[0].action, AgentAction::Bet(_)),
            "Original action should be a Bet"
        );

        // Verify the result action is a Bet at the call amount (not a raise)
        if let AgentAction::Bet(amount) = failed_actions[0].result.action {
            assert_eq!(
                amount, failed_actions[0].result.starting_bet,
                "Result should be a call at the current bet level"
            );
        } else {
            panic!("Result action should be a Bet");
        }
    }
}
