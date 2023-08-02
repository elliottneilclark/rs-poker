use core::fmt;
use std::collections::BTreeMap;

use rand::rngs::ThreadRng;
use rand::{thread_rng, Rng};
use tracing::{debug_span, event, trace_span, Level};

use crate::arena::action::{FailedActionPayload, PlayedActionPayload};
use crate::arena::game_state::Round;
use crate::core::{Card, Deck, FlatDeck, Rank, Rankable};

use super::action::{
    Action, AgentAction, DealStartingHandPayload, ForcedBetPayload, GameStartPayload,
    PlayerSitPayload,
};
use super::agent::FoldingAgent;
use super::errors::HoldemSimulationError;
use super::Agent;
use super::GameState;

/// # Description
///
/// This code is implementing a version of Texas Hold'em poker. It is a
/// simulation of the game that can be played with computer agents. The game
/// progresses through a number of rounds: Starting,
/// Preflop,
/// Flop,
/// Turn,
/// River, and
/// Showdown.
///
/// The simulation creates a deck of cards, shuffles it, and deals cards to the
/// players. The players then take turns making bets, raising or folding until
/// the round is complete. Then, the game moves to the next round, and the
/// process repeats. At the end of the game, the player with the best hand wins.
///
/// The simulation is designed to be used with agents that can make decisions
/// based on the game state. The `HoldemSimulation` struct keeps track of the
/// game state, the deck, and the actions taken in the game.
///
/// The `run` method can be used to run the entire game
///
/// # Behavior
///
/// - Any agent bet that is an over bet will silently turn into an all in. That
///   is to say if an agent has 100 in their stack and bet `100_000_000` that
///   will be accepted and will be equivilant to bet `100`
/// - Any bet that `GameState` rules as being impossible, those that turn into
///   [`rs-poker::arena::errors::GameStateError`] will instead be turned into
///   fold.
/// - It's expected that you have the same number of agents as you have chip
///   stacks in the game state. If players are not active, you can use the
///   `FoldingAgent` as a stand in and set the active bit to false.
pub struct HoldemSimulation {
    agents: Vec<Box<dyn Agent>>,
    pub game_state: GameState,
    pub deck: FlatDeck,
    pub actions: Vec<Action>,
}

impl HoldemSimulation {
    pub fn more_rounds(&self) -> bool {
        !matches!(self.game_state.round, Round::Complete)
    }

    /// Run the simulation all the way to completion. This will mutate the
    /// current state.
    pub fn run(&mut self) {
        let span = debug_span!("run",
            game_state = ?self.game_state,
            deck = ?self.deck);
        let _enter = span.enter();

        while self.more_rounds() {
            self.step();
        }
    }

    pub fn step(&mut self) {
        let span = trace_span!("step");
        let _enter = span.enter();

        match self.game_state.round {
            // Dealing the user hand is dealt with as its own round
            // in order to use the per round active bit set
            // for iterating players
            Round::Starting => self.start(),
            Round::Preflop => self.preflop(),
            Round::Flop => self.flop(),
            Round::Turn => self.turn(),
            Round::River => self.river(),
            Round::Showdown => self.showdown(),

            // There's nothing left to do to this.
            Round::Complete => (),
        }
    }

    fn start(&mut self) {
        let span = trace_span!("start");
        let _enter = span.enter();

        // Add an action to record the sb and bb
        // This should allow recreating starting game state
        // together with PlayerSit actions.
        self.add_action(Action::GameStart(GameStartPayload {
            small_blind: self.game_state.small_blind,
            big_blind: self.game_state.big_blind,
        }));

        // We deal the cards before advancing the round
        // This allows us to use the round active bitset
        while self.game_state.num_active_players_in_round() > 0 {
            let c1 = self.deck.deal().unwrap();
            let c2 = self.deck.deal().unwrap();

            // Keep an order of cards to keep the number of permutations down.
            let first_card = c1.min(c2);
            let second_card = c1.max(c2);

            let idx = self.game_state.current_round_data().to_act_idx;

            // Add an action that records starting stack for each player
            // Starting with to the left of the dealer
            // and ending with dealer button.
            self.add_action(Action::PlayerSit(PlayerSitPayload {
                player_stack: self.game_state.stacks[idx],
                idx,
            }));

            // Put the cards in the hand
            self.game_state.hands[idx].push(first_card);
            self.game_state.hands[idx].push(second_card);

            // set the active bit on the player to false.
            // This allows us to not deal to players that
            // are sitting out, while also going in the same
            // order of dealing
            self.game_state
                .mut_current_round_data()
                .player_active
                .disable(idx);

            self.add_action(Action::DealStartingHand(DealStartingHandPayload {
                card_one: first_card,
                card_two: second_card,
                idx,
            }));

            self.game_state.mut_current_round_data().advance();
        }

        // We're done with the non-betting dealing only round
        self.advance_round();
    }

    fn preflop(&mut self) {
        let span = trace_span!("preflop");
        let _enter = span.enter();

        // We have two different bets to force.
        let sb = self.game_state.small_blind;
        let sb_idx = self.game_state.current_round_data().to_act_idx;
        self.game_state.do_bet(sb, true).unwrap();
        self.add_action(Action::ForcedBet(ForcedBetPayload {
            bet: sb,
            idx: sb_idx,
            player_stack: self.game_state.stacks[sb_idx],
        }));

        let bb = self.game_state.big_blind;
        let bb_idx = self.game_state.current_round_data().to_act_idx;
        self.game_state.do_bet(bb, true).unwrap();
        self.add_action(Action::ForcedBet(ForcedBetPayload {
            bet: bb,
            idx: bb_idx,
            player_stack: self.game_state.stacks[bb_idx],
        }));

        self.run_betting_round();
        self.advance_round();
    }

    fn flop(&mut self) {
        let span = trace_span!("flop");
        let _enter = span.enter();

        self.deal_comunity(3);
        self.run_betting_round();
        self.advance_round();
    }

    fn turn(&mut self) {
        let span = trace_span!("turn");
        let _enter = span.enter();

        self.deal_comunity(1);
        self.run_betting_round();
        self.advance_round();
    }

    fn river(&mut self) {
        let span = trace_span!("river");
        let _enter = span.enter();

        self.deal_comunity(1);
        self.run_betting_round();
        self.advance_round();
    }

    fn showdown(&mut self) {
        let span = trace_span!("showdown");
        let _enter = span.enter();

        // Rank each player that still has a chance.
        let active = self.game_state.player_active | self.game_state.player_all_in;

        let mut bets = self.game_state.player_bet.clone();

        // Create a map where the keys are the ranks of hands and
        // the values are vectors of player index, for players that had that hand
        let ranks = active
            .ones()
            .map(|idx| (idx, self.game_state.hands[idx].rank()))
            .fold(
                BTreeMap::new(),
                |mut map: BTreeMap<Rank, Vec<usize>>, (idx, rank)| {
                    map.entry(rank)
                        .and_modify(|m| {
                            m.push(idx);
                            m.sort_by(|a, b| bets[*a].cmp(&bets[*b]));
                        })
                        .or_insert_with(|| vec![idx]);

                    map
                },
            );

        // By default the map gives keys in assending order. We want them descending.
        // The actual player vector is sorted in ascending order according to bet size.
        for (_rank, players) in ranks.into_iter().rev() {
            let mut start_idx = 0;
            let end_idx = players.len();

            while start_idx < end_idx {
                // Becasue our lists are ordered from smallest bets to largest
                // we can just assume the first one is the smallest
                let max_wager = bets[players[start_idx]];
                let mut pot = 0;

                // Most common is that ties will
                // be for wagers that are all the same.
                // So check if there's no more
                // bets to award for this player.
                if max_wager == 0 {
                    start_idx += 1;
                    continue;
                }

                // Take all the wagers into a singular pot.
                for b in bets.iter_mut() {
                    let w = (*b).min(max_wager);
                    *b -= w;
                    pot += w;
                }

                // Now all the winning players get
                // an equal share of the total pot
                let num_players = (end_idx - start_idx) as i32;
                let split = pot / num_players;
                for idx in &players[start_idx..end_idx] {
                    self.game_state.award(*idx, split);
                }

                // Since the first player is bet size
                // that we used. They have won everything that they're eligible for.
                start_idx += 1;
            }
        }

        self.game_state.complete();
    }

    fn deal_comunity(&mut self, num_cards: usize) {
        let mut community_cards: Vec<Card> =
            (0..num_cards).map(|_| self.deck.deal().unwrap()).collect();

        // Keep the community cards sorted in min to max order
        // this keeps the number of permutations down since
        // its now AsKsKd is the same as KdAsKs after sorting.
        community_cards.sort();

        for c in &community_cards {
            self.add_action(Action::DealCommunity(*c));
        }
        // Add all the cards to the hands as well.
        for h in &mut self.game_state.hands {
            h.extend(community_cards.to_owned());
        }
        // Drain the community_cards vec into the game_state board.
        self.game_state.board.append(&mut community_cards);
    }

    fn run_betting_round(&mut self) {
        // There's no need to run any betting round if there's no on left in the round.
        if self.game_state.num_active_players_in_round() > 1 {
            let current_round = self.game_state.round;
            // However if there is more than one player,
            // we need to run the betting until everyone has acted.
            // or until the round has been completed
            // because no one can act anymore
            while self.game_state.num_active_players_in_round() > 0
                && self.game_state.num_active_players() > 1
                && current_round == self.game_state.round
            {
                let idx = self.game_state.current_round_data().to_act_idx;
                let action = self.agents[idx].act(&self.game_state);
                self.run_agent_action(action)
            }
        }
    }

    fn run_agent_action(&mut self, agent_action: AgentAction) {
        event!(Level::TRACE, ?agent_action, "run_agent_action");

        let idx = self.game_state.current_round_data().to_act_idx;
        match agent_action {
            AgentAction::Fold => {
                let player_bet = self.game_state.current_round_data().player_bet[idx];
                let current_bet = self.game_state.current_round_data().bet;
                if player_bet == current_bet {
                    event!(Level::WARN, "fold_error");

                    let new_action = AgentAction::Bet(current_bet);

                    self.game_state.do_bet(current_bet, false).unwrap();
                    self.add_action(Action::FailedAction(FailedActionPayload {
                        action: agent_action,
                        result_action: new_action,
                        player_stack: self.game_state.stacks[idx],
                        idx,
                    }));
                } else {
                    self.add_action(Action::PlayedAction(PlayedActionPayload {
                        action: agent_action,
                        player_stack: self.game_state.stacks[idx],
                        idx,
                    }));
                    self.player_fold();
                }
            }
            AgentAction::Bet(bet_ammount) => {
                let bet_result = self.game_state.do_bet(bet_ammount, false);
                match bet_result {
                    Err(error) => {
                        // If the agent failed to give us a good bet then they lose this round.
                        //
                        // We emit the error as an event
                        // Assume that game_state.do_bet() will have changed nothing since it
                        // errored out Add an action that shows the user was
                        // force folded. Actually fold the user
                        event!(Level::WARN, ?error, "bet_error");

                        // Record this errant action
                        self.add_action(Action::FailedAction(FailedActionPayload {
                            action: agent_action,
                            result_action: AgentAction::Fold,
                            player_stack: self.game_state.stacks[idx],
                            idx,
                        }));

                        // Actually fold the user
                        self.player_fold();
                    }
                    Ok(_added) => {
                        let new_action =
                            AgentAction::Bet(self.game_state.current_round_data().player_bet[idx]);
                        // If the game_state.do_bet function returned Ok then
                        // the state is already changed so record the action as played.
                        self.add_action(Action::PlayedAction(PlayedActionPayload {
                            action: new_action,
                            player_stack: self.game_state.stacks[idx],
                            idx,
                        }));
                    }
                }
            }
        }
    }

    fn player_fold(&mut self) {
        self.game_state.fold();
        let left = self.game_state.player_active | self.game_state.player_all_in;

        // If there's only one person left then they win.
        // If there's no one left, and one person went all in they win.
        //
        if left.count() <= 1 {
            if let Some(winning_idx) = left.ones().next() {
                self.game_state
                    .award(winning_idx, self.game_state.total_pot);
            }

            self.game_state.complete()
        }
    }

    fn advance_round(&mut self) {
        self.game_state.advance_round();
        self.add_action(Action::RoundAdvance(self.game_state.round));
    }

    fn add_action(&mut self, action: Action) {
        event!(Level::TRACE, action = ?action, game_state = ?self.game_state, "add_action");
        self.actions.push(action);
    }
}

impl fmt::Debug for HoldemSimulation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HoldemSimulation")
            .field("game_state", &self.game_state)
            .field("deck", &self.deck)
            .finish()
    }
}

// Some builder methods to help with turning a builder struct into a ready
// simulation
fn build_flat_deck<R: Rng>(game_state: &GameState, rng: &mut R) -> FlatDeck {
    let mut d = Deck::default();

    for hand in game_state.hands.iter() {
        for card in hand.iter() {
            d.remove(card);
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

/// # Builder
///
/// `RngHoldemSimulationBuilder` is a builder to allow for complex
/// configurations of a holdem simulation played via agents. A game state is
/// required, other fields are optional.
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
/// let game_state = GameState::new(vec![100; 5], 2, 1, 3);
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
/// let game_state = GameState::new(vec![100; 5], 2, 1, 3);
/// let rng = StdRng::seed_from_u64(420);
/// let sim = RngHoldemSimulationBuilder::default()
///     .game_state(game_state)
///     .rng(rng)
///     .build()
///     .unwrap();
/// ```
pub struct RngHoldemSimulationBuilder<R: Rng> {
    agents: Option<Vec<Box<dyn Agent>>>,
    game_state: Option<GameState>,
    deck: Option<FlatDeck>,
    rng: Option<R>,
}

impl<R: Rng> RngHoldemSimulationBuilder<R> {
    pub fn agents(mut self, agents: Vec<Box<dyn Agent>>) -> Self {
        self.agents = Some(agents);
        self
    }

    pub fn game_state(mut self, game_state: GameState) -> Self {
        self.game_state = Some(game_state);
        self
    }

    pub fn deck(mut self, deck: FlatDeck) -> Self {
        self.deck = Some(deck);
        self
    }

    pub fn rng(mut self, rng: R) -> Self {
        self.rng = Some(rng);
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
        Ok(HoldemSimulation {
            agents,
            game_state,
            deck,
            actions: vec![],
        })
    }
}

impl<R: Rng> Default for RngHoldemSimulationBuilder<R> {
    fn default() -> Self {
        Self {
            agents: None,
            game_state: None,
            deck: None,
            rng: None,
        }
    }
}

pub type HoldemSimulationBuilder = RngHoldemSimulationBuilder<ThreadRng>;

#[cfg(test)]
mod tests {
    use std::convert::TryFrom;

    use rand::{rngs::StdRng, SeedableRng};

    use super::*;

    #[test_log::test]
    fn test_single_step_agent() {
        let stacks = vec![100; 9];
        let game_state = GameState::new(stacks, 10, 5, 0);
        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .build()
            .unwrap();

        assert_eq!(100, sim.game_state.stacks[1]);
        assert_eq!(100, sim.game_state.stacks[2]);
        // We are starting out.
        sim.step();
        assert_eq!(100, sim.game_state.stacks[1]);
        assert_eq!(100, sim.game_state.stacks[2]);

        sim.step();
        // assert that blinds are there
        assert_eq!(5, sim.game_state.player_bet[1]);
        assert_eq!(10, sim.game_state.player_bet[2]);
    }

    #[test_log::test]
    fn test_flatdeck_order() {
        let stacks = vec![100; 2];
        let game_state = GameState::new(stacks, 10, 5, 0);

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
        let stacks = vec![100, 5, 10, 100, 200];
        let mut game_state = GameState::new(stacks, 10, 5, 0);
        let mut deck = Deck::default();

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
        // Preflop
        game_state.do_bet(5, true).unwrap(); // blinds@idx 1
        game_state.do_bet(10, true).unwrap(); // blinds@idx 2
        game_state.fold(); // idx 3
        game_state.do_bet(10, false).unwrap(); // idx 4
        game_state.do_bet(10, false).unwrap(); // idx 0

        game_state.advance_round();
        assert_eq!(game_state.num_active_players(), 2);

        deal_community_card("6c", &mut deck, &mut game_state);
        deal_community_card("2d", &mut deck, &mut game_state);
        deal_community_card("3d", &mut deck, &mut game_state);
        // Flop
        game_state.do_bet(90, false).unwrap(); // idx 4
        game_state.do_bet(90, false).unwrap(); // idx 0
        game_state.advance_round();
        assert_eq!(game_state.num_active_players(), 1);

        deal_community_card("8h", &mut deck, &mut game_state);
        // Turn
        game_state.do_bet(0, false).unwrap(); // idx 4
        game_state.advance_round();
        assert_eq!(game_state.num_active_players(), 1);

        // River
        deal_community_card("8s", &mut deck, &mut game_state);
        game_state.do_bet(100, false).unwrap(); // idx 4
        game_state.advance_round();
        assert_eq!(game_state.num_active_players(), 0);

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .build()
            .unwrap();
        sim.run();

        assert_eq!(Round::Complete, sim.game_state.round);

        assert_eq!(180, sim.game_state.player_winnings[0]);
        assert_eq!(10, sim.game_state.player_winnings[1]);
        assert_eq!(25, sim.game_state.player_winnings[2]);
        assert_eq!(0, sim.game_state.player_winnings[3]);
        assert_eq!(100, sim.game_state.player_winnings[4]);

        assert_eq!(180, sim.game_state.stacks[0]);
        assert_eq!(10, sim.game_state.stacks[1]);
        assert_eq!(25, sim.game_state.stacks[2]);
        assert_eq!(100, sim.game_state.stacks[3]);
        assert_eq!(100, sim.game_state.stacks[4]);
    }

    fn deal_hand_card(idx: usize, card_str: &str, deck: &mut Deck, game_state: &mut GameState) {
        let c = Card::try_from(card_str).unwrap();
        assert!(deck.remove(&c));
        game_state.hands[idx].push(c);
    }

    fn deal_community_card(card_str: &str, deck: &mut Deck, game_state: &mut GameState) {
        let c = Card::try_from(card_str).unwrap();
        assert!(deck.remove(&c));
        for h in &mut game_state.hands {
            h.push(c);
        }

        game_state.board.push(c);
    }
}
