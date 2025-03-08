use std::collections::BTreeMap;
use std::fmt;

use tracing::{Level, debug_span, event, instrument, trace_span};
use uuid::Uuid;

use crate::arena::action::{FailedActionPayload, PlayedActionPayload};
use crate::arena::game_state::Round;
use crate::core::{Card, FlatDeck, Rank, Rankable};

use super::action::{
    Action, AgentAction, AwardPayload, DealStartingHandPayload, ForcedBetPayload, GameStartPayload,
    PlayerSitPayload,
};

use super::Agent;
use super::GameState;
use super::historian::Historian;

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
    /// A randomly generated UUID to represent the simulation.
    pub id: Uuid,
    pub agents: Vec<Box<dyn Agent>>,
    pub game_state: GameState,
    pub deck: FlatDeck,
    pub historians: Vec<Box<dyn Historian>>,
    pub panic_on_historian_error: bool,
}

impl HoldemSimulation {
    pub fn more_rounds(&self) -> bool {
        !matches!(self.game_state.round, Round::Complete)
    }
    /// Returns the number of poker agents participating in this simulation.
    pub fn num_agents(&self) -> usize {
        self.agents.len()
    }

    /// Run the simulation all the way to completion. This will mutate the
    /// current state.
    pub fn run(&mut self) {
        let span = debug_span!("run",
            game_state = ?self.game_state,
            deck = ?self.deck);
        let _enter = span.enter();

        while self.more_rounds() {
            self.run_round();
        }
    }

    #[instrument]
    pub fn run_round(&mut self) {
        let span = trace_span!("run_round");
        let _enter = span.enter();

        match self.game_state.round {
            // Dealing the user hand is dealt with as its own round
            // in order to use the per round active bit set
            // for iterating players
            Round::Starting => self.start(),
            Round::Ante => self.ante(),

            Round::DealPreflop => self.deal_preflop(),
            Round::Preflop => self.preflop(),

            Round::DealFlop => self.deal_flop(),
            Round::Flop => self.flop(),

            Round::DealTurn => self.deal_turn(),
            Round::Turn => self.turn(),

            Round::DealRiver => self.deal_river(),
            Round::River => self.river(),

            Round::Showdown => self.showdown(),

            // There's nothing left to do to this.
            Round::Complete => (),
        }
    }

    fn start(&mut self) {
        let span = trace_span!("start");
        let _enter = span.enter();

        // Add an action to record the ante, sb and bb
        // This should allow recreating starting game state
        // together with PlayerSit actions.
        self.record_action(Action::GameStart(GameStartPayload {
            ante: self.game_state.ante,
            small_blind: self.game_state.small_blind,
            big_blind: self.game_state.big_blind,
        }));

        while self.game_state.current_round_num_active_players() > 0 {
            let idx = self.game_state.to_act_idx();

            // Add an action that records starting stack for each player
            // Starting with to the left of the dealer
            // and ending with dealer button.
            self.record_action(Action::PlayerSit(PlayerSitPayload {
                player_stack: self.game_state.stacks[idx],
                idx,
            }));

            // set the active bit on the player to false.
            // This allows us to not deal to players that
            // are sitting out, while also going in the same
            // order of dealing
            self.game_state.round_data.needs_action.disable(idx);

            self.game_state.round_data.advance_action();
        }

        // We're done with the non-betting dealing only round
        self.advance_round();
    }

    fn ante(&mut self) {
        let span = trace_span!("ante");
        let _enter = span.enter();

        let ante = self.game_state.ante;
        if ante > 0.0 {
            // Force the ante from each active player.
            while self.game_state.current_round_num_active_players() > 0 {
                let idx = self.game_state.to_act_idx();

                self.game_state.do_bet(ante, true).unwrap();
                self.record_action(Action::ForcedBet(ForcedBetPayload {
                    bet: ante,
                    idx,
                    player_stack: self.game_state.stacks[idx],
                    forced_bet_type: super::action::ForcedBetType::Ante,
                }));

                self.game_state.round_data.needs_action.disable(idx);
            }
        }
        self.advance_round();
    }

    fn deal_preflop(&mut self) {
        let span = trace_span!("deal_preflop");
        let _enter = span.enter();
        // We deal the cards before advancing the round
        // This allows us to use the round active bitset
        while self.game_state.current_round_num_active_players() > 0 {
            let idx = self.game_state.to_act_idx();

            self.deal_player_cards(2);

            // This allows us to not deal to players that
            // are sitting out, while also going in the same
            // order of dealing
            self.game_state.round_data.needs_action.disable(idx);

            self.game_state.round_data.advance_action();
        }
        self.advance_round()
    }

    fn preflop(&mut self) {
        let span = trace_span!("preflop");
        let _enter = span.enter();

        // Force the small blind and the big blind.
        if !self.game_state.sb_posted {
            let sb = self.game_state.small_blind;
            let sb_idx = self.game_state.to_act_idx();
            self.game_state.do_bet(sb, true).unwrap();
            self.game_state.sb_posted = true;

            self.record_action(Action::ForcedBet(ForcedBetPayload {
                bet: sb,
                idx: sb_idx,
                forced_bet_type: super::action::ForcedBetType::SmallBlind,
                player_stack: self.game_state.stacks[sb_idx],
            }));
        }

        if !self.game_state.bb_posted {
            let bb = self.game_state.big_blind;
            let bb_idx = self.game_state.to_act_idx();
            self.game_state.do_bet(bb, true).unwrap();
            self.game_state.bb_posted = true;
            self.record_action(Action::ForcedBet(ForcedBetPayload {
                bet: bb,
                idx: bb_idx,
                forced_bet_type: super::action::ForcedBetType::BigBlind,
                player_stack: self.game_state.stacks[bb_idx],
            }));
        }

        self.run_betting_round();
        self.advance_round();
    }

    fn deal_flop(&mut self) {
        let span = trace_span!("deal_flop");
        let _enter = span.enter();

        self.deal_comunity_cards(3);
        self.advance_round();
    }

    fn flop(&mut self) {
        let span = trace_span!("flop");
        let _enter = span.enter();

        self.run_betting_round();
        self.advance_round();
    }

    fn deal_turn(&mut self) {
        let span = trace_span!("turn");
        let _enter = span.enter();

        self.deal_comunity_cards(1);
        self.advance_round();
    }

    fn turn(&mut self) {
        let span = trace_span!("turn");
        let _enter = span.enter();

        self.run_betting_round();
        self.advance_round();
    }

    fn deal_river(&mut self) {
        let span = trace_span!("river");
        let _enter = span.enter();

        self.deal_comunity_cards(1);
        self.advance_round();
    }
    fn river(&mut self) {
        let span = trace_span!("river");
        let _enter = span.enter();

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
                            m.sort_by(|a, b| bets[*a].partial_cmp(&bets[*b]).unwrap());
                        })
                        .or_insert_with(|| vec![idx]);

                    map
                },
            );
        // There can be bets that players made but didn't take to showdown they should
        // be added to the main pot. Keep them here and then split them up
        // between the winners of the first rank pot. resetting the ammount to
        // zero.
        let mut folded_pot = bets
            .iter()
            .enumerate()
            .filter(|(idx, _)| !active.get(*idx))
            .map(|(_, bet)| *bet)
            .sum::<f32>();
        bets = bets
            .iter()
            .enumerate()
            .map(|(idx, v)| if active.get(idx) { *v } else { 0.0 })
            .collect();

        // By default the map gives keys in assending order. We want them descending.
        // The actual player vector is sorted in ascending order according to bet size.
        for (rank, players) in ranks.into_iter().rev() {
            let mut start_idx = 0;
            let end_idx = players.len();

            // Set the rank on the current round since we know the ranks
            for idx in players.iter() {
                self.game_state.computed_rank[*idx] = Some(rank);
            }

            // We'll conitune until every player has been given the matching money
            // up to their wager. However since some players might have gone allin
            // earlier we keep removing from the pot and splitting it equally to all
            // those players still left in the pot.
            while start_idx < end_idx {
                // Becasue our lists are ordered from smallest bets to largest
                // we can just assume the first one is the smallest
                //
                // Here we use that property to find the max bet that this pot
                // will give for this round of splitting ties.
                let max_wager = bets[players[start_idx]];
                let mut pot: f64 = folded_pot as f64;
                folded_pot = 0.0;

                // Most common is that ties will
                // be for wagers that are all the same.
                // So check if there's no more
                // bets to award for this player.
                if max_wager <= 0.0 {
                    start_idx += 1;
                    continue;
                }

                // Take all the wagers remaining into a
                // side pot. However this side pot might
                // be the only pot if there were no allins
                for b in bets.iter_mut() {
                    let w = (*b).min(max_wager);
                    *b -= w;
                    pot += w as f64;
                }

                // Now all the winning players get
                // an equal share of the side pot
                let num_players = (end_idx - start_idx) as f64;
                let split = pot / num_players;

                for idx in &players[start_idx..end_idx] {
                    // Record that this player won something
                    event!(parent: &span, Level::INFO, idx, split, pot, ?rank, "pot_awarded");
                    self.game_state.award(*idx, split as f32);
                    self.record_action(Action::Award(AwardPayload {
                        idx: *idx,
                        total_pot: pot as f32,
                        award_amount: split as f32,
                        // Since we had a showdown we cen copy the hand
                        // and the resulting rank.
                        rank: Some(rank),
                        hand: Some(self.game_state.hands[*idx].clone()),
                    }));
                }

                // Since the first player is bet size
                // that we used. They have won everything that they're eligible for.
                start_idx += 1;
            }
        }

        self.end_game();
    }

    fn deal_player_cards(&mut self, num_cards: usize) {
        let new_hand: Vec<Card> = self.deal_cards(num_cards);
        for c in &new_hand {
            self.record_action(Action::DealStartingHand(DealStartingHandPayload {
                card: *c,
                idx: self.game_state.to_act_idx(),
            }));
        }

        let idx = self.game_state.to_act_idx();
        self.game_state.hands[idx].extend(new_hand);
    }

    fn deal_comunity_cards(&mut self, num_cards: usize) {
        let mut community_cards = self.deal_cards(num_cards);
        for c in &community_cards {
            self.record_action(Action::DealCommunity(*c));
        }
        // Add all the cards to the hands as well.
        for h in &mut self.game_state.hands {
            h.extend(community_cards.to_owned());
        }
        // Drain the community_cards vec into the game_state board.
        self.game_state.board.append(&mut community_cards);
    }

    /// Pull num_cards from the deck and return them as a vector.
    fn deal_cards(&mut self, num_cards: usize) -> Vec<Card> {
        let mut cards: Vec<Card> = (0..num_cards).map(|_| self.deck.deal().unwrap()).collect();

        // Keep the cards sorted in min to max order
        // this keeps the number of permutations down since
        // its now AsKsKd is the same as KdAsKs after sorting.
        cards.sort();

        cards
    }

    /// This runs betting for the round to completion. It will run until
    /// everyone has acted or until the round has been completed because no one
    /// can act anymore.
    fn run_betting_round(&mut self) {
        let current_round = self.game_state.round;
        while self.needs_action() && self.game_state.round == current_round {
            self.run_single_agent();
        }
    }

    fn needs_action(&self) -> bool {
        // active but not this player
        let mut other_active = self.game_state.player_active;
        other_active.disable(self.game_state.to_act_idx());

        // Listed as still needing action
        let need_action = self.game_state.round_data.needs_action;

        let result = need_action & other_active;
        !result.empty()
    }

    /// Run the next agent in the game state to act.
    fn run_single_agent(&mut self) {
        let idx = self.game_state.to_act_idx();
        let span = trace_span!("run_agent", idx);
        let _enter = span.enter();
        let action = self.agents[idx].act(&self.id, &self.game_state);

        event!(parent: &span, Level::TRACE, ?action, idx);
        self.run_agent_action(action);
    }

    /// Given the action that an agent wants to take, this function will
    /// determine if the action is valid and then apply it to the game state.
    /// If the action is invalid, the agent will be forced to fold.
    fn run_agent_action(&mut self, agent_action: AgentAction) {
        event!(Level::TRACE, ?agent_action, "run_agent_action");

        let idx = self.game_state.to_act_idx();
        let starting_bet = self.game_state.current_round_bet();
        let starting_player_bet = self.game_state.current_round_player_bet(idx);
        let starting_min_raise = self.game_state.current_round_min_raise();
        let starting_pot = self.game_state.total_pot;

        match agent_action {
            AgentAction::Fold => {
                // It plays hell on verifying games if there are players that quit when there's
                // no money in being asked for. For now do exact equality, but
                // this might be a place we should use approxiate compaison
                // crate.
                if starting_player_bet == starting_bet {
                    event!(Level::WARN, "fold_error");

                    let new_action = AgentAction::Bet(starting_bet);

                    self.game_state.do_bet(starting_bet, false).unwrap();
                    self.record_action(Action::FailedAction(FailedActionPayload {
                        action: agent_action,
                        result: PlayedActionPayload {
                            action: new_action,
                            player_stack: self.game_state.stacks[idx],
                            idx,
                            starting_bet,
                            final_bet: starting_bet,
                            starting_min_raise,
                            final_min_raise: self.game_state.current_round_min_raise(),
                            starting_player_bet,
                            final_player_bet: starting_player_bet,
                            players_active: self.game_state.player_active,
                            players_all_in: self.game_state.player_all_in,
                            starting_pot,
                            final_pot: self.game_state.total_pot,
                        },
                    }));
                } else {
                    self.record_action(Action::PlayedAction(PlayedActionPayload {
                        action: agent_action,
                        player_stack: self.game_state.stacks[idx],
                        idx,

                        // None of the bets move if this is a fold
                        starting_bet,
                        final_bet: starting_bet,
                        starting_min_raise,
                        final_min_raise: self.game_state.current_round_min_raise(),
                        starting_player_bet,
                        final_player_bet: starting_player_bet,
                        players_active: self.game_state.player_active,
                        players_all_in: self.game_state.player_all_in,
                        starting_pot,
                        final_pot: self.game_state.total_pot,
                    }));
                    self.player_fold();
                }
            }
            AgentAction::Bet(bet_amount) => {
                let bet_result = self.game_state.do_bet(bet_amount, false);

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
                        self.record_action(Action::FailedAction(FailedActionPayload {
                            action: agent_action,
                            result: PlayedActionPayload {
                                action: AgentAction::Fold,
                                player_stack: self.game_state.stacks[idx],
                                idx,
                                starting_bet,
                                final_bet: starting_bet,
                                starting_min_raise,
                                final_min_raise: self.game_state.current_round_min_raise(),
                                starting_player_bet,
                                final_player_bet: starting_player_bet,
                                players_active: self.game_state.player_active,
                                players_all_in: self.game_state.player_all_in,
                                // What's the pot worth
                                starting_pot,
                                final_pot: self.game_state.total_pot,
                            },
                        }));

                        // Actually fold the user
                        self.player_fold();
                    }
                    Ok(_added) => {
                        let player_bet = self.game_state.current_round_player_bet(idx);

                        let new_action = AgentAction::Bet(player_bet);
                        // If the game_state.do_bet function returned Ok then
                        // the state is already changed so record the action as played.
                        self.record_action(Action::PlayedAction(PlayedActionPayload {
                            action: new_action,
                            player_stack: self.game_state.stacks[idx],
                            idx,
                            starting_bet,
                            final_bet: self.game_state.current_round_bet(),
                            starting_min_raise,
                            final_min_raise: self.game_state.current_round_min_raise(),
                            starting_player_bet,
                            final_player_bet: player_bet,
                            // Keep track of who's in a
                            players_active: self.game_state.player_active,
                            players_all_in: self.game_state.player_all_in,

                            // What's the pot worth
                            starting_pot,
                            final_pot: self.game_state.total_pot,
                        }));
                    }
                }
            }
        }
    }

    #[instrument]
    fn player_fold(&mut self) {
        self.game_state.fold();
        let left = self.game_state.player_active | self.game_state.player_all_in;

        // If there's only one person left then they win.
        // If there's no one left, and one person went all in they win.
        //
        if left.count() <= 1 {
            if let Some(winning_idx) = left.ones().next() {
                let total_pot = self.game_state.total_pot;
                event!(Level::INFO, winning_idx, total_pot, "folded_to_winner");
                self.game_state.award(winning_idx, total_pot);
                self.record_action(Action::Award(AwardPayload {
                    idx: winning_idx,
                    total_pot,
                    award_amount: total_pot,
                    rank: None,
                    hand: None,
                }))
            }

            self.end_game();
        }
    }

    #[instrument]
    fn end_game(&mut self) {
        let current_round = self.game_state.round;
        self.game_state.complete();
        if current_round != self.game_state.round {
            self.record_action(Action::RoundAdvance(self.game_state.round));
        }
    }

    #[instrument]
    fn advance_round(&mut self) {
        let current_round = self.game_state.round;
        self.game_state.advance_round();
        if self.game_state.round != current_round {
            // Notify historians through record_action
            self.record_action(Action::RoundAdvance(self.game_state.round));
        }
    }

    // Make sure that all modifications to game_state are complete before calling
    // `record_action`. This is critical for making sure replays are deterministic.
    fn record_action(&mut self, action: Action) {
        event!(Level::TRACE, action = ?action, game_state = ?self.game_state, "add_action");
        // Iterate over the historians and record the action
        // If there's an error, log it and remove the historian
        self.historians = self
            .historians
            .drain(..)
            .filter_map(|mut historian| {
                match historian.record_action(&self.id, &self.game_state, action.clone()) {
                    Ok(_) => Some(historian),
                    Err(error) => {
                        event!(Level::ERROR, ?error, "historian_error");

                        // Some user might never error.
                        // For them it's a panic.
                        if self.panic_on_historian_error {
                            panic!(
                                "Historian error {}\naction={:?}\ngame_state = {:?}",
                                error, action, self.game_state
                            );
                        }
                        None
                    }
                }
            })
            .collect();
    }
}

impl fmt::Debug for HoldemSimulation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HoldemSimulation")
            .field("game_state", &self.game_state)
            .field("deck", &self.deck.len())
            .field("historians", &self.historians.len())
            .field("agents", &self.agents.len())
            .field("panic_on_historian_error", &self.panic_on_historian_error)
            .finish()
    }
}
