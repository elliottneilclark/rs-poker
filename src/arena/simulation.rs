use core::fmt;
use std::collections::BTreeMap;

use crate::arena::game_state::Round;
use crate::core::{Card, Deck, FlatDeck, Rank, Rankable};

use super::action::AgentAction;
use super::agent::Agent;

use super::game_state::GameState;

pub struct HoldemSimulation {
    agents: Vec<Box<dyn Agent>>,
    pub game_state: GameState,
    pub deck: FlatDeck,
}

impl HoldemSimulation {
    pub fn new(game_state: GameState) -> Self {
        HoldemSimulation::new_with_agents(game_state, vec![])
    }

    pub fn new_with_agents(game_state: GameState, agents: Vec<Box<dyn Agent>>) -> Self {
        let mut d = Deck::default();

        for hand in game_state.hands.iter() {
            for card in hand.iter() {
                d.remove(card);
            }
        }
        let mut flat_deck: FlatDeck = d.into();
        flat_deck.shuffle();

        Self::new_with_agents_and_deck(game_state, flat_deck, agents)
    }

    pub fn new_with_agents_and_deck(
        game_state: GameState,
        deck: FlatDeck,
        agents: Vec<Box<dyn Agent>>,
    ) -> Self {
        Self {
            game_state,
            agents,
            deck,
        }
    }

    pub fn more_rounds(&self) -> bool {
        !matches!(self.game_state.round, Round::Complete)
    }

    pub fn run(&mut self) {
        while self.more_rounds() {
            self.step();
        }
    }

    pub fn step(&mut self) {
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
        while self.game_state.num_active_players_in_round() > 0 {
            let c1 = self.deck.deal().unwrap();
            let c2 = self.deck.deal().unwrap();

            let idx = self.game_state.current_round_data().to_act_idx;
            self.game_state.hands[idx].push(c1);
            self.game_state.hands[idx].push(c2);

            // set the active bit on the player to false.
            // This allows us to not deal to players that
            // are sitting out, while also going in the same
            // order of dealing
            self.game_state
                .mut_current_round_data()
                .player_active
                .set(idx, false);

            self.game_state.mut_current_round_data().advance();
        }
        // We're done with the non-betting dealing only round
        self.game_state.advance_round();
    }

    fn preflop(&mut self) {
        dbg!(&self.game_state);
        self.run_betting_round();
        self.game_state.advance_round();
    }

    fn flop(&mut self) {
        self.deal_comunity(3);
        self.run_betting_round();
        self.game_state.advance_round();
    }

    fn turn(&mut self) {
        self.deal_comunity(1);
        self.run_betting_round();
        self.game_state.advance_round();
    }

    fn river(&mut self) {
        self.deal_comunity(1);
        self.run_betting_round();
        self.game_state.advance_round();
    }

    fn showdown(&mut self) {
        // Rank each player that still has a chance.
        let mut active = self.game_state.player_active.clone();
        active.union_with(&self.game_state.player_all_in);

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
        // Add all the cards to the hands as well.
        for h in &mut self.game_state.hands {
            for c in &community_cards {
                // push a copy
                h.push(*c);
            }
        }
        // Drain the community_cards vec into the game_state board.
        self.game_state.board.append(&mut community_cards);
    }

    fn run_betting_round(&mut self) {
        // There's no need to run any betting round if there's no on left in the round.
        if self.game_state.num_active_players_in_round() > 1 {
            // Howevwer if there is more than
            // one, we need to run the betting until everyone has acted.
            while self.game_state.num_active_players_in_round() > 0 {
                let idx = self.game_state.current_round_data().to_act_idx;
                let action = self.agents[idx].act(&self.game_state);
                self.run_agent_action(action)
            }
        }
    }

    fn run_agent_action(&mut self, action: AgentAction) {
        match action {
            AgentAction::Bet(bet_ammount) => {
                if self.game_state.do_bet(bet_ammount, false).is_err() {
                    // If the agent failed to give us a good bet then they lose this round.
                    self.player_fold()
                }
            }
            AgentAction::Fold => self.player_fold(),
        }
    }

    fn player_fold(&mut self) {
        self.game_state.fold();

        // If there's only one person left then they win.
        // If there's no one left, and one person went all in they win.
        //
        if self.game_state.num_active_players() == 1 && self.game_state.num_all_in_players() == 0
            || self.game_state.num_active_players() == 0
                && self.game_state.num_all_in_players() == 1
        {
            let to_award: Vec<usize> = self
                .game_state
                .player_active
                .union(&self.game_state.player_all_in)
                .collect();

            if let Some(winning_idx) = to_award.first() {
                self.game_state
                    .award(*winning_idx, self.game_state.total_pot);
            }

            self.game_state.complete()
        }
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

#[cfg(test)]
mod tests {
    use std::convert::TryFrom;

    use crate::arena::agent::{CallingAgent, FoldingAgent};

    use super::*;

    #[test]
    fn test_single_step_agent() {
        let stacks = vec![100; 9];
        let game_state = GameState::new(stacks, 10, 5, 0);
        let mut sim = HoldemSimulation::new(game_state);

        assert_eq!(100, sim.game_state.stacks[1]);
        // We are starting out.
        sim.step();
        // assert that blinds are there
        assert_eq!(95, sim.game_state.stacks[1]);
        assert_eq!(90, sim.game_state.stacks[2]);
    }

    #[test]
    fn test_call_agents() {
        let stacks = vec![100; 4];
        let game_state = GameState::new(stacks, 10, 5, 0);
        let mut sim = HoldemSimulation::new_with_agents(
            game_state,
            vec![
                Box::new(CallingAgent {}),
                Box::new(CallingAgent {}),
                Box::new(CallingAgent {}),
                Box::new(CallingAgent {}),
            ],
        );

        sim.run();

        assert_eq!(sim.game_state.num_active_players(), 4);

        assert_ne!(0, sim.game_state.player_winnings.iter().sum());
        assert_eq!(40, sim.game_state.player_winnings.iter().sum());
    }

    #[test]
    fn test_folding_agents() {
        let stacks = vec![100; 2];
        let game_state = GameState::new(stacks, 10, 5, 0);
        let mut sim = HoldemSimulation::new_with_agents(
            game_state,
            vec![Box::new(FoldingAgent {}), Box::new(FoldingAgent {})],
        );

        sim.run();

        assert_eq!(sim.game_state.num_active_players(), 1);
        assert_eq!(sim.game_state.round, Round::Complete);

        assert_eq!(15, sim.game_state.player_winnings.iter().sum());
        assert_eq!(15, sim.game_state.player_winnings[0]);
    }

    #[test]
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

        let mut sim = HoldemSimulation::new(game_state);
        sim.step();

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
