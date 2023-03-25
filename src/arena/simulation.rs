use core::fmt;
use std::collections::BTreeMap;

use crate::arena::game_state::Round;
use crate::core::{Card, Deck, FlatDeck, Rank, Rankable};

use super::action::Action;
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
        Self {
            game_state,
            agents,
            deck: flat_deck,
        }
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

    pub fn step(&mut self) {
        match self.game_state.round {
            Round::Starting => self.start(),
            Round::Preflop => self.preflop(),
            Round::Flop => self.flop(),
            Round::Turn => self.turn(),
            Round::River => self.river(),
            Round::Showdown => self.showdown(),
            Round::Complete => (),
        }
    }

    fn start(&mut self) {
        for h in &mut self.game_state.hands {
            h.push(self.deck.deal().unwrap());
            h.push(self.deck.deal().unwrap());
        }
        self.game_state.advance_round().unwrap();
    }

    fn preflop(&mut self) {
        self.run_betting_round();
        self.game_state.advance_round().unwrap()
    }

    fn flop(&mut self) {
        self.deal_comunity(3);
        self.run_betting_round();
        self.game_state.advance_round().unwrap()
    }

    fn turn(&mut self) {
        self.deal_comunity(1);
        self.run_betting_round();
        self.game_state.advance_round().unwrap()
    }

    fn river(&mut self) {
        self.deal_comunity(1);
        self.run_betting_round();
        self.game_state.advance_round().unwrap()
    }

    fn showdown(&mut self) {
        // Rank each player that still has a chance.
        let mut active = self.game_state.player_active.clone();
        active.union_with(&self.game_state.player_all_in);

        let mut bets = self.game_state.player_bet.clone();

        // Create a map where the keys are the ranks and the values are vectors of player index.
        let ranks = active
            .ones()
            .into_iter()
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

        self.game_state.advance_round().unwrap()
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
        while self.game_state.num_active_players_in_round() > 0 {
            let round = self.game_state.current_round_data().unwrap();
            let idx = round.to_act_idx;
            let action = self.agents[idx].act(&self.game_state);
            self.run_action(action)
        }
    }

    fn run_action(&mut self, action: Action) {
        match action {
            Action::Bet(bet_ammount) => {
                let result = self.game_state.do_bet(bet_ammount, false);
                if result.is_err() {
                    self.game_state.fold().unwrap();
                }
            }
            Action::Fold => self.game_state.fold().unwrap(),
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

    use crate::arena::agent::CallingAgent;

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

        while sim.more_rounds() {
            sim.step();
        }

        assert_eq!(sim.game_state.num_active_players(), 4);

        assert_ne!(0, sim.game_state.player_winnings.iter().sum());
        assert_eq!(40, sim.game_state.player_winnings.iter().sum());
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
        game_state.advance_round().unwrap();
        // Preflop
        game_state.fold().unwrap(); // idx 3
        game_state.do_bet(10, false).unwrap(); // idx 4
        game_state.do_bet(10, false).unwrap(); // idx 0

        game_state.advance_round().unwrap();
        assert_eq!(game_state.num_active_players(), 2);

        deal_community_card("6c", &mut deck, &mut game_state);
        deal_community_card("2d", &mut deck, &mut game_state);
        deal_community_card("3d", &mut deck, &mut game_state);
        // Flop
        game_state.do_bet(90, false).unwrap(); // idx 4
        game_state.do_bet(90, false).unwrap(); // idx 0
        game_state.advance_round().unwrap();
        assert_eq!(game_state.num_active_players(), 1);

        deal_community_card("8h", &mut deck, &mut game_state);
        // Turn
        game_state.do_bet(0, false).unwrap(); // idx 4
        game_state.advance_round().unwrap();
        assert_eq!(game_state.num_active_players(), 1);

        // River
        deal_community_card("8s", &mut deck, &mut game_state);
        game_state.do_bet(100, false).unwrap(); // idx 4
        game_state.advance_round().unwrap();
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
