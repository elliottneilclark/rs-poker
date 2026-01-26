use rand::rng;

use crate::core::{CardBitSet, FlatDeck, Hand, PlayerBitSet, RSPokerError, Rank, Rankable};

/// Current state of a game.
#[derive(Debug)]
pub struct MonteCarloGame {
    /// Flatten deck
    deck: FlatDeck,
    /// Hands still playing.
    hands: Vec<Hand>,
    starting_hands: Vec<Hand>,
    // The number of community cards that will be dealt to each player.
    num_community_cards: usize,
    // The number of needed cards each round
    cards_needed: usize,
    current_offset: usize,
}

impl MonteCarloGame {
    /// If we already have hands then lets start there.
    pub fn new(hands: Vec<Hand>) -> Result<Self, RSPokerError> {
        let mut deck = CardBitSet::default();
        let mut max_hand_size: usize = 0;
        let mut cards_needed = 0;

        for hand in &hands {
            let hand_size = hand.count();
            if hand_size > 7 {
                return Err(RSPokerError::HoldemHandSize);
            }

            // The largest hand size sets how many community cards to add
            max_hand_size = max_hand_size.max(hand_size);
            // Compute the number of cards needed per round.
            cards_needed += 7 - hand_size;

            for card in hand.iter() {
                deck.remove(card);
            }
        }

        let num_community_cards = (7 - max_hand_size).max(0);

        let flat_deck: FlatDeck = deck.into();
        // Grab the deck.len() so that any call to shuffle_if_needed
        // will result in a shuffling.
        let offset = flat_deck.len();

        Ok(Self {
            deck: flat_deck,
            starting_hands: hands.clone(),
            hands,
            num_community_cards,
            cards_needed,
            current_offset: offset,
        })
    }

    /// Simulate finishing a holdem game.
    ///
    /// This will fill out the board and then return the tuple
    /// of which hand had the best rank in end.
    pub fn simulate(&mut self) -> (PlayerBitSet, Rank) {
        self.shuffle_if_needed();

        let community_start_idx = self.current_offset;
        let community_end_idx = self.current_offset + self.num_community_cards;
        self.current_offset += self.num_community_cards;

        for h in &mut self.hands {
            h.extend(self.deck[community_start_idx..community_end_idx].to_owned());
            let hole_needed = 7 - h.count();
            let range = &self.deck[self.current_offset..self.current_offset + hole_needed];
            h.extend(range.to_owned());
            self.current_offset += hole_needed;
        }

        // Now get the best rank of all the possible hands.
        self.hands.iter().map(|h| h.rank()).enumerate().fold(
            (PlayerBitSet::default(), Rank::HighCard(0)),
            |(mut found, max_rank), (idx, rank)| {
                match rank.cmp(&max_rank) {
                    std::cmp::Ordering::Equal => {
                        // If this is a tie then add the index.
                        found.enable(idx);
                        (found, rank)
                    }
                    std::cmp::Ordering::Greater => {
                        // If this is the higest then reset all the bitset
                        // Then set only the current hand's index as true
                        found = PlayerBitSet::default();
                        found.enable(idx);
                        (found, rank)
                    }
                    // Otherwise keep what we've already found.
                    _ => (found, max_rank),
                }
            },
        )
    }

    /// Reset the game state.
    pub fn reset(&mut self) {
        self.hands
            .iter_mut()
            .zip(self.starting_hands.iter())
            .for_each(|(h, s)| {
                h.clear();
                h.extend(s.iter());
            });
    }
    fn shuffle_if_needed(&mut self) {
        if self.current_offset + self.cards_needed >= self.deck.len() {
            self.current_offset = 0;
            let mut rng = rng();
            self.deck.shuffle(&mut rng);
        }
    }

    /// Estimate the equity of each hand by simulating the game `iterations`
    /// times. This will return a vector of floats where each
    /// float is the estimated percentage of the pot that the player has in
    /// expected value.
    ///
    /// This does not take in account subsequent betting rounds.
    ///
    /// # Arguments
    ///
    /// * `iterations` - The number of times to simulate the game.
    ///
    /// # Example
    ///
    /// ```
    /// use rs_poker::core::{Card, Hand, Suit, Value};
    /// use rs_poker::holdem::MonteCarloGame;
    ///
    /// let hero = Hand::new_with_cards(vec![
    ///     Card::new(Value::Jack, Suit::Spade),
    ///     Card::new(Value::Jack, Suit::Heart),
    /// ]);
    ///
    /// let villan = Hand::new_with_cards(vec![
    ///     Card::new(Value::Ace, Suit::Spade),
    ///     Card::new(Value::King, Suit::Spade),
    /// ]);
    ///
    /// let mut monte_sim = MonteCarloGame::new(vec![hero, villan]).unwrap();
    /// let equity = monte_sim.estimate_equity(1000);
    ///
    /// // Jacks with the blocker on spades hold up most of the time
    /// assert!(equity[0] > equity[1]);
    /// ```
    pub fn estimate_equity(&mut self, iterations: usize) -> Vec<f32> {
        let mut values = vec![0.0; self.hands.len()];
        for _ in 0..iterations {
            let (winners, _) = self.simulate();

            // Reset the hands
            self.reset();
            // each player gets the pot divided by the number of people with exactly the
            // same hand value. This is to make sure that ties are correctly valued.
            let value = 1.0 / winners.count() as f32;

            for idx in winners.ones() {
                values[idx] += value;
            }
        }

        // Normalize later on in the hopes of not making
        // each value actually zero
        for v in values.iter_mut() {
            *v /= iterations as f32;
        }
        values
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::core::Card;
    use crate::core::Suit;
    use crate::core::Value;

    #[test]
    fn test_simulate_pocket_pair() {
        let hands = ["AdAh", "2c2s"]
            .iter()
            .map(|s| Hand::new_from_str(s).unwrap())
            .collect();
        let mut g = MonteCarloGame::new(hands).unwrap();
        let result = g.simulate();
        assert!(result.1 >= Rank::OnePair(0));
    }

    #[test]
    fn test_simulate_pocket_pair_with_board() {
        let board = vec![
            Card {
                suit: Suit::Spade,
                value: Value::Ace,
            },
            Card {
                suit: Suit::Diamond,
                value: Value::Three,
            },
            Card {
                suit: Suit::Diamond,
                value: Value::Four,
            },
        ];
        let mut hands: Vec<Hand> = ["AdAh", "2c2s"]
            .iter()
            .map(|s| Hand::new_from_str(s).unwrap())
            .collect();

        for h in hands.iter_mut() {
            for c in &board {
                (*h).insert(*c);
            }
        }

        let mut g = MonteCarloGame::new(hands).unwrap();
        let result = g.simulate();
        assert!(result.1 >= Rank::ThreeOfAKind(0));
    }

    #[test]
    fn test_unseen_hole_cards() {
        let hands = vec![Hand::new_from_str("KsKd").unwrap(), Hand::default()];
        let mut g = MonteCarloGame::new(hands).unwrap();
        for _i in 0..10_000 {
            let result = g.simulate();
            assert!(result.1 >= Rank::OnePair(11 << 13));
            g.reset();
        }
    }

    #[test]
    fn test_simulate_set() {
        let mut hands: Vec<Hand> = ["6d6h", "3d3h"]
            .iter()
            .map(|s| Hand::new_from_str(s).unwrap())
            .collect();
        let board: Vec<Card> = vec![
            Card {
                value: Value::Six,
                suit: Suit::Spade,
            },
            Card {
                value: Value::King,
                suit: Suit::Diamond,
            },
            Card {
                value: Value::Queen,
                suit: Suit::Heart,
            },
        ];

        for h in hands.iter_mut() {
            for c in &board {
                (*h).insert(*c);
            }
        }

        let mut g = MonteCarloGame::new(hands).unwrap();
        let result = g.simulate();
        assert!(result.1 >= Rank::ThreeOfAKind(4));
    }

    #[test]
    fn test_simulate_equity_three_kind() {
        let hero = Hand::new_with_cards(vec![
            Card {
                value: Value::Six,
                suit: Suit::Spade,
            },
            Card {
                value: Value::Six,
                suit: Suit::Club,
            },
            Card {
                value: Value::Six,
                suit: Suit::Heart,
            },
            Card {
                value: Value::King,
                suit: Suit::Heart,
            },
            Card {
                value: Value::Ten,
                suit: Suit::Diamond,
            },
        ]);

        let board = vec![
            Card {
                value: Value::Six,
                suit: Suit::Heart,
            },
            Card {
                value: Value::King,
                suit: Suit::Heart,
            },
            Card {
                value: Value::Ten,
                suit: Suit::Diamond,
            },
        ];
        let hands = vec![
            hero,
            Hand::new_with_cards(board.clone()),
            Hand::new_with_cards(board.clone()),
            Hand::new_with_cards(board.clone()),
            Hand::new_with_cards(board),
        ];

        let mut g = MonteCarloGame::new(hands).unwrap();
        let equity = g.estimate_equity(1000);

        assert!(equity[0] > equity[1]);
        assert!(equity[0] > equity[2]);
        assert!(equity[0] > equity[3]);
        assert!(equity[0] > equity[4]);
    }

    #[test]
    fn test_simulate_equity_lots_players() {
        let mut rng = rng();
        for in_hand in 2..7 {
            for num_players in 3..9 {
                let mut deck = FlatDeck::default();
                deck.shuffle(&mut rng);

                let hands: Vec<Hand> = deck[..]
                    .chunks(in_hand)
                    .take(num_players)
                    .map(|cards| Hand::new_with_cards(cards.to_owned()))
                    .collect();

                let mut g = MonteCarloGame::new(hands.clone()).unwrap();
                let _result = g.estimate_equity(1_000);
            }
        }
    }
    #[test]
    fn test_simulate_equity_cleaned_hands() {
        let mut rng = rng();
        for in_hand in 2..7 {
            for num_players in 3..9 {
                let mut deck = FlatDeck::default();
                deck.shuffle(&mut rng);

                let clean_hands = deck[..]
                    .chunks(in_hand)
                    .take(num_players)
                    .map(|cards| Hand::new_with_cards(cards.to_owned()))
                    .enumerate()
                    .map(|(idx, h)| if idx == 0 { h } else { Hand::default() })
                    .collect();

                let mut g = MonteCarloGame::new(clean_hands).unwrap();
                let _result = g.estimate_equity(1_000);
            }
        }
    }

    /// Test that ties are handled correctly in simulate.
    #[test]
    fn test_simulate_tie() {
        // Give both players the exact same hand
        let hands = vec![
            Hand::new_from_str("AsKh").unwrap(),
            Hand::new_from_str("AdKc").unwrap(),
        ];
        let mut g = MonteCarloGame::new(hands).unwrap();

        // Run multiple simulations
        let mut tie_found = false;
        for _ in 0..1000 {
            let (winners, _rank) = g.simulate();
            // If both players win, it's a tie
            if winners.count() == 2 {
                tie_found = true;
                assert!(winners.get(0));
                assert!(winners.get(1));
            }
            g.reset();
        }
        // With same starting hands, there should be ties
        assert!(tie_found, "Expected to find at least one tie");
    }

    /// Test that equity values sum to approximately 1.0.
    #[test]
    fn test_estimate_equity_sums_to_one() {
        let hands = vec![
            Hand::new_from_str("AsAh").unwrap(),
            Hand::new_from_str("KsKh").unwrap(),
        ];
        let mut g = MonteCarloGame::new(hands).unwrap();
        let equity = g.estimate_equity(1000);

        let sum: f32 = equity.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Equity should sum to 1.0, got {}",
            sum
        );
    }

    /// Test equity with ties is correctly split between players.
    #[test]
    fn test_estimate_equity_tie_splitting() {
        // Same cards = should be close to 50/50
        let hands = vec![
            Hand::new_from_str("AsKh").unwrap(),
            Hand::new_from_str("AdKc").unwrap(),
        ];
        let mut g = MonteCarloGame::new(hands).unwrap();
        let equity = g.estimate_equity(5000);

        // Should be roughly equal
        assert!(
            (equity[0] - equity[1]).abs() < 0.1,
            "Same hands should have similar equity, got {} vs {}",
            equity[0],
            equity[1]
        );
    }

    /// Test that many simulations correctly trigger reshuffles.
    #[test]
    fn test_many_simulations_reshuffles() {
        let hands = vec![
            Hand::new_from_str("AsAh").unwrap(),
            Hand::new_from_str("KsKh").unwrap(),
        ];
        let mut g = MonteCarloGame::new(hands).unwrap();

        // Run enough simulations to trigger multiple reshuffles
        for _ in 0..100 {
            let _ = g.simulate();
            g.reset();
        }
        // If we get here without panic/error, shuffling works
    }
}
