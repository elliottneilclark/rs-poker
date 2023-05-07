use crate::{core::*, utils::PlayerBitSet};

/// Current state of a game.
#[derive(Debug)]
pub struct MonteCarloGame {
    /// Flatten deck
    deck: FlatDeck,
    /// Hands still playing.
    hands: Vec<Hand>,
    // The origional size of each of the hands.
    // This is used to reset each hand after a round
    hand_sizes: Vec<usize>,
    // The number of community cards that will be dealt to each player.
    num_community_cards: usize,
    // The number of needed cards each round
    cards_needed: usize,
    current_offset: usize,
}

impl MonteCarloGame {
    /// If we already have hands then lets start there.
    pub fn new(hands: Vec<Hand>) -> Result<Self, String> {
        let mut deck = Deck::default();
        let mut max_hand_size: usize = 0;
        let mut cards_needed = 0;
        let mut hand_sizes: Vec<usize> = vec![];

        for hand in &hands {
            let hand_size = hand.len();
            assert!(hand_size < 7, "Holdem only has 7 cards in a hand.");
            max_hand_size = max_hand_size.max(hand_size);
            hand_sizes.push(hand_size);
            cards_needed += 7 - hand_size;

            for card in hand.iter() {
                deck.remove(card);
            }
        }

        let num_community_cards = (7 - max_hand_size).min(5).max(0);

        let flat_deck: FlatDeck = deck.into();
        // Grab the deck.len() so that any call to shuffle_if_needed
        // will result in a shuffling.
        let offset = flat_deck.len();

        Ok(Self {
            deck: flat_deck,
            hands,
            hand_sizes,
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
            let hole_needed = 7 - h.len();
            h.extend(self.deck[self.current_offset..self.current_offset + hole_needed].to_owned());
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
        for (h, hand_size) in self.hands.iter_mut().zip(self.hand_sizes.iter()) {
            h.truncate(*hand_size);
        }
    }
    fn shuffle_if_needed(&mut self) {
        if self.current_offset + self.cards_needed >= self.deck.len() {
            self.current_offset = 0;
            self.deck.shuffle();
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::core::Hand;
    use crate::core::Rank;

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
                (*h).push(*c);
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
                (*h).push(*c);
            }
        }

        let mut g = MonteCarloGame::new(hands).unwrap();
        let result = g.simulate();
        assert!(result.1 >= Rank::ThreeOfAKind(4));
    }
}
