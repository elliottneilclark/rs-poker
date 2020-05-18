use crate::core::*;

/// Current state of a game.
#[derive(Debug)]
pub struct MonteCarloGame {
    /// Flatten deck
    deck: FlatDeck,
    /// Community cards.
    pub board: Vec<Card>,
    /// Hands still playing.
    hands: Vec<Hand>,
    current_offset: usize,
}

impl MonteCarloGame {
    /// If we already have hands then lets start there.
    pub fn new_with_hands(hands: Vec<Hand>) -> Result<Self, String> {
        let mut d = Deck::default();
        for h in &hands {
            if h.len() != 2 {
                return Err(String::from("Hand passed in doesn't have 2 cards."));
            }
            for c in h.iter() {
                if !d.remove(*c) {
                    return Err(format!("Card {} was already removed from the deck.", c));
                }
            }
        }
        Ok(Self {
            deck: d.flatten(),
            hands,
            board: vec![],
            current_offset: 52,
        })
    }

    pub fn new_with_board(hands: Vec<Hand>, board: Vec<Card>) -> Result<Self, String> {
        let mut deck = Deck::default();
        if board.len() > 5 {
            return Err(String::from("Board passed in has more than 5 cards"));
        }

        for hand in &hands {
            if hand.len() != 2 {
                return Err(String::from("Hand passed in doesn't have 2 cards."));
            }
            for card in hand.iter() {
                if !deck.remove(*card) {
                    return Err(format!("Card {} was already removed from the deck.", card));
                }
            }
        }

        for card in &board {
            if !deck.remove(*card) {
                return Err(format!("Card {} was already removed from the deck.", card));
            }
        }

        Ok(Self {
            deck: deck.flatten(),
            hands,
            board,
            current_offset: 52,
        })
    }

    /// Simulate finishing a holdem game.
    ///
    /// This will fill out the board and then return the tuple
    /// of which hand had the best rank in end.
    pub fn simulate(&mut self) -> Result<(usize, Rank), String> {
        if self.hands.is_empty() {
            return Err(String::from("There are no hands."));
        }
        // Add the board cards to all the hands.
        for c in &self.board {
            for h in &mut self.hands {
                h.push(*c);
            }
        }
        // Figure out how many cards to deal.
        let num_cards = 5 - self.board.len();
        // Now iterate over a sample of the deck.
        self.shuffle_if_needed();
        for c in &self.deck[self.current_offset..self.current_offset + num_cards] {
            for h in &mut self.hands {
                h.push(*c);
            }
        }
        self.current_offset += num_cards;

        // Now get the best rank of all the possible hands.
        let best_rank = self
            .hands
            .iter()
            .map(|h| h.rank())
            .enumerate()
            .max_by_key(|&(_, ref rank)| rank.clone())
            .ok_or_else(|| String::from("Unable to determine best rank."));
        Ok(best_rank?)
    }
    /// Reset the game state.
    pub fn reset(&mut self) {
        for h in &mut self.hands {
            h.truncate(2 + self.board.len());
        }
    }
    fn shuffle_if_needed(&mut self) {
        if self.current_offset + 5 > self.deck.len() {
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
        let mut g = MonteCarloGame::new_with_hands(hands).unwrap();
        let result = g.simulate().unwrap();
        assert!(result.1 >= Rank::OnePair(0));
    }

    #[test]
    fn test_simulate_set() {
        let hands: Vec<Hand> = ["6d6h", "3d3h"]
            .iter()
            .map(|s| Hand::new_from_str(s).unwrap())
            .collect();
        let board: Vec<Card> = vec![
            Card {
                value: Value::Six,
                suit: Suit::Spade
            }, Card {
            value: Value::King,
            suit: Suit::Diamond
            }, Card {
                value: Value::Queen,
                suit: Suit::Heart
            }    
        ];
        let mut g = MonteCarloGame::new_with_board(hands, board).unwrap();
        let result = g.simulate().unwrap();
        assert!(result.1 >= Rank::ThreeOfAKind(4));

    }
}
