use core::*;

/// Current state of a game.
#[derive(Debug)]
pub struct Game {
    /// Flatten deck
    deck: FlatDeck,
    /// Community cards.
    board: Vec<Card>,
    /// Hands still playing.
    hands: Vec<Hand>,
    current_offset: usize,
}

impl Game {
    /// Create a new game with no cards dealt and `num_players` empty hands.
    pub fn new(num_players: usize) -> Result<Game, String> {
        Ok(Game {
               deck: Deck::default().flatten(),
               board: Vec::with_capacity(5),
               hands: (0..num_players).map(|_| Hand::default()).collect(),
               current_offset: 52,
           })
    }

    /// If we already have hands then lets start there.
    pub fn new_with_hands(hands: Vec<Hand>) -> Result<Game, String> {
        let mut d = Deck::default();
        for h in &hands {
            if h.len() != 2 {
                return Err(String::from("Hand passed in doesn't have 2 cards."));
            }
            for c in h.iter() {
                if !d.remove(c) {
                    return Err(format!("Card {} was already removed from the deck.", c));
                }
            }
        }
        Ok(Game {
               deck: d.flatten(),
               hands: hands,
               board: vec![],
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
        let best_rank = self.hands
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
    use core::Hand;
    use core::Rank;

    #[test]
    fn test_create_game() {
        let g = Game::new(9).unwrap();
        assert!(g.deck.len() == 52);
        assert!(g.hands.len() == 9);
    }


    #[test]
    fn test_simulate_pocket_pair() {
        let hands = ["AdAh", "2c2s"]
            .iter()
            .map(|s| Hand::new_from_str(s).unwrap())
            .collect();
        let mut g = Game::new_with_hands(hands).unwrap();
        let result = g.simulate().unwrap();
        println!("h0 = {:?}", g.hands[0]);
        println!("h1 = {:?}", g.hands[1]);
        assert!(result.1 >= Rank::OnePair(0));

    }
}
