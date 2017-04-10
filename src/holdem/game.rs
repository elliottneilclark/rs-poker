use core::*;

extern crate rand;
use self::rand::{thread_rng, sample};

/// Current state of a game.
#[derive(Debug)]
pub struct Game {
    /// Flatten deck
    deck: FlatDeck,
    /// Community cards.
    board: Vec<Card>,
    /// Hands still playing.
    hands: Vec<Hand>,
}

impl Game {
    /// Create a new game with no cards dealt and `num_players` empty hands.
    pub fn new(num_players: usize) -> Result<Game, String> {
        Ok(Game {
               deck: Deck::default().flatten(),
               board: Vec::with_capacity(5),
               hands: (0..num_players).map(|_| Hand::default()).collect(),
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
           })
    }

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
        // Get an rng to help out
        let mut rng = thread_rng();
        // Now iterate over a sample of the deck.
        for c in &sample(&mut rng, &self.deck[..], num_cards) {
            for h in &mut self.hands {
                // This sucks.
                // Sample gives a reference to a ref.
                h.push(*(*c));
            }
        }

        // Now get the best rank of all the possible hands.
        let best_rank = self.hands
            .iter()
            .map(|h| h.rank_seven())
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
