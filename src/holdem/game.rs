use core::FlatDeck;
use core::Deck;
use core::Card;
use core::Hand;
use core::Flattenable;

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
            for c in h.iter() {
                if !d.remove(c) {
                    return Err(format!("Card {:?} was already removed from the deck.", c));
                }
            }
        }
        Ok(Game {
               deck: d.flatten(),
               hands: hands,
               board: vec![],
           })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_create_game() {
        let g = Game::new(9).unwrap();
        assert!(g.deck.len() == 52);
        assert!(g.hands.len() == 9);
    }
}
