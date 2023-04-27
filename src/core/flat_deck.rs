use crate::core::card::Card;
use crate::core::deck::Deck;
use std::ops::{Index, Range, RangeFrom, RangeFull, RangeTo};

extern crate rand;
use rand::seq::*;
use rand::thread_rng;

/// `FlatDeck` is a deck of cards that allows easy
/// indexing into the cards. It does not provide
/// contains methods.
#[derive(Debug, Clone)]
pub struct FlatDeck {
    /// Card storage.
    cards: Vec<Card>,
}

impl FlatDeck {
    /// How many cards are there in the deck ?
    pub fn len(&self) -> usize {
        self.cards.len()
    }
    /// Have all cards been dealt ?
    /// This probably won't be used as it's unlikely
    /// that someone will deal all 52 cards from a deck.
    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }

    /// Give a random sample of the cards still left in the deck
    pub fn sample(&self, n: usize) -> Vec<Card> {
        let mut rng = thread_rng();
        self.cards.choose_multiple(&mut rng, n).cloned().collect()
    }

    /// Randomly shuffle the flat deck.
    /// This will ensure the there's no order to the deck.
    pub fn shuffle(&mut self) {
        let mut rng = thread_rng();
        self.cards.shuffle(&mut rng)
    }

    /// Deal a card if there is one there to deal.
    /// None if the deck is empty
    pub fn deal(&mut self) -> Option<Card> {
        self.cards.pop()
    }
}

impl Index<usize> for FlatDeck {
    type Output = Card;
    fn index(&self, index: usize) -> &Card {
        &self.cards[index]
    }
}
impl Index<Range<usize>> for FlatDeck {
    type Output = [Card];
    fn index(&self, index: Range<usize>) -> &[Card] {
        &self.cards[index]
    }
}
impl Index<RangeTo<usize>> for FlatDeck {
    type Output = [Card];
    fn index(&self, index: RangeTo<usize>) -> &[Card] {
        &self.cards[index]
    }
}
impl Index<RangeFrom<usize>> for FlatDeck {
    type Output = [Card];
    fn index(&self, index: RangeFrom<usize>) -> &[Card] {
        &self.cards[index]
    }
}
impl Index<RangeFull> for FlatDeck {
    type Output = [Card];
    fn index(&self, index: RangeFull) -> &[Card] {
        &self.cards[index]
    }
}

impl From<Vec<Card>> for FlatDeck {
    fn from(value: Vec<Card>) -> Self {
        Self { cards: value }
    }
}

/// Allow creating a flat deck from a Deck
impl From<Deck> for FlatDeck {
    /// Flatten this deck, consuming it to produce a `FlatDeck` that's
    /// easier to get random access to.
    fn from(value: Deck) -> Self {
        Self {
            cards: value.into_iter().collect(),
        }
    }
}
impl Default for FlatDeck {
    fn default() -> Self {
        let mut cards: Vec<Card> = Deck::default().into_iter().collect();
        let mut rng = thread_rng();
        cards.shuffle(&mut rng);
        Self { cards }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::card::{Suit, Value};

    #[test]
    fn test_deck_from() {
        let fd: FlatDeck = Deck::default().into();
        assert_eq!(52, fd.len());
    }

    #[test]
    fn test_from_vec() {
        let c = Card {
            value: Value::Nine,
            suit: Suit::Heart,
        };
        let v = vec![c];

        let mut flat_deck: FlatDeck = v.into();

        assert_eq!(1, flat_deck.len());
        assert_eq!(c, flat_deck.deal().unwrap());
    }
}
