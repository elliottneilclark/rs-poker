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
    #[must_use]
    pub fn len(&self) -> usize {
        self.cards.len()
    }
    /// Have all cards been dealt ?
    /// This probably won't be used as it's unlikely
    /// that someone will deal all 52 cards from a deck.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }

    /// Give a random sample of the cards still left in the deck
    #[must_use]
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
    #[must_use]
    fn index(&self, index: usize) -> &Card {
        &self.cards[index]
    }
}
impl Index<Range<usize>> for FlatDeck {
    type Output = [Card];
    #[must_use]
    fn index(&self, index: Range<usize>) -> &[Card] {
        &self.cards[index]
    }
}
impl Index<RangeTo<usize>> for FlatDeck {
    type Output = [Card];
    #[must_use]
    fn index(&self, index: RangeTo<usize>) -> &[Card] {
        &self.cards[index]
    }
}
impl Index<RangeFrom<usize>> for FlatDeck {
    type Output = [Card];
    #[must_use]
    fn index(&self, index: RangeFrom<usize>) -> &[Card] {
        &self.cards[index]
    }
}
impl Index<RangeFull> for FlatDeck {
    type Output = [Card];
    #[must_use]
    fn index(&self, index: RangeFull) -> &[Card] {
        &self.cards[index]
    }
}

/// Trait that means a deck can be made into a `FlatDeck`
pub trait Flattenable {
    /// Consume a `Deck` and produce a deck suitable for random index.
    fn flatten(self) -> FlatDeck;
}

/// Allow creating a flat deck from a Deck
impl Flattenable for Deck {
    /// Flatten this deck, consuming it to produce a `FlatDeck` that's
    /// easier to get random access to.
    #[must_use]
    fn flatten(self) -> FlatDeck {
        FlatDeck {
            cards: self.into_iter().collect(),
        }
    }
}

impl Into<FlatDeck> for Deck {
    /// Flatten a `Deck` into a `FlatDeck`.
    #[must_use]
    fn into(self) -> FlatDeck {
        self.flatten()
    }
}
