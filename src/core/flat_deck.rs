use std::ops::{Index, Range, RangeTo, RangeFrom, RangeFull};
use core::card::Card;
use core::deck::Deck;

extern crate rand;
use self::rand::{thread_rng, sample, Rng};

/// `FlatDeck` is a deck of cards that allows easy
/// indexing into the cards. It does not provide
/// contains methods.
#[derive(Debug)]
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
        sample(&mut rng, &self.cards[..], n)
            .iter()
            .map(|c| *(*c))
            .collect()
    }

    /// Randomly shuffle the flat deck.
    /// This will ensure the there's no order to the deck.
    pub fn shuffle(&mut self) {
        let mut rng = thread_rng();
        rng.shuffle(&mut self.cards)
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


/// Trait that means a deck can be made into a `FlatDeck`
pub trait Flattenable {
    /// Consume a `Deck` and produce a deck suitable for random index.
    fn flatten(self) -> FlatDeck;
}

/// Allow creating a flat deck from a Deck
impl Flattenable for Deck {
    /// Flatten this deck, consuming it to produce a `FlatDeck` that's
    /// easier to get random access to.
    fn flatten(self) -> FlatDeck {
        FlatDeck { cards: self.into_iter().collect() }
    }
}

impl Into<FlatDeck> for Deck {
    fn into(self) -> FlatDeck {
        self.flatten()
    }
}
