use crate::core::card::{Card, Suit, Value};
use std::collections::hash_set::{IntoIter, Iter};
use std::collections::HashSet;

/// Deck struct that can tell quickly if a card is in the deck
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct Deck {
    /// Card storage.
    /// Used to figure out quickly
    /// if this card is in the deck.
    cards: HashSet<Card>,
}

impl Deck {
    /// Given a card, is it in the current deck?
    pub fn contains(&self, c: &Card) -> bool {
        self.cards.contains(c)
    }
    /// Given a card remove it from the deck if it is present.
    pub fn remove(&mut self, c: &Card) -> bool {
        self.cards.remove(c)
    }
    /// How many cards are there in the deck.
    pub fn len(&self) -> usize {
        self.cards.len()
    }
    /// Have all of the cards been dealt from this deck?
    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }
    /// Get an iterator from this deck
    pub fn iter(&self) -> Iter<Card> {
        self.cards.iter()
    }
}

/// Turn a deck into an iterator
impl IntoIterator for Deck {
    type Item = Card;
    type IntoIter = IntoIter<Card>;
    /// Consume this deck and create a new iterator.
    fn into_iter(self) -> IntoIter<Card> {
        self.cards.into_iter()
    }
}

impl Default for Deck {
    /// Create the default 52 card deck
    ///
    /// ```
    /// use rs_poker::core::Deck;
    ///
    /// assert_eq!(52, Deck::default().len());
    /// ```
    fn default() -> Self {
        let mut cards: HashSet<Card> = HashSet::new();
        for v in &Value::values() {
            for s in &Suit::suits() {
                cards.insert(Card {
                    value: *v,
                    suit: *s,
                });
            }
        }
        Self { cards }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::card::*;

    #[test]
    fn test_contains_in() {
        let d = Deck::default();
        assert!(d.contains(&Card {
            value: Value::Eight,
            suit: Suit::Heart,
        }));
    }

    #[test]
    fn test_remove() {
        let mut d = Deck::default();
        let c = Card {
            value: Value::Ace,
            suit: Suit::Heart,
        };
        assert!(d.contains(&c));
        assert!(d.remove(&c));
        assert!(!d.contains(&c));
        assert!(!d.remove(&c));
    }
}
