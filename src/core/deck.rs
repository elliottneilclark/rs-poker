use crate::core::card::{Card, Suit, Value};
use std::collections::hash_set::{IntoIter, Iter};
use std::collections::HashSet;

/// Deck struct that can tell quickly if a card is in the deck
///
/// # Examples
///
/// ```
/// use rs_poker::core::{Card, Deck, Suit, Value};
///
/// // create a new deck
/// let mut deck = Deck::new();
///
/// // add some cards to the deck
/// deck.insert(Card::new(Value::Ace, Suit::Club));
/// deck.insert(Card::new(Value::King, Suit::Diamond));
/// deck.insert(Card::new(Value::Queen, Suit::Heart));
///
/// // check if a card is in the deck
/// let card = Card::new(Value::Ace, Suit::Club);
/// assert!(deck.contains(&card));
///
/// // remove a card from the deck
/// assert!(deck.remove(&card));
/// assert!(!deck.contains(&card));
///
/// // get the number of cards in the deck
/// assert_eq!(deck.len(), 2);
///
/// // check if the deck is empty
/// assert!(!deck.is_empty());
///
/// // get an iterator from the deck
/// for card in deck.iter() {
///     println!("{:?}", card);
/// }
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct Deck {
    /// Card storage.
    /// Used to figure out quickly
    /// if this card is in the deck.
    cards: HashSet<Card>,
}

impl Deck {
    pub fn new() -> Self {
        Self {
            cards: HashSet::new(),
        }
    }
    /// Given a card, is it in the current deck?
    pub fn contains(&self, c: &Card) -> bool {
        self.cards.contains(c)
    }
    /// Given a card remove it from the deck if it is present.
    pub fn remove(&mut self, c: &Card) -> bool {
        self.cards.remove(c)
    }
    /// Add a given card to the deck.
    pub fn insert(&mut self, c: Card) -> bool {
        self.cards.insert(c)
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
