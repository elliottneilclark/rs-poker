use crate::core::card::Card;

use super::{CardBitSet, CardBitSetIter};

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
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct Deck(CardBitSet);

impl Deck {
    /// Create a new empty deck
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::core::Deck;
    ///
    /// let deck = Deck::new();
    ///
    /// assert!(deck.is_empty());
    /// assert_eq!(0, deck.len());
    /// ```
    pub fn new() -> Self {
        Self(CardBitSet::new())
    }
    /// Given a card, is it in the current deck?
    pub fn contains(&self, c: &Card) -> bool {
        self.0.contains(*c)
    }
    /// Given a card remove it from the deck if it is present.
    pub fn remove(&mut self, c: &Card) -> bool {
        let contains = self.contains(c);
        self.0.remove(*c);
        contains
    }
    /// Add a given card to the deck.
    pub fn insert(&mut self, c: Card) -> bool {
        let contains = self.contains(&c);
        self.0.insert(c);
        !contains
    }
    /// How many cards are there in the deck.
    pub fn count(&self) -> usize {
        self.0.count()
    }
    /// Have all of the cards been dealt from this deck?
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    /// Get an iterator from this deck
    pub fn iter(&self) -> CardBitSetIter {
        self.0.into_iter()
    }

    pub fn len(&self) -> usize {
        self.0.count()
    }
}

/// Turn a deck into an iterator
impl IntoIterator for Deck {
    type Item = Card;
    type IntoIter = CardBitSetIter;
    /// Consume this deck and create a new iterator.
    fn into_iter(self) -> CardBitSetIter {
        self.0.into_iter()
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
        Self(CardBitSet::default())
    }
}

#[cfg(test)]
mod tests {
    use crate::core::{Suit, Value};

    use super::*;

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
