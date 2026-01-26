use rand::Rng;

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
#[derive(Debug, Clone, Copy, PartialEq)]
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

    pub fn deal<R: Rng>(&mut self, rng: &mut R) -> Option<Card> {
        let card = self.0.sample_one(rng);
        if let Some(c) = card {
            // remove the card from the deck
            self.remove(&c);
            Some(c)
        } else {
            None
        }
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

impl From<CardBitSet> for Deck {
    /// Convert a `CardBitSet` into a `Deck`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::core::CardBitSet;
    /// use rs_poker::core::{Card, Deck, Suit, Value};
    ///
    /// let mut card_bit_set = CardBitSet::new();
    ///
    /// // Add some cards to the CardBitSet
    ///
    /// card_bit_set.insert(Card::new(Value::Ace, Suit::Club));
    /// card_bit_set.insert(Card::new(Value::King, Suit::Diamond));
    ///
    /// // Convert the CardBitSet into a Deck
    /// let deck: Deck = card_bit_set.into();
    ///
    /// assert_eq!(2, deck.len());
    /// assert!(deck.contains(&Card::new(Value::Ace, Suit::Club)));
    /// assert!(deck.contains(&Card::new(Value::King, Suit::Diamond)));
    /// ```
    fn from(val: CardBitSet) -> Self {
        Deck(val)
    }
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};

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

    #[test]
    fn test_deal() {
        let mut d = Deck::default();
        let mut rng = rand::rng();
        let c = d.deal(&mut rng);
        assert!(c.is_some());
        assert!(!d.contains(&c.unwrap()));

        let other = d.deal(&mut rng);
        assert!(other.is_some());

        assert_ne!(c, other);
        assert_eq!(d.len(), 50);
    }

    #[test]
    fn test_deal_all() {
        let mut cards_dealt = 0;
        let mut d = Deck::default();

        let mut rng = rand::rng();

        while let Some(_c) = d.deal(&mut rng) {
            cards_dealt += 1;
        }
        assert_eq!(cards_dealt, 52);
        assert!(d.is_empty());
    }

    #[test]
    fn test_stable_deal_order_with_seed_rng() {
        let mut rng_one = StdRng::seed_from_u64(420);
        let mut rng_two = StdRng::seed_from_u64(420);

        let mut d_one = Deck::default();
        let mut d_two = Deck::default();

        let mut cards_dealt_one = Vec::with_capacity(52);
        let mut cards_dealt_two = Vec::with_capacity(52);

        while let Some(c) = d_one.deal(&mut rng_one) {
            cards_dealt_one.push(c);
        }
        while let Some(c) = d_two.deal(&mut rng_two) {
            cards_dealt_two.push(c);
        }
        assert_eq!(cards_dealt_one, cards_dealt_two);
        assert!(d_one.is_empty());
        assert!(d_two.is_empty());
    }

    #[test]
    fn test_insert_returns_bool() {
        let mut d = Deck::new();
        let c = Card {
            value: Value::Ace,
            suit: Suit::Heart,
        };
        assert!(d.insert(c));
        assert!(!d.insert(c));
        assert!(d.contains(&c));
        assert_eq!(d.len(), 1);

        let c2 = Card {
            value: Value::Two,
            suit: Suit::Heart,
        };
        assert!(d.insert(c2));
        assert!(!d.insert(c2));
        assert!(d.contains(&c2));
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn test_count_zero() {
        let d = Deck::new();
        assert_eq!(0, d.count());
    }

    #[test]
    fn test_count_after_adding() {
        let mut d = Deck::new();

        let c = Card {
            value: Value::Ace,
            suit: Suit::Heart,
        };

        d.insert(c);
        assert_eq!(1, d.count());
        d.insert(Card {
            value: Value::Two,
            suit: Suit::Heart,
        });

        assert_eq!(2, d.count());
    }

    /// Test is_empty returns false for non-empty deck and true for empty deck.
    #[test]
    fn test_is_empty() {
        let d = Deck::new();
        assert!(d.is_empty());

        let d = Deck::default();
        assert!(!d.is_empty());

        let mut d = Deck::new();
        d.insert(Card::new(Value::Ace, Suit::Spade));
        assert!(!d.is_empty());
    }

    /// Test From<CardBitSet> for Deck preserves the cards.
    #[test]
    fn test_from_card_bit_set() {
        use crate::core::CardBitSet;

        let mut cbs = CardBitSet::new();
        let ace_spade = Card::new(Value::Ace, Suit::Spade);
        let king_heart = Card::new(Value::King, Suit::Heart);

        cbs.insert(ace_spade);
        cbs.insert(king_heart);

        let deck: Deck = cbs.into();

        assert_eq!(deck.len(), 2);
        assert!(deck.contains(&ace_spade));
        assert!(deck.contains(&king_heart));
        assert!(!deck.contains(&Card::new(Value::Two, Suit::Club)));
    }
}
