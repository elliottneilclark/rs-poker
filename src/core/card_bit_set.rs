use std::ops::{BitOr, BitOrAssign, BitXor, BitXorAssign};

use super::{Card, FlatDeck};
use std::fmt::Debug;

/// This struct is a bitset for cards
/// Each card is represented by a bit in a 64 bit integer
///
/// The bit is set if the card present
/// The bit is unset if the card not in the set
///
/// It implements the BitOr, BitAnd, and BitXor traits
/// It implements the Display trait
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CardBitSet {
    // The bitset
    cards: u64,
}

const FIFTY_TWO_ONES: u64 = (1 << 52) - 1;

impl CardBitSet {
    /// Create a new empty bitset
    ///
    /// ```
    /// use rs_poker::core::CardBitSet;
    /// let cards = CardBitSet::new();
    /// assert!(cards.is_empty());
    /// ```
    pub fn new() -> Self {
        Self { cards: 0 }
    }

    /// This does what it says on the tin it insertes a card into the bitset
    ///
    /// ```
    /// use rs_poker::core::{Card, CardBitSet, Deck, Suit, Value};
    /// let mut cards = CardBitSet::new();
    ///
    /// cards.insert(Card::new(Value::Six, Suit::Club));
    /// cards.insert(Card::new(Value::King, Suit::Club));
    /// cards.insert(Card::new(Value::Ace, Suit::Club));
    /// assert_eq!(3, cards.count());
    /// ```
    pub fn insert(&mut self, card: Card) {
        self.cards |= 1 << u8::from(card);
    }

    /// Remove a card from the bitset
    ///
    /// ```
    /// use rs_poker::core::{Card, CardBitSet, Deck, Suit, Value};
    /// let mut cards = CardBitSet::new();
    /// cards.insert(Card::from(17));
    ///
    /// // We're using the u8 but it's got a value as well
    /// assert_eq!(Card::new(Value::Six, Suit::Club), Card::from(17));
    ///
    /// // The card is in the bitset
    /// assert!(cards.contains(Card::new(Value::Six, Suit::Club)));
    /// // We can remove the card
    /// cards.remove(Card::new(Value::Six, Suit::Club));
    ///
    /// // show that the card is no longer in the bitset
    /// assert!(!cards.contains(Card::from(17)));
    /// ```
    pub fn remove(&mut self, card: Card) {
        self.cards &= !(1 << u8::from(card));
    }

    /// Is the card in the bitset ?
    ///
    /// ```
    /// use rs_poker::core::{Card, CardBitSet, Deck, Suit, Value};
    ///
    /// let mut cards = CardBitSet::new();
    /// cards.insert(Card::from(17));
    ///
    /// assert!(cards.contains(Card::new(Value::Six, Suit::Club)));
    /// ```
    pub fn contains(&self, card: Card) -> bool {
        (self.cards & (1 << u8::from(card))) != 0
    }

    /// Is the bitset empty ?
    ///
    /// ```
    /// use rs_poker::core::{Card, CardBitSet};
    ///
    /// let mut cards = CardBitSet::new();
    /// assert!(cards.is_empty());
    ///
    /// cards.insert(Card::from(17));
    /// assert!(!cards.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.cards == 0
    }

    /// How many cards are in the bitset ?
    ///
    /// ```
    /// use rs_poker::core::{Card, CardBitSet};
    /// let mut cards = CardBitSet::new();
    ///
    /// assert_eq!(0, cards.count());
    /// for card in 0..13 {
    ///    cards.insert(Card::from(card));
    ///    assert_eq!(card as usize + 1, cards.count());
    /// }
    /// assert_eq!(13, cards.count());
    pub fn count(&self) -> usize {
        self.cards.count_ones() as usize
    }
}

impl Default for CardBitSet {
    /// Create a new bitset with all the cards in it
    /// ```
    /// use rs_poker::core::CardBitSet;
    ///
    /// let cards = CardBitSet::default();
    ///
    /// assert_eq!(52, cards.count());
    /// assert!(!cards.is_empty());
    /// ```
    fn default() -> Self {
        Self {
            cards: FIFTY_TWO_ONES,
        }
    }
}

// Trait for converting a CardBitSet into a FlatDeck
// Create the vec for storage and then return the flatdeck
impl From<CardBitSet> for FlatDeck {
    fn from(value: CardBitSet) -> Self {
        value.into_iter().collect::<Vec<Card>>().into()
    }
}

impl Debug for CardBitSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;

        for idx in (0..52).rev() {
            let card = Card::from(idx);
            if self.contains(card) {
                write!(f, "A")?;
            } else {
                write!(f, "_")?;
            }
        }

        write!(f, "]")
    }
}

impl BitOr<CardBitSet> for CardBitSet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            cards: self.cards | rhs.cards,
        }
    }
}

impl BitOr<Card> for CardBitSet {
    type Output = Self;

    fn bitor(self, rhs: Card) -> Self::Output {
        Self {
            cards: self.cards | (1 << u8::from(rhs)),
        }
    }
}

impl BitOrAssign<CardBitSet> for CardBitSet {
    fn bitor_assign(&mut self, rhs: Self) {
        self.cards |= rhs.cards;
    }
}

impl BitOrAssign<Card> for CardBitSet {
    fn bitor_assign(&mut self, rhs: Card) {
        self.cards |= 1 << u8::from(rhs);
    }
}

impl BitXor for CardBitSet {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self {
            cards: self.cards ^ rhs.cards,
        }
    }
}

impl BitXor<Card> for CardBitSet {
    type Output = Self;

    fn bitxor(self, rhs: Card) -> Self::Output {
        Self {
            cards: self.cards ^ (1 << u8::from(rhs)),
        }
    }
}

impl BitXorAssign<Card> for CardBitSet {
    fn bitxor_assign(&mut self, rhs: Card) {
        self.cards ^= 1 << u8::from(rhs);
    }
}

impl BitXorAssign<CardBitSet> for CardBitSet {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.cards ^= rhs.cards;
    }
}

/// The iterator for the CardBitSet
/// It iterates over the cards in the bitset
pub struct CardBitSetIter(u64);

impl IntoIterator for CardBitSet {
    type Item = Card;
    type IntoIter = CardBitSetIter;

    fn into_iter(self) -> Self::IntoIter {
        CardBitSetIter(self.cards)
    }
}

impl Iterator for CardBitSetIter {
    type Item = Card;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            return None;
        }

        let card = self.0.trailing_zeros();
        self.0 &= !(1 << card);

        Some(Card::from(card as u8))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::core::Deck;

    use super::*;

    #[test]
    fn test_empty() {
        let cards = CardBitSet::new();
        assert!(cards.is_empty());
    }

    #[test]
    fn test_insert_all() {
        let mut all_cards = CardBitSet::new();
        for card in Deck::default() {
            let mut single_card = CardBitSet::new();

            single_card.insert(card);
            all_cards |= single_card;

            assert!(single_card.contains(card));
        }

        assert_eq!(all_cards.count(), 52);

        for card in Deck::default() {
            assert!(all_cards.contains(card));
        }
    }

    #[test]
    fn test_xor_is_remove() {
        let mut all_cards = CardBitSet::new();
        for card in Deck::default() {
            all_cards |= card;
        }

        for card in Deck::default() {
            let xor_masked_set: CardBitSet = all_cards ^ card;
            assert!(!xor_masked_set.contains(card));

            let mut removed_set = all_cards;
            removed_set.remove(card);

            assert_eq!(removed_set, xor_masked_set);
        }
        assert_eq!(52, all_cards.count());
    }

    #[test]
    fn test_is_empty() {
        let empty = CardBitSet::new();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_not_empty() {
        let mut cards = CardBitSet::new();

        cards.insert(Card::from(17));
        assert!(!cards.is_empty());
    }

    #[test]
    fn test_add_cards_iter() {
        let mut hash_set: HashSet<Card> = HashSet::new();
        let mut bit_set = CardBitSet::new();

        let deck = FlatDeck::from(Deck::default());

        for card in deck.sample(13) {
            hash_set.insert(card);
            bit_set.insert(card);
        }

        assert_eq!(hash_set.len(), bit_set.count());
        for card in hash_set.clone() {
            assert!(bit_set.contains(card));
        }

        for card in bit_set {
            assert!(hash_set.contains(&card));
        }
    }

    #[test]
    fn test_default_contains() {
        let mut bitset_cards = CardBitSet::default();
        assert_eq!(52, bitset_cards.count());

        for card in Deck::default() {
            assert!(bitset_cards.contains(card));
            bitset_cards.remove(card);
        }

        assert_eq!(0, bitset_cards.count());
        assert!(bitset_cards.is_empty());
    }
}
