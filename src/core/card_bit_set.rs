use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use super::{Card, FlatDeck};
use std::fmt::Debug;

use rand::Rng;
#[cfg(feature = "serde")]
use serde::ser::SerializeSeq;

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

    pub fn clear(&mut self) {
        self.cards = 0;
    }

    /// Sample one card from the bitset
    ///
    /// Returns `None` if the bitset is empty
    ///
    ///
    /// # Examples
    ///
    /// Sample will give a random card from the bitset
    ///
    /// ```
    /// use rand::rng;
    /// use rs_poker::core::{Card, CardBitSet, Deck};
    ///
    /// let mut rng = rng();
    /// let cards = CardBitSet::default();
    /// let card = cards.sample_one(&mut rng);
    ///
    /// assert!(card.is_some());
    /// assert!(cards.contains(card.unwrap()));
    /// ```
    ///
    /// ```
    /// use rand::rng;
    /// use rs_poker::core::{Card, CardBitSet};
    ///
    /// let mut rng = rng();
    /// let cards = CardBitSet::new();
    /// assert!(cards.sample_one(&mut rng).is_none());
    /// ```
    pub fn sample_one<R: Rng>(&self, rng: &mut R) -> Option<Card> {
        let count = self.count();
        if count == 0 {
            return None;
        }

        let idx = rng.random_range(0..count) as u32;
        Some(Card::from(self.nth_set_bit(idx) as u8))
    }

    /// Find the position of the `n`th set bit (0-indexed) in this bitset.
    ///
    /// On x86_64 with BMI2 this compiles to `PDEP` + `TZCNT` (two
    /// instructions, branchless). Otherwise falls back to a binary search
    /// over popcount in 6 constant-time steps.
    #[inline]
    fn nth_set_bit(&self, n: u32) -> u32 {
        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        {
            // SAFETY: target_feature = "bmi2" guarantees PDEP is available.
            // PDEP deposits `1 << n` into the positions of the set bits of
            // `self.cards`, effectively placing a single 1-bit at the position
            // of the n-th set bit. TZCNT then reads off that position.
            unsafe {
                use core::arch::x86_64::_pdep_u64;
                return _pdep_u64(1u64 << n, self.cards).trailing_zeros();
            }
        }

        #[allow(unreachable_code)]
        self.nth_set_bit_fallback(n)
    }

    /// Fallback select implementation using binary search over popcount.
    ///
    /// At each step we split the remaining bits into a lower half and an upper
    /// half, then count the set bits in the lower half:
    ///
    /// - If `n` is **less than** that count, the target bit is in the lower
    ///   half — keep searching there.
    /// - If `n` is **greater than or equal to** that count, the target bit is
    ///   in the upper half — subtract the lower count from `n`, shift the upper
    ///   half down, and advance `pos` by the half-width.
    #[inline]
    fn nth_set_bit_fallback(&self, n: u32) -> u32 {
        let mut n = n;
        let mut bits = self.cards;
        let mut pos = 0u32;

        let c = (bits & 0xFFFF_FFFF).count_ones();
        if n >= c {
            n -= c;
            bits >>= 32;
            pos += 32;
        }

        let c = (bits & 0x0000_FFFF).count_ones();
        if n >= c {
            n -= c;
            bits >>= 16;
            pos += 16;
        }

        let c = (bits & 0x0000_00FF).count_ones();
        if n >= c {
            n -= c;
            bits >>= 8;
            pos += 8;
        }

        let c = (bits & 0x0000_000F).count_ones();
        if n >= c {
            n -= c;
            bits >>= 4;
            pos += 4;
        }

        let c = (bits & 0x0000_0003).count_ones();
        if n >= c {
            n -= c;
            bits >>= 2;
            pos += 2;
        }

        let c = (bits & 0x0000_0001) as u32;
        if n >= c {
            pos += 1;
        }

        pos
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
        f.debug_set().entries(*self).finish()
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

impl BitAnd for CardBitSet {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self {
            cards: self.cards & rhs.cards,
        }
    }
}

impl BitAndAssign for CardBitSet {
    fn bitand_assign(&mut self, rhs: Self) {
        self.cards &= rhs.cards;
    }
}

impl Not for CardBitSet {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self {
            cards: !self.cards & FIFTY_TWO_ONES, // Ensure we only keep the first 52 bits
        }
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

#[cfg(feature = "serde")]
impl serde::Serialize for CardBitSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.count()))?;
        for card in (*self).into_iter() {
            seq.serialize_element(&card)?;
        }
        seq.end()
    }
}

#[cfg(feature = "serde")]
struct CardBitSetVisitor;

#[cfg(feature = "serde")]
impl<'de> serde::de::Visitor<'de> for CardBitSetVisitor {
    type Value = CardBitSet;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a sequence of cards")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut deck = CardBitSet::new();
        while let Some(card) = seq.next_element()? {
            deck.insert(card);
        }
        Ok(deck)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for CardBitSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_seq(CardBitSetVisitor)
    }
}

#[cfg(test)]
mod tests {
    use core::panic;
    use std::collections::HashSet;

    use rand::{SeedableRng, rngs::StdRng};

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
    fn test_bit_or() {
        let mut cards = CardBitSet::new();
        cards.insert(Card::from(17));
        cards.insert(Card::from(18));

        let mut cards2 = CardBitSet::new();
        cards2.insert(Card::from(1));
        cards2.insert(Card::from(2));
        cards2.insert(Card::from(17));

        let or = cards | cards2;
        assert_eq!(or.count(), 4);
        assert!(or.contains(Card::from(17)));
        assert!(or.contains(Card::from(18)));
        assert!(or.contains(Card::from(1)));
        assert!(or.contains(Card::from(2)));
        assert!(!or.is_empty());
    }

    #[test]
    fn test_bit_or_assign() {
        let mut cards = CardBitSet::new();
        cards.insert(Card::from(17));
        cards.insert(Card::from(18));

        let mut cards2 = CardBitSet::new();
        cards2.insert(Card::from(1));
        cards2.insert(Card::from(2));
        cards2.insert(Card::from(17));

        cards |= cards2;

        assert_eq!(cards.count(), 4);
        assert!(cards.contains(Card::from(17)));
        assert!(cards.contains(Card::from(18)));
        assert!(cards.contains(Card::from(1)));
        assert!(cards.contains(Card::from(2)));
        assert!(!cards.is_empty());
        assert_eq!(cards, cards | cards2);
    }

    #[test]
    fn test_remove() {
        let mut cards = CardBitSet::new();
        cards.insert(Card::from(17));
        cards.insert(Card::from(18));

        assert!(cards.contains(Card::from(17)));
        cards.remove(Card::from(17));

        assert!(!cards.contains(Card::from(17)));
        assert!(cards.contains(Card::from(18)));
        assert_eq!(1, cards.count());
        assert!(!cards.is_empty());

        cards.remove(Card::from(18));
        assert!(!cards.contains(Card::from(18)));

        // Old cards don't come back
        assert!(!cards.contains(Card::from(17)));
        assert_eq!(0, cards.count());
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

    #[test]
    fn test_formatting_cards() {
        let mut cards = CardBitSet::new();
        cards.insert(Card::new(crate::core::Value::Ace, crate::core::Suit::Club));
        cards.insert(Card::new(
            crate::core::Value::King,
            crate::core::Suit::Diamond,
        ));
        cards.insert(Card::new(
            crate::core::Value::Three,
            crate::core::Suit::Heart,
        ));

        assert_eq!(format!("{cards:?}"), "{Card(Ac), Card(3h), Card(Kd)}");
    }

    #[test]
    fn test_bit_and() {
        let mut cards = CardBitSet::new();
        cards.insert(Card::new(crate::core::Value::Ace, crate::core::Suit::Club));
        cards.insert(Card::new(
            crate::core::Value::King,
            crate::core::Suit::Diamond,
        ));

        let mut cards2 = CardBitSet::new();
        cards2.insert(Card::new(
            crate::core::Value::Three,
            crate::core::Suit::Heart,
        ));
        cards2.insert(Card::new(
            crate::core::Value::King,
            crate::core::Suit::Diamond,
        ));

        let and = cards & cards2;
        assert_eq!(and.count(), 1);

        assert!(and.contains(Card::new(
            crate::core::Value::King,
            crate::core::Suit::Diamond,
        )));
        assert!(!and.contains(Card::new(crate::core::Value::Ace, crate::core::Suit::Club,)));
    }

    #[test]
    fn test_bit_and_assign() {
        let mut cards = CardBitSet::new();
        cards.insert(Card::new(crate::core::Value::Ace, crate::core::Suit::Club));
        cards.insert(Card::new(
            crate::core::Value::King,
            crate::core::Suit::Diamond,
        ));

        let mut cards2 = CardBitSet::new();
        cards2.insert(Card::new(
            crate::core::Value::Three,
            crate::core::Suit::Heart,
        ));
        cards2.insert(Card::new(
            crate::core::Value::King,
            crate::core::Suit::Diamond,
        ));

        cards &= cards2;

        assert_eq!(cards.count(), 1);

        // The shared card
        assert!(cards.contains(Card::new(
            crate::core::Value::King,
            crate::core::Suit::Diamond,
        )));

        // None of the non-shared are there.
        assert!(!cards.contains(Card::new(crate::core::Value::Ace, crate::core::Suit::Club,)));
        assert!(!cards.contains(Card::new(
            crate::core::Value::Three,
            crate::core::Suit::Heart,
        )));
    }

    #[test]
    fn test_bit_xor_assign() {
        let mut cards = CardBitSet::new();
        cards.insert(Card::new(crate::core::Value::Ace, crate::core::Suit::Club));
        cards.insert(Card::new(
            crate::core::Value::King,
            crate::core::Suit::Diamond,
        ));

        let mut cards2 = CardBitSet::new();
        cards2.insert(Card::new(
            crate::core::Value::Three,
            crate::core::Suit::Heart,
        ));
        cards2.insert(Card::new(
            crate::core::Value::King,
            crate::core::Suit::Diamond,
        ));

        cards ^= cards2;

        assert_eq!(cards.count(), 2);

        // These were in both card bit sets
        assert!(cards.contains(Card::new(crate::core::Value::Ace, crate::core::Suit::Club,)));
        assert!(cards.contains(Card::new(
            crate::core::Value::Three,
            crate::core::Suit::Heart,
        )));
    }

    #[test]
    fn test_pick_one() {
        let mut rng = rand::rng();
        let mut cards = CardBitSet::new();

        cards.insert(Card::new(crate::core::Value::Ace, crate::core::Suit::Club));

        let card = cards.sample_one(&mut rng);
        assert!(card.is_some(), "Card should be present");
        assert_eq!(
            card.unwrap(),
            Card::new(crate::core::Value::Ace, crate::core::Suit::Club)
        );
    }

    #[test]
    fn test_pick_one_all() {
        let mut rng = rand::rng();
        let mut cards = CardBitSet::default();

        let mut picked: HashSet<Card> = HashSet::new();

        for _i in 0..10 {
            let card = cards.sample_one(&mut rng);
            if let Some(c) = card {
                cards.remove(c);

                assert!(
                    !picked.contains(&c),
                    "Card already picked: {c:?} picked = {picked:?}"
                );
                picked.insert(c);
            } else {
                panic!("No more cards to pick");
            }
        }
        assert_eq!(cards.count(), 42); // 52 - 10 = 42
    }

    #[test]
    fn test_can_pick_one_for_all() {
        let mut rng = rand::rng();
        let mut cards_one = CardBitSet::default();
        let mut cards_two = CardBitSet::default();

        let mut picked_one = Vec::new();
        let mut picked_two = Vec::new();

        while cards_one.count() > 0 && cards_two.count() > 0 {
            if let Some(card_one) = cards_one.sample_one(&mut rng) {
                picked_one.push(card_one);
                cards_one.remove(card_one);
            }

            if let Some(card_two) = cards_two.sample_one(&mut rng) {
                picked_two.push(card_two);
                cards_two.remove(card_two);
            }
        }

        assert!(cards_one.is_empty(), "Cards one should be empty");
        assert!(cards_two.is_empty(), "Cards two should be empty");

        assert_eq!(picked_one.len(), 52);
        assert_eq!(picked_two.len(), 52);

        assert_ne!(picked_one, picked_two, "Picked cards should be different");

        // Check that all picked cards are unique
        let unique_one: HashSet<_> = picked_one.iter().cloned().collect();
        let unique_two: HashSet<_> = picked_two.iter().cloned().collect();

        assert_eq!(
            unique_one.len(),
            picked_one.len(),
            "Picked cards one should be unique"
        );
        assert_eq!(
            unique_two.len(),
            picked_two.len(),
            "Picked cards two should be unique"
        );
    }

    /// Test From<CardBitSet> for FlatDeck preserves the cards.
    #[test]
    fn test_from_card_bit_set_to_flat_deck() {
        use crate::core::FlatDeck;

        let mut cbs = CardBitSet::new();
        let ace_spade = Card::new(crate::core::Value::Ace, crate::core::Suit::Spade);
        let king_heart = Card::new(crate::core::Value::King, crate::core::Suit::Heart);

        cbs.insert(ace_spade);
        cbs.insert(king_heart);

        let fd: FlatDeck = cbs.into();

        assert_eq!(fd.len(), 2);
    }

    /// Test BitOr<Card> for CardBitSet adds cards correctly.
    #[test]
    fn test_bitor_card() {
        let mut cards = CardBitSet::new();
        let ace = Card::new(crate::core::Value::Ace, crate::core::Suit::Club);
        let king = Card::new(crate::core::Value::King, crate::core::Suit::Diamond);

        cards |= ace;
        assert_eq!(cards.count(), 1);
        assert!(cards.contains(ace));

        cards |= king;
        assert_eq!(cards.count(), 2);
        assert!(cards.contains(ace));
        assert!(cards.contains(king));

        // Oring the same card again shouldn't change count
        cards |= ace;
        assert_eq!(cards.count(), 2);
    }

    /// Test BitXor for CardBitSet returns symmetric difference.
    #[test]
    fn test_bitxor() {
        let mut cards1 = CardBitSet::new();
        let mut cards2 = CardBitSet::new();

        let ace = Card::new(crate::core::Value::Ace, crate::core::Suit::Club);
        let king = Card::new(crate::core::Value::King, crate::core::Suit::Diamond);
        let queen = Card::new(crate::core::Value::Queen, crate::core::Suit::Heart);

        cards1.insert(ace);
        cards1.insert(king);

        cards2.insert(king);
        cards2.insert(queen);

        let xor_result = cards1 ^ cards2;

        // XOR should have ace and queen but NOT king
        assert_eq!(xor_result.count(), 2);
        assert!(xor_result.contains(ace));
        assert!(xor_result.contains(queen));
        assert!(!xor_result.contains(king));
    }

    /// Test BitXorAssign<Card> for CardBitSet toggles card presence.
    #[test]
    fn test_bitxor_assign_card() {
        let mut cards = CardBitSet::new();
        let ace = Card::new(crate::core::Value::Ace, crate::core::Suit::Club);

        // XOR in a card (adds it)
        cards ^= ace;
        assert_eq!(cards.count(), 1);
        assert!(cards.contains(ace));

        // XOR again (removes it)
        cards ^= ace;
        assert_eq!(cards.count(), 0);
        assert!(!cards.contains(ace));

        // XOR again (adds it back)
        cards ^= ace;
        assert_eq!(cards.count(), 1);
        assert!(cards.contains(ace));
    }

    /// Test Not for CardBitSet returns the complement of the set.
    #[test]
    fn test_not() {
        let mut cards = CardBitSet::new();
        let ace = Card::new(crate::core::Value::Ace, crate::core::Suit::Club);
        let king = Card::new(crate::core::Value::King, crate::core::Suit::Diamond);

        cards.insert(ace);
        cards.insert(king);

        let not_cards = !cards;

        // Not should have 50 cards (52 - 2)
        assert_eq!(not_cards.count(), 50);
        assert!(!not_cards.contains(ace));
        assert!(!not_cards.contains(king));

        // All other cards should be present
        let queen = Card::new(crate::core::Value::Queen, crate::core::Suit::Heart);
        assert!(not_cards.contains(queen));
    }

    /// Test serde serialization and deserialization preserves the cards.
    #[cfg(feature = "serde")]
    #[test]
    fn test_serde() {
        let mut cards = CardBitSet::new();
        let ace = Card::new(crate::core::Value::Ace, crate::core::Suit::Club);
        let king = Card::new(crate::core::Value::King, crate::core::Suit::Diamond);

        cards.insert(ace);
        cards.insert(king);

        let json = serde_json::to_string(&cards).unwrap();
        let deserialized: CardBitSet = serde_json::from_str(&json).unwrap();

        assert_eq!(cards, deserialized);
        assert_eq!(deserialized.count(), 2);
        assert!(deserialized.contains(ace));
        assert!(deserialized.contains(king));
    }

    /// Test serde with empty set.
    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_empty() {
        let cards = CardBitSet::new();

        let json = serde_json::to_string(&cards).unwrap();
        let deserialized: CardBitSet = serde_json::from_str(&json).unwrap();

        assert_eq!(cards, deserialized);
        assert!(deserialized.is_empty());
    }

    #[test]
    fn test_nth_set_bit_all_positions() {
        // For a full deck, nth_set_bit(n) should return n
        let full = CardBitSet::default();
        for i in 0..52 {
            assert_eq!(full.nth_set_bit(i), i, "nth_set_bit({i}) on full deck");
        }
    }

    #[test]
    fn test_nth_set_bit_sparse() {
        // Cards at positions 3, 17, 42
        let mut cards = CardBitSet::new();
        cards.insert(Card::from(3));
        cards.insert(Card::from(17));
        cards.insert(Card::from(42));
        assert_eq!(cards.nth_set_bit(0), 3);
        assert_eq!(cards.nth_set_bit(1), 17);
        assert_eq!(cards.nth_set_bit(2), 42);
    }

    #[test]
    fn test_nth_set_bit_single() {
        for pos in 0..52u8 {
            let mut cards = CardBitSet::new();
            cards.insert(Card::from(pos));
            assert_eq!(cards.nth_set_bit(0), pos as u32);
        }
    }

    #[test]
    fn test_nth_set_bit_adjacent() {
        // Two adjacent cards
        let mut cards = CardBitSet::new();
        cards.insert(Card::from(30));
        cards.insert(Card::from(31));
        assert_eq!(cards.nth_set_bit(0), 30);
        assert_eq!(cards.nth_set_bit(1), 31);
    }

    /// Chi-squared test for uniform distribution of sample_one.
    ///
    /// Samples many times from a small set and checks that each card
    /// is selected with approximately equal frequency.
    #[test]
    fn test_sample_one_uniform_distribution() {
        let mut rng = StdRng::seed_from_u64(12345);

        // Create a set with 5 specific cards
        let mut cards = CardBitSet::new();
        let test_cards: Vec<Card> = (0..5).map(Card::from).collect();
        for &c in &test_cards {
            cards.insert(c);
        }

        let num_samples = 50_000;
        let mut counts = [0u32; 5];

        for _ in 0..num_samples {
            let card = cards.sample_one(&mut rng).unwrap();
            let idx = test_cards.iter().position(|&c| c == card).unwrap();
            counts[idx] += 1;
        }

        // Chi-squared test: expected frequency = num_samples / 5
        let expected = num_samples as f64 / 5.0;
        let chi_sq: f64 = counts
            .iter()
            .map(|&c| {
                let diff = c as f64 - expected;
                diff * diff / expected
            })
            .sum();

        // Critical value for chi-squared with 4 df at p=0.001 is 18.47
        assert!(
            chi_sq < 18.47,
            "Chi-squared {chi_sq} exceeds critical value 18.47 (p<0.001). \
             Counts: {counts:?}, expected: {expected}"
        );
    }

    /// Test uniform distribution with a sparse deck (2 cards far apart).
    #[test]
    fn test_sample_one_uniform_sparse() {
        let mut rng = StdRng::seed_from_u64(67890);

        let mut cards = CardBitSet::new();
        cards.insert(Card::from(0)); // Lowest position
        cards.insert(Card::from(51)); // Highest position

        let num_samples = 50_000;
        let mut count_low = 0u32;

        for _ in 0..num_samples {
            let card = cards.sample_one(&mut rng).unwrap();
            if card == Card::from(0) {
                count_low += 1;
            }
        }

        // For 2 cards, each should appear ~50% of the time.
        // Binomial test: z = (observed - expected) / sqrt(n * p * (1-p))
        let expected = num_samples as f64 / 2.0;
        let stddev = (num_samples as f64 * 0.25).sqrt(); // sqrt(n * 0.5 * 0.5)
        let z = (count_low as f64 - expected).abs() / stddev;

        // z > 3.89 corresponds to p < 0.0001
        assert!(
            z < 3.89,
            "z-score {z} exceeds 3.89 (p<0.0001). \
             Low: {count_low}, High: {}, expected: {expected}",
            num_samples - count_low
        );
    }

    /// Test uniform distribution on a full 52-card deck.
    #[test]
    fn test_sample_one_uniform_full_deck() {
        let mut rng = StdRng::seed_from_u64(11111);

        let cards = CardBitSet::default();
        let num_samples = 520_000; // 10,000 per card
        let mut counts = [0u32; 52];

        for _ in 0..num_samples {
            let card = cards.sample_one(&mut rng).unwrap();
            counts[u8::from(card) as usize] += 1;
        }

        let expected = num_samples as f64 / 52.0;
        let chi_sq: f64 = counts
            .iter()
            .map(|&c| {
                let diff = c as f64 - expected;
                diff * diff / expected
            })
            .sum();

        // Critical value for chi-squared with 51 df at p=0.001 is 82.29
        assert!(
            chi_sq < 82.29,
            "Chi-squared {chi_sq} exceeds critical value 82.29 (p<0.001). \
             Min count: {}, Max count: {}, expected: {expected}",
            counts.iter().min().unwrap(),
            counts.iter().max().unwrap()
        );
    }
}
