use std::ops::{BitAnd, BitAndAssign};

use super::{Card, CardBitSet, CardBitSetIter, RSPokerError, Suit, Value};

#[derive(Debug, Clone, PartialEq, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct Hand(CardBitSet);

impl Hand {
    /// Create a new empty hand
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::core::Hand;
    ///
    /// let hand = Hand::new();
    ///
    /// assert!(hand.is_empty());
    /// ```
    pub fn new() -> Self {
        Self(CardBitSet::new())
    }

    pub fn new_with_cards(cards: Vec<Card>) -> Self {
        let mut bitset = CardBitSet::new();
        for card in cards {
            bitset.insert(card);
        }
        Self(bitset)
    }

    /// Given a card, is it in the current hand?
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::core::{Card, Hand, Suit, Value};
    ///
    /// let mut hand = Hand::new();
    ///
    /// let card = Card::new(Value::Ace, Suit::Club);
    /// assert!(!hand.contains(&card));
    ///
    /// hand.insert(card);
    /// assert!(hand.contains(&card));
    /// ```
    pub fn contains(&self, c: &Card) -> bool {
        self.0.contains(*c)
    }

    /// Remove a card from the hand
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::core::{Card, Hand, Suit, Value};
    ///
    /// let mut hand = Hand::new();
    ///
    /// let card = Card::new(Value::Ace, Suit::Club);
    /// assert!(!hand.contains(&card));
    ///
    /// hand.insert(card);
    /// assert!(hand.contains(&card));
    ///
    /// hand.remove(&card);
    /// assert!(!hand.contains(&card));
    /// ```
    pub fn remove(&mut self, c: &Card) -> bool {
        let contains = self.contains(c);
        self.0.remove(*c);
        contains
    }

    pub fn insert(&mut self, c: Card) -> bool {
        let contains = self.contains(&c);
        self.0.insert(c);
        !contains
    }

    pub fn count(&self) -> usize {
        self.0.count()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> CardBitSetIter {
        self.0.into_iter()
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn new_from_str(hand_string: &str) -> Result<Self, RSPokerError> {
        let mut chars = hand_string.chars();
        let mut bitset = CardBitSet::new();

        // Keep looping until we explicitly break
        loop {
            let vco = chars.next();
            if vco.is_none() {
                break;
            } else {
                let sco = chars.next();
                let v = vco
                    .and_then(Value::from_char)
                    .ok_or(RSPokerError::UnexpectedValueChar)?;
                let s = sco
                    .and_then(Suit::from_char)
                    .ok_or(RSPokerError::UnexpectedSuitChar)?;

                let c = Card { value: v, suit: s };

                if bitset.contains(c) {
                    return Err(RSPokerError::DuplicateCardInHand(c));
                } else {
                    bitset.insert(c);
                }
            }
        }

        if chars.next().is_some() {
            return Err(RSPokerError::UnparsedCharsRemaining);
        }

        Ok(Self(bitset))
    }
}

impl Default for Hand {
    fn default() -> Self {
        Self(CardBitSet::new())
    }
}

impl Extend<Card> for Hand {
    fn extend<T: IntoIterator<Item = Card>>(&mut self, iter: T) {
        for card in iter {
            self.insert(card);
        }
    }
}

impl BitAnd for Hand {
    type Output = Hand;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for Hand {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl From<Hand> for CardBitSet {
    fn from(val: Hand) -> Self {
        val.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert() {
        let mut hand = Hand::new();
        for i in 1..7 {
            let c = Card::from(i);
            assert!(hand.insert(c));
            assert!(hand.contains(&c));
            assert_eq!(hand.count(), usize::from(i));
        }
    }

    #[test]
    fn test_is_empty() {
        let mut hand = Hand::new();
        assert!(hand.is_empty());

        hand.insert(Card::from(1));
        assert!(!hand.is_empty());
        hand.clear();

        assert!(hand.is_empty());
        hand.insert(Card::from(2));
        assert!(!hand.is_empty());
    }

    #[test]
    fn test_bit_and() {
        let mut hand1 = Hand::new();
        let mut hand2 = Hand::new();

        for i in 1..7 {
            let c = Card::from(i);
            hand1.insert(c);
        }

        for i in 4..10 {
            let c = Card::from(i);
            hand2.insert(c);
        }
        let hand3 = hand1 & hand2;
        assert_eq!(hand3.count(), 3);
        for i in 4..7 {
            let c = Card::from(i);
            assert!(hand3.contains(&c));
        }
        for i in 1..4 {
            let c = Card::from(i);
            assert!(!hand3.contains(&c));
        }
    }
}
