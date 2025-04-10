use crate::core::card::{Card, Suit, Value};
use std::ops::Index;
use std::ops::{RangeFrom, RangeFull, RangeTo};
use std::slice::Iter;

use super::RSPokerError;

/// Struct to hold cards.
///
/// This doesn't have the ability to easily check if a card is
/// in the hand. So do that before adding/removing a card.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct FlatHand(Vec<Card>);

impl FlatHand {
    /// Create the hand with specific hand.
    pub fn new_with_cards(cards: Vec<Card>) -> Self {
        Self(cards)
    }
    /// From a str create a new hand.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::core::FlatHand;
    /// let hand = FlatHand::new_from_str("AdKd").unwrap();
    /// ```
    ///
    /// Anything that can't be parsed will return an error.
    ///
    /// ```
    /// use rs_poker::core::FlatHand;
    /// let hand = FlatHand::new_from_str("AdKx");
    /// assert!(hand.is_err());
    /// ```
    pub fn new_from_str(hand_string: &str) -> Result<Self, RSPokerError> {
        // Get the chars iterator.
        let mut chars = hand_string.chars();
        // Where we will put the cards
        //
        // We make the assumption that the hands will have 2 plus five cards.
        let mut cards: Vec<Card> = Vec::with_capacity(7);

        // Keep looping until we explicitly break
        loop {
            // Now try and get a char.
            let vco = chars.next();
            // If there was no char then we are done.
            if vco.is_none() {
                break;
            } else {
                // If we got a value char then we should get a
                // suit.
                let sco = chars.next();
                // Now try and parse the two chars that we have.
                let v = vco
                    .and_then(Value::from_char)
                    .ok_or(RSPokerError::UnexpectedValueChar)?;
                let s = sco
                    .and_then(Suit::from_char)
                    .ok_or(RSPokerError::UnexpectedSuitChar)?;

                let c = Card { value: v, suit: s };

                match cards.binary_search(&c) {
                    Ok(_) => return Err(RSPokerError::DuplicateCardInHand(c)),
                    Err(i) => cards.insert(i, c),
                };
            }
        }

        if chars.next().is_some() {
            return Err(RSPokerError::UnparsedCharsRemaining);
        }

        cards.reserve(7);
        Ok(Self(cards))
    }
    /// Add card at to the hand.
    /// No verification is done at all.
    pub fn push(&mut self, c: Card) {
        self.0.push(c);
    }
    /// Truncate the hand to the given number of cards.
    pub fn truncate(&mut self, len: usize) {
        self.0.truncate(len);
    }
    /// How many cards are in this hand so far ?
    pub fn len(&self) -> usize {
        self.0.len()
    }
    /// Are there any cards at all ?
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    /// Create an iter on the cards.
    pub fn iter(&self) -> Iter<Card> {
        self.0.iter()
    }
}

impl Default for FlatHand {
    /// Create the default empty hand.
    fn default() -> Self {
        Self(Vec::with_capacity(7))
    }
}

/// Allow indexing into the hand.
impl Index<usize> for FlatHand {
    type Output = Card;
    fn index(&self, index: usize) -> &Card {
        &self.0[index]
    }
}

/// Allow the index to get refernce to every card.
impl Index<RangeFull> for FlatHand {
    type Output = [Card];
    fn index(&self, range: RangeFull) -> &[Card] {
        &self.0[range]
    }
}

impl Index<RangeTo<usize>> for FlatHand {
    type Output = [Card];
    fn index(&self, index: RangeTo<usize>) -> &[Card] {
        &self.0[index]
    }
}
impl Index<RangeFrom<usize>> for FlatHand {
    type Output = [Card];
    fn index(&self, index: RangeFrom<usize>) -> &[Card] {
        &self.0[index]
    }
}

impl Extend<Card> for FlatHand {
    fn extend<T: IntoIterator<Item = Card>>(&mut self, iter: T) {
        self.0.extend(iter);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_card() {
        let mut h = FlatHand::default();
        let c = Card {
            value: Value::Three,
            suit: Suit::Spade,
        };
        h.push(c);
        // Make sure that the card was added to the vec.
        //
        // This will also test that has len works
        assert_eq!(1, h.len());
    }

    #[test]
    fn test_index() {
        let mut h = FlatHand::default();
        h.push(Card {
            value: Value::Four,
            suit: Suit::Spade,
        });
        // Make sure the card is there
        assert_eq!(
            Card {
                value: Value::Four,
                suit: Suit::Spade,
            },
            h[0]
        );
    }
    #[test]
    fn test_parse_error() {
        assert!(FlatHand::new_from_str("BAD").is_err());
        assert!(FlatHand::new_from_str("Adx").is_err());
    }

    #[test]
    fn test_parse_one_hand() {
        let h = FlatHand::new_from_str("Ad").unwrap();
        assert_eq!(1, h.len())
    }

    #[test]
    fn test_parse_empty() {
        let h = FlatHand::new_from_str("").unwrap();
        assert!(h.is_empty());
    }

    #[test]
    fn test_new_with_cards() {
        let h = FlatHand::new_with_cards(vec![
            Card::new(Value::Jack, Suit::Spade),
            Card::new(Value::Jack, Suit::Heart),
        ]);

        assert_eq!(2, h.len());
    }

    #[test]
    fn test_error_on_duplicate_card() {
        assert!(FlatHand::new_from_str("AdAd").is_err());
    }

    #[test]
    fn test_deterministic_new_from_str() {
        let h = FlatHand::new_from_str("AdKd").unwrap();

        assert_eq!(h, FlatHand::new_from_str("AdKd").unwrap());
        assert_eq!(h, FlatHand::new_from_str("AdKd").unwrap());
        assert_eq!(h, FlatHand::new_from_str("AdKd").unwrap());
        assert_eq!(h, FlatHand::new_from_str("AdKd").unwrap());
        assert_eq!(h, FlatHand::new_from_str("AdKd").unwrap());
        assert_eq!(h, FlatHand::new_from_str("AdKd").unwrap());
    }
}
