//! Pre-flop chart system for Texas Hold'em.
//!
//! This module provides types for representing GTO-style action frequencies
//! for the 169 unique starting hands in Texas Hold'em.

use std::collections::HashMap;
use std::fmt;

use crate::core::{Hand, RSPokerError, Value};

/// A lightweight key for the 13x13 matrix of pre-flop hands.
///
/// Represents one of the 169 unique pre-flop starting hands:
/// - 13 pocket pairs (e.g., AA, KK, 22)
/// - 78 suited hands (e.g., AKs, T9s)
/// - 78 offsuit hands (e.g., AKo, 72o)
///
/// # Examples
///
/// ```
/// use rs_poker::holdem::PreflopHand;
/// use rs_poker::core::Value;
///
/// // Create from values
/// let aks = PreflopHand::new(Value::Ace, Value::King, true);
/// assert_eq!(aks.to_notation(), "AKs");
///
/// // Create from notation
/// let ako = PreflopHand::from_notation("AKo").unwrap();
/// assert!(!ako.suited());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "String", into = "String"))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct PreflopHand {
    /// Higher or equal value card
    high: Value,
    /// Lower or equal value card
    low: Value,
    /// true = suited, false = offsuit (pairs always false)
    suited: bool,
}

impl PreflopHand {
    /// Create a new PreflopHand from two values and suitedness.
    ///
    /// Values are automatically ordered so that `high >= low`.
    /// Pairs always have `suited = false` regardless of the input.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::holdem::PreflopHand;
    /// use rs_poker::core::Value;
    ///
    /// // Values are auto-ordered
    /// let hand1 = PreflopHand::new(Value::King, Value::Ace, true);
    /// let hand2 = PreflopHand::new(Value::Ace, Value::King, true);
    /// assert_eq!(hand1, hand2);
    ///
    /// // Pairs can't be suited
    /// let pair = PreflopHand::new(Value::Ace, Value::Ace, true);
    /// assert!(!pair.suited());
    /// ```
    pub fn new(v1: Value, v2: Value, suited: bool) -> Self {
        let (high, low) = if v1 >= v2 { (v1, v2) } else { (v2, v1) };
        // Pairs can't be suited
        let suited = if high == low { false } else { suited };
        Self { high, low, suited }
    }

    /// Returns true if this is a pocket pair.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::holdem::PreflopHand;
    /// use rs_poker::core::Value;
    ///
    /// let pair = PreflopHand::new(Value::Ace, Value::Ace, false);
    /// assert!(pair.is_pair());
    ///
    /// let non_pair = PreflopHand::new(Value::Ace, Value::King, true);
    /// assert!(!non_pair.is_pair());
    /// ```
    pub fn is_pair(&self) -> bool {
        self.high == self.low
    }

    /// Returns true if this hand is suited.
    ///
    /// Pairs always return false.
    pub fn suited(&self) -> bool {
        self.suited
    }

    /// Returns the high card value.
    pub fn high(&self) -> Value {
        self.high
    }

    /// Returns the low card value.
    pub fn low(&self) -> Value {
        self.low
    }

    /// Convert to standard notation string.
    ///
    /// Format:
    /// - Pairs: "AA", "KK", "22"
    /// - Suited: "AKs", "T9s"
    /// - Offsuit: "AKo", "72o"
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::holdem::PreflopHand;
    /// use rs_poker::core::Value;
    ///
    /// let aces = PreflopHand::new(Value::Ace, Value::Ace, false);
    /// assert_eq!(aces.to_notation(), "AA");
    ///
    /// let aks = PreflopHand::new(Value::Ace, Value::King, true);
    /// assert_eq!(aks.to_notation(), "AKs");
    ///
    /// let ako = PreflopHand::new(Value::Ace, Value::King, false);
    /// assert_eq!(ako.to_notation(), "AKo");
    /// ```
    pub fn to_notation(&self) -> String {
        let high_char = self.high.to_char();
        let low_char = self.low.to_char();

        if self.is_pair() {
            format!("{}{}", high_char, low_char)
        } else if self.suited {
            format!("{}{}s", high_char, low_char)
        } else {
            format!("{}{}o", high_char, low_char)
        }
    }

    /// Parse from notation string.
    ///
    /// Accepts formats:
    /// - Pairs: "AA", "KK", "22"
    /// - Suited: "AKs", "T9s"
    /// - Offsuit: "AKo", "72o"
    ///
    /// # Errors
    ///
    /// Returns `RSPokerError::InvalidPreflopNotation` if the notation is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::holdem::PreflopHand;
    /// use rs_poker::core::Value;
    ///
    /// let aces = PreflopHand::from_notation("AA").unwrap();
    /// assert!(aces.is_pair());
    /// assert_eq!(aces.high(), Value::Ace);
    ///
    /// let aks = PreflopHand::from_notation("AKs").unwrap();
    /// assert!(aks.suited());
    ///
    /// let ako = PreflopHand::from_notation("AKo").unwrap();
    /// assert!(!ako.suited());
    /// ```
    pub fn from_notation(s: &str) -> Result<Self, RSPokerError> {
        let chars: Vec<char> = s.chars().collect();

        if chars.len() < 2 || chars.len() > 3 {
            return Err(RSPokerError::InvalidPreflopNotation(s.to_string()));
        }

        let v1 = Value::from_char(chars[0])
            .ok_or_else(|| RSPokerError::InvalidPreflopNotation(s.to_string()))?;
        let v2 = Value::from_char(chars[1])
            .ok_or_else(|| RSPokerError::InvalidPreflopNotation(s.to_string()))?;

        let suited = if chars.len() == 2 {
            // Must be a pair for 2-char notation
            if v1 != v2 {
                return Err(RSPokerError::InvalidPreflopNotation(s.to_string()));
            }
            false
        } else {
            match chars[2].to_ascii_lowercase() {
                's' => {
                    // Can't have suited pair
                    if v1 == v2 {
                        return Err(RSPokerError::InvalidPreflopNotation(s.to_string()));
                    }
                    true
                }
                'o' => false,
                _ => return Err(RSPokerError::InvalidPreflopNotation(s.to_string())),
            }
        };

        Ok(Self::new(v1, v2, suited))
    }

    /// Generate all 169 unique pre-flop hands.
    ///
    /// Returns a vector containing:
    /// - 13 pocket pairs
    /// - 78 suited hands
    /// - 78 offsuit hands
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::holdem::PreflopHand;
    ///
    /// let all_hands = PreflopHand::all();
    /// assert_eq!(all_hands.len(), 169);
    /// ```
    pub fn all() -> Vec<Self> {
        let mut hands = Vec::with_capacity(169);
        let values = Value::values();

        for (i, &high) in values.iter().enumerate() {
            for &low in &values[..=i] {
                // Always add offsuit (or pair for same values)
                hands.push(Self::new(high, low, false));

                // Add suited for non-pairs
                if high != low {
                    hands.push(Self::new(high, low, true));
                }
            }
        }

        hands
    }
}

impl fmt::Display for PreflopHand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_notation())
    }
}

impl TryFrom<String> for PreflopHand {
    type Error = RSPokerError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::from_notation(&value)
    }
}

impl From<PreflopHand> for String {
    fn from(hand: PreflopHand) -> Self {
        hand.to_notation()
    }
}

/// Convert a Hand to a PreflopHand.
///
/// The hand must contain exactly 2 cards.
///
/// # Errors
///
/// Returns `RSPokerError::InvalidPreflopHandSize` if the hand doesn't have exactly 2 cards.
///
/// # Examples
///
/// ```
/// use rs_poker::holdem::PreflopHand;
/// use rs_poker::core::{Card, Hand, Value, Suit};
/// use std::convert::TryFrom;
///
/// let mut hand = Hand::new();
/// hand.insert(Card::new(Value::Ace, Suit::Spade));
/// hand.insert(Card::new(Value::King, Suit::Spade));
///
/// let preflop = PreflopHand::try_from(&hand).unwrap();
/// assert_eq!(preflop.to_notation(), "AKs");
/// ```
impl TryFrom<&Hand> for PreflopHand {
    type Error = RSPokerError;

    fn try_from(hand: &Hand) -> Result<Self, Self::Error> {
        let count = hand.count();
        if count != 2 {
            return Err(RSPokerError::InvalidPreflopHandSize(count));
        }

        let mut iter = hand.iter();
        let c1 = iter
            .next()
            .ok_or(RSPokerError::InvalidPreflopHandSize(count))?;
        let c2 = iter
            .next()
            .ok_or(RSPokerError::InvalidPreflopHandSize(count))?;

        let suited = c1.suit == c2.suit;
        Ok(Self::new(c1.value, c2.value, suited))
    }
}

/// Pre-flop action types.
///
/// These represent the common pre-flop decisions. Bet sizing is not included
/// as it's determined by context or separate configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum PreflopActionType {
    /// Fold the hand (or check when no bet to call)
    Fold,
    /// Call/limp (match the current bet)
    Call,
    /// Open raise or standard raise
    Raise,
    /// Re-raise (raise over a raise)
    ThreeBet,
    /// Re-raise over a 3-bet
    FourBet,
}

/// Stores action frequencies for a pre-flop hand.
///
/// Frequencies must sum to 1.0 (within floating point tolerance).
/// Only non-zero actions are stored.
///
/// # Examples
///
/// ```
/// use rs_poker::holdem::{PreflopStrategy, PreflopActionType};
///
/// // Pure strategy (100% one action)
/// let always_raise = PreflopStrategy::pure(PreflopActionType::Raise);
/// assert_eq!(always_raise.frequency(PreflopActionType::Raise), 1.0);
///
/// // Mixed strategy
/// let mixed = PreflopStrategy::new(vec![
///     (PreflopActionType::Raise, 0.7),
///     (PreflopActionType::Call, 0.3),
/// ]).unwrap();
/// assert_eq!(mixed.frequency(PreflopActionType::Raise), 0.7);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct PreflopStrategy {
    frequencies: Vec<(PreflopActionType, f32)>,
}

impl PreflopStrategy {
    /// Create a new strategy with validated frequencies.
    ///
    /// Frequencies must sum to 1.0 (within 0.001 tolerance).
    ///
    /// # Errors
    ///
    /// Returns `RSPokerError::InvalidStrategyFrequencies` if frequencies don't sum to 1.0.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::holdem::{PreflopStrategy, PreflopActionType};
    ///
    /// let strategy = PreflopStrategy::new(vec![
    ///     (PreflopActionType::Raise, 0.6),
    ///     (PreflopActionType::Call, 0.3),
    ///     (PreflopActionType::Fold, 0.1),
    /// ]).unwrap();
    /// ```
    pub fn new(frequencies: Vec<(PreflopActionType, f32)>) -> Result<Self, RSPokerError> {
        let sum: f32 = frequencies.iter().map(|(_, f)| f).sum();

        // Allow small floating point tolerance
        if (sum - 1.0).abs() > 0.001 {
            return Err(RSPokerError::InvalidStrategyFrequencies(sum.to_string()));
        }

        // Filter out zero frequencies
        let frequencies: Vec<_> = frequencies.into_iter().filter(|(_, f)| *f > 0.0).collect();

        Ok(Self { frequencies })
    }

    /// Create a pure strategy (100% one action).
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::holdem::{PreflopStrategy, PreflopActionType};
    ///
    /// let always_fold = PreflopStrategy::pure(PreflopActionType::Fold);
    /// assert_eq!(always_fold.frequency(PreflopActionType::Fold), 1.0);
    /// ```
    pub fn pure(action: PreflopActionType) -> Self {
        Self {
            frequencies: vec![(action, 1.0)],
        }
    }

    /// Create a fold-only strategy.
    ///
    /// Convenience method equivalent to `pure(PreflopActionType::Fold)`.
    pub fn fold() -> Self {
        Self::pure(PreflopActionType::Fold)
    }

    /// Get the frequency for a specific action.
    ///
    /// Returns 0.0 if the action is not in the strategy.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::holdem::{PreflopStrategy, PreflopActionType};
    ///
    /// let strategy = PreflopStrategy::new(vec![
    ///     (PreflopActionType::Raise, 0.8),
    ///     (PreflopActionType::Fold, 0.2),
    /// ]).unwrap();
    ///
    /// assert_eq!(strategy.frequency(PreflopActionType::Raise), 0.8);
    /// assert_eq!(strategy.frequency(PreflopActionType::Call), 0.0);
    /// ```
    pub fn frequency(&self, action: PreflopActionType) -> f32 {
        self.frequencies
            .iter()
            .find(|(a, _)| *a == action)
            .map(|(_, f)| *f)
            .unwrap_or(0.0)
    }

    /// Get all non-zero action frequencies.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::holdem::{PreflopStrategy, PreflopActionType};
    ///
    /// let strategy = PreflopStrategy::pure(PreflopActionType::Raise);
    /// let freqs = strategy.frequencies();
    /// assert_eq!(freqs.len(), 1);
    /// assert_eq!(freqs[0], (PreflopActionType::Raise, 1.0));
    /// ```
    pub fn frequencies(&self) -> &[(PreflopActionType, f32)] {
        &self.frequencies
    }

    /// Sample an action based on frequencies using the provided random value.
    ///
    /// The `random_value` should be in the range [0.0, 1.0).
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::holdem::{PreflopStrategy, PreflopActionType};
    ///
    /// let strategy = PreflopStrategy::new(vec![
    ///     (PreflopActionType::Raise, 0.7),
    ///     (PreflopActionType::Call, 0.3),
    /// ]).unwrap();
    ///
    /// // Values 0.0..0.7 will return Raise
    /// assert_eq!(strategy.sample(0.0), PreflopActionType::Raise);
    /// assert_eq!(strategy.sample(0.69), PreflopActionType::Raise);
    ///
    /// // Values 0.7..1.0 will return Call
    /// assert_eq!(strategy.sample(0.7), PreflopActionType::Call);
    /// assert_eq!(strategy.sample(0.99), PreflopActionType::Call);
    /// ```
    pub fn sample(&self, random_value: f32) -> PreflopActionType {
        let mut cumulative = 0.0;

        for (action, freq) in &self.frequencies {
            cumulative += freq;
            if random_value < cumulative {
                return *action;
            }
        }

        // Return last action if we somehow reach here (floating point edge case)
        self.frequencies
            .last()
            .map(|(a, _)| *a)
            .unwrap_or(PreflopActionType::Fold)
    }
}

impl Default for PreflopStrategy {
    fn default() -> Self {
        Self::fold()
    }
}

/// A pre-flop chart mapping hands to strategies.
///
/// Hands not in the chart are implicitly 100% fold.
///
/// # Examples
///
/// ```
/// use rs_poker::holdem::{PreflopChart, PreflopHand, PreflopStrategy, PreflopActionType};
/// use rs_poker::core::Value;
///
/// let mut chart = PreflopChart::new();
///
/// // Set strategy for AA
/// let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
/// chart.set(aa, PreflopStrategy::pure(PreflopActionType::Raise));
///
/// // Get strategy
/// assert!(chart.get(&aa).is_some());
/// assert_eq!(chart.get(&aa).unwrap().frequency(PreflopActionType::Raise), 1.0);
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct PreflopChart {
    strategies: HashMap<PreflopHand, PreflopStrategy>,
}

impl PreflopChart {
    /// Create a new empty pre-flop chart.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the strategy for a hand.
    ///
    /// Returns `None` if no strategy is set for this hand.
    pub fn get(&self, hand: &PreflopHand) -> Option<&PreflopStrategy> {
        self.strategies.get(hand)
    }

    /// Get the strategy for a hand, returning fold strategy if not found.
    ///
    /// This is useful when you always want a valid strategy back.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::holdem::{PreflopChart, PreflopHand, PreflopActionType};
    /// use rs_poker::core::Value;
    ///
    /// let chart = PreflopChart::new();
    /// let unknown = PreflopHand::new(Value::Seven, Value::Two, false);
    ///
    /// // Returns fold strategy for unknown hands
    /// let strategy = chart.get_or_fold(&unknown);
    /// assert_eq!(strategy.frequency(PreflopActionType::Fold), 1.0);
    /// ```
    pub fn get_or_fold(&self, hand: &PreflopHand) -> PreflopStrategy {
        self.strategies
            .get(hand)
            .cloned()
            .unwrap_or_else(PreflopStrategy::fold)
    }

    /// Set the strategy for a hand.
    pub fn set(&mut self, hand: PreflopHand, strategy: PreflopStrategy) {
        self.strategies.insert(hand, strategy);
    }

    /// Remove the strategy for a hand.
    ///
    /// Returns the removed strategy if it existed.
    pub fn remove(&mut self, hand: &PreflopHand) -> Option<PreflopStrategy> {
        self.strategies.remove(hand)
    }

    /// Iterate over all hands and their strategies.
    pub fn iter(&self) -> impl Iterator<Item = (&PreflopHand, &PreflopStrategy)> {
        self.strategies.iter()
    }

    /// Returns the number of hands with explicit strategies.
    pub fn len(&self) -> usize {
        self.strategies.len()
    }

    /// Returns true if no hands have explicit strategies.
    pub fn is_empty(&self) -> bool {
        self.strategies.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Card, Suit};

    // ========== PreflopHand tests ==========

    #[test]
    fn test_preflop_hand_ordering() {
        // Values should be auto-sorted so high >= low
        let hand1 = PreflopHand::new(Value::King, Value::Ace, true);
        let hand2 = PreflopHand::new(Value::Ace, Value::King, true);
        assert_eq!(hand1, hand2);
        assert_eq!(hand1.high(), Value::Ace);
        assert_eq!(hand1.low(), Value::King);
    }

    #[test]
    fn test_preflop_hand_try_from_hand() {
        // Suited hand
        let mut cards = Hand::new();
        cards.insert(Card::new(Value::Ace, Suit::Spade));
        cards.insert(Card::new(Value::King, Suit::Spade));
        let preflop = PreflopHand::try_from(&cards).expect("valid 2-card hand");
        assert!(preflop.suited());
        assert_eq!(preflop.high(), Value::Ace);
        assert_eq!(preflop.low(), Value::King);

        // Offsuit hand
        let mut cards = Hand::new();
        cards.insert(Card::new(Value::Ace, Suit::Spade));
        cards.insert(Card::new(Value::King, Suit::Heart));
        let preflop = PreflopHand::try_from(&cards).expect("valid 2-card hand");
        assert!(!preflop.suited());

        // Pair (both same value)
        let mut cards = Hand::new();
        cards.insert(Card::new(Value::Queen, Suit::Spade));
        cards.insert(Card::new(Value::Queen, Suit::Heart));
        let preflop = PreflopHand::try_from(&cards).expect("valid 2-card hand");
        assert!(preflop.is_pair());
        assert!(!preflop.suited()); // Pairs can't be suited
    }

    #[test]
    fn test_preflop_hand_try_from_invalid_size() {
        // Too few cards
        let cards = Hand::new();
        assert!(PreflopHand::try_from(&cards).is_err());

        // One card
        let mut cards = Hand::new();
        cards.insert(Card::new(Value::Ace, Suit::Spade));
        assert!(PreflopHand::try_from(&cards).is_err());

        // Three cards
        let mut cards = Hand::new();
        cards.insert(Card::new(Value::Ace, Suit::Spade));
        cards.insert(Card::new(Value::King, Suit::Spade));
        cards.insert(Card::new(Value::Queen, Suit::Spade));
        assert!(PreflopHand::try_from(&cards).is_err());
    }

    #[test]
    fn test_preflop_hand_is_pair() {
        let pair = PreflopHand::new(Value::Ace, Value::Ace, false);
        assert!(pair.is_pair());

        let non_pair = PreflopHand::new(Value::Ace, Value::King, true);
        assert!(!non_pair.is_pair());
    }

    #[test]
    fn test_preflop_hand_pair_not_suited() {
        // Even if you pass suited=true for a pair, it should be false
        let pair = PreflopHand::new(Value::King, Value::King, true);
        assert!(!pair.suited());
    }

    #[test]
    fn test_preflop_hand_all_count() {
        let all = PreflopHand::all();
        assert_eq!(all.len(), 169);

        // Count pairs, suited, and offsuit
        let pairs = all.iter().filter(|h| h.is_pair()).count();
        let suited = all.iter().filter(|h| h.suited()).count();
        let offsuit = all.iter().filter(|h| !h.is_pair() && !h.suited()).count();

        assert_eq!(pairs, 13);
        assert_eq!(suited, 78);
        assert_eq!(offsuit, 78);
    }

    // ========== Notation tests ==========

    #[test]
    fn test_notation_pairs() {
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        assert_eq!(aa.to_notation(), "AA");

        let twos = PreflopHand::new(Value::Two, Value::Two, false);
        assert_eq!(twos.to_notation(), "22");

        // Roundtrip
        let parsed = PreflopHand::from_notation("KK").unwrap();
        assert_eq!(parsed.high(), Value::King);
        assert_eq!(parsed.low(), Value::King);
        assert!(parsed.is_pair());
    }

    #[test]
    fn test_notation_suited() {
        let aks = PreflopHand::new(Value::Ace, Value::King, true);
        assert_eq!(aks.to_notation(), "AKs");

        let t9s = PreflopHand::new(Value::Ten, Value::Nine, true);
        assert_eq!(t9s.to_notation(), "T9s");

        // Roundtrip
        let parsed = PreflopHand::from_notation("QJs").unwrap();
        assert!(parsed.suited());
        assert_eq!(parsed.high(), Value::Queen);
        assert_eq!(parsed.low(), Value::Jack);
    }

    #[test]
    fn test_notation_offsuit() {
        let ako = PreflopHand::new(Value::Ace, Value::King, false);
        assert_eq!(ako.to_notation(), "AKo");

        let seven_two = PreflopHand::new(Value::Seven, Value::Two, false);
        assert_eq!(seven_two.to_notation(), "72o");

        // Roundtrip
        let parsed = PreflopHand::from_notation("A5o").unwrap();
        assert!(!parsed.suited());
        assert_eq!(parsed.high(), Value::Ace);
        assert_eq!(parsed.low(), Value::Five);
    }

    #[test]
    fn test_notation_roundtrip() {
        // Test all 169 hands
        for hand in PreflopHand::all() {
            let notation = hand.to_notation();
            let parsed = PreflopHand::from_notation(&notation).unwrap();
            assert_eq!(hand, parsed, "Failed roundtrip for {}", notation);
        }
    }

    #[test]
    fn test_notation_case_insensitive() {
        let parsed_lower = PreflopHand::from_notation("aks").unwrap();
        let parsed_upper = PreflopHand::from_notation("AKS").unwrap();
        let parsed_mixed = PreflopHand::from_notation("AkS").unwrap();

        assert_eq!(parsed_lower, parsed_upper);
        assert_eq!(parsed_lower, parsed_mixed);
    }

    #[test]
    fn test_notation_invalid() {
        // Too short
        assert!(PreflopHand::from_notation("A").is_err());

        // Too long
        assert!(PreflopHand::from_notation("AKso").is_err());

        // Invalid value chars
        assert!(PreflopHand::from_notation("XKs").is_err());

        // Non-pair without suitedness indicator
        assert!(PreflopHand::from_notation("AK").is_err());

        // Suited pair (invalid)
        assert!(PreflopHand::from_notation("AAs").is_err());

        // Invalid suitedness char
        assert!(PreflopHand::from_notation("AKx").is_err());
    }

    // ========== PreflopStrategy tests ==========

    #[test]
    fn test_strategy_pure() {
        let strategy = PreflopStrategy::pure(PreflopActionType::Raise);
        assert_eq!(strategy.frequency(PreflopActionType::Raise), 1.0);
        assert_eq!(strategy.frequency(PreflopActionType::Fold), 0.0);
        assert_eq!(strategy.frequencies().len(), 1);
    }

    #[test]
    fn test_strategy_fold() {
        let strategy = PreflopStrategy::fold();
        assert_eq!(strategy.frequency(PreflopActionType::Fold), 1.0);
    }

    #[test]
    fn test_strategy_mixed() {
        let strategy = PreflopStrategy::new(vec![
            (PreflopActionType::Raise, 0.6),
            (PreflopActionType::Call, 0.3),
            (PreflopActionType::Fold, 0.1),
        ])
        .unwrap();

        assert_eq!(strategy.frequency(PreflopActionType::Raise), 0.6);
        assert_eq!(strategy.frequency(PreflopActionType::Call), 0.3);
        assert_eq!(strategy.frequency(PreflopActionType::Fold), 0.1);
        assert_eq!(strategy.frequencies().len(), 3);
    }

    #[test]
    fn test_strategy_invalid_sum_too_low() {
        let result = PreflopStrategy::new(vec![
            (PreflopActionType::Raise, 0.5),
            (PreflopActionType::Call, 0.3),
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_strategy_invalid_sum_too_high() {
        let result = PreflopStrategy::new(vec![
            (PreflopActionType::Raise, 0.7),
            (PreflopActionType::Call, 0.5),
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_strategy_tolerance() {
        // Should accept small floating point errors
        let strategy = PreflopStrategy::new(vec![
            (PreflopActionType::Raise, 0.333),
            (PreflopActionType::Call, 0.333),
            (PreflopActionType::Fold, 0.334),
        ])
        .unwrap();

        // Sum is 1.0 within tolerance
        let total: f32 = strategy.frequencies().iter().map(|(_, f)| f).sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_strategy_filters_zero() {
        let strategy = PreflopStrategy::new(vec![
            (PreflopActionType::Raise, 1.0),
            (PreflopActionType::Call, 0.0),
            (PreflopActionType::Fold, 0.0),
        ])
        .unwrap();

        // Zero frequencies should be filtered out
        assert_eq!(strategy.frequencies().len(), 1);
    }

    #[test]
    fn test_strategy_sample() {
        let strategy = PreflopStrategy::new(vec![
            (PreflopActionType::Raise, 0.5),
            (PreflopActionType::Call, 0.3),
            (PreflopActionType::Fold, 0.2),
        ])
        .unwrap();

        // Test boundary conditions
        assert_eq!(strategy.sample(0.0), PreflopActionType::Raise);
        assert_eq!(strategy.sample(0.49), PreflopActionType::Raise);
        assert_eq!(strategy.sample(0.5), PreflopActionType::Call);
        assert_eq!(strategy.sample(0.79), PreflopActionType::Call);
        assert_eq!(strategy.sample(0.8), PreflopActionType::Fold);
        assert_eq!(strategy.sample(0.99), PreflopActionType::Fold);
    }

    #[test]
    fn test_strategy_default() {
        let strategy = PreflopStrategy::default();
        assert_eq!(strategy.frequency(PreflopActionType::Fold), 1.0);
    }

    // ========== PreflopChart tests ==========

    #[test]
    fn test_chart_get_set() {
        let mut chart = PreflopChart::new();
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);

        assert!(chart.get(&aa).is_none());

        chart.set(aa, PreflopStrategy::pure(PreflopActionType::Raise));

        assert!(chart.get(&aa).is_some());
        assert_eq!(
            chart.get(&aa).unwrap().frequency(PreflopActionType::Raise),
            1.0
        );
    }

    #[test]
    fn test_chart_get_or_fold() {
        let chart = PreflopChart::new();
        let unknown = PreflopHand::new(Value::Seven, Value::Two, false);

        let strategy = chart.get_or_fold(&unknown);
        assert_eq!(strategy.frequency(PreflopActionType::Fold), 1.0);
    }

    #[test]
    fn test_chart_remove() {
        let mut chart = PreflopChart::new();
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);

        chart.set(aa, PreflopStrategy::pure(PreflopActionType::Raise));
        assert_eq!(chart.len(), 1);

        let removed = chart.remove(&aa);
        assert!(removed.is_some());
        assert_eq!(chart.len(), 0);

        // Remove non-existent
        let removed = chart.remove(&aa);
        assert!(removed.is_none());
    }

    #[test]
    fn test_chart_iter() {
        let mut chart = PreflopChart::new();
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        let kk = PreflopHand::new(Value::King, Value::King, false);

        chart.set(aa, PreflopStrategy::pure(PreflopActionType::Raise));
        chart.set(kk, PreflopStrategy::pure(PreflopActionType::ThreeBet));

        let count = chart.iter().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_chart_len_is_empty() {
        let mut chart = PreflopChart::new();
        assert!(chart.is_empty());
        assert_eq!(chart.len(), 0);

        chart.set(
            PreflopHand::new(Value::Ace, Value::Ace, false),
            PreflopStrategy::pure(PreflopActionType::Raise),
        );
        assert!(!chart.is_empty());
        assert_eq!(chart.len(), 1);
    }

    // ========== Serde tests (feature-gated) ==========

    #[cfg(feature = "serde")]
    mod serde_tests {
        use super::*;

        #[test]
        fn test_serde_hand_roundtrip() {
            let hand = PreflopHand::new(Value::Ace, Value::King, true);
            let json = serde_json::to_string(&hand).unwrap();
            assert_eq!(json, "\"AKs\"");

            let parsed: PreflopHand = serde_json::from_str(&json).unwrap();
            assert_eq!(hand, parsed);
        }

        #[test]
        fn test_serde_action_type() {
            let action = PreflopActionType::ThreeBet;
            let json = serde_json::to_string(&action).unwrap();
            assert_eq!(json, "\"ThreeBet\"");

            let parsed: PreflopActionType = serde_json::from_str(&json).unwrap();
            assert_eq!(action, parsed);
        }

        #[test]
        fn test_serde_strategy_roundtrip() {
            let strategy = PreflopStrategy::new(vec![
                (PreflopActionType::Raise, 0.7),
                (PreflopActionType::Call, 0.3),
            ])
            .unwrap();

            let json = serde_json::to_string(&strategy).unwrap();
            let parsed: PreflopStrategy = serde_json::from_str(&json).unwrap();

            assert_eq!(
                strategy.frequency(PreflopActionType::Raise),
                parsed.frequency(PreflopActionType::Raise)
            );
            assert_eq!(
                strategy.frequency(PreflopActionType::Call),
                parsed.frequency(PreflopActionType::Call)
            );
        }

        #[test]
        fn test_serde_chart_json() {
            let mut chart = PreflopChart::new();
            chart.set(
                PreflopHand::new(Value::Ace, Value::Ace, false),
                PreflopStrategy::new(vec![
                    (PreflopActionType::ThreeBet, 0.85),
                    (PreflopActionType::Call, 0.15),
                ])
                .unwrap(),
            );
            chart.set(
                PreflopHand::new(Value::Ace, Value::King, true),
                PreflopStrategy::new(vec![
                    (PreflopActionType::ThreeBet, 0.50),
                    (PreflopActionType::Call, 0.40),
                    (PreflopActionType::Fold, 0.10),
                ])
                .unwrap(),
            );

            let json = serde_json::to_string_pretty(&chart).unwrap();
            let parsed: PreflopChart = serde_json::from_str(&json).unwrap();

            assert_eq!(chart.len(), parsed.len());

            let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
            assert_eq!(
                chart
                    .get(&aa)
                    .unwrap()
                    .frequency(PreflopActionType::ThreeBet),
                parsed
                    .get(&aa)
                    .unwrap()
                    .frequency(PreflopActionType::ThreeBet)
            );
        }
    }
}
