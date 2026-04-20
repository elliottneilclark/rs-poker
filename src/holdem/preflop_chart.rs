//! Pre-flop chart system for Texas Hold'em.
//!
//! This module provides types for representing GTO-style action frequencies
//! for the 169 unique starting hands in Texas Hold'em.
//!
//! The chart format is scoped by *scenario* — the decision point is derived
//! from `total_raise_count` at act-time, and a separate [`PreflopChart`] holds
//! the hand→strategy map for each (position, scenario). A strategy is just a
//! `(raise, call)` frequency pair; fold is implicit as `1 - raise - call`.

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
    pub fn new(v1: Value, v2: Value, suited: bool) -> Self {
        let (high, low) = if v1 >= v2 { (v1, v2) } else { (v2, v1) };
        // Pairs can't be suited
        let suited = if high == low { false } else { suited };
        Self { high, low, suited }
    }

    /// Returns true if this is a pocket pair.
    pub fn is_pair(&self) -> bool {
        self.high == self.low
    }

    /// Returns true if this hand is suited. Pairs always return false.
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

    /// Convert to standard notation string ("AA", "AKs", "AKo").
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
    /// # Errors
    ///
    /// Returns `RSPokerError::InvalidPreflopNotation` if the notation is
    /// invalid.
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
            if v1 != v2 {
                return Err(RSPokerError::InvalidPreflopNotation(s.to_string()));
            }
            false
        } else {
            match chars[2].to_ascii_lowercase() {
                's' => {
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

    /// Generate all 169 unique pre-flop hands: 13 pairs + 78 suited + 78
    /// offsuit.
    pub fn all() -> Vec<Self> {
        let mut hands = Vec::with_capacity(169);
        let values = Value::values();

        for (i, &high) in values.iter().enumerate() {
            for &low in &values[..=i] {
                hands.push(Self::new(high, low, false));
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
/// Returns `RSPokerError::InvalidPreflopHandSize` if the hand doesn't have
/// exactly 2 cards.
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

/// Pre-flop action types. Bet sizing is supplied by the action generator,
/// not stored on the action.
///
/// Scenario is external — a single [`PreflopChart`] holds the strategy for
/// one (position, scenario) pair, and the caller knows which scenario
/// applies from `total_raise_count`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum PreflopActionType {
    /// Fold the hand (or check when no bet to call).
    Fold,
    /// Flat-call the current bet.
    Call,
    /// Raise (sizing is scenario-specific: open for RFI, 3-bet for VsOpen,
    /// 4-bet for Vs3Bet).
    Raise,
}

/// Which preflop decision point a strategy entry applies to.
///
/// Derived at action-generation time from `round_data.total_raise_count`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum PreflopScenario {
    /// Unopened pot (0 prior raises this street). Valid: `Raise`, `Fold`.
    Rfi,
    /// Facing exactly 1 raise. Valid: `Raise` (3-bet), `Call`, `Fold`.
    VsOpen,
    /// Facing exactly 2 raises. Valid: `Raise` (4-bet), `Call`, `Fold`.
    Vs3Bet,
    /// Facing 3+ raises. Valid: `Call`, `Fold` (raise is capped).
    Vs4Bet,
}

impl PreflopScenario {
    /// Derive the scenario from `total_raise_count` (non-forced raises this
    /// street).
    pub fn from_raise_count(raises: u8) -> Self {
        match raises {
            0 => Self::Rfi,
            1 => Self::VsOpen,
            2 => Self::Vs3Bet,
            _ => Self::Vs4Bet,
        }
    }

    /// All scenarios in a stable order.
    pub const fn all() -> [Self; 4] {
        [Self::Rfi, Self::VsOpen, Self::Vs3Bet, Self::Vs4Bet]
    }

    /// Short human-readable label ("RFI", "vs Open", ...).
    pub fn label(self) -> &'static str {
        match self {
            Self::Rfi => "RFI",
            Self::VsOpen => "vs Open",
            Self::Vs3Bet => "vs 3-Bet",
            Self::Vs4Bet => "vs 4-Bet",
        }
    }
}

/// Action frequencies for a single hand in a single scenario.
///
/// `raise` and `call` are both in `[0, 1]`, with `raise + call ≤ 1`. Fold is
/// implicit as `1 - raise - call`. Both fields default to 0 when absent from
/// JSON, so `{}` is all-fold, `{"raise": 1.0}` is pure-raise, etc.
///
/// The interpretation of "raise" depends on the scenario in which this
/// strategy is looked up: an open in RFI, a 3-bet in VsOpen, a 4-bet in
/// Vs3Bet. Sizing is supplied by the action generator's config.
///
/// # Examples
///
/// ```
/// use rs_poker::holdem::{PreflopActionType, PreflopStrategy};
///
/// let pure_raise = PreflopStrategy::pure_raise();
/// assert_eq!(pure_raise.raise(), 1.0);
/// assert_eq!(pure_raise.fold_freq(), 0.0);
///
/// let mixed = PreflopStrategy::new(0.5, 0.3).unwrap();
/// assert_eq!(mixed.raise(), 0.5);
/// assert_eq!(mixed.call(), 0.3);
/// assert!((mixed.fold_freq() - 0.2).abs() < 1e-5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct PreflopStrategy {
    /// Raise frequency (open / 3-bet / 4-bet depending on scenario).
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "is_zero"))]
    raise: f32,
    /// Call frequency. Ignored in RFI.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "is_zero"))]
    call: f32,
}

#[cfg(feature = "serde")]
fn is_zero(f: &f32) -> bool {
    *f == 0.0
}

impl PreflopStrategy {
    /// Create a new strategy.
    ///
    /// # Errors
    ///
    /// Returns `RSPokerError::InvalidStrategyFrequencies` if either value is
    /// outside `[0.0, 1.0]` or if `raise + call > 1.001`.
    pub fn new(raise: f32, call: f32) -> Result<Self, RSPokerError> {
        if !(0.0..=1.0 + 0.001).contains(&raise) {
            return Err(RSPokerError::InvalidStrategyFrequencies(format!(
                "raise = {raise}"
            )));
        }
        if !(0.0..=1.0 + 0.001).contains(&call) {
            return Err(RSPokerError::InvalidStrategyFrequencies(format!(
                "call = {call}"
            )));
        }
        if raise + call > 1.0 + 0.001 {
            return Err(RSPokerError::InvalidStrategyFrequencies(format!(
                "raise + call = {:.3}",
                raise + call
            )));
        }
        Ok(Self { raise, call })
    }

    /// Pure-fold strategy (raise=0, call=0).
    pub const fn fold() -> Self {
        Self {
            raise: 0.0,
            call: 0.0,
        }
    }

    /// Pure-raise strategy (raise=1, call=0).
    pub const fn pure_raise() -> Self {
        Self {
            raise: 1.0,
            call: 0.0,
        }
    }

    /// Pure-call strategy (raise=0, call=1).
    pub const fn pure_call() -> Self {
        Self {
            raise: 0.0,
            call: 1.0,
        }
    }

    /// Raise frequency.
    pub fn raise(&self) -> f32 {
        self.raise
    }

    /// Call frequency.
    pub fn call(&self) -> f32 {
        self.call
    }

    /// Implicit fold frequency (`1 - raise - call`, clamped at 0).
    pub fn fold_freq(&self) -> f32 {
        (1.0 - self.raise - self.call).max(0.0)
    }

    /// True when this is the fold-everywhere strategy.
    pub fn is_pure_fold(&self) -> bool {
        self.raise == 0.0 && self.call == 0.0
    }

    /// Look up the frequency for a specific action type.
    pub fn frequency(&self, action: PreflopActionType) -> f32 {
        match action {
            PreflopActionType::Raise => self.raise,
            PreflopActionType::Call => self.call,
            PreflopActionType::Fold => self.fold_freq(),
        }
    }

    /// Sample an action using `random_value` in `[0, 1)`.
    ///
    /// Order is Raise → Call → Fold.
    pub fn sample(&self, random_value: f32) -> PreflopActionType {
        if random_value < self.raise {
            PreflopActionType::Raise
        } else if random_value < self.raise + self.call {
            PreflopActionType::Call
        } else {
            PreflopActionType::Fold
        }
    }
}

impl Default for PreflopStrategy {
    fn default() -> Self {
        Self::fold()
    }
}

/// A pre-flop chart mapping hands to strategies, scoped to one (position,
/// scenario) pair.
///
/// Hands not in the chart are implicitly pure-fold.
///
/// # Examples
///
/// ```
/// use rs_poker::holdem::{PreflopActionType, PreflopChart, PreflopHand, PreflopStrategy};
/// use rs_poker::core::Value;
///
/// let mut chart = PreflopChart::new();
/// let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
/// chart.set(aa, PreflopStrategy::pure_raise());
///
/// assert_eq!(chart.get(&aa).unwrap().raise(), 1.0);
/// assert_eq!(chart.get_or_fold(&PreflopHand::from_notation("72o").unwrap()).fold_freq(), 1.0);
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct PreflopChart {
    strategies: HashMap<PreflopHand, PreflopStrategy>,
}

impl PreflopChart {
    /// Create a new empty pre-flop chart.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the strategy for a hand, or `None` if not set.
    pub fn get(&self, hand: &PreflopHand) -> Option<&PreflopStrategy> {
        self.strategies.get(hand)
    }

    /// Get the strategy for a hand, falling back to pure-fold when absent.
    pub fn get_or_fold(&self, hand: &PreflopHand) -> PreflopStrategy {
        self.strategies.get(hand).copied().unwrap_or_default()
    }

    /// Set the strategy for a hand.
    pub fn set(&mut self, hand: PreflopHand, strategy: PreflopStrategy) {
        self.strategies.insert(hand, strategy);
    }

    /// Remove the strategy for a hand.
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

    /// Returns true if no hands have explicit strategies (every hand folds).
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
        let hand1 = PreflopHand::new(Value::King, Value::Ace, true);
        let hand2 = PreflopHand::new(Value::Ace, Value::King, true);
        assert_eq!(hand1, hand2);
        assert_eq!(hand1.high(), Value::Ace);
        assert_eq!(hand1.low(), Value::King);
    }

    #[test]
    fn test_preflop_hand_try_from_hand() {
        let mut cards = Hand::new();
        cards.insert(Card::new(Value::Ace, Suit::Spade));
        cards.insert(Card::new(Value::King, Suit::Spade));
        let preflop = PreflopHand::try_from(&cards).expect("valid 2-card hand");
        assert!(preflop.suited());
        assert_eq!(preflop.high(), Value::Ace);
        assert_eq!(preflop.low(), Value::King);

        let mut cards = Hand::new();
        cards.insert(Card::new(Value::Ace, Suit::Spade));
        cards.insert(Card::new(Value::King, Suit::Heart));
        let preflop = PreflopHand::try_from(&cards).expect("valid 2-card hand");
        assert!(!preflop.suited());

        let mut cards = Hand::new();
        cards.insert(Card::new(Value::Queen, Suit::Spade));
        cards.insert(Card::new(Value::Queen, Suit::Heart));
        let preflop = PreflopHand::try_from(&cards).expect("valid 2-card hand");
        assert!(preflop.is_pair());
        assert!(!preflop.suited());
    }

    #[test]
    fn test_preflop_hand_try_from_invalid_size() {
        let cards = Hand::new();
        assert!(PreflopHand::try_from(&cards).is_err());

        let mut cards = Hand::new();
        cards.insert(Card::new(Value::Ace, Suit::Spade));
        assert!(PreflopHand::try_from(&cards).is_err());

        let mut cards = Hand::new();
        cards.insert(Card::new(Value::Ace, Suit::Spade));
        cards.insert(Card::new(Value::King, Suit::Spade));
        cards.insert(Card::new(Value::Queen, Suit::Spade));
        assert!(PreflopHand::try_from(&cards).is_err());
    }

    #[test]
    fn test_preflop_hand_all_count() {
        let all = PreflopHand::all();
        assert_eq!(all.len(), 169);
        let pairs = all.iter().filter(|h| h.is_pair()).count();
        let suited = all.iter().filter(|h| h.suited()).count();
        let offsuit = all.iter().filter(|h| !h.is_pair() && !h.suited()).count();
        assert_eq!(pairs, 13);
        assert_eq!(suited, 78);
        assert_eq!(offsuit, 78);
    }

    // ========== Notation tests ==========

    #[test]
    fn test_notation_roundtrip() {
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
        assert!(PreflopHand::from_notation("A").is_err());
        assert!(PreflopHand::from_notation("AKso").is_err());
        assert!(PreflopHand::from_notation("XKs").is_err());
        assert!(PreflopHand::from_notation("AK").is_err());
        assert!(PreflopHand::from_notation("AAs").is_err());
        assert!(PreflopHand::from_notation("AKx").is_err());
    }

    // ========== PreflopStrategy tests ==========

    #[test]
    fn test_strategy_pure_raise() {
        let s = PreflopStrategy::pure_raise();
        assert_eq!(s.raise(), 1.0);
        assert_eq!(s.call(), 0.0);
        assert_eq!(s.fold_freq(), 0.0);
        assert_eq!(s.frequency(PreflopActionType::Raise), 1.0);
        assert_eq!(s.frequency(PreflopActionType::Fold), 0.0);
    }

    #[test]
    fn test_strategy_pure_call() {
        let s = PreflopStrategy::pure_call();
        assert_eq!(s.raise(), 0.0);
        assert_eq!(s.call(), 1.0);
        assert_eq!(s.fold_freq(), 0.0);
    }

    #[test]
    fn test_strategy_fold() {
        let s = PreflopStrategy::fold();
        assert_eq!(s.fold_freq(), 1.0);
        assert_eq!(s.raise(), 0.0);
        assert_eq!(s.call(), 0.0);
        assert!(s.is_pure_fold());
    }

    #[test]
    fn test_strategy_mixed() {
        let s = PreflopStrategy::new(0.6, 0.3).unwrap();
        assert_eq!(s.raise(), 0.6);
        assert_eq!(s.call(), 0.3);
        assert!((s.fold_freq() - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_strategy_rejects_out_of_range() {
        assert!(PreflopStrategy::new(1.5, 0.0).is_err());
        assert!(PreflopStrategy::new(-0.1, 0.0).is_err());
        assert!(PreflopStrategy::new(0.0, 1.5).is_err());
        assert!(PreflopStrategy::new(0.0, -0.1).is_err());
    }

    #[test]
    fn test_strategy_rejects_sum_over_one() {
        let result = PreflopStrategy::new(0.6, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_strategy_allows_partial_sum() {
        let s = PreflopStrategy::new(0.5, 0.0).unwrap();
        assert!((s.fold_freq() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_strategy_sample() {
        let s = PreflopStrategy::new(0.5, 0.3).unwrap();
        assert_eq!(s.sample(0.0), PreflopActionType::Raise);
        assert_eq!(s.sample(0.49), PreflopActionType::Raise);
        assert_eq!(s.sample(0.5), PreflopActionType::Call);
        assert_eq!(s.sample(0.79), PreflopActionType::Call);
        assert_eq!(s.sample(0.85), PreflopActionType::Fold);
    }

    #[test]
    fn test_strategy_default_is_fold() {
        let s = PreflopStrategy::default();
        assert!(s.is_pure_fold());
    }

    // ========== PreflopChart tests ==========

    #[test]
    fn test_chart_get_set() {
        let mut chart = PreflopChart::new();
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        assert!(chart.get(&aa).is_none());
        chart.set(aa, PreflopStrategy::pure_raise());
        assert_eq!(chart.get(&aa).unwrap().raise(), 1.0);
    }

    #[test]
    fn test_chart_get_or_fold() {
        let chart = PreflopChart::new();
        let unknown = PreflopHand::new(Value::Seven, Value::Two, false);
        let strategy = chart.get_or_fold(&unknown);
        assert!(strategy.is_pure_fold());
    }

    #[test]
    fn test_chart_remove() {
        let mut chart = PreflopChart::new();
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        chart.set(aa, PreflopStrategy::pure_raise());
        assert_eq!(chart.len(), 1);
        assert!(chart.remove(&aa).is_some());
        assert_eq!(chart.len(), 0);
        assert!(chart.remove(&aa).is_none());
    }

    #[test]
    fn test_chart_len_is_empty() {
        let mut chart = PreflopChart::new();
        assert!(chart.is_empty());
        chart.set(
            PreflopHand::new(Value::Ace, Value::Ace, false),
            PreflopStrategy::pure_raise(),
        );
        assert!(!chart.is_empty());
        assert_eq!(chart.len(), 1);
    }

    // ========== PreflopScenario tests ==========

    #[test]
    fn test_scenario_from_raise_count() {
        assert_eq!(PreflopScenario::from_raise_count(0), PreflopScenario::Rfi);
        assert_eq!(
            PreflopScenario::from_raise_count(1),
            PreflopScenario::VsOpen
        );
        assert_eq!(
            PreflopScenario::from_raise_count(2),
            PreflopScenario::Vs3Bet
        );
        assert_eq!(
            PreflopScenario::from_raise_count(3),
            PreflopScenario::Vs4Bet
        );
        assert_eq!(
            PreflopScenario::from_raise_count(10),
            PreflopScenario::Vs4Bet
        );
    }

    // ========== Serde tests ==========

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
            let action = PreflopActionType::Raise;
            let json = serde_json::to_string(&action).unwrap();
            assert_eq!(json, "\"Raise\"");
            let parsed: PreflopActionType = serde_json::from_str(&json).unwrap();
            assert_eq!(action, parsed);
        }

        #[test]
        fn test_serde_strategy_minimal() {
            // {} should deserialize as pure-fold.
            let s: PreflopStrategy = serde_json::from_str("{}").unwrap();
            assert!(s.is_pure_fold());
        }

        #[test]
        fn test_serde_strategy_raise_only() {
            // {"raise": 1.0} should deserialize with call=0.
            let s: PreflopStrategy = serde_json::from_str(r#"{"raise": 1.0}"#).unwrap();
            assert_eq!(s.raise(), 1.0);
            assert_eq!(s.call(), 0.0);
        }

        #[test]
        fn test_serde_strategy_call_only() {
            let s: PreflopStrategy = serde_json::from_str(r#"{"call": 0.5}"#).unwrap();
            assert_eq!(s.raise(), 0.0);
            assert_eq!(s.call(), 0.5);
        }

        #[test]
        fn test_serde_strategy_both() {
            let s: PreflopStrategy =
                serde_json::from_str(r#"{"raise": 0.5, "call": 0.3}"#).unwrap();
            assert_eq!(s.raise(), 0.5);
            assert_eq!(s.call(), 0.3);
        }

        #[test]
        fn test_serde_strategy_skip_zero() {
            // Zero fields should not appear in output.
            let s = PreflopStrategy::pure_raise();
            let json = serde_json::to_string(&s).unwrap();
            assert_eq!(json, r#"{"raise":1.0}"#);

            let s = PreflopStrategy::fold();
            let json = serde_json::to_string(&s).unwrap();
            assert_eq!(json, "{}");
        }

        #[test]
        fn test_serde_chart_json() {
            let mut chart = PreflopChart::new();
            chart.set(
                PreflopHand::new(Value::Ace, Value::Ace, false),
                PreflopStrategy::new(0.85, 0.15).unwrap(),
            );
            chart.set(
                PreflopHand::new(Value::Ace, Value::King, true),
                PreflopStrategy::pure_raise(),
            );

            let json = serde_json::to_string(&chart).unwrap();
            let parsed: PreflopChart = serde_json::from_str(&json).unwrap();
            assert_eq!(chart, parsed);
        }

        #[test]
        fn test_serde_chart_transparent() {
            // PreflopChart serializes as the underlying map.
            let chart = PreflopChart::new();
            let json = serde_json::to_string(&chart).unwrap();
            assert_eq!(json, "{}");
        }
    }
}
