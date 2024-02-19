use std::collections::HashSet;
use crate::core::{Card, Hand, RSPokerError, Suit, Value};
use crate::holdem::Suitedness;
use std::iter::Iterator;

/// Inclusive Range of card values.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
struct InclusiveValueRange {
    /// Lowest value allowed in the range
    start: Value,
    /// Highest value in the range.
    end: Value,
}

impl InclusiveValueRange {
    /// Just incase the start and end are messed up sort them.
    fn sort(&mut self) {
        if self.start > self.end {
            std::mem::swap(&mut self.start, &mut self.end);
        }
    }
    /// Is a value in the range
    #[inline]
    fn include(&self, v: Value) -> bool {
        self.start <= v && v <= self.end
    }
    /// This range defines only a single value.
    fn is_single(&self) -> bool {
        self.start == self.end
    }
}

/// Modifier on the end of a hand range.
enum Modifier {
    /// Keep cards higher than the one preceding this, and less than the first
    /// one. However if the plus is applied to a card range where the first
    /// and second card are connectors, then it means all connectors above
    /// the current ones.
    Plus,
    /// The range modifier means that the next
    /// cards after the dash will make new ends of the range.
    Range,
    /// Only keep cards that have the same suit
    Suited,
    /// Only keep sets of cards that have different suits.
    Offsuit,
}

impl Modifier {
    /// From a character try and extract a modifier.
    /// Returns None if it doesn't match anything.
    fn from_char(c: char) -> Option<Self> {
        match c {
            '+' => Some(Self::Plus),
            's' => Some(Self::Suited),
            'o' => Some(Self::Offsuit),
            '-' => Some(Self::Range),
            _ => None,
        }
    }
}

/// Enum to specify how a value will be decided.
#[derive(Debug, PartialEq)]
enum RangeIterValueSpecifier {
    /// This value will be a gap away from the smaller
    Gap(u8),
    /// This value will be static.
    Static(Value),
    /// Pair
    Pair,
}

/// This is an `Iterator` that will iterate over two card hands
/// in a way that range parser can use them. It is not meant to be used
/// outside of the parser module.
#[derive(Debug)]
struct RangeIter {
    /// Specifier for how to determine the value of the first card.
    /// See `RangeIterValueSpecifier` for the different options.
    value_one: RangeIterValueSpecifier,
    /// Range for the second card
    range: InclusiveValueRange,
    /// How far into the range we are.
    offset: u8,
    /// Which suit to use for the first card.
    suit_one_offset: u8,
    /// Which suit to use for the second card
    suit_two_offset: u8,
}

impl RangeIter {
    /// Create a new parser by giving it a static value and a range.
    fn stat(value: Value, range_two: InclusiveValueRange) -> Self {
        Self {
            value_one: RangeIterValueSpecifier::Static(value),
            range: range_two,
            offset: 0,
            suit_one_offset: 0,
            suit_two_offset: 0,
        }
    }
    /// Create a range iterator where the first card is a gap away from
    /// the second.
    fn gap(gap: u8, range_two: InclusiveValueRange) -> Self {
        Self {
            value_one: RangeIterValueSpecifier::Gap(gap),
            range: range_two,
            offset: 0,
            suit_one_offset: 0,
            suit_two_offset: 0,
        }
    }
    /// Create an iterator over a range of pocket pairs
    fn pair(range: InclusiveValueRange) -> Self {
        Self {
            value_one: RangeIterValueSpecifier::Pair,
            range,
            offset: 0,
            suit_one_offset: 0,
            suit_two_offset: 1,
        }
    }
    /// Is this iterator creating pocket pairs?
    #[inline]
    fn is_pair(&self) -> bool {
        self.value_one == RangeIterValueSpecifier::Pair
            || (self.range.is_single()
                && RangeIterValueSpecifier::Static(self.range.start) == self.value_one)
    }
    /// Check if this iterator can create more items.
    #[inline]
    fn has_more(&self) -> bool {
        let v = self.offset + self.range.start as u8;
        v < 13 && self.range.end >= Value::from_u8(v)
    }
    /// Move the indecies to the next value.
    /// This movex first the suits then it moves the values.
    #[inline]
    fn incr(&mut self) {
        // Move the suit forward on.
        self.suit_two_offset += 1;
        // See if we've gone past the end of the suit.
        // If that happens we need move other indecies.
        if self.suit_two_offset == 4 {
            self.suit_one_offset += 1;
            // Reset where the smaller card's suit goes.
            self.suit_two_offset = if self.is_pair() {
                // If this is a range of pairs then we can't
                // ever have two of the smake suit so start ater this
                self.suit_one_offset + 1
            } else {
                // If this isn't a pair then start at the begining
                0
            };
        }
        if self.suit_one_offset == 4 || self.suit_two_offset == 4 {
            self.suit_one_offset = 0;
            // Pairs can't have two of the same suit.
            if self.is_pair() {
                self.suit_two_offset = 1;
            }
            self.offset += 1;
        }
    }
    /// Create the first card.
    /// This card will depend on the value specifier.
    fn first_card(&self) -> Card {
        // Figure out the value.
        let v = match self.value_one {
            RangeIterValueSpecifier::Gap(gap) => {
                Value::from_u8(self.range.start as u8 + self.offset + gap)
            }
            RangeIterValueSpecifier::Static(value) => value,
            RangeIterValueSpecifier::Pair => Value::from_u8(self.range.start as u8 + self.offset),
        };
        // Create the card.
        Card {
            value: v,
            suit: Suit::from_u8(self.suit_one_offset),
        }
    }
    /// Create the second smaller card.
    fn second_card(&self) -> Card {
        // Create the card.
        Card {
            value: Value::from_u8(self.range.start as u8 + self.offset),
            suit: Suit::from_u8(self.suit_two_offset),
        }
    }
}

/// `Iterator` implementation for `RangeIter`
impl Iterator for RangeIter {
    type Item = Hand;
    /// Get the next value if there are any.
    fn next(&mut self) -> Option<Hand> {
        if self.has_more() {
            let h = Hand::new_with_cards(vec![self.first_card(), self.second_card()]);
            self.incr();
            Some(h)
        } else {
            None
        }
    }
}

/// Unit struct to provide starting hand parse functions. Use this to parse
/// things like `RangeParser::parse_one("AKo")` and
/// `RangeParser::parse_one("TT+")`
pub struct RangeParser;

impl RangeParser {
    /// Parse a string and return all the starting hands
    ///
    /// # Examples
    ///
    /// You can send in hands where the suits are specified for all hands.
    ///
    /// ```
    /// use rs_poker::holdem::RangeParser;
    ///
    /// let hand = RangeParser::parse_one("AhKh").unwrap();
    /// assert_eq!(1, hand.len());
    /// ```
    ///
    /// You can also specify hands were the suits are not specified,
    /// but you want them to be suited.
    ///
    /// ```
    /// use rs_poker::core::Value;
    /// use rs_poker::holdem::RangeParser;
    ///
    /// let hands = RangeParser::parse_one("AKs").unwrap();
    /// assert!(hands.len() == 4);
    /// for hand in hands {
    ///     assert!(hand[0].suit == hand[1].suit);
    ///     assert_eq!(Value::Ace, hand[0].value);
    ///     assert_eq!(Value::King, hand[1].value);
    /// }
    /// ```
    ///
    /// You can also specify that the cards are not of the same suit.
    ///
    /// ```
    /// use rs_poker::core::Value;
    /// use rs_poker::holdem::RangeParser;
    ///
    /// let hands = RangeParser::parse_one("AKo").unwrap();
    ///
    /// for hand in hands {
    ///     assert!(hand[0].suit != hand[1].suit);
    ///     assert_eq!(Value::Ace, hand[0].value);
    ///     assert_eq!(Value::King, hand[1].value);
    /// }
    /// ```
    ///
    /// You can also use the + modifier after a set of
    /// starting cards. The modifier will mean different things
    /// for different sets of cards.
    ///
    /// If the starting cards are pairs then the + means all pairs
    /// equal to or above the specified values.
    ///
    /// ```
    /// use rs_poker::core::Value;
    /// use rs_poker::holdem::RangeParser;
    ///
    /// let hands = RangeParser::parse_one("TT+").unwrap();
    ///
    /// for hand in hands {
    ///     assert!(hand[0].value == hand[1].value);
    ///     assert!(hand[0].value >= Value::Ten);
    ///     assert!(hand[1].value >= Value::Ten);
    /// }
    /// ```
    ///
    /// If the cards are connectors then the plus means all
    /// connectors where the cards are above the specified
    /// values.
    ///
    /// ```
    /// use rs_poker::core::Value;
    /// use rs_poker::holdem::RangeParser;
    ///
    /// let hands = RangeParser::parse_one("T9o+").unwrap();
    ///
    /// for hand in hands {
    ///     assert_eq!(hand[0].value, Value::from_u8(hand[1].value as u8 + 1));
    ///     assert!(hand[0].suit != hand[1].suit);
    /// }
    /// ```
    ///
    /// If the cards are not paired and not connectors then plus
    /// after the hand means all hands where the second card
    /// is greater than or equal to the specified second card,
    /// and below the first card.
    ///
    /// ```
    /// use rs_poker::core::Value;
    /// use rs_poker::holdem::RangeParser;
    /// let hands = RangeParser::parse_one("A9s+").unwrap();
    /// for hand in hands {
    ///     assert!(hand[0].value > hand[1].value);
    ///     assert!(hand[1].value >= Value::Nine);
    ///     assert!(hand[0].suit == hand[1].suit);
    /// }
    /// ```
    ///
    /// It's also possible to do more complex ranges using
    /// dash modifer. For example if you wanted to represent
    /// Suited middle connectors you could do something like this.
    ///
    /// ```
    /// use rs_poker::core::Value;
    /// use rs_poker::holdem::RangeParser;
    ///
    /// let hands = RangeParser::parse_one("JT-67s").unwrap();
    /// for hand in hands {
    ///     // The largest card is always the first
    ///     assert!(hand[0].value > hand[1].value);
    ///     // first card is great or equal to seven
    ///     assert!(hand[0].value >= Value::Seven);
    ///     // Second card is greater or equal to six
    ///     assert!(hand[1].value >= Value::Six);
    ///     // First card is less than or equal to Jack
    ///     assert!(hand[0].value <= Value::Jack);
    ///     // Second card is less than or equal to Ten
    ///     assert!(hand[1].value <= Value::Ten);
    ///     // All hands are connectors.
    ///     assert_eq!(1, hand[0].value.gap(hand[1].value));
    /// }
    /// ```
    ///
    /// Also with the dash modifier there's no need to only have
    /// connected cards. It's possible to represent ranges with gappers.
    /// For example if you wanted to do high one gappers.
    ///
    ///
    /// ```
    /// use rs_poker::holdem::RangeParser;
    /// let hands = RangeParser::parse_one("AQ-J9").unwrap();
    /// println!("Hands = {:?}", hands);
    /// ```
    ///
    /// Since the dash modifier represents a range the difference
    /// between cards ( the gap ) must remain constant.
    /// If it's not `parse_one will` return an `Err`.
    ///
    /// ```rust,should_panic
    /// use rs_poker::holdem::RangeParser;
    /// // This will not work since the difference between Ace and King is one
    /// // while the diffence between Jack and Nine is two.
    /// let hands = RangeParser::parse_one("AK-J9").unwrap();
    /// // We'll never get here
    /// println!("Hands = {:?}", hands);
    /// ```
    pub fn parse_one(r_str: &str) -> Result<Vec<Hand>, RSPokerError> {
        let mut iter = r_str.chars().peekable();
        let mut first_range = InclusiveValueRange {
            start: Value::Two,
            end: Value::Ace,
        };
        let mut second_range = InclusiveValueRange {
            start: Value::Two,
            end: Value::Ace,
        };
        // Assume that we know nothing about suits.
        let mut first_suit: Option<Suit> = None;
        let mut second_suit: Option<Suit> = None;
        // Assume that we don't care about suited/offsuit
        let mut suited = Suitedness::Any;
        // Assume that this is not a set of connectors
        let mut gap: Option<u8> = None;

        // Get the first char.
        let fv_char = iter.next().ok_or(RSPokerError::TooFewChars)?;
        // It should be a value.
        first_range.start = Value::from_char(fv_char).ok_or(RSPokerError::TooFewChars)?;
        // Make the assumption that there's no ranges involved.
        first_range.end = first_range.start;

        // Try and get a suit.
        if let Some(s) = Suit::from_char(*iter.peek().unwrap_or(&':')) {
            first_suit = Some(s);
            iter.next();
        }

        // Now there should be another value char.
        let sv_char = iter.next().ok_or(RSPokerError::TooFewChars)?;
        // that char should parse correctly.
        second_range.start = Value::from_char(sv_char).ok_or(RSPokerError::TooFewChars)?;
        second_range.end = second_range.start;

        // If the first one had a suit then it's possible that
        // the second on can have it.
        // parse this first so that s is assumed to be a spade rather
        // than suited if the first card had a suit.
        if first_suit.is_some() {
            // Try and parse the suit.
            if let Some(s) = Suit::from_char(*iter.peek().unwrap_or(&':')) {
                // If we got it then keep it.
                second_suit = Some(s);
                // And consume the char.
                iter.next();
            }
        }

        // Now check to see how the modifier change all this.
        loop {
            if let Some(m) = Modifier::from_char(*iter.peek().unwrap_or(&':')) {
                // Consume the modifier character.
                iter.next();
                // Now do something with it.
                match m {
                    Modifier::Offsuit => {
                        if first_suit.is_some() && first_suit == second_suit {
                            return Err(RSPokerError::OffSuitWithMatchingSuit);
                        }
                        suited = Suitedness::OffSuit;
                    }
                    Modifier::Suited => {
                        if first_suit.is_some()
                            && second_suit.is_some()
                            && first_suit != second_suit
                        {
                            return Err(RSPokerError::SuitedWithNoMatchingSuit);
                        }
                        suited = Suitedness::Suited;
                    }
                    Modifier::Plus => {
                        if gap.is_some() {
                            return Err(RSPokerError::InvalidPlusModifier);
                        }
                        let ex_gap = first_range.end.gap(second_range.end);
                        if ex_gap <= 1 {
                            // This is either a pocket pair (ex_gap == 0)
                            // or connectors (ex_gap == 1).
                            first_range.end = Value::Ace;
                            second_range.end = Value::from_u8(Value::Ace as u8 - ex_gap);
                            gap = Some(ex_gap);
                        } else if first_range.end < second_range.end {
                            return Err(RSPokerError::InvalidPlusModifier);
                        } else {
                            second_range.end = Value::from_u8(first_range.end as u8 - 1);
                        }
                    }
                    Modifier::Range => {
                        let fr_char = iter.next().ok_or(RSPokerError::TooFewChars)?;
                        let sr_char = iter.next().ok_or(RSPokerError::TooFewChars)?;
                        first_range.end =
                            Value::from_char(fr_char).ok_or(RSPokerError::UnexpectedValueChar)?;
                        second_range.end =
                            Value::from_char(sr_char).ok_or(RSPokerError::UnexpectedValueChar)?;

                        let first_gap = first_range.start.gap(second_range.start);
                        let second_gap = first_range.end.gap(second_range.end);

                        if first_gap != second_gap {
                            return Err(RSPokerError::InvalidGap);
                        }
                        gap = Some(first_gap);
                    }
                }
            } else {
                // there is no modifier, but we still want to recognize pocket pairs
                if (first_range.start.gap(second_range.start)) == 0 {
                    gap = Some(0)
                }
                break;
            }
        }

        // It's possible that the ordering was weird.
        first_range.sort();
        second_range.sort();
        if first_range < second_range {
            std::mem::swap(&mut first_range, &mut second_range);
            std::mem::swap(&mut first_suit, &mut second_suit);
        }

        // Now create an iterator for two cards.
        let citer = match gap {
            Some(0) => RangeIter::pair(first_range.clone()),
            Some(g) => RangeIter::gap(g, second_range.clone()),
            None => RangeIter::stat(first_range.start, second_range.clone()),
        };

        // There can not be suited pairs
        if citer.is_pair() {
            // Do the two cards have a suit specified and it is the same suit.
            let explicitly_suited = first_suit.is_some() && first_suit == second_suit;
            if suited == Suitedness::Suited || explicitly_suited {
                return Err(RSPokerError::InvalidSuitedPairs);
            }
        }

        let filtered: Vec<Hand> = citer
            // Need to make sure that the first card is in the range
            .filter(|hand| first_range.include(hand[0].value))
            // Make sure the second card is in the range
            .filter(|hand| second_range.include(hand[1].value))
            // If this is suited then make sure that they are suited.
            .filter(|h| {
                (suited == Suitedness::Any)
                    || (suited == Suitedness::OffSuit && h[0].suit != h[1].suit)
                    || (suited == Suitedness::Suited && h[0].suit == h[1].suit)
            })
            // Make sure the suits match if specified
            .filter(|h| {
                if h[0].value == h[1].value {
                    // This is a pair so ordering on suits can be weird.
                    first_suit.map_or(true, |s| h[0].suit == s || h[1].suit == s)
                        && second_suit.map_or(true, |s| h[0].suit == s || h[1].suit == s)
                } else {
                    first_suit.map_or(true, |s| h[0].suit == s)
                        && second_suit.map_or(true, |s| h[1].suit == s)
                }
            })
            // If there is a gap make sure it's enforced.
            .filter(|h| gap.map_or(true, |g| g == h[0].value.gap(h[1].value)))
            .collect();

        Ok(filtered)
    }

    /// Parse a string and return all the starting hands
    ///
    /// # Examples
    ///
    /// Same as `parse_one` but this will parse a comma separated list of hands.
    ///
    /// ```
    /// use rs_poker::holdem::RangeParser;
    ///
    /// let hand = RangeParser::parse_many("KK+,A2s+").unwrap();
    /// assert_eq!(60, hand.len());
    ///
    /// // Filters out duplicates.
    /// assert_eq!(RangeParser::parse_many("AK-87s,A2s+").unwrap().len(), 72)
    /// ```
    pub fn parse_many(r_str: &str) -> Result<Vec<Hand>, Vec<RSPokerError>> {
        let mut errors: Vec<RSPokerError> = vec![];
        let mut set = HashSet::new();

        let hands: Vec<_> = r_str.split(',')
            .map(|r| RangeParser::parse_one(r.trim()))
            .filter_map(|r| r.map_err(|e| errors.push(e)).ok())
            .flatten()
            .collect();

        return match errors.is_empty() {
            true => {
                set.extend(hands);

                Ok(set.into_iter().collect())
            },
            false => Err(errors)
        };
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::core::Value;

    #[test]
    fn test_range_iter_static() {
        let c = RangeIter::stat(
            Value::Ace,
            InclusiveValueRange {
                start: Value::Two,
                end: Value::King,
            },
        );

        let mut count = 0;
        for hand in c {
            count += 1;
            assert!(hand[0] > hand[1]);
        }
        assert_eq!(12 * 4 * 4, count);
    }

    #[test]
    fn test_easy_parse() {
        // Parse something easy.
        let c = RangeParser::parse_one("AK").unwrap();
        assert_eq!(16, c.len());
    }

    #[test]
    fn test_single_pair() {
        // Test to make sure a single pair is parsed correctly
        let c = RangeParser::parse_one("22").unwrap();
        assert_eq!(6, c.len());
    }

    #[test]
    fn test_easy_parse_sorted() {
        // Test to make sure that order doesn't matter for the easy parsing.
        assert_eq!(
            RangeParser::parse_one("AK").unwrap(),
            RangeParser::parse_one("KA").unwrap()
        );
    }

    #[test]
    fn test_easy_parse_offsuit() {
        // Make sure that the off suit works.
        let c = RangeParser::parse_one("AKo").unwrap();
        let mut count = 0;
        for h in c {
            count += 1;
            assert!(h[0].suit != h[1].suit);
            assert_eq!(Value::Ace, h[0].value);
            assert_eq!(Value::King, h[1].value);
        }
        assert_eq!(4 * 3, count);
    }

    #[test]
    fn test_easy_parse_suited() {
        let c = RangeParser::parse_one("AKs").unwrap();
        let mut count = 0;
        for h in c {
            count += 1;
            // Needs to be suited
            assert!(h[0].suit == h[1].suit);
            // Needs to be aces and kings.
            assert_eq!(Value::Ace, h[0].value);
            assert_eq!(Value::King, h[1].value);
        }
        assert_eq!(4, count);
    }

    #[test]
    fn test_plus_top_gap() {
        let c = RangeParser::parse_one("KQ+").unwrap();
        // 4 Suits for first card.
        // 4 Suits for second card.
        // AK and KQ
        assert_eq!(4 * 4 * 2, c.len());
    }

    #[test]
    fn test_plus_low_gap() {
        let c = RangeParser::parse_one("32+").unwrap();
        assert_eq!(12 * 16, c.len());
    }

    #[test]
    fn test_plus_ungapped() {
        let c = RangeParser::parse_one("A9+").unwrap();
        assert_eq!(4 * 4 * 5, c.len());
    }

    #[test]
    fn test_plus_pair() {
        let c = RangeParser::parse_one("KK+").unwrap();
        let mut count = 0;
        for h in c {
            count += 1;
            // Same value
            assert_eq!(h[0].value, h[1].value);
            // But not the same card.
            assert!(h[0] != h[1]);
        }
        assert_eq!(6 * 2, count);
    }

    #[test]
    fn test_range_parse_suited() {
        let c = RangeParser::parse_one("87-JTs").unwrap();
        assert_eq!(4 * 4, c.len());
    }
    #[test]
    fn test_range_parse_flipped() {
        let c = RangeParser::parse_one("JT-87").unwrap();
        let mut count = 0;
        for h in c {
            count += 1;
            assert!(h[0] > h[1]);
        }
        assert_eq!(4 * 4 * 4, count);
    }
    #[test]
    fn test_range_parse_flipped_flipped() {
        let c = RangeParser::parse_one("TJ-78").unwrap();
        let mut count = 0;
        for h in c {
            count += 1;
            assert!(h[0] > h[1]);
        }
        assert_eq!(4 * 4 * 4, count);
    }
    #[test]
    fn test_range_parse() {
        let c = RangeParser::parse_one("87-JT").unwrap();
        assert_eq!(4 * 4 * 4, c.len());
    }

    #[test]
    fn test_cant_suit_pairs() {
        let shs = RangeParser::parse_one(&String::from("88s"));
        assert!(shs.is_err());
    }
    #[test]
    fn test_cant_suit_pairs_explicit() {
        let shs = RangeParser::parse_one(&String::from("8s8s"));
        assert!(shs.is_err());
    }

    #[test]
    fn test_explicit_pair_good() {
        assert!(!RangeParser::parse_one(&String::from("2c2s"))
            .unwrap()
            .is_empty());
    }
    #[test]
    fn test_explicit_suit_good() {
        assert!(!RangeParser::parse_one(&String::from("6c2c"))
            .unwrap()
            .is_empty());
    }
    #[test]
    fn test_explicit_suited_no_good() {
        assert!(RangeParser::parse_one(&String::from("6c2co")).is_err());
        assert!(RangeParser::parse_one(&String::from("6h2cs")).is_err());
    }

    #[test]
    fn test_bad_input() {
        assert!(RangeParser::parse_one(&String::from("4f7")).is_err());
    }
    #[test]
    fn test_explicit_suit_plus() {
        assert!(!RangeParser::parse_one(&String::from("2s2+"))
            .unwrap()
            .is_empty());
    }
    #[test]
    fn test_explicit_suit_pair() {
        assert!(!RangeParser::parse_one(&String::from("8D8"))
            .unwrap()
            .is_empty());
    }
    #[test]
    fn test_ok_with_trailing_plus() {
        assert!(RangeParser::parse_one(&String::from("8Q-62+")).is_err());
    }

    #[test]
    fn test_ok_with_multiple() {
        assert!(RangeParser::parse_many(&String::from("KK+, AJs+")).is_ok());
    }

    #[test]
    fn test_ok_with_single() {
        assert!(RangeParser::parse_many(&String::from("KK+")).is_ok());
    }

    #[test]
    fn test_parse_multiple() {
        assert_eq!(
            RangeParser::parse_many(&String::from("KK+, A2s+"))
                .unwrap()
                .len(),
            60
        );
    }

    #[test]
    fn test_filters_duplicates() {
        assert_eq!(
            RangeParser::parse_many(&String::from("AK-87s,A2s+"))
                .unwrap()
                .len(),
            72
        );
    }
}
