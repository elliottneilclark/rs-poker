use std::cmp;
use std::fmt;
use std::mem;

use super::error::RSPokerError;

/// Card rank or value.
/// This is basically the face value - 2
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(PartialEq, PartialOrd, Eq, Ord, Debug, Clone, Copy, Hash)]
pub enum Value {
    /// 2
    Two = 0,
    /// 3
    Three = 1,
    /// 4
    Four = 2,
    /// 5
    Five = 3,
    /// 6
    Six = 4,
    /// 7
    Seven = 5,
    /// 8
    Eight = 6,
    /// 9
    Nine = 7,
    /// T
    Ten = 8,
    /// J
    Jack = 9,
    /// Q
    Queen = 10,
    /// K
    King = 11,
    /// A
    Ace = 12,
}

/// Constant of all the values.
/// This is what `Value::values()` returns
const VALUES: [Value; 13] = [
    Value::Two,
    Value::Three,
    Value::Four,
    Value::Five,
    Value::Six,
    Value::Seven,
    Value::Eight,
    Value::Nine,
    Value::Ten,
    Value::Jack,
    Value::Queen,
    Value::King,
    Value::Ace,
];

impl Value {
    /// Take a u32 and convert it to a value.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::core::Value;
    /// assert_eq!(Value::Four, Value::from_u8(Value::Four as u8));
    /// ```
    pub fn from_u8(v: u8) -> Self {
        Self::from(v)
    }
    /// Get all of the `Value`'s that are possible.
    /// This is used to iterate through all possible
    /// values when creating a new deck, or
    /// generating all possible starting hands.
    pub const fn values() -> [Self; 13] {
        VALUES
    }

    /// Given a character parse that char into a value.
    /// Case is ignored as long as the char is in the ascii range (It should
    /// be).
    ///
    /// @returns None if there's no valid value there. Otherwise a Value enum
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::core::Value;
    ///
    /// assert_eq!(Value::Ace, Value::from_char('A').unwrap());
    /// ```
    pub fn from_char(c: char) -> Option<Self> {
        Self::try_from(c).ok()
    }

    /// Convert this Value to a char.
    pub fn to_char(self) -> char {
        char::from(self)
    }

    /// How card ranks seperate the two values.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::core::Value;
    /// assert_eq!(1, Value::Ace.gap(Value::King));
    /// ```
    pub fn gap(self, other: Self) -> u8 {
        let min = cmp::min(self as u8, other as u8);
        let max = cmp::max(self as u8, other as u8);
        max - min
    }
}

impl From<u8> for Value {
    fn from(value: u8) -> Self {
        unsafe { mem::transmute(cmp::min(value, Self::Ace as u8)) }
    }
}

impl TryFrom<char> for Value {
    type Error = RSPokerError;

    /// ```
    /// use rs_poker::core::*;
    /// use std::convert::TryFrom;
    ///
    /// assert_eq!(Value::Jack, Value::try_from('j').unwrap());
    /// assert_eq!(Value::Jack, Value::try_from('J').unwrap());
    /// ```
    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value.to_ascii_uppercase() {
            'A' => Ok(Self::Ace),
            'K' => Ok(Self::King),
            'Q' => Ok(Self::Queen),
            'J' => Ok(Self::Jack),
            'T' => Ok(Self::Ten),
            '9' => Ok(Self::Nine),
            '8' => Ok(Self::Eight),
            '7' => Ok(Self::Seven),
            '6' => Ok(Self::Six),
            '5' => Ok(Self::Five),
            '4' => Ok(Self::Four),
            '3' => Ok(Self::Three),
            '2' => Ok(Self::Two),
            _ => Err(RSPokerError::UnexpectedValueChar),
        }
    }
}

impl From<Value> for char {
    fn from(value: Value) -> Self {
        match value {
            Value::Ace => 'A',
            Value::King => 'K',
            Value::Queen => 'Q',
            Value::Jack => 'J',
            Value::Ten => 'T',
            Value::Nine => '9',
            Value::Eight => '8',
            Value::Seven => '7',
            Value::Six => '6',
            Value::Five => '5',
            Value::Four => '4',
            Value::Three => '3',
            Value::Two => '2',
        }
    }
}

/// Implement the From trait
///
/// # Examples
///
/// Ace is a high card in our counting so it's the max value 12
/// ```
/// use rs_poker::core::Value;
/// let v = Value::try_from('A').unwrap();
/// assert_eq!(Value::Ace, v);
/// assert_eq!(12, u8::from(v));
/// ```
///
/// Values are zero indexed with 2 being the lowest value.
/// ```
/// use rs_poker::core::Value;
/// let v = Value::try_from('4').unwrap();
/// assert_eq!(Value::Four, v);
/// assert_eq!(2, u8::from(v));
/// ```
impl From<Value> for u8 {
    fn from(value: Value) -> Self {
        value as u8
    }
}

/// Enum for the four different suits.
/// While this has support for ordering it's not
/// sensical. The sorting is only there to allow sorting cards.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(PartialEq, PartialOrd, Eq, Ord, Debug, Clone, Copy, Hash)]
pub enum Suit {
    /// Spades
    Spade = 0,
    /// Clubs
    Club = 1,
    /// Hearts
    Heart = 2,
    /// Diamonds
    Diamond = 3,
}

/// All of the `Suit`'s. This is what `Suit::suits()` returns.
const SUITS: [Suit; 4] = [Suit::Spade, Suit::Club, Suit::Heart, Suit::Diamond];

/// Impl of Suit
///
/// This is just here to provide a list of all `Suit`'s.
impl Suit {
    /// Provide all the Suit's that there are.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::core::Suit;
    /// let suits = Suit::suits();
    /// assert_eq!(4, suits.len());
    /// ```
    pub const fn suits() -> [Self; 4] {
        SUITS
    }

    /// Translate a Suit from a u8. If the u8 is above the expected value
    /// then Diamond will be the result.
    ///
    /// #Examples
    /// ```
    /// use rs_poker::core::Suit;
    /// let idx = Suit::Club as u8;
    /// assert_eq!(Suit::Club, Suit::from_u8(idx));
    /// ```
    pub fn from_u8(s: u8) -> Self {
        Self::from(s)
    }

    /// Given a character that represents a suit try and parse that char.
    /// If the char can represent a suit return it.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::core::Suit;
    ///
    /// let s = Suit::from_char('s');
    /// assert_eq!(Some(Suit::Spade), s);
    /// ```
    ///
    /// ```
    /// use rs_poker::core::Suit;
    ///
    /// let s = Suit::from_char('X');
    /// assert_eq!(None, s);
    /// ```
    pub fn from_char(s: char) -> Option<Self> {
        TryFrom::try_from(s).ok()
    }

    /// This Suit to a character.
    pub fn to_char(self) -> char {
        char::from(self)
    }
}

impl From<u8> for Suit {
    fn from(value: u8) -> Self {
        unsafe { mem::transmute(cmp::min(value, Self::Diamond as u8)) }
    }
}

impl From<Suit> for u8 {
    fn from(value: Suit) -> Self {
        value as u8
    }
}

impl TryFrom<char> for Suit {
    type Error = RSPokerError;

    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value.to_ascii_lowercase() {
            'd' => Ok(Self::Diamond),
            's' => Ok(Self::Spade),
            'h' => Ok(Self::Heart),
            'c' => Ok(Self::Club),
            _ => Err(RSPokerError::UnexpectedSuitChar),
        }
    }
}

impl From<Suit> for char {
    fn from(value: Suit) -> Self {
        match value {
            Suit::Diamond => 'd',
            Suit::Spade => 's',
            Suit::Heart => 'h',
            Suit::Club => 'c',
        }
    }
}

/// The main struct of this library.
/// This is a carrier for Suit and Value combined.
#[derive(PartialEq, PartialOrd, Eq, Ord, Clone, Copy, Hash)]
pub struct Card {
    /// The face value of this card.
    pub value: Value,
    /// The suit of this card.
    pub suit: Suit,
}

impl Card {
    /// Given a value and a suit create a new card.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::core::{Card, Suit, Value};
    /// let c = Card::new(Value::Ace, Suit::Spade);
    /// assert_eq!(Value::Ace, c.value);
    /// assert_eq!(Suit::Spade, c.suit);
    /// ```
    pub fn new(value: Value, suit: Suit) -> Self {
        Self { value, suit }
    }
}

impl From<Card> for u8 {
    fn from(card: Card) -> Self {
        u8::from(card.suit) * 13 + u8::from(card.value)
    }
}

impl From<u8> for Card {
    fn from(value: u8) -> Self {
        Self {
            value: Value::from(value % 13),
            suit: Suit::from(value / 13),
        }
    }
}

impl fmt::Debug for Card {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Card({}{})",
            char::from(self.value),
            char::from(self.suit)
        )
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", char::from(self.value), char::from(self.suit))
    }
}

impl TryFrom<&str> for Card {
    type Error = RSPokerError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let mut chars = value.chars();
        let value_char = chars.next().ok_or(RSPokerError::TooFewChars)?;
        let suit_char = chars.next().ok_or(RSPokerError::TooFewChars)?;
        Ok(Self {
            value: Value::try_from(value_char)?,
            suit: Suit::try_from(suit_char)?,
        })
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for Card {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Use existing Display implementation which formats as "Ah", "Kc", etc
        serializer.serialize_str(&self.to_string())
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Card {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Card::try_from(s.as_str()).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constructor() {
        let c = Card {
            value: Value::Three,
            suit: Suit::Spade,
        };
        assert_eq!(Suit::Spade, c.suit);
        assert_eq!(Value::Three, c.value);
    }

    #[test]
    fn test_suit_from_u8() {
        assert_eq!(Suit::Spade, Suit::from_u8(0));
        assert_eq!(Suit::Club, Suit::from_u8(1));
        assert_eq!(Suit::Heart, Suit::from_u8(2));
        assert_eq!(Suit::Diamond, Suit::from_u8(3));
    }

    #[test]
    fn test_value_from_u8() {
        assert_eq!(Value::Two, Value::from_u8(0));
        assert_eq!(Value::Ace, Value::from_u8(12));
    }

    #[test]
    fn test_roundtrip_from_u8_all_cards() {
        for suit in SUITS {
            for value in VALUES {
                let c = Card { suit, value };
                let u = u8::from(c);
                assert_eq!(c, Card::from(u));
            }
        }
    }

    #[test]
    fn test_try_parse_card() {
        let expected = Card {
            value: Value::King,
            suit: Suit::Spade,
        };

        assert_eq!(expected, Card::try_from("Ks").unwrap())
    }

    #[test]
    fn test_parse_all_cards() {
        for suit in SUITS {
            for value in VALUES {
                let e = Card { suit, value };
                let card_string = format!("{}{}", char::from(value), char::from(suit));
                assert_eq!(e, Card::try_from(card_string.as_str()).unwrap());
            }
        }
    }

    #[test]
    fn test_compare() {
        let c1 = Card {
            value: Value::Three,
            suit: Suit::Spade,
        };
        let c2 = Card {
            value: Value::Four,
            suit: Suit::Spade,
        };
        let c3 = Card {
            value: Value::Four,
            suit: Suit::Club,
        };

        // Make sure that the values are ordered
        assert!(c1 < c2);
        assert!(c2 > c1);
        // Make sure that suit is used.
        assert!(c3 > c2);
    }

    #[test]
    fn test_value_cmp() {
        assert!(Value::Two < Value::Ace);
        assert!(Value::King < Value::Ace);
        assert_eq!(Value::Two, Value::Two);
    }

    #[test]
    fn test_from_u8() {
        assert_eq!(Value::Two, Value::from_u8(0));
        assert_eq!(Value::Ace, Value::from_u8(12));
    }

    #[test]
    fn test_size_card() {
        // Card should be really small. Hopefully just two u8's
        assert!(mem::size_of::<Card>() <= 2);
    }

    #[test]
    fn test_size_suit() {
        // One byte for Suit
        assert!(mem::size_of::<Suit>() <= 1);
    }

    #[test]
    fn test_size_value() {
        // One byte for Value
        assert!(mem::size_of::<Value>() <= 1);
    }

    #[test]
    fn test_gap() {
        // test on gap
        assert!(1 == Value::Ace.gap(Value::King));
        // test no gap at the high end
        assert!(0 == Value::Ace.gap(Value::Ace));
        // test no gap at the low end
        assert!(0 == Value::Two.gap(Value::Two));
        // Test one gap at the low end
        assert!(1 == Value::Two.gap(Value::Three));
        // test that ordering doesn't matter
        assert!(1 == Value::Three.gap(Value::Two));
        // Test things that are far apart
        assert!(12 == Value::Ace.gap(Value::Two));
        assert!(12 == Value::Two.gap(Value::Ace));
    }

    #[test]
    fn test_suit_to_char() {
        let s = Suit::Spade;
        assert_eq!('s', s.to_char());

        let s = Suit::Club;
        assert_eq!('c', s.to_char());

        let s = Suit::Heart;
        assert_eq!('h', s.to_char());

        let s = Suit::Diamond;
        assert_eq!('d', s.to_char());
    }

    #[test]
    fn test_value_to_char() {
        let v = Value::Ace;
        assert_eq!('A', v.to_char());

        let v = Value::King;
        assert_eq!('K', v.to_char());

        let v = Value::Queen;
        assert_eq!('Q', v.to_char());

        let v = Value::Jack;
        assert_eq!('J', v.to_char());

        let v = Value::Ten;
        assert_eq!('T', v.to_char());

        for i in 2..=9 {
            // The eunm is 0 based so the first value is 0 while cards
            // start at 2
            let v = Value::from(i - 2);
            assert_eq!(char::from_digit(u32::from(i), 10).unwrap(), v.to_char());
        }
    }

    /// Verifies that Display for Card produces the expected string representation.
    #[test]
    fn test_card_display() {
        let c = Card {
            value: Value::Ace,
            suit: Suit::Spade,
        };
        assert_eq!("As", c.to_string());

        let c = Card {
            value: Value::King,
            suit: Suit::Heart,
        };
        assert_eq!("Kh", c.to_string());

        let c = Card {
            value: Value::Two,
            suit: Suit::Diamond,
        };
        assert_eq!("2d", c.to_string());

        let c = Card {
            value: Value::Ten,
            suit: Suit::Club,
        };
        assert_eq!("Tc", c.to_string());
    }
}
