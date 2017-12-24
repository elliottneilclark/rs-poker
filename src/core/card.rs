use std::mem;
use std::cmp;
use std::ascii::AsciiExt;
use std::fmt;

/// Card rank or value.
/// This is basically the face value - 2
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
    pub fn from_u8(v: u8) -> Value {
        unsafe { mem::transmute(cmp::min(v, Value::Ace as u8)) }
    }
    /// Get all of the `Value`'s that are possible.
    /// This is used to iterate through all possible
    /// values when creating a new deck, or
    /// generating all possible starting hands.
    pub fn values() -> [Value; 13] {
        VALUES
    }

    /// Given a character parse that char into a value.
    /// Case is ignored as long as the char is in the ascii range (It should be).
    /// @returns None if there's no value there.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::core::Value;
    ///
    /// assert_eq!(Value::Ace, Value::from_char('A').unwrap());
    /// ```
    pub fn from_char(c: char) -> Option<Value> {
        match c.to_ascii_uppercase() {
            'A' => Some(Value::Ace),
            'K' => Some(Value::King),
            'Q' => Some(Value::Queen),
            'J' => Some(Value::Jack),
            'T' => Some(Value::Ten),
            '9' => Some(Value::Nine),
            '8' => Some(Value::Eight),
            '7' => Some(Value::Seven),
            '6' => Some(Value::Six),
            '5' => Some(Value::Five),
            '4' => Some(Value::Four),
            '3' => Some(Value::Three),
            '2' => Some(Value::Two),
            _ => None,
        }
    }

    /// Convert this Value to a char.
    pub fn to_char(&self) -> char {
        match *self {
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

    /// How card ranks seperate the two values.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::core::Value;
    /// assert_eq!(1, Value::Ace.gap(&Value::King));
    /// ```
    pub fn gap(&self, other: &Value) -> u8 {
        let min = cmp::min(*self as u8, *other as u8);
        let max = cmp::max(*self as u8, *other as u8);
        max - min
    }
}

/// Enum for the four different suits.
/// While this has support for ordering it's not
/// sensical. The sorting is only there to allow sorting cards.
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
    pub fn suits() -> [Suit; 4] {
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
    pub fn from_u8(s: u8) -> Suit {
        unsafe { mem::transmute(cmp::min(s, Suit::Diamond as u8)) }
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
    pub fn from_char(s: char) -> Option<Suit> {
        match s.to_ascii_lowercase() {
            'd' => Some(Suit::Diamond),
            's' => Some(Suit::Spade),
            'h' => Some(Suit::Heart),
            'c' => Some(Suit::Club),
            _ => None,
        }
    }

    /// This Suit to a character.
    pub fn to_char(&self) -> char {
        match *self {
            Suit::Diamond => 'd',
            Suit::Spade => 's',
            Suit::Heart => 'h',
            Suit::Club => 'c',
        }
    }
}

/// The main struct of this library.
/// This is a carrier for Suit and Value combined.
#[derive(PartialEq, PartialOrd, Eq, Ord, Debug, Clone, Copy, Hash)]
pub struct Card {
    /// The face value of this card.
    pub value: Value,
    /// The suit of this card.
    pub suit: Suit,
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", self.value.to_char(), self.suit.to_char())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

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

        // Make sure that equals works
        assert!(c1 == c1);
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
        assert!(1 == Value::Ace.gap(&Value::King));
        // test no gap at the high end
        assert!(0 == Value::Ace.gap(&Value::Ace));
        // test no gap at the low end
        assert!(0 == Value::Two.gap(&Value::Two));
        // Test one gap at the low end
        assert!(1 == Value::Two.gap(&Value::Three));
        // test that ordering doesn't matter
        assert!(1 == Value::Three.gap(&Value::Two));
        // Test things that are far apart
        assert!(12 == Value::Ace.gap(&Value::Two));
        assert!(12 == Value::Two.gap(&Value::Ace));
    }
}
