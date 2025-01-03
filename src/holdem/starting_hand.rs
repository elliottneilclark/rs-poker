use crate::core::{Card, FlatHand, Suit, Value};

/// Enum to represent how the suits of a hand correspond to each other.
/// `Suitedness::Suited` will mean that all cards have the same suit
/// `Suitedness::OffSuit` will mean that all cards have the different suit
/// `Suitedness::Any` makes no promises.
#[derive(Debug, Eq, PartialEq, PartialOrd, Ord, Clone, Copy)]
pub enum Suitedness {
    /// All of the cards are the same suit
    Suited,
    /// None of the cards are the same suit
    OffSuit,
    /// No promises about suit.
    Any,
}

/// `HoldemStartingHand` represents the two card starting hand of texas holdem.
/// It can generate all the possible actual starting hands.
///
/// Give two values and if you only want suited variants.
#[derive(Debug, Eq, PartialEq, PartialOrd, Ord, Clone)]
pub struct Default {
    /// The first value.
    value_one: Value,
    /// The second value.
    value_two: Value,
    /// should we only consider possible starting hands of the same suit?
    suited: Suitedness,
}

impl Default {
    /// Is this starting hand a pocket pair?
    fn is_pair(&self) -> bool {
        self.value_one == self.value_two
    }

    /// Create a new vector of all suited hands.
    fn create_suited(&self) -> Vec<FlatHand> {
        // Can't have a suited pair. Not unless you're cheating.
        if self.is_pair() {
            return vec![];
        }
        Suit::suits()
            .iter()
            .map(|s| {
                FlatHand::new_with_cards(vec![
                    Card {
                        value: self.value_one,
                        suit: *s,
                    },
                    Card {
                        value: self.value_two,
                        suit: *s,
                    },
                ])
            })
            .collect()
    }

    /// Create a new vector of all the off suit hands.
    fn create_offsuit(&self) -> Vec<FlatHand> {
        // Since the values are the same there is no reason to swap the suits.
        let expected_hands = if self.is_pair() { 6 } else { 12 };
        self.append_offsuit(Vec::with_capacity(expected_hands))
    }

    /// Append all the off suit hands to the passed in vec and
    /// then return it.
    ///
    /// @returns the passed in vector with offsuit hands appended.
    fn append_offsuit(&self, mut hands: Vec<FlatHand>) -> Vec<FlatHand> {
        let suits = Suit::suits();
        for (i, suit_one) in suits.iter().enumerate() {
            for suit_two in &suits[i + 1..] {
                // Push the hands in.
                hands.push(FlatHand::new_with_cards(vec![
                    Card {
                        value: self.value_one,
                        suit: *suit_one,
                    },
                    Card {
                        value: self.value_two,
                        suit: *suit_two,
                    },
                ]));

                // If this isn't a pair then the flipped suits is needed.
                if self.value_one != self.value_two {
                    hands.push(FlatHand::new_with_cards(vec![
                        Card {
                            value: self.value_one,
                            suit: *suit_two,
                        },
                        Card {
                            value: self.value_two,
                            suit: *suit_one,
                        },
                    ]));
                }
            }
        }
        hands
    }

    /// Get all the possible starting hands represented by the
    /// two values of this starting hand.
    fn possible_hands(&self) -> Vec<FlatHand> {
        match self.suited {
            Suitedness::Suited => self.create_suited(),
            Suitedness::OffSuit => self.create_offsuit(),
            Suitedness::Any => self.append_offsuit(self.create_suited()),
        }
    }
}

/// Starting hand struct to represent where it's one
/// static card and a range for the other.
#[derive(Debug, PartialEq, Eq, PartialOrd)]
pub struct SingleCardRange {
    /// First value; this one will not change.
    value_one: Value,
    /// Inclusive start range
    start: Value,
    /// Inclusive end range
    end: Value,
    /// What Suits can this have.
    suited: Suitedness,
}

impl SingleCardRange {
    /// Generate all the possible hands for this starting hand type.
    fn possible_hands(&self) -> Vec<FlatHand> {
        let mut cur_value = self.start;
        let mut hands = vec![];
        // TODO: Make a better iterator for values.
        while cur_value <= self.end {
            let mut new_hands = Default {
                value_one: self.value_one,
                value_two: cur_value,
                suited: self.suited,
            }
            .possible_hands();
            hands.append(&mut new_hands);
            cur_value = Value::from_u8(cur_value as u8 + 1);
        }

        hands
    }
}

/// Enum to represent all the possible ways to specify a starting hand.
#[derive(Debug, PartialEq, Eq, PartialOrd)]
pub enum StartingHand {
    /// Default starting hand type. This means that we
    /// specify two cards and their suitedness.
    Def(Default),

    /// A starting hand where the second card is a range.
    SingleCardRange(SingleCardRange),
}

impl StartingHand {
    /// Create a default starting hand with two `Value`'s and a `Suitedness`.
    pub fn default(value_one: Value, value_two: Value, suited: Suitedness) -> Self {
        Self::Def(Default {
            value_one,
            value_two,
            suited,
        })
    }

    /// Create a new StartingHand with the second card being a range.
    pub fn single_range(value_one: Value, start: Value, end: Value, suited: Suitedness) -> Self {
        Self::SingleCardRange(SingleCardRange {
            value_one,
            start,
            end,
            suited,
        })
    }

    /// Create every possible unique StartingHand.
    pub fn all() -> Vec<Self> {
        let mut hands = Vec::with_capacity(169);
        let values = Value::values();
        for (i, value_one) in values.iter().enumerate() {
            for value_two in &values[i..] {
                hands.push(Self::Def(Default {
                    value_one: *value_one,
                    value_two: *value_two,
                    suited: Suitedness::OffSuit,
                }));
                if value_one != value_two {
                    hands.push(Self::Def(Default {
                        value_one: *value_one,
                        value_two: *value_two,
                        suited: Suitedness::Suited,
                    }));
                }
            }
        }
        hands
    }

    /// From a `StartingHand` specify all the hands this could represent.
    pub fn possible_hands(&self) -> Vec<FlatHand> {
        match *self {
            Self::Def(ref h) => h.possible_hands(),
            Self::SingleCardRange(ref h) => h.possible_hands(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aces() {
        let sh = Default {
            value_one: Value::Ace,
            value_two: Value::Ace,
            suited: Suitedness::OffSuit,
        };
        assert!(6 == sh.possible_hands().len());
    }

    #[test]
    fn test_suited_connector() {
        let sh = Default {
            value_one: Value::Ace,
            value_two: Value::King,
            suited: Suitedness::Suited,
        };
        assert!(4 == sh.possible_hands().len());
    }
    #[test]
    fn test_unsuited_connector() {
        let sh = Default {
            value_one: Value::Ace,
            value_two: Value::King,
            suited: Suitedness::OffSuit,
        };
        assert!(12 == sh.possible_hands().len());
    }

    #[test]
    fn test_starting_hand_count() {
        let num_to_test: usize = StartingHand::all()
            .iter()
            .map(|h| h.possible_hands().len())
            .sum();
        assert!(1326 == num_to_test);
    }
}
