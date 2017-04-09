use core::hand::Hand;
use core::card::Value;
use core::card::Card;

/// All the different possible hand ranks.
/// For each hand rank the u32 corresponds to
/// the strength of the hand in comparison to others
/// of the same rank.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub enum Rank {
    /// The lowest rank.
    /// No matches
    HighCard(u32),
    /// One Card matches another.
    OnePair(u32),
    /// Two diffent pair of matching cards.
    TwoPair(u32),
    /// Three of the same value.
    ThreeOfAKind(u32),
    /// Five cards in a sequence
    Straight(u32),
    /// Five cards of the same suit
    Flush(u32),
    /// Three of one value and two of another value
    FullHouse(u32),
    /// Four of the same value.
    FourOfAKind(u32),
    /// Five cards in a sequence all for the same suit.
    StraightFlush(u32),
}

/// Big ugly constant for all the straghts.
const STRAIGHT0: u32 = 1 << (Value::Ace as u32) | 1 << (Value::Two as u32) |
                       1 << (Value::Three as u32) |
                       1 << (Value::Four as u32) | 1 << (Value::Five as u32);
/// "Normal" straights starting at two to six.
const STRAIGHT1: u32 = 1 << (Value::Two as u32) | 1 << (Value::Three as u32) |
                       1 << (Value::Four as u32) |
                       1 << (Value::Five as u32) | 1 << (Value::Six as u32);
/// Three to Seven
const STRAIGHT2: u32 = 1 << (Value::Three as u32) | 1 << (Value::Four as u32) |
                       1 << (Value::Five as u32) |
                       1 << (Value::Six as u32) | 1 << (Value::Seven as u32);
/// Four to Eight
const STRAIGHT3: u32 =
    1 << (Value::Four as u32) | 1 << (Value::Five as u32) | 1 << (Value::Six as u32) |
    1 << (Value::Seven as u32) | 1 << (Value::Eight as u32);
/// Five to Nine
const STRAIGHT4: u32 =
    1 << (Value::Five as u32) | 1 << (Value::Six as u32) | 1 << (Value::Seven as u32) |
    1 << (Value::Eight as u32) | 1 << (Value::Nine as u32);
/// Six to Ten
const STRAIGHT5: u32 = 1 << (Value::Six as u32) | 1 << (Value::Seven as u32) |
                       1 << (Value::Eight as u32) |
                       1 << (Value::Nine as u32) | 1 << (Value::Ten as u32);
/// Seven to Jack.
const STRAIGHT6: u32 = 1 << (Value::Seven as u32) | 1 << (Value::Eight as u32) |
                       1 << (Value::Nine as u32) |
                       1 << (Value::Ten as u32) | 1 << (Value::Jack as u32);
/// Eight to Queen
const STRAIGHT7: u32 = 1 << (Value::Eight as u32) | 1 << (Value::Nine as u32) |
                       1 << (Value::Ten as u32) | 1 << (Value::Jack as u32) |
                       1 << (Value::Queen as u32);
/// Nine to king
const STRAIGHT8: u32 =
    1 << (Value::Nine as u32) | 1 << (Value::Ten as u32) | 1 << (Value::Jack as u32) |
    1 << (Value::Queen as u32) | 1 << (Value::King as u32);
/// Royal straight
const STRAIGHT9: u32 = 1 << (Value::Ten as u32) | 1 << (Value::Jack as u32) |
                       1 << (Value::Queen as u32) |
                       1 << (Value::King as u32) | 1 << (Value::Ace as u32);

/// Can this turn into a hand rank?
pub trait Rankable {
    /// Rank the current 5 card hand.
    /// This will no cache the value.
    fn cards(&self) -> &[Card];

    /// Rank this hand. It doesn't do any caching so it's left up to the user
    /// to understand that duplicate work will be done if this is called more than once.
    fn rank(&self) -> Rank {
        // use for bitset
        let mut suit_set: u32 = 0;
        // Use for bitset
        let mut value_set: u32 = 0;
        let mut value_to_count: [u8; 13] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        // count => bitset of values.
        let mut count_to_value: [u32; 5] = [0, 0, 0, 0, 0];
        for c in self.cards() {
            let v = c.value as u8;
            let s = c.suit as u8;

            // Will be used for flush
            suit_set |= 1 << s;
            value_set |= 1 << v;
            // Keep track of counts for each card.
            value_to_count[v as usize] += 1;
        }

        // Now rotate the value to count map.
        for (value, &count) in value_to_count.iter().enumerate() {
            // Get the entry for the map, or insert it into the map.
            count_to_value[count as usize] |= 1 << value;
        }

        // The major deciding factor for hand rank
        // is the number of unique card values.
        let unique_card_count = value_set.count_ones();

        // Now that we should have all the information needed.
        // Lets do this.

        match unique_card_count {
            5 => {
                // If there are five different cards it can be a straight
                // a straight flush, a flush, or just a high card.
                // Need to check for all of them.
                let suit_count = suit_set.count_ones();
                let is_flush = suit_count == 1;
                match (self.rank_straight(&value_set), is_flush) {
                    // This is the most likely outcome.
                    // Not a flush and not a straight.
                    (None, false) => Rank::HighCard(value_set),
                    (Some(rank), false) => Rank::Straight(rank),
                    (None, true) => Rank::Flush(value_set),
                    (Some(rank), true) => Rank::StraightFlush(rank),
                }
            }
            4 => {
                // this is unique_card_count == 4
                // It is always one pair
                let major_rank = count_to_value[2];
                let minor_rank = value_set ^ major_rank;
                Rank::OnePair(major_rank << 13 | minor_rank)
            }
            3 => {
                // this can be three of a kind or two pair.
                let three_value = count_to_value[3];
                if three_value > 0 {
                    let major_rank = three_value;
                    let minor_rank = value_set ^ major_rank;
                    Rank::ThreeOfAKind(major_rank << 13 | minor_rank)
                } else {
                    // get the values of the pairs
                    let major_rank = count_to_value[2];
                    let minor_rank = value_set ^ major_rank;
                    Rank::TwoPair(major_rank << 13 | minor_rank)

                }
            }
            2 => {
                // This can either be full house, or four of a kind.
                let three_value = count_to_value[3];
                if three_value > 0 {
                    let major_rank = three_value;
                    // Remove the card that we have three of from the minor rank.
                    let minor_rank = value_set ^ major_rank;
                    // then join the two ranks
                    Rank::FullHouse(major_rank << 13 | minor_rank)
                } else {
                    let major_rank = count_to_value[4];
                    let minor_rank = value_set ^ major_rank;
                    Rank::FourOfAKind(major_rank << 13 | minor_rank)
                }
            }
            _ => unreachable!(),
        }
    }

    /// Given a bitset of hand ranks. This method
    /// will determine if there's a staright, and will give the
    /// rank. Wheel is the lowest, broadway is the highest value.
    ///
    /// Returns None if the hand ranks represented don't correspond
    /// to a straight.
    #[inline]
    fn rank_straight(&self, hand_rank: &u32) -> Option<u32> {
        match *hand_rank {
            STRAIGHT0 => Some(0),
            STRAIGHT1 => Some(1),
            STRAIGHT2 => Some(2),
            STRAIGHT3 => Some(3),
            STRAIGHT4 => Some(4),
            STRAIGHT5 => Some(5),
            STRAIGHT6 => Some(6),
            STRAIGHT7 => Some(7),
            STRAIGHT8 => Some(8),
            STRAIGHT9 => Some(9),
            _ => None,
        }
    }
}

/// Implementation for `Hand`
impl Rankable for Hand {
    fn cards(&self) -> &[Card] {
        &self[..]
    }
}
impl Rankable for Vec<Card> {
    fn cards(&self) -> &[Card] {
        &self[..]
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use core::hand::*;
    use core::card::*;


    #[test]
    fn test_cmp() {
        assert!(Rank::HighCard(0) < Rank::StraightFlush(0));
        assert!(Rank::HighCard(0) < Rank::FourOfAKind(0));
        assert!(Rank::HighCard(0) < Rank::ThreeOfAKind(0));
    }

    #[test]
    fn test_cmp_high() {
        assert!(Rank::HighCard(0) < Rank::HighCard(100));
    }

    #[test]
    fn test_high_card_hand() {
        let hand = Hand::new_with_cards(vec![Card {
                                                 value: Value::Ace,
                                                 suit: Suit::Diamond,
                                             },
                                             Card {
                                                 value: Value::Eight,
                                                 suit: Suit::Heart,
                                             },
                                             Card {
                                                 value: Value::Nine,
                                                 suit: Suit::Club,
                                             },
                                             Card {
                                                 value: Value::Ten,
                                                 suit: Suit::Club,
                                             },
                                             Card {
                                                 value: Value::Five,
                                                 suit: Suit::Club,
                                             }]);

        let rank = 1 << Value::Ace as u32 | 1 << Value::Eight as u32 | 1 << Value::Nine as u32 |
                   1 << Value::Ten as u32 |
                   1 << Value::Five as u32;

        assert!(Rank::HighCard(rank) == hand.rank());
    }

    #[test]
    fn test_flush() {
        let hand = Hand::new_with_cards(vec![Card {
                                                 value: Value::Ace,
                                                 suit: Suit::Diamond,
                                             },
                                             Card {
                                                 value: Value::Eight,
                                                 suit: Suit::Diamond,
                                             },
                                             Card {
                                                 value: Value::Nine,
                                                 suit: Suit::Diamond,
                                             },
                                             Card {
                                                 value: Value::Ten,
                                                 suit: Suit::Diamond,
                                             },
                                             Card {
                                                 value: Value::Five,
                                                 suit: Suit::Diamond,
                                             }]);

        let rank = 1 << Value::Ace as u32 | 1 << Value::Eight as u32 | 1 << Value::Nine as u32 |
                   1 << Value::Ten as u32 |
                   1 << Value::Five as u32;

        assert!(Rank::Flush(rank) == hand.rank());
    }

    #[test]
    fn test_full_house() {
        let hand = Hand::new_with_cards(vec![Card {
                                                 value: Value::Ace,
                                                 suit: Suit::Diamond,
                                             },
                                             Card {
                                                 value: Value::Ace,
                                                 suit: Suit::Club,
                                             },
                                             Card {
                                                 value: Value::Nine,
                                                 suit: Suit::Diamond,
                                             },
                                             Card {
                                                 value: Value::Nine,
                                                 suit: Suit::Club,
                                             },
                                             Card {
                                                 value: Value::Nine,
                                                 suit: Suit::Spade,
                                             }]);

        let rank = (1 << (Value::Nine as u32)) << 13 | 1 << (Value::Ace as u32);
        assert!(Rank::FullHouse(rank) == hand.rank());
    }

    #[test]
    fn test_two_pair() {
        // Make a two pair hand.
        let hand = Hand::new_with_cards(vec![Card {
                                                 value: Value::Ace,
                                                 suit: Suit::Diamond,
                                             },
                                             Card {
                                                 value: Value::Ace,
                                                 suit: Suit::Club,
                                             },
                                             Card {
                                                 value: Value::Nine,
                                                 suit: Suit::Diamond,
                                             },
                                             Card {
                                                 value: Value::Nine,
                                                 suit: Suit::Club,
                                             },
                                             Card {
                                                 value: Value::Ten,
                                                 suit: Suit::Spade,
                                             }]);

        let rank = (1 << Value::Ace as u32 | 1 << Value::Nine as u32) << 13 |
                   1 << Value::Ten as u32;
        assert!(Rank::TwoPair(rank) == hand.rank());
    }

    #[test]
    fn test_one_pair() {
        let hand = Hand::new_with_cards(vec![Card {
                                                 value: Value::Ace,
                                                 suit: Suit::Diamond,
                                             },
                                             Card {
                                                 value: Value::Ace,
                                                 suit: Suit::Club,
                                             },
                                             Card {
                                                 value: Value::Nine,
                                                 suit: Suit::Diamond,
                                             },
                                             Card {
                                                 value: Value::Eight,
                                                 suit: Suit::Club,
                                             },
                                             Card {
                                                 value: Value::Ten,
                                                 suit: Suit::Spade,
                                             }]);

        let rank = (1 << Value::Ace as u32) << 13 | 1 << Value::Nine as u32 |
                   1 << Value::Eight as u32 | 1 << Value::Ten as u32;

        assert!(Rank::OnePair(rank) == hand.rank());
    }

    #[test]
    fn test_four_of_a_kind() {
        let hand = Hand::new_with_cards(vec![Card {
                                                 value: Value::Ace,
                                                 suit: Suit::Diamond,
                                             },
                                             Card {
                                                 value: Value::Ace,
                                                 suit: Suit::Club,
                                             },
                                             Card {
                                                 value: Value::Ace,
                                                 suit: Suit::Spade,
                                             },
                                             Card {
                                                 value: Value::Ace,
                                                 suit: Suit::Heart,
                                             },
                                             Card {
                                                 value: Value::Ten,
                                                 suit: Suit::Spade,
                                             }]);

        assert!(Rank::FourOfAKind((1 << (Value::Ace as u32) << 13) | 1 << (Value::Ten as u32)) ==
                hand.rank());
    }

    #[test]
    fn test_wheel() {
        let hand = Hand::new_with_cards(vec![Card {
                                                 value: Value::Ace,
                                                 suit: Suit::Diamond,
                                             },
                                             Card {
                                                 value: Value::Two,
                                                 suit: Suit::Club,
                                             },
                                             Card {
                                                 value: Value::Three,
                                                 suit: Suit::Spade,
                                             },
                                             Card {
                                                 value: Value::Four,
                                                 suit: Suit::Heart,
                                             },
                                             Card {
                                                 value: Value::Five,
                                                 suit: Suit::Spade,
                                             }]);

        assert!(Rank::Straight(0) == hand.rank());
    }


    #[test]
    fn test_straight() {
        let hand = Hand::new_with_cards(vec![Card {
                                                 value: Value::Two,
                                                 suit: Suit::Club,
                                             },
                                             Card {
                                                 value: Value::Three,
                                                 suit: Suit::Spade,
                                             },
                                             Card {
                                                 value: Value::Four,
                                                 suit: Suit::Heart,
                                             },
                                             Card {
                                                 value: Value::Five,
                                                 suit: Suit::Spade,
                                             },
                                             Card {
                                                 value: Value::Six,
                                                 suit: Suit::Diamond,
                                             }]);

        assert!(Rank::Straight(1) == hand.rank());
    }

    #[test]
    fn test_three_of_a_kind() {
        let hand = Hand::new_with_cards(vec![Card {
                                                 value: Value::Two,
                                                 suit: Suit::Club,
                                             },
                                             Card {
                                                 value: Value::Two,
                                                 suit: Suit::Spade,
                                             },
                                             Card {
                                                 value: Value::Two,
                                                 suit: Suit::Heart,
                                             },
                                             Card {
                                                 value: Value::Five,
                                                 suit: Suit::Spade,
                                             },
                                             Card {
                                                 value: Value::Six,
                                                 suit: Suit::Diamond,
                                             }]);


        let rank = (1 << (Value::Two as u32)) << 13 | 1 << (Value::Five as u32) |
                   1 << (Value::Six as u32);

        assert!(Rank::ThreeOfAKind(rank) == hand.rank());
    }
}
