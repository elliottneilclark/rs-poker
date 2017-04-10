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

    /// This will rank 7 card hands.
    fn rank_seven(&self) -> Rank {
        let mut value_to_count: [u8; 13] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut count_to_value: [u32; 5] = [0, 0, 0, 0, 0];
        let mut suit_value_sets: [u32; 4] = [0, 0, 0, 0];
        let mut value_set: u32 = 0;

        for c in self.cards() {
            let v = c.value as u8;
            let s = c.suit as u8;
            value_set |= 1 << v;
            value_to_count[v as usize] += 1;
            suit_value_sets[s as usize] |= 1 << v;
        }

        // Now rotate the value to count map.
        for (value, &count) in value_to_count.iter().enumerate() {
            count_to_value[count as usize] |= 1 << value;
        }

        // Find out if there's a flush
        let flush: Option<usize> = self.find_flush(&suit_value_sets);

        // If this is a flush then it could be a straight flush
        // or a flush. So check only once.
        if let Some(flush_idx) = flush {
            // If we can find a straight in the flush then it's s flush
            if let Some(rank) = self.rank_straight(suit_value_sets[flush_idx]) {
                Rank::StraightFlush(rank)
            } else {
                // Else it's just a normal flush
                let rank = self.keep_n(suit_value_sets[flush_idx], 5);
                Rank::Flush(rank)
            }
        } else if count_to_value[4] != 0 {
            // Four of a kind.
            let high = self.keep_highest(value_set ^ count_to_value[4]);
            Rank::FourOfAKind(count_to_value[4] << 13 | high)
        } else if count_to_value[3] != 0 && count_to_value[3].count_ones() == 2 {
            // There are two sets. So the best we can make is a full house.
            let set = self.keep_highest(count_to_value[3]);
            let pair = count_to_value[3] ^ set;
            Rank::FullHouse(set << 13 | pair)
        } else if count_to_value[3] != 0 && count_to_value[2] != 0 {
            // there is a pair and a set.
            let set = count_to_value[3];
            let pair = self.keep_highest(count_to_value[2]);
            Rank::FullHouse(set << 13 | pair)
        } else if let Some(s_rank) = self.rank_straight(value_set) {
            // If there's a straight return it now.
            Rank::Straight(s_rank)
        } else if count_to_value[3] != 0 {
            // if there is a set then we need to keep 2 cards that
            // aren't in the set.
            let low = self.keep_n(value_set ^ count_to_value[3], 2);
            Rank::ThreeOfAKind(count_to_value[3] << 13 | low)
        } else if count_to_value[2].count_ones() >= 2 {
            // Two pair
            //
            // That can be because we have 3 pairs and a high card.
            // Or we could have two pair and two high cards.
            let pairs = self.keep_n(count_to_value[2], 2);
            let low = self.keep_highest(value_set ^ pairs);
            Rank::TwoPair(pairs << 13 | low)
        } else if count_to_value[2] == 0 {
            // This means that there's no pair
            // no sets, no straights, no flushes, so only a
            // high cards.
            Rank::HighCard(self.keep_n(value_set, 5))
        } else {
            // Otherwise there's only one pair.
            let pair = count_to_value[2];
            // Keep the highest three cards not in the pair.
            let low = self.keep_n(value_set ^ count_to_value[2], 3);
            Rank::OnePair(pair << 13 | low)
        }
    }

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
                match (self.rank_straight(value_set), is_flush) {
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
    fn rank_straight(&self, hand_rank: u32) -> Option<u32> {
        // We do a bunch of if/else statemens as there could be more than
        // 5 different bits set so the straight might not be equal to the
        // straight mask without masking them off.
        if hand_rank & STRAIGHT9 == STRAIGHT9 {
            Some(9)
        } else if hand_rank & STRAIGHT8 == STRAIGHT8 {
            Some(8)
        } else if hand_rank & STRAIGHT7 == STRAIGHT7 {
            Some(7)
        } else if hand_rank & STRAIGHT6 == STRAIGHT6 {
            Some(6)
        } else if hand_rank & STRAIGHT5 == STRAIGHT5 {
            Some(5)
        } else if hand_rank & STRAIGHT4 == STRAIGHT4 {
            Some(4)
        } else if hand_rank & STRAIGHT3 == STRAIGHT3 {
            Some(3)
        } else if hand_rank & STRAIGHT2 == STRAIGHT2 {
            Some(2)
        } else if hand_rank & STRAIGHT1 == STRAIGHT1 {
            Some(1)
        } else if hand_rank & STRAIGHT0 == STRAIGHT0 {
            Some(0)
        } else {
            None
        }
    }
    /// Keep only the most signifigant bit.
    fn keep_highest(&self, rank: u32) -> u32 {
        1 << (32 - rank.leading_zeros() - 1)
    }
    /// Keep the N most signifigant bits.
    ///
    /// This works by removing the least signifigant bits.
    fn keep_n(&self, rank: u32, to_keep: u32) -> u32 {
        let mut result = rank;
        while result.count_ones() > to_keep {
            result &= result - 1;
        }
        result
    }
    /// From a slice of values sets find if there's one that has a
    /// flush
    fn find_flush(&self, suit_value_sets: &[u32]) -> Option<usize> {
        for (i, sv) in suit_value_sets.iter().enumerate() {
            if sv.count_ones() >= 5 {
                return Some(i);
            }
        }
        None
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
        let hand = Hand::new_from_str("Ad8h9cTc5c").unwrap();
        let rank = 1 << Value::Ace as u32 | 1 << Value::Eight as u32 | 1 << Value::Nine as u32 |
                   1 << Value::Ten as u32 |
                   1 << Value::Five as u32;

        assert!(Rank::HighCard(rank) == hand.rank());
    }

    #[test]
    fn test_flush() {
        let hand = Hand::new_from_str("Ad8d9dTd5d").unwrap();
        let rank = 1 << Value::Ace as u32 | 1 << Value::Eight as u32 | 1 << Value::Nine as u32 |
                   1 << Value::Ten as u32 |
                   1 << Value::Five as u32;

        assert!(Rank::Flush(rank) == hand.rank());
    }

    #[test]
    fn test_full_house() {
        let hand = Hand::new_from_str("AdAc9d9c9s").unwrap();
        let rank = (1 << (Value::Nine as u32)) << 13 | 1 << (Value::Ace as u32);
        assert!(Rank::FullHouse(rank) == hand.rank());
    }

    #[test]
    fn test_two_pair() {
        // Make a two pair hand.
        let hand = Hand::new_from_str("AdAc9D9cTs").unwrap();
        let rank = (1 << Value::Ace as u32 | 1 << Value::Nine as u32) << 13 |
                   1 << Value::Ten as u32;
        assert!(Rank::TwoPair(rank) == hand.rank());
    }

    #[test]
    fn test_one_pair() {
        let hand = Hand::new_from_str("AdAc9d8cTs").unwrap();
        let rank = (1 << Value::Ace as u32) << 13 | 1 << Value::Nine as u32 |
                   1 << Value::Eight as u32 | 1 << Value::Ten as u32;

        assert!(Rank::OnePair(rank) == hand.rank());
    }

    #[test]
    fn test_four_of_a_kind() {
        let hand = Hand::new_from_str("AdAcAsAhTs").unwrap();
        assert!(Rank::FourOfAKind((1 << (Value::Ace as u32) << 13) | 1 << (Value::Ten as u32)) ==
                hand.rank());
    }

    #[test]
    fn test_wheel() {
        let hand = Hand::new_from_str("Ad2c3s4h5s").unwrap();
        assert!(Rank::Straight(0) == hand.rank());
    }


    #[test]
    fn test_straight() {
        let hand = Hand::new_from_str("2c3s4h5s6d").unwrap();
        assert!(Rank::Straight(1) == hand.rank());
    }

    #[test]
    fn test_three_of_a_kind() {
        let hand = Hand::new_from_str("2c2s2h5s6d").unwrap();
        let rank = (1 << (Value::Two as u32)) << 13 | 1 << (Value::Five as u32) |
                   1 << (Value::Six as u32);
        assert!(Rank::ThreeOfAKind(rank) == hand.rank());
    }

    #[test]
    fn test_rank_seven_straight_flush() {
        let h = Hand::new_from_str("AdKdQdJdTd9d8d").unwrap();
        assert_eq!(Rank::StraightFlush(9), h.rank_seven());
    }

    #[test]
    fn test_rank_seven_four_kind() {
        let h = Hand::new_from_str("2s2h2d2cKd9h4s").unwrap();
        let four_rank = (1 << Value::Two as u32) << 13;
        let low_rank = 1 << Value::King as u32;
        assert_eq!(Rank::FourOfAKind(four_rank | low_rank), h.rank_seven());
    }

    #[test]
    fn test_rank_seven_four_plus_set() {
        // Four of a kind plus a set.
        let h = Hand::new_from_str("2s2h2d2c8d8s8c").unwrap();
        let four_rank = (1 << Value::Two as u32) << 13;
        let low_rank = 1 << Value::Eight as u32;
        assert_eq!(Rank::FourOfAKind(four_rank | low_rank), h.rank_seven());
    }

    #[test]
    fn test_rank_seven_full_house_two_sets() {
        // We have two sets use the highest set.
        let h = Hand::new_from_str("As2h2d2c8d8s8c").unwrap();
        let set_rank = (1 << Value::Eight as u32) << 13;
        let low_rank = 1 << Value::Two as u32;
        assert_eq!(Rank::FullHouse(set_rank | low_rank), h.rank_seven());
    }

    #[test]
    fn test_rank_seven_full_house_two_pair() {
        // Test to make sure that we pick the best pair.
        let h = Hand::new_from_str("2h2d2c8d8sKdKs").unwrap();
        let set_rank = (1 << Value::Two as u32) << 13;
        let low_rank = 1 << Value::King as u32;
        assert_eq!(Rank::FullHouse(set_rank | low_rank), h.rank_seven());
    }

    #[test]
    fn test_two_pair_from_three_pair() {
        let h = Hand::new_from_str("2h2d8d8sKdKsTh").unwrap();
        let pair_rank = ((1 << Value::King as u32) | (1 << Value::Eight as u32)) << 13;
        let low_rank = 1 << Value::Ten as u32;
        assert_eq!(Rank::TwoPair(pair_rank | low_rank), h.rank_seven());
    }


    #[test]
    fn test_rank_seven_two_pair() {
        let h = Hand::new_from_str("2h2d8d8sKd6sTh").unwrap();
        let pair_rank = ((1 << Value::Two as u32) | (1 << Value::Eight as u32)) << 13;
        let low_rank = 1 << Value::King as u32;
        assert_eq!(Rank::TwoPair(pair_rank | low_rank), h.rank_seven());
    }
}
