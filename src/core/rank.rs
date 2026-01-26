use crate::core::card::Card;

use super::{CardBitSet, FlatHand, Hand};

/// All the different possible hand ranks.
/// For each hand rank the u32 corresponds to
/// the strength of the hand in comparison to others
/// of the same rank.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy)]
pub enum Rank {
    /// The lowest rank.
    /// No matches
    HighCard(u32),
    /// One Card matches another.
    OnePair(u32),
    /// Two different pair of matching cards.
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Copy)]
pub enum CoreRank {
    HighCard,
    OnePair,
    TwoPair,
    ThreeOfAKind,
    Straight,
    Flush,
    FullHouse,
    FourOfAKind,
    StraightFlush,
}

/// Convert from Rank to CoreRank by stripping the u32 detail.
/// This is useful to reduce the cardinality of ranks.
///
/// For example displaying the possible outcomes of a hand
/// without caring about the specific rank values.
impl From<Rank> for CoreRank {
    fn from(rank: Rank) -> Self {
        match rank {
            Rank::HighCard(_) => CoreRank::HighCard,
            Rank::OnePair(_) => CoreRank::OnePair,
            Rank::TwoPair(_) => CoreRank::TwoPair,
            Rank::ThreeOfAKind(_) => CoreRank::ThreeOfAKind,
            Rank::Straight(_) => CoreRank::Straight,
            Rank::Flush(_) => CoreRank::Flush,
            Rank::FullHouse(_) => CoreRank::FullHouse,
            Rank::FourOfAKind(_) => CoreRank::FourOfAKind,
            Rank::StraightFlush(_) => CoreRank::StraightFlush,
        }
    }
}

/// Bit mask for the wheel (Ace, two, three, four, five)
const WHEEL: u32 = 0b1_0000_0000_1111;
/// Given a bitset of hand ranks. This method
/// will determine if there's a straight, and will give the
/// rank. Wheel is the lowest, broadway is the highest value.
///
/// Returns None if the hand ranks represented don't correspond
/// to a straight.
fn rank_straight(value_set: u32) -> Option<u32> {
    // Example of something with a straight:
    //       0000111111100
    //       0001111111000
    //       0011111110000
    //       0111111100000
    //       1111111000000
    //       -------------
    //       0000111000000
    //
    // So there were seven ones in a row
    // we removed the bottom 4.
    //
    // Now an example of an almost straight:
    //
    //       0001110111100
    //       0011101111000
    //       0111011110000
    //       1110111100000
    //       1101111000000
    //       -------------
    //       0000000000000
    let left =
        value_set & (value_set << 1) & (value_set << 2) & (value_set << 3) & (value_set << 4);
    // Now count the leading 0's
    let idx = left.leading_zeros();
    // If this isn't all zeros then we found a straight
    if idx < 32 {
        Some(32 - 4 - idx)
    } else if value_set & WHEEL == WHEEL {
        // Check to see if this is the wheel. It's pretty unlikely.
        Some(0)
    } else {
        // We found nothing.
        None
    }
}
/// Keep only the most significant bit.
fn keep_highest(rank: u32) -> u32 {
    1 << (32 - rank.leading_zeros() - 1)
}
/// Keep the N most significant bits.
///
/// This works by removing the least significant bits.
fn keep_n(rank: u32, to_keep: u32) -> u32 {
    let mut result = rank;
    while result.count_ones() > to_keep {
        result &= result - 1;
    }
    result
}
/// From a slice of values sets find if there's one that has a
/// flush
fn find_flush(suit_value_sets: &[u32]) -> Option<usize> {
    suit_value_sets.iter().position(|sv| sv.count_ones() >= 5)
}
/// Can this turn into a hand rank? There are default implementations for
/// `Hand` and `Vec<Card>`.
pub trait Rankable {
    /// Rank the current 5 card hand.
    /// This will not cache the value.
    fn cards(&self) -> impl Iterator<Item = Card>;

    /// Rank the cards to find the best 5 card hand.
    /// This will work on 5 cards or more (specifically on 7 card holdem
    /// hands). If you know that the hand only contains 5 cards then
    /// `rank_five` will be faster.
    ///
    /// # Examples
    /// ```
    /// use rs_poker::core::{FlatHand, Rank, Rankable};
    ///
    /// let hand = FlatHand::new_from_str("2h2d8d8sKd6sTh").unwrap();
    /// let rank = hand.rank();
    /// assert!(Rank::TwoPair(0) <= rank);
    /// assert!(Rank::TwoPair(u32::max_value()) >= rank);
    /// ```
    fn rank(&self) -> Rank {
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
        let flush: Option<usize> = find_flush(&suit_value_sets);

        // If this is a flush then it could be a straight flush
        // or a flush. So check only once.
        if let Some(flush_idx) = flush {
            // If we can find a straight in the flush then it's a straight flush
            if let Some(rank) = rank_straight(suit_value_sets[flush_idx]) {
                Rank::StraightFlush(rank)
            } else {
                // Else it's just a normal flush
                let rank = keep_n(suit_value_sets[flush_idx], 5);
                Rank::Flush(rank)
            }
        } else if count_to_value[4] != 0 {
            // Four of a kind.
            let high = keep_highest(value_set ^ count_to_value[4]);
            Rank::FourOfAKind((count_to_value[4] << 13) | high)
        } else if count_to_value[3] != 0 && count_to_value[3].count_ones() == 2 {
            // There are two sets. So the best we can make is a full house.
            let set = keep_highest(count_to_value[3]);
            let pair = count_to_value[3] ^ set;
            Rank::FullHouse((set << 13) | pair)
        } else if count_to_value[3] != 0 && count_to_value[2] != 0 {
            // there is a pair and a set.
            let set = count_to_value[3];
            let pair = keep_highest(count_to_value[2]);
            Rank::FullHouse((set << 13) | pair)
        } else if let Some(s_rank) = rank_straight(value_set) {
            // If there's a straight return it now.
            Rank::Straight(s_rank)
        } else if count_to_value[3] != 0 {
            // if there is a set then we need to keep 2 cards that
            // aren't in the set.
            let low = keep_n(value_set ^ count_to_value[3], 2);
            Rank::ThreeOfAKind((count_to_value[3] << 13) | low)
        } else if count_to_value[2].count_ones() >= 2 {
            // Two pair
            //
            // That can be because we have 3 pairs and a high card.
            // Or we could have two pair and two high cards.
            let pairs = keep_n(count_to_value[2], 2);
            let low = keep_highest(value_set ^ pairs);
            Rank::TwoPair((pairs << 13) | low)
        } else if count_to_value[2] == 0 {
            // This means that there's no pair
            // no sets, no straights, no flushes, so only a
            // high card.
            Rank::HighCard(keep_n(value_set, 5))
        } else {
            // Otherwise there's only one pair.
            let pair = count_to_value[2];
            // Keep the highest three cards not in the pair.
            let low = keep_n(value_set ^ count_to_value[2], 3);
            Rank::OnePair((pair << 13) | low)
        }
    }

    /// Rank this hand. It doesn't do any caching so it's left up to the user
    /// to understand that duplicate work will be done if this is called more
    /// than once.
    fn rank_five(&self) -> Rank {
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
                match (rank_straight(value_set), is_flush) {
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
                Rank::OnePair((major_rank << 13) | minor_rank)
            }
            3 => {
                // this can be three of a kind or two pair.
                let three_value = count_to_value[3];
                if three_value > 0 {
                    let major_rank = three_value;
                    let minor_rank = value_set ^ major_rank;
                    Rank::ThreeOfAKind((major_rank << 13) | minor_rank)
                } else {
                    // get the values of the pairs
                    let major_rank = count_to_value[2];
                    let minor_rank = value_set ^ major_rank;
                    Rank::TwoPair((major_rank << 13) | minor_rank)
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
                    Rank::FullHouse((major_rank << 13) | minor_rank)
                } else {
                    let major_rank = count_to_value[4];
                    let minor_rank = value_set ^ major_rank;
                    Rank::FourOfAKind((major_rank << 13) | minor_rank)
                }
            }
            _ => unreachable!(),
        }
    }
}

/// Implementation for `Hand`
impl Rankable for FlatHand {
    fn cards(&self) -> impl Iterator<Item = Card> {
        self.iter().copied()
    }
}

impl Rankable for Vec<Card> {
    fn cards(&self) -> impl Iterator<Item = Card> {
        self.iter().copied()
    }
}

impl Rankable for [Card] {
    fn cards(&self) -> impl Iterator<Item = Card> {
        self.iter().copied()
    }
}

impl Rankable for &[Card] {
    fn cards(&self) -> impl Iterator<Item = Card> {
        self.iter().copied()
    }
}

impl Rankable for Hand {
    fn cards(&self) -> impl Iterator<Item = Card> {
        self.iter()
    }
}

impl Rankable for CardBitSet {
    fn cards(&self) -> impl Iterator<Item = Card> {
        self.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::card::*;
    use crate::core::flat_hand::*;

    #[test]
    fn test_keep_highest() {
        assert_eq!(0b100, keep_highest(0b111));
    }

    #[test]
    fn test_keep_n() {
        assert_eq!(3, keep_n(0b1111, 3).count_ones());
    }

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
        let hand = FlatHand::new_from_str("Ad8h9cTc5c").unwrap();
        let rank = (1 << Value::Ace as u32)
            | (1 << Value::Eight as u32)
            | (1 << Value::Nine as u32)
            | (1 << Value::Ten as u32)
            | (1 << Value::Five as u32);

        assert!(Rank::HighCard(rank) == hand.rank_five());
    }

    #[test]
    fn test_can_rank_two_card_hand() {
        let hand = FlatHand::new_from_str("Ad8h").unwrap();
        let rank = (1 << Value::Ace as u32) | (1 << Value::Eight as u32);
        assert!(Rank::HighCard(rank) == hand.rank());
    }

    #[test]
    fn test_flush() {
        let hand = FlatHand::new_from_str("Ad8d9dTd5d").unwrap();
        let rank = (1 << Value::Ace as u32)
            | (1 << Value::Eight as u32)
            | (1 << Value::Nine as u32)
            | (1 << Value::Ten as u32)
            | (1 << Value::Five as u32);

        assert!(Rank::Flush(rank) == hand.rank_five());
    }

    #[test]
    fn test_full_house() {
        let hand = FlatHand::new_from_str("AdAc9d9c9s").unwrap();
        let rank = ((1 << (Value::Nine as u32)) << 13) | (1 << (Value::Ace as u32));
        assert!(Rank::FullHouse(rank) == hand.rank_five());
    }

    #[test]
    fn test_two_pair() {
        // Make a two pair hand.
        let hand = FlatHand::new_from_str("AdAc9D9cTs").unwrap();
        let rank = (((1 << Value::Ace as u32) | (1 << Value::Nine as u32)) << 13)
            | (1 << Value::Ten as u32);
        assert!(Rank::TwoPair(rank) == hand.rank_five());
    }

    #[test]
    fn test_one_pair() {
        let hand = FlatHand::new_from_str("AdAc9d8cTs").unwrap();
        let rank = ((1 << Value::Ace as u32) << 13)
            | (1 << Value::Nine as u32)
            | (1 << Value::Eight as u32)
            | (1 << Value::Ten as u32);

        assert!(Rank::OnePair(rank) == hand.rank_five());
    }

    #[test]
    fn test_four_of_a_kind() {
        let hand = FlatHand::new_from_str("AdAcAsAhTs").unwrap();
        assert!(
            Rank::FourOfAKind((1 << (Value::Ace as u32) << 13) | (1 << (Value::Ten as u32)))
                == hand.rank_five()
        );
    }

    #[test]
    fn test_wheel() {
        let hand = FlatHand::new_from_str("Ad2c3s4h5s").unwrap();
        assert!(Rank::Straight(0) == hand.rank_five());
    }

    #[test]
    fn test_straight() {
        let hand = FlatHand::new_from_str("2c3s4h5s6d").unwrap();
        assert!(Rank::Straight(1) == hand.rank_five());
    }

    #[test]
    fn test_three_of_a_kind() {
        let hand = FlatHand::new_from_str("2c2s2h5s6d").unwrap();
        let rank = ((1 << (Value::Two as u32)) << 13)
            | (1 << (Value::Five as u32))
            | (1 << (Value::Six as u32));
        assert!(Rank::ThreeOfAKind(rank) == hand.rank_five());
    }

    #[test]
    fn test_rank_seven_straight_flush() {
        let h = FlatHand::new_from_str("AdKdQdJdTd9d8d").unwrap();
        assert_eq!(Rank::StraightFlush(9), h.rank());
    }

    #[test]
    fn test_rank_seven_straight_flush_wheel() {
        // Make sure that we pick up the wheel straight flush
        // over different straight.
        let h = FlatHand::new_from_str("2d3d4d5d6h7cAd").unwrap();
        assert_eq!(Rank::StraightFlush(0), h.rank());
    }
    #[test]
    fn test_rank_seven_straights() {
        let straights = [
            "2h3c4s5d6dTsKh",
            "3c4s5d6d7hTsKh",
            "4s5d6d7h8cTsKh",
            "5c6c7h8h9dAhAd",
            "6c7c8h9hTsKc6s",
            "7c8h9hTsKc6sJh",
            "8h9hTsQc6sJhAs",
            "9hTsQc6sJhKsKc",
            "TsQc6sJhKsAc5h",
        ];
        for (idx, s) in straights.iter().enumerate() {
            assert_eq!(
                Rank::Straight(idx as u32 + 1),
                FlatHand::new_from_str(s).unwrap().rank()
            );
        }
    }

    #[test]
    fn test_rank_seven_find_best_with_wheel() {
        let h = FlatHand::new_from_str("6dKdAd2d5d4d3d").unwrap();
        assert_eq!(Rank::StraightFlush(1), h.rank());
    }

    #[test]
    fn test_rank_seven_four_kind() {
        let h = FlatHand::new_from_str("2s2h2d2cKd9h4s").unwrap();
        let four_rank = (1 << Value::Two as u32) << 13;
        let low_rank = 1 << Value::King as u32;
        assert_eq!(Rank::FourOfAKind(four_rank | low_rank), h.rank());
    }

    #[test]
    fn test_rank_seven_four_plus_set() {
        // Four of a kind plus a set.
        let h = FlatHand::new_from_str("2s2h2d2c8d8s8c").unwrap();
        let four_rank = (1 << Value::Two as u32) << 13;
        let low_rank = 1 << Value::Eight as u32;
        assert_eq!(Rank::FourOfAKind(four_rank | low_rank), h.rank());
    }

    #[test]
    fn test_rank_seven_full_house_two_sets() {
        // We have two sets use the highest set.
        let h = FlatHand::new_from_str("As2h2d2c8d8s8c").unwrap();
        let set_rank = (1 << Value::Eight as u32) << 13;
        let low_rank = 1 << Value::Two as u32;
        assert_eq!(Rank::FullHouse(set_rank | low_rank), h.rank());
    }

    #[test]
    fn test_rank_seven_full_house_two_pair() {
        // Test to make sure that we pick the best pair.
        let h = FlatHand::new_from_str("2h2d2c8d8sKdKs").unwrap();
        let set_rank = (1 << Value::Two as u32) << 13;
        let low_rank = 1 << Value::King as u32;
        assert_eq!(Rank::FullHouse(set_rank | low_rank), h.rank());
    }

    #[test]
    fn test_two_pair_from_three_pair() {
        let h = FlatHand::new_from_str("2h2d8d8sKdKsTh").unwrap();
        let pair_rank = ((1 << Value::King as u32) | (1 << Value::Eight as u32)) << 13;
        let low_rank = 1 << Value::Ten as u32;
        assert_eq!(Rank::TwoPair(pair_rank | low_rank), h.rank());
    }

    #[test]
    fn test_rank_seven_two_pair() {
        let h = FlatHand::new_from_str("2h2d8d8sKd6sTh").unwrap();
        let pair_rank = ((1 << Value::Two as u32) | (1 << Value::Eight as u32)) << 13;
        let low_rank = 1 << Value::King as u32;
        assert_eq!(Rank::TwoPair(pair_rank | low_rank), h.rank());
    }

    // CoreRank conversion tests
    #[test]
    fn test_core_rank_from_high_card() {
        let rank = Rank::HighCard(12345);
        assert_eq!(CoreRank::HighCard, rank.into());
    }

    #[test]
    fn test_core_rank_from_one_pair() {
        let rank = Rank::OnePair(54321);
        assert_eq!(CoreRank::OnePair, rank.into());
    }

    #[test]
    fn test_core_rank_from_two_pair() {
        let rank = Rank::TwoPair(99999);
        assert_eq!(CoreRank::TwoPair, rank.into());
    }

    #[test]
    fn test_core_rank_from_three_of_a_kind() {
        let rank = Rank::ThreeOfAKind(11111);
        assert_eq!(CoreRank::ThreeOfAKind, rank.into());
    }

    #[test]
    fn test_core_rank_from_straight() {
        let rank = Rank::Straight(5);
        assert_eq!(CoreRank::Straight, rank.into());
    }

    #[test]
    fn test_core_rank_from_flush() {
        let rank = Rank::Flush(88888);
        assert_eq!(CoreRank::Flush, rank.into());
    }

    #[test]
    fn test_core_rank_from_full_house() {
        let rank = Rank::FullHouse(77777);
        assert_eq!(CoreRank::FullHouse, rank.into());
    }

    #[test]
    fn test_core_rank_from_four_of_a_kind() {
        let rank = Rank::FourOfAKind(66666);
        assert_eq!(CoreRank::FourOfAKind, rank.into());
    }

    #[test]
    fn test_core_rank_from_straight_flush() {
        let rank = Rank::StraightFlush(9);
        assert_eq!(CoreRank::StraightFlush, rank.into());
    }

    #[test]
    fn test_core_rank_ordering() {
        assert!(CoreRank::HighCard < CoreRank::OnePair);
        assert!(CoreRank::OnePair < CoreRank::TwoPair);
        assert!(CoreRank::TwoPair < CoreRank::ThreeOfAKind);
        assert!(CoreRank::ThreeOfAKind < CoreRank::Straight);
        assert!(CoreRank::Straight < CoreRank::Flush);
        assert!(CoreRank::Flush < CoreRank::FullHouse);
        assert!(CoreRank::FullHouse < CoreRank::FourOfAKind);
        assert!(CoreRank::FourOfAKind < CoreRank::StraightFlush);
    }

    #[test]
    fn test_core_rank_different_values_same_type() {
        // Different rank values should map to the same CoreRank
        let flush1: CoreRank = Rank::Flush(100).into();
        let flush2: CoreRank = Rank::Flush(200).into();
        let flush3: CoreRank = Rank::Flush(999999).into();

        assert_eq!(flush1, flush2);
        assert_eq!(flush2, flush3);
        assert_eq!(flush1, CoreRank::Flush);
    }

    /// Verifies that Vec<Card> correctly implements Rankable
    /// and produces the expected rank for a royal flush.
    #[test]
    fn test_rankable_vec_card() {
        let cards: Vec<Card> = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Spade),
            Card::new(Value::Jack, Suit::Spade),
            Card::new(Value::Ten, Suit::Spade),
        ];
        let rank = cards.rank_five();
        assert_eq!(Rank::StraightFlush(9), rank);
    }

    /// Verifies that [Card] slice correctly implements Rankable
    /// and produces the expected rank for a royal flush.
    #[test]
    fn test_rankable_slice_card() {
        let cards: [Card; 5] = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Spade),
            Card::new(Value::Jack, Suit::Spade),
            Card::new(Value::Ten, Suit::Spade),
        ];
        let rank = cards[..].rank_five();
        assert_eq!(Rank::StraightFlush(9), rank);
    }

    /// Verifies that &[Card] correctly implements Rankable
    /// and produces the expected rank for a royal flush.
    #[test]
    fn test_rankable_ref_slice_card() {
        let cards: Vec<Card> = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Spade),
            Card::new(Value::Jack, Suit::Spade),
            Card::new(Value::Ten, Suit::Spade),
        ];
        let slice: &[Card] = &cards;
        let rank = slice.rank_five();
        assert_eq!(Rank::StraightFlush(9), rank);
    }

    /// Verifies wheel (A-2-3-4-5) is correctly identified as the lowest straight,
    /// and hands missing a wheel card are not incorrectly classified as straights.
    #[test]
    fn test_wheel_straight_detection() {
        // Wheel: A-2-3-4-5
        let wheel = FlatHand::new_from_str("Ad2c3s4h5d").unwrap();
        assert_eq!(Rank::Straight(0), wheel.rank_five());

        // Not a wheel (missing 5)
        let not_wheel = FlatHand::new_from_str("Ad2c3s4h6d").unwrap();
        // Should be high card, not a straight
        assert!(matches!(not_wheel.rank_five(), Rank::HighCard(_)));

        // A-2-3-4-6 should NOT be a wheel
        let almost_wheel = FlatHand::new_from_str("Ad2c3s4h6c").unwrap();
        assert!(matches!(almost_wheel.rank_five(), Rank::HighCard(_)));
    }

    /// Verifies that rank values correctly encode hand components:
    /// the primary hand (quads, pair, etc.) and kickers are properly combined.
    #[test]
    fn test_rank_value_computation() {
        // Four of a kind with specific kicker
        let foak = FlatHand::new_from_str("AsAhAdAcKs").unwrap();
        let foak_rank = foak.rank_five();
        // The rank should be (ace_bits << 13) | king_bits
        let expected_foak =
            Rank::FourOfAKind(((1 << Value::Ace as u32) << 13) | (1 << Value::King as u32));
        assert_eq!(expected_foak, foak_rank);

        // One pair with specific kickers
        let pair = FlatHand::new_from_str("AsAhKdQcJs").unwrap();
        let pair_rank = pair.rank_five();
        let expected_pair = Rank::OnePair(
            ((1 << Value::Ace as u32) << 13)
                | (1 << Value::King as u32)
                | (1 << Value::Queen as u32)
                | (1 << Value::Jack as u32),
        );
        assert_eq!(expected_pair, pair_rank);
    }

    /// Verifies that hands with the same rank type but different card values
    /// are correctly ordered (e.g., pair of aces beats pair of kings).
    #[test]
    fn test_rank_ordering_within_same_type() {
        // Two different pairs
        let pair_aces = FlatHand::new_from_str("AsAhKdQcJs").unwrap();
        let pair_kings = FlatHand::new_from_str("KsKhAdQcJs").unwrap();
        assert!(pair_aces.rank_five() > pair_kings.rank_five());

        // Two different two-pairs
        let two_pair_ak = FlatHand::new_from_str("AsAhKdKcJs").unwrap();
        let two_pair_aq = FlatHand::new_from_str("AsAhQdQcKs").unwrap();
        assert!(two_pair_ak.rank_five() > two_pair_aq.rank_five());

        // Two different three of a kinds
        let trips_aces = FlatHand::new_from_str("AsAhAdKcJs").unwrap();
        let trips_kings = FlatHand::new_from_str("KsKhKdAcJs").unwrap();
        assert!(trips_aces.rank_five() > trips_kings.rank_five());
    }

    /// Test 7-card hands with Vec<Card>.
    #[test]
    fn test_seven_card_vec() {
        let cards: Vec<Card> = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Spade),
            Card::new(Value::Jack, Suit::Spade),
            Card::new(Value::Ten, Suit::Spade),
            Card::new(Value::Nine, Suit::Spade),
            Card::new(Value::Eight, Suit::Spade),
        ];
        let rank = cards.rank();
        assert_eq!(Rank::StraightFlush(9), rank);
    }

    /// Verifies four of a kind correctly encodes the quads and kicker.
    ///
    /// The rank bits should encode both the quad value (aces) in the high bits
    /// and the kicker value (king) in the low bits.
    #[test]
    fn test_four_of_a_kind_encoding() {
        let cards: Vec<Card> = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::Ace, Suit::Diamond),
            Card::new(Value::Ace, Suit::Club),
            Card::new(Value::King, Suit::Spade),
        ];

        let rank = cards.rank_five();

        let Rank::FourOfAKind(bits) = rank else {
            panic!("Expected FourOfAKind, got {:?}", rank);
        };

        let quads_value = bits >> 13;
        let kicker = bits & 0x1FFF;
        assert!(quads_value & (1 << 12) != 0, "Should have ace as quads");
        assert!(kicker & (1 << 11) != 0, "Should have king as kicker");
    }

    /// Verifies full house correctly encodes the trips and pair.
    ///
    /// The rank bits should encode the trips value (aces) in the high bits
    /// and the pair value (kings) in the low bits.
    #[test]
    fn test_full_house_encoding() {
        let cards: Vec<Card> = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::Ace, Suit::Diamond),
            Card::new(Value::King, Suit::Club),
            Card::new(Value::King, Suit::Spade),
        ];

        let rank = cards.rank_five();

        let Rank::FullHouse(bits) = rank else {
            panic!("Expected FullHouse, got {:?}", rank);
        };

        let set_value = bits >> 13;
        let pair_value = bits & 0x1FFF;
        assert!(set_value & (1 << 12) != 0, "Should have ace as trips");
        assert!(pair_value & (1 << 11) != 0, "Should have king as pair");
    }

    /// Verifies three of a kind correctly encodes the trips value.
    ///
    /// The rank bits should encode the trips value (aces) in the high bits.
    #[test]
    fn test_three_of_a_kind_encoding() {
        let cards: Vec<Card> = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::Ace, Suit::Diamond),
            Card::new(Value::King, Suit::Club),
            Card::new(Value::Queen, Suit::Spade),
        ];

        let rank = cards.rank_five();

        let Rank::ThreeOfAKind(bits) = rank else {
            panic!("Expected ThreeOfAKind, got {:?}", rank);
        };

        let set_value = bits >> 13;
        assert!(set_value & (1 << 12) != 0, "Should have ace as trips");
    }

    /// Verifies two pair correctly encodes both pairs and the kicker.
    ///
    /// The rank bits should encode both pair values (aces and kings) in the
    /// high bits and the kicker value (queen) in the low bits.
    #[test]
    fn test_two_pair_encoding() {
        let cards: Vec<Card> = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::King, Suit::Diamond),
            Card::new(Value::King, Suit::Club),
            Card::new(Value::Queen, Suit::Spade),
        ];

        let rank = cards.rank_five();

        let Rank::TwoPair(bits) = rank else {
            panic!("Expected TwoPair, got {:?}", rank);
        };

        let pairs_value = bits >> 13;
        let kicker = bits & 0x1FFF;
        assert!(pairs_value & (1 << 12) != 0, "Should have ace pair");
        assert!(pairs_value & (1 << 11) != 0, "Should have king pair");
        assert!(kicker & (1 << 10) != 0, "Should have queen as kicker");
    }

    /// Verifies one pair correctly encodes the pair and kickers.
    ///
    /// The rank bits should encode the pair value (aces) in the high bits
    /// and three kicker values in the low bits.
    #[test]
    fn test_one_pair_encoding() {
        let cards: Vec<Card> = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::King, Suit::Diamond),
            Card::new(Value::Queen, Suit::Club),
            Card::new(Value::Jack, Suit::Spade),
        ];

        let rank = cards.rank_five();

        let Rank::OnePair(bits) = rank else {
            panic!("Expected OnePair, got {:?}", rank);
        };

        let pair_value = bits >> 13;
        let kickers = bits & 0x1FFF;
        assert!(pair_value & (1 << 12) != 0, "Should have ace pair");
        assert_eq!(kickers.count_ones(), 3, "Should have 3 kickers");
    }

    /// Verifies that the cards() iterator returns all cards in the slice.
    #[test]
    fn test_slice_cards_iterator() {
        let cards: &[Card] = &[
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
        ];

        let card_vec: Vec<Card> = cards.cards().collect();

        assert_eq!(card_vec.len(), 2);
        assert!(card_vec.contains(&Card::new(Value::Ace, Suit::Spade)));
        assert!(card_vec.contains(&Card::new(Value::King, Suit::Heart)));
    }

    /// Verifies that five cards of the same suit are correctly identified as a flush.
    #[test]
    fn test_flush_detection() {
        let cards: Vec<Card> = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Spade),
            Card::new(Value::Jack, Suit::Spade),
            Card::new(Value::Nine, Suit::Spade),
        ];

        let rank = cards.rank_five();

        assert!(
            matches!(rank, Rank::Flush(_)),
            "Expected Flush, got {:?}",
            rank
        );
    }

    /// Verifies that four cards of the same value are correctly identified as four of a kind.
    #[test]
    fn test_four_of_a_kind_detection() {
        let cards: Vec<Card> = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::Ace, Suit::Diamond),
            Card::new(Value::Ace, Suit::Club),
            Card::new(Value::King, Suit::Spade),
        ];

        let rank = cards.rank_five();

        assert!(
            matches!(rank, Rank::FourOfAKind(_)),
            "Expected FourOfAKind, got {:?}",
            rank
        );
    }

    /// Verifies that two distinct pairs are correctly identified as two pair.
    #[test]
    fn test_two_pair_detection() {
        let cards: Vec<Card> = vec![
            Card::new(Value::Two, Suit::Spade),
            Card::new(Value::Two, Suit::Heart),
            Card::new(Value::Three, Suit::Diamond),
            Card::new(Value::Three, Suit::Club),
            Card::new(Value::Four, Suit::Spade),
        ];

        let rank = cards.rank_five();

        assert!(
            matches!(rank, Rank::TwoPair(_)),
            "Expected TwoPair, got {:?}",
            rank
        );
    }
}
