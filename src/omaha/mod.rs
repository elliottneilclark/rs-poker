//! Omaha poker hand evaluation.
//!
//! In Omaha Hold'em, each player receives hole cards (4 in PLO4, 5 in PLO5, 6
//! in PLO6, 7 in PLO7) and shares 5 community board cards. A valid 5-card hand must use
//! **exactly 2** hole cards and **exactly 3** board cards.
//!
//! [`OmahaHand`] stores the hole cards and board cards as separate
//! [`CardBitSet`]s and ranks the best possible 5-card combination.
//!
//! # Examples
//!
//! ```
//! use rs_poker::core::{Rank, Rankable};
//! use rs_poker::omaha::OmahaHand;
//!
//! // Hole: AhAsKhKs, Board: QhJhTh9h8h
//! // Best hand: AhKh (hole) + QhJhTh (board) = royal flush
//! let hand = OmahaHand::new_from_str("AhAsKhKs", "QhJhTh9h8h").unwrap();
//! let rank = hand.rank();
//! assert_eq!(rank, Rank::StraightFlush(9));
//! ```

use crate::core::{CardBitSet, CardIter, Hand, RSPokerError, Rank, RankFive, Rankable};

/// An Omaha poker hand consisting of private hole cards and shared board cards.
///
/// The hand is ranked by finding the best 5-card combination using exactly 2
/// hole cards and exactly 3 board cards.
///
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OmahaHand {
    /// The player's private hole cards (typically 4, but 5 or 6 for variants).
    hole: CardBitSet,
    /// The shared community board cards (3 on flop, 4 on turn, 5 on river).
    board: CardBitSet,
}

impl OmahaHand {
    /// Create a new `OmahaHand` from two `CardBitSet`s.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `hole` has fewer than 2 or more than 7 cards
    /// - `board` has fewer than 3 or more than 5 cards
    /// - `hole` and `board` share any cards
    pub fn new(hole: CardBitSet, board: CardBitSet) -> Result<Self, RSPokerError> {
        let hole_count = hole.count();
        if !(2..=7).contains(&hole_count) {
            return Err(RSPokerError::OmahaHoleCardCount(hole_count));
        }
        let board_count = board.count();
        if !(3..=5).contains(&board_count) {
            return Err(RSPokerError::OmahaBoardCardCount(board_count));
        }
        if !(hole & board).is_empty() {
            return Err(RSPokerError::OmahaOverlappingCards);
        }
        Ok(Self { hole, board })
    }

    /// Parse an `OmahaHand` from two card strings.
    ///
    /// Each string uses the same format as [`Hand::new_from_str`]: pairs of
    /// value+suit characters like `"AhKsQdJc"`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_poker::omaha::OmahaHand;
    ///
    /// let hand = OmahaHand::new_from_str("AhAsKhKs", "QhJhTh9h8h").unwrap();
    /// ```
    pub fn new_from_str(hole_str: &str, board_str: &str) -> Result<Self, RSPokerError> {
        let hole: CardBitSet = Hand::new_from_str(hole_str)?.into();
        let board: CardBitSet = Hand::new_from_str(board_str)?.into();
        Self::new(hole, board)
    }

    /// The player's private hole cards.
    pub fn hole(&self) -> CardBitSet {
        self.hole
    }

    /// The shared community board cards.
    pub fn board(&self) -> CardBitSet {
        self.board
    }
}

impl Rankable for OmahaHand {
    fn rank(&self) -> Rank {
        CardIter::new(self.hole, 2)
            .flat_map(|hole| {
                CardIter::new(self.board, 3).map(move |board| (hole | board).rank_five())
            })
            // OmahaHand::new() guarantees hole >= 2 and board >= 3,
            // so there is always at least C(2,2) * C(3,3) = 1 combination.
            .max()
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Card, CoreRank, Hand, Rank, RankFive, Rankable, Suit, Value};

    // ── Construction ─────────────────────────────────────────────

    #[test]
    fn test_new_valid() {
        let hand = OmahaHand::new_from_str("AhAsKhKs", "QhJhTh9h8h");
        assert!(hand.is_ok());
    }

    #[test]
    fn test_new_too_few_hole_cards() {
        // Only 1 hole card
        let mut hole = CardBitSet::new();
        hole.insert(Card::new(Value::Ace, Suit::Spade));
        let mut board = CardBitSet::new();
        for c in [
            Card::new(Value::Two, Suit::Heart),
            Card::new(Value::Three, Suit::Heart),
            Card::new(Value::Four, Suit::Heart),
        ] {
            board.insert(c);
        }
        assert!(matches!(
            OmahaHand::new(hole, board),
            Err(RSPokerError::OmahaHoleCardCount(1))
        ));
    }

    #[test]
    fn test_new_too_few_board_cards() {
        let mut hole = CardBitSet::new();
        hole.insert(Card::new(Value::Ace, Suit::Spade));
        hole.insert(Card::new(Value::King, Suit::Spade));
        let mut board = CardBitSet::new();
        board.insert(Card::new(Value::Two, Suit::Heart));
        board.insert(Card::new(Value::Three, Suit::Heart));
        assert!(matches!(
            OmahaHand::new(hole, board),
            Err(RSPokerError::OmahaBoardCardCount(2))
        ));
    }

    #[test]
    fn test_new_overlapping_cards() {
        let shared = Card::new(Value::Ace, Suit::Spade);
        let mut hole = CardBitSet::new();
        hole.insert(shared);
        hole.insert(Card::new(Value::King, Suit::Spade));
        let mut board = CardBitSet::new();
        board.insert(shared);
        board.insert(Card::new(Value::Two, Suit::Heart));
        board.insert(Card::new(Value::Three, Suit::Heart));
        assert!(matches!(
            OmahaHand::new(hole, board),
            Err(RSPokerError::OmahaOverlappingCards)
        ));
    }

    #[test]
    fn test_new_from_str_duplicate_in_hole() {
        let result = OmahaHand::new_from_str("AhAhKsQs", "2h3h4h5h6h");
        assert!(matches!(result, Err(RSPokerError::DuplicateCardInHand(_))));
    }

    // ── Ranking (issue #33 examples) ─────────────────────────────

    #[test]
    fn test_issue_33_example() {
        // From the issue: AhAsKhKs hole, QhJhTh9h8h board.
        // Best 5-card hand using exactly 2 hole + 3 board:
        //   AhKh (hole) + QhJhTh (board) = royal flush
        let hand = OmahaHand::new_from_str("AhAsKhKs", "QhJhTh9h8h").unwrap();
        let rank = hand.rank();
        let best_hand: CardBitSet = Hand::new_from_str("AhKhQhJhTh").unwrap().into();
        assert_eq!(rank, best_hand.rank_five());
    }

    #[test]
    fn test_straight_flush() {
        // Hole: 9s8s7c6c, Board: Ts7s6s5d4d
        // Best: 9s8s + Ts7s6s = T-high straight flush
        let hand = OmahaHand::new_from_str("9s8s7c6c", "Ts7s6s5d4d").unwrap();
        let rank = hand.rank();
        assert_eq!(CoreRank::StraightFlush, CoreRank::from(rank));
    }

    #[test]
    fn test_four_of_a_kind() {
        // Hole: AsAh9c2c, Board: AcAdKs3h4h
        // Best: AsAh (2 hole aces) + AcAdKs (2 board aces + K) = four aces + K
        let hand = OmahaHand::new_from_str("AsAh9c2c", "AcAdKs3h4h").unwrap();
        let rank = hand.rank();
        assert_eq!(CoreRank::FourOfAKind, CoreRank::from(rank));
    }

    #[test]
    fn test_full_house() {
        // Hole: AsAhKsQd, Board: AdKdKc3h4h
        // Best: AsAh + AdKdKc = AAA KK = full house (aces full of kings)
        let hand = OmahaHand::new_from_str("AsAhKsQd", "AdKdKc3h4h").unwrap();
        let rank = hand.rank();
        assert_eq!(CoreRank::FullHouse, CoreRank::from(rank));
    }

    #[test]
    fn test_flush() {
        // Hole: AhKh2s3s, Board: Qh9h5h4d6d
        // Best: AhKh + Qh9h5h = Ah Kh Qh 9h 5h = ace-high flush
        let hand = OmahaHand::new_from_str("AhKh2s3s", "Qh9h5h4d6d").unwrap();
        let rank = hand.rank();
        assert_eq!(CoreRank::Flush, CoreRank::from(rank));
    }

    #[test]
    fn test_straight() {
        // Hole: Ah Ks 2c 3d, Board: Qd Jh Tc 9s 4h
        // Best: AhKs + QdJhTc = AKQJT = broadway straight
        let hand = OmahaHand::new_from_str("AhKs2c3d", "QdJhTc9s4h").unwrap();
        let rank = hand.rank();
        assert_eq!(CoreRank::Straight, CoreRank::from(rank));
        // Broadway is straight index 9
        assert_eq!(Rank::Straight(9), rank);
    }

    #[test]
    fn test_must_use_exactly_two_hole_cards() {
        // This is the key Omaha rule test.
        // Hole: AhAdAcAs (four aces), Board: Kh Kd 2c 3c 4c
        // In holdem, this would be quads. In Omaha, you must use exactly 2
        // from hole (2 aces) and 3 from board (Kh Kd + one other).
        // Best: AhAd + KhKd2c = AA KK 2 = two pair
        // Or: AhAd + KhKd3c = AA KK 3 = two pair
        // So the best hand is two pair, NOT four of a kind.
        let hand = OmahaHand::new_from_str("AhAdAcAs", "KhKd2c3c4c").unwrap();
        let rank = hand.rank();
        assert_eq!(CoreRank::TwoPair, CoreRank::from(rank));
    }

    #[test]
    fn test_must_use_exactly_three_board_cards() {
        // Hole: AhKh2s3s, Board: Qh Jh Th 9h 8d
        // Board has 4 hearts (Qh Jh Th 9h). With Ah Kh from hole, you'd
        // have 6 hearts total. But you must use exactly 3 board cards.
        // Best with 2 hole hearts: AhKh + Qh Jh Th = royal flush
        // This IS a valid straight flush because we use 2 from hole, 3 from board.
        let hand = OmahaHand::new_from_str("AhKh2s3s", "QhJhTh9h8d").unwrap();
        let rank = hand.rank();
        assert_eq!(Rank::StraightFlush(9), rank);
    }

    #[test]
    fn test_board_flush_not_player_flush() {
        // Board has 5 hearts, but hole has no hearts.
        // Hole: 2s 3s 4d 5d, Board: Ah Kh Qh Jh 9h
        // Must use 2 from hole + 3 from board.
        // No combination gives a flush because hole has no hearts.
        // Best might be a straight: 2s3s (no), 4d5d (no)...
        // None of the 2-hole combos with 3 board cards make a flush.
        let hand = OmahaHand::new_from_str("2s3s4d5d", "AhKhQhJh9h").unwrap();
        let rank = hand.rank();
        assert_ne!(CoreRank::Flush, CoreRank::from(rank));
        assert_ne!(CoreRank::StraightFlush, CoreRank::from(rank));
    }

    #[test]
    fn test_plo5_five_hole_cards() {
        // PLO5: 5 hole cards
        let hand = OmahaHand::new_from_str("AhAsKhKs9d", "QhJhTh2c3c").unwrap();
        assert_eq!(hand.hole().count(), 5);
        let rank = hand.rank();
        // AhKh + QhJhTh = royal flush
        assert_eq!(Rank::StraightFlush(9), rank);
    }

    #[test]
    fn test_plo6_six_hole_cards() {
        // PLO6: 6 hole cards
        let hand = OmahaHand::new_from_str("AhAsKhKs9d8d", "QhJhTh2c3c").unwrap();
        assert_eq!(hand.hole().count(), 6);
        let rank = hand.rank();
        assert_eq!(Rank::StraightFlush(9), rank);
    }

    #[test]
    fn test_flop_three_board_cards() {
        // Partial board (flop): only 3 board cards
        let hand = OmahaHand::new_from_str("AhAsKhKs", "QhJhTh").unwrap();
        let rank = hand.rank();
        // AhKh + QhJhTh = royal flush
        assert_eq!(Rank::StraightFlush(9), rank);
    }

    #[test]
    fn test_turn_four_board_cards() {
        // Partial board (turn): 4 board cards
        let hand = OmahaHand::new_from_str("AhAsKhKs", "QhJhTh2c").unwrap();
        let rank = hand.rank();
        assert_eq!(Rank::StraightFlush(9), rank);
    }

    #[test]
    fn test_accessors() {
        let hand = OmahaHand::new_from_str("AhAsKhKs", "QhJhTh9h8h").unwrap();
        assert_eq!(hand.hole().count(), 4);
        assert_eq!(hand.board().count(), 5);
    }

    #[test]
    fn test_wheel_straight() {
        // Hole: Ah 2s 9c 8c, Board: 3d 4d 5h Kc Qc
        // Best: Ah2s + 3d4d5h = A2345 = wheel
        let hand = OmahaHand::new_from_str("Ah2s9c8c", "3d4d5hKcQc").unwrap();
        let rank = hand.rank();
        assert_eq!(Rank::Straight(0), rank);
    }

    #[test]
    fn test_high_card_only() {
        // No pair, no straight, no flush possible with Omaha constraints
        // Hole: 2s 7d 4c 9h, Board: As Kd Qh 3c Jh
        // Must use exactly 2 hole + 3 board. Best high card combos:
        // 9h7d + AsKdQh = A K Q 9 7
        let hand = OmahaHand::new_from_str("2s7d4c9h", "AsKdQh3cJh").unwrap();
        let rank = hand.rank();
        assert_eq!(CoreRank::HighCard, CoreRank::from(rank));
    }

    #[test]
    fn test_one_pair() {
        // Hole: AhKs2c3d, Board: Ad 9h 8c 7s 4d
        // Best: AhKs + Ad9h8c = pair of aces, K 9 8 kickers
        let hand = OmahaHand::new_from_str("AhKs2c3d", "Ad9h8c7s4d").unwrap();
        let rank = hand.rank();
        assert_eq!(CoreRank::OnePair, CoreRank::from(rank));
    }

    #[test]
    fn test_rankable_trait_impl() {
        // Verify OmahaHand can be used through the Rankable trait
        let hand = OmahaHand::new_from_str("AhAsKhKs", "QhJhTh9h8h").unwrap();
        let rank = hand.rank();
        assert_eq!(Rank::StraightFlush(9), rank);

        // Verify it works through a generic function
        fn rank_any(hand: &impl Rankable) -> Rank {
            hand.rank()
        }
        assert_eq!(Rank::StraightFlush(9), rank_any(&hand));
    }

    #[test]
    fn test_new_too_many_hole_cards() {
        // 8 hole cards should fail (max is 7 for PLO7)
        let hand = OmahaHand::new_from_str("AhAsKhKs9d8d7c6c", "QhJhTh2s3s");
        assert!(matches!(hand, Err(RSPokerError::OmahaHoleCardCount(8))));
    }

    #[test]
    fn test_new_too_many_board_cards() {
        // 6 board cards should fail (max is 5)
        let hand = OmahaHand::new_from_str("AhAsKhKs", "QhJhTh9h8h2c");
        assert!(matches!(hand, Err(RSPokerError::OmahaBoardCardCount(6))));
    }

    #[test]
    fn test_plo7_seven_hole_cards() {
        // PLO7: 7 hole cards should be valid
        let hand = OmahaHand::new_from_str("AhAsKhKs9d8d7c", "QdJdTd2s3s");
        assert!(hand.is_ok());
        assert_eq!(hand.unwrap().hole().count(), 7);
    }

    #[test]
    fn test_rank_comparison_between_hands() {
        // Hand A: has a flush
        let hand_a = OmahaHand::new_from_str("AhKh2s3s", "Qh9h5h4d6d").unwrap();
        // Hand B: has a straight
        let hand_b = OmahaHand::new_from_str("AhKs2c3d", "QdJhTc9s4h").unwrap();
        // Flush > Straight
        assert!(hand_a.rank() > hand_b.rank());
    }
}
