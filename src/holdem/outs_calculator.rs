use std::collections::HashMap;

use crate::core::{Card, CardBitSet, CardIter, Hand, PlayerBitSet, Rank, Rankable};

/// Result of calculating outs for a specific player
#[derive(Debug, Clone)]
pub struct PlayerOuts {
    /// Player index
    pub player_idx: usize,
    /// Map of Rank to the set of board card combinations that produce that rank
    /// The CardBitSet represents the board cards (not including player's hole
    /// cards)
    pub winning_boards: HashMap<Rank, Vec<CardBitSet>>,
    /// Total number of combinations where this player wins outright
    pub wins: usize,
    /// Total number of combinations where this player ties
    pub ties: usize,
    /// Total number of combinations evaluated
    pub total_combinations: usize,
}

impl PlayerOuts {
    /// Calculate win percentage (0.0 - 100.0)
    pub fn win_percentage(&self) -> f32 {
        if self.total_combinations == 0 {
            return 0.0;
        }
        (self.wins as f32 / self.total_combinations as f32) * 100.0
    }

    /// Calculate tie percentage (0.0 - 100.0)
    pub fn tie_percentage(&self) -> f32 {
        if self.total_combinations == 0 {
            return 0.0;
        }
        (self.ties as f32 / self.total_combinations as f32) * 100.0
    }

    /// Get the number of outs for a specific rank or better
    pub fn outs_for_rank(&self, target_rank: Rank) -> usize {
        self.winning_boards
            .iter()
            .filter(|(rank, _)| **rank >= target_rank)
            .map(|(_, boards)| boards.len())
            .sum()
    }
}

/// Calculator for determining player outs in Texas Hold'em
///
/// Given a board state and player hands, this calculator will enumerate
/// all possible remaining card combinations and determine which hands
/// win with which ranks.
#[derive(Debug)]
pub struct OutsCalculator {
    /// The current board (3, 4, or 5 cards typically)
    board: CardBitSet,
    /// Player hands (hole cards)
    player_hands: Vec<Hand>,
    /// Cards that can still be dealt
    remaining_cards: CardBitSet,
}

impl OutsCalculator {
    /// Create a new OutsCalculator
    ///
    /// # Arguments
    /// * `board` - The current board cards as a CardBitSet
    /// * `player_hands` - Vector of player hole cards
    ///
    /// # Returns
    /// A new OutsCalculator instance
    ///
    /// # Example
    /// ```
    /// use rs_poker::core::{Card, CardBitSet, Hand, Suit, Value};
    /// use rs_poker::holdem::OutsCalculator;
    ///
    /// let mut board = CardBitSet::new();
    /// board.insert(Card::new(Value::Ace, Suit::Spade));
    /// board.insert(Card::new(Value::King, Suit::Spade));
    /// board.insert(Card::new(Value::Queen, Suit::Spade));
    ///
    /// let player1 = Hand::new_from_str("JsTs").unwrap();
    /// let player2 = Hand::new_from_str("AhKd").unwrap();
    ///
    /// let calc = OutsCalculator::new(board, vec![player1, player2]);
    /// ```
    pub fn new(board: CardBitSet, player_hands: Vec<Hand>) -> Self {
        // Calculate which cards are still available
        let mut used_cards = board;
        for hand in &player_hands {
            for card in hand.iter() {
                used_cards.insert(card);
            }
        }

        let all_cards = CardBitSet::default();
        let remaining_cards = all_cards ^ used_cards;

        Self {
            board,
            player_hands,
            remaining_cards,
        }
    }

    /// Calculate outs by enumerating all possible board completions
    ///
    /// The board in Texas Hold'em always has 5 cards total (flop + turn +
    /// river). This method automatically determines how many cards need to
    /// be dealt based on the current board size.
    ///
    /// # Returns
    /// Vector of PlayerOuts, one for each player
    ///
    /// # Example
    /// ```
    /// use rs_poker::core::{Card, CardBitSet, Hand, Suit, Value};
    /// use rs_poker::holdem::OutsCalculator;
    ///
    /// let mut board = CardBitSet::new();
    /// board.insert(Card::new(Value::Ace, Suit::Spade));
    /// board.insert(Card::new(Value::King, Suit::Spade));
    /// board.insert(Card::new(Value::Queen, Suit::Spade));
    ///
    /// let player1 = Hand::new_from_str("JsTs").unwrap(); // Royal flush draw
    /// let player2 = Hand::new_from_str("AhKd").unwrap(); // Top two pair
    ///
    /// let calc = OutsCalculator::new(board, vec![player1, player2]);
    /// let results = calc.calculate_outs(); // Automatically deals turn + river
    ///
    /// // Player 1 should have a high win percentage with the royal flush draw
    /// assert!(results[0].win_percentage() > 0.0);
    /// ```
    pub fn calculate_outs(&self) -> Vec<PlayerOuts> {
        let num_players = self.player_hands.len();
        let board_size = self.board.count();
        let num_cards_to_deal = 5 - board_size;

        // Initialize results for each player
        let mut results: Vec<PlayerOuts> = (0..num_players)
            .map(|idx| PlayerOuts {
                player_idx: idx,
                winning_boards: HashMap::new(),
                wins: 0,
                ties: 0,
                total_combinations: 0,
            })
            .collect();

        let remaining_vec: Vec<Card> = self.remaining_cards.into_iter().collect();

        // Iterate through all possible combinations of remaining cards
        for combo in CardIter::new(&remaining_vec, num_cards_to_deal) {
            // Build the complete board using bitwise OR
            let mut full_board = self.board;
            for card in &combo {
                full_board.insert(*card);
            }

            // Evaluate each player's hand with the full board using bitwise OR
            // and fold to find best rank and winners
            let (best_rank, winners) = self
                .player_hands
                .iter()
                .enumerate()
                .map(|(idx, player_hand)| {
                    // Combine hole cards + board using bitwise OR
                    let player_bitset: CardBitSet = (*player_hand).into();
                    let combined_bitset = player_bitset | full_board;
                    // Rank directly on CardBitSet without allocation
                    (idx, combined_bitset.rank())
                })
                .fold(
                    (Rank::HighCard(0), PlayerBitSet::default()),
                    |(max_rank, mut winners), (idx, rank)| {
                        use std::cmp::Ordering;
                        match rank.cmp(&max_rank) {
                            Ordering::Greater => {
                                // New best rank, reset winners to just this player
                                let mut new_winners = PlayerBitSet::default();
                                new_winners.enable(idx);
                                (rank, new_winners)
                            }
                            Ordering::Equal => {
                                // Tie with current best, add to winners
                                winners.enable(idx);
                                (max_rank, winners)
                            }
                            Ordering::Less => {
                                // Not as good, keep current best
                                (max_rank, winners)
                            }
                        }
                    },
                );

            let is_tie = winners.count() > 1;

            // Update results for each player
            for (idx, result) in results.iter_mut().enumerate() {
                result.total_combinations += 1;

                if winners.get(idx) {
                    if is_tie {
                        result.ties += 1;
                    } else {
                        result.wins += 1;
                        // Record the winning board for this rank
                        result
                            .winning_boards
                            .entry(best_rank)
                            .or_default()
                            .push(full_board);
                    }
                }
            }
        }

        results
    }

    /// Get the current board
    pub fn board(&self) -> CardBitSet {
        self.board
    }

    /// Get the player hands
    pub fn player_hands(&self) -> &[Hand] {
        &self.player_hands
    }

    /// Get the remaining cards that can be dealt
    pub fn remaining_cards(&self) -> CardBitSet {
        self.remaining_cards
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Suit, Value};

    #[test]
    fn test_outs_calculator_basic() {
        let mut board = CardBitSet::new();
        board.insert(Card::new(Value::Ace, Suit::Spade));
        board.insert(Card::new(Value::King, Suit::Spade));
        board.insert(Card::new(Value::Queen, Suit::Spade));

        let player1 = Hand::new_from_str("JsTs").unwrap(); // Straight flush draw
        let player2 = Hand::new_from_str("AhKd").unwrap(); // Top two pair

        let calc = OutsCalculator::new(board, vec![player1, player2]);
        let results = calc.calculate_outs();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].player_idx, 0);
        assert_eq!(results[1].player_idx, 1);

        // Player 1 has a strong draw and should have high win percentage
        assert!(results[0].win_percentage() > 40.0);
    }

    #[test]
    fn test_outs_calculator_one_card() {
        let mut board = CardBitSet::new();
        board.insert(Card::new(Value::Ace, Suit::Spade));
        board.insert(Card::new(Value::King, Suit::Spade));
        board.insert(Card::new(Value::Queen, Suit::Spade));
        board.insert(Card::new(Value::Jack, Suit::Spade));

        let player1 = Hand::new_from_str("Ts9s").unwrap(); // Made straight flush
        let player2 = Hand::new_from_str("AhKd").unwrap(); // Top two pair

        let calc = OutsCalculator::new(board, vec![player1, player2]);
        let results = calc.calculate_outs();

        // Player 1 should have very high win percentage with made straight flush
        assert!(results[0].win_percentage() > 95.0);
    }

    #[test]
    fn test_outs_calculator_tie() {
        let mut board = CardBitSet::new();
        board.insert(Card::new(Value::Ace, Suit::Spade));
        board.insert(Card::new(Value::King, Suit::Spade));
        board.insert(Card::new(Value::Queen, Suit::Spade));

        // Both players have the same hand
        let player1 = Hand::new_from_str("JhTh").unwrap();
        let player2 = Hand::new_from_str("JdTd").unwrap();

        let calc = OutsCalculator::new(board, vec![player1, player2]);
        let results = calc.calculate_outs();

        // Both players should tie on all boards
        assert_eq!(results[0].wins, 0);
        assert_eq!(results[1].wins, 0);
        assert!(results[0].tie_percentage() > 95.0);
        assert!(results[1].tie_percentage() > 95.0);
    }

    #[test]
    fn test_flush_draw_vs_pocket_pair() {
        // Classic scenario: flush draw from behind has a chance to win
        // Flop: Kh 7h 2d
        let mut board = CardBitSet::new();
        board.insert(Card::new(Value::King, Suit::Heart));
        board.insert(Card::new(Value::Seven, Suit::Heart));
        board.insert(Card::new(Value::Two, Suit::Diamond));

        // Player 1: Ah 9h (flush draw, currently behind)
        let player1 = Hand::new_from_str("Ah9h").unwrap();

        // Player 2: Ks Kc (set of kings, currently ahead)
        let player2 = Hand::new_from_str("KsKc").unwrap();

        let calc = OutsCalculator::new(board, vec![player1, player2]);
        let results = calc.calculate_outs();

        // Player 1 is currently behind but has flush outs
        // Should have roughly 24-26% win rate (9 flush outs)
        assert!(results[0].win_percentage() > 20.0);
        assert!(results[0].win_percentage() < 30.0);

        // Player 2 should be ahead with the set
        assert!(results[1].win_percentage() > 70.0);
        assert!(results[1].win_percentage() < 80.0);

        // Verify player 1 has some wins (not just ties)
        assert!(results[0].wins > 0);

        // Check that player 1 can win with a flush
        let has_flush_win = results[0]
            .winning_boards
            .keys()
            .any(|rank| matches!(rank, Rank::Flush(_)));
        assert!(has_flush_win, "Player 1 should be able to win with a flush");
    }

    #[test]
    fn test_player_outs_percentages() {
        let outs = PlayerOuts {
            player_idx: 0,
            winning_boards: HashMap::new(),
            wins: 700,
            ties: 100,
            total_combinations: 1000,
        };

        // 700 / 1000 = 70%
        assert_eq!(outs.win_percentage(), 70.0);
        // 100 / 1000 = 10%
        assert_eq!(outs.tie_percentage(), 10.0);
    }
}
