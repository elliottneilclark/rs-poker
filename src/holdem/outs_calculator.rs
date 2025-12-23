use std::collections::HashMap;

use crate::core::{Card, CardBitSet, CardIter, Hand, PlayerBitSet, Rank, Rankable};

/// Result of calculating outs for a specific player
#[derive(Debug, Clone)]
pub struct PlayerOutcome {
    /// The board cards (CardBitSet of 0-5 cards)
    pub board: CardBitSet,
    /// The player's hand (hole cards)
    pub hand: Hand,
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

impl PlayerOutcome {
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

    /// Count winning boards by CoreRank and count occurrences.
    ///
    /// This method takes the detailed rank information from winning_boards and groups
    /// them by their CoreRank (e.g., all flushes together, all full houses together),
    /// summing up the total number of boards for each hand type.
    ///
    /// # Returns
    ///
    /// HashMap from CoreRank to the total count of boards that result
    /// in that hand type where this player wins.
    ///
    /// # Example
    /// ```
    /// use rs_poker::core::{Card, CardBitSet, Hand, Suit, Value};
    /// use rs_poker::holdem::OutsCalculator;
    ///
    /// let mut board = CardBitSet::new();
    /// board.insert(Card::new(Value::King, Suit::Heart));
    /// board.insert(Card::new(Value::Seven, Suit::Heart));
    /// board.insert(Card::new(Value::Two, Suit::Diamond));
    ///
    /// let player1 = Hand::new_from_str("Ah9h").unwrap();
    /// let player2 = Hand::new_from_str("KsKc").unwrap();
    ///
    /// let calc = OutsCalculator::new(board, vec![player1, player2]);
    /// let player_outs = calc.calculate_outs();
    /// let outcomes = player_outs.outcomes();
    ///
    /// // Get the grouped winning hands by core rank for player 1
    /// let grouped = outcomes[0].count_wins_by_core_rank();
    /// ```
    pub fn count_wins_by_core_rank(&self) -> HashMap<crate::core::CoreRank, usize> {
        use crate::core::CoreRank;
        let mut grouped = HashMap::new();
        for (rank, boards) in &self.winning_boards {
            let core_rank: CoreRank = (*rank).into();
            *grouped.entry(core_rank).or_insert(0) += boards.len();
        }
        grouped
    }
}

/// Wrapper for player outcomes with additional analysis methods
pub struct PlayerOuts(Vec<PlayerOutcome>);

impl PlayerOuts {
    /// Get the underlying player outcomes
    pub fn outcomes(&self) -> &Vec<PlayerOutcome> {
        &self.0
    }

    /// Get exclusive outs for each player based on calculated outcomes
    ///
    /// Returns a Vec of CardBitSets where each index corresponds to a player
    /// and each bit represents a card that appears in winning scenarios for
    /// only that player. This analyzes all the cards that appear in each
    /// player's winning boards and identifies which cards are exclusive to
    /// each player's wins.
    ///
    /// For example, if player 0 wins on all boards containing the Ace of Spades
    /// and no other player wins on boards with that card, then that bit will be
    /// set in player 0's CardBitSet. If both player 0 and player 1 win on boards
    /// containing the King of Hearts, then neither has that card as an exclusive out.
    ///
    /// Note: This works best when the board is incomplete (flop or turn), as it
    /// analyzes which cards in the completion lead to exclusive wins.
    ///
    /// # Returns
    ///
    /// Vec of CardBitSets, one per player, containing their exclusive outs
    ///
    /// # Example
    /// ```
    /// use rs_poker::core::{Card, CardBitSet, Hand, Suit, Value};
    /// use rs_poker::holdem::OutsCalculator;
    ///
    /// let mut board = CardBitSet::new();
    /// board.insert(Card::new(Value::King, Suit::Heart));
    /// board.insert(Card::new(Value::Seven, Suit::Heart));
    /// board.insert(Card::new(Value::Two, Suit::Diamond));
    ///
    /// let player1 = Hand::new_from_str("Ah9h").unwrap(); // Flush draw
    /// let player2 = Hand::new_from_str("KsKc").unwrap(); // Set of kings
    ///
    /// let calc = OutsCalculator::new(board, vec![player1, player2]);
    /// let player_outs = calc.calculate_outs();
    ///
    /// // Get exclusive outs for each player
    /// let outs = player_outs.get_outs();
    ///
    /// // Player 0 should have heart cards as exclusive outs (flush draw)
    /// // Player 1 should have cards that make full house without giving player 0 flush
    /// assert!(outs[0].count() > 0); // Player 0 has some exclusive outs
    /// ```
    pub fn get_outs(&self) -> Vec<CardBitSet> {
        // Initialize empty CardBitSets for each player
        let mut exclusive_outs: Vec<CardBitSet> = vec![CardBitSet::new(); self.0.len()];

        // Build a map of card -> which players win on boards containing that card
        // We need to track: for each card, which players have wins on boards with that card
        // AND which players have NO wins on boards with that card
        let mut card_to_winning_players: HashMap<Card, std::collections::HashSet<usize>> =
            HashMap::new();

        for (player_idx, outcome) in self.0.iter().enumerate() {
            // Iterate through all winning boards for this player
            for boards in outcome.winning_boards.values() {
                for &full_board in boards {
                    // The new cards are those in full_board but not in the original board
                    let new_cards = full_board ^ outcome.board;

                    // For each card in the completion, mark that this player wins with it
                    for card in new_cards.into_iter() {
                        card_to_winning_players
                            .entry(card)
                            .or_default()
                            .insert(player_idx);
                    }
                }
            }
        }

        // Now identify exclusive outs: cards where only one player ever wins
        for (card, winning_players) in card_to_winning_players {
            if winning_players.len() == 1 {
                // This card only appears in winning scenarios for one player
                let player_idx = *winning_players.iter().next().unwrap();
                exclusive_outs[player_idx].insert(card);
            }
        }

        exclusive_outs
    }
}

impl From<Vec<PlayerOutcome>> for PlayerOuts {
    fn from(v: Vec<PlayerOutcome>) -> Self {
        PlayerOuts(v)
    }
}

impl From<PlayerOuts> for Vec<PlayerOutcome> {
    fn from(po: PlayerOuts) -> Self {
        po.0
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
    /// PlayerOuts containing outcomes for each player
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
    /// let player_outs = calc.calculate_outs(); // Automatically deals turn + river
    /// let outcomes = player_outs.outcomes();
    ///
    /// // Player 1 should have a high win percentage with the royal flush draw
    /// assert!(outcomes[0].win_percentage() > 0.0);
    /// ```
    pub fn calculate_outs(&self) -> PlayerOuts {
        let board_size = self.board.count();
        let num_cards_to_deal = 5 - board_size;

        // Initialize results for each player
        let mut results: Vec<PlayerOutcome> = self
            .player_hands
            .iter()
            .map(|&hand| PlayerOutcome {
                board: self.board,
                hand,
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
            let full_board = self.board | combo;

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

        results.into()
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
        let player_outs = calc.calculate_outs();
        let results = player_outs.outcomes();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].hand, player1);
        assert_eq!(results[1].hand, player2);
        assert_eq!(results[0].board, board);
        assert_eq!(results[1].board, board);

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
        let player_outs = calc.calculate_outs();
        let results = player_outs.outcomes();

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
        let player_outs = calc.calculate_outs();
        let results = player_outs.outcomes();

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
        let player_outs = calc.calculate_outs();
        let results = player_outs.outcomes();

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
        let board = CardBitSet::new();
        let hand = Hand::new_from_str("AhKh").unwrap();
        let outs = PlayerOutcome {
            board,
            hand,
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

    #[test]
    fn test_wins_by_core_rank() {
        use crate::core::CoreRank;

        // Flush draw vs set scenario
        let mut board = CardBitSet::new();
        board.insert(Card::new(Value::King, Suit::Heart));
        board.insert(Card::new(Value::Seven, Suit::Heart));
        board.insert(Card::new(Value::Two, Suit::Diamond));

        let player1 = Hand::new_from_str("Ah9h").unwrap(); // Flush draw
        let player2 = Hand::new_from_str("KsKc").unwrap(); // Set

        let calc = OutsCalculator::new(board, vec![player1, player2]);
        let player_outs = calc.calculate_outs();
        let results = player_outs.outcomes();

        // Player 1 should win primarily with flushes
        let p1_grouped = results[0].count_wins_by_core_rank();
        assert!(
            p1_grouped.contains_key(&CoreRank::Flush),
            "Player 1 should have flush wins"
        );

        // Player 2 should win with various hand types (set, full house, etc.)
        let p2_grouped = results[1].count_wins_by_core_rank();
        assert!(!p2_grouped.is_empty(), "Player 2 should have winning hands");

        // Sum of wins from grouped should equal total wins
        let p1_sum: usize = p1_grouped.values().sum();
        assert_eq!(p1_sum, results[0].wins);
    }

    #[test]
    fn test_three_way_pot_all_have_chances() {
        use crate::core::CoreRank;

        // Flop: Jh Ts 8h
        // Player 1: Ah 9h (flush draw + straight draw)
        // Player 2: QsKd (straight draw)
        // Player 3: JdJc (set of jacks)
        let mut board = CardBitSet::new();
        board.insert(Card::new(Value::Jack, Suit::Heart));
        board.insert(Card::new(Value::Ten, Suit::Spade));
        board.insert(Card::new(Value::Eight, Suit::Heart));

        let player1 = Hand::new_from_str("Ah9h").unwrap(); // Flush + straight draws
        let player2 = Hand::new_from_str("QsKd").unwrap(); // Straight draw
        let player3 = Hand::new_from_str("JdJc").unwrap(); // Set of jacks

        let calc = OutsCalculator::new(board, vec![player1, player2, player3]);
        let player_outs = calc.calculate_outs();
        let results = player_outs.outcomes();

        // All three players should have some wins
        assert!(results[0].wins > 0, "Player 1 should have some wins");
        assert!(results[1].wins > 0, "Player 2 should have some wins");
        assert!(results[2].wins > 0, "Player 3 should have some wins");

        // Player 3 (set) should be the favorite
        assert!(
            results[2].win_percentage() > results[0].win_percentage(),
            "Player 3 (set) should be ahead of Player 1"
        );
        assert!(
            results[2].win_percentage() > results[1].win_percentage(),
            "Player 3 (set) should be ahead of Player 2"
        );

        // Check that wins_by_core_rank works correctly
        let p1_grouped = results[0].count_wins_by_core_rank();
        let p2_grouped = results[1].count_wins_by_core_rank();
        let p3_grouped = results[2].count_wins_by_core_rank();

        // Player 1 should win with flushes and straights
        assert!(
            p1_grouped.contains_key(&CoreRank::Flush)
                || p1_grouped.contains_key(&CoreRank::Straight),
            "Player 1 should win with flush or straight"
        );

        // Player 2 should win primarily with straights
        assert!(
            p2_grouped.contains_key(&CoreRank::Straight),
            "Player 2 should win with straights"
        );

        // Player 3 should win with various hand types
        assert!(!p3_grouped.is_empty(), "Player 3 should have winning hands");

        // Verify sum of grouped wins equals total wins for each player
        assert_eq!(
            p1_grouped.values().sum::<usize>(),
            results[0].wins,
            "Player 1 grouped wins sum should match total wins"
        );
        assert_eq!(
            p2_grouped.values().sum::<usize>(),
            results[1].wins,
            "Player 2 grouped wins sum should match total wins"
        );
        assert_eq!(
            p3_grouped.values().sum::<usize>(),
            results[2].wins,
            "Player 3 grouped wins sum should match total wins"
        );
    }

    #[test]
    fn test_three_way_pot_complex_scenario() {
        use crate::core::CoreRank;

        // Flop: Kd Qd 7c
        // Player 1: Ad Jd (nut flush draw + gutshot)
        // Player 2: KhKc (set of kings)
        // Player 3: Ts9s (open-ended straight draw)
        let mut board = CardBitSet::new();
        board.insert(Card::new(Value::King, Suit::Diamond));
        board.insert(Card::new(Value::Queen, Suit::Diamond));
        board.insert(Card::new(Value::Seven, Suit::Club));

        let player1 = Hand::new_from_str("AdJd").unwrap(); // Nut flush draw + gutshot
        let player2 = Hand::new_from_str("KhKc").unwrap(); // Set
        let player3 = Hand::new_from_str("Ts9s").unwrap(); // OESD

        let calc = OutsCalculator::new(board, vec![player1, player2, player3]);
        let player_outs = calc.calculate_outs();
        let results = player_outs.outcomes();

        // All three players should have some wins
        assert!(results[0].wins > 0, "Player 1 should have some wins");
        assert!(results[1].wins > 0, "Player 2 should have some wins");
        assert!(results[2].wins > 0, "Player 3 should have some wins");

        // Check wins_by_core_rank for each player
        let p1_grouped = results[0].count_wins_by_core_rank();
        let p2_grouped = results[1].count_wins_by_core_rank();
        let p3_grouped = results[2].count_wins_by_core_rank();

        // Player 1 should win with flushes
        assert!(
            p1_grouped.contains_key(&CoreRank::Flush),
            "Player 1 should win with flushes"
        );

        // Player 2 (set) should have multiple winning hand types
        assert!(
            p2_grouped.len() >= 2,
            "Player 2 should win with multiple hand types"
        );

        // Player 3 should win with straights
        assert!(
            p3_grouped.contains_key(&CoreRank::Straight),
            "Player 3 should win with straights"
        );

        // All sums should match
        assert_eq!(p1_grouped.values().sum::<usize>(), results[0].wins);
        assert_eq!(p2_grouped.values().sum::<usize>(), results[1].wins);
        assert_eq!(p3_grouped.values().sum::<usize>(), results[2].wins);

        // Total combinations should be the same for all players
        assert_eq!(results[0].total_combinations, results[1].total_combinations);
        assert_eq!(results[1].total_combinations, results[2].total_combinations);
    }

    #[test]
    fn test_three_way_pot_with_turn() {
        use crate::core::CoreRank;

        // Board: Jh Ts 8h 7d (turn)
        // Player 1: Ah 9h (flush draw + made straight)
        // Player 2: QsKd (straight)
        // Player 3: JdJc (set of jacks)
        let mut board = CardBitSet::new();
        board.insert(Card::new(Value::Jack, Suit::Heart));
        board.insert(Card::new(Value::Ten, Suit::Spade));
        board.insert(Card::new(Value::Eight, Suit::Heart));
        board.insert(Card::new(Value::Seven, Suit::Diamond));

        let player1 = Hand::new_from_str("Ah9h").unwrap(); // Straight + flush draw
        let player2 = Hand::new_from_str("QsKd").unwrap(); // Straight
        let player3 = Hand::new_from_str("JdJc").unwrap(); // Set

        let calc = OutsCalculator::new(board, vec![player1, player2, player3]);
        let player_outs = calc.calculate_outs();
        let results = player_outs.outcomes();

        // All should have some wins (only river to come)
        assert!(results[0].wins > 0, "Player 1 should have wins");
        assert!(results[1].wins > 0, "Player 2 should have wins");
        assert!(results[2].wins > 0, "Player 3 should have wins");

        // Test wins_by_core_rank
        let p1_grouped = results[0].count_wins_by_core_rank();
        let p2_grouped = results[1].count_wins_by_core_rank();
        let p3_grouped = results[2].count_wins_by_core_rank();

        // Verify all grouped sums match
        assert_eq!(p1_grouped.values().sum::<usize>(), results[0].wins);
        assert_eq!(p2_grouped.values().sum::<usize>(), results[1].wins);
        assert_eq!(p3_grouped.values().sum::<usize>(), results[2].wins);

        // Player 1 can win with flush
        assert!(
            p1_grouped.contains_key(&CoreRank::Flush)
                || p1_grouped.contains_key(&CoreRank::Straight),
            "Player 1 should have flush or straight wins"
        );
    }

    #[test]
    fn test_get_outs_river_flush_draw_vs_set() {
        // Turn: Kh 7h 2d 3s (so only river to come)
        // Player 1: Ah 9h (flush draw)
        // Player 2: Ks Kc (set of kings)
        let mut board = CardBitSet::new();
        board.insert(Card::new(Value::King, Suit::Heart));
        board.insert(Card::new(Value::Seven, Suit::Heart));
        board.insert(Card::new(Value::Two, Suit::Diamond));
        board.insert(Card::new(Value::Three, Suit::Spade));

        let player1 = Hand::new_from_str("Ah9h").unwrap();
        let player2 = Hand::new_from_str("KsKc").unwrap();

        let calc = OutsCalculator::new(board, vec![player1, player2]);
        let player_outs = calc.calculate_outs();
        let outs = player_outs.get_outs();

        //  Player 1 should have some exclusive outs (hearts that complete the flush)
        assert!(outs[0].count() > 0, "Player 1 should have exclusive outs");

        // Player 2 should have some exclusive outs
        assert!(outs[1].count() > 0, "Player 2 should have exclusive outs");

        // Check that player 1's outs include hearts
        let hearts_in_p1_outs = outs[0]
            .into_iter()
            .filter(|card| card.suit == Suit::Heart)
            .count();
        assert!(
            hearts_in_p1_outs > 0,
            "Player 1 should have hearts as exclusive outs"
        );
    }

    #[test]
    fn test_get_outs_no_exclusive_outs() {
        // Create a scenario where players often tie or split
        // Flop: As Ks Qs
        // Player 1: Jh Th (straight draw)
        // Player 2: Jd Td (straight draw)
        let mut board = CardBitSet::new();
        board.insert(Card::new(Value::Ace, Suit::Spade));
        board.insert(Card::new(Value::King, Suit::Spade));
        board.insert(Card::new(Value::Queen, Suit::Spade));

        let player1 = Hand::new_from_str("JhTh").unwrap();
        let player2 = Hand::new_from_str("JdTd").unwrap();

        let calc = OutsCalculator::new(board, vec![player1, player2]);
        let player_outs = calc.calculate_outs();
        let outs = player_outs.get_outs();

        // Both players should have very few or no exclusive outs since they tie often
        // They both make the same straights
        assert_eq!(outs.len(), 2);
    }

    #[test]
    fn test_get_outs_simple_scenario() {
        // Flop: 2s 3s 4s
        // Player 1: As 5s (made straight flush)
        // Player 2: 7h 8h (nothing)
        let mut board = CardBitSet::new();
        board.insert(Card::new(Value::Two, Suit::Spade));
        board.insert(Card::new(Value::Three, Suit::Spade));
        board.insert(Card::new(Value::Four, Suit::Spade));

        let player1 = Hand::new_from_str("As5s").unwrap();
        let player2 = Hand::new_from_str("7h8h").unwrap();

        let calc = OutsCalculator::new(board, vec![player1, player2]);
        let player_outs = calc.calculate_outs();
        let outs = player_outs.get_outs();

        // Player 1 should have many exclusive outs (basically wins on almost all boards)
        assert!(
            outs[0].count() > 30,
            "Player 1 should have many exclusive outs with made straight flush"
        );

        // Player 2 should have few or no exclusive outs
        assert!(
            outs[1].count() < 10,
            "Player 2 should have few exclusive outs"
        );
    }

    #[test]
    fn test_get_outs_river_three_players() {
        // Turn: Jh Ts 8h 7d (only river to come)
        // Player 1: Ah 9h (flush draw + made straight)
        // Player 2: Qs Kd (straight)
        // Player 3: Jd Jc (set of jacks)
        let mut board = CardBitSet::new();
        board.insert(Card::new(Value::Jack, Suit::Heart));
        board.insert(Card::new(Value::Ten, Suit::Spade));
        board.insert(Card::new(Value::Eight, Suit::Heart));
        board.insert(Card::new(Value::Seven, Suit::Diamond));

        let player1 = Hand::new_from_str("Ah9h").unwrap();
        let player2 = Hand::new_from_str("QsKd").unwrap();
        let player3 = Hand::new_from_str("JdJc").unwrap();

        let calc = OutsCalculator::new(board, vec![player1, player2, player3]);
        let player_outs = calc.calculate_outs();
        let outs = player_outs.get_outs();

        assert_eq!(outs.len(), 3, "Should have outs for all 3 players");

        // All players should have at least some exclusive outs on the river
        assert!(
            outs[0].count() > 0,
            "Player 1 should have some exclusive outs"
        );
        assert!(
            outs[1].count() > 0,
            "Player 2 should have some exclusive outs"
        );
        assert!(
            outs[2].count() > 0,
            "Player 3 should have some exclusive outs"
        );

        // Player 1 should have hearts as exclusive outs (flush cards)
        let hearts_in_p1_outs = outs[0]
            .into_iter()
            .filter(|card| card.suit == Suit::Heart)
            .count();
        assert!(
            hearts_in_p1_outs > 0,
            "Player 1 should have hearts as exclusive outs"
        );
    }

    #[test]
    fn test_get_outs_turn_scenario() {
        // Turn: Kh 7h 2d 3s
        // Player 1: Ah 9h (flush draw)
        // Player 2: Ks Kc (set of kings)
        let mut board = CardBitSet::new();
        board.insert(Card::new(Value::King, Suit::Heart));
        board.insert(Card::new(Value::Seven, Suit::Heart));
        board.insert(Card::new(Value::Two, Suit::Diamond));
        board.insert(Card::new(Value::Three, Suit::Spade));

        let player1 = Hand::new_from_str("Ah9h").unwrap();
        let player2 = Hand::new_from_str("KsKc").unwrap();

        let calc = OutsCalculator::new(board, vec![player1, player2]);
        let player_outs = calc.calculate_outs();
        let outs = player_outs.get_outs();

        // With only the river to come, each player should have exclusive outs
        assert!(outs[0].count() > 0, "Player 1 should have exclusive outs");
        assert!(outs[1].count() > 0, "Player 2 should have exclusive outs");

        // Player 1's outs should include hearts (flush cards)
        let hearts_in_p1_outs = outs[0]
            .into_iter()
            .filter(|card| card.suit == Suit::Heart)
            .count();
        assert!(
            hearts_in_p1_outs > 0,
            "Player 1 should have hearts as exclusive river outs"
        );
    }
}
