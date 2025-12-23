use rs_poker::core::{Card, CardBitSet, Hand, Suit, Value};
use rs_poker::holdem::OutsCalculator;

fn main() {
    println!("=== Texas Hold'em Outs Calculator Example ===\n");

    // Example 1: Flop scenario with a flush draw
    println!("Example 1: Flush Draw vs Top Pair");
    println!("Board: As Ks Qs");
    let mut board = CardBitSet::new();
    board.insert(Card::new(Value::Ace, Suit::Spade));
    board.insert(Card::new(Value::King, Suit::Spade));
    board.insert(Card::new(Value::Queen, Suit::Spade));

    println!("Player 1: Js Ts (Royal Flush Draw)");
    let player1 = Hand::new_from_str("JsTs").unwrap();

    println!("Player 2: Ah Kd (Top Two Pair)\n");
    let player2 = Hand::new_from_str("AhKd").unwrap();

    let calc = OutsCalculator::new(board, vec![player1, player2]);

    println!("Calculating all possible turn + river combinations...");
    println!(
        "Board has {} cards, so {} cards to deal",
        board.count(),
        5 - board.count()
    );

    let results = calc.calculate_outs();

    for result in &results {
        println!("Player {} Results:", result.player_idx + 1);
        println!("  Wins: {}", result.wins);
        println!("  Ties: {}", result.ties);
        println!("  Win %: {:.2}%", result.win_percentage());
        println!("  Tie %: {:.2}%", result.tie_percentage());

        if !result.winning_boards.is_empty() {
            println!("  Winning hand types:");
            let mut ranks: Vec<_> = result.winning_boards.keys().collect();
            ranks.sort();
            for rank in ranks.iter().rev() {
                let count = result.winning_boards[rank].len();
                println!("    {:?}: {} times", rank, count);
            }
        }
        println!();
    }

    // Example 2: Turn scenario - one card to come
    println!("\n=== Example 2: Turn Scenario ===");
    println!("Board: 7h 8h 9h Tc");
    let mut board2 = CardBitSet::new();
    board2.insert(Card::new(Value::Seven, Suit::Heart));
    board2.insert(Card::new(Value::Eight, Suit::Heart));
    board2.insert(Card::new(Value::Nine, Suit::Heart));
    board2.insert(Card::new(Value::Ten, Suit::Club));

    println!("Player 1: Jc Qs (Straight)");
    let player3 = Hand::new_from_str("JcQs").unwrap();

    println!("Player 2: Ah Kh (Flush Draw + Straight Draw)\n");
    let player4 = Hand::new_from_str("AhKh").unwrap();

    let calc2 = OutsCalculator::new(board2, vec![player3, player4]);

    println!("Calculating all possible river cards...");
    println!(
        "Board has {} cards, so {} card to deal",
        board2.count(),
        5 - board2.count()
    );
    let results2 = calc2.calculate_outs();

    for result in &results2 {
        println!("Player {} Results:", result.player_idx + 1);
        println!("  Wins: {}", result.wins);
        println!("  Ties: {}", result.ties);
        println!("  Win %: {:.2}%", result.win_percentage());
        println!("  Tie %: {:.2}%", result.tie_percentage());
        println!();
    }

    // Example 3: Flush draw from behind
    println!("\n=== Example 3: Flush Draw vs Set ===");
    println!("Board: Kh 7h 2d");
    let mut board3 = CardBitSet::new();
    board3.insert(Card::new(Value::King, Suit::Heart));
    board3.insert(Card::new(Value::Seven, Suit::Heart));
    board3.insert(Card::new(Value::Two, Suit::Diamond));

    println!("Player 1: Ah 9h (Flush Draw - currently behind)");
    let player5 = Hand::new_from_str("Ah9h").unwrap();

    println!("Player 2: Ks Kc (Set of Kings - currently ahead)\n");
    let player6 = Hand::new_from_str("KsKc").unwrap();

    let calc3 = OutsCalculator::new(board3, vec![player5, player6]);

    println!("Calculating all possible turn + river combinations...");
    let results3 = calc3.calculate_outs();

    for result in &results3 {
        println!("Player {} Results:", result.player_idx + 1);
        println!("  Wins: {}", result.wins);
        println!("  Ties: {}", result.ties);
        println!("  Win %: {:.2}%", result.win_percentage());
        println!("  Tie %: {:.2}%", result.tie_percentage());

        if !result.winning_boards.is_empty() {
            println!("  Winning hand types:");
            let mut ranks: Vec<_> = result.winning_boards.keys().collect();
            ranks.sort();
            for rank in ranks.iter().rev() {
                let count = result.winning_boards[rank].len();
                println!("    {:?}: {} times", rank, count);
            }
        }
        println!();
    }

    // Example 4: Heads-up preflop
    println!("\n=== Example 4: Preflop All-In ===");
    println!("Board: (empty)");
    let board4 = CardBitSet::new();

    println!("Player 1: Ac Ad (Pocket Aces)");
    let player7 = Hand::new_from_str("AcAd").unwrap();

    println!("Player 2: Ks Kh (Pocket Kings)\n");
    let player8 = Hand::new_from_str("KsKh").unwrap();

    let calc4 = OutsCalculator::new(board4, vec![player7, player8]);

    println!("Calculating all possible 5-card boards...");
    println!(
        "Board has {} cards, so {} cards to deal",
        board4.count(),
        5 - board4.count()
    );
    println!("(This may take a moment...)\n");

    let results4 = calc4.calculate_outs();

    for result in &results4 {
        println!("Player {} Results:", result.player_idx + 1);
        println!("  Wins: {}", result.wins);
        println!("  Ties: {}", result.ties);
        println!("  Win %: {:.2}%", result.win_percentage());
        println!("  Tie %: {:.2}%", result.tie_percentage());
        println!();
    }
}
