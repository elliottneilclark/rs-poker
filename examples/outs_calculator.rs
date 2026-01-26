//! # Outs Calculator Example
//!
//! Calculate win/tie probabilities and exclusive outs for Texas Hold'em hands.
//!
//! This example demonstrates how to use the `OutsCalculator` to analyze poker hands
//! at any stage of the game (preflop, flop, turn, or river). It calculates:
//! - Win and tie percentages for each player
//! - Exclusive outs (cards that only help that player win)
//! - Winning hand type distributions
//!
//! ## Usage
//!
//! ```bash
//! # Preflop equity
//! cargo run --example outs_calculator -- "AcAd" "KsKh"
//!
//! # With a flop
//! cargo run --example outs_calculator -- "JsTs" "AhKd" -b "AsKsQs"
//!
//! # Turn scenario showing exclusive outs (works best with 4 board cards)
//! cargo run --example outs_calculator -- "Ah9h" "KsKc" -b "Kh7h2d3s"
//!
//! # With verbose output showing winning hand types and actual out cards
//! cargo run --example outs_calculator -- "Ah9h" "KsKc" -b "Kh7h2d3s" -v
//! ```

mod common;

use clap::Parser;
use rs_poker::core::{CardBitSet, Hand};
use rs_poker::holdem::OutsCalculator;

#[derive(Parser, Debug)]
#[command(
    name = "outs_calculator",
    about = "Calculate outs and equity for Texas Hold'em hands",
    long_about = "Calculate win/tie probabilities and outs for Texas Hold'em hands.\n\
                  Provide player hands and optionally a board (flop/turn) to analyze all possible outcomes."
)]
struct Args {
    /// Tracing/logging options
    #[command(flatten)]
    tracing: common::TracingArgs,

    /// Player hands to analyze (e.g., "AcAd" "KsKh")
    #[arg(required = true, num_args = 2..)]
    hands: Vec<String>,

    /// Optional board cards (e.g., "AhKhQh" for flop, "AhKhQhTc" for turn)
    /// If not provided, calculates equity from preflop
    #[arg(short = 'b', long)]
    board: Option<String>,

    /// Show detailed winning hand types
    #[arg(short = 'v', long)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();
    args.tracing.init_tracing();

    println!("=== Texas Hold'em Outs Calculator ===\n");

    // Parse board (if provided)
    let board = if let Some(board_str) = &args.board {
        match Hand::new_from_str(board_str) {
            Ok(hand) => {
                let board_set: CardBitSet = hand.into();
                if board_set.count() > 5 {
                    eprintln!("Error: Board cannot have more than 5 cards");
                    std::process::exit(1);
                }
                println!("Board: {}", board_str);
                board_set
            }
            Err(e) => {
                eprintln!("Error parsing board '{}': {}", board_str, e);
                std::process::exit(1);
            }
        }
    } else {
        println!("Board: (empty - preflop)");
        CardBitSet::new()
    };

    // Parse player hands
    let mut hands = Vec::new();
    for (i, hand_str) in args.hands.iter().enumerate() {
        match Hand::new_from_str(hand_str) {
            Ok(hand) => {
                println!("Player {}: {}", i + 1, hand_str);
                hands.push(hand);
            }
            Err(e) => {
                eprintln!("Error parsing hand '{}': {}", hand_str, e);
                std::process::exit(1);
            }
        }
    }

    // Check for card conflicts between hands and board
    let mut all_cards = board;
    for hand in &hands {
        let hand_set: CardBitSet = (*hand).into();
        if !(all_cards & hand_set).is_empty() {
            eprintln!("Error: Duplicate cards detected between hands and/or board");
            std::process::exit(1);
        }
        all_cards |= hand_set;
    }

    println!();

    // Create calculator
    let calc = OutsCalculator::new(board, hands);

    // Calculate and display information
    let board_size = board.count();
    let cards_to_deal = 5 - board_size;

    match cards_to_deal {
        0 => println!("Board is complete. Analyzing final hand strengths..."),
        1 => println!(
            "Calculating all possible river cards ({} cards)...",
            cards_to_deal
        ),
        2 => println!(
            "Calculating all possible turn + river combinations ({} cards)...",
            cards_to_deal
        ),
        _ => {
            println!(
                "Calculating all possible board combinations ({} cards to deal)...",
                cards_to_deal
            );
            if board_size == 0 {
                println!("(This will take a moment for preflop calculations...)");
            }
        }
    }
    println!();

    // Calculate outs
    let player_outs = calc.calculate_outs();
    let results = player_outs.outcomes();

    // Get exclusive outs for each player
    let exclusive_outs = player_outs.get_outs();

    // Display results
    println!("Results:");
    println!("========\n");

    for (idx, result) in results.iter().enumerate() {
        println!("Player {} - {}:", idx + 1, args.hands[idx]);
        println!("  Wins:  {} ({:.2}%)", result.wins, result.win_percentage());
        println!("  Ties:  {} ({:.2}%)", result.ties, result.tie_percentage());

        // Display exclusive outs
        let outs_count = exclusive_outs[idx].count();
        if outs_count > 0 {
            println!("  Exclusive outs: {} cards", outs_count);
            if args.verbose {
                // Show the actual cards
                let outs_vec: Vec<_> = exclusive_outs[idx].into_iter().collect();
                let outs_str: Vec<String> = outs_vec.iter().map(|c| format!("{}", c)).collect();
                println!("    Cards: {}", outs_str.join(", "));
            }
        } else {
            println!("  Exclusive outs: None");
        }

        if args.verbose && !result.winning_boards.is_empty() {
            println!("\n  Winning hand types:");
            let grouped = result.count_wins_by_core_rank();
            let mut rank_counts: Vec<_> = grouped.iter().collect();

            // Sort by count (descending), then by rank (descending) as tiebreaker
            rank_counts.sort_by(|a, b| b.1.cmp(a.1).then_with(|| b.0.cmp(a.0)));

            for (rank, count) in rank_counts {
                println!("    {:?}: {} times", rank, count);
            }
        }
        println!();
    }

    // Summary
    println!("Summary:");
    println!("--------");
    println!(
        "Total possible outcomes analyzed: {}",
        results[0].total_combinations
    );

    // Find the favorite
    let favorite_idx = results
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.win_percentage().partial_cmp(&b.win_percentage()).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    println!(
        "Favorite: Player {} ({}) with {:.2}% win rate",
        favorite_idx + 1,
        args.hands[favorite_idx],
        results[favorite_idx].win_percentage()
    );
}
