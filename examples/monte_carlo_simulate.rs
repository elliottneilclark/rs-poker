extern crate rs_poker;

mod common;

use clap::Parser;
use rs_poker::core::Hand;
use rs_poker::holdem::MonteCarloGame;

#[derive(Parser, Debug)]
#[command(
    name = "monte_carlo_simulate",
    about = "Simulate poker hands using Monte Carlo simulation",
    long_about = "Run Monte Carlo simulations to calculate win probabilities for poker hands.\n\
                  Provide two or more hands to compare their equity."
)]
struct Args {
    /// Tracing/logging options
    #[command(flatten)]
    tracing: common::TracingArgs,

    /// Poker hands to simulate (e.g., "Adkh" "8c8s")
    #[arg(required = true, num_args = 2..)]
    hands: Vec<String>,

    /// Number of simulations to run
    #[arg(short = 'n', long, default_value_t = 3_000_000)]
    games: i32,
}

fn main() {
    let args = Args::parse();
    args.tracing.init_tracing();

    // Parse the hands from the command line arguments
    let hands: Vec<Hand> = args
        .hands
        .iter()
        .map(|s| {
            Hand::new_from_str(s).unwrap_or_else(|_| {
                panic!(
                    "Invalid hand string: '{}'. Use format like 'Adkh' or '8c8s'",
                    s
                )
            })
        })
        .collect();

    println!("Starting Monte Carlo simulation...");
    println!("Hands: {:?}", args.hands);
    println!("Number of games: {}", args.games);
    println!();

    let num_hands = hands.len();
    let mut g = MonteCarloGame::new(hands).expect("Should be able to create a game.");
    let mut wins: Vec<u64> = vec![0; num_hands];

    for _ in 0..args.games {
        let r = g.simulate();
        g.reset();
        wins[r.0.ones().next().unwrap()] += 1;
    }

    let normalized: Vec<f64> = wins
        .iter()
        .map(|cnt| *cnt as f64 / args.games as f64)
        .collect();

    println!("Results:");
    println!("========");
    for (i, hand) in args.hands.iter().enumerate() {
        println!(
            "Hand {}: {} - Wins: {} ({:.2}%)",
            i + 1,
            hand,
            wins[i],
            normalized[i] * 100.0
        );
    }
}
