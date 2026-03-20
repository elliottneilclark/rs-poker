use clap::Parser;
use rs_poker::core::{Hand, RSPokerError};
use rs_poker::holdem::MonteCarloGame;

#[derive(Debug, thiserror::Error)]
pub enum SimulateError {
    #[error(transparent)]
    Poker(#[from] RSPokerError),
}

#[derive(Parser, Debug)]
#[command(
    name = "simulate",
    about = "Simulate poker hands using Monte Carlo simulation",
    long_about = "Run Monte Carlo simulations to calculate win probabilities for poker hands.\n\
                  Provide two or more hands to compare their equity."
)]
pub struct SimulateArgs {
    /// Poker hands to simulate (e.g., "Adkh" "8c8s")
    #[arg(required = true, num_args = 2..)]
    hands: Vec<String>,

    /// Number of simulations to run
    #[arg(short = 'n', long = "num-games", default_value_t = 3_000_000)]
    num_games: usize,
}

pub fn run(args: SimulateArgs) -> Result<(), SimulateError> {
    let hands: Vec<Hand> = args
        .hands
        .iter()
        .map(|s| Hand::new_from_str(s))
        .collect::<Result<Vec<_>, _>>()?;

    println!("Starting Monte Carlo simulation...");
    println!("Hands: {:?}", args.hands);
    println!("Number of games: {}", args.num_games);
    println!();

    let num_hands = hands.len();
    let mut g = MonteCarloGame::new(hands)?;
    let mut wins: Vec<u64> = vec![0; num_hands];
    let mut ties: Vec<u64> = vec![0; num_hands];

    for _ in 0..args.num_games {
        let r = g.simulate();
        g.reset();
        let winners: Vec<usize> = r.0.ones().collect();
        if winners.len() == 1 {
            wins[winners[0]] += 1;
        } else {
            for &w in &winners {
                ties[w] += 1;
            }
        }
    }

    let total = args.num_games as f64;

    println!("Results:");
    println!("========");
    for (i, hand) in args.hands.iter().enumerate() {
        let win_pct = wins[i] as f64 / total * 100.0;
        let tie_pct = ties[i] as f64 / total * 100.0;
        println!(
            "Hand {}: {} - Wins: {} ({:.2}%), Ties: {} ({:.2}%)",
            i + 1,
            hand,
            wins[i],
            win_pct,
            ties[i],
            tie_pct,
        );
    }

    Ok(())
}
