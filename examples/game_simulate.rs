extern crate rs_poker;
use rs_poker::core::Hand;
use rs_poker::holdem::MonteCarloGame;

const GAMES_COUNT: i32 = 3_000_000;
const STARTING_HANDS: [&str; 2] = ["Adkh", "8c8s"];

fn main() {
    let hands = STARTING_HANDS
        .iter()
        .map(|s| Hand::new_from_str(s).expect("Should be able to create a hand."))
        .collect();

    let mut g = MonteCarloGame::new(hands).expect("Should be able to create a game.");
    let mut wins: [u64; 2] = [0, 0];

    for _ in 0..GAMES_COUNT {
        let r = g.simulate();
        g.reset();
        wins[r.0.ones().next().unwrap()] += 1
    }

    let normalized: Vec<f64> = wins
        .iter()
        .map(|cnt| *cnt as f64 / GAMES_COUNT as f64)
        .collect();

    println!("Starting Hands =\t{:?}", STARTING_HANDS);
    println!("Wins =\t\t\t{:?}", wins);
    println!("Normalized Wins =\t{:?}", normalized);
}
