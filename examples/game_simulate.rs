extern crate rs_poker;
use rs_poker::core::Hand;
use rs_poker::holdem::MonteCarloGame;

fn main() {
    let hands = ["Adkh", "8c8s"]
        .iter()
        .map(|s| Hand::new_from_str(s).expect("Should be able to create a hand."))
        .collect();
    let mut g = MonteCarloGame::new_with_hands(hands).expect("Should be able to create a game.");
    let mut wins: [u64; 2] = [0, 0];
    for _ in 0..2_000_000 {
        let r = g.simulate().expect("There should be one best rank.");
        g.reset();
        wins[r.0] += 1
    }
    println!("Wins = {:?}", wins);
}
