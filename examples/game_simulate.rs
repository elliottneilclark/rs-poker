extern crate rs_poker;
use rs_poker::core::{Hand, Card, Value, Suit};
use rs_poker::holdem::{MonteCarloGame};

fn main() {
    let hands = ["Ahjs", "td9d"]
        .iter()
        .map(|s| Hand::new_from_str(s).expect("Should be able to create a hand."))
        .collect();
    let board = vec![Card {
        value: Value::Jack,
        suit: Suit::Diamond
    },Card {
        value: Value::Eight,
        suit: Suit::Club
    },Card {
        value: Value::Three,
        suit: Suit::Diamond
    }];
    let mut g = MonteCarloGame::new_with_board(hands, board).expect("Should be able to create a game.");
    let mut wins: [u64; 2] = [0, 0];
    for _ in 0..2_000_000 {
        let r = g.simulate().expect("There should be one best rank.");
        g.reset();
        wins[r.0] += 1
    }
    println!("Wins = {:?}", wins);
}
