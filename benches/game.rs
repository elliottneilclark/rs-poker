#![feature(test)]
extern crate rs_poker;
extern crate test;
use rs_poker::core::Hand;
use rs_poker::holdem::Game;


#[bench]
fn simulate_one_game(b: &mut test::Bencher) {
    let hands = ["AdAh", "2c2s"]
        .iter()
        .map(|s| Hand::new_from_str(s).expect("Should be able to create a hand."))
        .collect();
    let mut g = Game::new_with_hands(hands).expect("Should be able to create a game.");
    b.iter(|| {
               let r = g.simulate().expect("There should be one best rank.");
               g.reset();
               r
           })

}
