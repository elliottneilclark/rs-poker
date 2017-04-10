#![feature(test)]
extern crate rs_poker;
extern crate test;
extern crate rand;

use rs_poker::core::{Deck, Rankable, Hand, Flattenable};

#[bench]
fn rank_one(b: &mut test::Bencher) {
    let d = Deck::default().flatten();
    let hand = Hand::new_with_cards(d.sample(5));
    b.iter(|| hand.rank())
}

#[bench]
fn rank_seven(b: &mut test::Bencher) {
    let d = Deck::default().flatten();
    let hand = Hand::new_with_cards(d.sample(7));
    b.iter(|| hand.rank_seven())
}
