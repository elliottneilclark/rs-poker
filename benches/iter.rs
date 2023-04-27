#[macro_use]
extern crate criterion;
extern crate rand;
extern crate rs_poker;

use criterion::Criterion;
use rs_poker::core::{CardIter, FlatDeck};

fn iter_in_deck(c: &mut Criterion) {
    c.bench_function("Iter all 5 cards hand in deck", |b| {
        b.iter(|| {
            let d: FlatDeck = FlatDeck::default();
            d.into_iter().count()
        })
    });
}

fn iter_hand(c: &mut Criterion) {
    let d: FlatDeck = FlatDeck::default();
    let hand = d.sample(7);

    c.bench_function("Iter all 5 cards hand in 7 card hand ", move |b| {
        b.iter(|| CardIter::new(&hand[..], 5).count())
    });
}

criterion_group!(benches, iter_in_deck, iter_hand);
criterion_main!(benches);
