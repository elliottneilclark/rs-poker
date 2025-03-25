#[macro_use]
extern crate criterion;
extern crate rs_poker;

use rand::rng;
use rs_poker::core::{Deck, FlatDeck};

fn deal_all_flat_deck(c: &mut criterion::Criterion) {
    let mut rng = rng();
    let mut flat_deck = FlatDeck::default();

    c.bench_function("deal all from FlatDeck", |b| {
        b.iter(|| {
            flat_deck.shuffle(&mut rng);
            while !flat_deck.is_empty() {
                let _card = flat_deck.deal().unwrap();
            }
        });
    });
}

fn deal_all_deck(c: &mut criterion::Criterion) {
    let mut rng = rng();
    let mut deck = Deck::default();

    c.bench_function("deal all from Deck", |b| {
        b.iter(|| {
            while !deck.is_empty() {
                let _card = deck.deal(&mut rng).unwrap();
            }
        });
    });
}

criterion_group!(benches, deal_all_flat_deck, deal_all_deck);
criterion_main!(benches);
