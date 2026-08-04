#[macro_use]
extern crate criterion;
extern crate rs_poker;

use std::hint::black_box;

use criterion::Criterion;
use rs_poker::core::{Card, CardBitSet, Deck, FlatDeck};

// Blog-post benchmark (mind-reader part 1): membership checks on a 7-card
// hand, Vec<Card> vs CardBitSet. Probes all 52 cards so hits and misses are
// both represented.

fn contains(c: &mut Criterion) {
    // One sampled 7-card hand and one probe deck, shared by both benches, so the
    // Vec and the CardBitSet are measured against strictly identical inputs.
    let d: FlatDeck = Deck::default().into();
    let hand_vec: Vec<Card> = d.sample(7);
    let mut hand_bits = CardBitSet::new();
    for &card in &hand_vec {
        hand_bits.insert(card);
    }
    let deck: Vec<Card> = Deck::default().into_iter().collect();

    c.bench_function("Vec<Card> contains, 52 probes of 7 cards", |b| {
        b.iter(|| {
            let mut hits = 0;
            for card in &deck {
                // Both arms probe with `black_box(*card)`. Passing
                // `black_box(card)` here instead would give the Vec arm an
                // opaque pointer to deref that the optimizer must assume can
                // alias the Vec's buffer, while the bitset arm below gets an
                // opaque value in a register. That skews the ratio this bench
                // exists to report.
                if hand_vec.contains(&black_box(*card)) {
                    hits += 1;
                }
            }
            hits
        })
    });

    c.bench_function("CardBitSet contains, 52 probes of 7 cards", |b| {
        b.iter(|| {
            let mut hits = 0;
            for card in &deck {
                if hand_bits.contains(black_box(*card)) {
                    hits += 1;
                }
            }
            hits
        })
    });
}

criterion_group!(benches, contains);
criterion_main!(benches);
