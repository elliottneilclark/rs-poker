#[macro_use]
extern crate criterion;
extern crate rs_poker;

use criterion::{BenchmarkId, Criterion};
use rand::rng;
use rs_poker::core::{CardBitSet, Deck};

fn sample_one_by_deck_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_one");

    for num_cards in [1, 2, 5, 13, 26, 45, 52] {
        group.bench_with_input(
            BenchmarkId::new("cards_remaining", num_cards),
            &num_cards,
            |b, &num_cards| {
                let mut rng = rng();
                // Build a CardBitSet with exactly num_cards cards
                let mut cards = CardBitSet::new();
                for card in Deck::default().into_iter().take(num_cards) {
                    cards.insert(card);
                }
                b.iter(|| std::hint::black_box(cards.sample_one(&mut rng)));
            },
        );
    }
    group.finish();
}

fn deal_all_cards_bitset(c: &mut Criterion) {
    c.bench_function("deal all 52 via sample_one", |b| {
        let mut rng = rng();
        b.iter(|| {
            let mut cards = CardBitSet::default();
            while !cards.is_empty() {
                let card = cards.sample_one(&mut rng).unwrap();
                cards.remove(card);
            }
        });
    });
}

criterion_group!(benches, sample_one_by_deck_size, deal_all_cards_bitset);
criterion_main!(benches);
