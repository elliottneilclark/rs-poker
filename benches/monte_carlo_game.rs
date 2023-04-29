#[macro_use]
extern crate criterion;
extern crate rs_poker;

use criterion::Criterion;
use rs_poker::core::Hand;
use rs_poker::holdem::MonteCarloGame;

fn simulate_one_monte_game(c: &mut Criterion) {
    let hands = ["AdAh", "2c2s"]
        .iter()
        .map(|s| Hand::new_from_str(s).expect("Should be able to create a hand."))
        .collect();
    let mut g = MonteCarloGame::new(hands).expect("Should be able to create a game.");

    c.bench_function("Simulate AdAh vs 2c2s", move |b| {
        b.iter(|| {
            let r = g.simulate();
            g.reset();
            r
        })
    });
}

fn simulate_unseen_hole_cards(c: &mut Criterion) {
    let hands = vec![Hand::new_from_str("KsKd").unwrap(), Hand::default()];
    let mut g = MonteCarloGame::new(hands).expect("Should be able to create a game.");

    c.bench_function("Simulate KsKd vs everything", move |b| {
        b.iter(|| {
            let r = g.simulate();
            g.reset();
            r
        })
    });
}

criterion_group!(benches, simulate_one_monte_game, simulate_unseen_hole_cards);
criterion_main!(benches);
