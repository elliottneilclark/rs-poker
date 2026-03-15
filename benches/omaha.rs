#[macro_use]
extern crate criterion;
extern crate rs_poker;

use criterion::Criterion;
use rs_poker::core::Rankable;
use rs_poker::omaha::OmahaHand;

fn rank_plo4(c: &mut Criterion) {
    let hand = OmahaHand::new_from_str("AhKhQs9d", "Ts7s6s5d4d").unwrap();
    c.bench_function("Rank PLO4 hand (4 hole, 5 board)", move |b| {
        b.iter(|| hand.rank())
    });
}

fn rank_plo5(c: &mut Criterion) {
    let hand = OmahaHand::new_from_str("AhKhQs9d8c", "Ts7s6s5d4d").unwrap();
    c.bench_function("Rank PLO5 hand (5 hole, 5 board)", move |b| {
        b.iter(|| hand.rank())
    });
}

fn rank_plo6(c: &mut Criterion) {
    let hand = OmahaHand::new_from_str("AhKhQs9d8c2c", "Ts7s6s5d4d").unwrap();
    c.bench_function("Rank PLO6 hand (6 hole, 5 board)", move |b| {
        b.iter(|| hand.rank())
    });
}

fn rank_plo4_flop(c: &mut Criterion) {
    let hand = OmahaHand::new_from_str("AhKhQs9d", "Ts7s6s").unwrap();
    c.bench_function("Rank PLO4 hand on flop (4 hole, 3 board)", move |b| {
        b.iter(|| hand.rank())
    });
}

criterion_group!(benches, rank_plo4, rank_plo5, rank_plo6, rank_plo4_flop);
criterion_main!(benches);
