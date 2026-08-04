//! Blog-post benchmark (mind-reader part 2): the frozen pre-perfect-hash
//! evaluator ("oracle") on the same 1024-random-7-card-hands workload as
//! benches/rank.rs, for an apples-to-apples old-vs-new comparison.
//!
//! The oracle body is `include!`d from src/core/rank_oracle.rs, the same source
//! the differential tests in src/core/rank.rs use. That module is
//! `cfg(test)`-only so it cannot be imported normally, and sharing the file
//! keeps the benchmarked oracle from drifting away from the tested one.

#[macro_use]
extern crate criterion;
extern crate rand;
extern crate rs_poker;

use criterion::Criterion;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rs_poker::core::Card;

#[allow(dead_code)] // `category()` is used by the tests, not by this bench.
mod oracle {
    include!("../src/core/rank_oracle.rs");
}
use oracle::rank_u64;

fn oracle_random_seven_throughput(c: &mut Criterion) {
    // Mirror benches/rank.rs exactly: same seed (identical hand sets) and the
    // same [Card; 7] starting point, packing each hand into the oracle's u64
    // mask *inside* the timed loop just as `h[..].rank()` packs there. That
    // keeps the old-vs-new throughput comparison apples-to-apples.
    let mut deck: Vec<Card> = (0u8..52).map(Card::from).collect();
    let mut rng = StdRng::seed_from_u64(42);
    let hands: Vec<[Card; 7]> = (0..1024)
        .map(|_| {
            deck.shuffle(&mut rng);
            [
                deck[0], deck[1], deck[2], deck[3], deck[4], deck[5], deck[6],
            ]
        })
        .collect();
    c.bench_function("Oracle rank 1024 random 7card hands", move |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for h in &hands {
                let mask = h.iter().fold(0u64, |a, &c| a | (1u64 << u8::from(c)));
                let r = rank_u64(mask);
                if r > acc {
                    acc = r;
                }
            }
            acc
        })
    });
}

criterion_group!(benches, oracle_random_seven_throughput);
criterion_main!(benches);
