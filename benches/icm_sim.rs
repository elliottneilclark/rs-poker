#[macro_use]
extern crate criterion;
extern crate rs_poker;

use criterion::{Bencher, BenchmarkId, Criterion};
use rs_poker::simulated_icm::simulate_icm_tournament;

use rand::{Rng, rng};

fn simulate_one_tournament(c: &mut Criterion) {
    let payments = vec![10_000, 6_000, 4_000, 1_000, 800];
    let mut rng = rng();
    let mut group = c.benchmark_group("Tournament ICM");

    for num_players in [2, 3, 4, 6, 128, 256, 8000].iter() {
        let id = BenchmarkId::new("num_players", num_players);
        group.bench_with_input(id, num_players, |b: &mut Bencher, num_players: &usize| {
            let chips: Vec<i32> = (0..*num_players)
                .map(|_pn| rng.random_range(1..500))
                .collect();
            b.iter(|| simulate_icm_tournament(&chips, &payments))
        });
    }

    group.finish();
}

criterion_group!(benches, simulate_one_tournament);
criterion_main!(benches);
