//! Microbenchmarks for [`SevenCardAccum::rank`] — the per-runout ranker that
//! dominates CFR fast-forward board enumeration. Benched once per [`Rank`]
//! archetype (high card, one pair, two pair, trips, straight, flush, full
//! house, quads, straight flush) so future changes to the branch ladder show
//! up where they matter; the relative cost between archetypes also makes the
//! impact of any reorder visible.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use rs_poker::core::{Card, SevenCardAccum, Suit, Value};

/// Build a 7-card `SevenCardAccum` from a card slice.
fn accum(cards: &[Card]) -> SevenCardAccum {
    let mut acc = SevenCardAccum::new();
    for &c in cards {
        acc.add(c);
    }
    acc
}

/// One representative 7-card hand per [`Rank`] archetype. The mix is what
/// `combo_reward`'s inner loop sees over a board enumeration: most runouts
/// produce a high card or one pair; flush/quads/straight-flush show up only
/// occasionally but cost differently per call, so each gets its own signal.
fn archetype_accums() -> Vec<(&'static str, SevenCardAccum)> {
    use Suit::*;
    use Value::*;
    let c = Card::new;
    vec![
        (
            "high_card",
            accum(&[
                c(Ace, Spade),
                c(King, Heart),
                c(Nine, Diamond),
                c(Seven, Club),
                c(Five, Spade),
                c(Four, Heart),
                c(Two, Diamond),
            ]),
        ),
        (
            "one_pair",
            accum(&[
                c(Ace, Spade),
                c(Ace, Heart),
                c(King, Diamond),
                c(Nine, Club),
                c(Seven, Spade),
                c(Four, Heart),
                c(Two, Diamond),
            ]),
        ),
        (
            "two_pair",
            accum(&[
                c(Ace, Spade),
                c(Ace, Heart),
                c(King, Diamond),
                c(King, Club),
                c(Seven, Spade),
                c(Four, Heart),
                c(Two, Diamond),
            ]),
        ),
        (
            "three_of_a_kind",
            accum(&[
                c(Ace, Spade),
                c(Ace, Heart),
                c(Ace, Diamond),
                c(King, Club),
                c(Seven, Spade),
                c(Four, Heart),
                c(Two, Diamond),
            ]),
        ),
        (
            "straight",
            accum(&[
                c(Nine, Spade),
                c(Eight, Heart),
                c(Seven, Diamond),
                c(Six, Club),
                c(Five, Spade),
                c(King, Heart),
                c(Two, Diamond),
            ]),
        ),
        (
            "flush",
            accum(&[
                c(Ace, Spade),
                c(King, Spade),
                c(Nine, Spade),
                c(Seven, Spade),
                c(Four, Spade),
                c(Queen, Heart),
                c(Two, Diamond),
            ]),
        ),
        (
            "full_house",
            accum(&[
                c(Ace, Spade),
                c(Ace, Heart),
                c(Ace, Diamond),
                c(King, Club),
                c(King, Spade),
                c(Four, Heart),
                c(Two, Diamond),
            ]),
        ),
        (
            "four_of_a_kind",
            accum(&[
                c(Ace, Spade),
                c(Ace, Heart),
                c(Ace, Diamond),
                c(Ace, Club),
                c(King, Spade),
                c(Four, Heart),
                c(Two, Diamond),
            ]),
        ),
        (
            "straight_flush",
            accum(&[
                c(Nine, Spade),
                c(Eight, Spade),
                c(Seven, Spade),
                c(Six, Spade),
                c(Five, Spade),
                c(King, Heart),
                c(Two, Diamond),
            ]),
        ),
    ]
}

fn bench_seven_card_accum_rank(c: &mut Criterion) {
    let mut group = c.benchmark_group("seven_card_accum_rank");
    group.throughput(Throughput::Elements(1));
    for (name, acc) in archetype_accums() {
        group.bench_with_input(BenchmarkId::from_parameter(name), &acc, |b, acc| {
            b.iter(|| black_box(acc).rank())
        });
    }
    group.finish();
}

criterion_group!(benches, bench_seven_card_accum_rank);
criterion_main!(benches);
