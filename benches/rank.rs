#[macro_use]
extern crate criterion;
extern crate rand;
extern crate rs_poker;

use criterion::Criterion;
use rs_poker::core::{Card, Deck, FlatDeck, FlatHand, Rank, Rankable, SevenCardAccum, Suit, Value};

fn rank_one(c: &mut Criterion) {
    let d: FlatDeck = Deck::default().into();
    let hand = FlatHand::new_with_cards(d.sample(5));
    c.bench_function("Rank one 5 card hand", move |b| b.iter(|| hand.rank()));
}

fn rank_best_seven(c: &mut Criterion) {
    let d: FlatDeck = Deck::default().into();
    let hand = FlatHand::new_with_cards(d.sample(7));
    c.bench_function("Rank best 5card hand from 7", move |b| {
        b.iter(|| hand.rank())
    });
}

// Two players, a flop, enumerate every turn+river runout via the
// tally-once / copy-and-extend pattern the arena uses.
fn rank_board_enumeration(c: &mut Criterion) {
    let p1 = [
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Spade),
    ];
    let p2 = [
        Card::new(Value::Queen, Suit::Heart),
        Card::new(Value::Queen, Suit::Diamond),
    ];
    let flop = [
        Card::new(Value::Jack, Suit::Spade),
        Card::new(Value::Seven, Suit::Club),
        Card::new(Value::Two, Suit::Diamond),
    ];
    let used: u64 = p1
        .iter()
        .chain(p2.iter())
        .chain(flop.iter())
        .fold(0u64, |a, c| a | (1u64 << u8::from(*c)));
    let deck: Vec<Card> = (0u8..52)
        .map(Card::from)
        .filter(|c| used & (1u64 << u8::from(*c)) == 0)
        .collect();

    let mut base1 = SevenCardAccum::new();
    let mut base2 = SevenCardAccum::new();
    for &c in p1.iter().chain(flop.iter()) {
        base1.add(c);
    }
    for &c in p2.iter().chain(flop.iter()) {
        base2.add(c);
    }

    c.bench_function("Rank board enumeration turn+river", move |b| {
        b.iter(|| {
            let mut wins = 0u32;
            for i in 0..deck.len() {
                for j in (i + 1)..deck.len() {
                    let mut a1 = base1;
                    a1.add(deck[i]);
                    a1.add(deck[j]);
                    let mut a2 = base2;
                    a2.add(deck[i]);
                    a2.add(deck[j]);
                    if a1.rank() > a2.rank() {
                        wins += 1;
                    }
                }
            }
            wins
        })
    });
}

fn rank_random_seven_throughput(c: &mut Criterion) {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
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
    c.bench_function("Rank 1024 random 7card hands", move |b| {
        b.iter(|| {
            let mut acc = Rank::HIGH_CARD_MIN;
            for h in &hands {
                let r = h[..].rank();
                if r > acc {
                    acc = r;
                }
            }
            acc
        })
    });
}

criterion_group!(
    benches,
    rank_one,
    rank_best_seven,
    rank_board_enumeration,
    rank_random_seven_throughput
);
criterion_main!(benches);
