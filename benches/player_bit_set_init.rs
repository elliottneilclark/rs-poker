//! Microbenchmark for `PlayerBitSet::new` initialization strategies.
//!
//! This benchmark does NOT call the real `PlayerBitSet::new`. It exists to
//! drive the choice between two ways of producing the initial `u16` bitmask
//! with the low `players` bits set, without having to commit to either one in
//! the library first.
//!
//! The two candidates:
//!
//! * `init_branch` — the current implementation: special-case the full-width
//!   input (`players == 16`) because `1u16 << 16` overflows, otherwise do
//!   `(1u16 << players) - 1`.
//!
//! * `init_widen`  — do the shift in `u32` where `1 << 16` is representable,
//!   subtract one, and truncate back to `u16`. No branch, at the cost of a
//!   32-bit shift/sub pair plus a truncation.
//!
//! Both are benchmarked on:
//!   1. Fixed per-count inputs (0, 2, 6, 9, 16) — shows steady-state cost
//!      when the branch predictor has a single answer and input is hot.
//!   2. A mixed batch with an unpredictable distribution of player counts —
//!      shows what happens when the branch predictor cannot lock onto one
//!      side and the CPU has to pay for mispredictions.

#[macro_use]
extern crate criterion;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

/// Branch-on-full-width variant. Matches the current `PlayerBitSet::new`.
#[inline(always)]
fn init_branch(players: usize) -> u16 {
    if players >= 16 {
        u16::MAX
    } else {
        (1u16 << players) - 1
    }
}

/// Widen-and-truncate variant. Does the `(1 << x) - 1` math in `u32` so the
/// `players == 16` case is representable, then truncates to `u16`.
#[inline(always)]
fn init_widen(players: usize) -> u16 {
    ((1u32 << players) - 1) as u16
}

fn bench_fixed_counts(c: &mut Criterion) {
    let mut group = c.benchmark_group("player_bit_set_init/fixed");

    // Covers: empty set, heads-up, 6-max, 9-max full ring, 16-player edge.
    for players in [0usize, 2, 6, 9, 16] {
        group.bench_with_input(BenchmarkId::new("branch", players), &players, |b, &n| {
            b.iter(|| init_branch(black_box(n)))
        });
        group.bench_with_input(BenchmarkId::new("widen", players), &players, |b, &n| {
            b.iter(|| init_widen(black_box(n)))
        });
    }

    group.finish();
}

fn bench_mixed_batch(c: &mut Criterion) {
    // Realistic-ish mix: common poker table sizes plus a sprinkle of the
    // 16-player edge case. The goal is to defeat branch prediction on the
    // `if players >= 16` check so the branch variant cannot hide the cost of
    // the extra compare.
    let inputs: Vec<usize> = (0..4096)
        .map(|i| match i % 11 {
            0 | 1 => 2,
            2..=4 => 6,
            5..=7 => 9,
            8 => 10,
            9 => 16,
            _ => 4,
        })
        .collect();

    let mut group = c.benchmark_group("player_bit_set_init/mixed_batch");

    group.bench_function("branch", |b| {
        b.iter(|| {
            let mut acc: u16 = 0;
            for &n in black_box(&inputs) {
                acc ^= init_branch(n);
            }
            acc
        })
    });

    group.bench_function("widen", |b| {
        b.iter(|| {
            let mut acc: u16 = 0;
            for &n in black_box(&inputs) {
                acc ^= init_widen(n);
            }
            acc
        })
    });

    group.finish();
}

criterion_group!(benches, bench_fixed_counts, bench_mixed_batch);
criterion_main!(benches);
