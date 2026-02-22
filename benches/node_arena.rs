use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use rs_poker::arena::cfr::{CFRState, NodeData};

use rs_poker::arena::GameStateBuilder;

fn make_game_state() -> rs_poker::arena::GameState {
    GameStateBuilder::new()
        .num_players_with_stack(2, 100.0)
        .blinds(10.0, 5.0)
        .build()
        .unwrap()
}

/// Populate a CFRState with `count` nodes in a chain.
fn populate_cfr_state(count: usize) -> CFRState {
    let state = CFRState::new(make_game_state());
    let mut parent = 0;
    for _ in 1..count {
        parent = state.add(parent, 0, NodeData::Chance);
    }
    state
}

fn bench_single_thread_push(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_thread_push");

    for count in [1000, 10_000] {
        group.bench_with_input(
            criterion::BenchmarkId::new("node_arena", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let state = CFRState::new(make_game_state());
                    let mut parent = 0;
                    for _ in 1..count {
                        parent = state.add(parent, 0, NodeData::Chance);
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_single_thread_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_thread_read");

    let count = 10_000;
    let cfr_state = populate_cfr_state(count);

    group.bench_function("node_arena", |b| {
        b.iter(|| {
            for i in 0..count {
                std::hint::black_box(cfr_state.get_node_data(i));
            }
        });
    });

    group.finish();
}

/// 4 threads all reading node data concurrently.
fn bench_concurrent_reads(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_reads");

    let count = 10_000usize;
    let num_threads = 4;

    let cfr_state = Arc::new(populate_cfr_state(count));

    group.bench_function("node_arena", |b| {
        b.iter(|| {
            std::thread::scope(|s| {
                for _ in 0..num_threads {
                    let state = &cfr_state;
                    s.spawn(move || {
                        for i in 0..count {
                            std::hint::black_box(state.get_node_data(i));
                        }
                    });
                }
            });
        });
    });

    group.finish();
}

/// 1 writer + 3 readers — the critical CFR scenario.
fn bench_concurrent_mixed_read_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_mixed");

    let count = 5_000usize;
    let num_readers = 3;

    let cfr_state = Arc::new(populate_cfr_state(count));

    group.bench_function("node_arena", |b| {
        b.iter(|| {
            std::thread::scope(|s| {
                let state = &cfr_state;
                s.spawn(move || {
                    for i in 0..count {
                        state.update_node(i, |_data| {}).unwrap();
                    }
                });

                for _ in 0..num_readers {
                    let state = &cfr_state;
                    s.spawn(move || {
                        for i in 0..count {
                            std::hint::black_box(state.get_node_data(i));
                        }
                    });
                }
            });
        });
    });

    group.finish();
}

/// 4 writers updating DIFFERENT nodes concurrently.
fn bench_concurrent_writes_different_nodes(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_writes_different_nodes");

    let count = 10_000usize;
    let num_threads = 4;
    let per_thread = count / num_threads;

    let cfr_state = Arc::new(populate_cfr_state(count));

    group.bench_function("node_arena", |b| {
        b.iter(|| {
            std::thread::scope(|s| {
                for t in 0..num_threads {
                    let state = &cfr_state;
                    let start = t * per_thread;
                    let end = start + per_thread;
                    s.spawn(move || {
                        for i in start..end {
                            state.update_node(i, |_data| {}).unwrap();
                        }
                    });
                }
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_thread_push,
    bench_single_thread_read,
    bench_concurrent_reads,
    bench_concurrent_mixed_read_write,
    bench_concurrent_writes_different_nodes,
);
criterion_main!(benches);
