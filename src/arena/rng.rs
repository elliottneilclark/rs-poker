//! Small RNG helpers shared across the arena.

use rand::SeedableRng;
use rand::rngs::StdRng;

/// Build a `StdRng` from an optional seed.
///
/// When `seed` is `Some`, the RNG is deterministically seeded so runs are
/// reproducible. When `seed` is `None`, the RNG is seeded from the thread RNG.
///
/// A concrete `StdRng` (rather than a `&mut dyn Rng`) is returned so callers
/// can own it across `Send` async boundaries, e.g. when driving a runner from
/// a `tokio::spawn`ed task.
pub fn seeded_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_rng(&mut rand::rng()),
    }
}
