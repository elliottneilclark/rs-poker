//! Concurrency primitives for the async CFR exploration engine.
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Cap on concurrently in-flight `compute_reward` tasks.
///
/// A single limiter instance is passed down through every level of one agent's
/// exploration, so that agent's recursive spawning composes to one bound
/// rather than multiplying with depth. Spawning is gated by `try_acquire`:
/// when no permit is free, a branch runs inline instead of spawning, which
/// keeps recursion deadlock-free at any depth.
///
/// The bound is per limiter instance, not process-wide. `build_default_limiter()`
/// allocates a fresh semaphore, so distinct agents — and the competition
/// runner — each get their own unless a caller deliberately shares one
/// `Arc<Semaphore>` across them (see `CFRAgentBuilder::limiter`).
pub type InFlightLimiter = Arc<Semaphore>;

/// Default in-flight permit count: `8 × available parallelism` (minimum 8).
///
/// Sized above the core count on purpose: the runtime's worker threads already
/// bound CPU parallelism, so this cap mainly governs memory when CPU-bound and,
/// once ML inference lands, how many subtree walkers can park at inference
/// awaits to feed batches. Exposed as a separate seam so a future
/// `LimiterConfig` can override it without touching call sites.
pub fn default_limiter_permits() -> usize {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    8 * cpus
}

/// Build a fresh limiter sized at [`default_limiter_permits`].
///
/// Builds (does not share) a new `Arc<Semaphore>`; distinct agents each get
/// their own bound unless a caller deliberately shares one (see
/// `CFRAgentBuilder::limiter`).
pub fn build_default_limiter() -> InFlightLimiter {
    Arc::new(Semaphore::new(default_limiter_permits()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_default_limiter_has_positive_permits() {
        let limiter = build_default_limiter();
        assert!(limiter.available_permits() >= 8);
    }

    #[test]
    fn default_limiter_permits_is_at_least_eight() {
        assert!(default_limiter_permits() >= 8);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn try_acquire_falls_back_when_exhausted() {
        let limiter = Arc::new(Semaphore::new(1));
        let _held = limiter.clone().try_acquire_owned().unwrap();
        // No permits left: try_acquire must fail (the engine then runs inline).
        assert!(limiter.clone().try_acquire_owned().is_err());
    }
}
