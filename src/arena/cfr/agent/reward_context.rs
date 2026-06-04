//! The per-exploration reward context.
//!
//! [`ComputeRewardContext`] is the cheap, `Send + 'static` snapshot of
//! everything `compute_reward` needs that stays constant across the
//! iterations and actions of a single `explore_all_actions` call. Every
//! field is an owned `Arc`-style handle or a `Copy` value. Randomness is
//! drawn from a thread-local RNG at each use site.
//!
//! The CFR cancellation token has been removed entirely — the lock-free
//! `stop: Arc<AtomicBool>` is the single cross-task stop signal.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use super::super::{
    ActionIndexMapper, Budget, CFRState, InFlightLimiter, TraversalSet, TraversalState,
    action_generator::ActionGenerator,
};
use crate::arena::HandDistributionEstimator;

pub(super) struct ComputeRewardContext<T: ActionGenerator> {
    pub(super) traversal_set: TraversalSet,
    pub(super) traversal_state: TraversalState,
    pub(super) cfr_state: CFRState,
    pub(super) action_gen_config: Arc<T::Config>,
    pub(super) action_index_mapper: ActionIndexMapper,
    /// Shared in-flight limiter for bounded concurrent exploration.
    pub(super) limiter: InFlightLimiter,
    /// The unified budget threaded down every recursive level.
    pub(super) budget: Arc<dyn Budget>,
    /// Lock-free stop flag shared by every recursive level. The per-act
    /// deadline timer (spawned by the engine when it sees
    /// `NextStep::StartTimer`) flips this; wave-boundary checks read it
    /// with a relaxed atomic load.
    pub(super) stop: Arc<AtomicBool>,
    pub(super) depth: usize,
    pub(super) fast_forward: bool,
    pub(super) allow_node_mutation: bool,
    pub(super) estimator: Arc<dyn HandDistributionEstimator>,
}

// Manual `Clone` so the bound is only on the `Arc`-backed handles, not on `T`.
impl<T: ActionGenerator> Clone for ComputeRewardContext<T> {
    fn clone(&self) -> Self {
        Self {
            traversal_set: self.traversal_set.clone(),
            traversal_state: self.traversal_state.clone(),
            cfr_state: self.cfr_state.clone(),
            action_gen_config: self.action_gen_config.clone(),
            action_index_mapper: self.action_index_mapper.clone(),
            limiter: self.limiter.clone(),
            budget: self.budget.clone(),
            stop: self.stop.clone(),
            depth: self.depth,
            fast_forward: self.fast_forward,
            allow_node_mutation: self.allow_node_mutation,
            estimator: self.estimator.clone(),
        }
    }
}
