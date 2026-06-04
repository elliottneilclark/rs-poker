//! Construction of [`CFRAgent`](super::CFRAgent) via [`CFRAgentBuilder`].
//!
//! The builder takes the required shared state (`CFRState`,
//! `TraversalSet`, action-generator config) and fills in optional pieces
//! with defaults: a `build_default_limiter()` in-flight limiter, a fresh
//! `Arc<AtomicBool>` stop flag, and `BudgetConfig::default().build()` as
//! the safety-net budget. Sub-agents are built with the same shared
//! handles so nested exploration composes correctly.

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;

use crate::arena::historian::SharedHistoryStorage;

use crate::arena::action::AgentAction;
use crate::arena::{HandDistributionEstimator, hand_estimator::KnownHandsEstimator};

use super::super::{
    ActionIndexMapper, ActionIndexMapperConfig, Budget, BudgetConfig, CFRState, InFlightLimiter,
    TraversalSet, action_generator::ActionGenerator, build_default_limiter,
};

use super::engine::CFRAgent;

pub struct CFRAgentBuilder<T>
where
    T: ActionGenerator,
{
    name: Option<Cow<'static, str>>,
    player_idx: Option<usize>,
    action_gen_config: Option<Arc<T::Config>>,
    traversal_set: Option<TraversalSet>,
    cfr_state: Option<CFRState>,
    mapper_config: Option<ActionIndexMapperConfig>,
    forced_action: Option<AgentAction>,
    depth: usize,
    allow_node_mutation: bool,
    limiter: Option<InFlightLimiter>,
    budget: Option<Arc<dyn Budget>>,
    stop: Option<Arc<AtomicBool>>,
    estimator: Option<Arc<dyn HandDistributionEstimator>>,
}

impl<T> Default for CFRAgentBuilder<T>
where
    T: ActionGenerator,
{
    fn default() -> Self {
        Self {
            name: None,
            player_idx: None,
            action_gen_config: None,
            traversal_set: None,
            cfr_state: None,
            mapper_config: None,
            forced_action: None,
            depth: 0,
            allow_node_mutation: true,
            limiter: None,
            budget: None,
            stop: None,
            estimator: None,
        }
    }
}

impl<T> CFRAgentBuilder<T>
where
    T: ActionGenerator,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn name(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn player_idx(mut self, player_idx: usize) -> Self {
        self.player_idx = Some(player_idx);
        self
    }

    pub fn action_gen_config(mut self, config: T::Config) -> Self {
        self.action_gen_config = Some(Arc::new(config));
        self
    }

    pub(super) fn action_gen_config_arc(mut self, config: Arc<T::Config>) -> Self {
        self.action_gen_config = Some(config);
        self
    }

    pub fn cfr_state(mut self, cfr_state: CFRState) -> Self {
        self.cfr_state = Some(cfr_state);
        self
    }

    pub fn traversal_set(mut self, traversal_set: TraversalSet) -> Self {
        self.traversal_set = Some(traversal_set);
        self
    }

    pub(super) fn depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    pub fn forced_action(mut self, action: AgentAction) -> Self {
        self.forced_action = Some(action);
        self
    }

    pub fn build(self) -> CFRAgent<T> {
        let name = self.name.expect("name is required");
        let player_idx = self.player_idx.expect("player_idx is required");
        let cfr_state = self.cfr_state.expect("cfr_state is required");
        let traversal_set = self.traversal_set.expect("traversal_set is required");
        let action_gen_config = self
            .action_gen_config
            .expect("action_gen_config is required");

        let traversal_state = traversal_set.get(player_idx);
        let action_generator = T::new(
            cfr_state.clone(),
            traversal_state.clone(),
            action_gen_config.clone(),
        );

        let mapper_config = self
            .mapper_config
            .unwrap_or_else(|| *cfr_state.mapper_config());
        let action_index_mapper = ActionIndexMapper::new(mapper_config);

        let limiter = self.limiter.unwrap_or_else(build_default_limiter);

        // Default to the small safe library exploration if no budget set.
        let budget = self
            .budget
            .unwrap_or_else(|| BudgetConfig::default().build());
        let stop = self
            .stop
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        let estimator = self
            .estimator
            .unwrap_or_else(|| Arc::new(KnownHandsEstimator) as Arc<dyn HandDistributionEstimator>);

        let log_storage: SharedHistoryStorage = Arc::new(Mutex::new(Vec::new()));

        CFRAgent {
            name,
            traversal_set,
            cfr_state,
            traversal_state,
            action_generator,
            action_gen_config,
            action_index_mapper,
            forced_action: self.forced_action,
            depth: self.depth,
            allow_node_mutation: self.allow_node_mutation,
            limiter,
            budget,
            stop,
            estimator,
            log_storage,
        }
    }

    pub(super) fn mapper_config(mut self, config: ActionIndexMapperConfig) -> Self {
        self.mapper_config = Some(config);
        self
    }

    pub fn allow_node_mutation(mut self, allow: bool) -> Self {
        self.allow_node_mutation = allow;
        self
    }

    pub fn limiter(mut self, limiter: InFlightLimiter) -> Self {
        self.limiter = Some(limiter);
        self
    }

    /// Set the unified [`Budget`] for this agent. Threaded into recursive
    /// sub-agents via the shared `Arc`. Defaults to
    /// `BudgetConfig::default().build()` (the small safe library
    /// exploration) when unset.
    pub fn budget(mut self, budget: Arc<dyn Budget>) -> Self {
        self.budget = Some(budget);
        self
    }

    /// Set the lock-free stop flag from an already-shared `Arc<AtomicBool>`.
    /// Used by recursive sub-agent construction so every level shares one
    /// flag.
    pub(super) fn stop_flag(mut self, stop: Arc<AtomicBool>) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Set the opponent hand-distribution estimator. Defaults to
    /// [`KnownHandsEstimator`], which reproduces the pre-estimator behavior.
    pub fn estimator(mut self, estimator: Arc<dyn HandDistributionEstimator>) -> Self {
        self.estimator = Some(estimator);
        self
    }
}
