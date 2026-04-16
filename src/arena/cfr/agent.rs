use std::borrow::Cow;
use std::marker::PhantomData;
use std::sync::Arc;

use little_sorry::{PcfrPlusRegretMatcher, RegretMinimizer};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use tracing::event;

use crate::arena::{
    Agent, GameState, HoldemSimulationBuilder, action::AgentAction, game_state::Round,
};
use crate::core::{Deck, PlayerBitSet, Rankable};

use super::{
    ActionIndexMapper, ActionIndexMapperConfig, ActionPicker, CFRState, CfrDepthConfig,
    NUM_ACTION_INDICES, NodeData, TraversalSet, TraversalState,
    action_bit_set::ActionBitSet,
    action_generator::ActionGenerator,
    action_validator::{ValidatorMode, validate_actions},
    get_regret_matcher_from_node,
};

/// Shared context for `compute_reward` calls, grouping the parameters that
/// are constant across all iterations and actions within `explore_all_actions`.
///
/// This struct allows `compute_reward` to accept a single reference instead
/// of many individual parameters.
struct ComputeRewardContext<'a, T: ActionGenerator> {
    traversal_set: &'a TraversalSet,
    traversal_state: &'a TraversalState,
    cfr_state: &'a CFRState,
    depth_config: &'a Arc<CfrDepthConfig>,
    action_gen_config: &'a Arc<T::Config>,
    action_index_mapper: &'a ActionIndexMapper,
    depth: usize,
    fast_forward: bool,
    allow_node_mutation: bool,
}

/// A CFR (Counterfactual Regret Minimization) agent for poker.
///
/// This agent uses CFR to compute optimal strategies by exploring the game tree
/// and learning from regret. It maintains state across simulations via shared
/// CFR states.
///
/// # Type Parameters
/// * `T` - The action generator type (implements `ActionGenerator`)
/// * `R` - The random number generator type (defaults to `StdRng`)
pub struct CFRAgent<T, R = StdRng>
where
    T: ActionGenerator,
    R: Rng + SeedableRng,
{
    name: Cow<'static, str>,
    traversal_set: TraversalSet,
    traversal_state: TraversalState,
    cfr_state: CFRState,
    action_generator: T,
    /// Shared recursion schedule. `depth_config.hands_for_depth(self.depth)`
    /// is the loop count for `explore_all_actions`; `0` means fast-forward.
    depth_config: Arc<CfrDepthConfig>,
    /// Shared config reference for cheap cloning in reward() sub-agent construction.
    action_gen_config: Arc<T::Config>,
    /// The action index mapper for consistent action-to-index mapping.
    action_index_mapper: ActionIndexMapper,

    // This will be the next action to play
    // This allows us to start exploration
    // from a specific action.
    forced_action: Option<AgentAction>,

    /// Recursion depth in CFR tree (0 = root agent, 1+ = sub-agent).
    depth: usize,

    /// Whether to allow mutating node types when a mismatch is found.
    /// When true, if the agent expects a Player node but finds a different type
    /// (e.g., Terminal), it will convert the node to the expected type.
    /// This is necessary for action generators that produce bet amounts mapping
    /// to the same index but resulting in different outcomes (all-in vs not).
    /// Default is true for ConfigurableActionGenerator, false for BasicCFRActionGenerator.
    allow_node_mutation: bool,

    /// Optional thread pool for parallel action exploration.
    /// When `Some`, `explore_all_actions` parallelizes reward computation using rayon.
    /// When `None`, the sequential path is used (zero overhead).
    thread_pool: Option<Arc<rayon::ThreadPool>>,

    /// Random number generator for action selection.
    rng: R,
}

/// Builder for creating CFR agents with a fluent API.
///
/// All CFR agents in a simulation should share the same `CFRState` and
/// `TraversalSet` to enable shared learning and coordinated tree traversal.
///
/// # Type Parameters
/// * `T` - The action generator type (implements `ActionGenerator`)
/// * `R` - The random number generator type (defaults to `StdRng`)
pub struct CFRAgentBuilder<T, R = StdRng>
where
    T: ActionGenerator,
    R: Rng + SeedableRng,
{
    name: Option<Cow<'static, str>>,
    player_idx: Option<usize>,
    depth_config: Option<Arc<CfrDepthConfig>>,
    action_gen_config: Option<Arc<T::Config>>,
    traversal_set: Option<TraversalSet>,
    /// Single shared CFR state for the entire game tree.
    cfr_state: Option<CFRState>,
    /// Pre-fetched mapper config to avoid lock acquisition in build().
    mapper_config: Option<ActionIndexMapperConfig>,
    forced_action: Option<AgentAction>,
    /// Recursion depth in CFR tree (0 = root, 1+ = sub-agent).
    depth: usize,
    /// Whether to allow mutating node types when a mismatch is found.
    allow_node_mutation: bool,
    /// Optional thread pool for parallel action exploration.
    thread_pool: Option<Arc<rayon::ThreadPool>>,
    /// Optional RNG instance. If None, creates one from system entropy.
    rng: Option<R>,
    /// Phantom data to satisfy the compiler.
    _marker: PhantomData<R>,
}

impl<T, R> Default for CFRAgentBuilder<T, R>
where
    T: ActionGenerator,
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        Self {
            name: None,
            player_idx: None,
            depth_config: None,
            action_gen_config: None,
            traversal_set: None,
            cfr_state: None,
            mapper_config: None,
            forced_action: None,
            depth: 0,
            allow_node_mutation: true,
            thread_pool: None,
            rng: None,
            _marker: PhantomData,
        }
    }
}

impl<T, R> CFRAgentBuilder<T, R>
where
    T: ActionGenerator,
    R: Rng + SeedableRng,
{
    /// Create a new CFRAgentBuilder with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the name for the agent.
    pub fn name(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the player index for the agent.
    pub fn player_idx(mut self, player_idx: usize) -> Self {
        self.player_idx = Some(player_idx);
        self
    }

    /// Set the depth-based recursion schedule.
    pub fn depth_config(mut self, config: CfrDepthConfig) -> Self {
        self.depth_config = Some(Arc::new(config));
        self
    }

    /// Set the depth config from a shared Arc.
    /// Used internally to avoid re-wrapping in Arc during sub-agent construction.
    fn depth_config_arc(mut self, config: Arc<CfrDepthConfig>) -> Self {
        self.depth_config = Some(config);
        self
    }

    /// Set the action generator configuration.
    pub fn action_gen_config(mut self, config: T::Config) -> Self {
        self.action_gen_config = Some(Arc::new(config));
        self
    }

    /// Set the action generator configuration from a shared Arc.
    /// Used internally to avoid re-wrapping in Arc during sub-agent construction.
    fn action_gen_config_arc(mut self, config: Arc<T::Config>) -> Self {
        self.action_gen_config = Some(config);
        self
    }

    /// Set the shared CFR state for this agent.
    ///
    /// All CFR agents in a simulation should share the same CFRState.
    /// CFRState is cheap to clone (just Arc bumps).
    pub fn cfr_state(mut self, cfr_state: CFRState) -> Self {
        self.cfr_state = Some(cfr_state);
        self
    }

    /// Set the traversal set for this agent.
    ///
    /// All CFR agents in a simulation should share the same TraversalSet.
    /// The traversal set tracks each player's position in the CFR tree.
    pub fn traversal_set(mut self, traversal_set: TraversalSet) -> Self {
        self.traversal_set = Some(traversal_set);
        self
    }

    /// Set the recursion depth for this agent.
    /// Used internally - depth 0 is the root agent, depth 1+ are sub-agents.
    fn depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    /// Set a forced action for the first act call.
    ///
    /// When set, the agent will play this action on its first act() call
    /// instead of computing one. Used for exploration in sub-simulations.
    pub fn forced_action(mut self, action: AgentAction) -> Self {
        self.forced_action = Some(action);
        self
    }

    /// Set the random number generator instance.
    ///
    /// When set, the agent will use this RNG for all random decisions.
    /// This allows passing in a pre-seeded or custom RNG.
    /// If not set, the agent creates an RNG from system entropy.
    pub fn rng(mut self, rng: R) -> Self {
        self.rng = Some(rng);
        self
    }

    /// Build the CFRAgent.
    ///
    /// # Panics
    ///
    /// Panics if any required fields are not set:
    /// - name
    /// - player_idx
    /// - cfr_state
    /// - traversal_set
    /// - depth_config
    /// - action_gen_config
    pub fn build(self) -> CFRAgent<T, R> {
        let name = self.name.expect("name is required");
        let _player_idx = self.player_idx.expect("player_idx is required");
        let cfr_state = self.cfr_state.expect("cfr_state is required");
        let traversal_set = self.traversal_set.expect("traversal_set is required");
        let depth_config = self.depth_config.expect("depth_config is required");
        let action_gen_config = self
            .action_gen_config
            .expect("action_gen_config is required");

        let traversal_state = traversal_set.get(_player_idx);
        let action_generator = T::new(
            cfr_state.clone(),
            traversal_state.clone(),
            action_gen_config.clone(),
        );

        // Use pre-fetched mapper config if available, otherwise fetch from CFR state.
        let mapper_config = self
            .mapper_config
            .unwrap_or_else(|| *cfr_state.mapper_config());
        let action_index_mapper = ActionIndexMapper::new(mapper_config);

        // Create the RNG - use provided RNG or generate from system entropy
        let rng = self.rng.unwrap_or_else(|| R::from_rng(&mut rand::rng()));

        CFRAgent {
            name,
            traversal_set,
            cfr_state,
            traversal_state,
            action_generator,
            depth_config,
            action_gen_config,
            action_index_mapper,
            forced_action: self.forced_action,
            depth: self.depth,
            allow_node_mutation: self.allow_node_mutation,
            thread_pool: self.thread_pool,
            rng,
        }
    }

    /// Set a pre-fetched mapper config for this agent.
    /// Used internally to avoid repeated lookups when constructing many sub-agents.
    fn mapper_config(mut self, config: ActionIndexMapperConfig) -> Self {
        self.mapper_config = Some(config);
        self
    }

    /// Set whether to allow mutating node types when a mismatch is found.
    ///
    /// When true (default), if the agent expects a Player node but finds a different
    /// type (e.g., Terminal), it will convert the node to the expected type.
    /// This is necessary for action generators that produce bet amounts mapping
    /// to the same index but resulting in different outcomes (all-in vs not).
    ///
    /// Set to false when testing with BasicCFRActionGenerator to catch bugs.
    pub fn allow_node_mutation(mut self, allow: bool) -> Self {
        self.allow_node_mutation = allow;
        self
    }

    /// Set a thread pool for parallel action exploration.
    ///
    /// When set, `explore_all_actions` parallelizes reward computation across
    /// iterations and actions using rayon. Only the root agent parallelizes;
    /// sub-agents run sequentially to preserve per-iteration regret update ordering.
    ///
    /// **Convergence note:** The parallel path runs all iterations x actions
    /// simultaneously against the same initial regret state, then applies regret
    /// updates sequentially afterward. This is a batch-style update schedule,
    /// unlike the sequential path where each iteration sees the regret state
    /// updated by all prior iterations. The parallel path trades some convergence
    /// efficiency for wall-clock speedup via parallelism.
    pub fn thread_pool(mut self, pool: Arc<rayon::ThreadPool>) -> Self {
        self.thread_pool = Some(pool);
        self
    }
}

impl<T, R> CFRAgent<T, R>
where
    T: ActionGenerator + 'static,
    R: Rng + SeedableRng + Send + 'static,
{
    /// Returns a reference to this agent's CFR state.
    ///
    /// The CFR state contains the game tree with regret information learned
    /// during simulations. This can be used for visualization or analysis.
    pub fn cfr_state(&self) -> &CFRState {
        &self.cfr_state
    }

    /// Returns a clone of this agent's traversal set.
    pub fn traversal_set(&self) -> &TraversalSet {
        &self.traversal_set
    }

    /// Returns whether this agent allows node mutation.
    pub fn allow_node_mutation(&self) -> bool {
        self.allow_node_mutation
    }

    /// Compute the expected reward for taking a specific action.
    ///
    /// This is an associated function (no `&self`) so it can be called from
    /// parallel contexts where `self` cannot be borrowed. Shared state is
    /// passed via `ComputeRewardContext`.
    ///
    /// There are two reward strategies, both defined in this file so a reader
    /// can see them side by side:
    ///
    /// 1. [`Self::compute_reward_recursive`] — the full-strength path. It
    ///    spawns a `HoldemSimulation` whose agents are fresh CFR sub-agents,
    ///    allowing the game tree to keep branching and mutual best-response
    ///    play to develop. Accurate but exponentially expensive with depth.
    ///
    /// 2. [`Self::compute_reward_fast_forward`] — the cheap path used when
    ///    `depth_config.hands_for_depth(self.depth)` is `0` (the schedule has
    ///    no entry at this depth). It applies the candidate action on a
    ///    cloned state and assumes every *subsequent* action (by any player,
    ///    in any round) is a check or call. Remaining community cards are
    ///    dealt and the pot is distributed with simple one-pot showdown
    ///    logic. This throws away the mutual-best-response signal deep in the
    ///    tree, but returns a realistic showdown reward and has bounded cost.
    fn compute_reward(
        game_state: &GameState,
        action: &AgentAction,
        rng: &mut R,
        ctx: &ComputeRewardContext<'_, T>,
    ) -> f32 {
        if ctx.fast_forward {
            let player_idx = ctx.traversal_state.player_idx() as usize;
            Self::compute_reward_fast_forward(game_state, action, player_idx, rng)
        } else {
            Self::compute_reward_recursive(game_state, action, rng, ctx)
        }
    }

    /// Full recursive reward: spawn a sub-simulation driven by new CFR agents.
    ///
    /// This is the expensive path — each call clones the game state, builds
    /// one CFR sub-agent per seat (the acting seat is forced to play `action`),
    /// and runs a complete `HoldemSimulation`. Those sub-agents may in turn
    /// call `compute_reward` themselves, which is where the exponential
    /// branching lives. `compute_reward` switches to the fast-forward sibling
    /// once the depth schedule runs out to cap that blowup.
    fn compute_reward_recursive(
        game_state: &GameState,
        action: &AgentAction,
        rng: &mut R,
        ctx: &ComputeRewardContext<'_, T>,
    ) -> f32 {
        let num_agents = game_state.num_players;

        // Get all traversal state fields in a single lock acquisition
        let (_before_node_idx, _before_child_idx, player_idx) = ctx.traversal_state.get_all();

        event!(
            tracing::Level::TRACE,
            num_agents,
            ?action,
            player_idx = player_idx,
            "Computing reward via sub-simulation"
        );

        // Fork the traversal set for sub-simulation isolation.
        // The forked set starts at the same positions but is independent —
        // mutations in the sub-simulation won't affect the parent.
        let forked_traversal_set = ctx.traversal_set.fork();

        let sub_depth = ctx.depth + 1;

        // Clone Arc configs (cheap atomic increment) instead of deep-cloning
        let depth_config = ctx.depth_config.clone();
        let action_config = ctx.action_gen_config.clone();
        let cached_mapper_config = *ctx.action_index_mapper.config();

        // All sub-agents share the same CFR state (single shared tree).
        let shared_cfr_state = ctx.cfr_state.clone();

        // Build directly into Vec<Box<dyn Agent>> to avoid an intermediate Vec
        let mut agents: Vec<Box<dyn Agent>> = Vec::with_capacity(num_agents);
        for i in 0..num_agents {
            let sub_rng = R::from_rng(rng);
            let mut builder = CFRAgentBuilder::<T, R>::new()
                .name("CFRAgent-sub")
                .player_idx(i)
                .cfr_state(shared_cfr_state.clone())
                .mapper_config(cached_mapper_config)
                .depth_config_arc(depth_config.clone())
                .action_gen_config_arc(action_config.clone())
                .traversal_set(forked_traversal_set.clone())
                .depth(sub_depth)
                .rng(sub_rng);

            // Note: we intentionally do NOT propagate the thread pool to sub-agents.
            // Only the root agent parallelizes explore_all_actions. Sub-agents run
            // sequentially to avoid lock contention on the shared CFR state.

            if i == player_idx as usize {
                builder = builder.forced_action((*action).clone());
            }

            agents.push(Box::new(builder.build()) as Box<dyn Agent>);
        }

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state.clone())
            .agents(agents)
            .cfr_context(
                shared_cfr_state,
                forked_traversal_set,
                ctx.allow_node_mutation,
            )
            .build_with_rng(rng)
            .unwrap();

        sim.run(rng);

        // Verify parent traversal is unaffected by the fork
        #[cfg(debug_assertions)]
        {
            let (after_node_idx, after_child_idx) = ctx.traversal_state.get_position();
            assert_eq!(
                _before_node_idx, after_node_idx,
                "Node index should be the same after exploration"
            );
            assert_eq!(
                _before_child_idx, after_child_idx,
                "Child index should be the same after exploration"
            );
        }

        sim.game_state.player_reward(player_idx as usize)
    }

    /// Fast-forward reward: apply `action` on a clone, then play the rest of
    /// the hand out assuming every further action is a check or call.
    ///
    /// This is the bounded-cost sibling to [`Self::compute_reward_recursive`].
    /// No CFR sub-agents are spawned and no `HoldemSimulation` is built — we
    /// just mutate a cloned `GameState` directly via the `fast_forward_*`
    /// helpers below, deal the remaining community cards from a reconstructed
    /// deck, and distribute a single pot at showdown.
    ///
    /// Simplifications (per design):
    /// - One pot only; side pots are collapsed into the main pot.
    /// - A player who cannot cover the current bet is treated as all-in for
    ///   what they have and remains eligible for the single pot.
    /// - Ties split the pot evenly.
    fn compute_reward_fast_forward(
        game_state: &GameState,
        action: &AgentAction,
        player_idx: usize,
        rng: &mut R,
    ) -> f32 {
        let mut gs = game_state.clone();
        fast_forward_apply_action(&mut gs, action);

        // Check if at most one player can contest the pot after the action.
        let contenders = gs.player_active.count() + gs.player_all_in.count();
        if contenders <= 1 {
            fast_forward_run_to_showdown(&mut gs, rng);
            fast_forward_distribute_pot(&mut gs);
            return gs.player_reward(player_idx);
        }

        // Try exhaustive board enumeration for improved reward accuracy.
        // Advance through any remaining betting (everyone calls) to reach
        // a deal round or showdown, then enumerate remaining community cards
        // instead of sampling.
        fast_forward_advance_betting(&mut gs);

        let cards_needed = match gs.round {
            Round::Showdown | Round::Complete => 0,
            Round::DealFlop => 3,
            Round::DealTurn => 2, // turn + river
            Round::DealRiver => 1,
            _ => {
                // Unexpected round after advancing betting. Fall back.
                fast_forward_run_to_showdown(&mut gs, rng);
                fast_forward_distribute_pot(&mut gs);
                return gs.player_reward(player_idx);
            }
        };

        // Enumerate all remaining board completions for zero-variance rewards.
        // Eliminating sampling noise from board completions produces
        // deterministic reward signals, improving CFR convergence.
        //
        // 0 cards: deterministic showdown (1 eval).
        // 1 card: ~46 evaluations (river only).
        // 2 cards: ~C(46,2) ≈ 1035 evaluations (turn + river).
        // 3 cards: sample FLOP_SAMPLES random flops, then enumerate all
        //   turn+river combinations for each (~1035 evals per flop).
        //   This gives low-variance rewards without the cost of full
        //   C(47,3) ≈ 16K enumeration.
        if cards_needed <= 2 {
            fast_forward_enumerate_showdowns(&gs, player_idx, cards_needed)
        } else {
            fast_forward_sample_flop_enumerate_runout(&gs, player_idx, rng)
        }
    }

    fn target_node_idx(&self) -> Option<usize> {
        let (from_node_idx, from_child_idx) = self.traversal_state.get_position();
        self.cfr_state.get_child(from_node_idx, from_child_idx)
    }

    /// Ensure that the target node is created and that it is a player node.
    ///
    /// Uses `CFRState::ensure_child` which handles the case where different bet
    /// amounts map to the same index but lead to different outcomes. If a node
    /// exists with a different type and `allow_node_mutation` is true, it will
    /// be updated to a Player node.
    fn ensure_target_node(&self) -> usize {
        // Get all traversal state fields in a single lock acquisition
        let (node_idx, chosen_child_idx, player_idx) = self.traversal_state.get_all();

        let expected_data = super::NodeData::Player(super::PlayerData {
            regret_matcher: None,
            player_idx,
        });

        self.cfr_state.ensure_child(
            node_idx,
            chosen_child_idx,
            expected_data,
            self.allow_node_mutation,
        )
    }

    fn ensure_regret_matcher(&mut self) {
        let target_node_idx = self.ensure_target_node();

        self.cfr_state
            .update_node(target_node_idx, |data| {
                if let NodeData::Player(player_data) = data
                    && player_data.regret_matcher.is_none()
                {
                    // Use the fixed constant for number of action indices (52)
                    let regret_matcher = Box::new(PcfrPlusRegretMatcher::new(NUM_ACTION_INDICES));
                    player_data.regret_matcher = Some(regret_matcher);
                }
            })
            .unwrap();
    }

    /// Update regret at a node, handling the case where a concurrent thread
    /// may have changed the node type (e.g., from Player to Chance) via
    /// `allow_node_mutation`. When this happens, we re-ensure the node as
    /// Player and create a fresh regret matcher before updating.
    fn update_regret_at_node(&self, target_node_idx: usize, rewards: &[f32]) {
        self.cfr_state
            .update_node(target_node_idx, |data| {
                if let NodeData::Player(player_data) = data {
                    if let Some(regret_matcher) = player_data.regret_matcher.as_mut() {
                        regret_matcher.update_regret(rewards);
                    }
                    // If regret_matcher is None, skip this update — it will
                    // be created on the next ensure_regret_matcher call.
                } else {
                    // A concurrent sub-simulation's historian overwrote this
                    // node's type (e.g., to Chance or Terminal) via
                    // allow_node_mutation. Restore it to Player with a fresh
                    // regret matcher so exploration can continue.
                    event!(
                        tracing::Level::DEBUG,
                        target_node_idx,
                        found_type = %data,
                        "Concurrent node type change detected — restoring Player"
                    );
                    let mut regret_matcher =
                        Box::new(PcfrPlusRegretMatcher::new(NUM_ACTION_INDICES));
                    regret_matcher.update_regret(rewards);
                    *data = NodeData::Player(super::PlayerData {
                        regret_matcher: Some(regret_matcher),
                        player_idx: self.traversal_state.player_idx(),
                    });
                }
            })
            .unwrap();
    }

    pub fn explore_all_actions(&mut self, game_state: &GameState) {
        // Read the schedule entry for this depth. `0` (absent from the
        // schedule) means: this agent does a single pass of exploration but
        // computes rewards via the fast-forward path rather than recursing.
        let scheduled_hands = self.depth_config.hands_for_depth(self.depth);
        let fast_forward = scheduled_hands == 0;
        let num_iterations = scheduled_hands.max(1);

        let mode = if fast_forward {
            ValidatorMode::Limited
        } else {
            ValidatorMode::Standard
        };

        // Generate actions and apply validation filters
        let raw_actions = self.action_generator.gen_possible_actions(game_state);
        let validated_actions = validate_actions(raw_actions, game_state, mode);

        // Filter actions to ensure each maps to a unique index.
        // Different bet amounts can map to the same index due to the logarithmic
        // mapping (only 49 slots for raises). We keep the first action for each index.
        // Using ActionBitSet for O(1) operations with no heap allocation.
        // Pre-compute action indices once to avoid repeated action_to_idx calls.
        let mut seen_indices = ActionBitSet::new();
        let indexed_actions: Vec<(AgentAction, usize)> = validated_actions
            .into_iter()
            .filter_map(|a| {
                let idx = self.action_index_mapper.action_to_idx(&a, game_state);
                if seen_indices.insert(idx) {
                    Some((a, idx))
                } else {
                    None
                }
            })
            .collect();

        // If no valid actions remain after filtering, skip exploration entirely.
        // This can happen at deep recursion depths in Limited mode when all
        // generated actions get filtered out by the validator chain.
        if indexed_actions.is_empty() {
            return;
        }

        // Penalty for invalid actions - using player's starting stack since
        // losing your whole stack is the worst outcome.
        let invalid_action_penalty =
            -(game_state.starting_stacks[self.traversal_state.player_idx() as usize]);

        let target_node_idx = self.target_node_idx().unwrap();

        // Pre-allocate rewards vec and reuse across iterations to avoid
        // repeated Vec allocations.
        let mut rewards: Vec<f32> = vec![invalid_action_penalty; NUM_ACTION_INDICES];

        // ── Regret-Based Pruning (Brown & Sandholm, NeurIPS 2015) ────
        //
        // After a warmup period, read the regret matcher's current strategy
        // to identify actions with zero strategy weight. These are actions
        // whose cumulative regret has been driven to 0 by PCFR+ clamping
        // and whose predicted future regret is also non-positive. Skipping
        // their reward computation (which involves expensive sub-simulations)
        // saves significant computation. Pruned actions keep the penalty
        // reward, which naturally maintains their zero cumulative regret.
        //
        // Every REPROBE_INTERVAL-th iteration, all actions are explored to
        // detect actions that may have become relevant again.
        //
        // Pruning is only applied when:
        // - The regret matcher has enough history (>= PRUNE_WARMUP updates)
        // - There are more than 2 actions (with only 2, pruning saves little)
        // - It's not a reprobe iteration
        const PRUNE_WARMUP: usize = 3;
        const REPROBE_INTERVAL: usize = 4;

        let (initial_active, initial_updates) = self.cfr_state.get_pruning_info(target_node_idx);
        let can_prune = indexed_actions.len() > 2 && initial_updates >= PRUNE_WARMUP;

        // Build context struct from self fields for Sync closure capture.
        // CFRAgent is not Sync (StdRng is not Sync), but all context fields are.
        let ctx = ComputeRewardContext::<T> {
            traversal_set: &self.traversal_set,
            traversal_state: &self.traversal_state,
            cfr_state: &self.cfr_state,
            depth_config: &self.depth_config,
            action_gen_config: &self.action_gen_config,
            action_index_mapper: &self.action_index_mapper,
            depth: self.depth,
            fast_forward,
            allow_node_mutation: self.allow_node_mutation,
        };

        if let Some(pool) = &self.thread_pool {
            // ── Parallel path with regret-based pruning ──
            //
            // All iterations run in parallel. When the regret matcher has
            // enough history (can_prune=true from a prior visit to this
            // node), non-reprobe iterations skip actions with zero strategy
            // weight. Otherwise all actions are explored.
            //
            // Reprobe iterations (every REPROBE_INTERVAL-th) always explore
            // all actions to detect changes in the active set.
            let num_actions = indexed_actions.len();
            let total_tasks = num_iterations * num_actions;

            // Pre-generate one RNG per task from the parent RNG.
            let task_rngs: Vec<R> = (0..total_tasks)
                .map(|_| R::from_rng(&mut self.rng))
                .collect();

            let active_actions = if can_prune {
                initial_active
            } else {
                seen_indices // all valid actions
            };

            let results: Vec<(usize, f32)> = pool.install(|| {
                task_rngs
                    .into_par_iter()
                    .enumerate()
                    .map(|(task_id, mut rng)| {
                        let iter_idx = task_id / num_actions;
                        let action_pos = task_id % num_actions;
                        let (action, reward_idx) = &indexed_actions[action_pos];

                        // Skip pruned actions on non-reprobe iterations
                        let is_reprobe = iter_idx.is_multiple_of(REPROBE_INTERVAL);
                        if can_prune && !is_reprobe && !active_actions.contains(*reward_idx) {
                            return (*reward_idx, invalid_action_penalty);
                        }

                        let reward = Self::compute_reward(game_state, action, &mut rng, &ctx);
                        (*reward_idx, reward)
                    })
                    .collect()
            });

            // Apply regret updates sequentially, grouped by iteration.
            for chunk in results.chunks(num_actions) {
                rewards.fill(invalid_action_penalty);
                for &(reward_idx, reward) in chunk {
                    rewards[reward_idx] = reward;
                }
                self.update_regret_at_node(target_node_idx, &rewards);
            }
        } else {
            // ── Sequential path with regret-based pruning ──
            //
            // Track the active action set. It starts from the initial read
            // and gets refreshed after each reprobe iteration so the pruning
            // mask stays current as regrets evolve within this call.
            let mut active_actions = initial_active;
            let mut updates_since_warmup = initial_updates;

            for iter_idx in 0..num_iterations {
                // Reset to penalty for all actions, valid ones get overwritten
                rewards.fill(invalid_action_penalty);

                // Decide whether to prune this iteration.
                // On reprobe iterations (every REPROBE_INTERVAL-th), explore all actions.
                let prune_this_iter = can_prune || updates_since_warmup >= PRUNE_WARMUP;
                let is_reprobe = iter_idx % REPROBE_INTERVAL == 0;
                let skip_pruned = prune_this_iter && !is_reprobe;

                for (action, reward_idx) in &indexed_actions {
                    // Regret-based pruning: skip actions with zero strategy weight
                    if skip_pruned && !active_actions.contains(*reward_idx) {
                        event!(
                            tracing::Level::TRACE,
                            action_idx = reward_idx,
                            iter = iter_idx,
                            "RBP: skipping pruned action"
                        );
                        continue;
                    }

                    debug_assert!(
                        *reward_idx < rewards.len(),
                        "Action index {} should be less than number of potential actions {}",
                        reward_idx,
                        rewards.len()
                    );

                    rewards[*reward_idx] =
                        Self::compute_reward(game_state, action, &mut self.rng, &ctx);
                }

                // Update regret immediately for this game state
                self.update_regret_at_node(target_node_idx, &rewards);
                updates_since_warmup += 1;

                // After a reprobe iteration, refresh the active action set
                // from the updated regret matcher.
                if is_reprobe && (can_prune || updates_since_warmup >= PRUNE_WARMUP) {
                    let (new_active, _) = self.cfr_state.get_pruning_info(target_node_idx);
                    active_actions = new_active;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Fast-forward helpers
//
// These free functions implement the cheap reward path used by
// `CFRAgent::compute_reward_fast_forward`. They mutate a cloned `GameState`
// directly: apply the candidate action, play out the rest of the hand
// assuming every further action is a check/call, and distribute a single pot.
// -----------------------------------------------------------------------------

/// Apply a single action on behalf of the current to-act player.
///
/// If the action fails validation (e.g. an illegal raise size because the
/// game state has drifted), we fall back to calling the current bet. An
/// unreachable edge case should degrade gracefully rather than poisoning
/// the reward signal.
fn fast_forward_apply_action(gs: &mut GameState, action: &AgentAction) {
    let call_current_bet = |gs: &mut GameState| {
        let _ = gs.do_bet(gs.current_round_bet(), false);
    };
    match action {
        AgentAction::Fold => gs.fold(),
        AgentAction::Call => call_current_bet(gs),
        AgentAction::Bet(amount) => {
            if gs.do_bet(*amount, false).is_err() {
                call_current_bet(gs);
            }
        }
        AgentAction::AllIn => {
            let idx = gs.to_act_idx();
            let target = gs.stacks[idx] + gs.current_round_player_bet(idx);
            if gs.do_bet(target, false).is_err() {
                call_current_bet(gs);
            }
        }
    }
}

/// Walk the game state forward through any remaining rounds. Betting rounds
/// are settled by having every still-needing-action player call; deal rounds
/// are settled by drawing fresh community cards from the remaining deck.
fn fast_forward_run_to_showdown<R: Rng>(gs: &mut GameState, rng: &mut R) {
    let mut deck = fast_forward_remaining_deck(gs);
    loop {
        // If at most one player can contest the pot, further play is moot —
        // skip straight to the pot distribution step.
        let contenders = gs.player_active.count() + gs.player_all_in.count();
        if contenders <= 1 {
            return;
        }
        match gs.round {
            Round::Showdown | Round::Complete => return,
            Round::Starting | Round::Ante | Round::DealPreflop => gs.advance_round(),
            Round::DealFlop => {
                fast_forward_deal_community_cards(gs, &mut deck, 3, rng);
                gs.advance_round();
            }
            Round::DealTurn | Round::DealRiver => {
                fast_forward_deal_community_cards(gs, &mut deck, 1, rng);
                gs.advance_round();
            }
            Round::Preflop | Round::Flop | Round::Turn | Round::River => {
                fast_forward_everyone_calls(gs);
                gs.advance_round();
            }
        }
    }
}

/// Have every player whose `needs_action` bit is still set call the current
/// bet. Players who cannot cover the call still put in what they have and
/// are marked all-in by `do_bet`.
fn fast_forward_everyone_calls(gs: &mut GameState) {
    // Safety cap: at most one call per seat per round. `do_bet` disables the
    // to-act player's `needs_action` bit, so this loop must terminate in at
    // most `num_players` iterations.
    for _ in 0..gs.num_players {
        if gs.round_data.num_players_need_action() == 0 {
            break;
        }
        let to_match = gs.current_round_bet();
        if gs.do_bet(to_match, false).is_err() {
            // The call validator can reject in pathological states (e.g.
            // NaN). Fall back to a check so we don't loop forever.
            let _ = gs.do_bet(0.0, false);
        }
    }
}

/// Build a deck of cards that haven't been dealt yet by removing every known
/// card from a fresh 52-card deck. Each player's hand already contains the
/// shared board cards, so iterating hands covers the board implicitly.
fn fast_forward_remaining_deck(gs: &GameState) -> Deck {
    let mut deck = Deck::default();
    for hand in &gs.hands {
        for card in hand.iter() {
            deck.remove(&card);
        }
    }
    deck
}

/// Draw `num_cards` from the deck and add them to the board and to every
/// player's hand, mirroring what `HoldemSimulation::deal_comunity_cards` does.
fn fast_forward_deal_community_cards<R: Rng>(
    gs: &mut GameState,
    deck: &mut Deck,
    num_cards: usize,
    rng: &mut R,
) {
    for _ in 0..num_cards {
        let Some(card) = deck.deal(rng) else { return };
        gs.board.push(card);
        for hand in gs.hands.iter_mut() {
            hand.insert(card);
        }
    }
}

/// Award the full pot to the best hand(s) among players still in the pot.
/// Uses a single pot (no side pots): ties split evenly.
fn fast_forward_distribute_pot(gs: &mut GameState) {
    let contenders = gs.player_active | gs.player_all_in;
    let count = contenders.count();
    if count == 0 {
        return;
    }
    let pot = gs.total_pot;
    if pot <= 0.0 {
        return;
    }
    if count == 1 {
        let winner = contenders.ones().next().unwrap();
        gs.award(winner, pot);
        gs.total_pot = 0.0;
        return;
    }
    let mut best_rank = None;
    let mut winners = PlayerBitSet::default();
    for idx in contenders.ones() {
        let rank = gs.hands[idx].rank();
        match best_rank {
            None => {
                best_rank = Some(rank);
                winners.enable(idx);
            }
            Some(current) if rank > current => {
                best_rank = Some(rank);
                winners = PlayerBitSet::default();
                winners.enable(idx);
            }
            Some(current) if rank == current => winners.enable(idx),
            _ => {}
        }
    }
    let split = pot / winners.count() as f32;
    for idx in winners.ones() {
        gs.award(idx, split);
    }
    gs.total_pot = 0.0;
}

/// Advance the game state through all remaining betting rounds (everyone
/// calls/checks) until a deal round or showdown is reached. This separates
/// the deterministic betting from the stochastic card dealing, allowing
/// the caller to enumerate board completions instead of sampling.
fn fast_forward_advance_betting(gs: &mut GameState) {
    // Safety cap: at most 8 round advances to prevent infinite loops.
    for _ in 0..8 {
        match gs.round {
            // Stop at deal rounds — the caller will enumerate cards.
            Round::DealFlop | Round::DealTurn | Round::DealRiver => return,
            // Stop at terminal states.
            Round::Showdown | Round::Complete => return,
            // Skip non-betting advance rounds.
            Round::Starting | Round::Ante | Round::DealPreflop => gs.advance_round(),
            // Betting rounds: everyone calls, then advance.
            Round::Preflop | Round::Flop | Round::Turn | Round::River => {
                fast_forward_everyone_calls(gs);
                gs.advance_round();
            }
        }
    }
}

/// Enumerate all possible board completions and compute the exact expected
/// reward for `player_idx`.
///
/// This replaces the random-sample approach in `fast_forward_run_to_showdown`
/// with deterministic enumeration when the number of remaining cards is small
/// enough (0, 1, or 2 cards). The result is zero variance in the reward
/// signal, which dramatically improves CFR convergence quality.
///
/// # Arguments
///
/// * `gs` - Game state positioned at a deal round (or showdown) after all
///   betting is resolved. The `total_pot` must already reflect all bets.
/// * `player_idx` - The player whose reward we compute.
/// * `cards_needed` - Number of community cards still to be dealt (0, 1, or 2).
fn fast_forward_enumerate_showdowns(gs: &GameState, player_idx: usize, cards_needed: usize) -> f32 {
    let contenders = gs.player_active | gs.player_all_in;
    let contender_count = contenders.count();

    // No contenders means everyone folded; the pot was already awarded.
    if contender_count == 0 {
        return gs.player_reward(player_idx);
    }

    // Single contender: they win everything regardless of board.
    if contender_count == 1 {
        let winner = contenders.ones().next().unwrap();
        let pot = gs.total_pot;
        let winnings = if winner == player_idx { pot } else { 0.0 };
        return winnings - gs.starting_stacks[player_idx];
    }

    let pot = gs.total_pot;
    if pot <= 0.0 {
        return gs.player_reward(player_idx);
    }

    // Collect the remaining deck into a Vec for indexed access.
    let deck = fast_forward_remaining_deck(gs);
    let remaining: Vec<crate::core::Card> = deck.iter().collect();

    if cards_needed == 0 {
        // Board is complete — just evaluate the showdown.
        return evaluate_showdown_reward(gs, &contenders, pot, player_idx);
    }

    let starting_stack = gs.starting_stacks[player_idx];
    let mut total_reward = 0.0f64;
    let mut count = 0u32;

    if cards_needed == 1 {
        // Enumerate single card (river).
        for &card in &remaining {
            let reward = evaluate_with_extra_cards(gs, &contenders, pot, player_idx, &[card]);
            total_reward += f64::from(reward - starting_stack);
            count += 1;
        }
    } else {
        // cards_needed == 2: enumerate all ordered pairs (turn + river).
        // We use ordered pairs (i < j) since card order doesn't matter for
        // hand evaluation, but we count each pair once.
        for i in 0..remaining.len() {
            for j in (i + 1)..remaining.len() {
                let reward = evaluate_with_extra_cards(
                    gs,
                    &contenders,
                    pot,
                    player_idx,
                    &[remaining[i], remaining[j]],
                );
                total_reward += f64::from(reward - starting_stack);
                count += 1;
            }
        }
    }

    (total_reward / f64::from(count)) as f32
}

/// Number of random flop samples to draw when 3 community cards remain.
/// For each sampled flop, all turn+river combinations are enumerated
/// exhaustively (~C(44,2) ≈ 946 evals per flop). This hybrid approach
/// gives much lower variance than a single random runout at modest cost.
const FLOP_SAMPLES: usize = 3;

/// Sample random flops and enumerate all turn+river completions for each.
///
/// When 3 community cards remain (pre-flop fast-forward), full enumeration
/// costs C(47,3) ≈ 16K evaluations — too expensive per action. Instead we
/// sample `FLOP_SAMPLES` random flop combinations and for each one
/// exhaustively enumerate all C(remaining,2) turn+river pairs. This
/// eliminates variance from 2 of the 3 unknown cards while keeping cost
/// at roughly `FLOP_SAMPLES × 1000` evaluations.
fn fast_forward_sample_flop_enumerate_runout<R: Rng>(
    gs: &GameState,
    player_idx: usize,
    rng: &mut R,
) -> f32 {
    fast_forward_sample_flop_enumerate_runout_n(gs, player_idx, rng, FLOP_SAMPLES)
}

/// Inner implementation parameterized by sample count for benchmarking.
fn fast_forward_sample_flop_enumerate_runout_n<R: Rng>(
    gs: &GameState,
    player_idx: usize,
    rng: &mut R,
    num_samples: usize,
) -> f32 {
    let contenders = gs.player_active | gs.player_all_in;
    let contender_count = contenders.count();

    if contender_count <= 1 {
        if contender_count == 0 {
            return gs.player_reward(player_idx);
        }
        let winner = contenders.ones().next().unwrap();
        let pot = gs.total_pot;
        let winnings = if winner == player_idx { pot } else { 0.0 };
        return winnings - gs.starting_stacks[player_idx];
    }

    let pot = gs.total_pot;
    if pot <= 0.0 {
        return gs.player_reward(player_idx);
    }

    let mut deck = fast_forward_remaining_deck(gs);
    let starting_stack = gs.starting_stacks[player_idx];
    let mut total_reward = 0.0f64;
    let mut total_count = 0u64;

    for _ in 0..num_samples {
        // Deal 3 random flop cards from the deck.
        let flop_cards: Vec<crate::core::Card> = (0..3).filter_map(|_| deck.deal(rng)).collect();
        if flop_cards.len() < 3 {
            // Not enough cards — shouldn't happen in practice.
            break;
        }

        // Build a temporary game state with the flop applied to hands.
        let mut gs_with_flop = gs.clone();
        for &card in &flop_cards {
            gs_with_flop.board.push(card);
            for hand in gs_with_flop.hands.iter_mut() {
                hand.insert(card);
            }
        }

        // Collect remaining cards (excluding the sampled flop).
        let flop_deck = fast_forward_remaining_deck(&gs_with_flop);
        let remaining: Vec<crate::core::Card> = flop_deck.iter().collect();

        // Enumerate all turn+river combinations exhaustively.
        for i in 0..remaining.len() {
            for j in (i + 1)..remaining.len() {
                let reward = evaluate_with_extra_cards(
                    &gs_with_flop,
                    &contenders,
                    pot,
                    player_idx,
                    &[remaining[i], remaining[j]],
                );
                total_reward += f64::from(reward - starting_stack);
                total_count += 1;
            }
        }

        // Put flop cards back in the deck for the next sample.
        for &card in &flop_cards {
            deck.insert(card);
        }
    }

    if total_count == 0 {
        return gs.player_reward(player_idx);
    }

    (total_reward / total_count as f64) as f32
}

/// Evaluate the showdown reward for a specific board completion.
///
/// Temporarily adds the given extra cards to each contender's hand,
/// ranks the hands, determines the winner(s), and returns the winnings
/// for `player_idx` (pot share if winner, 0 otherwise).
fn evaluate_with_extra_cards(
    gs: &GameState,
    contenders: &PlayerBitSet,
    pot: f32,
    player_idx: usize,
    extra_cards: &[crate::core::Card],
) -> f32 {
    use crate::core::Rankable;

    let mut best_rank = None;
    let mut winners = PlayerBitSet::default();
    let mut player_rank = None;

    for idx in contenders.ones() {
        let mut hand = gs.hands[idx];
        for &card in extra_cards {
            hand.insert(card);
        }
        let rank = hand.rank();

        if idx == player_idx {
            player_rank = Some(rank);
        }

        match best_rank {
            None => {
                best_rank = Some(rank);
                winners.enable(idx);
            }
            Some(current) if rank > current => {
                best_rank = Some(rank);
                winners = PlayerBitSet::default();
                winners.enable(idx);
            }
            Some(current) if rank == current => {
                winners.enable(idx);
            }
            _ => {}
        }
    }

    // Check if player_idx is among the winners.
    let _ = player_rank; // Used only if we wanted EV-based metrics.
    if winners.ones().any(|w| w == player_idx) {
        pot / winners.count() as f32
    } else {
        0.0
    }
}

/// Evaluate showdown with the current board (no extra cards).
fn evaluate_showdown_reward(
    gs: &GameState,
    contenders: &PlayerBitSet,
    pot: f32,
    player_idx: usize,
) -> f32 {
    let reward = evaluate_with_extra_cards(gs, contenders, pot, player_idx, &[]);
    reward - gs.starting_stacks[player_idx]
}

impl<T, R> Agent for CFRAgent<T, R>
where
    T: ActionGenerator + 'static,
    R: Rng + SeedableRng + Send + 'static,
{
    fn act(&mut self, id: u128, game_state: &GameState) -> crate::arena::action::AgentAction {
        event!(tracing::Level::TRACE, ?id, "Agent acting");
        assert!(
            game_state.round_data.to_act_idx == self.traversal_state.player_idx() as usize,
            "Agent should only be called when it's the player's turn"
        );

        // make sure that we have at least 2 cards
        let player_idx = self.traversal_state.player_idx() as usize;
        assert!(
            game_state.hands[player_idx].count() == 2 || game_state.hands[player_idx].count() >= 5,
            "Agent should only be called when it has at least 2 cards"
        );

        // Make sure that the CFR state has a regret matcher for this node
        self.ensure_target_node();

        if let Some(force_action) = self.forced_action.take() {
            event!(
                tracing::Level::DEBUG,
                ?force_action,
                "Playing forced action"
            );

            // Validate that the forced_action is still valid for this game state.
            // If not, we need to find a similar valid action.
            let valid_actions = self.action_generator.gen_possible_actions(game_state);

            // Check if the forced_action is in the valid actions (or close to it for Bet)
            match &force_action {
                AgentAction::Fold => {
                    if valid_actions.contains(&AgentAction::Fold) {
                        force_action
                    } else {
                        // Can't fold when there's nothing to call - this shouldn't happen
                        // but if it does, just call/check instead
                        event!(
                            tracing::Level::WARN,
                            "Forced Fold action invalid, using first valid action"
                        );
                        valid_actions.first().cloned().unwrap_or(AgentAction::Fold)
                    }
                }
                AgentAction::AllIn => {
                    // All-in should always be valid if we have chips
                    force_action
                }
                AgentAction::Call => {
                    // Call should always be valid
                    force_action
                }
                AgentAction::Bet(amount) => {
                    // For Bet, we need to verify the amount is still valid.
                    // The forced_action was generated for a specific game state, and while
                    // we expect the game state to be the same, let's validate to be safe.
                    // Use the ActionIndexMapper for consistent action-to-index mapping.
                    let forced_idx = self
                        .action_index_mapper
                        .action_to_idx(&force_action, game_state);

                    // Find a valid action with the same index
                    if let Some(valid_action) = valid_actions.iter().find(|a| {
                        self.action_index_mapper.action_to_idx(a, game_state) == forced_idx
                    }) {
                        // Found a valid action with the same index - use it
                        // (it might have a slightly different amount due to game state changes)
                        valid_action.clone()
                    } else {
                        // The forced action's index doesn't correspond to a valid action.
                        // This indicates a game state mismatch. Log and use a fallback.
                        event!(
                            tracing::Level::WARN,
                            ?force_action,
                            forced_idx = forced_idx,
                            amount = amount,
                            current_bet = game_state.current_round_bet(),
                            min_raise = game_state.current_round_min_raise(),
                            "Forced Bet action index not valid, using first valid action"
                        );
                        valid_actions.first().cloned().unwrap_or(AgentAction::Fold)
                    }
                }
            }
        } else {
            self.ensure_regret_matcher();
            // Explore all the potential actions
            self.explore_all_actions(game_state);

            // Use ActionPicker to select an action based on the regret matcher.
            // Run the raw generator output through `validate_actions` so the
            // picker sees the same filtered set used during training (e.g.,
            // honoring `max_raises_per_round`). Without this, a CFR strategy
            // trained on the filtered set could sample a raise-capped Bet at
            // play time, which `do_bet` then rejects.
            let raw_actions = self.action_generator.gen_possible_actions(game_state);
            let possible_actions =
                validate_actions(raw_actions, game_state, ValidatorMode::Standard);
            let target_node_idx = self.target_node_idx().unwrap();

            self.cfr_state.with_node_data(target_node_idx, |node_data| {
                let regret_matcher = get_regret_matcher_from_node(node_data);

                let picker = ActionPicker::new(
                    &self.action_index_mapper,
                    &possible_actions,
                    regret_matcher,
                    game_state,
                );
                picker.pick_action(&mut self.rng)
            })
        }
    }

    fn name(&self) -> &str {
        self.name.as_ref()
    }
}

#[cfg(test)]
mod tests {

    use crate::arena::GameStateBuilder;
    use crate::arena::agent::CallingAgent;
    use crate::arena::cfr::{
        BasicCFRActionGenerator, CfrDepthConfig, ConfigurableActionConfig,
        ConfigurableActionGenerator, TraversalSet,
    };

    use super::*;

    /// Helper to create CFR states for all players from a game state.
    fn make_cfr_state(game_state: &GameState) -> CFRState {
        CFRState::new(game_state.clone())
    }

    /// Test that a CFR agent can play against a non-CFR agent.
    /// This is a regression test for a bug where the CFR agent's reward()
    /// function assumed all players had CFR state initialized.
    ///
    /// The scenario: Player 0 is a CallingAgent (non-CFR), Player 1 is a CFR agent.
    /// The CFR agent uses shared CFR states for ALL players.
    #[test]
    fn test_cfr_vs_non_cfr_agent() {
        let stacks: Vec<f32> = vec![50.0, 50.0];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(5.0, 2.5)
            .build()
            .unwrap();

        // Create shared CFR states and TraversalSet
        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        let cfr_agent = Box::new(
            CFRAgentBuilder::<BasicCFRActionGenerator>::new()
                .name("CFRAgent-player1")
                .player_idx(1)
                .cfr_state(cfr_state.clone())
                .traversal_set(traversal_set.clone())
                .depth_config(CfrDepthConfig::new(vec![1]))
                .action_gen_config(())
                .build(),
        );

        // Player 0 is a simple calling agent (non-CFR)
        let calling_agent = Box::new(CallingAgent::new("CallingAgent-player0"));

        let agents: Vec<Box<dyn Agent>> = vec![calling_agent, cfr_agent];

        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .cfr_context(cfr_state.clone(), traversal_set, true)
            .build()
            .unwrap();

        // This should not panic - the CFR agent properly handles
        // mixed-agent simulations
        sim.run(&mut rng);
    }

    /// Regression test for B4: `CFRAgent::act` must route the raw generator
    /// output through `validate_actions` before handing it to the picker.
    /// Otherwise the picker can sample a raise after `max_raises_per_round`
    /// is reached, which `do_bet` would then reject.
    ///
    /// We pin the structural invariant: every action `act` returns is one of
    /// the actions present in `validate_actions(gen_possible_actions(state))`.
    /// `ConfigurableActionGenerator` is used because it emits raise sizings
    /// (the basic generator only emits Fold/Call/AllIn so the validator has
    /// nothing to filter).
    #[test]
    fn test_act_returns_only_validated_actions_when_cap_reached() {
        use crate::arena::cfr::action_validator::{ValidatorMode, validate_actions};
        use crate::core::{Card, Hand, Suit, Value};

        let mut game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .max_raises_per_round(Some(2))
            .build()
            .unwrap();

        // Advance to preflop and post blinds.
        game_state.advance_round(); // Starting -> Ante
        game_state.advance_round(); // Ante -> DealPreflop
        game_state.advance_round(); // DealPreflop -> Preflop
        game_state.do_bet(5.0, true).unwrap();
        game_state.do_bet(10.0, true).unwrap();

        let mut hand0 = Hand::default();
        hand0.insert(Card::new(Value::Ace, Suit::Spade));
        hand0.insert(Card::new(Value::King, Suit::Spade));
        let mut hand1 = Hand::default();
        hand1.insert(Card::new(Value::Queen, Suit::Heart));
        hand1.insert(Card::new(Value::Jack, Suit::Heart));
        game_state.hands[0] = hand0;
        game_state.hands[1] = hand1;

        // Cap raises so the validator must filter raise candidates.
        game_state.round_data.total_raise_count = 2;
        assert!(game_state.is_raise_capped());

        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        let mut agent = CFRAgentBuilder::<ConfigurableActionGenerator>::new()
            .name("CFRAgent-cap-test")
            .player_idx(game_state.to_act_idx())
            .cfr_state(cfr_state.clone())
            .traversal_set(traversal_set)
            .depth_config(CfrDepthConfig::new(vec![1]))
            .action_gen_config(ConfigurableActionConfig::default())
            .build();

        let raw_actions = agent.action_generator.gen_possible_actions(&game_state);
        let validated_actions =
            validate_actions(raw_actions.clone(), &game_state, ValidatorMode::Standard);
        // Sanity: validation actually removes raise candidates here.
        assert!(
            validated_actions.len() < raw_actions.len(),
            "validate_actions should filter raises once the cap is reached \
             (raw={raw_actions:?}, validated={validated_actions:?})"
        );

        for i in 0..32u128 {
            let action = agent.act(i, &game_state);
            assert!(
                validated_actions.contains(&action),
                "CFRAgent returned {action:?}, not in validated set \
                 {validated_actions:?} (raw set was {raw_actions:?})"
            );
        }
    }

    #[test]
    fn test_create_agent() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(3, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);
        let _ = CFRAgentBuilder::<BasicCFRActionGenerator>::new()
            .name("CFRAgent-test")
            .player_idx(0)
            .cfr_state(cfr_state.clone())
            .traversal_set(traversal_set)
            .depth_config(CfrDepthConfig::new(vec![1]))
            .action_gen_config(())
            .build();
    }

    #[test]
    fn test_run_heads_up() {
        let num_agents = 2;
        let stacks: Vec<f32> = vec![50.0, 50.0];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(5.0, 2.5)
            .build()
            .unwrap();

        // All CFR agents share the same CFR states and TraversalSet.
        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);
        let agents: Vec<Box<dyn Agent>> = (0..num_agents)
            .map(|i| {
                Box::new(
                    CFRAgentBuilder::<BasicCFRActionGenerator>::new()
                        .name(format!("CFRAgent-test-{i}"))
                        .player_idx(i)
                        .cfr_state(cfr_state.clone())
                        .traversal_set(traversal_set.clone())
                        .depth_config(CfrDepthConfig::new(vec![2, 1]))
                        .action_gen_config(())
                        .build(),
                ) as Box<dyn Agent>
            })
            .collect();

        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .cfr_context(cfr_state.clone(), traversal_set, true)
            .build()
            .unwrap();

        sim.run(&mut rng);
    }

    /// Test that agents sharing CFR states actually share the same CFR tree.
    /// This verifies the shared state pattern works correctly.
    #[test]
    fn test_shared_cfr_states_between_agents() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();

        // Create shared CFR states and traversal set
        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        let depth_config = CfrDepthConfig::new(vec![1]);

        // Create two agents with the same shared states
        let agent0 = CFRAgentBuilder::<BasicCFRActionGenerator>::new()
            .name("Agent0")
            .player_idx(0)
            .cfr_state(cfr_state.clone())
            .traversal_set(traversal_set.clone())
            .depth_config(depth_config.clone())
            .action_gen_config(())
            .build();

        let agent1 = CFRAgentBuilder::<BasicCFRActionGenerator>::new()
            .name("Agent1")
            .player_idx(1)
            .cfr_state(cfr_state.clone())
            .traversal_set(traversal_set)
            .depth_config(depth_config)
            .action_gen_config(())
            .build();

        // Both agents should have access to CFR state
        let state0 = agent0.cfr_state();
        let state1 = agent1.cfr_state();

        // Verify both see the same starting state (root node exists)
        assert!(state0.get_node_data(0).is_some());
        assert!(state1.get_node_data(0).is_some());

        // Both agents should be at valid positions
        assert!(state0.get_child(0, 0).is_none()); // No children yet at root
        assert!(state1.get_child(0, 0).is_none());
    }

    /// Test that the parallel code path (with a thread pool) completes
    /// successfully and produces valid results.
    #[test]
    fn test_run_heads_up_parallel() {
        let num_agents = 2;
        let stacks: Vec<f32> = vec![50.0, 50.0];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(5.0, 2.5)
            .build()
            .unwrap();

        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap(),
        );

        let agents: Vec<Box<dyn Agent>> = (0..num_agents)
            .map(|i| {
                Box::new(
                    CFRAgentBuilder::<BasicCFRActionGenerator>::new()
                        .name(format!("CFRAgent-par-{i}"))
                        .player_idx(i)
                        .cfr_state(cfr_state.clone())
                        .traversal_set(traversal_set.clone())
                        .depth_config(CfrDepthConfig::new(vec![2, 1]))
                        .action_gen_config(())
                        .thread_pool(pool.clone())
                        .build(),
                ) as Box<dyn Agent>
            })
            .collect();

        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .cfr_context(cfr_state.clone(), traversal_set, true)
            .build()
            .unwrap();

        sim.run(&mut rng);
    }

    /// Test that parallel and sequential paths both build the CFR tree.
    /// After running, the tree should have nodes beyond just the root.
    #[test]
    fn test_parallel_builds_cfr_tree() {
        let stacks: Vec<f32> = vec![50.0, 50.0];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(5.0, 2.5)
            .build()
            .unwrap();

        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap(),
        );

        let agents: Vec<Box<dyn Agent>> = (0..2)
            .map(|i| {
                Box::new(
                    CFRAgentBuilder::<BasicCFRActionGenerator>::new()
                        .name(format!("CFRAgent-par-{i}"))
                        .player_idx(i)
                        .cfr_state(cfr_state.clone())
                        .traversal_set(traversal_set.clone())
                        .depth_config(CfrDepthConfig::new(vec![2, 1]))
                        .action_gen_config(())
                        .thread_pool(pool.clone())
                        .build(),
                ) as Box<dyn Agent>
            })
            .collect();

        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .cfr_context(cfr_state.clone(), traversal_set, true)
            .build()
            .unwrap();

        sim.run(&mut rng);

        // After running, the shared CFR tree should have grown beyond just the root node
        assert!(
            cfr_state.arena().len() > 1,
            "CFR tree should have grown during simulation"
        );
    }

    /// Test that TraversalSet fork() provides proper sub-simulation isolation.
    #[test]
    fn test_sub_simulation_traversal_isolation() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let traversal_set = TraversalSet::new(game_state.num_players);

        // Get the initial traversal state position for player 0
        let initial = traversal_set.get(0);
        let initial_node = initial.node_idx();
        let initial_child = initial.chosen_child_idx();

        // Fork the traversal set (simulating sub-simulation start)
        let forked = traversal_set.fork();
        let sub_traversal = forked.get(0);

        // The sub-traversal should start at the same position
        assert_eq!(sub_traversal.node_idx(), initial_node);
        assert_eq!(sub_traversal.chosen_child_idx(), initial_child);

        // Move the sub-traversal
        sub_traversal.move_to(5, 3);

        // The sub-traversal should have moved
        assert_eq!(forked.get(0).node_idx(), 5);
        assert_eq!(forked.get(0).chosen_child_idx(), 3);

        // The original traversal should be unchanged
        assert_eq!(traversal_set.get(0).node_idx(), initial_node);
        assert_eq!(traversal_set.get(0).chosen_child_idx(), initial_child);
    }

    /// Reproduction test for the river call bug.
    ///
    /// Scenario: Player 1 holds K-high (7s Ks) facing an all-in on the river
    /// against Player 2 who has a pair of 9s (9h Qs). Board: 4s 4d 9d Ah Jd.
    /// Player 1 should ALWAYS fold, never call.
    ///
    /// This test verifies that:
    /// 1. The PCFR+ regret matcher correctly learns to fold with the reward structure
    /// 2. The full CFR agent picks fold when given this game state
    #[test]
    fn test_river_should_fold_king_high_vs_pair() {
        use crate::arena::cfr::action_generator::{
            ConfigurableActionConfig, ConfigurableActionGenerator,
        };
        use crate::arena::game_state::Round;
        use crate::core::{Card, Hand, PlayerBitSet, Suit, Value};
        use rand::SeedableRng;

        // Simplified version: 2-player, river, Player 1 facing all-in
        // Player 0 has pair of 9s (winning hand)
        // Player 1 has K-high (losing hand)
        //
        // Setup: preflop/flop/turn betting already happened.
        // Player 0 went all-in on river for 500.
        // Player 1 (with 300 remaining) must decide: fold or call.
        //
        // Board: 4s 4d 9d Ah Jd
        // Player 0 hand: 9h Qs (pair of 9s)
        // Player 1 hand: 7s Ks (K high)

        let board_cards = vec![
            Card::new(Value::Four, Suit::Spade),
            Card::new(Value::Four, Suit::Diamond),
            Card::new(Value::Nine, Suit::Diamond),
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::Jack, Suit::Diamond),
        ];

        // Player hands (including board cards for ranking)
        let p0_hole = vec![
            Card::new(Value::Nine, Suit::Heart),
            Card::new(Value::Queen, Suit::Spade),
        ];
        let p1_hole = vec![
            Card::new(Value::Seven, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];

        let mut p0_hand = Hand::new_with_cards(p0_hole.clone());
        p0_hand.extend(board_cards.iter().copied());
        let mut p1_hand = Hand::new_with_cards(p1_hole.clone());
        p1_hand.extend(board_cards.iter().copied());

        // Create game state at river decision point:
        // - Player 0 has gone all-in for 500 this round
        // - Player 1 has 300 left in stack, hasn't bet on river yet
        // - Previous rounds: both put in 200 each
        let stacks = vec![0.0, 300.0]; // P0 is all-in, P1 has 300 left
        let starting_stacks = vec![700.0, 700.0]; // Both started with 700
        let player_bet = vec![700.0, 400.0]; // Total bets over all rounds

        // Round data: P0 bet 500 this round, P1 hasn't bet yet
        let round_player_bet = vec![500.0, 0.0];
        let mut active = PlayerBitSet::new(2);
        // Only P1 is active (P0 is all-in)
        active.disable(0);

        let round_data = crate::arena::game_state::RoundData::new_with_bets(
            10.0, // min_raise
            active,
            1, // P1 to act
            round_player_bet,
        );

        let mut game_state = GameStateBuilder::new()
            .round(Round::River)
            .round_data(round_data)
            .stacks(stacks)
            .player_bet(player_bet)
            .big_blind(10.0)
            .small_blind(5.0)
            .hands(vec![p0_hand, p1_hand])
            .board(board_cards)
            .build()
            .unwrap();
        // Override starting_stacks (builder sets them = stacks)
        game_state.starting_stacks = starting_stacks;

        // Verify game state is correct
        assert_eq!(game_state.round, Round::River);
        assert_eq!(game_state.to_act_idx(), 1); // Player 1's turn
        assert_eq!(game_state.current_round_bet(), 500.0); // P0's all-in
        assert_eq!(game_state.current_player_stack(), 300.0); // P1's stack

        // Create CFR agent for Player 1
        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        // Use the configurable action generator (same as preflop chart post-flop)
        let action_config = ConfigurableActionConfig::default();

        let mut agent = CFRAgentBuilder::<ConfigurableActionGenerator, StdRng>::new()
            .name("TestCFRAgent")
            .player_idx(1)
            .cfr_state(cfr_state.clone())
            .traversal_set(traversal_set.clone())
            .depth_config(CfrDepthConfig::new(vec![24, 3, 1]))
            .action_gen_config(action_config)
            .rng(StdRng::seed_from_u64(42))
            .build();

        // Check what actions the generator produces
        let possible_actions = agent.action_generator.gen_possible_actions(&game_state);
        println!("Possible actions: {:?}", possible_actions);
        for action in &possible_actions {
            let idx = agent.action_index_mapper.action_to_idx(action, &game_state);
            println!("  {:?} -> index {}", action, idx);
        }

        // Run act() - this explores and picks an action
        let chosen_action = agent.act(0, &game_state);
        println!("Chosen action: {:?}", chosen_action);

        // The agent should fold with K-high facing an all-in
        assert!(
            matches!(chosen_action, AgentAction::Fold),
            "Agent should fold K-high facing all-in, but chose {:?}. \
             With a fresh regret matcher, 24 iterations of exploration \
             should overwhelmingly prefer fold over call.",
            chosen_action
        );

        // Also verify the regret matcher weights
        let target_node_idx = agent.target_node_idx().unwrap();
        agent
            .cfr_state
            .with_node_data(target_node_idx, |node_data| {
                let matcher = get_regret_matcher_from_node(node_data).unwrap();
                let weights = matcher.best_weight();
                let fold_weight = weights[0]; // ACTION_IDX_FOLD
                let call_weight = weights[1]; // ACTION_IDX_CALL

                println!(
                    "Weights after exploration: fold={:.6}, call={:.6}",
                    fold_weight, call_weight
                );

                assert!(
                    fold_weight > 0.99,
                    "Fold weight should be >0.99, got {:.6}. Call weight: {:.6}",
                    fold_weight,
                    call_weight
                );
            });
    }

    /// The same K-high river scenario as `test_river_should_fold_king_high_vs_pair`,
    /// but with an empty depth schedule. An empty schedule means the root
    /// agent itself has no recursive-hands budget, so it skips sub-simulation
    /// entirely and takes the fast-forward reward path — a direct end-to-end
    /// test that the fast-forward path still produces a decisive fold.
    #[test]
    fn test_river_fold_via_fast_forward_at_depth_zero() {
        use crate::arena::cfr::action_generator::{
            ConfigurableActionConfig, ConfigurableActionGenerator,
        };
        use crate::arena::game_state::Round;
        use crate::core::{Card, Hand, PlayerBitSet, Suit, Value};
        use rand::SeedableRng;

        let board_cards = vec![
            Card::new(Value::Four, Suit::Spade),
            Card::new(Value::Four, Suit::Diamond),
            Card::new(Value::Nine, Suit::Diamond),
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::Jack, Suit::Diamond),
        ];
        let p0_hole = vec![
            Card::new(Value::Nine, Suit::Heart),
            Card::new(Value::Queen, Suit::Spade),
        ];
        let p1_hole = vec![
            Card::new(Value::Seven, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];
        let mut p0_hand = Hand::new_with_cards(p0_hole);
        p0_hand.extend(board_cards.iter().copied());
        let mut p1_hand = Hand::new_with_cards(p1_hole);
        p1_hand.extend(board_cards.iter().copied());

        let stacks = vec![0.0, 300.0];
        let starting_stacks = vec![700.0, 700.0];
        let player_bet = vec![700.0, 400.0];
        let round_player_bet = vec![500.0, 0.0];
        let mut active = PlayerBitSet::new(2);
        active.disable(0);
        let round_data =
            crate::arena::game_state::RoundData::new_with_bets(10.0, active, 1, round_player_bet);

        let mut game_state = GameStateBuilder::new()
            .round(Round::River)
            .round_data(round_data)
            .stacks(stacks)
            .player_bet(player_bet)
            .big_blind(10.0)
            .small_blind(5.0)
            .hands(vec![p0_hand, p1_hand])
            .board(board_cards)
            .build()
            .unwrap();
        game_state.starting_stacks = starting_stacks;
        game_state.total_pot = 1100.0;

        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        let mut agent = CFRAgentBuilder::<ConfigurableActionGenerator, StdRng>::new()
            .name("CFR-ff")
            .player_idx(1)
            .cfr_state(cfr_state.clone())
            .traversal_set(traversal_set)
            .depth_config(CfrDepthConfig::new(vec![]))
            .action_gen_config(ConfigurableActionConfig::default())
            .rng(StdRng::seed_from_u64(42))
            .build();

        let chosen = agent.act(0, &game_state);
        assert!(
            matches!(chosen, AgentAction::Fold),
            "Agent should fold via fast-forward path, got {:?}",
            chosen
        );
    }

    /// Test that PCFR+ correctly handles the reward structure where call
    /// reward equals the invalid action penalty.
    #[test]
    fn test_pcfr_fold_vs_call_with_penalty_equal_to_call() {
        let mut matcher = PcfrPlusRegretMatcher::new(NUM_ACTION_INDICES);

        // Simulate the exact reward structure from the river call bug:
        // - Fold reward: -604 (lost what was already bet)
        // - Call reward: -1463 (lost entire stack)
        // - Invalid penalty: -1463 (same as call, since losing whole stack)
        let fold_reward = -604.0_f32;
        let call_reward = -1463.0_f32;
        let invalid_penalty = -1463.0_f32;

        for _ in 0..24 {
            let mut rewards = vec![invalid_penalty; NUM_ACTION_INDICES];
            rewards[0] = fold_reward; // Fold
            rewards[1] = call_reward; // Call
            matcher.update_regret(&rewards);
        }

        let weights = matcher.best_weight();
        let fold_weight = weights[0];
        let call_weight = weights[1];

        println!(
            "PCFR+ after 24 iterations: fold={:.6}, call={:.6}",
            fold_weight, call_weight
        );

        // Fold should dominate
        assert!(
            fold_weight > 0.99,
            "Fold should have >99% weight, got {:.4}%",
            fold_weight * 100.0
        );
        assert!(
            call_weight < 0.01,
            "Call should have <1% weight, got {:.4}%",
            call_weight * 100.0
        );
    }

    // -------------------------------------------------------------------------
    // Fast-forward helper tests
    //
    // These exercise the free `fast_forward_*` helpers directly, without
    // going through a CFRAgent. They verify pot distribution, tie-splitting,
    // and the deal-path from mid-hand rounds.
    // -------------------------------------------------------------------------

    /// River, heads-up: P0 has a pair of 9s, P1 has K-high. P0 is all-in for
    /// 500. P1 calls → P1 loses their remaining stack; P0 wins the pot.
    #[test]
    fn fast_forward_river_call_loses_with_worse_hand() {
        use crate::arena::game_state::{Round, RoundData};
        use crate::core::{Card, Hand, PlayerBitSet, Suit, Value};

        let board_cards = vec![
            Card::new(Value::Four, Suit::Spade),
            Card::new(Value::Four, Suit::Diamond),
            Card::new(Value::Nine, Suit::Diamond),
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::Jack, Suit::Diamond),
        ];

        let p0_hole = vec![
            Card::new(Value::Nine, Suit::Heart),
            Card::new(Value::Queen, Suit::Spade),
        ];
        let p1_hole = vec![
            Card::new(Value::Seven, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];

        let mut p0_hand = Hand::new_with_cards(p0_hole);
        p0_hand.extend(board_cards.iter().copied());
        let mut p1_hand = Hand::new_with_cards(p1_hole);
        p1_hand.extend(board_cards.iter().copied());

        let stacks = vec![0.0, 300.0];
        let starting_stacks = vec![700.0, 700.0];
        let player_bet = vec![700.0, 400.0];
        let round_player_bet = vec![500.0, 0.0];
        let mut active = PlayerBitSet::new(2);
        active.disable(0);

        let round_data = RoundData::new_with_bets(10.0, active, 1, round_player_bet);

        let mut game_state = GameStateBuilder::new()
            .round(Round::River)
            .round_data(round_data)
            .stacks(stacks)
            .player_bet(player_bet)
            .big_blind(10.0)
            .small_blind(5.0)
            .hands(vec![p0_hand, p1_hand])
            .board(board_cards)
            .build()
            .unwrap();
        game_state.starting_stacks = starting_stacks;
        // Total pot reflects what's already in: 700 + 400 = 1100.
        game_state.total_pot = 1100.0;
        game_state.player_all_in = {
            let mut pbs = PlayerBitSet::new(2);
            pbs.disable(1);
            pbs
        };

        let mut rng = StdRng::seed_from_u64(7);

        // Calling with K-high: P1 pays 300 more, goes all-in, loses at showdown.
        let mut call_state = game_state.clone();
        fast_forward_apply_action(&mut call_state, &AgentAction::Call);
        fast_forward_run_to_showdown(&mut call_state, &mut rng);
        fast_forward_distribute_pot(&mut call_state);
        let call_reward = call_state.player_reward(1);
        assert!(
            call_reward < -699.0,
            "Calling with the losing hand should cost ~700 stack, got {}",
            call_reward
        );

        // Folding: P1 forfeits only what's already in the pot (400).
        let mut fold_state = game_state.clone();
        fast_forward_apply_action(&mut fold_state, &AgentAction::Fold);
        fast_forward_run_to_showdown(&mut fold_state, &mut rng);
        fast_forward_distribute_pot(&mut fold_state);
        let fold_reward = fold_state.player_reward(1);
        assert!(
            (fold_reward - (-400.0)).abs() < 0.01,
            "Folding should cost exactly the 400 already committed, got {}",
            fold_reward
        );

        // Folding should be strictly better than calling here.
        assert!(fold_reward > call_reward);
    }

    /// Tie: both players play the board on a paired board → pot split.
    #[test]
    fn fast_forward_split_pot_on_tie() {
        use crate::arena::game_state::{Round, RoundData};
        use crate::core::{Card, Hand, PlayerBitSet, Suit, Value};

        let board_cards = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Ace, Suit::Diamond),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Queen, Suit::Club),
            Card::new(Value::Jack, Suit::Diamond),
        ];
        // Both players play A-A-K-Q-J from the board.
        let p0_hole = vec![
            Card::new(Value::Two, Suit::Club),
            Card::new(Value::Three, Suit::Club),
        ];
        let p1_hole = vec![
            Card::new(Value::Two, Suit::Heart),
            Card::new(Value::Three, Suit::Spade),
        ];
        let mut p0 = Hand::new_with_cards(p0_hole);
        p0.extend(board_cards.iter().copied());
        let mut p1 = Hand::new_with_cards(p1_hole);
        p1.extend(board_cards.iter().copied());

        let round_data = RoundData::new_with_bets(10.0, PlayerBitSet::new(2), 0, vec![0.0, 0.0]);

        let mut gs = GameStateBuilder::new()
            .round(Round::River)
            .round_data(round_data)
            .stacks(vec![500.0, 500.0])
            .player_bet(vec![500.0, 500.0])
            .big_blind(10.0)
            .small_blind(5.0)
            .hands(vec![p0, p1])
            .board(board_cards)
            .build()
            .unwrap();
        gs.starting_stacks = vec![1000.0, 1000.0];
        gs.total_pot = 1000.0;

        let mut rng = StdRng::seed_from_u64(11);
        fast_forward_apply_action(&mut gs, &AgentAction::Call);
        fast_forward_run_to_showdown(&mut gs, &mut rng);
        fast_forward_distribute_pot(&mut gs);
        let reward = gs.player_reward(0);
        // 1000 pot split between both players → ~0 net.
        assert!(
            reward.abs() < 0.01,
            "split should yield ~0 reward, got {}",
            reward
        );
    }

    /// Fast-forward from the flop: the helper must deal the turn and river
    /// before running the showdown rather than panicking on an empty board.
    #[test]
    fn fast_forward_from_flop_deals_turn_and_river() {
        use crate::arena::game_state::{Round, RoundData};
        use crate::core::{Card, Hand, PlayerBitSet, Suit, Value};

        let board_cards = vec![
            Card::new(Value::Two, Suit::Spade),
            Card::new(Value::Seven, Suit::Diamond),
            Card::new(Value::Jack, Suit::Heart),
        ];
        let p0_hole = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Ace, Suit::Diamond),
        ];
        let p1_hole = vec![
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Diamond),
        ];
        let mut p0 = Hand::new_with_cards(p0_hole);
        p0.extend(board_cards.iter().copied());
        let mut p1 = Hand::new_with_cards(p1_hole);
        p1.extend(board_cards.iter().copied());

        let round_data = RoundData::new_with_bets(10.0, PlayerBitSet::new(2), 0, vec![0.0, 0.0]);
        let mut gs = GameStateBuilder::new()
            .round(Round::Flop)
            .round_data(round_data)
            .stacks(vec![100.0, 100.0])
            .player_bet(vec![10.0, 10.0])
            .big_blind(10.0)
            .small_blind(5.0)
            .hands(vec![p0, p1])
            .board(board_cards)
            .build()
            .unwrap();
        gs.starting_stacks = vec![110.0, 110.0];
        gs.total_pot = 20.0;

        let mut rng = StdRng::seed_from_u64(3);
        fast_forward_apply_action(&mut gs, &AgentAction::Call);
        fast_forward_run_to_showdown(&mut gs, &mut rng);
        fast_forward_distribute_pot(&mut gs);
        // With random turn/river cards we don't assert an exact reward — the
        // point is that we reached showdown (board has 5 cards) instead of
        // panicking on an empty deck / missing cards.
        assert_eq!(gs.board.len(), 5);
    }

    /// Test that regret-based pruning produces the same correct result
    /// as the unpruned path. With enough iterations, the agent should
    /// still fold K-high facing an all-in.
    #[test]
    fn test_rbp_preserves_fold_decision() {
        use crate::arena::cfr::action_generator::{
            ConfigurableActionConfig, ConfigurableActionGenerator,
        };
        use crate::arena::game_state::Round;
        use crate::core::{Card, Hand, PlayerBitSet, Suit, Value};
        use rand::SeedableRng;

        let board_cards = vec![
            Card::new(Value::Four, Suit::Spade),
            Card::new(Value::Four, Suit::Diamond),
            Card::new(Value::Nine, Suit::Diamond),
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::Jack, Suit::Diamond),
        ];
        let p0_hole = vec![
            Card::new(Value::Nine, Suit::Heart),
            Card::new(Value::Queen, Suit::Spade),
        ];
        let p1_hole = vec![
            Card::new(Value::Seven, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];
        let mut p0_hand = Hand::new_with_cards(p0_hole);
        p0_hand.extend(board_cards.iter().copied());
        let mut p1_hand = Hand::new_with_cards(p1_hole);
        p1_hand.extend(board_cards.iter().copied());

        let stacks = vec![0.0, 300.0];
        let starting_stacks = vec![700.0, 700.0];
        let player_bet = vec![700.0, 400.0];
        let round_player_bet = vec![500.0, 0.0];
        let mut active = PlayerBitSet::new(2);
        active.disable(0);
        let round_data =
            crate::arena::game_state::RoundData::new_with_bets(10.0, active, 1, round_player_bet);

        let mut game_state = GameStateBuilder::new()
            .round(Round::River)
            .round_data(round_data)
            .stacks(stacks)
            .player_bet(player_bet)
            .big_blind(10.0)
            .small_blind(5.0)
            .hands(vec![p0_hand, p1_hand])
            .board(board_cards)
            .build()
            .unwrap();
        game_state.starting_stacks = starting_stacks;

        // Use 24 iterations — enough for RBP to kick in (warmup=3, reprobe every 4th).
        // Pruning should skip computing rewards for clearly-bad actions after warmup.
        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        let mut agent = CFRAgentBuilder::<ConfigurableActionGenerator, StdRng>::new()
            .name("CFR-RBP-test")
            .player_idx(1)
            .cfr_state(cfr_state.clone())
            .traversal_set(traversal_set)
            .depth_config(CfrDepthConfig::new(vec![24, 3, 1]))
            .action_gen_config(ConfigurableActionConfig::default())
            .rng(StdRng::seed_from_u64(42))
            .build();

        let chosen = agent.act(0, &game_state);
        assert!(
            matches!(chosen, AgentAction::Fold),
            "RBP should preserve the fold decision for K-high vs pair, got {:?}",
            chosen
        );
    }

    /// Test that regret-based pruning actually activates by verifying
    /// that the pruning info bitset becomes sparse after warmup.
    #[test]
    fn test_rbp_reduces_active_actions() {
        use crate::arena::cfr::action_generator::{
            ConfigurableActionConfig, ConfigurableActionGenerator,
        };
        use crate::arena::game_state::Round;
        use crate::core::{Card, Hand, PlayerBitSet, Suit, Value};
        use rand::SeedableRng;

        let board_cards = vec![
            Card::new(Value::Four, Suit::Spade),
            Card::new(Value::Four, Suit::Diamond),
            Card::new(Value::Nine, Suit::Diamond),
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::Jack, Suit::Diamond),
        ];
        let p0_hole = vec![
            Card::new(Value::Nine, Suit::Heart),
            Card::new(Value::Queen, Suit::Spade),
        ];
        let p1_hole = vec![
            Card::new(Value::Seven, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];
        let mut p0_hand = Hand::new_with_cards(p0_hole);
        p0_hand.extend(board_cards.iter().copied());
        let mut p1_hand = Hand::new_with_cards(p1_hole);
        p1_hand.extend(board_cards.iter().copied());

        let stacks = vec![0.0, 300.0];
        let starting_stacks = vec![700.0, 700.0];
        let player_bet = vec![700.0, 400.0];
        let round_player_bet = vec![500.0, 0.0];
        let mut active = PlayerBitSet::new(2);
        active.disable(0);
        let round_data =
            crate::arena::game_state::RoundData::new_with_bets(10.0, active, 1, round_player_bet);

        let mut game_state = GameStateBuilder::new()
            .round(Round::River)
            .round_data(round_data)
            .stacks(stacks)
            .player_bet(player_bet)
            .big_blind(10.0)
            .small_blind(5.0)
            .hands(vec![p0_hand, p1_hand])
            .board(board_cards)
            .build()
            .unwrap();
        game_state.starting_stacks = starting_stacks;

        let cfr_state = make_cfr_state(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        let mut agent = CFRAgentBuilder::<ConfigurableActionGenerator, StdRng>::new()
            .name("CFR-RBP-sparse")
            .player_idx(1)
            .cfr_state(cfr_state.clone())
            .traversal_set(traversal_set)
            .depth_config(CfrDepthConfig::new(vec![24, 3, 1]))
            .action_gen_config(ConfigurableActionConfig::default())
            .rng(StdRng::seed_from_u64(42))
            .build();

        // Run exploration
        let _ = agent.act(0, &game_state);

        // After 24 iterations with clear fold > call, the active set
        // should have fewer actions than the total valid set.
        let target_node_idx = agent.target_node_idx().unwrap();
        let (active_set, num_updates) = agent.cfr_state.get_pruning_info(target_node_idx);

        println!(
            "After exploration: {} active actions, {} updates",
            active_set.count(),
            num_updates
        );

        // The regret matcher should have been updated at least 24 times
        assert!(
            num_updates >= 24,
            "Expected >= 24 updates, got {}",
            num_updates
        );

        // With a clear fold vs call scenario, not all actions should remain active.
        // Fold should dominate, so call and other actions should have 0 weight.
        // We expect at most 2 active actions (fold + possibly one other)
        // in this lopsided scenario.
        assert!(
            active_set.count() <= 3,
            "Expected <= 3 active actions after pruning, got {}. \
             In this lopsided fold-vs-call scenario, most actions should be pruned.",
            active_set.count()
        );
    }

    /// Test that multi-sample flop + exhaustive runout produces lower
    /// variance than single-sample for a preflop scenario.
    ///
    /// Runs both approaches many times with different RNG seeds and
    /// compares the standard deviation of the rewards.
    #[test]
    fn test_flop_sample_variance_vs_single_sample() {
        use crate::arena::game_state::Round;
        use crate::core::{Card, Hand, Suit, Value};
        use rand::SeedableRng;

        // Set up a preflop-like game state where both players are all-in
        // and we need to deal flop+turn+river.
        let p0_hole = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];
        let p1_hole = vec![
            Card::new(Value::Queen, Suit::Heart),
            Card::new(Value::Jack, Suit::Heart),
        ];
        let p0_hand = Hand::new_with_cards(p0_hole);
        let p1_hand = Hand::new_with_cards(p1_hole);

        // stacks=0 + bets>0 in a non-Starting round → both players all-in
        let mut game_state = GameStateBuilder::new()
            .round(Round::DealFlop)
            .stacks(vec![0.0, 0.0])
            .player_bet(vec![500.0, 500.0])
            .big_blind(10.0)
            .small_blind(5.0)
            .hands(vec![p0_hand, p1_hand])
            .build()
            .unwrap();
        game_state.starting_stacks = vec![500.0, 500.0];

        let num_trials = 50;

        // Multi-sample flop + exhaustive runout (current approach)
        let multi_results: Vec<f32> = (0..num_trials)
            .map(|seed| {
                let mut rng = StdRng::seed_from_u64(seed);
                fast_forward_sample_flop_enumerate_runout(&game_state, 0, &mut rng)
            })
            .collect();

        // Single-sample baseline (old approach: random runout)
        let single_results: Vec<f32> = (0..num_trials)
            .map(|seed| {
                let mut gs = game_state.clone();
                let mut rng = StdRng::seed_from_u64(seed);
                fast_forward_run_to_showdown(&mut gs, &mut rng);
                fast_forward_distribute_pot(&mut gs);
                gs.player_reward(0)
            })
            .collect();

        fn std_dev(vals: &[f32]) -> f64 {
            let n = vals.len() as f64;
            let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / n;
            let var = vals.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
            var.sqrt()
        }

        let multi_std = std_dev(&multi_results);
        let single_std = std_dev(&single_results);

        let multi_mean: f64 =
            multi_results.iter().map(|&v| v as f64).sum::<f64>() / num_trials as f64;
        let single_mean: f64 =
            single_results.iter().map(|&v| v as f64).sum::<f64>() / num_trials as f64;

        println!(
            "Multi-sample flop (k={}): mean={:.2}, std={:.2}",
            FLOP_SAMPLES, multi_mean, multi_std
        );
        println!(
            "Single-sample runout:     mean={:.2}, std={:.2}",
            single_mean, single_std
        );
        println!(
            "Variance reduction: {:.1}x",
            if multi_std > 0.0 {
                single_std / multi_std
            } else {
                f64::INFINITY
            }
        );

        // The multi-sample approach should have meaningfully lower variance.
        // With k=3 flops and exact runout, we expect at least 2x reduction.
        assert!(
            multi_std < single_std,
            "Multi-sample flop should have lower variance than single-sample. \
             multi_std={:.2}, single_std={:.2}",
            multi_std,
            single_std
        );

        // Both means should be in roughly the same ballpark (same underlying EV).
        // AKs vs QJs preflop is roughly 60/40, so EV for player 0 ≈ +200 (win 1000 * 0.6 - 500).
        // Allow a wide range since we're measuring with limited trials.
        assert!(
            (multi_mean - single_mean).abs() < 200.0,
            "Means should be broadly similar: multi={:.2}, single={:.2}",
            multi_mean,
            single_mean
        );
    }

    /// Test that multi-sample flop enumeration produces correct sign
    /// for a strongly dominated hand. AKs should beat 72o.
    #[test]
    fn test_flop_sample_dominated_hand() {
        use crate::arena::game_state::Round;
        use crate::core::{Card, Hand, Suit, Value};
        use rand::SeedableRng;

        let p0_hole = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];
        let p1_hole = vec![
            Card::new(Value::Seven, Suit::Diamond),
            Card::new(Value::Two, Suit::Club),
        ];
        let p0_hand = Hand::new_with_cards(p0_hole);
        let p1_hand = Hand::new_with_cards(p1_hole);

        // stacks=0 + bets>0 in a non-Starting round → both players all-in
        let mut game_state = GameStateBuilder::new()
            .round(Round::DealFlop)
            .stacks(vec![0.0, 0.0])
            .player_bet(vec![500.0, 500.0])
            .big_blind(10.0)
            .small_blind(5.0)
            .hands(vec![p0_hand, p1_hand])
            .build()
            .unwrap();
        game_state.starting_stacks = vec![500.0, 500.0];

        // Average over multiple seeds — AKs should consistently beat 72o.
        let num_trials = 20;
        let total_reward: f64 = (0..num_trials)
            .map(|seed| {
                let mut rng = StdRng::seed_from_u64(seed + 100);
                fast_forward_sample_flop_enumerate_runout(&game_state, 0, &mut rng) as f64
            })
            .sum();
        let avg_reward = total_reward / num_trials as f64;

        println!("AKs vs 72o avg reward for AKs: {:.2}", avg_reward);

        // AKs vs 72o has ~66% equity, so EV ≈ 1000 * 0.66 - 500 = +160.
        // Should be solidly positive.
        assert!(
            avg_reward > 50.0,
            "AKs should have positive EV vs 72o, got {:.2}",
            avg_reward
        );
    }

    /// Compare variance and cost across different flop sample counts.
    /// Run with `--nocapture` to see the table:
    ///   cargo test --all-features test_flop_sample_count_comparison -- --nocapture
    #[test]
    fn test_flop_sample_count_comparison() {
        use crate::arena::game_state::Round;
        use crate::core::{Card, Hand, Suit, Value};
        use rand::SeedableRng;
        use std::time::Instant;

        let p0_hole = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];
        let p1_hole = vec![
            Card::new(Value::Queen, Suit::Heart),
            Card::new(Value::Jack, Suit::Heart),
        ];
        let p0_hand = Hand::new_with_cards(p0_hole);
        let p1_hand = Hand::new_with_cards(p1_hole);

        let mut game_state = GameStateBuilder::new()
            .round(Round::DealFlop)
            .stacks(vec![0.0, 0.0])
            .player_bet(vec![500.0, 500.0])
            .big_blind(10.0)
            .small_blind(5.0)
            .hands(vec![p0_hand, p1_hand])
            .build()
            .unwrap();
        game_state.starting_stacks = vec![500.0, 500.0];

        let num_trials = 100;

        fn std_dev(vals: &[f32]) -> f64 {
            let n = vals.len() as f64;
            let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / n;
            let var = vals.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
            var.sqrt()
        }

        println!(
            "\n{:<8} {:>10} {:>10} {:>12} {:>10}",
            "k", "mean", "std", "var_red", "time_us"
        );
        println!("{}", "-".repeat(56));

        // Single-sample baseline
        let start = Instant::now();
        let single_results: Vec<f32> = (0..num_trials)
            .map(|seed| {
                let mut gs = game_state.clone();
                let mut rng = StdRng::seed_from_u64(seed);
                fast_forward_run_to_showdown(&mut gs, &mut rng);
                fast_forward_distribute_pot(&mut gs);
                gs.player_reward(0)
            })
            .collect();
        let single_time = start.elapsed();
        let single_std = std_dev(&single_results);
        let single_mean: f64 =
            single_results.iter().map(|&v| v as f64).sum::<f64>() / num_trials as f64;

        println!(
            "{:<8} {:>10.2} {:>10.2} {:>12} {:>10}",
            "1-samp",
            single_mean,
            single_std,
            "baseline",
            single_time.as_micros()
        );

        // Test k = 1, 2, 3, 5, 8, 13
        for k in [1, 2, 3, 5, 8, 13] {
            let start = Instant::now();
            let results: Vec<f32> = (0..num_trials)
                .map(|seed| {
                    let mut rng = StdRng::seed_from_u64(seed);
                    fast_forward_sample_flop_enumerate_runout_n(&game_state, 0, &mut rng, k)
                })
                .collect();
            let elapsed = start.elapsed();

            let multi_std = std_dev(&results);
            let multi_mean: f64 =
                results.iter().map(|&v| v as f64).sum::<f64>() / num_trials as f64;
            let var_reduction = if multi_std > 0.0 {
                single_std / multi_std
            } else {
                f64::INFINITY
            };

            println!(
                "{:<8} {:>10.2} {:>10.2} {:>11.1}x {:>10}",
                format!("k={k}"),
                multi_mean,
                multi_std,
                var_reduction,
                elapsed.as_micros()
            );
        }

        // Sanity: k=3 should reduce variance vs single sample
        let k3_results: Vec<f32> = (0..num_trials)
            .map(|seed| {
                let mut rng = StdRng::seed_from_u64(seed);
                fast_forward_sample_flop_enumerate_runout_n(&game_state, 0, &mut rng, 3)
            })
            .collect();
        assert!(
            std_dev(&k3_results) < single_std,
            "k=3 should reduce variance vs single sample"
        );
    }
}
