use std::borrow::Cow;
use std::marker::PhantomData;
use std::sync::Arc;

use little_sorry::{PcfrPlusRegretMatcher, RegretMinimizer};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use tracing::event;

use crate::arena::{Agent, GameState, HoldemSimulationBuilder, action::AgentAction};

use super::{
    ActionIndexMapper, ActionIndexMapperConfig, ActionPicker, CFRState, GameStateIteratorGen,
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
struct ComputeRewardContext<'a, T: ActionGenerator, I: GameStateIteratorGen> {
    traversal_set: &'a TraversalSet,
    traversal_state: &'a TraversalState,
    cfr_states: &'a Arc<[CFRState]>,
    iter_gen_config: &'a Arc<I::Config>,
    action_gen_config: &'a Arc<T::Config>,
    action_index_mapper: &'a ActionIndexMapper,
    depth: usize,
    limited_exploration_depth: Option<usize>,
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
/// * `I` - The game state iterator generator type
/// * `R` - The random number generator type (defaults to `StdRng`)
pub struct CFRAgent<T, I, R = StdRng>
where
    T: ActionGenerator,
    I: GameStateIteratorGen,
    R: Rng + SeedableRng,
{
    name: Cow<'static, str>,
    cfr_states: Arc<[CFRState]>,
    traversal_set: TraversalSet,
    traversal_state: TraversalState,
    cfr_state: CFRState,
    action_generator: T,
    gamestate_iterator_gen: I,
    /// Shared config references for cheap cloning in reward() sub-agent construction.
    action_gen_config: Arc<T::Config>,
    iter_gen_config: Arc<I::Config>,
    /// The action index mapper for consistent action-to-index mapping.
    action_index_mapper: ActionIndexMapper,

    // This will be the next action to play
    // This allows us to start exploration
    // from a specific action.
    forced_action: Option<AgentAction>,

    /// Recursion depth in CFR tree (0 = root agent, 1+ = sub-agent).
    depth: usize,

    /// Depth at which to switch to limited action exploration.
    /// When depth >= this value, only Fold, Call, AllIn actions are explored.
    /// None means no limit (always use full action set).
    limited_exploration_depth: Option<usize>,

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
/// All CFR agents in a simulation should share the same `Vec<CFRState>` and
/// `TraversalSet` to enable shared learning and coordinated tree traversal.
///
/// # Type Parameters
/// * `T` - The action generator type (implements `ActionGenerator`)
/// * `I` - The game state iterator generator type
/// * `R` - The random number generator type (defaults to `StdRng`)
pub struct CFRAgentBuilder<T, I, R = StdRng>
where
    T: ActionGenerator,
    I: GameStateIteratorGen,
    R: Rng + SeedableRng,
{
    name: Option<Cow<'static, str>>,
    player_idx: Option<usize>,
    gamestate_iterator_gen_config: Option<Arc<I::Config>>,
    action_gen_config: Option<Arc<T::Config>>,
    cfr_states: Option<Arc<[CFRState]>>,
    traversal_set: Option<TraversalSet>,
    /// Pre-fetched CFR state to avoid lock acquisition in build().
    cfr_state: Option<CFRState>,
    /// Pre-fetched mapper config to avoid lock acquisition in build().
    mapper_config: Option<ActionIndexMapperConfig>,
    forced_action: Option<AgentAction>,
    /// Recursion depth in CFR tree (0 = root, 1+ = sub-agent).
    depth: usize,
    /// Depth at which to switch to limited action exploration.
    limited_exploration_depth: Option<usize>,
    /// Whether to allow mutating node types when a mismatch is found.
    allow_node_mutation: bool,
    /// Optional thread pool for parallel action exploration.
    thread_pool: Option<Arc<rayon::ThreadPool>>,
    /// Optional RNG instance. If None, creates one from system entropy.
    rng: Option<R>,
    /// Phantom data to satisfy the compiler.
    _marker: PhantomData<R>,
}

impl<T, I, R> Default for CFRAgentBuilder<T, I, R>
where
    T: ActionGenerator,
    I: GameStateIteratorGen,
    R: Rng + SeedableRng,
{
    fn default() -> Self {
        Self {
            name: None,
            player_idx: None,
            gamestate_iterator_gen_config: None,
            action_gen_config: None,
            cfr_states: None,
            traversal_set: None,
            cfr_state: None,
            mapper_config: None,
            forced_action: None,
            depth: 0,
            limited_exploration_depth: Some(3),
            allow_node_mutation: true,
            thread_pool: None,
            rng: None,
            _marker: PhantomData,
        }
    }
}

impl<T, I, R> CFRAgentBuilder<T, I, R>
where
    T: ActionGenerator,
    I: GameStateIteratorGen,
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

    /// Set the game state iterator generator configuration.
    pub fn gamestate_iterator_gen_config(mut self, config: I::Config) -> Self {
        self.gamestate_iterator_gen_config = Some(Arc::new(config));
        self
    }

    /// Set the game state iterator generator configuration from a shared Arc.
    /// Used internally to avoid re-wrapping in Arc during sub-agent construction.
    fn gamestate_iterator_gen_config_arc(mut self, config: Arc<I::Config>) -> Self {
        self.gamestate_iterator_gen_config = Some(config);
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

    /// Set the shared CFR states for this agent.
    ///
    /// All CFR agents in a simulation should share the same Vec<CFRState>.
    /// CFRState is cheap to clone (just Arc bumps).
    pub fn cfr_states(mut self, cfr_states: Vec<CFRState>) -> Self {
        self.cfr_states = Some(cfr_states.into());
        self
    }

    /// Set the shared CFR states from a pre-built Arc.
    /// Used internally by sub-agent construction to avoid re-allocating.
    fn cfr_states_arc(mut self, cfr_states: Arc<[CFRState]>) -> Self {
        self.cfr_states = Some(cfr_states);
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

    /// Set the depth at which to switch to limited action exploration.
    ///
    /// When depth >= this value, only Fold, Call, AllIn actions are explored.
    /// This reduces the branching factor deep in the CFR tree.
    pub fn limited_exploration_depth(mut self, depth: usize) -> Self {
        self.limited_exploration_depth = Some(depth);
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
    /// - cfr_states
    /// - traversal_set
    /// - gamestate_iterator_gen_config
    /// - action_gen_config
    pub fn build(self) -> CFRAgent<T, I, R> {
        let name = self.name.expect("name is required");
        let player_idx = self.player_idx.expect("player_idx is required");
        let cfr_states = self.cfr_states.expect("cfr_states is required");
        let traversal_set = self.traversal_set.expect("traversal_set is required");
        let iter_gen_config = self
            .gamestate_iterator_gen_config
            .expect("gamestate_iterator_gen_config is required");
        let action_gen_config = self
            .action_gen_config
            .expect("action_gen_config is required");

        // Use pre-fetched CFR state if available, otherwise get from the vec
        let cfr_state = self.cfr_state.unwrap_or_else(|| {
            cfr_states
                .get(player_idx)
                .cloned()
                .expect("CFR state for player not found")
        });
        let traversal_state = traversal_set.get(player_idx);
        let action_generator = T::new(
            cfr_state.clone(),
            traversal_state.clone(),
            action_gen_config.clone(),
        );
        let gamestate_iterator_gen = I::new(&iter_gen_config, self.depth);

        // Use pre-fetched mapper config if available, otherwise fetch from CFR state.
        let mapper_config = self
            .mapper_config
            .unwrap_or_else(|| *cfr_state.mapper_config());
        let action_index_mapper = ActionIndexMapper::new(mapper_config);

        // Create the RNG - use provided RNG or generate from system entropy
        let rng = self.rng.unwrap_or_else(|| R::from_rng(&mut rand::rng()));

        CFRAgent {
            name,
            cfr_states,
            traversal_set,
            cfr_state,
            traversal_state,
            action_generator,
            gamestate_iterator_gen,
            action_gen_config,
            iter_gen_config,
            action_index_mapper,
            forced_action: self.forced_action,
            depth: self.depth,
            limited_exploration_depth: self.limited_exploration_depth,
            allow_node_mutation: self.allow_node_mutation,
            thread_pool: self.thread_pool,
            rng,
        }
    }

    /// Set a pre-fetched CFR state for this agent.
    /// Used internally by sub-agent construction to avoid re-indexing.
    fn cfr_state(mut self, cfr_state: CFRState) -> Self {
        self.cfr_state = Some(cfr_state);
        self
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

impl<T, I, R> CFRAgent<T, I, R>
where
    T: ActionGenerator + 'static,
    I: GameStateIteratorGen + 'static,
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

    /// Returns a reference to this agent's CFR states.
    pub fn cfr_states(&self) -> &[CFRState] {
        &self.cfr_states
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
    fn compute_reward(
        game_state: &GameState,
        action: &AgentAction,
        rng: &mut R,
        ctx: &ComputeRewardContext<'_, T, I>,
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
        let iter_config = ctx.iter_gen_config.clone();
        let action_config = ctx.action_gen_config.clone();
        let cached_mapper_config = *ctx.action_index_mapper.config();

        // Build directly into Vec<Box<dyn Agent>> to avoid an intermediate Vec
        let mut agents: Vec<Box<dyn Agent>> = Vec::with_capacity(num_agents);
        for (i, cfr_state_i) in ctx.cfr_states.iter().cloned().enumerate() {
            let sub_rng = R::from_rng(rng);
            let mut builder = CFRAgentBuilder::<T, I, R>::new()
                .name("CFRAgent-sub")
                .player_idx(i)
                .cfr_state(cfr_state_i)
                .mapper_config(cached_mapper_config)
                .gamestate_iterator_gen_config_arc(iter_config.clone())
                .action_gen_config_arc(action_config.clone())
                .cfr_states_arc(ctx.cfr_states.clone())
                .traversal_set(forked_traversal_set.clone())
                .depth(sub_depth)
                .rng(sub_rng);

            // Note: we intentionally do NOT propagate the thread pool to sub-agents.
            // Only the root agent parallelizes explore_all_actions. Sub-agents run
            // sequentially to avoid lock contention on the shared CFR state.

            if let Some(limited_depth) = ctx.limited_exploration_depth {
                builder = builder.limited_exploration_depth(limited_depth);
            }

            if i == player_idx as usize {
                builder = builder.forced_action((*action).clone());
            }

            agents.push(Box::new(builder.build()) as Box<dyn Agent>);
        }

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state.clone())
            .agents(agents)
            .cfr_context_arc(
                ctx.cfr_states.clone(),
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
        // Determine validation mode based on depth
        let mode = if self.depth >= self.limited_exploration_depth.unwrap_or(usize::MAX) {
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

        let num_iterations = self.gamestate_iterator_gen.num_iterations();
        let target_node_idx = self.target_node_idx().unwrap();

        // Pre-allocate rewards vec and reuse across iterations to avoid
        // repeated Vec allocations.
        let mut rewards: Vec<f32> = vec![invalid_action_penalty; NUM_ACTION_INDICES];

        if let Some(pool) = &self.thread_pool {
            let num_actions = indexed_actions.len();
            let total_tasks = num_iterations * num_actions;

            // Pre-generate one RNG per task from the parent RNG.
            // Each parallel task gets its own RNG — no sharing, no locking.
            let task_rngs: Vec<R> = (0..total_tasks)
                .map(|_| R::from_rng(&mut self.rng))
                .collect();

            // Build context struct from self fields for Sync closure capture.
            // CFRAgent is not Sync (StdRng is not Sync), but all context fields are.
            let ctx = ComputeRewardContext::<T, I> {
                traversal_set: &self.traversal_set,
                traversal_state: &self.traversal_state,
                cfr_states: &self.cfr_states,
                iter_gen_config: &self.iter_gen_config,
                action_gen_config: &self.action_gen_config,
                action_index_mapper: &self.action_index_mapper,
                depth: self.depth,
                limited_exploration_depth: self.limited_exploration_depth,
                allow_node_mutation: self.allow_node_mutation,
            };

            // Run ALL iterations × actions in parallel.
            // Results are ordered by task_id (iter_idx * num_actions + action_pos)
            // because Rayon's indexed parallel iterators preserve ordering.
            let results: Vec<(usize, f32)> = pool.install(|| {
                task_rngs
                    .into_par_iter()
                    .enumerate()
                    .map(|(task_id, mut rng)| {
                        let action_pos = task_id % num_actions;
                        let (action, reward_idx) = &indexed_actions[action_pos];

                        debug_assert!(
                            *reward_idx < NUM_ACTION_INDICES,
                            "Action index {} should be less than number of potential actions {}",
                            reward_idx,
                            NUM_ACTION_INDICES
                        );

                        let reward = Self::compute_reward(game_state, action, &mut rng, &ctx);
                        (*reward_idx, reward)
                    })
                    .collect()
            });

            // Apply regret updates sequentially, grouped by iteration.
            // Results are ordered by task_id (iter_idx * num_actions + action_pos),
            // so we can process them in chunks of num_actions — O(n * actions).
            for chunk in results.chunks(num_actions) {
                rewards.fill(invalid_action_penalty);
                for &(reward_idx, reward) in chunk {
                    rewards[reward_idx] = reward;
                }
                self.update_regret_at_node(target_node_idx, &rewards);
            }
        } else {
            // Sequential path — used when no thread pool is configured.
            // Process each iteration independently and update regret immediately.
            // This helps the regret matcher converge better than averaging rewards
            // across all iterations before updating.
            let ctx = ComputeRewardContext::<T, I> {
                traversal_set: &self.traversal_set,
                traversal_state: &self.traversal_state,
                cfr_states: &self.cfr_states,
                iter_gen_config: &self.iter_gen_config,
                action_gen_config: &self.action_gen_config,
                action_index_mapper: &self.action_index_mapper,
                depth: self.depth,
                limited_exploration_depth: self.limited_exploration_depth,
                allow_node_mutation: self.allow_node_mutation,
            };

            for _ in 0..num_iterations {
                // Reset to penalty for all actions, valid ones get overwritten
                rewards.fill(invalid_action_penalty);

                for (action, reward_idx) in &indexed_actions {
                    debug_assert!(
                        *reward_idx < rewards.len(),
                        "Action index {} should be less than number of potential actions {}",
                        reward_idx,
                        rewards.len()
                    );
                    debug_assert!(
                        seen_indices.contains(*reward_idx),
                        "Action {:?} mapped to index {} which should be in seen_indices",
                        action,
                        reward_idx
                    );

                    rewards[*reward_idx] =
                        Self::compute_reward(game_state, action, &mut self.rng, &ctx);
                }

                // Update regret immediately for this game state
                self.update_regret_at_node(target_node_idx, &rewards);
            }
        }
    }
}

impl<T, I, R> Agent for CFRAgent<T, I, R>
where
    T: ActionGenerator + 'static,
    I: GameStateIteratorGen + 'static,
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

            // Use ActionPicker to select an action based on the regret matcher
            // We use with_node_data to avoid cloning the entire NodeData
            let possible_actions = self.action_generator.gen_possible_actions(game_state);
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
        BasicCFRActionGenerator, DepthBasedIteratorGen, DepthBasedIteratorGenConfig, TraversalSet,
    };

    use super::*;

    /// Helper to create CFR states for all players from a game state.
    fn make_cfr_states(game_state: &GameState) -> Vec<CFRState> {
        (0..game_state.num_players)
            .map(|_| CFRState::new(game_state.clone()))
            .collect()
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
        let cfr_states = make_cfr_states(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        let cfr_agent = Box::new(
            CFRAgentBuilder::<BasicCFRActionGenerator, DepthBasedIteratorGen>::new()
                .name("CFRAgent-player1")
                .player_idx(1)
                .cfr_states(cfr_states.clone())
                .traversal_set(traversal_set.clone())
                .gamestate_iterator_gen_config(DepthBasedIteratorGenConfig::new(vec![1]))
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
            .cfr_context(cfr_states, traversal_set, true)
            .build()
            .unwrap();

        // This should not panic - the CFR agent properly handles
        // mixed-agent simulations
        sim.run(&mut rng);
    }

    #[test]
    fn test_create_agent() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(3, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let cfr_states = make_cfr_states(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);
        let _ = CFRAgentBuilder::<BasicCFRActionGenerator, DepthBasedIteratorGen>::new()
            .name("CFRAgent-test")
            .player_idx(0)
            .cfr_states(cfr_states)
            .traversal_set(traversal_set)
            .gamestate_iterator_gen_config(DepthBasedIteratorGenConfig::new(vec![1]))
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
        let cfr_states = make_cfr_states(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);
        let agents: Vec<Box<dyn Agent>> = (0..num_agents)
            .map(|i| {
                Box::new(
                    CFRAgentBuilder::<BasicCFRActionGenerator, DepthBasedIteratorGen>::new()
                        .name(format!("CFRAgent-test-{i}"))
                        .player_idx(i)
                        .cfr_states(cfr_states.clone())
                        .traversal_set(traversal_set.clone())
                        .gamestate_iterator_gen_config(DepthBasedIteratorGenConfig::new(vec![2, 1]))
                        .action_gen_config(())
                        .build(),
                ) as Box<dyn Agent>
            })
            .collect();

        let mut rng = rand::rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .cfr_context(cfr_states, traversal_set, true)
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
        let cfr_states = make_cfr_states(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        let iter_config = DepthBasedIteratorGenConfig::new(vec![1]);

        // Create two agents with the same shared states
        let agent0 = CFRAgentBuilder::<BasicCFRActionGenerator, DepthBasedIteratorGen>::new()
            .name("Agent0")
            .player_idx(0)
            .cfr_states(cfr_states.clone())
            .traversal_set(traversal_set.clone())
            .gamestate_iterator_gen_config(iter_config.clone())
            .action_gen_config(())
            .build();

        let agent1 = CFRAgentBuilder::<BasicCFRActionGenerator, DepthBasedIteratorGen>::new()
            .name("Agent1")
            .player_idx(1)
            .cfr_states(cfr_states)
            .traversal_set(traversal_set)
            .gamestate_iterator_gen_config(iter_config)
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

        let cfr_states = make_cfr_states(&game_state);
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
                    CFRAgentBuilder::<BasicCFRActionGenerator, DepthBasedIteratorGen>::new()
                        .name(format!("CFRAgent-par-{i}"))
                        .player_idx(i)
                        .cfr_states(cfr_states.clone())
                        .traversal_set(traversal_set.clone())
                        .gamestate_iterator_gen_config(DepthBasedIteratorGenConfig::new(vec![2, 1]))
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
            .cfr_context(cfr_states, traversal_set, true)
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

        let cfr_states = make_cfr_states(&game_state);
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
                    CFRAgentBuilder::<BasicCFRActionGenerator, DepthBasedIteratorGen>::new()
                        .name(format!("CFRAgent-par-{i}"))
                        .player_idx(i)
                        .cfr_states(cfr_states.clone())
                        .traversal_set(traversal_set.clone())
                        .gamestate_iterator_gen_config(DepthBasedIteratorGenConfig::new(vec![2, 1]))
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
            .cfr_context(cfr_states.clone(), traversal_set, true)
            .build()
            .unwrap();

        sim.run(&mut rng);

        // After running, the CFR tree should have grown beyond just the root node
        for state in &cfr_states {
            assert!(
                state.arena().len() > 1,
                "CFR tree should have grown during simulation"
            );
        }
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
        let cfr_states = make_cfr_states(&game_state);
        let traversal_set = TraversalSet::new(game_state.num_players);

        // Use the configurable action generator (same as preflop chart post-flop)
        let action_config = ConfigurableActionConfig::default();

        let mut agent =
            CFRAgentBuilder::<ConfigurableActionGenerator, DepthBasedIteratorGen, StdRng>::new()
                .name("TestCFRAgent")
                .player_idx(1)
                .cfr_states(cfr_states.clone())
                .traversal_set(traversal_set.clone())
                .gamestate_iterator_gen_config(DepthBasedIteratorGenConfig::new(vec![24, 3, 1]))
                .action_gen_config(action_config)
                .limited_exploration_depth(3)
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
}
