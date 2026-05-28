//! The CFR exploration engine: [`CFRAgent`] and its async tree walk.
//!
//! `explore_all_actions` is the core loop. For the agent's decision node it
//! runs a budget-driven series of **waves** (mini-batch PCFR+). Each wave fans
//! out `wave_width × (non-pruned) actions` reward samples against the same
//! pre-wave strategy snapshot, averages the per-slot samples
//! ([`wave_mean`]), and applies exactly one atomic regret update. Rewards are
//! produced either by recursing into a sub-simulation (when the budget
//! returns `NextStep::Wave`) or by the fast-forward Monte-Carlo path (when
//! it returns `NextStep::FastForward`). `wave_width == 1` reproduces the prior
//! single-sample-per-action behavior (one sample per slot → mean == sample).
//!
//! Concurrency follows a "try-acquire-or-inline" model bounded by a shared
//! [`InFlightLimiter`](super::super::InFlightLimiter): at the shallow spawn
//! frontier a sample is `tokio::spawn`ed when a permit is free and otherwise
//! runs inline, so recursion is deadlock-free at any depth. Regret-based
//! pruning (Brown & Sandholm, 2015) skips actions driven to zero strategy
//! weight, re-probing periodically; the prune decision is computed once per
//! wave so every sample in the wave agrees.
//!
//! Stopping is cooperative and checked only at the wave boundary: the
//! [`Budget`](super::super::Budget) decides what each wave does (recursive
//! `Wave`, one-shot `FastForward`, `Stop`, or `StartTimer` to arm a deadline),
//! and a lock-free `Arc<AtomicBool>` stop flag is the single cross-task stop
//! signal. A stop never leaves a partial regret update — it simply means
//! fewer completed waves, and `act` picks from whatever regret has
//! accumulated.

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use little_sorry::{PcfrPlusRegretMatcher, RegretMinimizer};
use rand::SeedableRng;
use rand::rngs::StdRng;
use smallvec::SmallVec;
use tracing::event;

use crate::arena::{
    Agent, GameState, HoldemSimulationBuilder, action::AgentAction, game_state::Round,
};

use super::super::{
    ActionIndexMapper, Budget, CFRState, ExplorationStats, InFlightLimiter, NUM_ACTION_INDICES,
    NextStep, NodeData, PlayerData, TraversalSet, TraversalState, action_bit_set::ActionBitSet,
    action_generator::ActionGenerator, action_validator::validate_actions,
};

/// Why `explore_all_actions` exited. Five reasons end the wave loop
/// (Deadline, BudgetStop, BudgetStartTimer, FastForward, StableStrategy);
/// a sixth (SingleAction) bypasses the loop entirely when there's
/// nothing to explore. The budget tree's `MostRestrictive` composer
/// collapses internal Stop causes, so callers disambiguate by inspecting
/// the emitted field values (final_iterations vs configured cap,
/// last(regret_series) vs epsilon, elapsed vs deadline).
#[derive(Copy, Clone, Debug)]
enum StopCause {
    /// The lock-free stop atomic flipped — the deadline timer fired or
    /// an external cancellation was requested.
    Deadline,
    /// The budget tree returned `NextStep::Stop` or `NextStep::Pass`.
    BudgetStop,
    /// `NextStep::StartTimer` arrived after the timer was already armed
    /// (the engine treats this as Stop). Degenerate but observable.
    BudgetStartTimer,
    /// The wave loop completed a one-shot `FastForward` step.
    FastForward,
    /// Only one legal action remained after validation; the wave loop
    /// was skipped because there is no decision to learn. The regret
    /// matcher's trivial strategy `[1.0]` over that action is what the
    /// picker will return.
    SingleAction,
    /// Strategy stabilized: L1 distance between consecutive waves'
    /// strategies stayed below `EARLY_EXIT_EPSILON` for
    /// `EARLY_EXIT_STABLE_ITERS` consecutive waves. Modeled on
    /// Stockfish's stability-based time management — if the answer
    /// hasn't moved in a while, more iterations won't help.
    StableStrategy,
}

impl std::fmt::Display for StopCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            StopCause::Deadline => "deadline",
            StopCause::BudgetStop => "budget_stop",
            StopCause::BudgetStartTimer => "budget_start_timer",
            StopCause::FastForward => "fast_forward",
            StopCause::SingleAction => "single_action",
            StopCause::StableStrategy => "stable_strategy",
        };
        f.write_str(s)
    }
}

/// Strategy-stability early exit thresholds. See Tier-1 Experiment 3 in
/// `docs/superpowers/specs/2026-05-28-cfr-tier1-tuning-experiments.md`.
///
/// `MIN_ITERS` protects against pre-warmup noise: the first few waves
/// can hit a temporary plateau before real exploration kicks in.
/// `STABLE_ITERS` requires the L1 strategy delta to stay below
/// `EPSILON` for that many consecutive waves before we declare
/// convergence and stop the loop.
const EARLY_EXIT_MIN_ITERS: usize = 4;
const EARLY_EXIT_STABLE_ITERS: u32 = 3;
const EARLY_EXIT_EPSILON: f32 = 0.01;

use super::builder::CFRAgentBuilder;
use super::fast_forward::{
    fast_forward_advance_betting, fast_forward_apply_action, fast_forward_distribute_pot,
    fast_forward_enumerate_showdowns, fast_forward_run_to_showdown,
    fast_forward_sample_flop_enumerate_runout,
};
use super::reward_context::ComputeRewardContext;

/// Write the per-slot mean over the samples a wave produced into `out`. Slots
/// with zero samples (pruned / never explored) get `penalty`, matching invariant
/// #1's full vector. Writes into a caller-owned buffer so the wave loop reuses
/// one allocation-free array across every wave.
pub(super) fn wave_mean_into(out: &mut [f32], sums: &[f32], counts: &[u32], penalty: f32) {
    for (o, (&s, &c)) in out.iter_mut().zip(sums.iter().zip(counts)) {
        *o = if c > 0 { s / c as f32 } else { penalty };
    }
}

/// A CFR (Counterfactual Regret Minimization) agent for poker.
///
/// This agent uses CFR to compute optimal strategies by exploring the game tree
/// and learning from regret. It maintains state across simulations via shared
/// CFR states.
///
/// # Type Parameters
/// * `T` - The action generator type (implements `ActionGenerator`)
pub struct CFRAgent<T>
where
    T: ActionGenerator,
{
    pub(super) name: Cow<'static, str>,
    pub(super) traversal_set: TraversalSet,
    pub(super) traversal_state: TraversalState,
    pub(super) cfr_state: CFRState,
    pub(super) action_generator: T,
    pub(super) action_gen_config: Arc<T::Config>,
    pub(super) action_index_mapper: ActionIndexMapper,
    pub(super) forced_action: Option<AgentAction>,
    pub(super) depth: usize,
    pub(super) allow_node_mutation: bool,
    pub(super) limiter: InFlightLimiter,
    pub(super) budget: Arc<dyn Budget>,
    pub(super) stop: Arc<AtomicBool>,
}

/// Spawn a tokio task that flips `stop` to `true` after `duration`. The
/// returned `AbortOnDrop` wrapper aborts the task when dropped, so a fast
/// `act` leaves no lingering timer.
pub(super) fn spawn_stop_timer(
    duration: std::time::Duration,
    stop: Arc<AtomicBool>,
) -> super::AbortOnDrop {
    super::AbortOnDrop(tokio::spawn(async move {
        tokio::time::sleep(duration).await;
        stop.store(true, Ordering::Relaxed);
    }))
}

impl<T> CFRAgent<T>
where
    T: ActionGenerator + Send + 'static,
    T::Config: Send + Sync,
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
    ///    the budget returns `NextStep::FastForward` (typically deep in the
    ///    tree). It applies the candidate action on a cloned state and assumes
    ///    every *subsequent* action (by any player, in any round) is a check
    ///    or call. Remaining community cards are dealt and the pot is
    ///    distributed with simple one-pot showdown logic. This throws away the
    ///    mutual-best-response signal deep in the tree, but returns a
    ///    realistic showdown reward and has bounded cost.
    async fn compute_reward(
        game_state: &GameState,
        action: &AgentAction,
        ctx: &ComputeRewardContext<T>,
    ) -> f32 {
        if ctx.fast_forward {
            let player_idx = ctx.traversal_state.player_idx() as usize;
            // Fast-forward is pure CPU on a cloned GameState — no awaiting.
            Self::compute_reward_fast_forward(game_state, action, player_idx)
        } else {
            Self::compute_reward_recursive(game_state, action, ctx).await
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
    async fn compute_reward_recursive(
        game_state: &GameState,
        action: &AgentAction,
        ctx: &ComputeRewardContext<T>,
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

        let action_config = ctx.action_gen_config.clone();
        let cached_mapper_config = *ctx.action_index_mapper.config();

        // All sub-agents share the same CFR state (single shared tree).
        let shared_cfr_state = ctx.cfr_state.clone();

        let mut agents: Vec<Box<dyn Agent>> = Vec::with_capacity(num_agents);
        for i in 0..num_agents {
            let mut builder = CFRAgentBuilder::<T>::new()
                .name("CFRAgent-sub")
                .player_idx(i)
                .cfr_state(shared_cfr_state.clone())
                .mapper_config(cached_mapper_config)
                .action_gen_config_arc(action_config.clone())
                .traversal_set(forked_traversal_set.clone())
                .depth(sub_depth)
                .limiter(ctx.limiter.clone())
                .budget(ctx.budget.clone())
                .stop_flag(ctx.stop.clone());

            // Sub-agents share the SAME in-flight limiter, budget, and stop
            // flag as the root, so adaptive `try_acquire`-or-inline spawning
            // at deeper levels draws from one global bound instead of
            // multiplying into oversubscription, and budget/stop signals
            // reach every recursive level.

            if i == player_idx as usize {
                builder = builder.forced_action((*action).clone());
            }

            agents.push(Box::new(builder.build()) as Box<dyn Agent>);
        }

        // Seed the sub-simulation's RNG from the thread-local generator.
        let sub_sim_rng = StdRng::from_rng(&mut rand::rng());
        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state.clone())
            .agents(agents)
            .cfr_context(
                shared_cfr_state,
                forked_traversal_set,
                ctx.allow_node_mutation,
            )
            .build_with_rng(sub_sim_rng)
            .unwrap();

        sim.run().await;

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
    ) -> f32 {
        // Thread-local RNG: fn is sync so this never crosses an await.
        let mut rng = rand::rng();
        let mut gs = game_state.clone();
        fast_forward_apply_action(&mut gs, action);

        // Check if at most one player can contest the pot after the action.
        let contenders = gs.player_active.count() + gs.player_all_in.count();
        if contenders <= 1 {
            fast_forward_run_to_showdown(&mut gs, &mut rng);
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
                fast_forward_run_to_showdown(&mut gs, &mut rng);
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
            fast_forward_sample_flop_enumerate_runout(&gs, player_idx, &mut rng)
        }
    }

    pub(super) fn target_node_idx(&self) -> Option<usize> {
        let (from_node_idx, from_child_idx) = self.traversal_state.get_position();
        self.cfr_state.get_child(from_node_idx, from_child_idx)
    }

    /// Ensure that the target node is created and that it is a player node.
    ///
    /// Uses `CFRState::ensure_child` which handles the case where different bet
    /// amounts map to the same index but lead to different outcomes. If a node
    /// exists with a different type and `allow_node_mutation` is true, it will
    /// be updated to a Player node.
    pub(super) fn ensure_target_node(&self) -> usize {
        // Get all traversal state fields in a single lock acquisition
        let (node_idx, chosen_child_idx, player_idx) = self.traversal_state.get_all();

        let expected_data = NodeData::Player(PlayerData {
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

    pub(super) fn ensure_regret_matcher(&mut self) {
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
                    *data = NodeData::Player(PlayerData {
                        regret_matcher: Some(regret_matcher),
                        player_idx: self.traversal_state.player_idx(),
                    });
                }
            })
            .unwrap();
    }

    /// Run the budget-driven wave loop for this node.
    ///
    /// Stopping is cooperative via `self.stop` (a lock-free `Arc<AtomicBool>`
    /// shared with every recursive sub-agent). The `Budget` decides what each
    /// wave does (recursive wave, fast-forward, stop, or arm a deadline timer);
    /// when the budget asks for `StartTimer`, the engine spawns a tokio task
    /// that flips `self.stop` after the requested duration, so every recursive
    /// level sees the stop at its next wave boundary.
    pub async fn explore_all_actions(&mut self, game_state: &GameState) {
        let raw_actions = self.action_generator.gen_possible_actions(game_state);
        let validated_actions = validate_actions(raw_actions, game_state);

        // Filter actions to ensure each maps to a unique index.
        // Different bet amounts can map to the same index due to the logarithmic
        // mapping (only 49 slots for raises). We keep the first action for each index.
        // Using ActionBitSet for O(1) operations with no heap allocation.
        // Pre-compute action indices once to avoid repeated action_to_idx calls.
        let mut seen_indices = ActionBitSet::new();
        let indexed_actions: SmallVec<[(AgentAction, usize); 8]> = validated_actions
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

        // Single-action shortcut. When only one action survives validation
        // the strategy is forced ([1.0] over that action) and a wave loop
        // would only waste budget — there's no alternative to regret
        // against. Empirically this fires for a large fraction of root
        // acts in heads-up / short-stack scenarios where the betting line
        // collapses to a single legal move. Emit a diag event so the
        // analyzer can still count these in the cross-tab and
        // actions-considered histogram.
        if indexed_actions.len() == 1 {
            if tracing::event_enabled!(target: "cfr_diag", tracing::Level::TRACE) {
                let nodes = self.cfr_state.node_count() as u64;
                let empty: &[f32] = &[];
                tracing::event!(
                    target: "cfr_diag",
                    tracing::Level::TRACE,
                    depth = self.depth as u64,
                    stop_cause = %StopCause::SingleAction,
                    final_iterations = 0u64,
                    final_elapsed_us = 0u64,
                    nodes_touched_start = nodes,
                    nodes_touched_end = nodes,
                    timer_armed = false,
                    actions_considered = 1u64,
                    regret_series = ?empty,
                );
            }
            return;
        }

        // Penalty for invalid actions - using player's starting stack since
        // losing your whole stack is the worst outcome.
        let invalid_action_penalty =
            -(game_state.starting_stacks[self.traversal_state.player_idx() as usize]);

        let target_node_idx = self.target_node_idx().unwrap();

        // Per-slot wave accumulators, reused across waves to avoid repeated Vec
        // allocations. Each wave sums every sample landing in a slot and counts
        // them; `wave_mean` then averages (slots with zero samples — pruned or
        // never explored — fall back to `invalid_action_penalty`).
        // Stack-allocated, fixed-size accumulators (NUM_ACTION_INDICES is a
        // compile-time const) reused across waves — no per-node heap allocation.
        let mut sums = [0.0f32; NUM_ACTION_INDICES];
        let mut counts = [0u32; NUM_ACTION_INDICES];
        // Reused per-wave reward buffer (the averaged vector handed to the regret
        // matcher), also stack-allocated and reused — no per-wave allocation.
        let mut rewards = [0.0f32; NUM_ACTION_INDICES];

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

        // Coarse spawn frontier. We only fan reward computations out to tokio
        // tasks while `depth < SPAWN_FRONTIER_DEPTH`; deeper nodes recurse
        // inline. The per-action spawn cost (a `try_acquire` atomic on the
        // shared semaphore + a `tokio::spawn` + a `GameState` clone) is paid
        // once per spawned node, so spawning at *every* depth makes it scale
        // with total node count and dominates on large trees. Confining
        // spawning to the top few levels turns each spawned task into a whole
        // subtree (work amortized like a work-stealing unit) while keeping that
        // overhead O(shallow nodes). Iteration × action fan-out at depths
        // 0..SPAWN_FRONTIER_DEPTH still provides ample parallelism. (Measured:
        // K=2 matches spawning at every depth while allocating far fewer tasks;
        // K=1 under-parallelizes.)
        // Tuning note: the poker action tree is highly unbalanced (a raise
        // branch explodes while fold is trivial), so spawning only at the top
        // 1-2 levels left the deep fast-forward enumeration — the bulk of the
        // work — running inline and unstealable, idling most workers (measured
        // ~4 of 16 cores busy at K=2). Pushing the frontier one level deeper
        // turns each depth-2 node's fan-out of fast-forward leaves into
        // stealable tasks the multi-thread scheduler load-balances.
        const SPAWN_FRONTIER_DEPTH: usize = 3;

        let (initial_active, initial_updates) = self.cfr_state.get_pruning_info(target_node_idx);
        let can_prune = indexed_actions.len() > 2 && initial_updates >= PRUNE_WARMUP;

        // ── Budget-driven ordered waves: try-acquire-or-inline spawning ──
        //
        // The loop runs WAVES until the budget says stop (or the lock-free
        // `stop` flag flips). A wave fans out `wave_width` reward samples for
        // every non-pruned action against the same pre-wave strategy snapshot,
        // averages the per-slot samples (`wave_mean`), and applies exactly one
        // atomic PCFR+ regret update (invariant #1 — mini-batch PCFR+). At
        // `wave_width == 1` each slot gets one sample → mean == sample → the
        // prior single-sample behavior exactly. Each sample's `compute_reward`
        // is fanned out: a permit is immediately available → `spawn` the
        // subtree concurrently; none → run it inline. Acquisition is always
        // `try_acquire` (never blocking), so recursion at any depth is
        // deadlock-free (invariant #3).
        //
        // Regret-based pruning is preserved: the active set starts from the
        // initial read and is refreshed after each reprobe wave, gated
        // identically on `can_prune` / `updates_since_warmup` / the `len() > 2`
        // guard. The prune decision is computed ONCE per wave (below) so every
        // sample in the wave agrees.
        let mut active_actions = initial_active;
        let mut updates_since_warmup = initial_updates;

        // Wall-clock start for `Budget` accounting. `std::time::Instant` (not
        // tokio's) so the elapsed value is meaningful regardless of runtime.
        let started = std::time::Instant::now();

        // Node-local convergence signal from the *previous* completed update,
        // surfaced to the budget at the next wave boundary. `None` until the
        // first update lands.
        let mut latest_avg_regret: Option<f32> = None;

        // Completed waves at this node (= matcher updates). This is what the
        // budget's `iterations` signal reports.
        let mut iter_idx: u64 = 0;

        // ── Strategy-stability early exit ──
        //
        // Track the L1 distance between consecutive waves' strategies. If
        // the strategy stops moving for STABLE_ITERS consecutive waves,
        // further iterations are unlikely to help — bail and return
        // budget to the rest of the simulation. Stack-allocated buffers,
        // no heap alloc.
        let mut early_exit_prev_strategy = [0.0f32; NUM_ACTION_INDICES];
        let mut early_exit_curr_strategy = [0.0f32; NUM_ACTION_INDICES];
        let mut early_exit_stable_count: u32 = 0;
        let mut early_exit_has_prev = false;

        // Sub-agents inherit "timer already armed" — only the root agent at
        // depth 0 ever sees StartTimer from the Deadline leaf.
        let mut timer_armed = self.depth > 0;
        // Holds the spawned timer's abort guard so it's cancelled when this fn
        // returns. Only ever Some at the root after a successful arm.
        let mut _timer_guard: Option<super::AbortOnDrop> = None;

        // ── Diagnostics: per-act ExplorationSummary ──
        //
        // Entire diagnostic path is gated by `event_enabled!`. When no
        // subscriber is interested, `diag_on == false`, the Vec stays empty
        // (capacity 0, no allocation), and the per-wave push is skipped.
        // When enabled, the Vec is pre-sized to a conservative estimate (32)
        // to avoid reallocs in common configurations.
        let diag_on = tracing::event_enabled!(target: "cfr_diag", tracing::Level::TRACE);
        let diag_nodes_touched_start: u64 = if diag_on {
            self.cfr_state.node_count() as u64
        } else {
            0
        };
        let mut diag_regret_series: Vec<f32> = if diag_on {
            Vec::with_capacity(32)
        } else {
            Vec::new()
        };
        // Default; overwritten by every break path below. If you add a new
        // break without tagging it, the emitted stop_cause will incorrectly
        // say "budget_stop".
        let mut diag_stop_cause: StopCause = StopCause::BudgetStop;

        loop {
            // ── Budget / stop check at the WAVE BOUNDARY ──
            //
            // INVARIANT #1: the pre-wave stop check sits before any reward
            // computation for this wave, so breaking here can never leave a
            // partial reward vector — it simply means fewer completed waves.
            // `act` then picks from whatever regret has accumulated, returning
            // the best-known action. Budget exhaustion and stop are NOT errors.
            let stats = ExplorationStats {
                elapsed: started.elapsed(),
                iterations: iter_idx,
                nodes_touched: self.cfr_state.node_count() as u64,
                depth: self.depth,
                avg_regret: latest_avg_regret,
                timer_armed,
            };
            if self.stop.load(Ordering::Relaxed) {
                if diag_on {
                    diag_stop_cause = StopCause::Deadline;
                }
                break;
            }

            let (wave_width, fast_forward) = match self.budget.next_step(&stats) {
                NextStep::Stop | NextStep::Pass => {
                    if diag_on {
                        diag_stop_cause = StopCause::BudgetStop;
                    }
                    break;
                }
                NextStep::StartTimer { duration } if !timer_armed => {
                    debug_assert!(
                        self.depth == 0,
                        "NextStep::StartTimer should only arrive at the root (depth 0). \
                         Sub-agents inherit `timer_armed = true`, so a StartTimer here means \
                         a budget returned StartTimer at depth > 0, which is unsupported by \
                         the current engine. If you want per-depth timers, the engine needs \
                         to grow per-depth timer slots."
                    );
                    _timer_guard = Some(spawn_stop_timer(duration, self.stop.clone()));
                    timer_armed = true;
                    continue;
                }
                NextStep::StartTimer { .. } => {
                    if diag_on {
                        diag_stop_cause = StopCause::BudgetStartTimer;
                    }
                    break;
                }
                NextStep::Wave { width } => (width, false),
                NextStep::FastForward => (1, true),
            };

            // Build the per-wave ComputeRewardContext. `fast_forward` varies
            // per iteration so we rebuild ctx each wave; every field is an
            // `Arc`-style handle, so this is cheap.
            let ctx = ComputeRewardContext::<T> {
                traversal_set: self.traversal_set.clone(),
                traversal_state: self.traversal_state.clone(),
                cfr_state: self.cfr_state.clone(),
                action_gen_config: self.action_gen_config.clone(),
                action_index_mapper: self.action_index_mapper.clone(),
                limiter: self.limiter.clone(),
                budget: self.budget.clone(),
                stop: self.stop.clone(),
                depth: self.depth,
                fast_forward,
                allow_node_mutation: self.allow_node_mutation,
            };

            // Decide whether to prune this wave. On reprobe waves (every
            // REPROBE_INTERVAL-th), explore all actions. Computed ONCE per wave
            // so all `wave_width` samples make the same prune decision and the
            // averaged vector is a mean over a consistent active set.
            //
            // The len() > 2 guard is required here even though `can_prune`
            // already checks it. The second disjunct (`updates_since_warmup
            // >= PRUNE_WARMUP`) handles nodes that cross the warmup
            // threshold mid-call, but it does not carry the action-count
            // check. Without the outer guard, 2-action nodes could have one
            // action pruned on 75% of waves, collapsing to a fixed
            // policy with no exploration.
            let prune_this_iter =
                indexed_actions.len() > 2 && (can_prune || updates_since_warmup >= PRUNE_WARMUP);
            let is_reprobe = iter_idx.is_multiple_of(REPROBE_INTERVAL as u64);
            let skip_pruned = prune_this_iter && !is_reprobe;

            // INVARIANT #1: accumulate the COMPLETE per-slot sample sums/counts
            // for this wave before updating regret. Pruned (and never-sampled)
            // slots stay at count 0 and fall back to `invalid_action_penalty`
            // via `wave_mean`, never omitted.
            sums.fill(0.0);
            counts.fill(0);

            let mut set: tokio::task::JoinSet<(usize, f32)> = tokio::task::JoinSet::new();
            let mut inline: Vec<(usize, f32)> = Vec::new();

            // Only spawn at the shallow frontier. Below it, recurse inline with
            // no semaphore traffic, no spawn, and no per-sample clone. Build the
            // shared `Arc<GameState>` snapshot once (cloned once, then cheap Arc
            // clones per spawned task) and only when we may actually spawn.
            let spawn_here = self.depth < SPAWN_FRONTIER_DEPTH;
            let gs_arc = spawn_here.then(|| std::sync::Arc::new(game_state.clone()));

            // `wave_width` samples × each active action.
            for _sample in 0..wave_width {
                for (action, reward_idx) in &indexed_actions {
                    let reward_idx = *reward_idx;

                    // Regret-based pruning: skip actions with zero strategy
                    // weight. Pruned actions take no sample and keep the penalty
                    // via `wave_mean` (count stays 0).
                    if skip_pruned && !active_actions.contains(reward_idx) {
                        event!(
                            tracing::Level::TRACE,
                            action_idx = reward_idx,
                            wave = iter_idx,
                            "RBP: skipping pruned action"
                        );
                        continue;
                    }

                    debug_assert!(
                        reward_idx < sums.len(),
                        "Action index {} should be less than number of potential actions {}",
                        reward_idx,
                        sums.len()
                    );

                    let action = action.clone();

                    // At the frontier, spawn the subtree if a permit is free
                    // (`try_acquire` never blocks → recursion stays
                    // deadlock-free, invariant #3). Below the frontier, or when
                    // saturated, compute inline.
                    if let Some(gs_arc) = &gs_arc
                        && let Ok(permit) = ctx.limiter.clone().try_acquire_owned()
                    {
                        let ctx = ctx.clone();
                        let gs = gs_arc.clone();
                        set.spawn(async move {
                            // Permit is held for the whole subtree's lifetime.
                            let _permit = permit;
                            let r = CFRAgent::<T>::compute_reward(&gs, &action, &ctx).await;
                            (reward_idx, r)
                        });
                        continue;
                    }
                    let r = Self::compute_reward(game_state, &action, &ctx).await;
                    inline.push((reward_idx, r));
                }
            }

            // Accumulate inline samples, then join all spawned handles, so the
            // complete per-slot sums/counts are ready before the single update.
            for (idx, r) in inline.drain(..) {
                sums[idx] += r;
                counts[idx] += 1;
            }
            while let Some(joined) = set.join_next().await {
                match joined {
                    Ok((idx, r)) => {
                        sums[idx] += r;
                        counts[idx] += 1;
                    }
                    Err(join_err) => {
                        // A `JoinError` here means a spawned exploration task
                        // panicked — a bug to surface, not swallow. Re-raise
                        // the original panic on this thread.
                        if join_err.is_panic() {
                            std::panic::resume_unwind(join_err.into_panic());
                        } else {
                            panic!("CFR exploration task failed to join: {join_err}");
                        }
                    }
                }
            }

            // INVARIANT #1 — discard a stopped wave. The lock-free `stop`
            // flag can flip mid-wave (after the boundary check, while samples
            // are in flight). Without aborting in-flight `compute_reward`
            // calls we wait for them to finish on their own, but if `stop`
            // fired during the wave we drop the whole wave — no
            // `update_regret_at_node`, no `iter_idx` increment, no active-set
            // refresh — and break. The next iteration's boundary check would
            // be too late: this wave's samples would already have updated.
            if self.stop.load(Ordering::Relaxed) {
                if diag_on {
                    diag_stop_cause = StopCause::Deadline;
                }
                break;
            }

            // INVARIANT #1: one atomic, complete averaged vector → one update.
            wave_mean_into(&mut rewards, &sums, &counts, invalid_action_penalty);
            self.update_regret_at_node(target_node_idx, &rewards);
            updates_since_warmup += 1;
            iter_idx += 1;
            // Refresh the convergence signal for the next wave's budget check
            // from the regret matrix this update just produced.
            latest_avg_regret = self.cfr_state.node_avg_regret(target_node_idx);
            if diag_on && let Some(r) = latest_avg_regret {
                diag_regret_series.push(r);
            }

            // Strategy-stability early exit. Read the post-update strategy,
            // compare against the snapshot we took after the previous wave,
            // and accumulate a stable-iteration counter. We only consider
            // the gate once the warmup has passed (MIN_ITERS) — the first
            // few waves often look "stable" before real exploration begins.
            if self
                .cfr_state
                .node_current_strategy_into(target_node_idx, &mut early_exit_curr_strategy)
            {
                if early_exit_has_prev && (iter_idx as usize) >= EARLY_EXIT_MIN_ITERS {
                    let mut l1 = 0.0f32;
                    for (a, b) in early_exit_curr_strategy
                        .iter()
                        .zip(early_exit_prev_strategy.iter())
                    {
                        l1 += (a - b).abs();
                    }
                    if l1 < EARLY_EXIT_EPSILON {
                        early_exit_stable_count += 1;
                        if early_exit_stable_count >= EARLY_EXIT_STABLE_ITERS {
                            if diag_on {
                                diag_stop_cause = StopCause::StableStrategy;
                            }
                            break;
                        }
                    } else {
                        early_exit_stable_count = 0;
                    }
                }
                early_exit_prev_strategy.copy_from_slice(&early_exit_curr_strategy);
                early_exit_has_prev = true;
            }

            // After a reprobe wave, refresh the active action set from the
            // updated regret matcher. The len() > 2 guard keeps this consistent
            // with the pruning decision above — there is no point refreshing an
            // active set we will never use.
            if is_reprobe
                && indexed_actions.len() > 2
                && (can_prune || updates_since_warmup >= PRUNE_WARMUP)
            {
                let (new_active, _) = self.cfr_state.get_pruning_info(target_node_idx);
                active_actions = new_active;
            }

            // `FastForward` is a one-shot — fast-forward is deterministic for
            // 0–2 remaining community cards (full enumeration) and samples
            // flops internally for 3 cards; doing it more than once per node
            // yields no new information.
            if fast_forward {
                if diag_on {
                    diag_stop_cause = StopCause::FastForward;
                }
                break;
            }
        }

        if diag_on {
            let elapsed_us = started.elapsed().as_micros() as u64;
            let nodes_touched_end = self.cfr_state.node_count() as u64;
            tracing::event!(
                target: "cfr_diag",
                tracing::Level::TRACE,
                depth = self.depth as u64,
                stop_cause = %diag_stop_cause,
                final_iterations = iter_idx,
                final_elapsed_us = elapsed_us,
                nodes_touched_start = diag_nodes_touched_start,
                nodes_touched_end = nodes_touched_end,
                timer_armed = timer_armed,
                actions_considered = indexed_actions.len() as u64,
                regret_series = ?diag_regret_series.as_slice(),
            );
        }
    }
}

#[cfg(test)]
mod wave_tests {
    use super::wave_mean_into;

    #[test]
    fn wave_mean_averages_only_sampled_slots() {
        // 3 action slots, penalty -100. Slots 0 and 2 each got 2 samples; slot 1 none.
        let penalty = -100.0_f32;
        let sums = [3.0, 0.0, 8.0];
        let counts = [2u32, 0, 2];
        let mut mean = [0.0f32; 3];
        wave_mean_into(&mut mean, &sums, &counts, penalty);
        assert_eq!(mean, [1.5, -100.0, 4.0]);
    }

    #[test]
    fn wave_mean_single_sample_equals_sample() {
        // wave_width == 1: each sampled slot has count 1, so mean == sample.
        let penalty = -7.0_f32;
        let sums = [5.0, -2.0, 0.0];
        let counts = [1u32, 1, 0];
        let mut mean = [0.0f32; 3];
        wave_mean_into(&mut mean, &sums, &counts, penalty);
        assert_eq!(mean, [5.0, -2.0, -7.0]);
    }
}
