//! Pluggable budget for CFR exploration.
//!
//! A [`Budget`] answers one question — *"what should the next wave-loop
//! iteration do?"* — and is consulted once at each iteration boundary of
//! `explore_all_actions`. The returned [`NextStep`] tells the engine to
//! `Stop`, run a recursive `Wave` of a given width, run a one-shot
//! `FastForward`, arm a deadline timer via `StartTimer`, or `Pass` (no
//! opinion — only meaningful inside a composer).
//!
//! Composition is via [`MostRestrictive`] (tightest answer wins) and
//! [`PerDepth`] (dispatch by recursion depth). With `Pass` doing the "no
//! opinion" job and `Stop` being the most restrictive answer, every
//! useful composition is "tightest-wins"; there is no `LeastRestrictive`.
//!
//! Deadlines live in the budget tree. A [`Deadline`] leaf returns
//! `StartTimer { duration }` on its first call at the root; the engine
//! arms a stop-flag timer in response, sets `stats.timer_armed = true`,
//! and re-queries. `Deadline` returns `Pass` thereafter. The engine
//! enforces the deadline by polling a lock-free `Arc<AtomicBool>` stop
//! flag at every wave boundary.

use std::sync::Arc;
use std::time::Duration;

/// Snapshot of exploration progress handed to a [`Budget`] each iteration.
///
/// Every field is `Copy`; `Budget` impls are pure functions of this.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ExplorationStats {
    /// Wall-clock time since exploration began at this node.
    pub elapsed: Duration,
    /// Completed wave iterations at THIS node.
    pub iterations: u64,
    /// Tree nodes touched (created or updated) so far (global signal).
    pub nodes_touched: u64,
    /// This agent's recursion depth (0 = root).
    pub depth: usize,
    /// Node-local average regret from the most recent completed update at
    /// this node. `None` until the first update lands.
    pub avg_regret: Option<f32>,
    /// Whether the per-act stop-flag timer has been armed already. Set by
    /// the engine after it processes a `NextStep::StartTimer`. Lets the
    /// `Deadline` leaf return `StartTimer` exactly once per act.
    pub timer_armed: bool,
}

/// What the next wave-loop iteration should do.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NextStep {
    /// Stop the wave loop. The agent picks from accumulated regret.
    Stop,
    /// No opinion — filtered by composers. If the top-level answer is
    /// `Pass`, the engine treats it as `Stop` (no budget wants to keep
    /// going).
    Pass,
    /// Arm a stop-flag timer for `duration` and re-query. This iteration
    /// runs NO wave. The engine sets `stats.timer_armed = true` after
    /// arming, so subsequent calls from the same `Deadline` leaf return
    /// `Pass`.
    StartTimer { duration: Duration },
    /// Run a recursive wave: `width` samples per action, each via a full
    /// sub-simulation.
    Wave { width: usize },
    /// Run exactly one fast-forward computation per action, then stop.
    /// Fast-forward is deterministic for 0–2 remaining community cards
    /// (full enumeration) and samples flops internally for 3 cards; doing
    /// it more than once per node yields no new information.
    FastForward,
}

/// A budget — decides what the next wave-loop iteration should do.
///
/// Implementations must be cheap (called once per iteration on the hot
/// path) and `Send + Sync` (the engine shares budgets across spawned
/// tasks via `Arc<dyn Budget>`).
pub trait Budget: Send + Sync {
    fn next_step(&self, stats: &ExplorationStats) -> NextStep;
}

// ─── Stop-leaf budgets ────────────────────────────────────────────────

/// Stop after a fixed number of completed wave iterations at this node.
#[derive(Debug, Clone, Copy)]
pub struct IterationCount {
    pub max: u64,
}

impl IterationCount {
    pub fn new(max: u64) -> Self {
        Self { max }
    }
}

impl Budget for IterationCount {
    fn next_step(&self, stats: &ExplorationStats) -> NextStep {
        if stats.iterations >= self.max {
            NextStep::Stop
        } else {
            NextStep::Pass
        }
    }
}

/// Stop once the global tree has touched `max` nodes.
#[derive(Debug, Clone, Copy)]
pub struct NodeCount {
    pub max: u64,
}

impl NodeCount {
    pub fn new(max: u64) -> Self {
        Self { max }
    }
}

impl Budget for NodeCount {
    fn next_step(&self, stats: &ExplorationStats) -> NextStep {
        if stats.nodes_touched >= self.max {
            NextStep::Stop
        } else {
            NextStep::Pass
        }
    }
}

/// Stop once this node has completed at least `min_iterations` and its
/// average regret has fallen to/below `epsilon`. Folds the
/// `MinIterations + RegretThreshold` floor + threshold pattern into one
/// leaf so a fresh matcher's initial `0.0` isn't read as convergence.
#[derive(Debug, Clone, Copy)]
pub struct RegretBelow {
    pub epsilon: f32,
    pub min_iterations: u64,
}

impl RegretBelow {
    pub fn new(epsilon: f32, min_iterations: u64) -> Self {
        Self {
            epsilon,
            min_iterations,
        }
    }
}

impl Budget for RegretBelow {
    fn next_step(&self, stats: &ExplorationStats) -> NextStep {
        if stats.iterations >= self.min_iterations
            && stats.avg_regret.is_some_and(|r| r <= self.epsilon)
        {
            NextStep::Stop
        } else {
            NextStep::Pass
        }
    }
}

// ─── Deadline (timer-emitter) ─────────────────────────────────────────

/// Per-act wall-clock ceiling. On its very first call at the root
/// (`depth == 0 && !timer_armed`), returns
/// `NextStep::StartTimer { duration: self.0 }`. The engine spawns a
/// timer task that flips the lock-free `stop` flag when it elapses, and
/// sets `stats.timer_armed = true`; subsequent calls return `Pass`. At
/// every depth > 0, always returns `Pass` (sub-agents inherit the root's
/// stop flag — they don't arm their own timers today, though a future
/// per-depth-deadline leaf could).
#[derive(Debug, Clone, Copy)]
pub struct Deadline(pub Duration);

impl Deadline {
    pub fn new(duration: Duration) -> Self {
        Self(duration)
    }
}

impl Budget for Deadline {
    fn next_step(&self, stats: &ExplorationStats) -> NextStep {
        if stats.depth == 0 && !stats.timer_armed {
            NextStep::StartTimer { duration: self.0 }
        } else {
            NextStep::Pass
        }
    }
}

// ─── Action leaf — recurse/FF schedule + wave width per depth ─────────

/// At depth `d < recursive_widths.len()`, returns
/// `Wave { width: recursive_widths[d] }`. At depth `d >= len`, returns
/// `FastForward`. Replaces `RecursionConfig`: the vec length sets the
/// fast-forward boundary, and the entries set per-depth wave widths.
///
/// Under `MostRestrictive`, two `MaxWidth` leaves compose by taking the
/// per-depth minimum width — the tightest cap wins.
#[derive(Debug, Clone)]
pub struct MaxWidth {
    pub recursive_widths: Vec<usize>,
}

impl MaxWidth {
    pub fn new(recursive_widths: Vec<usize>) -> Self {
        Self { recursive_widths }
    }
}

impl Budget for MaxWidth {
    fn next_step(&self, stats: &ExplorationStats) -> NextStep {
        match self.recursive_widths.get(stats.depth) {
            Some(&width) => NextStep::Wave { width },
            None => NextStep::FastForward,
        }
    }
}

// ─── Composer ─────────────────────────────────────────────────────────

/// "Tightest answer wins" composer.
///
/// Variant priority for the result (most-restrictive first): `StartTimer`,
/// `Stop`, `FastForward`, `Wave` (smaller width is more restrictive), then
/// `Pass`. `Pass` is filtered before comparison.
///
/// - If any child returns `StartTimer`, that `StartTimer` is returned
///   (engine handles it as setup and re-queries).
/// - Else if any child returns `Stop`, returns `Stop`.
/// - Else if any returns `FastForward`, returns `FastForward`.
/// - Else if any returns `Wave`, returns `Wave { width: min(widths) }`.
/// - Else (all children `Pass`, or empty), returns `Pass`.
pub struct MostRestrictive {
    pub children: Vec<Arc<dyn Budget>>,
}

impl MostRestrictive {
    pub fn new(children: Vec<Arc<dyn Budget>>) -> Self {
        Self { children }
    }
}

impl Budget for MostRestrictive {
    fn next_step(&self, stats: &ExplorationStats) -> NextStep {
        let mut start_timer: Option<Duration> = None;
        let mut any_stop = false;
        let mut any_ff = false;
        let mut min_wave_width: Option<usize> = None;
        for c in &self.children {
            match c.next_step(stats) {
                NextStep::StartTimer { duration } => {
                    start_timer = Some(start_timer.map_or(duration, |d| d.min(duration)));
                }
                NextStep::Stop => any_stop = true,
                NextStep::FastForward => any_ff = true,
                NextStep::Wave { width } => {
                    min_wave_width = Some(min_wave_width.map_or(width, |m| m.min(width)));
                }
                NextStep::Pass => {}
            }
        }
        if let Some(d) = start_timer {
            return NextStep::StartTimer { duration: d };
        }
        if any_stop {
            return NextStep::Stop;
        }
        if any_ff {
            return NextStep::FastForward;
        }
        if let Some(w) = min_wave_width {
            return NextStep::Wave { width: w };
        }
        NextStep::Pass
    }
}

// ─── Dispatcher ───────────────────────────────────────────────────────

/// Depth-dispatching budget: consults `by_depth[stats.depth]` if in
/// range, else `fallback`.
pub struct PerDepth {
    pub by_depth: Vec<Arc<dyn Budget>>,
    pub fallback: Arc<dyn Budget>,
}

impl PerDepth {
    pub fn new(by_depth: Vec<Arc<dyn Budget>>, fallback: Arc<dyn Budget>) -> Self {
        Self { by_depth, fallback }
    }
}

impl Budget for PerDepth {
    fn next_step(&self, stats: &ExplorationStats) -> NextStep {
        match self.by_depth.get(stats.depth) {
            Some(b) => b.next_step(stats),
            None => self.fallback.next_step(stats),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats() -> ExplorationStats {
        ExplorationStats {
            elapsed: Duration::ZERO,
            iterations: 0,
            nodes_touched: 0,
            depth: 0,
            avg_regret: None,
            timer_armed: false,
        }
    }

    #[test]
    fn iteration_count_stops_at_limit() {
        let b = IterationCount::new(10);
        let mut s = stats();
        s.iterations = 9;
        assert_eq!(b.next_step(&s), NextStep::Pass);
        s.iterations = 10;
        assert_eq!(b.next_step(&s), NextStep::Stop);
        s.iterations = 11;
        assert_eq!(b.next_step(&s), NextStep::Stop);
    }

    #[test]
    fn node_count_stops_at_limit() {
        let b = NodeCount::new(100);
        let mut s = stats();
        s.nodes_touched = 99;
        assert_eq!(b.next_step(&s), NextStep::Pass);
        s.nodes_touched = 100;
        assert_eq!(b.next_step(&s), NextStep::Stop);
    }

    #[test]
    fn regret_below_needs_both_floor_and_threshold() {
        let b = RegretBelow::new(0.01, 8);
        let mut s = stats();
        s.iterations = 5;
        s.avg_regret = Some(0.001);
        assert_eq!(b.next_step(&s), NextStep::Pass);
        s.iterations = 8;
        s.avg_regret = Some(0.5);
        assert_eq!(b.next_step(&s), NextStep::Pass);
        s.avg_regret = None;
        assert_eq!(b.next_step(&s), NextStep::Pass);
        s.avg_regret = Some(0.01);
        assert_eq!(b.next_step(&s), NextStep::Stop);
        s.avg_regret = Some(0.001);
        assert_eq!(b.next_step(&s), NextStep::Stop);
    }

    #[test]
    fn deadline_emits_start_timer_once_at_root() {
        let b = Deadline::new(Duration::from_millis(250));
        let mut s = stats();
        s.depth = 0;
        s.timer_armed = false;
        assert_eq!(
            b.next_step(&s),
            NextStep::StartTimer {
                duration: Duration::from_millis(250)
            }
        );
        s.timer_armed = true;
        assert_eq!(b.next_step(&s), NextStep::Pass);
        s.depth = 1;
        s.timer_armed = false;
        assert_eq!(b.next_step(&s), NextStep::Pass);
        s.depth = 5;
        assert_eq!(b.next_step(&s), NextStep::Pass);
    }

    #[test]
    fn max_width_returns_wave_in_range_else_fast_forward() {
        let b = MaxWidth::new(vec![8, 4, 2]);
        let mut s = stats();
        s.depth = 0;
        assert_eq!(b.next_step(&s), NextStep::Wave { width: 8 });
        s.depth = 1;
        assert_eq!(b.next_step(&s), NextStep::Wave { width: 4 });
        s.depth = 2;
        assert_eq!(b.next_step(&s), NextStep::Wave { width: 2 });
        s.depth = 3;
        assert_eq!(b.next_step(&s), NextStep::FastForward);
        s.depth = 100;
        assert_eq!(b.next_step(&s), NextStep::FastForward);
    }

    #[test]
    fn most_restrictive_start_timer_wins() {
        let b = MostRestrictive::new(vec![
            Arc::new(Deadline::new(Duration::from_millis(100))),
            Arc::new(MaxWidth::new(vec![8])),
        ]);
        let mut s = stats();
        s.depth = 0;
        s.timer_armed = false;
        assert_eq!(
            b.next_step(&s),
            NextStep::StartTimer {
                duration: Duration::from_millis(100)
            }
        );
    }

    #[test]
    fn most_restrictive_picks_min_start_timer_duration() {
        // Two Deadlines disagree; the tighter (smaller) duration must win.
        let b = MostRestrictive::new(vec![
            Arc::new(Deadline::new(Duration::from_millis(500))),
            Arc::new(Deadline::new(Duration::from_millis(100))),
            Arc::new(Deadline::new(Duration::from_millis(250))),
        ]);
        let mut s = stats();
        s.depth = 0;
        s.timer_armed = false;
        assert_eq!(
            b.next_step(&s),
            NextStep::StartTimer {
                duration: Duration::from_millis(100)
            }
        );
    }

    #[test]
    fn most_restrictive_stop_wins_over_action() {
        let b = MostRestrictive::new(vec![
            Arc::new(IterationCount::new(5)),
            Arc::new(MaxWidth::new(vec![8])),
        ]);
        let mut s = stats();
        s.iterations = 5;
        assert_eq!(b.next_step(&s), NextStep::Stop);
    }

    #[test]
    fn most_restrictive_fast_forward_wins_over_wave() {
        let b = MostRestrictive::new(vec![
            Arc::new(MaxWidth::new(vec![8, 4])),
            Arc::new(MaxWidth::new(vec![])),
        ]);
        let mut s = stats();
        s.depth = 0;
        assert_eq!(b.next_step(&s), NextStep::FastForward);
    }

    #[test]
    fn most_restrictive_picks_min_wave_width() {
        let b = MostRestrictive::new(vec![
            Arc::new(MaxWidth::new(vec![8, 4])),
            Arc::new(MaxWidth::new(vec![2, 1])),
        ]);
        let mut s = stats();
        s.depth = 0;
        assert_eq!(b.next_step(&s), NextStep::Wave { width: 2 });
        s.depth = 1;
        assert_eq!(b.next_step(&s), NextStep::Wave { width: 1 });
    }

    #[test]
    fn most_restrictive_empty_returns_pass() {
        let b = MostRestrictive::new(vec![]);
        assert_eq!(b.next_step(&stats()), NextStep::Pass);
    }

    #[test]
    fn most_restrictive_all_pass_returns_pass() {
        let b = MostRestrictive::new(vec![
            Arc::new(IterationCount::new(1000)),
            Arc::new(NodeCount::new(1000)),
        ]);
        assert_eq!(b.next_step(&stats()), NextStep::Pass);
    }

    #[test]
    fn per_depth_does_not_fall_through_when_child_returns_pass() {
        // by_depth[0] = IterationCount(10) — returns Pass while iter < 10.
        // fallback = MaxWidth([5]) at depth 0 — would return Wave { 5 } if consulted.
        // PerDepth should consult by_depth[0] ONLY, NOT fall through to fallback.
        let b = PerDepth::new(
            vec![Arc::new(IterationCount::new(10))],
            Arc::new(MaxWidth::new(vec![5])),
        );
        let mut s = stats();
        s.depth = 0;
        s.iterations = 0;
        // by_depth[0].next_step → Pass. Result must be Pass, not Wave { 5 }.
        assert_eq!(b.next_step(&s), NextStep::Pass);
    }

    #[test]
    fn per_depth_dispatches_by_depth() {
        let b = PerDepth::new(
            vec![
                Arc::new(IterationCount::new(10)),
                Arc::new(IterationCount::new(3)),
            ],
            Arc::new(IterationCount::new(1)),
        );
        let mut s = stats();
        s.depth = 0;
        s.iterations = 9;
        assert_eq!(b.next_step(&s), NextStep::Pass);
        s.iterations = 10;
        assert_eq!(b.next_step(&s), NextStep::Stop);
        s.depth = 1;
        s.iterations = 2;
        assert_eq!(b.next_step(&s), NextStep::Pass);
        s.iterations = 3;
        assert_eq!(b.next_step(&s), NextStep::Stop);
        s.depth = 5;
        s.iterations = 0;
        assert_eq!(b.next_step(&s), NextStep::Pass);
        s.iterations = 1;
        assert_eq!(b.next_step(&s), NextStep::Stop);
    }
}
