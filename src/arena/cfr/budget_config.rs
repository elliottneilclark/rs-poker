//! Serializable mirror of the runtime [`Budget`] tree.
//!
//! A [`BudgetConfig`] is a flat list of [`BudgetItem`] variants,
//! implicitly aggregated under [`MostRestrictive`]. JSON form is a
//! top-level array. `BudgetConfig::default()` is a small safe
//! exploration (100 ms deadline, 5 root iterations, no recursion past
//! depth 0) used as a fallback when no operational budget has been
//! supplied.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::budget::{
    Budget, Deadline, IterationCount, MaxWidth, MostRestrictive, NodeCount, PerDepth, RegretBelow,
};

fn one() -> u64 {
    1
}

/// A budget config — a list of items, implicitly aggregated as
/// `MostRestrictive`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(transparent)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct BudgetConfig(pub Vec<BudgetItem>);

/// One item in a [`BudgetConfig`] list.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum BudgetItem {
    /// Arm a per-act stop-flag timer for `millis`. Only meaningful at
    /// the root (depth 0) — see `Deadline` in budget.rs.
    Deadline { millis: u64 },
    /// Stop after `max` waves at this node.
    IterationCount { max: u64 },
    /// Stop after `max` global tree nodes touched.
    NodeCount { max: u64 },
    /// Stop once node-local avg regret ≤ `epsilon` AND iterations ≥
    /// `min_iterations`.
    RegretBelow { epsilon: f32, min_iterations: u64 },
    /// Per-depth wave widths; depths past the vec fast-forward.
    MaxWidth { recursive_widths: Vec<usize> },
    /// Dispatch to a depth-indexed inner item; depths past the vec use
    /// `fallback`. Both fields are required in JSON — use
    /// `{"type": "iteration_count", "max": 1}` (or similar) as the fallback
    /// for a "stop at depth past the vec" effect.
    PerDepth {
        by_depth: Vec<BudgetItem>,
        fallback: Box<BudgetItem>,
    },
    /// Sugar: build a `PerDepth` of `IterationCount` from a plain count
    /// list. Past-vec depths use `IterationCount::new(fallback)`.
    PerDepthIterations {
        counts: Vec<u64>,
        #[serde(default = "one")]
        fallback: u64,
    },
}

impl Default for BudgetConfig {
    /// Small safe exploration: a bounded, terminating exploration used
    /// when no operational budget has been supplied. 100 ms cap, up to 5
    /// root iterations, width 1, no recursion past depth 0.
    fn default() -> Self {
        BudgetConfig(vec![
            BudgetItem::Deadline { millis: 100 },
            BudgetItem::IterationCount { max: 5 },
            BudgetItem::MaxWidth {
                recursive_widths: vec![1],
            },
        ])
    }
}

impl BudgetConfig {
    /// Compile this config into a live `Arc<dyn Budget>` (a
    /// `MostRestrictive` over the items' built children).
    pub fn build(&self) -> Arc<dyn Budget> {
        let children: Vec<Arc<dyn Budget>> = self.0.iter().map(BudgetItem::build).collect();
        Arc::new(MostRestrictive::new(children))
    }
}

impl BudgetItem {
    fn build(&self) -> Arc<dyn Budget> {
        match self {
            BudgetItem::Deadline { millis } => {
                Arc::new(Deadline::new(Duration::from_millis(*millis)))
            }
            BudgetItem::IterationCount { max } => Arc::new(IterationCount::new(*max)),
            BudgetItem::NodeCount { max } => Arc::new(NodeCount::new(*max)),
            BudgetItem::RegretBelow {
                epsilon,
                min_iterations,
            } => Arc::new(RegretBelow::new(*epsilon, *min_iterations)),
            BudgetItem::MaxWidth { recursive_widths } => {
                Arc::new(MaxWidth::new(recursive_widths.clone()))
            }
            BudgetItem::PerDepth { by_depth, fallback } => {
                let by: Vec<Arc<dyn Budget>> = by_depth.iter().map(BudgetItem::build).collect();
                Arc::new(PerDepth::new(by, fallback.build()))
            }
            BudgetItem::PerDepthIterations { counts, fallback } => {
                let by: Vec<Arc<dyn Budget>> = counts
                    .iter()
                    .map(|c| Arc::new(IterationCount::new(*c)) as Arc<dyn Budget>)
                    .collect();
                Arc::new(PerDepth::new(by, Arc::new(IterationCount::new(*fallback))))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::cfr::budget::{ExplorationStats, NextStep};

    fn stats(iterations: u64, depth: usize) -> ExplorationStats {
        ExplorationStats {
            elapsed: Duration::ZERO,
            iterations,
            nodes_touched: 0,
            depth,
            avg_regret: None,
            timer_armed: true, // most tests want to skip StartTimer
        }
    }

    #[test]
    fn json_round_trips_flat_array() {
        let cfg = BudgetConfig(vec![
            BudgetItem::Deadline { millis: 250 },
            BudgetItem::PerDepthIterations {
                counts: vec![24, 3, 1],
                fallback: 1,
            },
            BudgetItem::RegretBelow {
                epsilon: 0.001,
                min_iterations: 8,
            },
            BudgetItem::MaxWidth {
                recursive_widths: vec![8, 1, 1],
            },
        ]);
        let json = serde_json::to_string(&cfg).unwrap();
        // Should serialize as a top-level array.
        assert!(json.starts_with('['));
        let back: BudgetConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn default_is_small_safe_exploration() {
        let cfg = BudgetConfig::default();
        assert_eq!(
            cfg,
            BudgetConfig(vec![
                BudgetItem::Deadline { millis: 100 },
                BudgetItem::IterationCount { max: 5 },
                BudgetItem::MaxWidth {
                    recursive_widths: vec![1]
                },
            ])
        );
    }

    #[test]
    fn build_produces_most_restrictive_over_items() {
        let b = BudgetConfig(vec![
            BudgetItem::IterationCount { max: 3 },
            BudgetItem::MaxWidth {
                recursive_widths: vec![8, 4],
            },
        ])
        .build();
        // depth 0, iterations 2 → Wave { 8 } (MaxWidth wins; IterCount Passes)
        assert_eq!(b.next_step(&stats(2, 0)), NextStep::Wave { width: 8 });
        // iterations 3 → Stop (IterationCount fires)
        assert_eq!(b.next_step(&stats(3, 0)), NextStep::Stop);
    }

    #[test]
    fn per_depth_iterations_expands_to_per_depth_of_iteration_count() {
        let b = BudgetConfig(vec![BudgetItem::PerDepthIterations {
            counts: vec![10, 3],
            fallback: 1,
        }])
        .build();
        // depth 0, iter 9 → Pass; iter 10 → Stop.
        assert_eq!(b.next_step(&stats(9, 0)), NextStep::Pass);
        assert_eq!(b.next_step(&stats(10, 0)), NextStep::Stop);
        // depth 1, iter 2 → Pass; iter 3 → Stop.
        assert_eq!(b.next_step(&stats(2, 1)), NextStep::Pass);
        assert_eq!(b.next_step(&stats(3, 1)), NextStep::Stop);
        // depth 5 (fallback), iter 0 → Pass; iter 1 → Stop.
        assert_eq!(b.next_step(&stats(0, 5)), NextStep::Pass);
        assert_eq!(b.next_step(&stats(1, 5)), NextStep::Stop);
    }

    #[test]
    fn per_depth_dispatches_by_depth() {
        // depth 0 → IterationCount(10); depth 1 → MaxWidth([4]); fallback → IterationCount(1).
        let b = BudgetConfig(vec![BudgetItem::PerDepth {
            by_depth: vec![
                BudgetItem::IterationCount { max: 10 },
                BudgetItem::MaxWidth {
                    recursive_widths: vec![4],
                },
            ],
            fallback: Box::new(BudgetItem::IterationCount { max: 1 }),
        }])
        .build();
        // depth 0: IterationCount(10) → Pass while iter < 10, Stop at 10.
        assert_eq!(b.next_step(&stats(9, 0)), NextStep::Pass);
        assert_eq!(b.next_step(&stats(10, 0)), NextStep::Stop);
        // depth 1: MaxWidth([4]) at depth 1 is past its vec (len=1) → FastForward.
        // Note: MaxWidth.recursive_widths is indexed by stats.depth, so at depth 1
        // with widths=[4], depth 1 is past the vec → FastForward.
        assert_eq!(b.next_step(&stats(0, 1)), NextStep::FastForward);
        // depth 5: fallback IterationCount(1) → Pass at 0, Stop at 1.
        assert_eq!(b.next_step(&stats(0, 5)), NextStep::Pass);
        assert_eq!(b.next_step(&stats(1, 5)), NextStep::Stop);
    }
}
