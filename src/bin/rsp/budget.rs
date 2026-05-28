//! Binary-level budget plumbing for the `rsp` CLI.
//!
//! The library has no implicit operational default; this module owns the
//! `rsp`-specific operational default and the `--budget` / `RSP_BUDGET`
//! resolution logic.

use std::path::Path;

use rs_poker::arena::agent::AgentConfig;
use rs_poker::arena::cfr::{BudgetConfig, BudgetItem};

#[derive(Debug, thiserror::Error)]
pub enum BudgetError {
    #[error("failed to read budget file '{path}': {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse budget JSON: {0}")]
    Json(#[from] serde_json::Error),
}

/// Resolve the effective budget for this run.
///
/// Priority: explicit `--budget` CLI arg > `RSP_BUDGET` env var >
/// `operational_default()`. The CLI arg / env var value is either a
/// path to a JSON file or inline JSON if it starts with `{` or `[`.
pub fn effective_budget(cli_value: Option<&str>) -> Result<BudgetConfig, BudgetError> {
    let raw = cli_value
        .map(str::to_owned)
        .or_else(|| std::env::var("RSP_BUDGET").ok());
    match raw.as_deref() {
        Some(s) => parse_budget_arg(s),
        None => Ok(operational_default()),
    }
}

fn parse_budget_arg(s: &str) -> Result<BudgetConfig, BudgetError> {
    let trimmed = s.trim_start();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        Ok(serde_json::from_str(s)?)
    } else {
        let bytes = std::fs::read(Path::new(s)).map_err(|source| BudgetError::Io {
            path: s.to_owned(),
            source,
        })?;
        Ok(serde_json::from_slice(&bytes)?)
    }
}

/// The `rsp` operational default budget: 800 ms deadline, `[48, 3, 1]`
/// per-depth iteration schedule, regret early-stop, root-wave parallelism
/// up to `available_parallelism`.
///
/// Picked from a sweep over (deadline, root-cap) at 50 sims/point. The
/// 250 ms / `[24, 3, 1]` predecessor was both deadline-bound (68 % of
/// depth-0 acts) and cap-bound (iter p50 = 24). Doubling the cap and
/// tripling the deadline gives ~2× regret p50, ~4× p90, and ~8× p99
/// improvement at ~2.6× wall (~1.08 s/game, ~3 300 games/hour).
pub fn operational_default() -> BudgetConfig {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    BudgetConfig(vec![
        BudgetItem::Deadline { millis: 800 },
        BudgetItem::PerDepth {
            by_depth: vec![
                BudgetItem::IterationCount { max: 48 },
                BudgetItem::IterationCount { max: 3 },
                BudgetItem::IterationCount { max: 1 },
            ],
            fallback: Box::new(BudgetItem::IterationCount { max: 1 }),
        },
        BudgetItem::RegretBelow {
            epsilon: 1e-3,
            min_iterations: 8,
        },
        BudgetItem::MaxWidth {
            recursive_widths: vec![cores, 1, 1],
        },
    ])
}

/// Fill `exploration.budget = Some(default)` on every CFR-flavored
/// `AgentConfig` where it's currently `None`. Explicit per-config budgets
/// are preserved.
pub fn override_budgets(configs: &mut [AgentConfig], default: &BudgetConfig) {
    for c in configs {
        c.fill_default_budget(default);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_cli_no_env_returns_operational_default() {
        // SAFETY: test touches env vars; assume serial execution.
        unsafe {
            std::env::remove_var("RSP_BUDGET");
        }
        let b = effective_budget(None).unwrap();
        assert_eq!(b, operational_default());
    }

    #[test]
    fn cli_inline_json_array_parses() {
        let b = effective_budget(Some(r#"[{"type":"deadline","millis":500}]"#)).unwrap();
        assert_eq!(b, BudgetConfig(vec![BudgetItem::Deadline { millis: 500 }]));
    }

    #[test]
    fn operational_default_starts_with_deadline_and_schedule() {
        let b = operational_default();
        assert!(matches!(
            b.0.first(),
            Some(BudgetItem::Deadline { millis: 800 })
        ));
        match &b.0[1] {
            BudgetItem::PerDepth { by_depth, fallback } => {
                assert_eq!(by_depth.len(), 3);
                assert!(matches!(
                    fallback.as_ref(),
                    BudgetItem::IterationCount { max: 1 }
                ));
            }
            other => panic!("expected PerDepth, got {other:?}"),
        }
    }
}
