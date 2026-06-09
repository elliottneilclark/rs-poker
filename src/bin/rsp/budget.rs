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

/// The `rsp` operational default budget
pub fn operational_default() -> BudgetConfig {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    BudgetConfig(vec![
        BudgetItem::Deadline { millis: 2048 },
        BudgetItem::PerDepth {
            by_depth: vec![
                BudgetItem::IterationCount { max: 128 },
                BudgetItem::IterationCount { max: 3 },
                BudgetItem::IterationCount { max: 1 },
                BudgetItem::IterationCount { max: 1 },
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
            Some(BudgetItem::Deadline { millis: 2048 })
        ));
        match &b.0[1] {
            BudgetItem::PerDepth { by_depth, fallback } => {
                assert_eq!(by_depth.len(), 5);
                assert!(matches!(
                    fallback.as_ref(),
                    BudgetItem::IterationCount { max: 1 }
                ));
            }
            other => panic!("expected PerDepth, got {other:?}"),
        }
    }
}
