//! `rsp arena charts` — view preflop charts from agent configs.
//!
//! Loads an agent config file, extracts or synthesizes a
//! [`PreflopChartConfig`], and launches an interactive TUI that shows the
//! 13×13 hand grid for each seat along with range-weighted action
//! breakdowns.

use std::io::IsTerminal;
use std::path::PathBuf;

use clap::Args;

use rs_poker::arena::agent::{AgentConfig, ConfigAgentBuilder};
use rs_poker::arena::cfr::{PositionCharts, PreflopChartConfig};
use rs_poker::holdem::{PreflopChart, PreflopHand, PreflopStrategy};

#[cfg(test)]
use rs_poker::holdem::PreflopScenario;

use crate::tui::chart_app::{ChartApp, run_chart_app};
use crate::tui::screens::chart_viewer::ChartViewerState;

/// Arguments for `rsp arena charts`.
#[derive(Args, Debug, Clone)]
pub struct ChartsArgs {
    /// Path to an agent config file.
    path: PathBuf,
    /// Seat to start on (0 = BB, 1 = SB, 2 = BTN, ...).
    #[arg(long, default_value_t = 2)]
    seat: usize,
    /// Override the number of seat tabs (default: 6).
    #[arg(long)]
    num_seats: Option<usize>,
}

#[derive(Debug, thiserror::Error)]
pub enum ChartsError {
    #[error("failed to load agent config: {0}")]
    LoadConfig(#[from] rs_poker::arena::agent::AgentConfigError),
    #[error("chart viewer requires a TTY; run in a terminal (or unset RSP_NO_TUI)")]
    NotATty,
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub fn run(args: ChartsArgs) -> Result<(), ChartsError> {
    if !std::io::stdout().is_terminal() {
        return Err(ChartsError::NotATty);
    }

    let builder = ConfigAgentBuilder::from_file(&args.path)?;
    let config = builder.config();
    let config_name = config_label(config, &args.path);
    let num_seats = args.num_seats.unwrap_or(6).max(1);

    let (preflop_config, synthesized, banner) = resolve_preflop_config(config);
    let mut state = ChartViewerState::new(preflop_config, config_name, num_seats, synthesized);
    if let Some(msg) = banner {
        state = state.with_banner(msg);
    }
    state.set_seat(args.seat);

    let mut app = ChartApp::new(state);
    run_chart_app(&mut app)?;
    Ok(())
}

/// Best-effort human-readable label for the config: use the agent's `name`
/// field if present, else the file stem.
fn config_label(config: &AgentConfig, path: &std::path::Path) -> String {
    let name = match config {
        AgentConfig::AllIn { name, .. }
        | AgentConfig::Calling { name, .. }
        | AgentConfig::Folding { name, .. }
        | AgentConfig::Random { name, .. }
        | AgentConfig::RandomPotControl { name, .. }
        | AgentConfig::CfrBasic { name, .. }
        | AgentConfig::CfrSimple { name, .. }
        | AgentConfig::CfrConfigurable { name, .. }
        | AgentConfig::CfrPreflopChart { name, .. } => name.clone(),
    };
    name.unwrap_or_else(|| {
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("config")
            .to_string()
    })
}

/// Extract or synthesize a `PreflopChartConfig` for any agent config.
///
/// Returns `(config, synthesized, optional banner)`. `synthesized=true`
/// means the chart is a reasonable approximation of the agent's behavior
/// rather than an authoritative preflop range.
fn resolve_preflop_config(config: &AgentConfig) -> (PreflopChartConfig, bool, Option<String>) {
    match config {
        AgentConfig::CfrPreflopChart { preflop_config, .. } => match preflop_config.resolve() {
            Ok(cfg) => (cfg, false, None),
            Err(e) => (
                PreflopChartConfig::with_single_position(PositionCharts::default()),
                true,
                Some(format!("failed to resolve preflop config: {}", e)),
            ),
        },
        AgentConfig::AllIn { .. } => (
            PreflopChartConfig::with_single_position(uniform_position_charts(
                PreflopStrategy::pure_raise(),
            )),
            true,
            None,
        ),
        AgentConfig::Calling { .. } => (
            PreflopChartConfig::with_single_position(uniform_position_charts(
                PreflopStrategy::pure_call(),
            )),
            true,
            None,
        ),
        AgentConfig::Folding { .. } => (
            PreflopChartConfig::with_single_position(PositionCharts::default()),
            true,
            None,
        ),
        AgentConfig::Random {
            percent_fold,
            percent_call,
            ..
        } => (
            PreflopChartConfig::with_single_position(random_position_charts(
                percent_fold,
                percent_call,
            )),
            true,
            Some(
                "Synthesized from first element of percent_fold/percent_call \
                 (preflop, no raises yet)"
                    .to_string(),
            ),
        ),
        AgentConfig::RandomPotControl { percent_call, .. } => (
            PreflopChartConfig::with_single_position(random_position_charts(&[], percent_call)),
            true,
            Some(
                "Synthesized from first element of percent_call \
                 (RandomPotControl only defines call frequency)"
                    .to_string(),
            ),
        ),
        AgentConfig::CfrBasic { .. }
        | AgentConfig::CfrSimple { .. }
        | AgentConfig::CfrConfigurable { .. } => (
            PreflopChartConfig::with_single_position(PositionCharts::default()),
            true,
            Some(
                "CFR agent without preflop chart — preflop action is learned \
                 at runtime via CFR exploration. Use cfr_preflop_chart to \
                 constrain preflop."
                    .to_string(),
            ),
        ),
    }
}

/// Build a chart containing `strategy` for every hand.
fn uniform_chart(strategy: PreflopStrategy) -> PreflopChart {
    let mut chart = PreflopChart::new();
    for hand in PreflopHand::all() {
        chart.set(hand, strategy);
    }
    chart
}

/// Build position charts with `strategy` applied across every scenario.
///
/// Note: `rfi` does not populate `call` even if the input strategy has it —
/// call is invalid in RFI (no limping). Similarly `vs_4bet` drops raise.
fn uniform_position_charts(strategy: PreflopStrategy) -> PositionCharts {
    // Strip disallowed components per scenario.
    let rfi_strategy = PreflopStrategy::new(strategy.raise(), 0.0).unwrap_or_default();
    let vs_4bet_strategy = PreflopStrategy::new(0.0, strategy.call()).unwrap_or_default();
    PositionCharts {
        rfi: uniform_chart(rfi_strategy),
        vs_open: uniform_chart(strategy),
        vs_3bet: uniform_chart(strategy),
        vs_4bet: uniform_chart(vs_4bet_strategy),
    }
}

/// Build `PositionCharts` from a random agent's fold/call probability
/// vectors. The preflop (no raises yet) entry is index 0.
fn random_position_charts(percent_fold: &[f64], percent_call: &[f64]) -> PositionCharts {
    let fold = percent_fold.first().copied().unwrap_or(0.0).clamp(0.0, 1.0);
    let call = percent_call.first().copied().unwrap_or(0.0).clamp(0.0, 1.0);
    let call = call.min(1.0 - fold);
    let raise = (1.0 - fold - call).max(0.0);
    let strategy =
        PreflopStrategy::new(raise as f32, call as f32).unwrap_or_else(|_| PreflopStrategy::fold());
    uniform_position_charts(strategy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn config_from_json(json: &str) -> AgentConfig {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn label_falls_back_to_file_stem() {
        let cfg = config_from_json(r#"{"type":"all_in"}"#);
        let label = config_label(&cfg, Path::new("/foo/my_agent.json"));
        assert_eq!(label, "my_agent");
    }

    #[test]
    fn label_uses_name_field_when_present() {
        let cfg = config_from_json(r#"{"type":"all_in","name":"YOLO"}"#);
        let label = config_label(&cfg, Path::new("/foo/other.json"));
        assert_eq!(label, "YOLO");
    }

    #[test]
    fn all_in_synthesizes_pure_raise() {
        let cfg = config_from_json(r#"{"type":"all_in"}"#);
        let (chart_cfg, synth, _) = resolve_preflop_config(&cfg);
        assert!(synth);
        let aa = PreflopHand::from_notation("AA").unwrap();
        let strategy = chart_cfg
            .chart_for(0, PreflopScenario::Rfi)
            .get(&aa)
            .unwrap();
        assert!((strategy.raise() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn folding_synthesizes_empty_chart() {
        let cfg = config_from_json(r#"{"type":"folding"}"#);
        let (chart_cfg, synth, _) = resolve_preflop_config(&cfg);
        assert!(synth);
        assert!(chart_cfg.chart_for(0, PreflopScenario::Rfi).is_empty());
    }

    #[test]
    fn random_synthesizes_mixed_strategy() {
        let cfg =
            config_from_json(r#"{"type":"random","percent_fold":[0.2],"percent_call":[0.5]}"#);
        let (chart_cfg, synth, _) = resolve_preflop_config(&cfg);
        assert!(synth);
        let aa = PreflopHand::from_notation("AA").unwrap();
        // fold=0.2, call=0.5, raise=0.3 — but RFI strips call.
        let rfi = chart_cfg
            .chart_for(0, PreflopScenario::Rfi)
            .get(&aa)
            .unwrap();
        assert!((rfi.raise() - 0.3).abs() < 1e-5);
        assert_eq!(rfi.call(), 0.0);
        // VsOpen keeps both raise and call.
        let vso = chart_cfg
            .chart_for(0, PreflopScenario::VsOpen)
            .get(&aa)
            .unwrap();
        assert!((vso.raise() - 0.3).abs() < 1e-5);
        assert!((vso.call() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn cfr_basic_has_banner_but_empty_chart() {
        let cfg = config_from_json(r#"{"type":"cfr_basic"}"#);
        let (chart_cfg, synth, banner) = resolve_preflop_config(&cfg);
        assert!(synth);
        assert!(banner.is_some());
        assert!(chart_cfg.chart_for(0, PreflopScenario::Rfi).is_empty());
    }
}
