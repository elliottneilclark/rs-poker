use std::fs;
use std::path::{Path, PathBuf};

use clap::Args;
use tracing::{info, warn};

use rs_poker::arena::agent::{AgentConfig, ConfigAgentBuilder};
use rs_poker::arena::cfr::{PositionCharts, PreflopChartConfig};
use rs_poker::holdem::{PreflopChart, PreflopHand, PreflopScenario};

#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    #[error("failed to read path '{path}': {source}")]
    ReadPath {
        path: String,
        source: std::io::Error,
    },
    #[error("{failed} of {total} config(s) failed verification")]
    VerificationFailed { failed: usize, total: usize },
}

/// Verify agent config files load correctly and, when they carry preflop
/// charts, that the charts are well-formed with reasonable range coverage.
///
/// Accepts either a single config file or a directory of `.json` configs.
/// Matches `arena generate`'s loading behavior: non-recursive, top-level
/// `.json` files only. Exits non-zero if any configs fail to load.
#[derive(Args, Debug, Clone)]
pub struct VerifyArgs {
    /// Path to an agent config file or a directory of config files.
    path: PathBuf,
    /// Print combo-weighted range summaries for each (position, scenario).
    #[arg(long, default_value_t = false)]
    summary: bool,
}

pub fn run(args: VerifyArgs) -> Result<(), VerifyError> {
    let paths = collect_config_paths(&args.path)?;
    if paths.is_empty() {
        info!("No .json files found at '{}'", args.path.display());
        return Ok(());
    }

    let mut ok = 0usize;
    let mut failed = 0usize;
    for path in &paths {
        match ConfigAgentBuilder::from_file(path) {
            Ok(builder) => {
                info!("OK   {}", path.display());
                ok += 1;
                if args.summary
                    && let AgentConfig::CfrPreflopChart { preflop_config, .. } = builder.config()
                {
                    match preflop_config.resolve() {
                        Ok(cfg) => print_chart_summary(&cfg),
                        Err(e) => warn!("  could not resolve preflop config: {}", e),
                    }
                }
            }
            Err(e) => {
                warn!("FAIL {}: {}", path.display(), e);
                failed += 1;
            }
        }
    }

    let total = paths.len();
    info!("{} OK, {} failed ({} total)", ok, failed, total);

    if failed > 0 {
        return Err(VerifyError::VerificationFailed { failed, total });
    }
    Ok(())
}

fn collect_config_paths(path: &Path) -> Result<Vec<PathBuf>, VerifyError> {
    let meta = fs::metadata(path).map_err(|e| VerifyError::ReadPath {
        path: path.display().to_string(),
        source: e,
    })?;

    if meta.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }

    let entries = fs::read_dir(path).map_err(|e| VerifyError::ReadPath {
        path: path.display().to_string(),
        source: e,
    })?;

    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    paths.sort();
    Ok(paths)
}

fn print_chart_summary(cfg: &PreflopChartConfig) {
    info!(
        "  raise {:.2}bb · 3bet×{:.1} · 4bet+×{:.1} · positions {}",
        cfg.raise_size_bb,
        cfg.three_bet_multiplier,
        cfg.four_bet_plus_multiplier,
        cfg.positions.len()
    );
    for (idx, position) in cfg.positions.iter().enumerate() {
        let label = position_label(idx, cfg.positions.len());
        info!("  position {} ({}):", idx, label);
        for scenario in PreflopScenario::all() {
            print_scenario_summary(position, scenario);
        }
        if let Some(warning) = sanity_warning(idx, position) {
            warn!("    ⚠ {}", warning);
        }
    }
}

fn print_scenario_summary(position: &PositionCharts, scenario: PreflopScenario) {
    let chart = position.chart_for(scenario);
    let stats = range_stats(chart);
    info!(
        "    {:<8}  raise {:>5.1}%  call {:>5.1}%  fold {:>5.1}%  ({} hands)",
        scenario.label(),
        stats.raise_pct * 100.0,
        stats.call_pct * 100.0,
        stats.fold_pct * 100.0,
        chart.len()
    );
}

struct RangeStats {
    raise_pct: f32,
    call_pct: f32,
    fold_pct: f32,
}

/// Combo-weighted summary of a single chart. Pairs count as 6, suited as 4,
/// offsuit as 12.
fn range_stats(chart: &PreflopChart) -> RangeStats {
    let mut raise = 0.0f32;
    let mut call = 0.0f32;
    let mut fold = 0.0f32;
    let mut total = 0.0f32;

    for hand in PreflopHand::all() {
        let weight = combo_weight(&hand);
        total += weight;
        match chart.get(&hand) {
            Some(strategy) => {
                raise += strategy.raise() * weight;
                call += strategy.call() * weight;
                fold += strategy.fold_freq() * weight;
            }
            None => fold += weight,
        }
    }

    if total <= 0.0 {
        return RangeStats {
            raise_pct: 0.0,
            call_pct: 0.0,
            fold_pct: 0.0,
        };
    }
    RangeStats {
        raise_pct: raise / total,
        call_pct: call / total,
        fold_pct: fold / total,
    }
}

fn combo_weight(hand: &PreflopHand) -> f32 {
    if hand.is_pair() {
        6.0
    } else if hand.suited() {
        4.0
    } else {
        12.0
    }
}

fn position_label(idx: usize, num_positions: usize) -> String {
    match (idx, num_positions) {
        (0, _) => "BB".to_string(),
        (1, 2) => "BTN".to_string(),
        (1, _) => "SB".to_string(),
        (2, _) => "BTN".to_string(),
        (3, _) => "CO".to_string(),
        (4, _) => "HJ".to_string(),
        (5, _) => "UTG".to_string(),
        (n, _) => format!("UTG+{}", n - 5),
    }
}

/// Flag structurally odd charts. These aren't errors — just hints that the
/// config may be incomplete.
///
/// BB having RFI entries is legitimate: when the SB completes the blind, BB
/// faces a 0-raise decision (check or raise the option), so the RFI chart
/// applies.
fn sanity_warning(position: usize, charts: &PositionCharts) -> Option<String> {
    // BTN (idx 2) and CO (idx 3) should have open ranges. Empty RFI there
    // almost always indicates an incomplete chart rather than a strategic
    // choice.
    if (position == 2 || position == 3) && charts.rfi.is_empty() {
        return Some(format!(
            "position {} has no RFI chart — likely incomplete",
            position
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn write_json(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        path
    }

    #[test]
    fn valid_config_dir_passes() {
        let tmp = TempDir::new().unwrap();
        write_json(tmp.path(), "a.json", r#"{"type":"all_in"}"#);
        write_json(tmp.path(), "b.json", r#"{"type":"calling"}"#);
        run(VerifyArgs {
            path: tmp.path().to_path_buf(),
            summary: false,
        })
        .unwrap();
    }

    #[test]
    fn invalid_config_dir_fails() {
        let tmp = TempDir::new().unwrap();
        write_json(tmp.path(), "ok.json", r#"{"type":"all_in"}"#);
        write_json(tmp.path(), "bad.json", r#"{"name":"nope"}"#);
        let err = run(VerifyArgs {
            path: tmp.path().to_path_buf(),
            summary: false,
        })
        .unwrap_err();
        assert!(matches!(
            err,
            VerifyError::VerificationFailed {
                failed: 1,
                total: 2
            }
        ));
    }

    #[test]
    fn single_file_verifies() {
        let tmp = TempDir::new().unwrap();
        let file = write_json(tmp.path(), "single.json", r#"{"type":"folding"}"#);
        run(VerifyArgs {
            path: file,
            summary: false,
        })
        .unwrap();
    }

    #[test]
    fn missing_path_errors() {
        let err = run(VerifyArgs {
            path: PathBuf::from("/does/not/exist/configs"),
            summary: false,
        })
        .unwrap_err();
        assert!(matches!(err, VerifyError::ReadPath { .. }));
    }

    #[test]
    fn non_json_files_ignored() {
        let tmp = TempDir::new().unwrap();
        write_json(tmp.path(), "a.json", r#"{"type":"all_in"}"#);
        write_json(tmp.path(), "notes.txt", "not json at all");
        run(VerifyArgs {
            path: tmp.path().to_path_buf(),
            summary: false,
        })
        .unwrap();
    }

    #[test]
    fn empty_dir_passes() {
        let tmp = TempDir::new().unwrap();
        run(VerifyArgs {
            path: tmp.path().to_path_buf(),
            summary: false,
        })
        .unwrap();
    }

    #[test]
    fn subdirectory_not_recursed() {
        let tmp = TempDir::new().unwrap();
        let sub = tmp.path().join("sub");
        fs::create_dir(&sub).unwrap();
        write_json(&sub, "bad.json", r#"{"name":"no type field"}"#);
        run(VerifyArgs {
            path: tmp.path().to_path_buf(),
            summary: false,
        })
        .unwrap();
    }

    #[test]
    fn range_stats_empty_chart_is_all_fold() {
        let chart = PreflopChart::new();
        let stats = range_stats(&chart);
        assert!((stats.fold_pct - 1.0).abs() < 1e-5);
        assert!(stats.raise_pct.abs() < 1e-5);
        assert!(stats.call_pct.abs() < 1e-5);
    }

    #[test]
    fn range_stats_pure_raise_hand_contributes_combo_weight() {
        use rs_poker::core::Value;
        use rs_poker::holdem::PreflopStrategy;

        let mut chart = PreflopChart::new();
        // AA pure raise → 6 combos out of 1326 = ~0.452%.
        let aa = PreflopHand::new(Value::Ace, Value::Ace, false);
        chart.set(aa, PreflopStrategy::pure_raise());

        let stats = range_stats(&chart);
        let expected = 6.0 / 1326.0;
        assert!((stats.raise_pct - expected).abs() < 1e-5);
    }

    #[test]
    fn sanity_warning_silent_for_bb_rfi() {
        // BB can legitimately have RFI entries (SB-completes scenario).
        use rs_poker::core::Value;
        use rs_poker::holdem::PreflopStrategy;

        let mut charts = PositionCharts::default();
        charts.rfi.set(
            PreflopHand::new(Value::Ace, Value::Ace, false),
            PreflopStrategy::pure_raise(),
        );
        assert!(sanity_warning(0, &charts).is_none());
    }

    #[test]
    fn sanity_warning_flags_btn_missing_rfi() {
        let charts = PositionCharts::default();
        assert!(sanity_warning(2, &charts).is_some());
    }

    #[test]
    fn sanity_warning_silent_for_utg_empty_rfi() {
        // UTG may legitimately have a tight/empty chart in some studies.
        let charts = PositionCharts::default();
        assert!(sanity_warning(5, &charts).is_none());
    }
}
