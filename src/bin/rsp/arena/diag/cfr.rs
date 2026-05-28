use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::PathBuf;

use clap::Args;
use serde::Deserialize;

/// Number of log2-spaced histogram bins. Bin i covers
/// `[2^(i-60), 2^(i-59))`, so the full range is roughly `[1e-18, 1e18)`.
const N_BINS: usize = 120;

/// Summarize a captured `cfr_diag` JSONL stream into three plain-text sections:
/// stop_cause × depth cross-tab, depth-0 deadline-utilization histogram,
/// and final-regret quantiles by depth (plus last/first convergence ratio).
///
/// Streams the input line-by-line. Memory is O(causes × depths + bins × depths)
/// — bounded by the cardinality of the data, not the event count. Quantile
/// estimates are bin-midpoint geometric means over log2-spaced buckets,
/// accurate to ~one bin width; this is plenty sharp for triage.
///
/// Capture a JSONL stream with:
///   RSP_DIAG_LOG=cfr_diag=trace rsp arena generate ... 2> stats.jsonl
#[derive(Args, Debug, Clone)]
pub struct CfrArgs {
    /// JSONL file captured via `RSP_DIAG_LOG=cfr_diag=trace ... 2> file.jsonl`.
    /// Pass `-` to read from stdin.
    file: PathBuf,
    /// Configured Deadline budget in milliseconds, used for the depth-0
    /// utilization histogram.
    #[arg(long, default_value_t = 250)]
    deadline_ms: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum CfrDiagError {
    #[error("failed to open file '{path}': {source}")]
    Open { path: String, source: io::Error },
    #[error("deadline_ms must be > 0")]
    InvalidDeadline,
    #[error("read error: {0}")]
    Read(#[from] io::Error),
}

#[derive(Deserialize)]
struct Event {
    #[serde(default)]
    target: String,
    fields: Fields,
}

#[derive(Deserialize)]
struct Fields {
    depth: u64,
    stop_cause: String,
    #[serde(default)]
    final_elapsed_us: u64,
    /// Encoded as a JSON-quoted string by the tracing emitter
    /// (`?slice` Debug → e.g. `"[0.42, 0.31]"`).
    #[serde(default)]
    regret_series: String,
}

struct Accum {
    cross: HashMap<(String, u64), u64>,
    ratios: [u64; 6],
    d0_n: u64,
    regret: HashMap<u64, Box<[u64; N_BINS]>>,
    conv: HashMap<u64, Box<[u64; N_BINS]>>,
    depths: HashMap<u64, u64>,
    causes: BTreeSet<String>,
    total: u64,
}

impl Accum {
    fn new() -> Self {
        Self {
            cross: HashMap::new(),
            ratios: [0; 6],
            d0_n: 0,
            regret: HashMap::new(),
            conv: HashMap::new(),
            depths: HashMap::new(),
            causes: BTreeSet::new(),
            total: 0,
        }
    }

    fn observe(&mut self, f: &Fields, deadline_us: u64) {
        self.total += 1;
        *self.depths.entry(f.depth).or_insert(0) += 1;
        if !self.causes.contains(&f.stop_cause) {
            self.causes.insert(f.stop_cause.clone());
        }
        *self
            .cross
            .entry((f.stop_cause.clone(), f.depth))
            .or_insert(0) += 1;

        if f.depth == 0 {
            self.d0_n += 1;
            let r = f.final_elapsed_us as f64 / deadline_us as f64;
            let bucket = if r < 0.25 {
                0
            } else if r < 0.5 {
                1
            } else if r < 0.75 {
                2
            } else if r < 0.95 {
                3
            } else if r < 1.0 {
                4
            } else {
                5
            };
            self.ratios[bucket] += 1;
        }

        let Ok(series) = serde_json::from_str::<Vec<f32>>(&f.regret_series) else {
            return;
        };
        if let Some(&last) = series.last() {
            let hist = self
                .regret
                .entry(f.depth)
                .or_insert_with(|| Box::new([0; N_BINS]));
            hist[bin_of(last as f64)] += 1;
        }
        if series.len() > 1 && series[0] != 0.0 {
            let ratio = (*series.last().unwrap() / series[0]) as f64;
            let hist = self
                .conv
                .entry(f.depth)
                .or_insert_with(|| Box::new([0; N_BINS]));
            hist[bin_of(ratio)] += 1;
        }
    }
}

fn bin_of(v: f64) -> usize {
    if v <= 0.0 || !v.is_finite() {
        return 0;
    }
    let idx = v.log2().floor() as i32 + 60;
    idx.clamp(0, (N_BINS - 1) as i32) as usize
}

fn bin_midpoint(i: usize) -> f64 {
    2f64.powi(i as i32 - 60) * std::f64::consts::SQRT_2
}

fn hist_quantile(h: &[u64; N_BINS], p: f64) -> Option<f64> {
    let total: u64 = h.iter().sum();
    if total == 0 {
        return None;
    }
    let target = total as f64 * p;
    let mut acc: u64 = 0;
    for (i, &c) in h.iter().enumerate() {
        acc += c;
        if (acc as f64) >= target {
            return Some(bin_midpoint(i));
        }
    }
    Some(bin_midpoint(N_BINS - 1))
}

/// Round to one decimal place and append `%`. Matches the jq prototype's
/// formatting so existing analysis scripts read the same numbers.
fn pct(x: f64) -> String {
    let r = (x * 1000.0).round() / 10.0;
    format!("{r}%")
}

pub fn run(args: CfrArgs) -> Result<(), CfrDiagError> {
    if args.deadline_ms == 0 {
        return Err(CfrDiagError::InvalidDeadline);
    }
    let deadline_us = args.deadline_ms * 1000;

    let acc = if args.file.as_os_str() == "-" {
        let stdin = io::stdin();
        let reader = stdin.lock();
        ingest(reader, deadline_us)?
    } else {
        let file = File::open(&args.file).map_err(|e| CfrDiagError::Open {
            path: args.file.display().to_string(),
            source: e,
        })?;
        ingest(BufReader::new(file), deadline_us)?
    };

    let mut out = io::BufWriter::new(io::stdout().lock());
    render(&mut out, &acc, args.deadline_ms)?;
    Ok(())
}

fn ingest<R: BufRead>(mut reader: R, deadline_us: u64) -> Result<Accum, CfrDiagError> {
    let mut acc = Accum::new();
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let Ok(event) = serde_json::from_str::<Event>(&line) else {
            continue;
        };
        if event.target != "cfr_diag" {
            continue;
        }
        acc.observe(&event.fields, deadline_us);
    }
    Ok(acc)
}

fn render<W: io::Write>(w: &mut W, acc: &Accum, deadline_ms: u64) -> io::Result<()> {
    let mut depths: Vec<u64> = acc.depths.keys().copied().collect();
    depths.sort_unstable();

    writeln!(w, "stop_cause × depth (n={} events)", acc.total)?;
    write!(w, "  ")?;
    for (i, d) in depths.iter().enumerate() {
        if i > 0 {
            write!(w, "  ")?;
        }
        write!(w, "depth={d}")?;
    }
    writeln!(w)?;
    for cause in &acc.causes {
        write!(w, "{cause}  ")?;
        for (i, d) in depths.iter().enumerate() {
            if i > 0 {
                write!(w, "  ")?;
            }
            let n = acc.cross.get(&(cause.clone(), *d)).copied().unwrap_or(0);
            let tot = acc.depths.get(d).copied().unwrap_or(0);
            if tot == 0 {
                write!(w, "    -")?;
            } else {
                write!(w, "{}", pct(n as f64 / tot as f64))?;
            }
        }
        writeln!(w)?;
    }
    writeln!(w)?;

    writeln!(
        w,
        "deadline utilization (depth=0, deadline={}ms, n={})",
        deadline_ms, acc.d0_n
    )?;
    let labels = [
        "[0-25%)  ",
        "[25-50%) ",
        "[50-75%) ",
        "[75-95%) ",
        "[95-100%)",
        ">=100%   ",
    ];
    for (i, label) in labels.iter().enumerate() {
        let count = acc.ratios[i];
        let frac_str = if acc.d0_n == 0 {
            "-".to_string()
        } else {
            pct(count as f64 / acc.d0_n as f64)
        };
        writeln!(w, "{label} {count}  {frac_str}")?;
    }
    writeln!(w)?;

    writeln!(
        w,
        "final regret last(regret_series) quantiles by depth (approx, log2-binned)"
    )?;
    writeln!(w, "depth  n          p10        p50        p90        p99")?;
    for d in &depths {
        let hist = acc.regret.get(d);
        let n: u64 = hist.map(|h| h.iter().sum()).unwrap_or(0);
        if n == 0 {
            writeln!(w, "{d}      0          -          -          -          -")?;
            continue;
        }
        let h = hist.unwrap();
        let q10 = hist_quantile(h, 0.10).unwrap();
        let q50 = hist_quantile(h, 0.50).unwrap();
        let q90 = hist_quantile(h, 0.90).unwrap();
        let q99 = hist_quantile(h, 0.99).unwrap();
        let n_str = n.to_string();
        let n_pad = " ".repeat(11_usize.saturating_sub(n_str.len()));
        writeln!(w, "{d}      {n_str}{n_pad}{q10}    {q50}    {q90}    {q99}")?;
    }
    writeln!(w)?;

    writeln!(
        w,
        "convergence ratio last/first by depth (lower = converging more)"
    )?;
    writeln!(w, "depth  n          p10        p50        p90")?;
    for d in &depths {
        let hist = acc.conv.get(d);
        let n: u64 = hist.map(|h| h.iter().sum()).unwrap_or(0);
        if n == 0 {
            writeln!(w, "{d}      0          -          -          -")?;
            continue;
        }
        let h = hist.unwrap();
        let q10 = hist_quantile(h, 0.10).unwrap();
        let q50 = hist_quantile(h, 0.50).unwrap();
        let q90 = hist_quantile(h, 0.90).unwrap();
        let n_str = n.to_string();
        let n_pad = " ".repeat(11_usize.saturating_sub(n_str.len()));
        writeln!(w, "{d}      {n_str}{n_pad}{q10}    {q50}    {q90}")?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn run_to_string(input: &str, deadline_us: u64) -> String {
        let acc = ingest(Cursor::new(input.as_bytes()), deadline_us).unwrap();
        let mut out = Vec::new();
        render(&mut out, &acc, deadline_us / 1000).unwrap();
        String::from_utf8(out).unwrap()
    }

    fn event(depth: u64, stop_cause: &str, elapsed_us: u64, regret_series: &str) -> String {
        format!(
            r#"{{"timestamp":"2026-01-01T00:00:00Z","level":"TRACE","target":"cfr_diag","fields":{{"depth":{depth},"stop_cause":"{stop_cause}","final_iterations":1,"final_elapsed_us":{elapsed_us},"nodes_touched_start":0,"nodes_touched_end":0,"timer_armed":false,"actions_considered":2,"regret_series":"{regret_series}"}}}}"#
        )
    }

    #[test]
    fn empty_input_produces_skeleton_output() {
        let out = run_to_string("", 250_000);
        assert!(out.contains("stop_cause × depth (n=0 events)"));
        assert!(out.contains("deadline utilization (depth=0, deadline=250ms, n=0)"));
        assert!(out.contains("[0-25%)   0  -"));
    }

    #[test]
    fn non_json_lines_are_skipped() {
        let input = format!(
            "   Compiling tracing v0.1.41\n{}\nignored garbage\n",
            event(0, "deadline", 250_000, "[10.0, 5.0]")
        );
        let out = run_to_string(&input, 250_000);
        assert!(out.contains("stop_cause × depth (n=1 events)"));
        assert!(out.contains("deadline  100%"));
    }

    #[test]
    fn non_cfr_diag_target_skipped() {
        let other = r#"{"timestamp":"x","level":"INFO","target":"other","fields":{"msg":"hi"}}"#;
        let input = format!("{other}\n{}\n", event(0, "deadline", 250_000, "[10.0]"));
        let out = run_to_string(&input, 250_000);
        assert!(out.contains("stop_cause × depth (n=1 events)"));
    }

    #[test]
    fn deadline_utilization_buckets_correct() {
        // 4 events at depth=0 with ratios 0.1, 0.4, 0.8, 1.5
        let input = [
            event(0, "budget_stop", 25_000, "[1.0]"),  // 0.10 → [0-25%)
            event(0, "budget_stop", 100_000, "[1.0]"), // 0.40 → [25-50%)
            event(0, "budget_stop", 200_000, "[1.0]"), // 0.80 → [75-95%)
            event(0, "deadline", 375_000, "[1.0]"),    // 1.50 → >=100%
        ]
        .join("\n");
        let out = run_to_string(&input, 250_000);
        assert!(
            out.contains("[0-25%)   1  25%"),
            "missing [0-25%) row in:\n{out}"
        );
        assert!(out.contains("[25-50%)  1  25%"));
        assert!(out.contains("[75-95%)  1  25%"));
        assert!(out.contains(">=100%    1  25%"));
    }

    #[test]
    fn cross_tab_percentages_per_depth() {
        // depth=0: 1 deadline, 3 budget_stop  → 25% / 75%
        // depth=1: 2 budget_stop              → 100% budget_stop
        let input = [
            event(0, "deadline", 250_000, "[1.0]"),
            event(0, "budget_stop", 100_000, "[1.0]"),
            event(0, "budget_stop", 100_000, "[1.0]"),
            event(0, "budget_stop", 100_000, "[1.0]"),
            event(1, "budget_stop", 0, "[1.0]"),
            event(1, "budget_stop", 0, "[1.0]"),
        ]
        .join("\n");
        let out = run_to_string(&input, 250_000);
        assert!(out.contains("budget_stop  75%  100%"));
        assert!(out.contains("deadline  25%  0%"));
    }

    #[test]
    fn invalid_deadline_rejected() {
        let err = run(CfrArgs {
            file: PathBuf::from("/tmp/whatever"),
            deadline_ms: 0,
        })
        .unwrap_err();
        assert!(matches!(err, CfrDiagError::InvalidDeadline));
    }

    #[test]
    fn missing_file_errors() {
        let err = run(CfrArgs {
            file: PathBuf::from("/does/not/exist.jsonl"),
            deadline_ms: 250,
        })
        .unwrap_err();
        assert!(matches!(err, CfrDiagError::Open { .. }));
    }

    #[test]
    fn quantile_bin_midpoint_known_values() {
        // bin 60 covers [1.0, 2.0); midpoint = sqrt(2).
        let mut h = [0u64; N_BINS];
        h[60] = 100;
        let q = hist_quantile(&h, 0.5).unwrap();
        assert!((q - std::f64::consts::SQRT_2).abs() < 1e-9);
    }

    #[test]
    fn bin_of_handles_edge_cases() {
        assert_eq!(bin_of(0.0), 0);
        assert_eq!(bin_of(-1.0), 0);
        assert_eq!(bin_of(f64::NAN), 0);
        assert_eq!(bin_of(f64::INFINITY), 0);
        assert_eq!(bin_of(1.0), 60);
        assert_eq!(bin_of(2.0), 61);
        assert_eq!(bin_of(0.5), 59);
    }

    #[test]
    fn regret_series_first_zero_excluded_from_convergence() {
        // .[0] == 0 → division would be inf; should be skipped.
        let input = [
            event(0, "budget_stop", 0, "[0.0, 10.0]"),
            event(0, "budget_stop", 0, "[5.0, 10.0]"),
        ]
        .join("\n");
        let out = run_to_string(&input, 250_000);
        // Both events count in the final-regret section
        assert!(out.contains("0      2"));
        // But only the second contributes to convergence ratio
        let conv_section = out.split("convergence ratio").nth(1).unwrap();
        assert!(conv_section.contains("0      1"));
    }
}
