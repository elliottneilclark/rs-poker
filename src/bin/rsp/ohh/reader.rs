use std::ffi::OsStr;
use std::path::Path;

use rs_poker::open_hand_history::{HandHistory, OpenHandHistoryWrapper};

#[derive(Debug, thiserror::Error)]
pub enum ReaderError {
    #[error("failed to read file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse line {line}: {source}")]
    Parse {
        line: usize,
        source: serde_json::Error,
    },
}

/// Read all `.ohh` files from a directory (sorted by filename).
///
/// Files that do not have an `.ohh` extension are skipped. Without this
/// filter, a directory that happens to contain other artifacts (e.g.
/// `results.json`, `hands.jsonl`, or markdown reports produced by
/// `rsp arena compare`) would cause the reader to try parsing them as
/// JSONL OHH and fail on the first non-OHH line.
pub fn read_ohh_dir(dir: &Path) -> Result<Vec<HandHistory>, ReaderError> {
    let mut entries: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && has_ohh_extension(p))
        .collect();
    entries.sort();

    let mut all_hands = Vec::new();
    for path in entries {
        let hands = read_ohh_file(&path)?;
        all_hands.extend(hands);
    }
    Ok(all_hands)
}

/// Returns `true` if the given path's extension is `ohh`
/// (case-insensitive).
pub fn has_ohh_extension(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .is_some_and(|ext| ext.eq_ignore_ascii_case("ohh"))
}

/// Read an OHH file (JSONL format: one JSON object per line).
pub fn read_ohh_file(path: &Path) -> Result<Vec<HandHistory>, ReaderError> {
    let content = std::fs::read_to_string(path)?;
    let mut hands = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let wrapper: OpenHandHistoryWrapper =
            serde_json::from_str(trimmed).map_err(|e| ReaderError::Parse {
                line: line_num + 1,
                source: e,
            })?;
        hands.push(wrapper.ohh);
    }

    Ok(hands)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::{NamedTempFile, tempdir};

    #[test]
    fn test_empty_file() {
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "").unwrap();
        let hands = read_ohh_file(f.path()).unwrap();
        assert!(hands.is_empty());
    }

    #[test]
    fn test_invalid_json_returns_error() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "not json").unwrap();
        let result = read_ohh_file(f.path());
        assert!(result.is_err());
    }

    /// Regression test for B6: `read_ohh_dir` must ignore non-`.ohh`
    /// files. Previously it walked every regular file and tried to parse
    /// them, so running it on an `arena compare` output directory (which
    /// also contains `results.json`, `hands.jsonl`, `results.md`, etc.)
    /// would fail on the first non-OHH line.
    #[test]
    fn test_read_ohh_dir_skips_non_ohh_files() {
        let dir = tempdir().unwrap();
        // An empty .ohh file (valid: 0 hands)
        fs::write(dir.path().join("a.ohh"), "").unwrap();
        // Unrelated files that would fail to parse as JSONL OHH.
        fs::write(dir.path().join("results.json"), "{ invalid ohh }").unwrap();
        fs::write(dir.path().join("report.md"), "# Report\n").unwrap();
        fs::write(dir.path().join("hands.jsonl"), "not a hand").unwrap();

        let hands = read_ohh_dir(dir.path()).unwrap();
        assert!(hands.is_empty());
    }

    #[test]
    fn test_has_ohh_extension() {
        assert!(has_ohh_extension(Path::new("hand.ohh")));
        assert!(has_ohh_extension(Path::new("/tmp/hand.OHH")));
        assert!(!has_ohh_extension(Path::new("hand.json")));
        assert!(!has_ohh_extension(Path::new("results")));
    }
}
