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
    use std::io::Write;
    use tempfile::NamedTempFile;

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
}
