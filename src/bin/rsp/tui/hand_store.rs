use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use rs_poker::open_hand_history::{HandHistory, OpenHandHistoryWrapper};

use crate::tui::state::GameLogEntry;

#[derive(Debug, thiserror::Error)]
pub enum HandStoreError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
}

struct Inner {
    path: Option<PathBuf>,
    offsets: Vec<u64>,
    file: Option<File>,
}

/// Disk-backed index into an OHH JSONL file for on-demand HandHistory loading.
///
/// Thread-safe via `Arc<Mutex<Inner>>`. The writer thread calls `push_offset`
/// (one `u64` append) while the TUI calls `fetch` on user keypress. Lock
/// contention is negligible.
#[derive(Clone)]
pub struct HandStore(Arc<Mutex<Inner>>);

impl HandStore {
    /// No OHH file available; `fetch` always returns `Ok(None)`.
    pub fn none() -> Self {
        Self(Arc::new(Mutex::new(Inner {
            path: None,
            offsets: Vec::new(),
            file: None,
        })))
    }

    /// For live simulations: index built incrementally via `push_offset`.
    pub fn new(path: PathBuf) -> Self {
        Self(Arc::new(Mutex::new(Inner {
            path: Some(path),
            offsets: Vec::new(),
            file: None,
        })))
    }

    /// For static viewers (`ohh view`): scans the file to build the full index.
    pub fn from_existing(path: &Path) -> Result<Self, HandStoreError> {
        let file = File::open(path)?;
        let reader = BufReader::new(&file);
        let mut offsets = Vec::new();
        let mut pos: u64 = 0;
        let mut in_record = false;

        for line in reader.lines() {
            let line = line?;
            let line_bytes = line.len() as u64 + 1; // +1 for the newline
            if !line.trim().is_empty() {
                if !in_record {
                    offsets.push(pos);
                    in_record = true;
                }
            } else {
                in_record = false;
            }
            pos += line_bytes;
        }

        Ok(Self(Arc::new(Mutex::new(Inner {
            path: Some(path.to_path_buf()),
            offsets,
            file: None, // lazy-open on first fetch
        }))))
    }

    /// Record the byte offset of a newly written game in the JSONL file.
    /// Called by the simulation thread after each game is flushed to disk.
    pub fn push_offset(&self, offset: u64) {
        let mut inner = self.0.lock().unwrap();
        inner.offsets.push(offset);
    }

    /// Load a HandHistory by game number (1-based).
    /// Returns `Ok(None)` if no OHH file is configured or game_number is out of range.
    pub fn fetch(&self, game_number: usize) -> Result<Option<HandHistory>, HandStoreError> {
        let mut inner = self.0.lock().unwrap();
        let path = match inner.path {
            Some(ref p) => p.clone(),
            None => return Ok(None),
        };

        let idx = game_number.saturating_sub(1);
        if idx >= inner.offsets.len() {
            return Ok(None);
        }
        let offset = inner.offsets[idx];

        // Lazy-open the read handle
        let file = match inner.file {
            Some(ref mut f) => f,
            None => {
                inner.file = Some(File::open(&path)?);
                inner.file.as_mut().unwrap()
            }
        };

        file.seek(SeekFrom::Start(offset))?;
        let mut reader = BufReader::new(&*file);
        let mut line = String::new();
        reader.read_line(&mut line)?;

        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }

        let wrapper: OpenHandHistoryWrapper = serde_json::from_str(trimmed)?;
        Ok(Some(wrapper.ohh))
    }

    /// Number of indexed hands.
    pub fn len(&self) -> usize {
        self.0.lock().unwrap().offsets.len()
    }

    /// Load a `GameLogEntry` by game number (1-based).
    /// Returns `Ok(None)` if no OHH file is configured or game_number is out of range.
    pub fn fetch_entry(&self, game_number: usize) -> Result<Option<GameLogEntry>, HandStoreError> {
        let hand = match self.fetch(game_number)? {
            Some(h) => h,
            None => return Ok(None),
        };
        Ok(Some(GameLogEntry::from_hand(game_number, &hand)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rs_poker::open_hand_history::{GameType, OpenHandHistoryWrapper};
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_test_hand(game_number: &str) -> HandHistory {
        HandHistory {
            spec_version: "1.4.7".into(),
            site_name: "test".into(),
            network_name: "test".into(),
            internal_version: "1.0".into(),
            tournament: false,
            tournament_info: None,
            game_number: game_number.into(),
            start_date_utc: None,
            table_name: "test".into(),
            table_handle: None,
            table_skin: None,
            game_type: GameType::Holdem,
            bet_limit: None,
            table_size: 2,
            currency: "USD".into(),
            dealer_seat: 0,
            small_blind_amount: 5.0,
            big_blind_amount: 10.0,
            ante_amount: 0.0,
            hero_player_id: None,
            players: vec![],
            rounds: vec![],
            pots: vec![],
            tournament_bounties: None,
        }
    }

    /// Write a hand to the file in the same format as append_hand (JSON + \n\n).
    /// Returns the byte offset where this hand starts.
    fn write_hand(file: &mut std::fs::File, hand: HandHistory) -> u64 {
        use std::io::Seek;
        let offset = file.stream_position().unwrap();
        let wrapped = OpenHandHistoryWrapper { ohh: hand };
        serde_json::to_writer(&mut *file, &wrapped).unwrap();
        writeln!(file).unwrap();
        writeln!(file).unwrap();
        offset
    }

    #[test]
    fn test_none_store_fetch_returns_none() {
        let store = HandStore::none();
        assert_eq!(store.len(), 0);
        let result = store.fetch(1).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_new_store_starts_empty() {
        let store = HandStore::new(PathBuf::from("/tmp/nonexistent.ohh"));
        assert_eq!(store.len(), 0);
        let result = store.fetch(1).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_push_offset_and_fetch() {
        let mut tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        // Write two hands, tracking offsets
        let offset1 = write_hand(tmp.as_file_mut(), make_test_hand("1"));
        let offset2 = write_hand(tmp.as_file_mut(), make_test_hand("2"));

        // Build store incrementally (simulating live sim)
        let store = HandStore::new(path);
        store.push_offset(offset1);
        store.push_offset(offset2);

        assert_eq!(store.len(), 2);

        // Fetch game 1
        let hand1 = store.fetch(1).unwrap().expect("should find game 1");
        assert_eq!(hand1.game_number, "1");

        // Fetch game 2
        let hand2 = store.fetch(2).unwrap().expect("should find game 2");
        assert_eq!(hand2.game_number, "2");

        // Fetch out-of-range returns None
        assert!(store.fetch(3).unwrap().is_none());
    }

    #[test]
    fn test_from_existing_scans_file() {
        let mut tmp = NamedTempFile::new().unwrap();

        // Write three hands
        write_hand(tmp.as_file_mut(), make_test_hand("10"));
        write_hand(tmp.as_file_mut(), make_test_hand("20"));
        write_hand(tmp.as_file_mut(), make_test_hand("30"));

        // Build store from existing file
        let store = HandStore::from_existing(tmp.path()).unwrap();
        assert_eq!(store.len(), 3);

        let hand1 = store.fetch(1).unwrap().expect("game 1");
        assert_eq!(hand1.game_number, "10");

        let hand3 = store.fetch(3).unwrap().expect("game 3");
        assert_eq!(hand3.game_number, "30");
    }

    #[test]
    fn test_clone_shares_state() {
        let store = HandStore::new(PathBuf::from("/tmp/test.ohh"));
        let clone = store.clone();

        store.push_offset(0);
        store.push_offset(100);

        // Clone sees the same offsets
        assert_eq!(clone.len(), 2);
    }

    #[test]
    fn test_fetch_game_zero_returns_none() {
        let store = HandStore::none();
        // game_number 0 is invalid (1-based), should not panic
        assert!(store.fetch(0).unwrap().is_none());
    }
}
