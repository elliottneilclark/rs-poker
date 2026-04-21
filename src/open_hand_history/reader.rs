//! Streaming reader for the OHH storage format.
//!
//! The on-disk format is specified at
//! <https://hh-specs.handhistory.org/storage-format>: a file is a
//! sequence of `{"ohh": ...}` JSON objects separated by at least one
//! blank line. It is *not* a valid top-level JSON array, so it cannot
//! be parsed with a single `serde_json::from_reader` call.
//!
//! [`HandReader`] yields one [`HandHistory`] at a time from either an
//! in-memory buffer or a filesystem path. Memory usage is bounded by
//! the size of a single record regardless of input size, so
//! multi-gigabyte inputs stream without issue.
//!
//! ## Construction
//!
//! * [`HandReader::open`] — a path that is either a `.ohh` file or a
//!   directory of `.ohh` files. Directories are enumerated in sorted
//!   filename order and non-`.ohh` files are skipped.
//! * [`HandReader::from_reader`] — any [`BufRead`] (Cursor, stdin,
//!   pipe, etc.). Useful for tests and for plugging the reader into
//!   non-file data sources.
//!
//! The reader tolerates:
//! * Pretty-printed records (the spec does not forbid embedded
//!   newlines inside a record; only the blank-line separator is
//!   required).
//! * Multiple consecutive blank lines between records.
//! * Leading or trailing blank lines.
//! * Parse errors on individual records — iteration continues past
//!   the next blank-line boundary so one bad hand cannot poison the
//!   rest of the file.
//!
//! # Examples
//!
//! Stream hands one at a time (bounded memory):
//!
//! ```no_run
//! use rs_poker::open_hand_history::HandReader;
//!
//! // Works for either a single file or a directory of .ohh files.
//! for item in HandReader::open("hands.ohh")? {
//!     let hand = item?;
//!     println!("hand {}", hand.game_number);
//! }
//! # Ok::<_, Box<dyn std::error::Error>>(())
//! ```
//!
//! Collect everything into a `Vec` when bounded memory doesn't matter
//! (e.g. a GUI viewer):
//!
//! ```no_run
//! use rs_poker::open_hand_history::{HandHistory, HandReader, ReaderError};
//!
//! let hands: Vec<HandHistory> = HandReader::open("hands.ohh")?
//!     .collect::<Result<Vec<_>, ReaderError>>()?;
//! # Ok::<_, ReaderError>(())
//! ```
use std::collections::VecDeque;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

use super::hand_history::{HandHistory, OpenHandHistoryWrapper};

/// Maximum characters of a failing record to include in a parse-error
/// message. Long enough to see the relevant context, short enough to
/// keep error output readable.
const PREVIEW_LEN: usize = 160;

/// Errors produced by [`HandReader`].
///
/// [`Io`](ReaderError::Io) and [`Parse`](ReaderError::Parse) carry a
/// 1-based line number that is local to the source being read. When
/// the reader is iterating a directory those errors are wrapped in
/// [`InFile`](ReaderError::InFile) so the offending filename is also
/// reported.
#[derive(Debug, thiserror::Error)]
pub enum ReaderError {
    /// Reading from the underlying buffered source failed.
    #[error("I/O error on line {line}: {source}")]
    Io {
        /// 1-based line number where the failure occurred.
        line: usize,
        /// Underlying cause.
        #[source]
        source: io::Error,
    },

    /// A record was not valid OHH JSON.
    #[error("parse error on line {line}: {source}\n  record preview: {preview}")]
    Parse {
        /// 1-based line number where the failing record began.
        line: usize,
        /// Truncated preview of the failing record.
        preview: String,
        /// Underlying cause.
        #[source]
        source: serde_json::Error,
    },

    /// Opening or listing a filesystem path failed.
    #[error("failed to access {path}: {source}")]
    Open {
        /// The file or directory that could not be opened.
        path: PathBuf,
        /// Underlying cause.
        #[source]
        source: io::Error,
    },

    /// An inner error, annotated with the file it came from.
    ///
    /// Produced when iterating a directory: the inner variant carries
    /// the per-file line number and is boxed to keep the enum compact.
    #[error("in {path}: {source}")]
    InFile {
        /// The file that produced the inner error.
        path: PathBuf,
        /// Underlying cause.
        #[source]
        source: Box<ReaderError>,
    },
}

/// Streaming reader over [`HandHistory`] records.
///
/// See the [module docs](self) for construction options and behavior.
/// `HandReader` is an [`Iterator`]: every call to [`Iterator::next`]
/// returns `Some(Ok(HandHistory))` for the next record,
/// `Some(Err(ReaderError))` on a malformed record or I/O failure, or
/// `None` once every input has been exhausted.
pub struct HandReader {
    state: State,
}

/// Internal dispatch between the two reader modes.
///
/// Both variants produce records through the same [`Stream`]
/// record-accumulator; the only difference is where the bytes come
/// from and whether errors get annotated with a path.
enum State {
    /// Finished — iteration returns `None` forever.
    Done,
    /// A single buffered source (file, stdin, Cursor, …).
    Single(Stream),
    /// A queue of `.ohh` files to read in order.
    Chain {
        pending: VecDeque<PathBuf>,
        current: Option<(PathBuf, Stream)>,
    },
}

impl HandReader {
    /// Open a filesystem path.
    ///
    /// * If `path` is a regular file, hands are read from it directly.
    /// * If `path` is a directory, every `.ohh` file inside it is
    ///   read in sorted filename order; non-`.ohh` files are skipped.
    /// * Any other kind of path (symlink to neither, missing, etc.)
    ///   returns [`ReaderError::Open`].
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, ReaderError> {
        let path = path.as_ref();

        if path.is_dir() {
            let pending = enumerate_ohh_files(path)?;
            Ok(Self {
                state: State::Chain {
                    pending,
                    current: None,
                },
            })
        } else {
            let stream = open_stream(path)?;
            Ok(Self {
                state: State::Single(stream),
            })
        }
    }

    /// Read hands from any buffered reader.
    ///
    /// The type is erased to `Box<dyn BufRead>` so the returned
    /// `HandReader` has a single concrete type regardless of the
    /// source. This matches the ergonomics of [`open`](Self::open)
    /// and keeps the iterator's item type consistent.
    pub fn from_reader<R: BufRead + 'static>(reader: R) -> Self {
        Self {
            state: State::Single(Stream::new(Box::new(reader))),
        }
    }
}

impl Iterator for HandReader {
    type Item = Result<HandHistory, ReaderError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match &mut self.state {
                State::Done => return None,
                State::Single(stream) => return stream.next_hand(),
                State::Chain { pending, current } => {
                    // Try to pull another record from the currently
                    // open file.
                    if let Some((path, stream)) = current.as_mut() {
                        if let Some(item) = stream.next_hand() {
                            return Some(item.map_err(|e| ReaderError::InFile {
                                path: path.clone(),
                                source: Box::new(e),
                            }));
                        }
                        // Current file exhausted — drop it.
                        *current = None;
                    }
                    // Open the next pending file, or finish.
                    match pending.pop_front() {
                        None => {
                            self.state = State::Done;
                            return None;
                        }
                        Some(path) => match open_stream(&path) {
                            Ok(stream) => *current = Some((path, stream)),
                            Err(e) => return Some(Err(e)),
                        },
                    }
                }
            }
        }
    }
}

/// Returns `true` if `path` has the `.ohh` extension
/// (case-insensitive).
///
/// Exposed so callers can pre-filter filesystem walks with the same
/// rule the reader uses internally.
pub fn has_ohh_extension(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .is_some_and(|ext| ext.eq_ignore_ascii_case("ohh"))
}

/// List `.ohh` files inside `dir`, sorted by filename.
///
/// Non-`.ohh` files (e.g. `results.json`, markdown reports produced
/// by `rsp arena compare`) are skipped, matching the same rule
/// [`HandReader::open`] applies when handed a directory.
///
/// Exposed for callers like the TUI hand store that maintain their
/// own index over the same set of files the reader would traverse.
pub fn ohh_files_in_dir(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && has_ohh_extension(p))
        .collect();
    entries.sort();
    Ok(entries)
}

/// Wrap [`ohh_files_in_dir`] for internal use, attaching the
/// directory path to any I/O error.
fn enumerate_ohh_files(dir: &Path) -> Result<VecDeque<PathBuf>, ReaderError> {
    ohh_files_in_dir(dir)
        .map(VecDeque::from)
        .map_err(|source| ReaderError::Open {
            path: dir.to_path_buf(),
            source,
        })
}

/// Open a single `.ohh` file as a buffered [`Stream`].
fn open_stream(path: &Path) -> Result<Stream, ReaderError> {
    let file = File::open(path).map_err(|source| ReaderError::Open {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(Stream::new(Box::new(BufReader::new(file))))
}

/// Per-source record accumulator.
///
/// Owns the [`BufRead`] and the scratch buffers used to stitch
/// multi-line records together. Line numbering is local to this
/// source — the [`HandReader::Chain`] mode wraps any errors in
/// [`ReaderError::InFile`] to add the filename.
struct Stream {
    inner: Box<dyn BufRead>,
    line: usize,
    scratch: String,
    record: String,
    record_start: usize,
    done: bool,
}

impl Stream {
    fn new(inner: Box<dyn BufRead>) -> Self {
        Self {
            inner,
            line: 0,
            scratch: String::new(),
            record: String::new(),
            record_start: 0,
            done: false,
        }
    }

    /// Return the next parsed hand from this source, or `None` at
    /// end of input. A parse error does not set `done` — the next
    /// call proceeds with the following record.
    fn next_hand(&mut self) -> Option<Result<HandHistory, ReaderError>> {
        if self.done {
            return None;
        }
        match self.read_record() {
            Ok(true) => Some(self.parse_current()),
            Ok(false) => {
                self.done = true;
                None
            }
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }

    /// Accumulate lines into `self.record` until a blank line or EOF
    /// terminates the record. Leading and duplicate blank lines are
    /// skipped.
    ///
    /// Returns `Ok(true)` if a record is ready, `Ok(false)` on clean
    /// EOF with no pending data.
    fn read_record(&mut self) -> Result<bool, ReaderError> {
        self.record.clear();
        self.record_start = 0;

        loop {
            self.scratch.clear();
            let bytes =
                self.inner
                    .read_line(&mut self.scratch)
                    .map_err(|source| ReaderError::Io {
                        line: self.line + 1,
                        source,
                    })?;

            if bytes == 0 {
                // EOF — any pending record is terminated by the file
                // boundary, matching how a blank line would.
                return Ok(!self.record.is_empty());
            }

            self.line += 1;

            if self.scratch.trim().is_empty() {
                if self.record.is_empty() {
                    // Leading / duplicate blank — skip.
                    continue;
                }
                return Ok(true);
            }

            if self.record.is_empty() {
                self.record_start = self.line;
            }
            self.record.push_str(&self.scratch);
        }
    }

    /// Parse whatever is currently accumulated in `self.record`.
    fn parse_current(&self) -> Result<HandHistory, ReaderError> {
        let trimmed = self.record.trim();
        serde_json::from_str::<OpenHandHistoryWrapper>(trimmed)
            .map(|w| w.ohh)
            .map_err(|source| ReaderError::Parse {
                line: self.record_start,
                preview: record_preview(trimmed),
                source,
            })
    }
}

/// Truncate a potentially long, potentially multi-line record to a
/// compact single-line preview for error messages.
fn record_preview(record: &str) -> String {
    let flat = record.replace('\n', " ");
    let mut out = String::new();
    for (i, c) in flat.chars().enumerate() {
        if i >= PREVIEW_LEN {
            out.push('…');
            break;
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::open_hand_history::{GameType, HandHistory, PlayerObj};
    use std::io::{Cursor, Write};
    use tempfile::{NamedTempFile, tempdir};

    fn sample_hand(id: &str) -> HandHistory {
        HandHistory {
            spec_version: "1.4.7".into(),
            site_name: "Site".into(),
            network_name: "Net".into(),
            internal_version: "1.0".into(),
            tournament: false,
            tournament_info: None,
            game_number: id.into(),
            start_date_utc: None,
            table_name: "T".into(),
            table_handle: None,
            table_skin: None,
            game_type: GameType::Holdem,
            bet_limit: None,
            table_size: 2,
            currency: "USD".into(),
            dealer_seat: 1,
            small_blind_amount: 1.0,
            big_blind_amount: 2.0,
            ante_amount: 0.0,
            hero_player_id: None,
            players: vec![PlayerObj {
                id: 1,
                seat: 1,
                name: "p".into(),
                display: None,
                starting_stack: 100.0,
                player_bounty: None,
                is_sitting_out: None,
            }],
            rounds: vec![],
            pots: vec![],
            tournament_bounties: None,
        }
    }

    /// Build a valid OHH blob from a list of hand IDs, using the
    /// canonical writer so tests match the real on-disk format.
    fn valid_blob(ids: &[&str]) -> Vec<u8> {
        let mut buf = Vec::new();
        for id in ids {
            super::super::write_hand(&mut buf, sample_hand(id)).unwrap();
        }
        buf
    }

    fn ids(hands: Vec<HandHistory>) -> Vec<String> {
        hands.into_iter().map(|h| h.game_number).collect()
    }

    fn reader_over(bytes: Vec<u8>) -> HandReader {
        HandReader::from_reader(Cursor::new(bytes))
    }

    #[test]
    fn empty_input_yields_no_hands() {
        assert!(
            reader_over(vec![])
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn blank_only_input_yields_no_hands() {
        assert!(
            reader_over(b"\n\n\n".to_vec())
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn reads_single_hand() {
        let r = reader_over(valid_blob(&["42"]));
        assert_eq!(ids(r.collect::<Result<Vec<_>, _>>().unwrap()), vec!["42"]);
    }

    #[test]
    fn reads_many_hands_in_order() {
        let r = reader_over(valid_blob(&["1", "2", "3"]));
        assert_eq!(
            ids(r.collect::<Result<Vec<_>, _>>().unwrap()),
            vec!["1", "2", "3"]
        );
    }

    #[test]
    fn tolerates_extra_blank_lines_between_records() {
        let mut blob = Vec::new();
        super::super::write_hand(&mut blob, sample_hand("1")).unwrap();
        blob.extend_from_slice(b"\n\n\n");
        super::super::write_hand(&mut blob, sample_hand("2")).unwrap();
        assert_eq!(
            ids(reader_over(blob).collect::<Result<Vec<_>, _>>().unwrap()),
            vec!["1", "2"]
        );
    }

    #[test]
    fn tolerates_leading_blank_lines() {
        let mut blob: Vec<u8> = b"\n\n".to_vec();
        blob.extend(valid_blob(&["1"]));
        assert_eq!(
            ids(reader_over(blob).collect::<Result<Vec<_>, _>>().unwrap()),
            vec!["1"]
        );
    }

    #[test]
    fn tolerates_final_record_without_trailing_blank() {
        let mut blob = serde_json::to_vec(&OpenHandHistoryWrapper {
            ohh: sample_hand("9"),
        })
        .unwrap();
        blob.push(b'\n');
        assert_eq!(
            ids(reader_over(blob).collect::<Result<Vec<_>, _>>().unwrap()),
            vec!["9"]
        );
    }

    #[test]
    fn parses_pretty_printed_records() {
        let a = serde_json::to_string_pretty(&OpenHandHistoryWrapper {
            ohh: sample_hand("a"),
        })
        .unwrap();
        let b = serde_json::to_string_pretty(&OpenHandHistoryWrapper {
            ohh: sample_hand("b"),
        })
        .unwrap();
        let blob = format!("{a}\n\n{b}\n").into_bytes();
        assert_eq!(
            ids(reader_over(blob).collect::<Result<Vec<_>, _>>().unwrap()),
            vec!["a", "b"]
        );
    }

    #[test]
    fn parse_error_reports_line_and_does_not_halt_iteration() {
        let mut blob = Vec::new();
        super::super::write_hand(&mut blob, sample_hand("1")).unwrap();
        blob.extend_from_slice(b"not valid json\n\n");
        super::super::write_hand(&mut blob, sample_hand("3")).unwrap();

        let results: Vec<_> = reader_over(blob).collect();
        assert_eq!(results.len(), 3, "expected ok, err, ok");
        assert_eq!(results[0].as_ref().unwrap().game_number, "1");

        match results[1].as_ref() {
            Err(ReaderError::Parse { line, preview, .. }) => {
                assert_eq!(*line, 3);
                assert!(preview.contains("not valid json"));
            }
            other => panic!("expected parse error, got {other:?}"),
        }

        assert_eq!(results[2].as_ref().unwrap().game_number, "3");
    }

    #[test]
    fn open_file_reads_hands() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&valid_blob(&["x", "y"])).unwrap();
        let hands = HandReader::open(f.path())
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(ids(hands), vec!["x", "y"]);
    }

    #[test]
    fn open_directory_chains_files_in_sorted_order() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("b.ohh"), valid_blob(&["beta"])).unwrap();
        std::fs::write(dir.path().join("a.ohh"), valid_blob(&["alpha1", "alpha2"])).unwrap();
        std::fs::write(dir.path().join("report.md"), b"# skip me").unwrap();
        std::fs::write(dir.path().join("skip.json"), b"not OHH").unwrap();

        let hands = HandReader::open(dir.path())
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(ids(hands), vec!["alpha1", "alpha2", "beta"]);
    }

    #[test]
    fn directory_errors_are_annotated_with_filename() {
        let dir = tempdir().unwrap();
        let good = dir.path().join("good.ohh");
        let bad = dir.path().join("zbad.ohh"); // sorts after "good"
        std::fs::write(&good, valid_blob(&["g"])).unwrap();
        std::fs::write(&bad, b"not valid json\n\n").unwrap();

        let results: Vec<_> = HandReader::open(dir.path()).unwrap().collect();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].as_ref().unwrap().game_number, "g");

        match results[1].as_ref() {
            Err(ReaderError::InFile { path, source }) => {
                assert_eq!(path, &bad);
                assert!(matches!(**source, ReaderError::Parse { .. }));
            }
            other => panic!("expected InFile/Parse, got {other:?}"),
        }
    }

    #[test]
    fn open_missing_path_returns_open_error() {
        match HandReader::open("/definitely/does/not/exist.ohh") {
            Ok(_) => panic!("expected error, got Ok"),
            Err(ReaderError::Open { path, .. }) => {
                assert_eq!(path, Path::new("/definitely/does/not/exist.ohh"));
            }
            Err(other) => panic!("expected Open, got {other:?}"),
        }
    }

    #[test]
    fn open_empty_directory_yields_no_hands() {
        let dir = tempdir().unwrap();
        let hands = HandReader::open(dir.path())
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(hands.is_empty());
    }

    #[test]
    fn ohh_files_in_dir_filters_and_sorts() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("b.ohh"), b"").unwrap();
        std::fs::write(dir.path().join("a.ohh"), b"").unwrap();
        std::fs::write(dir.path().join("skip.json"), b"").unwrap();
        std::fs::write(dir.path().join("README.md"), b"").unwrap();

        let files = ohh_files_in_dir(dir.path()).unwrap();
        let names: Vec<_> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(names, vec!["a.ohh", "b.ohh"]);
    }

    #[test]
    fn ohh_files_in_dir_missing_directory_is_error() {
        assert!(ohh_files_in_dir(Path::new("/does/not/exist")).is_err());
    }

    #[test]
    fn has_ohh_extension_is_case_insensitive() {
        assert!(has_ohh_extension(Path::new("hand.ohh")));
        assert!(has_ohh_extension(Path::new("hand.OHH")));
        assert!(has_ohh_extension(Path::new("/tmp/a.OhH")));
        assert!(!has_ohh_extension(Path::new("hand.json")));
        assert!(!has_ohh_extension(Path::new("hand")));
    }

    #[test]
    fn record_preview_flattens_newlines_and_truncates() {
        assert_eq!(record_preview("a\nb\nc"), "a b c");
        let long = "x".repeat(PREVIEW_LEN + 50);
        let p = record_preview(&long);
        assert!(p.ends_with('…'));
        assert_eq!(p.chars().count(), PREVIEW_LEN + 1);
    }
}
