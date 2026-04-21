use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::Path;

use super::hand_history::{HandHistory, OpenHandHistoryWrapper};

/// Serialize one hand in the OHH storage format and write it to
/// `writer`.
///
/// The on-disk format is a single-line JSON `{"ohh": ...}` object
/// followed by `\n\n`: the first newline terminates the record, the
/// second is the blank separator line mandated by
/// <https://hh-specs.handhistory.org/storage-format>. Serialization
/// happens into a scratch buffer so the whole record reaches `writer`
/// in one `write_all` call, which keeps it atomic at EOF for
/// `O_APPEND` files on Linux.
pub fn write_hand<W: Write>(writer: &mut W, hand: HandHistory) -> io::Result<()> {
    let wrapped = OpenHandHistoryWrapper { ohh: hand };
    let mut buf = serde_json::to_vec(&wrapped)?;
    buf.extend_from_slice(b"\n\n");
    writer.write_all(&buf)
}

/// Appends a hand history to a file in the OHH storage format.
///
/// Opens the file with `O_APPEND`; the kernel locks the inode around
/// seek-to-EOF + write, so the single `write_all` inside
/// [`write_hand`] is atomic at EOF and concurrent writers cannot
/// interleave their JSON mid-record.
pub fn append_hand(path: &Path, hand: HandHistory) -> io::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    write_hand(&mut file, hand)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;

    #[test]
    fn test_append_hand() {
        // Create temp file that gets cleaned up after test
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Create minimal hand history for testing
        let hand = HandHistory {
            spec_version: "1.4.7".to_string(),
            site_name: "Test Site".to_string(),
            network_name: "Test Network".to_string(),
            internal_version: "1.0".to_string(),
            tournament: false,
            tournament_info: None,
            game_number: "123".to_string(),
            start_date_utc: None,
            table_name: "Test Table".to_string(),
            table_handle: None,
            table_skin: None,
            game_type: super::super::hand_history::GameType::Holdem,
            bet_limit: None,
            table_size: 9,
            currency: "USD".to_string(),
            dealer_seat: 1,
            small_blind_amount: 1.0,
            big_blind_amount: 2.0,
            ante_amount: 0.0,
            hero_player_id: None,
            players: vec![],
            rounds: vec![],
            pots: vec![],
            tournament_bounties: None,
        };

        // Write hand to file
        append_hand(path, hand).unwrap();

        // Verify file contents
        let contents = fs::read_to_string(path).unwrap();
        dbg!(&contents);
        assert!(contents.contains("Test Site"));
        assert!(contents.ends_with("\n\n")); // Check for double newline at end

        // Parse contents back to verify format
        let parsed: OpenHandHistoryWrapper = serde_json::from_str(contents.trim()).unwrap();
        assert_eq!(parsed.ohh.site_name, "Test Site");
    }
}
