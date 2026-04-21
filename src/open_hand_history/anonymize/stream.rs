//! Streaming driver: [`HandReader`] → [`Anonymizer`] → [`write_hand`].
//!
//! This module is the glue that binds the three building blocks
//! together. All of the real work lives elsewhere:
//!
//! * Parsing the OHH storage format — [`HandReader`].
//! * Rewriting identifying fields — [`Anonymizer`].
//! * Emitting the OHH storage format — [`write_hand`].
//!
//! Because each of those is independently tested, this file is
//! almost entirely plumbing. It exists so callers can ask a single
//! question — "anonymize this stream" — without knowing about
//! readers, writers, or the on-disk layout.
use std::io::{BufRead, Write};

use super::anonymizer::Anonymizer;
use crate::open_hand_history::{HandReader, ReaderError, write_hand};

/// Errors produced by [`anonymize_stream`].
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    /// The input could not be parsed as an OHH stream.
    #[error(transparent)]
    Read(#[from] ReaderError),
    /// Writing a record to the output failed.
    #[error("write error: {0}")]
    Write(#[source] std::io::Error),
}

/// Anonymize every hand read from `input` and write the results to
/// `output`.
///
/// Both sides stream: input is pulled one record at a time via
/// [`HandReader`], and output is emitted one record at a time via
/// [`write_hand`]. Memory usage stays O(one hand) regardless of
/// input size.
///
/// The anonymizer is taken by mutable reference so its learned state
/// (stable name map, site rotations, global time shift) persists
/// across the whole stream and so callers can inspect it after the
/// call returns.
///
/// On success, returns the number of hands processed.
pub fn anonymize_stream<R, W>(
    input: R,
    mut output: W,
    anonymizer: &mut Anonymizer,
) -> Result<usize, StreamError>
where
    R: BufRead + 'static,
    W: Write,
{
    let mut count = 0usize;
    for item in HandReader::from_reader(input) {
        let mut hand = item?;
        anonymizer.anonymize(&mut hand);
        write_hand(&mut output, hand).map_err(StreamError::Write)?;
        count += 1;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::super::config::{AnonymizeConfig, NameStrategy};
    use super::*;
    use crate::open_hand_history::{GameType, HandHistory, HandReader, PlayerObj, write_hand};
    use std::io::Cursor;

    fn sample_hand(id: &str, player_name: &str) -> HandHistory {
        HandHistory {
            spec_version: "1.4.7".into(),
            site_name: "RealSite".into(),
            network_name: "RealNet".into(),
            internal_version: "9.9.9".into(),
            tournament: false,
            tournament_info: None,
            game_number: id.into(),
            start_date_utc: None,
            table_name: "RealTable".into(),
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
                name: player_name.into(),
                display: None,
                starting_stack: 200.0,
                player_bounty: None,
                is_sitting_out: None,
            }],
            rounds: vec![],
            pots: vec![],
            tournament_bounties: None,
        }
    }

    fn blob(hands: &[HandHistory]) -> Vec<u8> {
        let mut out = Vec::new();
        for h in hands {
            write_hand(&mut out, h.clone()).unwrap();
        }
        out
    }

    fn read_back(buf: Vec<u8>) -> Vec<HandHistory> {
        // `HandReader::from_reader` requires `'static`, so we take
        // ownership of the buffer.
        HandReader::from_reader(Cursor::new(buf))
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    }

    fn config() -> AnonymizeConfig {
        AnonymizeConfig {
            seed: Some(1),
            name_strategy: NameStrategy::Stable,
            ..AnonymizeConfig::default()
        }
    }

    #[test]
    fn stream_preserves_hand_count() {
        let input = blob(&[sample_hand("1", "Alice"), sample_hand("2", "Bob")]);
        let mut out = Vec::new();
        let mut a = Anonymizer::new(config());
        let n = anonymize_stream(Cursor::new(input), &mut out, &mut a).unwrap();
        assert_eq!(n, 2);
        assert_eq!(read_back(out.clone()).len(), 2);
    }

    #[test]
    fn stream_anonymizes_names_consistently() {
        let input = blob(&[sample_hand("1", "Alice"), sample_hand("2", "Alice")]);
        let mut out = Vec::new();
        let mut a = Anonymizer::new(config());
        anonymize_stream(Cursor::new(input), &mut out, &mut a).unwrap();
        let hands = read_back(out.clone());
        assert_eq!(hands[0].players[0].name, hands[1].players[0].name);
        assert_ne!(hands[0].players[0].name, "Alice");
    }

    #[test]
    fn output_round_trips_through_hand_reader() {
        // The output of the anonymizer must itself be a valid OHH
        // file — HandReader should parse it back without error.
        let input = blob(&[sample_hand("1", "Alice")]);
        let mut out = Vec::new();
        let mut a = Anonymizer::new(config());
        anonymize_stream(Cursor::new(input), &mut out, &mut a).unwrap();
        let parsed = read_back(out.clone());
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn parse_error_is_surfaced() {
        let mut out = Vec::new();
        let mut a = Anonymizer::new(config());
        let err = anonymize_stream(Cursor::new("not valid json\n"), &mut out, &mut a).unwrap_err();
        assert!(matches!(err, StreamError::Read(_)));
    }
}
