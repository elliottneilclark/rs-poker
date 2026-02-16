//! Open Hand History Historian Implementation
//!
//! This module provides the `OpenHandHistoryHistorian` which implements the `Historian` trait
//! to record arena game simulations in the standardized Open Hand History (OHH) JSON format.

use std::{cell::RefCell, path::PathBuf, rc::Rc};

use tracing::{debug, instrument, trace};

use crate::arena::game_state::Round;
use crate::arena::{GameState, action::Action};

use crate::arena::historian::{Historian, HistorianError};

use crate::open_hand_history::{ConverterConfig, HandHistory, HandHistoryBuilder, append_hand};

/// A historian that records game simulations in Open Hand History format
///
/// This historian implements the `Historian` trait to convert arena game simulations
/// into the standardized Open Hand History (OHH) JSON format and write them to a file
/// in JSON Lines format (one JSON object per line).
///
/// ## Output Format
///
/// The historian writes each completed game as a single JSON object on its own line
/// in the output file. This JSON Lines format makes it easy to process multiple games
/// and append new games to existing files.
///
/// ## Usage
///
/// ```no_run
/// # #[cfg(all(feature = "open-hand-history", feature = "arena"))] {
/// use rs_poker::arena::historian::OpenHandHistoryHistorian;
/// use rs_poker::open_hand_history::ConverterConfig;
/// use std::path::PathBuf;
///
/// // With default configuration
/// let historian = OpenHandHistoryHistorian::new(PathBuf::from("hands.jsonl"));
///
/// // With custom configuration
/// let config = ConverterConfig {
///     site_name: "my_site".to_string(),
///     network_name: "my_network".to_string(),
///     currency: "EUR".to_string(),
/// };
/// let historian = OpenHandHistoryHistorian::new_with_config(
///     PathBuf::from("hands.jsonl"),
///     config
/// );
/// # }
/// ```
#[derive(Debug)]
pub struct OpenHandHistoryHistorian {
    output_path: PathBuf,
    builder: Option<HandHistoryBuilder>,
}

impl OpenHandHistoryHistorian {
    /// Create a new OpenHandHistoryHistorian with default configuration
    ///
    /// Uses default values:
    /// - site_name: "rs_poker"
    /// - network_name: "rs_poker_arena"
    /// - currency: "USD"
    pub fn new(output_path: PathBuf) -> Self {
        Self::new_with_config(output_path, ConverterConfig::default())
    }

    /// Create a new OpenHandHistoryHistorian with custom configuration
    ///
    /// Allows customization of site name, network name, and currency that will
    /// appear in the generated OHH files.
    pub fn new_with_config(output_path: PathBuf, config: ConverterConfig) -> Self {
        Self {
            output_path,
            builder: Some(HandHistoryBuilder::new(config)),
        }
    }
}

impl Historian for OpenHandHistoryHistorian {
    #[instrument(level = "trace", skip(self, game_state), fields(output_path = ?self.output_path))]
    fn record_action(
        &mut self,
        id: u128,
        game_state: &GameState,
        action: Action,
    ) -> Result<(), HistorianError> {
        let builder = self
            .builder
            .as_mut()
            .ok_or(HistorianError::UnableToRecordAction)?;

        // Record the action (builder handles game_id internally)
        builder.record_action(id, &action, game_state)?;

        // Check if game is complete
        if matches!(game_state.round, Round::Complete) {
            let completed_builder = self
                .builder
                .take()
                .ok_or(HistorianError::UnableToRecordAction)?;

            // Build hand history (consumes builder)
            let hand_history = completed_builder.build()?;

            // Ensure output directory exists
            if let Some(parent) = self.output_path.parent() {
                debug!(?parent, "Creating output directory");
                std::fs::create_dir_all(parent)?;
            }

            debug!(id, ?self.output_path, "Writing completed hand to OHH file");

            // Write to file using existing writer
            append_hand(&self.output_path, hand_history)?;
        }

        Ok(())
    }
}

/// A historian that records hand histories into in-memory storage for later inspection.
///
/// This mirrors the behavior of [`VecHistorian`](crate::arena::historian::VecHistorian) but
/// stores fully converted Open Hand History records instead of raw arena actions.
#[derive(Debug)]
pub struct OpenHandHistoryVecHistorian {
    builder: Option<HandHistoryBuilder>,
    storage: Rc<RefCell<Vec<HandHistory>>>,
}

impl OpenHandHistoryVecHistorian {
    /// Create a new vector-backed historian using the default converter configuration.
    pub fn new() -> Self {
        Self::new_with_config(
            Rc::new(RefCell::new(Vec::new())),
            ConverterConfig::default(),
        )
    }

    /// Create a historian with a custom converter configuration.
    pub fn new_with_config(
        storage: Rc<RefCell<Vec<HandHistory>>>,
        config: ConverterConfig,
    ) -> Self {
        Self {
            builder: Some(HandHistoryBuilder::new(config)),
            storage,
        }
    }

    /// Create a historian backed by the provided storage and default converter configuration.
    pub fn new_with_storage(storage: Rc<RefCell<Vec<HandHistory>>>) -> Self {
        Self::new_with_config(storage, ConverterConfig::default())
    }

    /// Access the underlying storage so tests/fuzz targets can inspect recorded hands.
    pub fn get_storage(&self) -> Rc<RefCell<Vec<HandHistory>>> {
        self.storage.clone()
    }
}

impl Default for OpenHandHistoryVecHistorian {
    fn default() -> Self {
        Self::new()
    }
}

impl Historian for OpenHandHistoryVecHistorian {
    #[instrument(level = "trace", skip(self, game_state))]
    fn record_action(
        &mut self,
        id: u128,
        game_state: &GameState,
        action: Action,
    ) -> Result<(), HistorianError> {
        let builder = self
            .builder
            .as_mut()
            .ok_or(HistorianError::UnableToRecordAction)?;

        // Record the action (builder handles game_id internally)
        builder.record_action(id, &action, game_state)?;

        // Check if game is complete
        if matches!(game_state.round, Round::Complete) {
            let completed_builder = self
                .builder
                .take()
                .ok_or(HistorianError::UnableToRecordAction)?;

            // Build hand history (consumes builder)
            let hand_history = completed_builder.build()?;

            // Store in memory
            let mut storage = self.storage.try_borrow_mut()?;
            storage.push(hand_history);
            trace!(
                id,
                storage_count = storage.len(),
                "Stored completed hand in memory"
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{
        Agent, GameStateBuilder, HoldemSimulationBuilder,
        action::{Action, AwardPayload, GameStartPayload},
        agent::CallingAgent,
        game_state::GameState,
    };
    use crate::open_hand_history::OpenHandHistoryWrapper;
    use rand::rng;
    use tempfile::NamedTempFile;

    fn create_test_game_state() -> GameState {
        GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .blinds(2.0, 1.0)
            .build()
            .unwrap()
    }

    fn record_simple_hand<H: Historian>(historian: &mut H, game_id: u128) {
        let mut game_state = create_test_game_state();
        historian
            .record_action(
                game_id,
                &game_state,
                Action::GameStart(GameStartPayload {
                    small_blind: 1.0,
                    big_blind: 2.0,
                    ante: 0.0,
                }),
            )
            .unwrap();

        game_state.complete();
        historian
            .record_action(
                game_id,
                &game_state,
                Action::Award(AwardPayload {
                    idx: 0,
                    award_amount: 10.0,
                    total_pot: 10.0,
                    rank: None,
                    hand: None,
                }),
            )
            .unwrap();
    }

    #[test]
    fn test_historian_creation_with_default_config() {
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_path_buf();
        temp_file.close().unwrap();

        let mut historian = OpenHandHistoryHistorian::new(temp_path.clone());
        let mut game_state = create_test_game_state();

        historian
            .record_action(
                12345,
                &game_state,
                Action::GameStart(GameStartPayload {
                    small_blind: 1.0,
                    big_blind: 2.0,
                    ante: 0.0,
                }),
            )
            .unwrap();

        game_state.complete();
        historian
            .record_action(
                12345,
                &game_state,
                Action::Award(AwardPayload {
                    idx: 0,
                    award_amount: 10.0,
                    total_pot: 10.0,
                    rank: None,
                    hand: None,
                }),
            )
            .unwrap();

        let content = std::fs::read_to_string(&temp_path).unwrap();
        let wrapper: OpenHandHistoryWrapper = serde_json::from_str(content.trim()).unwrap();

        assert_eq!(wrapper.ohh.site_name, "rs_poker");
        assert_eq!(wrapper.ohh.network_name, "rs_poker_arena");
        assert_eq!(wrapper.ohh.currency, "USD");
    }

    #[test]
    fn test_historian_creation_with_custom_config() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = ConverterConfig {
            site_name: "custom_site".to_string(),
            network_name: "custom_network".to_string(),
            currency: "EUR".to_string(),
        };
        let temp_path = temp_file.path().to_path_buf();
        temp_file.close().unwrap();

        let mut historian = OpenHandHistoryHistorian::new_with_config(temp_path.clone(), config);
        let mut game_state = create_test_game_state();

        historian
            .record_action(
                67890,
                &game_state,
                Action::GameStart(GameStartPayload {
                    small_blind: 1.0,
                    big_blind: 2.0,
                    ante: 0.0,
                }),
            )
            .unwrap();

        game_state.complete();
        historian
            .record_action(
                67890,
                &game_state,
                Action::Award(AwardPayload {
                    idx: 1,
                    award_amount: 5.0,
                    total_pot: 5.0,
                    rank: None,
                    hand: None,
                }),
            )
            .unwrap();

        let content = std::fs::read_to_string(&temp_path).unwrap();
        let wrapper: OpenHandHistoryWrapper = serde_json::from_str(content.trim()).unwrap();

        assert_eq!(wrapper.ohh.site_name, "custom_site");
        assert_eq!(wrapper.ohh.network_name, "custom_network");
        assert_eq!(wrapper.ohh.currency, "EUR");
    }

    #[test]
    fn test_file_not_created_until_game_completes() {
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_path_buf();
        temp_file.close().unwrap(); // Remove the file so we can check it doesn't exist

        let mut historian = OpenHandHistoryHistorian::new(temp_path.clone());
        let mut game_state = create_test_game_state();

        // Start game but don't complete it
        historian
            .record_action(
                12345,
                &game_state,
                Action::GameStart(GameStartPayload {
                    small_blind: 1.0,
                    big_blind: 2.0,
                    ante: 0.0,
                }),
            )
            .unwrap();

        // File should not exist yet
        assert!(!temp_path.exists());

        // Complete the game and record an award
        game_state.complete();
        historian
            .record_action(
                12345,
                &game_state,
                Action::Award(AwardPayload {
                    idx: 0,
                    award_amount: 10.0,
                    total_pot: 10.0,
                    rank: None,
                    hand: None,
                }),
            )
            .unwrap();

        // Now file should exist
        assert!(temp_path.exists());
    }

    #[test]
    fn test_historian_with_simple_game_completion() {
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_path_buf();
        temp_file.close().unwrap();

        let mut historian = OpenHandHistoryHistorian::new(temp_path.clone());
        let mut game_state = create_test_game_state();

        // Simple game start
        historian
            .record_action(
                12345,
                &game_state,
                Action::GameStart(GameStartPayload {
                    small_blind: 1.0,
                    big_blind: 2.0,
                    ante: 0.0,
                }),
            )
            .unwrap();

        // Complete the game
        game_state.complete();
        historian
            .record_action(
                12345,
                &game_state,
                Action::Award(AwardPayload {
                    idx: 0,
                    award_amount: 200.0,
                    total_pot: 200.0,
                    rank: None,
                    hand: None,
                }),
            )
            .unwrap();

        // Verify file was created
        assert!(temp_path.exists());

        // Verify we can read the file back
        let content = std::fs::read_to_string(&temp_path).unwrap();
        assert!(!content.is_empty());

        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert!(parsed["ohh"].is_object());
    }

    #[test]
    fn test_vec_historian_collects_hand_history() {
        let historian = Box::new(OpenHandHistoryVecHistorian::new());
        let storage = historian.get_storage();

        let stacks = vec![50.0; 2];
        let agents: Vec<Box<dyn Agent>> = vec![
            Box::<CallingAgent>::default(),
            Box::<CallingAgent>::default(),
        ];
        let game_state = GameStateBuilder::new()
            .stacks(stacks)
            .blinds(2.0, 1.0)
            .build()
            .unwrap();
        let mut rng = rng();

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![historian])
            .build()
            .unwrap();

        sim.run(&mut rng);

        let hands = storage.borrow();
        assert_eq!(hands.len(), 1);
    }

    #[test]
    fn test_vec_historian_records_single_hand() {
        let mut historian = OpenHandHistoryVecHistorian::new();
        let storage = historian.get_storage();

        record_simple_hand(&mut historian, 1);

        assert_eq!(storage.borrow().len(), 1);
    }

    #[test]
    fn test_vec_historian_errors_after_completion() {
        let mut historian = OpenHandHistoryVecHistorian::new();
        let storage = historian.get_storage();

        record_simple_hand(&mut historian, 1);
        assert_eq!(storage.borrow().len(), 1);

        // Attempting to record another hand should fail since the builder was consumed
        let game_state = create_test_game_state();
        let err = historian
            .record_action(
                2,
                &game_state,
                Action::GameStart(GameStartPayload {
                    small_blind: 1.0,
                    big_blind: 2.0,
                    ante: 0.0,
                }),
            )
            .unwrap_err();

        assert!(matches!(err, HistorianError::UnableToRecordAction));
    }

    #[test]
    fn test_historian_errors_after_completion() {
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_path_buf();
        temp_file.close().unwrap();

        let mut historian = OpenHandHistoryHistorian::new(temp_path.clone());

        let mut game_state = create_test_game_state();
        historian
            .record_action(
                11111,
                &game_state,
                Action::GameStart(GameStartPayload {
                    small_blind: 1.0,
                    big_blind: 2.0,
                    ante: 0.0,
                }),
            )
            .unwrap();
        game_state.complete();
        historian
            .record_action(
                11111,
                &game_state,
                Action::Award(AwardPayload {
                    idx: 0,
                    award_amount: 3.0,
                    total_pot: 3.0,
                    rank: None,
                    hand: None,
                }),
            )
            .unwrap();

        let next_state = create_test_game_state();
        let err = historian
            .record_action(
                22222,
                &next_state,
                Action::GameStart(GameStartPayload {
                    small_blind: 1.0,
                    big_blind: 2.0,
                    ante: 0.0,
                }),
            )
            .unwrap_err();

        assert!(matches!(err, HistorianError::UnableToRecordAction));

        assert!(temp_path.exists());
        let content = std::fs::read_to_string(&temp_path).unwrap();
        let lines: Vec<&str> = content
            .trim()
            .split('\n')
            .filter(|line| !line.is_empty())
            .collect();
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn test_deserialize_generated_file_structure() {
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_path_buf();
        temp_file.close().unwrap();

        let mut historian = OpenHandHistoryHistorian::new(temp_path.clone());
        let mut game_state = create_test_game_state();

        // Create a basic complete game
        historian
            .record_action(
                12345,
                &game_state,
                Action::GameStart(GameStartPayload {
                    small_blind: 1.0,
                    big_blind: 2.0,
                    ante: 0.0,
                }),
            )
            .unwrap();

        game_state.complete();
        historian
            .record_action(
                12345,
                &game_state,
                Action::Award(AwardPayload {
                    idx: 0,
                    award_amount: 3.0,
                    total_pot: 3.0,
                    rank: None,
                    hand: None,
                }),
            )
            .unwrap();

        // Read and deserialize the file
        let content = std::fs::read_to_string(&temp_path).unwrap();
        let wrapper: OpenHandHistoryWrapper = serde_json::from_str(content.trim()).unwrap();

        // Verify structure
        assert_eq!(wrapper.ohh.spec_version, "1.4.7");
        assert_eq!(wrapper.ohh.game_number, "12345");
        assert_eq!(wrapper.ohh.small_blind_amount, 1.0);
        assert_eq!(wrapper.ohh.big_blind_amount, 2.0);
        assert_eq!(wrapper.ohh.ante_amount, 0.0);
        assert_eq!(wrapper.ohh.table_size, 2);
    }
}
