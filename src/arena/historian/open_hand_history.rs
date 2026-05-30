//! Open Hand History Historian Implementation
//!
//! This module provides the `OpenHandHistoryHistorian` which implements the `Historian` trait
//! to record arena game simulations in the standardized Open Hand History (OHH) JSON format.

use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use tokio::io::AsyncWriteExt;
use tracing::{debug, instrument, trace};

use crate::arena::game_state::Round;
use crate::arena::{GameState, action::Action};

use crate::arena::historian::{Historian, HistorianError, HistorianLock};

use crate::open_hand_history::{ConverterConfig, HandHistory, HandHistoryBuilder, write_hand};

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
    config: ConverterConfig,
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
        let builder = HandHistoryBuilder::new(config.clone());
        Self {
            output_path,
            config,
            builder: Some(builder),
        }
    }
}

#[async_trait]
impl Historian for OpenHandHistoryHistorian {
    #[instrument(level = "trace", skip(self, game_state), fields(output_path = ?self.output_path))]
    async fn record_action(
        &mut self,
        id: u128,
        game_state: &GameState,
        action: &Action,
    ) -> Result<(), HistorianError> {
        let builder = self
            .builder
            .as_mut()
            .ok_or(HistorianError::UnableToRecordAction)?;

        // Record the action (builder handles game_id internally)
        builder.record_action(id, action, game_state)?;

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
                tokio::fs::create_dir_all(parent).await?;
            }

            debug!(id, ?self.output_path, "Writing completed hand to OHH file");

            // Serialize the hand into the OHH on-disk record up front, then
            // append it with a single async `write_all`. `write_hand` produces
            // the exact byte sequence `append_hand` would, and opening with
            // `O_APPEND` keeps the write atomic at EOF (see `write_hand`); using
            // async fs keeps the runtime worker thread off the blocking syscall.
            let mut record = Vec::new();
            write_hand(&mut record, hand_history)?;
            let mut file = tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.output_path)
                .await?;
            file.write_all(&record).await?;
            file.flush().await?;

            // Reset the builder so the historian can be reused across hands
            // (matching `OpenHandHistoryVecHistorian`'s behaviour below).
            // Without this, a reused historian would silently no-op on
            // every subsequent action with `UnableToRecordAction`.
            self.builder = Some(HandHistoryBuilder::new(self.config.clone()));
        }

        Ok(())
    }
}

/// Shared storage backing an [`OpenHandHistoryVecHistorian`]. `Arc<Mutex<..>>`
/// (rather than `Rc<RefCell<..>>`) so the historian is `Send`.
pub type SharedHandHistoryStorage = Arc<Mutex<Vec<HandHistory>>>;

/// A historian that records hand histories into in-memory storage for later inspection.
///
/// This mirrors the behavior of [`VecHistorian`](crate::arena::historian::VecHistorian) but
/// stores fully converted Open Hand History records instead of raw arena actions.
#[derive(Debug)]
pub struct OpenHandHistoryVecHistorian {
    config: ConverterConfig,
    builder: Option<HandHistoryBuilder>,
    storage: SharedHandHistoryStorage,
}

impl OpenHandHistoryVecHistorian {
    /// Create a new vector-backed historian using the default converter configuration.
    pub fn new() -> Self {
        Self::new_with_config(Arc::new(Mutex::new(Vec::new())), ConverterConfig::default())
    }

    /// Create a historian with a custom converter configuration.
    pub fn new_with_config(storage: SharedHandHistoryStorage, config: ConverterConfig) -> Self {
        let builder = HandHistoryBuilder::new(config.clone());
        Self {
            config,
            builder: Some(builder),
            storage,
        }
    }

    /// Create a historian backed by the provided storage and default converter configuration.
    pub fn new_with_storage(storage: SharedHandHistoryStorage) -> Self {
        Self::new_with_config(storage, ConverterConfig::default())
    }

    /// Access the underlying storage so tests/fuzz targets can inspect recorded hands.
    pub fn get_storage(&self) -> SharedHandHistoryStorage {
        self.storage.clone()
    }
}

impl Default for OpenHandHistoryVecHistorian {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Historian for OpenHandHistoryVecHistorian {
    #[instrument(level = "trace", skip(self, game_state))]
    async fn record_action(
        &mut self,
        id: u128,
        game_state: &GameState,
        action: &Action,
    ) -> Result<(), HistorianError> {
        let builder = self
            .builder
            .as_mut()
            .ok_or(HistorianError::UnableToRecordAction)?;

        // Record the action (builder handles game_id internally)
        builder.record_action(id, action, game_state)?;

        // Check if game is complete
        if matches!(game_state.round, Round::Complete) {
            let completed_builder = self
                .builder
                .take()
                .ok_or(HistorianError::UnableToRecordAction)?;

            // Build hand history (consumes builder)
            let hand_history = completed_builder.build()?;

            // Store in memory
            let mut storage = self
                .storage
                .lock()
                .map_err(|_| HistorianError::LockPoisoned {
                    lock: HistorianLock::VecRecords,
                })?;
            storage.push(hand_history);
            trace!(
                id,
                storage_count = storage.len(),
                "Stored completed hand in memory"
            );
            drop(storage);

            // Reset the builder so the historian can be reused across hands.
            self.builder = Some(HandHistoryBuilder::new(self.config.clone()));
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
    use tempfile::NamedTempFile;

    fn create_test_game_state() -> GameState {
        GameStateBuilder::new()
            .stacks(vec![100.0, 100.0])
            .blinds(2.0, 1.0)
            .build()
            .unwrap()
    }

    async fn record_simple_hand<H: Historian>(historian: &mut H, game_id: u128) {
        let mut game_state = create_test_game_state();
        historian
            .record_action(
                game_id,
                &game_state,
                &Action::GameStart(GameStartPayload {
                    small_blind: 1.0,
                    big_blind: 2.0,
                    ante: 0.0,
                }),
            )
            .await
            .unwrap();

        game_state.complete();
        historian
            .record_action(
                game_id,
                &game_state,
                &Action::Award(AwardPayload {
                    idx: 0,
                    award_amount: 10.0,
                    total_pot: 10.0,
                    rank: None,
                    hand: None,
                }),
            )
            .await
            .unwrap();
    }

    /// Regression test for M6: the file-backed historian's builder must
    /// be re-initialised after `Round::Complete` so the historian can
    /// be reused across hands. Previously `self.builder.take()` left
    /// it `None`, and any subsequent action returned
    /// `UnableToRecordAction`.
    #[tokio::test(flavor = "current_thread")]
    async fn test_file_historian_reusable_across_hands() {
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_path_buf();
        temp_file.close().unwrap();

        let mut historian = OpenHandHistoryHistorian::new(temp_path.clone());
        record_simple_hand(&mut historian, 1).await;
        // The failing precondition: this second record_action used to
        // return `UnableToRecordAction` because the builder was consumed.
        record_simple_hand(&mut historian, 2).await;

        // And both hands should have been appended to the file.
        let contents = std::fs::read_to_string(&temp_path).unwrap();
        let line_count = contents.lines().filter(|l| !l.trim().is_empty()).count();
        assert_eq!(line_count, 2, "expected 2 hands in the output file");

        let _ = std::fs::remove_file(&temp_path);
    }

    /// Regression test for M6: the in-memory historian also re-initialises
    /// the builder after completion (it already survived the same pattern,
    /// but we pin it so the contract matches the file-backed variant).
    #[tokio::test(flavor = "current_thread")]
    async fn test_vec_historian_reusable_across_hands() {
        let storage: SharedHandHistoryStorage = Arc::new(Mutex::new(Vec::new()));
        let mut historian = OpenHandHistoryVecHistorian::new_with_config(
            storage.clone(),
            ConverterConfig::default(),
        );

        record_simple_hand(&mut historian, 1).await;
        record_simple_hand(&mut historian, 2).await;
        assert_eq!(storage.lock().unwrap().len(), 2);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_historian_creation_with_default_config() {
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_path_buf();
        temp_file.close().unwrap();

        let mut historian = OpenHandHistoryHistorian::new(temp_path.clone());
        let mut game_state = create_test_game_state();

        historian
            .record_action(
                12345,
                &game_state,
                &Action::GameStart(GameStartPayload {
                    small_blind: 1.0,
                    big_blind: 2.0,
                    ante: 0.0,
                }),
            )
            .await
            .unwrap();

        game_state.complete();
        historian
            .record_action(
                12345,
                &game_state,
                &Action::Award(AwardPayload {
                    idx: 0,
                    award_amount: 10.0,
                    total_pot: 10.0,
                    rank: None,
                    hand: None,
                }),
            )
            .await
            .unwrap();

        let content = std::fs::read_to_string(&temp_path).unwrap();
        let wrapper: OpenHandHistoryWrapper = serde_json::from_str(content.trim()).unwrap();

        assert_eq!(wrapper.ohh.site_name, "rs_poker");
        assert_eq!(wrapper.ohh.network_name, "rs_poker_arena");
        assert_eq!(wrapper.ohh.currency, "USD");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_historian_creation_with_custom_config() {
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
                &Action::GameStart(GameStartPayload {
                    small_blind: 1.0,
                    big_blind: 2.0,
                    ante: 0.0,
                }),
            )
            .await
            .unwrap();

        game_state.complete();
        historian
            .record_action(
                67890,
                &game_state,
                &Action::Award(AwardPayload {
                    idx: 1,
                    award_amount: 5.0,
                    total_pot: 5.0,
                    rank: None,
                    hand: None,
                }),
            )
            .await
            .unwrap();

        let content = std::fs::read_to_string(&temp_path).unwrap();
        let wrapper: OpenHandHistoryWrapper = serde_json::from_str(content.trim()).unwrap();

        assert_eq!(wrapper.ohh.site_name, "custom_site");
        assert_eq!(wrapper.ohh.network_name, "custom_network");
        assert_eq!(wrapper.ohh.currency, "EUR");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_file_not_created_until_game_completes() {
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
                &Action::GameStart(GameStartPayload {
                    small_blind: 1.0,
                    big_blind: 2.0,
                    ante: 0.0,
                }),
            )
            .await
            .unwrap();

        // File should not exist yet
        assert!(!temp_path.exists());

        // Complete the game and record an award
        game_state.complete();
        historian
            .record_action(
                12345,
                &game_state,
                &Action::Award(AwardPayload {
                    idx: 0,
                    award_amount: 10.0,
                    total_pot: 10.0,
                    rank: None,
                    hand: None,
                }),
            )
            .await
            .unwrap();

        // Now file should exist
        assert!(temp_path.exists());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_historian_with_simple_game_completion() {
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
                &Action::GameStart(GameStartPayload {
                    small_blind: 1.0,
                    big_blind: 2.0,
                    ante: 0.0,
                }),
            )
            .await
            .unwrap();

        // Complete the game
        game_state.complete();
        historian
            .record_action(
                12345,
                &game_state,
                &Action::Award(AwardPayload {
                    idx: 0,
                    award_amount: 200.0,
                    total_pot: 200.0,
                    rank: None,
                    hand: None,
                }),
            )
            .await
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

    #[tokio::test]
    async fn test_vec_historian_collects_hand_history() {
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

        let mut sim = HoldemSimulationBuilder::default()
            .game_state(game_state)
            .agents(agents)
            .historians(vec![historian])
            .build()
            .unwrap();

        sim.run().await;

        let hands = storage.lock().unwrap();
        assert_eq!(hands.len(), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_vec_historian_records_single_hand() {
        let mut historian = OpenHandHistoryVecHistorian::new();
        let storage = historian.get_storage();

        record_simple_hand(&mut historian, 1).await;

        assert_eq!(storage.lock().unwrap().len(), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_deserialize_generated_file_structure() {
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
                &Action::GameStart(GameStartPayload {
                    small_blind: 1.0,
                    big_blind: 2.0,
                    ante: 0.0,
                }),
            )
            .await
            .unwrap();

        game_state.complete();
        historian
            .record_action(
                12345,
                &game_state,
                &Action::Award(AwardPayload {
                    idx: 0,
                    award_amount: 3.0,
                    total_pot: 3.0,
                    rank: None,
                    hand: None,
                }),
            )
            .await
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
