use std::{collections::HashMap, path::PathBuf};

use async_trait::async_trait;
use tracing::{debug, instrument};

use crate::arena::action::Action;

use super::Historian;

/// A historian implementation that records game actions in a directory.
#[derive(Debug, Clone)]
pub struct DirectoryHistorian {
    base_path: PathBuf,
    sequence: HashMap<u128, Vec<Action>>,
}

impl DirectoryHistorian {
    /// Creates a new `DirectoryHistorian` with the specified base path.
    ///
    /// # Arguments
    ///
    /// * `base_path` - The base path where the game action files will be
    ///   stored.
    pub fn new(base_path: PathBuf) -> Self {
        debug!(?base_path, "Creating DirectoryHistorian");
        DirectoryHistorian {
            base_path,
            sequence: HashMap::new(),
        }
    }
}
#[async_trait]
impl Historian for DirectoryHistorian {
    /// Records all the game actions into a file in the specified directory.
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the game.
    /// * `_game_state` - The current game state.
    /// * `action` - The action to record.
    ///
    /// # Errors
    ///
    /// Returns an error if there was a problem recording the action.
    #[instrument(level = "trace", skip(self, _game_state), fields(base_path = ?self.base_path))]
    async fn record_action(
        &mut self,
        id: u128,
        _game_state: &crate::arena::GameState,
        action: &crate::arena::action::Action,
    ) -> Result<(), super::HistorianError> {
        // First make sure the base_path exists at all
        if !tokio::fs::try_exists(&self.base_path).await? {
            debug!(?self.base_path, "Creating directory for game history");
            tokio::fs::create_dir_all(&self.base_path).await?;
        }

        let game_path = self.base_path.join(id.to_string()).with_extension("json");
        // Add the new action to the sequence
        let sequence = self.sequence.entry(id).or_default();
        sequence.push(action.clone());

        debug!(
            ?game_path,
            action_count = sequence.len(),
            "Writing game history"
        );

        // Serialize the whole sequence up front, then write the file every time
        // just in case something fails. The serialization releases the borrow of
        // `self.sequence` before the `.await`, and async fs keeps the runtime
        // worker thread off the blocking write syscall.
        let bytes = serde_json::to_vec_pretty(sequence)?;
        tokio::fs::write(&game_path, &bytes).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::GameStateBuilder;
    use crate::arena::action::{ForcedBetPayload, ForcedBetType, GameStartPayload};
    use tempfile::TempDir;

    /// Verifies that record_action creates the directory tree if it doesn't exist.
    #[tokio::test(flavor = "current_thread")]
    async fn test_creates_directory_when_missing() {
        let temp_dir = TempDir::new().unwrap();
        let non_existent_path = temp_dir.path().join("subdir").join("game_history");

        // Verify directory doesn't exist yet
        assert!(
            !non_existent_path.exists(),
            "Directory should not exist initially"
        );

        let mut historian = DirectoryHistorian::new(non_existent_path.clone());

        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let action = Action::GameStart(GameStartPayload {
            ante: 0.0,
            small_blind: 5.0,
            big_blind: 10.0,
        });

        historian
            .record_action(12345, &game_state, &action)
            .await
            .unwrap();

        // Directory should now exist
        assert!(
            non_existent_path.exists(),
            "Directory should be created by record_action"
        );

        // File should also exist
        let expected_file = non_existent_path.join("12345.json");
        assert!(
            expected_file.exists(),
            "Game history file should be created"
        );
    }

    /// Verifies that record_action writes the action data to a file.
    #[tokio::test(flavor = "current_thread")]
    async fn test_records_action_to_file() {
        let temp_dir = TempDir::new().unwrap();
        let history_path = temp_dir.path().join("history");

        let mut historian = DirectoryHistorian::new(history_path.clone());

        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let action = Action::ForcedBet(ForcedBetPayload {
            bet: 5.0,
            player_stack: 100.0,
            idx: 0,
            forced_bet_type: ForcedBetType::SmallBlind,
        });

        historian
            .record_action(42, &game_state, &action)
            .await
            .unwrap();

        // Read the file and verify content
        let file_path = history_path.join("42.json");
        let content = std::fs::read_to_string(&file_path).unwrap();

        // Verify the file contains the action
        assert!(
            content.contains("ForcedBet"),
            "File should contain the action type"
        );
        assert!(
            content.contains("SmallBlind"),
            "File should contain forced bet type"
        );
    }

    /// Test that multiple actions are recorded in sequence.
    #[tokio::test(flavor = "current_thread")]
    async fn test_records_multiple_actions() {
        let temp_dir = TempDir::new().unwrap();
        let history_path = temp_dir.path().join("history");

        let mut historian = DirectoryHistorian::new(history_path.clone());
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();

        let action1 = Action::ForcedBet(ForcedBetPayload {
            bet: 5.0,
            player_stack: 100.0,
            idx: 0,
            forced_bet_type: ForcedBetType::SmallBlind,
        });
        let action2 = Action::ForcedBet(ForcedBetPayload {
            bet: 10.0,
            player_stack: 100.0,
            idx: 1,
            forced_bet_type: ForcedBetType::BigBlind,
        });

        historian
            .record_action(100, &game_state, &action1)
            .await
            .unwrap();
        historian
            .record_action(100, &game_state, &action2)
            .await
            .unwrap();

        // Read the file and verify both actions are recorded
        let file_path = history_path.join("100.json");
        let content = std::fs::read_to_string(&file_path).unwrap();

        // File should be a JSON array with 2 elements
        let actions: Vec<Action> = serde_json::from_str(&content).unwrap();
        assert_eq!(actions.len(), 2, "Should have 2 recorded actions");
    }
}
