use std::{collections::HashMap, fs::File, path::PathBuf};

use uuid::Uuid;

use crate::arena::action::Action;

use super::Historian;

/// A historian implementation that records game actions in a directory.
#[derive(Debug, Clone)]
pub struct DirectoryHistorian {
    base_path: PathBuf,
    sequence: HashMap<Uuid, Vec<Action>>,
}

impl DirectoryHistorian {
    /// Creates a new `DirectoryHistorian` with the specified base path.
    ///
    /// # Arguments
    ///
    /// * `base_path` - The base path where the game action files will be
    ///   stored.
    pub fn new(base_path: PathBuf) -> Self {
        DirectoryHistorian {
            base_path,
            sequence: HashMap::new(),
        }
    }
}
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
    fn record_action(
        &mut self,
        id: &uuid::Uuid,
        _game_state: &crate::arena::GameState,
        action: crate::arena::action::Action,
    ) -> Result<(), super::HistorianError> {
        // First make sure the base_path exists at all
        if !self.base_path.exists() {
            std::fs::create_dir_all(&self.base_path)?;
        }

        let game_path = self.base_path.join(id.to_string()).with_extension("json");
        // Create and write the whole sequence to the file every time just in case
        // something fails.
        let file = File::create(game_path)?;
        // Add the new action to the sequence
        let sequence = self.sequence.entry(*id).or_default();
        sequence.push(action);

        // Write the sequence to the file
        Ok(serde_json::to_writer_pretty(&file, sequence)?)
    }
}
