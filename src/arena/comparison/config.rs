use std::path::PathBuf;

use super::error::{ComparisonError, Result};

/// Configuration for running agent comparisons
#[derive(Debug, Clone)]
pub struct ComparisonConfig {
    /// Number of unique game states to test
    pub num_games: usize,
    /// Number of players per table
    pub players_per_table: usize,
    /// Big blind amount
    pub big_blind: f32,
    /// Small blind amount
    pub small_blind: f32,
    /// Minimum starting stack in big blinds
    pub min_stack_bb: f32,
    /// Maximum starting stack in big blinds
    pub max_stack_bb: f32,
    /// Ante amount (0.0 for no ante)
    pub ante: f32,
    /// Optional directory to save game history and results
    pub output_dir: Option<PathBuf>,
    /// Optional random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            num_games: 1000,
            players_per_table: 3,
            big_blind: 10.0,
            small_blind: 5.0,
            min_stack_bb: 100.0,
            max_stack_bb: 100.0,
            ante: 0.0,
            output_dir: None,
            seed: None,
        }
    }
}

impl ComparisonConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate the comparison configuration
    pub fn validate(&self, num_agents: usize) -> Result<()> {
        // Verify players_per_table >= 2
        if self.players_per_table < 2 {
            return Err(ComparisonError::ValidationError(
                "players_per_table must be at least 2".to_string(),
            ));
        }

        // Verify players_per_table <= num_agents
        if self.players_per_table > num_agents {
            return Err(ComparisonError::ValidationError(format!(
                "players_per_table ({}) cannot exceed number of agents ({})",
                self.players_per_table, num_agents
            )));
        }

        // Verify num_games > 0
        if self.num_games == 0 {
            return Err(ComparisonError::ValidationError(
                "num_games must be greater than 0".to_string(),
            ));
        }

        // Verify blinds are positive
        if self.big_blind <= 0.0 {
            return Err(ComparisonError::ValidationError(
                "big_blind must be positive".to_string(),
            ));
        }

        if self.small_blind <= 0.0 {
            return Err(ComparisonError::ValidationError(
                "small_blind must be positive".to_string(),
            ));
        }

        // Verify small blind is less than big blind
        if self.small_blind >= self.big_blind {
            return Err(ComparisonError::ValidationError(
                "small_blind must be less than big_blind".to_string(),
            ));
        }

        // Verify stack sizes are positive
        if self.min_stack_bb <= 0.0 {
            return Err(ComparisonError::ValidationError(
                "min_stack_bb must be positive".to_string(),
            ));
        }

        if self.max_stack_bb <= 0.0 {
            return Err(ComparisonError::ValidationError(
                "max_stack_bb must be positive".to_string(),
            ));
        }

        // Verify min <= max
        if self.min_stack_bb > self.max_stack_bb {
            return Err(ComparisonError::ValidationError(
                "min_stack_bb cannot exceed max_stack_bb".to_string(),
            ));
        }

        // Verify ante is non-negative
        if self.ante < 0.0 {
            return Err(ComparisonError::ValidationError(
                "ante must be non-negative".to_string(),
            ));
        }

        Ok(())
    }

    /// Get the minimum stack size in chips
    pub fn min_stack(&self) -> f32 {
        self.min_stack_bb * self.big_blind
    }

    /// Get the maximum stack size in chips
    pub fn max_stack(&self) -> f32 {
        self.max_stack_bb * self.big_blind
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ComparisonConfig::default();
        assert_eq!(config.num_games, 1000);
        assert_eq!(config.players_per_table, 3);
        assert_eq!(config.big_blind, 10.0);
        assert_eq!(config.small_blind, 5.0);
        assert_eq!(config.min_stack_bb, 100.0);
        assert_eq!(config.max_stack_bb, 100.0);
        assert_eq!(config.ante, 0.0);
        assert!(config.output_dir.is_none());
        assert!(config.seed.is_none());
    }

    #[test]
    fn test_validate_valid_config() {
        let config = ComparisonConfig::default();
        assert!(config.validate(5).is_ok());
    }

    #[test]
    fn test_validate_too_few_players() {
        let config = ComparisonConfig {
            players_per_table: 1,
            ..Default::default()
        };
        assert!(config.validate(5).is_err());
    }

    #[test]
    fn test_validate_too_many_players() {
        let config = ComparisonConfig {
            players_per_table: 6,
            ..Default::default()
        };
        assert!(config.validate(5).is_err());
    }

    #[test]
    fn test_validate_zero_games() {
        let config = ComparisonConfig {
            num_games: 0,
            ..Default::default()
        };
        assert!(config.validate(5).is_err());
    }

    #[test]
    fn test_validate_invalid_blinds() {
        let config = ComparisonConfig {
            big_blind: 5.0,
            small_blind: 10.0,
            ..Default::default()
        };
        assert!(config.validate(5).is_err());
    }

    #[test]
    fn test_validate_invalid_stack_range() {
        let config = ComparisonConfig {
            min_stack_bb: 200.0,
            max_stack_bb: 100.0,
            ..Default::default()
        };
        assert!(config.validate(5).is_err());
    }

    #[test]
    fn test_stack_calculations() {
        let config = ComparisonConfig {
            big_blind: 10.0,
            min_stack_bb: 100.0,
            max_stack_bb: 200.0,
            ..Default::default()
        };
        assert_eq!(config.min_stack(), 1000.0);
        assert_eq!(config.max_stack(), 2000.0);
    }
}
