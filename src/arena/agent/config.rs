//! # Agent Configuration System
//!
//! This module provides JSON-based configuration for poker agents, enabling
//! declarative agent creation via configuration files or inline JSON strings.
//!
//! ## Basic Usage
//!
//! ```rust
//! use rs_poker::arena::agent::ConfigAgentGenerator;
//!
//! // From inline JSON
//! let generator = ConfigAgentGenerator::from_json(r#"{"type": "all_in"}"#).unwrap();
//!
//! // From file path
//! # // let generator = ConfigAgentGenerator::from_file("agents/random.json").unwrap();
//!
//! // Smart parsing (tries file first, then inline JSON)
//! # // let generator = ConfigAgentGenerator::from_str_or_file("agents/calling.json").unwrap();
//! ```
//!
//! ## Supported Agent Types
//!
//! ### Simple Agents (no parameters)
//! - `all_in` - Always goes all-in
//! - `calling` - Always calls
//! - `folding` - Always folds
//!
//! ### Configurable Agents
//! - `random` - Random decision making with probability vectors
//! - `random_pot_control` - Monte Carlo-based pot control
//!
//! ## Examples
//!
//! ### Random Agent with Default Parameters
//! ```json
//! {"type": "random"}
//! ```
//!
//! ### Random Agent with Custom Parameters
//! ```json
//! {
//!   "type": "random",
//!   "percent_fold": [0.2, 0.3, 0.5],
//!   "percent_call": [0.5, 0.6, 0.45]
//! }
//! ```
//!
//! ### Random Pot Control Agent
//! ```json
//! {
//!   "type": "random_pot_control",
//!   "percent_call": [0.5, 0.3]
//! }
//! ```
//!
//! ### CFR Agent with Fixed Iteration
//! ```json
//! {
//!   "type": "cfr_basic",
//!   "num_hands": 10
//! }
//! ```
//!
//! ### CFR Agent with Per-Round Iteration
//! ```json
//! {
//!   "type": "cfr_per_round",
//!   "pre_flop_hands": 10,
//!   "flop_hands": 10,
//!   "turn_hands": 10,
//!   "river_hands": 1
//! }
//! ```
//!
//! ## Extending with New Agent Types
//!
//! To add a new agent type to the configuration system:
//!
//! 1. **Add variant to `AgentConfig` enum**:
//! ```rust,ignore
//! #[derive(Debug, Clone, Serialize, Deserialize)]
//! #[serde(tag = "type", rename_all = "snake_case")]
//! pub enum AgentConfig {
//!     // ... existing variants ...
//!     MyNewAgent {
//!         param1: f64,
//!         #[serde(default = "default_param2")]
//!         param2: Vec<f64>,
//!     },
//! }
//!
//! fn default_param2() -> Vec<f64> {
//!     vec![1.0, 2.0]
//! }
//! ```
//!
//! 2. **Add validation logic** in `AgentConfig::validate()`:
//! ```rust,ignore
//! impl AgentConfig {
//!     pub fn validate(&self) -> Result<(), AgentConfigError> {
//!         match self {
//!             AgentConfig::MyNewAgent { param1, param2 } => {
//!                 // Validate parameters
//!                 if *param1 < 0.0 {
//!                     return Err(AgentConfigError::ValidationError(
//!                         "param1 must be non-negative".to_string()
//!                     ));
//!                 }
//!             }
//!             // ... other variants ...
//!         }
//!         Ok(())
//!     }
//! }
//! ```
//!
//! 3. **Add handling in `ConfigAgentGenerator::generate()`**:
//! ```rust,ignore
//! impl AgentGenerator for ConfigAgentGenerator {
//!     fn generate(&self, player_idx: usize, game_state: &GameState) -> Box<dyn Agent> {
//!         match &self.config {
//!             AgentConfig::MyNewAgent { param1, param2 } => {
//!                 Box::new(MyNewAgent::new(*param1, param2.clone()))
//!             }
//!             // ... other variants ...
//!         }
//!     }
//! }
//! ```
//!
//! 4. **Add tests** for serialization, deserialization, and validation
//! 5. **Create example config file** in `examples/configs/`
//! 6. **Update documentation** with usage examples
//!
//! ## Future CFR Extensions
//!
//! The CFR configuration system is designed for extensibility. To add new
//! `ActionGenerator` or `GameStateIteratorGen` implementations:
//!
//! 1. **Define new config variants**:
//! ```rust,ignore
//! #[derive(Debug, Clone, Serialize, Deserialize)]
//! #[serde(tag = "type", rename_all = "snake_case")]
//! pub enum AgentConfig {
//!     // ... existing variants ...
//!     CfrAdvanced {
//!         action_generator: ActionGeneratorConfig,
//!         gamestate_iterator: GameStateIteratorConfig,
//!     },
//! }
//!
//! #[derive(Debug, Clone, Serialize, Deserialize)]
//! #[serde(tag = "type")]
//! pub enum ActionGeneratorConfig {
//!     Basic,
//!     Advanced { params: Vec<f64> },
//! }
//! ```
//!
//! 2. **Update `ConfigAgentGenerator::generate()`** to handle new CFR types
//! 3. **Add appropriate tests and examples**

use crate::arena::agent::{
    AgentGenerator, AllInAgent, CallingAgent, FoldingAgent, RandomAgent, RandomPotControlAgent,
};
use crate::arena::cfr::{
    BasicCFRActionGenerator, CFRAgent, FixedGameStateIteratorGen,
    PerRoundFixedGameStateIteratorGen, SimpleActionGenerator,
};
use crate::arena::{Agent, GameState};
use serde::{Deserialize, Serialize};
use std::{io::ErrorKind, path::Path};
use thiserror::Error;

/// Configuration for different agent types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum AgentConfig {
    /// Agent that always goes all-in
    AllIn {
        /// Optional explicit name for the agent
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// Agent that always calls
    Calling {
        /// Optional explicit name for the agent
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// Agent that always folds
    Folding {
        /// Optional explicit name for the agent
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// Agent that makes random decisions based on probability vectors
    #[serde(alias = "random")]
    Random {
        /// Optional explicit name for the agent
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Probability of folding indexed by raise count
        #[serde(default = "default_percent_fold")]
        percent_fold: Vec<f64>,
        /// Probability of calling indexed by raise count
        #[serde(default = "default_percent_call")]
        percent_call: Vec<f64>,
    },
    /// Agent that uses Monte Carlo simulation for pot control
    RandomPotControl {
        /// Optional explicit name for the agent
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Probability of calling indexed by raise count
        percent_call: Vec<f64>,
    },
    /// CFR agent with fixed number of game state iterations
    CfrBasic {
        /// Optional explicit name for the agent
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Number of game state hands to iterate per action exploration
        #[serde(default = "default_cfr_num_hands")]
        num_hands: usize,
    },
    /// CFR agent with per-round configurable game state iterations
    CfrPerRound {
        /// Optional explicit name for the agent
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Number of hands to iterate during pre-flop
        #[serde(default = "default_pre_flop_hands")]
        pre_flop_hands: usize,
        /// Number of hands to iterate during flop
        #[serde(default = "default_flop_hands")]
        flop_hands: usize,
        /// Number of hands to iterate during turn
        #[serde(default = "default_turn_hands")]
        turn_hands: usize,
        /// Number of hands to iterate during river
        #[serde(default = "default_river_hands")]
        river_hands: usize,
    },
    /// CFR agent with SimpleActionGenerator (more bet sizing options)
    ///
    /// Uses 6 actions: fold, check/call, min raise, 33% pot, 66% pot, all-in
    CfrSimple {
        /// Optional explicit name for the agent
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Number of game state hands to iterate per action exploration
        #[serde(default = "default_cfr_num_hands")]
        num_hands: usize,
    },
    /// CFR agent with SimpleActionGenerator and per-round iterations
    ///
    /// Uses 6 actions: fold, check/call, min raise, 33% pot, 66% pot, all-in
    CfrSimplePerRound {
        /// Optional explicit name for the agent
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Number of hands to iterate during pre-flop
        #[serde(default = "default_pre_flop_hands")]
        pre_flop_hands: usize,
        /// Number of hands to iterate during flop
        #[serde(default = "default_flop_hands")]
        flop_hands: usize,
        /// Number of hands to iterate during turn
        #[serde(default = "default_turn_hands")]
        turn_hands: usize,
        /// Number of hands to iterate during river
        #[serde(default = "default_river_hands")]
        river_hands: usize,
    },
}

fn default_percent_fold() -> Vec<f64> {
    vec![0.25, 0.30, 0.50]
}

fn default_percent_call() -> Vec<f64> {
    vec![0.5, 0.6, 0.45]
}

fn default_cfr_num_hands() -> usize {
    10
}

fn default_pre_flop_hands() -> usize {
    10
}

fn default_flop_hands() -> usize {
    10
}

fn default_turn_hands() -> usize {
    10
}

fn default_river_hands() -> usize {
    1
}

/// Errors that can occur during agent configuration
#[derive(Debug, Error)]
pub enum AgentConfigError {
    /// Invalid probability value (must be between 0.0 and 1.0)
    #[error("Invalid probability value: {0} (must be between 0.0 and 1.0)")]
    InvalidProbability(f64),

    /// JSON parsing error
    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// File I/O error
    #[error("File I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Generic validation error
    #[error("Validation error: {0}")]
    ValidationError(String),
}

impl AgentConfig {
    /// Validate that the configuration is correct
    pub fn validate(&self) -> Result<(), AgentConfigError> {
        match self {
            AgentConfig::Random {
                percent_fold,
                percent_call,
                ..
            } => {
                validate_probabilities(percent_fold)?;
                validate_probabilities(percent_call)?;
            }
            AgentConfig::RandomPotControl { percent_call, .. } => {
                validate_probabilities(percent_call)?;
            }
            _ => {}
        }
        Ok(())
    }
}

fn validate_probabilities(probs: &[f64]) -> Result<(), AgentConfigError> {
    for &p in probs {
        if !(0.0..=1.0).contains(&p) {
            return Err(AgentConfigError::InvalidProbability(p));
        }
    }
    Ok(())
}

fn default_agent_name(agent_kind: &str, player_idx: usize) -> String {
    format!("{agent_kind}-{player_idx}")
}

fn resolve_agent_name(name: &Option<String>, agent_kind: &str, player_idx: usize) -> String {
    name.clone()
        .unwrap_or_else(|| default_agent_name(agent_kind, player_idx))
}

/// Agent generator that creates agents from configuration.
///
/// CFR agents created by this generator will automatically initialize
/// CFR state for all players in the game, enabling proper mixed-agent
/// simulations.
#[derive(Debug, Clone)]
pub struct ConfigAgentGenerator {
    config: AgentConfig,
}

impl ConfigAgentGenerator {
    /// Create a new generator from a validated config
    pub fn new(config: AgentConfig) -> Result<Self, AgentConfigError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Create a new generator with an optional StateStore for CFR agents.
    ///
    /// Note: The state_store parameter is now ignored. CFR agents create
    /// their own state store that initializes states for all players.
    /// This method is kept for backwards compatibility.
    #[deprecated(
        since = "0.6.0",
        note = "CFR agents now create their own StateStore. Use new() instead."
    )]
    pub fn with_state_store(
        config: AgentConfig,
        _state_store: Option<std::sync::Arc<crate::arena::cfr::StateStore>>,
    ) -> Result<Self, AgentConfigError> {
        Self::new(config)
    }

    /// Get a reference to the configuration
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Create from a JSON string
    pub fn from_json(json: &str) -> Result<Self, AgentConfigError> {
        let config: AgentConfig = serde_json::from_str(json)?;
        Self::new(config)
    }

    /// Create from a file path
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, AgentConfigError> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json)
    }

    /// Try to parse as file path first, then as inline JSON
    pub fn from_str_or_file(input: &str) -> Result<Self, AgentConfigError> {
        match Self::from_file(input) {
            Ok(generator) => Ok(generator),
            Err(AgentConfigError::IoError(err)) if err.kind() == ErrorKind::NotFound => {
                Self::from_json(input)
            }
            Err(err) => Err(err),
        }
    }
}

impl AgentGenerator for ConfigAgentGenerator {
    fn generate(&self, player_idx: usize, game_state: &GameState) -> Box<dyn Agent> {
        match &self.config {
            AgentConfig::AllIn { name } => Box::new(AllInAgent::new(resolve_agent_name(
                name,
                "AllInAgent",
                player_idx,
            ))),
            AgentConfig::Calling { name } => Box::new(CallingAgent::new(resolve_agent_name(
                name,
                "CallingAgent",
                player_idx,
            ))),
            AgentConfig::Folding { name } => Box::new(FoldingAgent::new(resolve_agent_name(
                name,
                "FoldingAgent",
                player_idx,
            ))),
            AgentConfig::Random {
                name,
                percent_fold,
                percent_call,
            } => Box::new(RandomAgent::new(
                resolve_agent_name(name, "RandomAgent", player_idx),
                percent_fold.clone(),
                percent_call.clone(),
            )),
            AgentConfig::RandomPotControl { name, percent_call } => {
                Box::new(RandomPotControlAgent::new(
                    resolve_agent_name(name, "RandomPotControlAgent", player_idx),
                    percent_call.clone(),
                ))
            }
            AgentConfig::CfrBasic { name, num_hands } => {
                // Each CFR agent creates its own StateStore that initializes
                // states for ALL players. This enables proper mixed-agent play.
                Box::new(
                    CFRAgent::<BasicCFRActionGenerator, FixedGameStateIteratorGen>::new(
                        resolve_agent_name(name, "CFRAgent", player_idx),
                        player_idx,
                        game_state.clone(),
                        FixedGameStateIteratorGen::new(*num_hands),
                    ),
                )
            }
            AgentConfig::CfrPerRound {
                name,
                pre_flop_hands,
                flop_hands,
                turn_hands,
                river_hands,
            } => {
                // Each CFR agent creates its own StateStore that initializes
                // states for ALL players. This enables proper mixed-agent play.
                Box::new(CFRAgent::<
                    BasicCFRActionGenerator,
                    PerRoundFixedGameStateIteratorGen,
                >::new(
                    resolve_agent_name(name, "CFRAgent", player_idx),
                    player_idx,
                    game_state.clone(),
                    PerRoundFixedGameStateIteratorGen::new(
                        *pre_flop_hands,
                        *flop_hands,
                        *turn_hands,
                        *river_hands,
                    ),
                ))
            }
            AgentConfig::CfrSimple { name, num_hands } => Box::new(CFRAgent::<
                SimpleActionGenerator,
                FixedGameStateIteratorGen,
            >::new(
                resolve_agent_name(name, "CFRSimpleAgent", player_idx),
                player_idx,
                game_state.clone(),
                FixedGameStateIteratorGen::new(*num_hands),
            )),
            AgentConfig::CfrSimplePerRound {
                name,
                pre_flop_hands,
                flop_hands,
                turn_hands,
                river_hands,
            } => Box::new(CFRAgent::<
                SimpleActionGenerator,
                PerRoundFixedGameStateIteratorGen,
            >::new(
                resolve_agent_name(name, "CFRSimpleAgent", player_idx),
                player_idx,
                game_state.clone(),
                PerRoundFixedGameStateIteratorGen::new(
                    *pre_flop_hands,
                    *flop_hands,
                    *turn_hands,
                    *river_hands,
                ),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_all_in() {
        let config = AgentConfig::AllIn { name: None };
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(json, r#"{"type":"all_in"}"#);
    }

    #[test]
    fn test_serialize_calling() {
        let config = AgentConfig::Calling { name: None };
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(json, r#"{"type":"calling"}"#);
    }

    #[test]
    fn test_serialize_folding() {
        let config = AgentConfig::Folding { name: None };
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(json, r#"{"type":"folding"}"#);
    }

    #[test]
    fn test_deserialize_all_in() {
        let json = r#"{"type":"all_in"}"#;
        let config: AgentConfig = serde_json::from_str(json).unwrap();
        match config {
            AgentConfig::AllIn { name } => assert!(name.is_none()),
            _ => panic!("Expected AllIn variant"),
        }
    }

    #[test]
    fn test_deserialize_calling() {
        let json = r#"{"type":"calling"}"#;
        let config: AgentConfig = serde_json::from_str(json).unwrap();
        match config {
            AgentConfig::Calling { name } => assert!(name.is_none()),
            _ => panic!("Expected Calling variant"),
        }
    }

    #[test]
    fn test_deserialize_folding() {
        let json = r#"{"type":"folding"}"#;
        let config: AgentConfig = serde_json::from_str(json).unwrap();
        match config {
            AgentConfig::Folding { name } => assert!(name.is_none()),
            _ => panic!("Expected Folding variant"),
        }
    }

    #[test]
    fn test_deserialize_random_with_defaults() {
        let json = r#"{"type":"random"}"#;
        let config: AgentConfig = serde_json::from_str(json).unwrap();
        match config {
            AgentConfig::Random {
                name,
                percent_fold,
                percent_call,
            } => {
                assert!(name.is_none());
                assert_eq!(percent_fold, vec![0.25, 0.30, 0.50]);
                assert_eq!(percent_call, vec![0.5, 0.6, 0.45]);
            }
            _ => panic!("Expected Random variant"),
        }
    }

    #[test]
    fn test_deserialize_random_with_params() {
        let json = r#"{"type":"random","percent_fold":[0.1,0.2],"percent_call":[0.6,0.7]}"#;
        let config: AgentConfig = serde_json::from_str(json).unwrap();
        match config {
            AgentConfig::Random {
                name,
                percent_fold,
                percent_call,
            } => {
                assert!(name.is_none());
                assert_eq!(percent_fold, vec![0.1, 0.2]);
                assert_eq!(percent_call, vec![0.6, 0.7]);
            }
            _ => panic!("Expected Random variant"),
        }
    }

    #[test]
    fn test_deserialize_random_pot_control() {
        let json = r#"{"type":"random_pot_control","percent_call":[0.5,0.3]}"#;
        let config: AgentConfig = serde_json::from_str(json).unwrap();
        match config {
            AgentConfig::RandomPotControl { name, percent_call } => {
                assert!(name.is_none());
                assert_eq!(percent_call, vec![0.5, 0.3]);
            }
            _ => panic!("Expected RandomPotControl variant"),
        }
    }

    #[test]
    fn test_validate_invalid_probability_too_high() {
        let config = AgentConfig::Random {
            name: None,
            percent_fold: vec![1.5],
            percent_call: vec![0.5],
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_probability_negative() {
        let config = AgentConfig::Random {
            name: None,
            percent_fold: vec![0.25],
            percent_call: vec![-0.1],
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_valid_config() {
        let config = AgentConfig::Random {
            name: None,
            percent_fold: vec![0.25, 0.30],
            percent_call: vec![0.5, 0.6],
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_edge_cases() {
        let config = AgentConfig::Random {
            name: None,
            percent_fold: vec![0.0, 1.0],
            percent_call: vec![0.0, 1.0],
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_round_trip_serialization() {
        let configs = vec![
            AgentConfig::AllIn { name: None },
            AgentConfig::Calling { name: None },
            AgentConfig::Folding { name: None },
            AgentConfig::Random {
                name: None,
                percent_fold: vec![0.2],
                percent_call: vec![0.5],
            },
            AgentConfig::RandomPotControl {
                name: None,
                percent_call: vec![0.4, 0.3],
            },
        ];

        for config in configs {
            let json = serde_json::to_string(&config).unwrap();
            let deserialized: AgentConfig = serde_json::from_str(&json).unwrap();
            // Use Debug comparison since AgentConfig derives Debug
            assert_eq!(format!("{:?}", config), format!("{:?}", deserialized));
        }
    }

    // Generator tests
    #[test]
    fn test_create_from_config() {
        let config = AgentConfig::AllIn { name: None };
        let generator = ConfigAgentGenerator::new(config).unwrap();
        let game_state = GameState::new_starting(vec![100.0; 2], 10.0, 5.0, 0.0, 0);
        let _agent = generator.generate(0, &game_state);
        // Agent should be created successfully (test passes if no panic)
    }

    #[test]
    fn test_from_json() {
        let json = r#"{"type":"calling"}"#;
        let generator = ConfigAgentGenerator::from_json(json).unwrap();
        assert!(matches!(generator.config, AgentConfig::Calling { .. }));
    }

    #[test]
    fn test_from_json_with_params() {
        let json = r#"{"type":"random","percent_fold":[0.2],"percent_call":[0.5]}"#;
        let generator = ConfigAgentGenerator::from_json(json).unwrap();
        match generator.config {
            AgentConfig::Random {
                name,
                percent_fold,
                percent_call,
            } => {
                assert!(name.is_none());
                assert_eq!(percent_fold, vec![0.2]);
                assert_eq!(percent_call, vec![0.5]);
            }
            _ => panic!("Expected Random variant"),
        }
    }

    #[test]
    fn test_validation_on_construction() {
        let json = r#"{"type":"random","percent_fold":[1.5],"percent_call":[0.5]}"#;
        assert!(ConfigAgentGenerator::from_json(json).is_err());
    }

    #[test]
    fn test_generate_multiple_agents() {
        let config = AgentConfig::Random {
            name: None,
            percent_fold: vec![0.25],
            percent_call: vec![0.5],
        };
        let generator = ConfigAgentGenerator::new(config).unwrap();
        let game_state = GameState::new_starting(vec![100.0; 3], 10.0, 5.0, 0.0, 0);

        // Generate multiple agents from same generator
        let _agent1 = generator.generate(0, &game_state);
        let _agent2 = generator.generate(1, &game_state);
        // Both should be created successfully (test passes if no panic)
    }

    #[test]
    fn test_from_str_or_file_inline_json() {
        let json = r#"{"type":"all_in"}"#;
        let generator = ConfigAgentGenerator::from_str_or_file(json).unwrap();
        assert!(matches!(generator.config, AgentConfig::AllIn { .. }));
    }

    // CFR tests
    #[test]
    fn test_cfr_basic_config() {
        let json = r#"{"type":"cfr_basic","num_hands":5}"#;
        let config: AgentConfig = serde_json::from_str(json).unwrap();
        match config {
            AgentConfig::CfrBasic { name, num_hands } => {
                assert!(name.is_none());
                assert_eq!(num_hands, 5);
            }
            _ => panic!("Expected CfrBasic variant"),
        }
    }

    #[test]
    fn test_cfr_basic_defaults() {
        let json = r#"{"type":"cfr_basic"}"#;
        let config: AgentConfig = serde_json::from_str(json).unwrap();
        match config {
            AgentConfig::CfrBasic { name, num_hands } => {
                assert!(name.is_none());
                assert_eq!(num_hands, 10); // default
            }
            _ => panic!("Expected CfrBasic variant"),
        }
    }

    #[test]
    fn test_cfr_per_round_config() {
        let json = r#"{
            "type":"cfr_per_round",
            "pre_flop_hands":15,
            "flop_hands":12,
            "turn_hands":8,
            "river_hands":2
        }"#;
        let config: AgentConfig = serde_json::from_str(json).unwrap();
        match config {
            AgentConfig::CfrPerRound {
                name,
                pre_flop_hands,
                flop_hands,
                turn_hands,
                river_hands,
            } => {
                assert!(name.is_none());
                assert_eq!(pre_flop_hands, 15);
                assert_eq!(flop_hands, 12);
                assert_eq!(turn_hands, 8);
                assert_eq!(river_hands, 2);
            }
            _ => panic!("Expected CfrPerRound variant"),
        }
    }

    #[test]
    fn test_cfr_agent_generator() {
        let config = AgentConfig::CfrBasic {
            name: None,
            num_hands: 5,
        };
        // CFR agents now create their own state store, no need to provide one
        let generator = ConfigAgentGenerator::new(config).unwrap();

        let game_state = GameState::new_starting(vec![100.0; 2], 10.0, 5.0, 0.0, 0);
        let _agent = generator.generate(0, &game_state);
        // Agent should be created successfully (test passes if no panic)
    }

    #[test]
    fn test_cfr_agent_generator_per_round() {
        let config = AgentConfig::CfrPerRound {
            name: Some("TestCFR".to_string()),
            pre_flop_hands: 3,
            flop_hands: 3,
            turn_hands: 3,
            river_hands: 1,
        };
        let generator = ConfigAgentGenerator::new(config).unwrap();

        let game_state = GameState::new_starting(vec![100.0; 3], 10.0, 5.0, 0.0, 0);
        let agent = generator.generate(1, &game_state);
        assert_eq!(agent.name(), "TestCFR");
    }

    /// Verifies default_pre_flop_hands returns the expected default value.
    #[test]
    fn test_default_pre_flop_hands() {
        let result = default_pre_flop_hands();
        assert!(
            result > 1,
            "default_pre_flop_hands should be > 1, got {}",
            result
        );
        assert_eq!(result, 10, "default_pre_flop_hands should be 10");
    }

    /// Verifies default_flop_hands returns the expected default value.
    #[test]
    fn test_default_flop_hands() {
        let result = default_flop_hands();
        assert!(
            result > 1,
            "default_flop_hands should be > 1, got {}",
            result
        );
        assert_eq!(result, 10, "default_flop_hands should be 10");
    }

    /// Verifies default_turn_hands returns the expected default value.
    #[test]
    fn test_default_turn_hands() {
        let result = default_turn_hands();
        assert!(
            result > 1,
            "default_turn_hands should be > 1, got {}",
            result
        );
        assert_eq!(result, 10, "default_turn_hands should be 10");
    }

    /// Verifies default_river_hands returns the expected default value.
    #[test]
    fn test_default_river_hands() {
        let result = default_river_hands();
        assert!(
            result >= 1,
            "default_river_hands should be >= 1, got {}",
            result
        );
        assert_eq!(result, 1, "default_river_hands should be 1");
    }

    #[test]
    fn test_default_agent_name_format() {
        // default_agent_name should return "{agent_kind}-{player_idx}", not "xyzzy" or ""
        let name = default_agent_name("TestAgent", 5);

        assert!(!name.is_empty(), "agent name should not be empty");
        assert_ne!(name, "xyzzy", "agent name should not be 'xyzzy'");
        assert_eq!(name, "TestAgent-5", "agent name should be 'TestAgent-5'");
    }

    #[test]
    fn test_validate_random_pot_control_match_arm() {
        // Test that the RandomPotControl match arm in validate() is executed
        let config = AgentConfig::RandomPotControl {
            name: Some("Test".to_string()),
            percent_call: vec![0.5, 0.6], // Valid probabilities
        };

        // Should validate successfully
        let result = config.validate();
        assert!(
            result.is_ok(),
            "Valid RandomPotControl should pass validation"
        );

        // Test with invalid probability
        let invalid_config = AgentConfig::RandomPotControl {
            name: Some("Test".to_string()),
            percent_call: vec![1.5], // Invalid - > 1.0
        };

        let result = invalid_config.validate();
        assert!(
            result.is_err(),
            "Invalid RandomPotControl should fail validation"
        );
    }

    #[test]
    fn test_from_str_or_file_match_guard() {
        // Test the match guard: err.kind() == ErrorKind::NotFound
        // When file not found, should try parsing as JSON

        // This should fail as file (not found) but succeed as JSON
        // The format uses serde tag="type" with rename_all="snake_case"
        let json_input = r#"{"type": "all_in"}"#;
        let result = ConfigAgentGenerator::from_str_or_file(json_input);

        // Should succeed by parsing as JSON (not as file)
        assert!(
            result.is_ok(),
            "Valid JSON should parse when file not found"
        );

        // Test with a path that looks like a file but doesn't exist
        let nonexistent = "/nonexistent/path/to/config.json";
        let result = ConfigAgentGenerator::from_str_or_file(nonexistent);

        // Should try as JSON and fail (not valid JSON)
        assert!(
            result.is_err(),
            "Non-existent file with invalid JSON should fail"
        );
    }

    #[test]
    fn test_resolve_agent_name_with_none() {
        // Test that resolve_agent_name uses default when name is None
        let name = resolve_agent_name(&None, "TestKind", 3);
        assert_eq!(name, "TestKind-3", "Should use default name format");
    }

    #[test]
    fn test_resolve_agent_name_with_some() {
        // Test that resolve_agent_name uses provided name when Some
        let name = resolve_agent_name(&Some("CustomName".to_string()), "TestKind", 3);
        assert_eq!(name, "CustomName", "Should use provided name");
    }
}
