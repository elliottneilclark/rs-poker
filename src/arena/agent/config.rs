//! # Agent Configuration System
//!
//! This module provides JSON-based configuration for poker agents, enabling
//! declarative agent creation via configuration files or inline JSON strings.
//!
//! ## Basic Usage
//!
//! ```rust
//! use rs_poker::arena::agent::ConfigAgentBuilder;
//!
//! // From inline JSON
//! let agent = ConfigAgentBuilder::from_json(r#"{"type": "all_in"}"#)
//!     .unwrap()
//!     .player_idx(0)
//!     .build();
//!
//! // From file path
//! # // let agent = ConfigAgentBuilder::from_file("agents/random.json")
//! # //     .unwrap().player_idx(0).build();
//!
//! // Smart parsing (tries file first, then inline JSON)
//! # // let agent = ConfigAgentBuilder::from_str_or_file("agents/calling.json")
//! # //     .unwrap().player_idx(0).build();
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
//! ### CFR Agent with an Exploration Policy
//! ```json
//! {
//!   "type": "cfr_basic",
//!   "exploration": {
//!     "budget": [
//!       { "type": "deadline", "millis": 250 },
//!       { "type": "per_depth_iterations", "counts": [20, 5, 1] }
//!     ]
//!   }
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
//!                 // Validate parameters via a domain-specific error
//!                 // variant; add a new variant to AgentConfigError if
//!                 // none of the existing typed variants fits.
//!                 if *param1 < 0.0 || *param1 > 1.0 {
//!                     return Err(AgentConfigError::InvalidProbability(*param1));
//!                 }
//!             }
//!             // ... other variants ...
//!         }
//!         Ok(())
//!     }
//! }
//! ```
//!
//! 3. **Add handling in `ConfigAgentBuilder::build()`**:
//! ```rust,ignore
//! // In the build() method's match on self.config:
//! AgentConfig::MyNewAgent { param1, param2 } => {
//!     Box::new(MyNewAgent::new(*param1, param2.clone()))
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
//! `ActionGenerator` implementations:
//!
//! 1. **Define new config variants**:
//! ```rust,ignore
//! #[derive(Debug, Clone, Serialize, Deserialize)]
//! #[serde(tag = "type", rename_all = "snake_case")]
//! pub enum AgentConfig {
//!     // ... existing variants ...
//!     CfrAdvanced {
//!         action_generator: ActionGeneratorConfig,
//!         exploration: CfrExploration,
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
//! 2. **Update `ConfigAgentBuilder::build()`** to handle new CFR types
//! 3. **Add appropriate tests and examples**

use std::sync::Arc;

use crate::arena::agent::{
    AllInAgent, CallingAgent, FoldingAgent, RandomAgent, RandomPotControlAgent,
};
use crate::arena::cfr::{
    BasicCFRActionGenerator, BudgetConfig, CFRAgentBuilder, CFRState, ConfigurableActionConfig,
    ConfigurableActionConfigError, ConfigurableActionGenerator, PreflopChartActionConfig,
    PreflopChartActionGenerator, PreflopChartConfig, PreflopChartConfigError,
    SimpleActionGenerator, TraversalSet,
};
use crate::arena::hand_estimator::{
    HandDistributionEstimator, KnownHandsEstimator, UniformRandomEstimator,
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
    /// CFR agent with depth-based game state iterations
    ///
    /// Uses BasicCFRActionGenerator with 3 actions: fold, check/call, and all-in.
    CfrBasic {
        /// Optional explicit name for the agent
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Exploration policy: recursion depth, stopping budget, wave width,
        /// and the per-action real-time ceiling.
        #[serde(default)]
        exploration: CfrExploration,
    },
    /// CFR agent with SimpleActionGenerator (more bet sizing options)
    ///
    /// Uses 6 actions: fold, check/call, min raise, 33% pot, 66% pot, all-in
    CfrSimple {
        /// Optional explicit name for the agent
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Exploration policy: recursion depth, stopping budget, wave width,
        /// and the per-action real-time ceiling.
        #[serde(default)]
        exploration: CfrExploration,
    },
    /// CFR agent with configurable action generator
    ///
    /// Allows full customization of bet sizing options per round:
    /// - Raise multiples (e.g., 1x, 2x, 3x min raise)
    /// - Pot multiples (e.g., 33%, 50%, 75%, 100% pot)
    /// - Setup shove action
    /// - Enable/disable check/call and all-in
    CfrConfigurable {
        /// Optional explicit name for the agent
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Exploration policy: recursion depth, stopping budget, wave width,
        /// and the per-action real-time ceiling.
        #[serde(default)]
        exploration: CfrExploration,
        /// Action generator configuration
        action_config: Box<ConfigurableActionConfig>,
    },
    /// CFR agent with preflop charts (limits preflop exploration to chart actions)
    ///
    /// This agent uses pre-configured preflop charts to limit preflop exploration
    /// to only actions that have non-zero probability in the chart for the current
    /// hand/position, while still using CFR for post-flop streets.
    CfrPreflopChart {
        /// Optional explicit name for the agent
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Exploration policy for post-flop CFR: recursion depth, stopping
        /// budget, wave width, and the per-action real-time ceiling.
        #[serde(default)]
        exploration: CfrExploration,
        /// Preflop chart configuration (inline or preset name)
        #[serde(default)]
        preflop_config: PreflopChartConfigOption,
        /// Post-flop action configuration (bet sizing options)
        #[serde(default, skip_serializing_if = "Option::is_none")]
        postflop_config: Option<Box<ConfigurableActionConfig>>,
    },
}

/// Option for specifying preflop chart configuration.
///
/// Can be either a preset name (e.g., "6max_gto", "tight", "loose")
/// or an inline chart configuration.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum PreflopChartConfigOption {
    /// Use a preset chart by name
    Preset(String),
    /// Inline chart configuration
    Inline(PreflopChartConfig),
}

// Custom Deserialize so failures on the inline path surface the *specific*
// PreflopChartConfig parse error (e.g. "missing field `charts`", "unknown
// field `position`") instead of serde's generic "data did not match any
// variant of untagged enum" message, which is useless when debugging a
// malformed config.
impl<'de> Deserialize<'de> for PreflopChartConfigOption {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        if let serde_json::Value::String(s) = &value {
            return Ok(PreflopChartConfigOption::Preset(s.clone()));
        }
        serde_json::from_value::<PreflopChartConfig>(value)
            .map(PreflopChartConfigOption::Inline)
            .map_err(serde::de::Error::custom)
    }
}

impl Default for PreflopChartConfigOption {
    fn default() -> Self {
        PreflopChartConfigOption::Preset("6max_gto".to_string())
    }
}

impl PreflopChartConfigOption {
    /// Resolve to a PreflopChartConfig.
    ///
    /// Preset names are not currently supported - use inline configuration.
    pub fn resolve(&self) -> Result<PreflopChartConfig, AgentConfigError> {
        match self {
            PreflopChartConfigOption::Preset(name) => Err(
                AgentConfigError::PreflopChartPresetUnavailable(name.clone()),
            ),
            PreflopChartConfigOption::Inline(config) => Ok(config.clone()),
        }
    }
}

fn default_percent_fold() -> Vec<f64> {
    vec![0.25, 0.30, 0.50]
}

fn default_percent_call() -> Vec<f64> {
    vec![0.5, 0.6, 0.45]
}

/// Exploration policy for a CFR agent — just the unified budget. All
/// the previously-separate knobs (`max_recursion_depth`, `concurrency`,
/// `act_deadline_ms`) now live inside the budget tree.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct CfrExploration {
    /// Per-config budget. `None` = use whatever default the caller
    /// (binary or library user) decides. Explicit `Some(...)` is the
    /// agent's own budget — used to mix stronger and weaker agents in
    /// one comparison.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget: Option<BudgetConfig>,
}

/// Which opponent hand-distribution estimator a CFR agent uses.
// Task 12 will wire this into AgentConfig variants; suppress dead_code until then.
#[expect(dead_code, reason = "wired into AgentConfig variants in Task 12")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EstimatorConfig {
    /// Use the real opponent hands (today's behavior). Default.
    #[default]
    Known,
    /// Sample opponents uniformly from the remaining deck.
    Uniform,
}

impl EstimatorConfig {
    /// Construct the configured estimator.
    #[expect(dead_code, reason = "called from AgentConfig variants in Task 12")]
    pub fn build(&self) -> Arc<dyn HandDistributionEstimator> {
        match self {
            EstimatorConfig::Known => Arc::new(KnownHandsEstimator),
            EstimatorConfig::Uniform => Arc::new(UniformRandomEstimator),
        }
    }
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

    /// Failed to validate a configurable action generator config.
    #[error("invalid configurable action config: {0}")]
    ConfigurableActionConfig(#[from] ConfigurableActionConfigError),

    /// Failed to validate a preflop chart config.
    #[error("invalid preflop chart config: {0}")]
    PreflopChartConfig(#[from] PreflopChartConfigError),

    /// A preflop chart preset name was supplied, but presets are not yet
    /// supported. Use an inline configuration instead.
    #[error(
        "preflop chart preset '{0}' is not available; use inline configuration instead. \
         See examples/configs/preflop_6max_rfi.json for an example"
    )]
    PreflopChartPresetUnavailable(String),
}

impl AgentConfig {
    /// Returns true if this agent config produces a CFR-based agent.
    ///
    /// CFR agents benefit from sharing CFR states and a `TraversalSet`
    /// across all agents in a simulation. Use this to decide whether to
    /// create shared CFR context.
    pub fn is_cfr(&self) -> bool {
        matches!(
            self,
            AgentConfig::CfrBasic { .. }
                | AgentConfig::CfrSimple { .. }
                | AgentConfig::CfrConfigurable { .. }
                | AgentConfig::CfrPreflopChart { .. }
        )
    }

    /// Build the optional shared CFR context for a group of agent configs.
    ///
    /// If any config in `configs` is CFR-based, all CFR agents in the
    /// simulation should share a single `CFRState` (rooted at `game_state`)
    /// and a single `TraversalSet` (sized for `num_players`). When no config
    /// is CFR-based, `None` is returned and no CFR allocations are made.
    pub fn maybe_shared_cfr_context<'a, I>(
        configs: I,
        game_state: &GameState,
        num_players: usize,
    ) -> Option<(CFRState, TraversalSet)>
    where
        I: IntoIterator<Item = &'a AgentConfig>,
    {
        let has_cfr = configs.into_iter().any(|c| c.is_cfr());
        has_cfr.then(|| {
            (
                CFRState::new(game_state.clone()),
                TraversalSet::new(num_players),
            )
        })
    }

    /// Fill `exploration.budget` on every CFR variant where it's `None`.
    /// Explicit `Some(...)` values are preserved untouched — that's how
    /// per-agent budgets express stronger-vs-weaker comparisons.
    pub fn fill_default_budget(&mut self, default: &BudgetConfig) {
        let target = match self {
            AgentConfig::CfrBasic { exploration, .. }
            | AgentConfig::CfrSimple { exploration, .. }
            | AgentConfig::CfrConfigurable { exploration, .. }
            | AgentConfig::CfrPreflopChart { exploration, .. } => &mut exploration.budget,
            _ => return,
        };
        if target.is_none() {
            *target = Some(default.clone());
        }
    }

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
            AgentConfig::CfrConfigurable { action_config, .. } => {
                action_config.validate()?;
            }
            AgentConfig::CfrPreflopChart { preflop_config, .. } => {
                let resolved = preflop_config.resolve()?;
                resolved.validate()?;
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

/// Builder that creates agents from configuration.
///
/// CFR agents created by this builder will automatically initialize
/// CFR state for all players in the game, enabling proper mixed-agent
/// simulations.
///
/// For shared CFR learning across agents, use `cfr_context()` to provide
/// shared CFR states and a `TraversalSet`. When absent, each CFR agent
/// creates its own.
///
/// # Example
///
/// ```rust
/// use rs_poker::arena::agent::ConfigAgentBuilder;
/// use rs_poker::arena::GameStateBuilder;
///
/// let game_state = GameStateBuilder::new()
///     .num_players_with_stack(2, 100.0)
///     .blinds(10.0, 5.0)
///     .build()
///     .unwrap();
///
/// let agent = ConfigAgentBuilder::from_json(r#"{"type": "calling"}"#)
///     .unwrap()
///     .player_idx(0)
///     .game_state(game_state)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct ConfigAgentBuilder {
    config: AgentConfig,
    player_idx: Option<usize>,
    game_state: Option<GameState>,
    cfr_state: Option<CFRState>,
    traversal_set: Option<TraversalSet>,
    rng_seed: Option<u64>,
}

impl ConfigAgentBuilder {
    /// Create a new builder from a validated config.
    pub fn new(config: AgentConfig) -> Result<Self, AgentConfigError> {
        config.validate()?;
        Ok(Self {
            config,
            player_idx: None,
            game_state: None,
            cfr_state: None,
            traversal_set: None,
            rng_seed: None,
        })
    }

    /// Set the player index for the agent.
    pub fn player_idx(mut self, idx: usize) -> Self {
        self.player_idx = Some(idx);
        self
    }

    /// Set the game state for the agent.
    ///
    /// For CFR agents, this eagerly initializes a shared CFR state and a
    /// `TraversalSet` (unless explicit context was already provided via
    /// `cfr_context()`). This means cloned builders automatically share
    /// the same CFR state (CFRState is cheap to clone via Arc).
    pub fn game_state(mut self, game_state: GameState) -> Self {
        // Eagerly create CFR context so that cloned builders share the
        // same Arc-backed state. Explicit cfr_context() takes priority.
        if self.config.is_cfr() && self.cfr_state.is_none() {
            self.cfr_state = Some(CFRState::new(game_state.clone()));
            self.traversal_set = Some(TraversalSet::new(game_state.num_players));
        }
        self.game_state = Some(game_state);
        self
    }

    /// Provide shared CFR context.
    ///
    /// When set, all CFR agents built will share the same CFR states
    /// and `TraversalSet`, enabling shared learning across agents.
    /// When not set, each CFR agent creates its own.
    pub fn cfr_context(mut self, cfr_state: CFRState, traversal_set: TraversalSet) -> Self {
        self.cfr_state = Some(cfr_state);
        self.traversal_set = Some(traversal_set);
        self
    }

    /// Set an RNG seed for the agent.
    ///
    /// When set, agents that use randomness (Random, RandomPotControl) will use
    /// a deterministic RNG seeded from this value instead of system entropy.
    /// CFR agents draw from a thread-local RNG and ignore this seed.
    pub fn rng_seed(mut self, seed: u64) -> Self {
        self.rng_seed = Some(seed);
        self
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Create from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, AgentConfigError> {
        let config: AgentConfig = serde_json::from_str(json)?;
        Self::new(config)
    }

    /// Create from a file path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, AgentConfigError> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json)
    }

    /// Try to parse as file path first, then as inline JSON.
    pub fn from_str_or_file(input: &str) -> Result<Self, AgentConfigError> {
        match Self::from_file(input) {
            Ok(builder) => Ok(builder),
            Err(AgentConfigError::IoError(err)) if err.kind() == ErrorKind::NotFound => {
                Self::from_json(input)
            }
            Err(err) => Err(err),
        }
    }

    /// Build the agent.
    ///
    /// # Panics
    ///
    /// Panics if `player_idx` has not been set.
    /// Panics if `game_state` has not been set for CFR agent types.
    pub fn build(self) -> Box<dyn Agent> {
        let player_idx = self.player_idx.expect("player_idx is required");

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
            } => {
                let agent_name = resolve_agent_name(name, "RandomAgent", player_idx);
                if let Some(seed) = self.rng_seed {
                    Box::new(RandomAgent::new_with_seed(
                        agent_name,
                        percent_fold.clone(),
                        percent_call.clone(),
                        seed,
                    ))
                } else {
                    Box::new(RandomAgent::new(
                        agent_name,
                        percent_fold.clone(),
                        percent_call.clone(),
                    ))
                }
            }
            AgentConfig::RandomPotControl { name, percent_call } => {
                let agent_name = resolve_agent_name(name, "RandomPotControlAgent", player_idx);
                if let Some(seed) = self.rng_seed {
                    Box::new(RandomPotControlAgent::new_with_seed(
                        agent_name,
                        percent_call.clone(),
                        seed,
                    ))
                } else {
                    Box::new(RandomPotControlAgent::new(agent_name, percent_call.clone()))
                }
            }
            AgentConfig::CfrBasic { name, exploration } => {
                let (cfr_state, traversal_set) = self.resolve_cfr_context();
                let budget = exploration.budget.clone().unwrap_or_default().build();
                let builder = CFRAgentBuilder::<BasicCFRActionGenerator>::new()
                    .name(resolve_agent_name(name, "CFRAgent", player_idx))
                    .player_idx(player_idx)
                    .cfr_state(cfr_state)
                    .traversal_set(traversal_set)
                    .budget(budget)
                    .action_gen_config(());
                Box::new(builder.build())
            }
            AgentConfig::CfrSimple { name, exploration } => {
                let (cfr_state, traversal_set) = self.resolve_cfr_context();
                let budget = exploration.budget.clone().unwrap_or_default().build();
                let builder = CFRAgentBuilder::<SimpleActionGenerator>::new()
                    .name(resolve_agent_name(name, "CFRSimpleAgent", player_idx))
                    .player_idx(player_idx)
                    .cfr_state(cfr_state)
                    .traversal_set(traversal_set)
                    .budget(budget)
                    .action_gen_config(());
                Box::new(builder.build())
            }
            AgentConfig::CfrConfigurable {
                name,
                exploration,
                action_config,
            } => {
                let (cfr_state, traversal_set) = self.resolve_cfr_context();
                let budget = exploration.budget.clone().unwrap_or_default().build();
                let builder = CFRAgentBuilder::<ConfigurableActionGenerator>::new()
                    .name(resolve_agent_name(name, "CFRConfigurableAgent", player_idx))
                    .player_idx(player_idx)
                    .cfr_state(cfr_state)
                    .traversal_set(traversal_set)
                    .budget(budget)
                    .action_gen_config(action_config.as_ref().clone());
                Box::new(builder.build())
            }
            AgentConfig::CfrPreflopChart {
                name,
                exploration,
                preflop_config,
                postflop_config,
            } => {
                let resolved_preflop_config = preflop_config
                    .resolve()
                    .expect("Invalid preflop config - should have been validated");
                let (cfr_state, traversal_set) = self.resolve_cfr_context();
                let action_config = PreflopChartActionConfig {
                    preflop_config: resolved_preflop_config,
                    postflop_config: postflop_config
                        .as_ref()
                        .map(|c| c.as_ref().clone())
                        .unwrap_or_default(),
                };
                let budget = exploration.budget.clone().unwrap_or_default().build();
                let builder = CFRAgentBuilder::<PreflopChartActionGenerator>::new()
                    .name(resolve_agent_name(name, "CFRPreflopChartAgent", player_idx))
                    .player_idx(player_idx)
                    .cfr_state(cfr_state)
                    .traversal_set(traversal_set)
                    .budget(budget)
                    .action_gen_config(action_config);
                Box::new(builder.build())
            }
        }
    }

    /// Get the shared CFR states and TraversalSet for CFR agents.
    ///
    /// CFR context is initialized either by `cfr_context()` or eagerly
    /// by `game_state()` when the config is CFR. Cloned builders share
    /// the same Arc-backed states.
    ///
    /// # Panics
    ///
    /// Panics if neither `cfr_context()` nor `game_state()` was called.
    fn resolve_cfr_context(&self) -> (CFRState, TraversalSet) {
        let cfr_state = self
            .cfr_state
            .clone()
            .expect("cfr_context() or game_state() is required for CFR agents");
        let traversal_set = self
            .traversal_set
            .clone()
            .expect("cfr_context() or game_state() is required for CFR agents");
        (cfr_state, traversal_set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::GameStateBuilder;
    use crate::arena::cfr::BudgetItem;

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

    // Builder tests
    #[test]
    fn test_create_from_config() {
        let config = AgentConfig::AllIn { name: None };
        let _agent = ConfigAgentBuilder::new(config)
            .unwrap()
            .player_idx(0)
            .build();
        // Agent should be created successfully (test passes if no panic)
    }

    #[test]
    fn test_from_json() {
        let json = r#"{"type":"calling"}"#;
        let generator = ConfigAgentBuilder::from_json(json).unwrap();
        assert!(matches!(generator.config, AgentConfig::Calling { .. }));
    }

    #[test]
    fn test_from_json_with_params() {
        let json = r#"{"type":"random","percent_fold":[0.2],"percent_call":[0.5]}"#;
        let generator = ConfigAgentBuilder::from_json(json).unwrap();
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
        assert!(ConfigAgentBuilder::from_json(json).is_err());
    }

    #[test]
    fn test_build_multiple_agents() {
        let config = AgentConfig::Random {
            name: None,
            percent_fold: vec![0.25],
            percent_call: vec![0.5],
        };

        // Build multiple agents by cloning the builder
        let builder = ConfigAgentBuilder::new(config).unwrap();
        let _agent1 = builder.clone().player_idx(0).build();
        let _agent2 = builder.player_idx(1).build();
        // Both should be created successfully (test passes if no panic)
    }

    #[test]
    fn test_from_str_or_file_inline_json() {
        let json = r#"{"type":"all_in"}"#;
        let generator = ConfigAgentBuilder::from_str_or_file(json).unwrap();
        assert!(matches!(generator.config, AgentConfig::AllIn { .. }));
    }

    // CFR tests
    #[test]
    fn test_cfr_basic_config() {
        let json = r#"{
            "type": "cfr_basic",
            "exploration": {
                "budget": [
                    { "type": "per_depth_iterations", "counts": [10, 5, 1] }
                ]
            }
        }"#;
        let config: AgentConfig = serde_json::from_str(json).unwrap();
        match config {
            AgentConfig::CfrBasic { name, exploration } => {
                assert!(name.is_none());
                assert_eq!(
                    exploration.budget,
                    Some(BudgetConfig(vec![BudgetItem::PerDepthIterations {
                        counts: vec![10, 5, 1],
                        fallback: 1,
                    }]))
                );
            }
            _ => panic!("Expected CfrBasic variant"),
        }
    }

    #[test]
    fn test_cfr_basic_defaults() {
        let json = r#"{"type":"cfr_basic"}"#;
        let config: AgentConfig = serde_json::from_str(json).unwrap();
        match config {
            AgentConfig::CfrBasic { name, exploration } => {
                assert!(name.is_none());
                assert_eq!(exploration, CfrExploration::default());
                // Default = budget unset; the build path will substitute
                // BudgetConfig::default() at construction time.
                assert!(exploration.budget.is_none());
            }
            _ => panic!("Expected CfrBasic variant"),
        }
    }

    #[test]
    fn cfr_configurable_parses_exploration() {
        let json = r#"{
            "type": "cfr_configurable",
            "name": "X",
            "exploration": {
                "budget": [
                    { "type": "per_depth_iterations", "counts": [100,5,3,1] }
                ]
            },
            "action_config": {
                "preflop": {"call_enabled": true,"raise_mult":[2.5],"pot_mult":[],"setup_shove":false,"all_in":true},
                "flop":    {"call_enabled": true,"raise_mult":[],"pot_mult":[0.33,1.0],"setup_shove":false,"all_in":false},
                "turn":    {"call_enabled": true,"raise_mult":[],"pot_mult":[0.67,1.0],"setup_shove":true,"all_in":false},
                "river":   {"call_enabled": true,"raise_mult":[],"pot_mult":[0.67,1.5],"setup_shove":true,"all_in":true}
            }
        }"#;
        let cfg: AgentConfig = serde_json::from_str(json).unwrap();
        match cfg {
            AgentConfig::CfrConfigurable { exploration, .. } => {
                assert_eq!(
                    exploration.budget,
                    Some(BudgetConfig(vec![BudgetItem::PerDepthIterations {
                        counts: vec![100, 5, 3, 1],
                        fallback: 1,
                    }]))
                );
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_cfr_agent_builder() {
        let config = AgentConfig::CfrBasic {
            name: None,
            exploration: CfrExploration {
                budget: Some(BudgetConfig(vec![BudgetItem::PerDepthIterations {
                    counts: vec![5, 1],
                    fallback: 1,
                }])),
            },
        };
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let _agent = ConfigAgentBuilder::new(config)
            .unwrap()
            .player_idx(0)
            .game_state(game_state)
            .build();
        // Agent should be created successfully (test passes if no panic)
    }

    #[test]
    fn test_cfr_agent_builder_depth_based() {
        let config = AgentConfig::CfrBasic {
            name: Some("TestCFR".to_string()),
            exploration: CfrExploration {
                budget: Some(BudgetConfig(vec![BudgetItem::PerDepthIterations {
                    counts: vec![3, 2, 1],
                    fallback: 1,
                }])),
            },
        };
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(3, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let agent = ConfigAgentBuilder::new(config)
            .unwrap()
            .player_idx(1)
            .game_state(game_state)
            .build();
        assert_eq!(agent.name(), "TestCFR");
    }

    #[test]
    fn cfr_configurable_example_loads_and_builds() {
        let json = std::fs::read_to_string("examples/configs/cfr_configurable.json")
            .expect("example config should be readable");
        let config: AgentConfig =
            serde_json::from_str(&json).expect("example config should deserialize");
        assert!(matches!(&config, AgentConfig::CfrConfigurable { .. }));
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(2, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let _agent = ConfigAgentBuilder::new(config)
            .unwrap()
            .player_idx(0)
            .game_state(game_state)
            .build();
        // Builds against a GameState without panicking.
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
        let result = ConfigAgentBuilder::from_str_or_file(json_input);

        // Should succeed by parsing as JSON (not as file)
        assert!(
            result.is_ok(),
            "Valid JSON should parse when file not found"
        );

        // Test with a path that looks like a file but doesn't exist
        let nonexistent = "/nonexistent/path/to/config.json";
        let result = ConfigAgentBuilder::from_str_or_file(nonexistent);

        // Should try as JSON and fail (not valid JSON)
        assert!(
            result.is_err(),
            "Non-existent file with invalid JSON should fail"
        );
    }

    #[test]
    fn cfr_exploration_default_is_empty_budget() {
        // Default is "no budget set" — callers (binary / library users)
        // are expected to fill it in via `fill_default_budget`.
        let e = CfrExploration::default();
        assert!(e.budget.is_none());
    }

    #[test]
    fn cfr_exploration_round_trips() {
        let json = r#"{
            "budget": [
                { "type": "deadline", "millis": 250 },
                { "type": "per_depth_iterations", "counts": [50,40,30,20,10,5] }
            ]
        }"#;
        let e: CfrExploration = serde_json::from_str(json).unwrap();
        assert_eq!(
            e.budget,
            Some(BudgetConfig(vec![
                BudgetItem::Deadline { millis: 250 },
                BudgetItem::PerDepthIterations {
                    counts: vec![50, 40, 30, 20, 10, 5],
                    fallback: 1,
                },
            ]))
        );
    }

    #[test]
    fn fill_default_budget_fills_none_and_preserves_some() {
        let mut none_cfg = AgentConfig::CfrBasic {
            name: None,
            exploration: CfrExploration::default(),
        };
        let custom = BudgetConfig(vec![BudgetItem::IterationCount { max: 7 }]);
        none_cfg.fill_default_budget(&custom);
        match &none_cfg {
            AgentConfig::CfrBasic { exploration, .. } => {
                assert_eq!(exploration.budget.as_ref(), Some(&custom));
            }
            _ => unreachable!(),
        }

        let explicit = BudgetConfig(vec![BudgetItem::Deadline { millis: 42 }]);
        let mut some_cfg = AgentConfig::CfrBasic {
            name: None,
            exploration: CfrExploration {
                budget: Some(explicit.clone()),
            },
        };
        some_cfg.fill_default_budget(&custom);
        match &some_cfg {
            AgentConfig::CfrBasic { exploration, .. } => {
                assert_eq!(exploration.budget.as_ref(), Some(&explicit));
            }
            _ => unreachable!(),
        }

        // Non-CFR variants are no-ops.
        let mut non_cfr = AgentConfig::AllIn { name: None };
        non_cfr.fill_default_budget(&custom);
        assert!(matches!(non_cfr, AgentConfig::AllIn { name: None }));
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

    #[test]
    fn hand_estimator_config_defaults_to_known() {
        assert_eq!(EstimatorConfig::default(), EstimatorConfig::Known);
    }

    #[test]
    fn hand_estimator_config_round_trips_json() {
        let parsed: EstimatorConfig = serde_json::from_str("\"uniform\"").unwrap();
        assert_eq!(parsed, EstimatorConfig::Uniform);
    }
}
