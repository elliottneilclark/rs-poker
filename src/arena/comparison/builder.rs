use std::path::{Path, PathBuf};

use crate::arena::agent::AgentConfig;

use super::config::ComparisonConfig;
use super::error::{ComparisonError, Result};
use super::runner::ArenaComparison;

/// Builder for constructing ArenaComparison instances
///
/// # Example
///
/// ```ignore
/// use rs_poker::arena::comparison::ComparisonBuilder;
///
/// let comparison = ComparisonBuilder::new()
///     .num_games(1000)
///     .players_per_table(3)
///     .big_blind(10.0)
///     .small_blind(5.0)
///     .add_agent_config(agent_config)
///     .seed(42)
///     .build()?;
/// ```
#[derive(Debug, Default)]
pub struct ComparisonBuilder {
    agents: Vec<(String, AgentConfig)>,
    num_games: Option<usize>,
    players_per_table: Option<usize>,
    big_blind: Option<f32>,
    small_blind: Option<f32>,
    min_stack_bb: Option<f32>,
    max_stack_bb: Option<f32>,
    ante: Option<f32>,
    output_dir: Option<PathBuf>,
    seed: Option<u64>,
}

impl ComparisonBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of unique game states to test
    pub fn num_games(mut self, num_games: usize) -> Self {
        self.num_games = Some(num_games);
        self
    }

    /// Set the number of players per table
    pub fn players_per_table(mut self, players_per_table: usize) -> Self {
        self.players_per_table = Some(players_per_table);
        self
    }

    /// Set the big blind amount
    pub fn big_blind(mut self, big_blind: f32) -> Self {
        self.big_blind = Some(big_blind);
        self
    }

    /// Set the small blind amount
    pub fn small_blind(mut self, small_blind: f32) -> Self {
        self.small_blind = Some(small_blind);
        self
    }

    /// Set the minimum starting stack in big blinds
    pub fn min_stack_bb(mut self, min_stack_bb: f32) -> Self {
        self.min_stack_bb = Some(min_stack_bb);
        self
    }

    /// Set the maximum starting stack in big blinds
    pub fn max_stack_bb(mut self, max_stack_bb: f32) -> Self {
        self.max_stack_bb = Some(max_stack_bb);
        self
    }

    /// Set the ante amount
    pub fn ante(mut self, ante: f32) -> Self {
        self.ante = Some(ante);
        self
    }

    /// Set the output directory for results and hand histories
    pub fn output_dir<P: AsRef<Path>>(mut self, output_dir: P) -> Self {
        self.output_dir = Some(output_dir.as_ref().to_path_buf());
        self
    }

    /// Set the random seed for reproducibility
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Add an agent configuration with an optional name
    pub fn add_agent(mut self, name: String, config: AgentConfig) -> Self {
        self.agents.push((name, config));
        self
    }

    /// Add an agent configuration, using the config's name or a default
    pub fn add_agent_config(mut self, config: AgentConfig) -> Self {
        let name = get_agent_name(&config, &format!("Agent{}", self.agents.len()));
        self.agents.push((name, config));
        self
    }

    /// Add multiple agent configurations
    pub fn add_agents(mut self, agents: Vec<(String, AgentConfig)>) -> Self {
        self.agents.extend(agents);
        self
    }

    /// Load agent configurations from a directory of JSON files
    pub fn load_agents_from_dir<P: AsRef<Path>>(mut self, dir: P) -> Result<Self> {
        let agents = load_agents_from_dir(dir.as_ref())?;
        self.agents.extend(agents);
        Ok(self)
    }

    /// Build the ArenaComparison
    ///
    /// Returns an error if configuration is invalid or no agents are configured.
    pub fn build(self) -> Result<ArenaComparison> {
        // Check we have agents
        if self.agents.is_empty() {
            return Err(ComparisonError::MissingConfig(
                "No agents configured. Use add_agent(), add_agent_config(), or load_agents_from_dir()".to_string(),
            ));
        }

        // Build config with defaults
        let config = ComparisonConfig {
            num_games: self.num_games.unwrap_or(1000),
            players_per_table: self.players_per_table.unwrap_or(3),
            big_blind: self.big_blind.unwrap_or(10.0),
            small_blind: self.small_blind.unwrap_or(5.0),
            min_stack_bb: self.min_stack_bb.unwrap_or(100.0),
            max_stack_bb: self.max_stack_bb.unwrap_or(100.0),
            ante: self.ante.unwrap_or(0.0),
            output_dir: self.output_dir,
            seed: self.seed,
        };

        // Validate configuration
        config.validate(self.agents.len())?;

        // Validate all agent configs up front
        for (_, agent_config) in &self.agents {
            agent_config
                .validate()
                .map_err(ComparisonError::InvalidConfig)?;
        }

        Ok(ArenaComparison::new(config, self.agents))
    }
}

/// Extract the name from an AgentConfig, using the name field if present
fn get_agent_name(config: &AgentConfig, fallback_name: &str) -> String {
    match config {
        AgentConfig::AllIn { name, .. }
        | AgentConfig::Calling { name, .. }
        | AgentConfig::Folding { name, .. }
        | AgentConfig::Random { name, .. }
        | AgentConfig::RandomPotControl { name, .. }
        | AgentConfig::CfrBasic { name, .. }
        | AgentConfig::CfrSimple { name, .. }
        | AgentConfig::CfrConfigurable { name, .. }
        | AgentConfig::CfrPreflopChart { name, .. } => {
            name.clone().unwrap_or_else(|| fallback_name.to_string())
        }
    }
}

/// Load agent configurations from a directory of JSON files
fn load_agents_from_dir(dir: &Path) -> Result<Vec<(String, AgentConfig)>> {
    if !dir.exists() {
        return Err(ComparisonError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Directory does not exist: {}", dir.display()),
        )));
    }

    if !dir.is_dir() {
        return Err(ComparisonError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Path is not a directory: {}", dir.display()),
        )));
    }

    let mut agents = Vec::new();
    let entries = std::fs::read_dir(dir)?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        // Only process .json files
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }

        let fallback_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        match load_agent_config(&path) {
            Ok(config) => {
                let agent_name = get_agent_name(&config, &fallback_name);
                agents.push((agent_name, config));
            }
            Err(e) => {
                tracing::warn!("Skipping invalid config file {}: {}", path.display(), e);
            }
        }
    }

    if agents.is_empty() {
        return Err(ComparisonError::NoAgentsFound(dir.display().to_string()));
    }

    // Sort by agent name for consistent ordering
    agents.sort_by(|a, b| a.0.cmp(&b.0));

    Ok(agents)
}

/// Load and validate a single agent configuration from a file
fn load_agent_config(path: &Path) -> Result<AgentConfig> {
    let contents = std::fs::read_to_string(path)?;
    let config: AgentConfig =
        serde_json::from_str(&contents).map_err(|source| ComparisonError::ParseConfig {
            path: path.display().to_string(),
            source,
        })?;

    // Validate the configuration
    config.validate()?;

    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let builder = ComparisonBuilder::new()
            .add_agent_config(AgentConfig::Folding { name: None })
            .add_agent_config(AgentConfig::Calling { name: None })
            .add_agent_config(AgentConfig::AllIn { name: None });

        let comparison = builder.build().unwrap();
        let config = comparison.config();

        assert_eq!(config.num_games, 1000);
        assert_eq!(config.players_per_table, 3);
        assert_eq!(config.big_blind, 10.0);
        assert_eq!(config.small_blind, 5.0);
    }

    #[test]
    fn test_builder_custom_config() {
        let comparison = ComparisonBuilder::new()
            .num_games(500)
            .players_per_table(2)
            .big_blind(20.0)
            .small_blind(10.0)
            .min_stack_bb(50.0)
            .max_stack_bb(150.0)
            .seed(42)
            .add_agent_config(AgentConfig::Folding { name: None })
            .add_agent_config(AgentConfig::Calling { name: None })
            .build()
            .unwrap();

        let config = comparison.config();
        assert_eq!(config.num_games, 500);
        assert_eq!(config.players_per_table, 2);
        assert_eq!(config.big_blind, 20.0);
        assert_eq!(config.small_blind, 10.0);
        assert_eq!(config.min_stack_bb, 50.0);
        assert_eq!(config.max_stack_bb, 150.0);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_builder_no_agents_error() {
        let result = ComparisonBuilder::new().build();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ComparisonError::MissingConfig(_)
        ));
    }

    #[test]
    fn test_builder_validation_error() {
        let result = ComparisonBuilder::new()
            .players_per_table(1) // Invalid: must be >= 2
            .add_agent_config(AgentConfig::Folding { name: None })
            .add_agent_config(AgentConfig::Calling { name: None })
            .build();

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ComparisonError::ValidationError(_)
        ));
    }

    #[test]
    fn test_get_agent_name_with_name() {
        let config = AgentConfig::AllIn {
            name: Some("MyAgent".to_string()),
        };
        assert_eq!(get_agent_name(&config, "fallback"), "MyAgent");
    }

    #[test]
    fn test_get_agent_name_without_name() {
        let config = AgentConfig::AllIn { name: None };
        assert_eq!(get_agent_name(&config, "fallback"), "fallback");
    }
}
