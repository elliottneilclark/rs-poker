use std::collections::HashMap;
use std::path::Path;

use super::config::ComparisonConfig;
use super::error::{ComparisonError, Result};
use super::stats::AgentStats;

/// Results of an agent comparison
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Agent names in order
    agent_names: Vec<String>,
    /// Statistics for each agent, keyed by agent name
    agent_stats: HashMap<String, AgentStats>,
    /// The configuration used for this comparison
    config: ComparisonConfig,
    /// Total number of permutations run
    total_permutations: usize,
}

impl ComparisonResult {
    /// Create a new comparison result
    pub fn new(
        agent_names: Vec<String>,
        agent_stats: HashMap<String, AgentStats>,
        config: ComparisonConfig,
        total_permutations: usize,
    ) -> Self {
        Self {
            agent_names,
            agent_stats,
            config,
            total_permutations,
        }
    }

    /// Get the agent names
    pub fn agent_names(&self) -> &[String] {
        &self.agent_names
    }

    /// Get statistics for a specific agent
    pub fn get_agent_stats(&self, name: &str) -> Option<&AgentStats> {
        self.agent_stats.get(name)
    }

    /// Get all agent statistics
    pub fn all_stats(&self) -> &HashMap<String, AgentStats> {
        &self.agent_stats
    }

    /// Get the configuration used
    pub fn config(&self) -> &ComparisonConfig {
        &self.config
    }

    /// Get total permutations run
    pub fn total_permutations(&self) -> usize {
        self.total_permutations
    }

    /// Get total games simulated
    pub fn total_games(&self) -> usize {
        self.total_permutations * self.config.num_games
    }

    /// Get rankings sorted by profit per game (descending)
    pub fn get_rankings(&self) -> Vec<(&String, &AgentStats)> {
        let mut rankings: Vec<_> = self.agent_stats.iter().collect();
        rankings.sort_by(|a, b| {
            b.1.profit_per_game
                .partial_cmp(&a.1.profit_per_game)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rankings
    }

    /// Format results as Markdown output
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        // Header
        output.push_str(&format!("{}\n", "=".repeat(80)));
        output.push_str("# Agent Comparison Results\n");
        output.push_str(&format!("{}\n\n", "=".repeat(80)));

        // Configuration section
        output.push_str("## Configuration\n\n");
        output.push_str(&format!(
            "- **Agents Tested**: {}\n",
            self.agent_names.len()
        ));
        output.push_str(&format!(
            "- **Players per Table**: {}\n",
            self.config.players_per_table
        ));
        output.push_str(&format!(
            "- **Unique Game States**: {}\n",
            self.config.num_games
        ));
        output.push_str(&format!(
            "- **Total Permutations**: {}\n",
            self.total_permutations
        ));
        output.push_str(&format!(
            "- **Total Games Simulated**: {}\n",
            self.total_games()
        ));
        output.push_str(&format!("- **Big Blind**: {}\n", self.config.big_blind));
        output.push_str(&format!("- **Small Blind**: {}\n", self.config.small_blind));
        output.push_str(&format!(
            "- **Stack Range**: {}-{} BB\n",
            self.config.min_stack_bb, self.config.max_stack_bb
        ));
        if let Some(seed) = self.config.seed {
            output.push_str(&format!("- **Random Seed**: {}\n", seed));
        }
        output.push('\n');

        // Rankings
        output.push_str("## Rankings (by Profit per Game)\n\n");
        let rankings = self.get_rankings();
        for (rank, (agent_name, stats)) in rankings.iter().enumerate() {
            let profit_bb = stats.profit_per_game / self.config.big_blind;
            output.push_str(&format!(
                "{}. **{}**: {:+.2} bb/game (ROI: {:+.1}%)\n",
                rank + 1,
                agent_name,
                profit_bb,
                stats.roi_percent
            ));
        }
        output.push('\n');

        // Detailed statistics for each agent
        output.push_str("## Detailed Statistics\n\n");
        for (agent_name, stats) in rankings.iter() {
            output.push_str(&format!("### {}\n\n", agent_name));

            // Financial Performance
            output.push_str("#### Financial Performance\n\n");
            output.push_str("| Metric | Value |\n");
            output.push_str("|--------|-------|\n");
            output.push_str(&format!(
                "| Total Profit | {:+.2} chips ({:+.2} bb) |\n",
                stats.total_profit,
                stats.total_profit / self.config.big_blind
            ));
            output.push_str(&format!("| Games Played | {} |\n", stats.total_games));
            output.push_str(&format!(
                "| Wins | {} ({:.1}%) |\n",
                stats.wins,
                if stats.total_games > 0 {
                    100.0 * stats.wins as f32 / stats.total_games as f32
                } else {
                    0.0
                }
            ));
            output.push_str(&format!(
                "| Losses | {} ({:.1}%) |\n",
                stats.losses,
                if stats.total_games > 0 {
                    100.0 * stats.losses as f32 / stats.total_games as f32
                } else {
                    0.0
                }
            ));
            output.push_str(&format!(
                "| Breakeven | {} ({:.1}%) |\n",
                stats.breakeven,
                if stats.total_games > 0 {
                    100.0 * stats.breakeven as f32 / stats.total_games as f32
                } else {
                    0.0
                }
            ));
            output.push_str(&format!(
                "| Profit/Game | {:+.2} bb |\n",
                stats.profit_per_game / self.config.big_blind
            ));
            output.push_str(&format!(
                "| Profit/100 Hands | {:+.2} bb |\n",
                stats.profit_per_100_hands / self.config.big_blind
            ));
            output.push_str(&format!("| ROI | {:+.1}% |\n", stats.roi_percent));
            output.push('\n');

            // Playing Style
            output.push_str("#### Playing Style\n\n");
            output.push_str("| Metric | Value |\n");
            output.push_str("|--------|-------|\n");
            output.push_str(&format!("| VPIP | {:.1}% |\n", stats.vpip_percent));
            output.push_str(&format!("| PFR | {:.1}% |\n", stats.pfr_percent));
            output.push_str(&format!("| 3-Bet % | {:.1}% |\n", stats.three_bet_percent));
            output.push_str(&format!("| ATS % | {:.1}% |\n", stats.steal_percent));
            output.push_str(&format!(
                "| Aggression Factor | {:.2} |\n",
                stats.aggression_factor
            ));
            output.push_str(&format!(
                "| Aggression Frequency | {:.1}% |\n",
                stats.aggression_frequency
            ));
            output.push('\n');

            // Post-Flop Stats
            output.push_str("#### Post-Flop Stats\n\n");
            output.push_str("| Metric | Value |\n");
            output.push_str("|--------|-------|\n");
            output.push_str(&format!("| C-Bet % | {:.1}% |\n", stats.cbet_percent));
            output.push_str(&format!("| WTSD % | {:.1}% |\n", stats.wtsd_percent));
            output.push_str(&format!("| W$SD % | {:.1}% |\n", stats.wsd_percent));

            let flop_af_str = if stats.flop_aggression_factor.is_infinite() {
                "∞".to_string()
            } else {
                format!("{:.2}", stats.flop_aggression_factor)
            };
            let turn_af_str = if stats.turn_aggression_factor.is_infinite() {
                "∞".to_string()
            } else {
                format!("{:.2}", stats.turn_aggression_factor)
            };
            let river_af_str = if stats.river_aggression_factor.is_infinite() {
                "∞".to_string()
            } else {
                format!("{:.2}", stats.river_aggression_factor)
            };
            output.push_str(&format!("| Flop AF | {} |\n", flop_af_str));
            output.push_str(&format!("| Turn AF | {} |\n", turn_af_str));
            output.push_str(&format!("| River AF | {} |\n", river_af_str));
            output.push('\n');

            // Round-by-Round Win Rates
            output.push_str("#### Round-by-Round Win Rates\n\n");
            output.push_str("| Round | Win Rate |\n");
            output.push_str("|-------|----------|\n");
            output.push_str(&format!("| Preflop | {:.1}% |\n", stats.preflop_win_rate));
            output.push_str(&format!("| Flop | {:.1}% |\n", stats.flop_win_rate));
            output.push_str(&format!("| Turn | {:.1}% |\n", stats.turn_win_rate));
            output.push_str(&format!("| River | {:.1}% |\n", stats.river_win_rate));
            output.push('\n');

            // Position Performance
            if !stats.position_stats.is_empty() {
                output.push_str("#### Position Performance\n\n");
                output.push_str("| Position (Seat) | Profit/Game | Games Played |\n");
                output.push_str("|-----------------|-------------|-------------|\n");
                for pos_stat in &stats.position_stats {
                    output.push_str(&format!(
                        "| Seat {} | {:+.2} bb | {} |\n",
                        pos_stat.seat_index,
                        pos_stat.profit_per_game / self.config.big_blind,
                        pos_stat.games_played
                    ));
                }
                output.push('\n');
            }

            output.push_str("---\n\n");
        }

        output
    }

    /// Serialize agent stats to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.agent_stats).map_err(ComparisonError::from)
    }

    /// Save results to JSON and Markdown files
    pub fn save_to_dir(&self, output_dir: &Path) -> Result<()> {
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(output_dir)?;

        // Save JSON
        let json_path = output_dir.join("results.json");
        let json_output = self.to_json()?;
        std::fs::write(&json_path, json_output)?;

        // Save Markdown
        let md_path = output_dir.join("results.md");
        let md_output = self.to_markdown();
        std::fs::write(&md_path, md_output)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::comparison::stats::PositionStats;

    fn create_test_agent_stats(name: &str, profit: f32) -> AgentStats {
        AgentStats {
            agent_name: name.to_string(),
            total_profit: profit,
            total_games: 100,
            wins: 50,
            losses: 40,
            breakeven: 10,
            profit_per_game: profit / 100.0,
            profit_per_100_hands: profit,
            roi_percent: profit / 100.0,
            position_stats: vec![PositionStats {
                seat_index: 0,
                games_played: 100,
                profit,
                profit_per_game: profit / 100.0,
            }],
            vpip_percent: 25.0,
            pfr_percent: 15.0,
            three_bet_percent: 5.0,
            aggression_factor: 2.0,
            cbet_percent: 60.0,
            wtsd_percent: 30.0,
            wsd_percent: 50.0,
            steal_percent: 25.0,
            aggression_frequency: 40.0,
            flop_aggression_factor: 2.5,
            turn_aggression_factor: 2.0,
            river_aggression_factor: 1.5,
            preflop_win_rate: 20.0,
            flop_win_rate: 30.0,
            turn_win_rate: 35.0,
            river_win_rate: 40.0,
        }
    }

    #[test]
    fn test_get_rankings() {
        let mut agent_stats = HashMap::new();
        agent_stats.insert(
            "Agent1".to_string(),
            create_test_agent_stats("Agent1", 100.0),
        );
        agent_stats.insert(
            "Agent2".to_string(),
            create_test_agent_stats("Agent2", -50.0),
        );
        agent_stats.insert(
            "Agent3".to_string(),
            create_test_agent_stats("Agent3", 200.0),
        );

        let result = ComparisonResult::new(
            vec![
                "Agent1".to_string(),
                "Agent2".to_string(),
                "Agent3".to_string(),
            ],
            agent_stats,
            ComparisonConfig::default(),
            10,
        );

        let rankings = result.get_rankings();
        assert_eq!(rankings.len(), 3);
        assert_eq!(rankings[0].0, "Agent3");
        assert_eq!(rankings[1].0, "Agent1");
        assert_eq!(rankings[2].0, "Agent2");
    }

    #[test]
    fn test_total_games() {
        let result = ComparisonResult::new(
            vec![],
            HashMap::new(),
            ComparisonConfig {
                num_games: 100,
                ..Default::default()
            },
            10,
        );

        assert_eq!(result.total_games(), 1000);
    }

    #[test]
    fn test_to_markdown_contains_sections() {
        let mut agent_stats = HashMap::new();
        agent_stats.insert(
            "TestAgent".to_string(),
            create_test_agent_stats("TestAgent", 100.0),
        );

        let result = ComparisonResult::new(
            vec!["TestAgent".to_string()],
            agent_stats,
            ComparisonConfig::default(),
            10,
        );

        let markdown = result.to_markdown();
        assert!(markdown.contains("# Agent Comparison Results"));
        assert!(markdown.contains("## Configuration"));
        assert!(markdown.contains("## Rankings"));
        assert!(markdown.contains("## Detailed Statistics"));
        assert!(markdown.contains("### TestAgent"));
        assert!(markdown.contains("#### Financial Performance"));
        assert!(markdown.contains("#### Playing Style"));
        assert!(markdown.contains("#### Post-Flop Stats"));
    }

    #[test]
    fn test_to_json() {
        let mut agent_stats = HashMap::new();
        agent_stats.insert(
            "TestAgent".to_string(),
            create_test_agent_stats("TestAgent", 100.0),
        );

        let result = ComparisonResult::new(
            vec!["TestAgent".to_string()],
            agent_stats,
            ComparisonConfig::default(),
            10,
        );

        let json = result.to_json().unwrap();
        assert!(json.contains("\"TestAgent\""));
        assert!(json.contains("\"total_profit\""));
        assert!(json.contains("\"roi_percent\""));

        // Verify it's valid JSON by parsing it
        let parsed: HashMap<String, AgentStats> = serde_json::from_str(&json).unwrap();
        assert!(parsed.contains_key("TestAgent"));
    }

    #[test]
    fn test_save_to_dir() {
        let mut agent_stats = HashMap::new();
        agent_stats.insert(
            "TestAgent".to_string(),
            create_test_agent_stats("TestAgent", 100.0),
        );

        let result = ComparisonResult::new(
            vec!["TestAgent".to_string()],
            agent_stats,
            ComparisonConfig::default(),
            10,
        );

        // Create a temporary directory
        let temp_dir = std::env::temp_dir().join(format!("rs_poker_test_{}", std::process::id()));

        // Save results
        result.save_to_dir(&temp_dir).unwrap();

        // Verify files were created
        assert!(temp_dir.join("results.json").exists());
        assert!(temp_dir.join("results.md").exists());

        // Verify JSON content
        let json_content = std::fs::read_to_string(temp_dir.join("results.json")).unwrap();
        assert!(json_content.contains("TestAgent"));

        // Verify markdown content
        let md_content = std::fs::read_to_string(temp_dir.join("results.md")).unwrap();
        assert!(md_content.contains("# Agent Comparison Results"));

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_markdown_roi_format() {
        let mut agent_stats = HashMap::new();
        let mut stats = create_test_agent_stats("Winner", 500.0);
        stats.roi_percent = 25.5;
        agent_stats.insert("Winner".to_string(), stats);

        let result = ComparisonResult::new(
            vec!["Winner".to_string()],
            agent_stats,
            ComparisonConfig::default(),
            10,
        );

        let markdown = result.to_markdown();
        // Verify ROI is formatted with sign and percentage
        assert!(markdown.contains("ROI: +25.5%") || markdown.contains("ROI: +25.5%)"));
    }

    #[test]
    fn test_markdown_negative_profit() {
        let mut agent_stats = HashMap::new();
        agent_stats.insert(
            "Loser".to_string(),
            create_test_agent_stats("Loser", -200.0),
        );

        let result = ComparisonResult::new(
            vec!["Loser".to_string()],
            agent_stats,
            ComparisonConfig {
                big_blind: 10.0,
                ..Default::default()
            },
            10,
        );

        let markdown = result.to_markdown();
        // Verify negative profit is shown with minus sign
        assert!(markdown.contains("-20.00 bb") || markdown.contains("-2.00 bb"));
    }

    #[test]
    fn test_accessor_methods() {
        let mut agent_stats = HashMap::new();
        agent_stats.insert(
            "Agent1".to_string(),
            create_test_agent_stats("Agent1", 100.0),
        );

        let config = ComparisonConfig {
            num_games: 50,
            players_per_table: 3,
            ..Default::default()
        };

        let result = ComparisonResult::new(vec!["Agent1".to_string()], agent_stats, config, 6);

        assert_eq!(result.agent_names(), &["Agent1".to_string()]);
        assert_eq!(result.total_permutations(), 6);
        assert_eq!(result.total_games(), 300); // 6 * 50
        assert_eq!(result.config().num_games, 50);
        assert!(result.get_agent_stats("Agent1").is_some());
        assert!(result.get_agent_stats("NonExistent").is_none());
        assert_eq!(result.all_stats().len(), 1);
    }
}
