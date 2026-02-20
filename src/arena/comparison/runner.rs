use std::path::PathBuf;

use itertools::Itertools;
use rand::{Rng, SeedableRng, rngs::StdRng};
use tracing::event;

use crate::arena::agent::{AgentConfig, ConfigAgentBuilder};
use crate::arena::cfr::{StateStore, TraversalSet};
use crate::arena::errors::HoldemSimulationError;
use crate::arena::game_state::{GameState, RandomGameStateGenerator};
use crate::arena::historian::OpenHandHistoryHistorian;
use crate::arena::historian::StatsTrackingHistorian;
use crate::arena::{Agent, Historian, HoldemSimulationBuilder};

use super::config::ComparisonConfig;
use super::error::{ComparisonError, Result};
use super::result::ComparisonResult;
use super::stats::AgentStatsBuilder;

/// Runs agent comparisons across all permutations of seat arrangements
///
/// This struct orchestrates the comparison of multiple poker agents by:
/// 1. Generating random game states
/// 2. Running all permutations of agent seat assignments
/// 3. Collecting and aggregating statistics
/// 4. Producing comparison results
#[derive(Debug)]
pub struct ArenaComparison {
    config: ComparisonConfig,
    agents: Vec<(String, AgentConfig)>,
}

impl ArenaComparison {
    /// Create a new ArenaComparison (internal - use ComparisonBuilder instead)
    pub(crate) fn new(config: ComparisonConfig, agents: Vec<(String, AgentConfig)>) -> Self {
        Self { config, agents }
    }

    /// Get the comparison configuration
    pub fn config(&self) -> &ComparisonConfig {
        &self.config
    }

    /// Get the configured agents
    pub fn agents(&self) -> &[(String, AgentConfig)] {
        &self.agents
    }

    /// Get the number of agents
    pub fn num_agents(&self) -> usize {
        self.agents.len()
    }

    /// Calculate the total number of permutations
    pub fn total_permutations(&self) -> usize {
        (0..self.agents.len())
            .permutations(self.config.players_per_table)
            .count()
    }

    /// Calculate the total number of games to simulate
    pub fn total_games(&self) -> usize {
        self.total_permutations() * self.config.num_games
    }

    /// Run the comparison and return results
    pub fn run(&self) -> Result<ComparisonResult> {
        event!(
            tracing::Level::INFO,
            num_agents = self.agents.len(),
            num_games = self.config.num_games,
            players_per_table = self.config.players_per_table,
            "Starting agent comparison"
        );

        // Create RNG
        let mut rng = if let Some(seed) = self.config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let seed = rand::random::<u64>();
            StdRng::seed_from_u64(seed)
        };

        // Extract agent names and configs
        let agent_names: Vec<String> = self.agents.iter().map(|(name, _)| name.clone()).collect();
        let agent_configs: Vec<AgentConfig> = self.agents.iter().map(|(_, c)| c.clone()).collect();

        // Create stats builder
        let mut builder = AgentStatsBuilder::new(agent_names.clone());

        // Create game state generator (seeded if seed is provided)
        let min_stack = self.config.min_stack();
        let max_stack = self.config.max_stack();
        let mut game_state_gen = if let Some(seed) = self.config.seed {
            // Use a derived seed for game state generation to keep it separate from simulation RNG
            RandomGameStateGenerator::with_seed(
                self.config.players_per_table,
                min_stack,
                max_stack,
                self.config.big_blind,
                self.config.small_blind,
                self.config.ante,
                seed.wrapping_add(1),
            )
        } else {
            RandomGameStateGenerator::new(
                self.config.players_per_table,
                min_stack,
                max_stack,
                self.config.big_blind,
                self.config.small_blind,
                self.config.ante,
            )
        };

        // Compute OHH output path if output_dir is specified
        let ohh_output_path = self
            .config
            .output_dir
            .as_ref()
            .map(|dir| dir.join("hands.jsonl"));

        // Run simulations for each game state
        let total_permutations = self.total_permutations();
        let log_interval = 100; // Log progress every 100 game states

        for game_idx in 0..self.config.num_games {
            // Log progress every log_interval games
            if game_idx > 0 && game_idx % log_interval == 0 {
                let games_completed = game_idx * total_permutations;
                let total_games = self.config.num_games * total_permutations;
                let percent = (games_completed as f64 / total_games as f64) * 100.0;
                println!(
                    "Progress: {}/{} game states ({}/{} total games, {:.1}%)",
                    game_idx, self.config.num_games, games_completed, total_games, percent
                );
            }

            // Generate a game state
            let game_state = game_state_gen.next().ok_or_else(|| {
                ComparisonError::SimulationError("Failed to generate game state".to_string())
            })?;

            // Run all permutations with this game state
            self.run_single_game_state(
                game_state,
                &agent_configs,
                &mut builder,
                &mut rng,
                ohh_output_path.as_ref(),
            )?;
        }

        // Build the final aggregated statistics
        let agent_stats = builder.build();
        let total_permutations = self.total_permutations();

        Ok(ComparisonResult::new(
            agent_names,
            agent_stats,
            self.config.clone(),
            total_permutations,
        ))
    }

    /// Run all permutations for a single game state
    fn run_single_game_state(
        &self,
        game_state: GameState,
        agent_configs: &[AgentConfig],
        builder: &mut AgentStatsBuilder,
        rng: &mut impl Rng,
        ohh_output_path: Option<&PathBuf>,
    ) -> Result<()> {
        let players_per_table = self.config.players_per_table;
        let total_permutations = (0..agent_configs.len())
            .permutations(players_per_table)
            .count();

        event!(
            tracing::Level::DEBUG,
            total_permutations,
            players_per_table,
            "Running permutations for game state"
        );

        // Generate all permutations of size players_per_table from num_agents
        for (perm_idx, permutation) in (0..agent_configs.len())
            .permutations(players_per_table)
            .enumerate()
        {
            event!(
                tracing::Level::TRACE,
                perm_idx,
                total_permutations,
                ?permutation,
                "Starting permutation"
            );
            // Check if any agent in this permutation is CFR-based.
            // If so, create shared context so all CFR agents share the same
            // state store and traversal set.
            let has_cfr = permutation
                .iter()
                .any(|&agent_idx| agent_configs[agent_idx].is_cfr());

            let cfr_context = if has_cfr {
                let state_store = StateStore::new(game_state.clone());
                let traversal_set = TraversalSet::new(players_per_table);
                Some((state_store, traversal_set))
            } else {
                None
            };

            // Create agents for this permutation
            let boxed_agents: Vec<Box<dyn Agent>> = permutation
                .iter()
                .enumerate()
                .map(|(idx, &agent_idx)| {
                    let mut builder = ConfigAgentBuilder::new(agent_configs[agent_idx].clone())
                        .expect("Failed to create agent builder")
                        .player_idx(idx)
                        .game_state(game_state.clone());
                    if let Some((ref ss, ref ts)) = cfr_context {
                        builder = builder.cfr_context(ss.clone(), ts.clone());
                    }
                    builder.build()
                })
                .collect();

            // Create stats historian and get a clone of its storage
            let stats_historian = StatsTrackingHistorian::new_with_num_players(players_per_table);
            let stats_storage = stats_historian.get_storage();

            // Build historians list
            #[allow(unused_mut)]
            let mut historians: Vec<Box<dyn Historian>> = vec![Box::new(stats_historian)];

            // Add OpenHandHistory historian if output path is specified
            #[cfg(feature = "open-hand-history")]
            if let Some(output_path) = ohh_output_path {
                historians.push(Box::new(OpenHandHistoryHistorian::new(output_path.clone())));
            }

            // Run simulation with the cloned game state
            let mut sim_builder = HoldemSimulationBuilder::default()
                .game_state(game_state.clone())
                .agents(boxed_agents)
                .historians(historians);
            if let Some((state_store, traversal_set)) = cfr_context {
                sim_builder = sim_builder.cfr_context(state_store, traversal_set, true);
            }
            let mut sim = sim_builder.build().map_err(|e: HoldemSimulationError| {
                ComparisonError::SimulationError(e.to_string())
            })?;

            sim.run(rng);

            event!(
                tracing::Level::TRACE,
                perm_idx,
                total_permutations,
                final_round = ?sim.game_state.round,
                "Completed permutation"
            );

            // Extract statistics from the historian via the shared storage
            let stats = stats_storage.try_read().map_err(|e| {
                ComparisonError::SimulationError(format!("Failed to read stats: {}", e))
            })?;

            // Merge the stats into the builder using agent indices
            builder.merge_permutation_stats(&permutation, &stats);
        }

        Ok(())
    }

    /// Print a summary of the configuration to stdout
    pub fn print_configuration_summary(&self) {
        println!("Agent Comparison Configuration");
        println!("==============================");
        println!();
        println!("Number of Agents: {}", self.agents.len());
        println!("Players per Table: {}", self.config.players_per_table);
        println!("Number of Games: {}", self.config.num_games);
        println!();
        println!("Game Settings:");
        println!("  Big Blind: {}", self.config.big_blind);
        println!("  Small Blind: {}", self.config.small_blind);
        if self.config.ante > 0.0 {
            println!("  Ante: {}", self.config.ante);
        }
        println!(
            "  Stack Range: {}-{} BB",
            self.config.min_stack_bb, self.config.max_stack_bb
        );
        if let Some(seed) = self.config.seed {
            println!("  Random Seed: {}", seed);
        }
        if let Some(ref output_dir) = self.config.output_dir {
            println!("  Output Directory: {}", output_dir.display());
        }
        println!();
        println!("Loaded Agents:");
        for (name, _) in &self.agents {
            println!("  - {}", name);
        }
        println!();

        // Calculate and display permutation count
        let total_permutations = self.total_permutations();
        let total_games = self.total_games();

        println!("Simulation Scale:");
        println!("  Total permutations: {}", total_permutations);
        println!(
            "  Total games to simulate: {} × {} = {}",
            total_permutations, self.config.num_games, total_games
        );
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::comparison::ComparisonBuilder;

    #[test]
    fn test_arena_comparison_total_permutations() {
        let comparison = ComparisonBuilder::new()
            .players_per_table(2)
            .add_agent_config(AgentConfig::Folding {
                name: Some("A".to_string()),
            })
            .add_agent_config(AgentConfig::Folding {
                name: Some("B".to_string()),
            })
            .add_agent_config(AgentConfig::Folding {
                name: Some("C".to_string()),
            })
            .build()
            .unwrap();

        // P(3, 2) = 3 * 2 = 6
        assert_eq!(comparison.total_permutations(), 6);
    }

    #[test]
    fn test_arena_comparison_total_games() {
        let comparison = ComparisonBuilder::new()
            .num_games(100)
            .players_per_table(2)
            .add_agent_config(AgentConfig::Folding {
                name: Some("A".to_string()),
            })
            .add_agent_config(AgentConfig::Folding {
                name: Some("B".to_string()),
            })
            .build()
            .unwrap();

        // P(2, 2) = 2, so total = 2 * 100 = 200
        assert_eq!(comparison.total_games(), 200);
    }

    #[test]
    fn test_arena_comparison_run() {
        let comparison = ComparisonBuilder::new()
            .num_games(10)
            .players_per_table(2)
            .seed(42)
            .add_agent_config(AgentConfig::Folding {
                name: Some("Folder".to_string()),
            })
            .add_agent_config(AgentConfig::Calling {
                name: Some("Caller".to_string()),
            })
            .build()
            .unwrap();

        let result = comparison.run().unwrap();

        // Check we got results for both agents
        assert_eq!(result.agent_names().len(), 2);
        assert!(result.get_agent_stats("Folder").is_some());
        assert!(result.get_agent_stats("Caller").is_some());

        // Check total permutations
        assert_eq!(result.total_permutations(), 2);

        // Check total games
        assert_eq!(result.total_games(), 20);
    }

    #[test]
    fn test_arena_comparison_run_with_three_agents() {
        let comparison = ComparisonBuilder::new()
            .num_games(5)
            .players_per_table(2)
            .seed(123)
            .add_agent_config(AgentConfig::Folding {
                name: Some("A".to_string()),
            })
            .add_agent_config(AgentConfig::Calling {
                name: Some("B".to_string()),
            })
            .add_agent_config(AgentConfig::AllIn {
                name: Some("C".to_string()),
            })
            .build()
            .unwrap();

        let result = comparison.run().unwrap();

        // P(3, 2) = 6 permutations
        assert_eq!(result.total_permutations(), 6);

        // Total games = 6 * 5 = 30
        assert_eq!(result.total_games(), 30);

        // All three agents should have stats
        assert!(result.get_agent_stats("A").is_some());
        assert!(result.get_agent_stats("B").is_some());
        assert!(result.get_agent_stats("C").is_some());
    }

    #[test]
    fn test_arena_comparison_roi_tracking() {
        // Test that ROI is properly calculated based on investment
        let comparison = ComparisonBuilder::new()
            .num_games(20)
            .players_per_table(2)
            .seed(42)
            .add_agent_config(AgentConfig::Calling {
                name: Some("Caller".to_string()),
            })
            .add_agent_config(AgentConfig::Folding {
                name: Some("Folder".to_string()),
            })
            .build()
            .unwrap();

        let result = comparison.run().unwrap();

        let caller_stats = result.get_agent_stats("Caller").unwrap();
        let folder_stats = result.get_agent_stats("Folder").unwrap();

        // Caller should have some investment (they call bets)
        // We can't assert exact values due to randomness, but:
        // - If caller has positive profit and positive investment, ROI should be profit/investment * 100
        // - If caller has negative profit, ROI should be negative

        // Folder has 0% VPIP, so no voluntary investment
        // ROI should be 0 since total_invested is 0
        assert_eq!(
            folder_stats.roi_percent, 0.0,
            "Folder should have 0% ROI (no voluntary investment)"
        );

        // Caller invests money, so they should have non-zero ROI
        // (unless by chance they break exactly even on investment, which is unlikely)
        // We just verify that the stats exist and are reasonable
        assert!(
            caller_stats.total_games > 0,
            "Caller should have played games"
        );
    }

    #[test]
    fn test_arena_comparison_position_stats() {
        // Test that position stats are tracked for each agent
        let comparison = ComparisonBuilder::new()
            .num_games(10)
            .players_per_table(2)
            .seed(999)
            .add_agent_config(AgentConfig::Calling {
                name: Some("Player1".to_string()),
            })
            .add_agent_config(AgentConfig::Folding {
                name: Some("Player2".to_string()),
            })
            .build()
            .unwrap();

        let result = comparison.run().unwrap();

        let player1 = result.get_agent_stats("Player1").unwrap();

        // With 2 agents and 2 players_per_table, each agent plays in both positions
        // Each agent should have played 10 games in each of 2 positions
        assert_eq!(
            player1.position_stats.len(),
            2,
            "Should have stats for both seat positions"
        );

        // Total games across positions should equal total games
        let total_position_games: usize =
            player1.position_stats.iter().map(|p| p.games_played).sum();
        assert_eq!(
            total_position_games, player1.total_games,
            "Position games should sum to total games"
        );
    }

    #[test]
    fn test_arena_comparison_profit_zero_sum() {
        // In a heads-up game, total profit should be approximately zero
        // (one player's gain is another's loss)
        let comparison = ComparisonBuilder::new()
            .num_games(50)
            .players_per_table(2)
            .seed(12345)
            .add_agent_config(AgentConfig::Calling {
                name: Some("A".to_string()),
            })
            .add_agent_config(AgentConfig::Calling {
                name: Some("B".to_string()),
            })
            .build()
            .unwrap();

        let result = comparison.run().unwrap();

        let a_stats = result.get_agent_stats("A").unwrap();
        let b_stats = result.get_agent_stats("B").unwrap();

        // Total profit should sum to approximately zero (within floating point tolerance)
        let total_profit = a_stats.total_profit + b_stats.total_profit;
        assert!(
            total_profit.abs() < 1.0,
            "Total profit should be approximately zero, got {}",
            total_profit
        );
    }

    #[test]
    fn test_arena_comparison_deterministic_with_seed() {
        // Running with the same seed should produce identical results
        let build_comparison = || {
            ComparisonBuilder::new()
                .num_games(10)
                .players_per_table(2)
                .seed(42)
                .add_agent_config(AgentConfig::Folding {
                    name: Some("A".to_string()),
                })
                .add_agent_config(AgentConfig::Calling {
                    name: Some("B".to_string()),
                })
                .build()
                .unwrap()
        };

        let result1 = build_comparison().run().unwrap();
        let result2 = build_comparison().run().unwrap();

        let a1 = result1.get_agent_stats("A").unwrap();
        let a2 = result2.get_agent_stats("A").unwrap();

        assert_eq!(
            a1.total_profit, a2.total_profit,
            "Same seed should produce same profit"
        );
        assert_eq!(a1.wins, a2.wins, "Same seed should produce same wins");
        assert_eq!(a1.losses, a2.losses, "Same seed should produce same losses");
    }

    /// Verifies that num_agents() returns the actual number of agents added.
    #[test]
    fn test_arena_comparison_num_agents() {
        let comparison = ComparisonBuilder::new()
            .players_per_table(2)
            .add_agent_config(AgentConfig::Folding {
                name: Some("A".to_string()),
            })
            .add_agent_config(AgentConfig::Calling {
                name: Some("B".to_string()),
            })
            .add_agent_config(AgentConfig::AllIn {
                name: Some("C".to_string()),
            })
            .build()
            .unwrap();

        assert_eq!(comparison.num_agents(), 3);
        assert_ne!(comparison.num_agents(), 0);
    }

    /// Verifies that agents() returns the agents with their correct names.
    #[test]
    fn test_arena_comparison_agents() {
        let comparison = ComparisonBuilder::new()
            .players_per_table(2)
            .add_agent_config(AgentConfig::Folding {
                name: Some("Agent1".to_string()),
            })
            .add_agent_config(AgentConfig::Calling {
                name: Some("Agent2".to_string()),
            })
            .build()
            .unwrap();

        let agents = comparison.agents();
        assert_eq!(agents.len(), 2);
        assert_eq!(agents[0].0, "Agent1");
        assert_eq!(agents[1].0, "Agent2");
    }

    /// Verifies that print_configuration_summary executes without panicking
    /// and uses the correct configuration values.
    #[test]
    fn test_arena_comparison_print_configuration_summary() {
        let comparison = ComparisonBuilder::new()
            .players_per_table(2)
            .add_agent_config(AgentConfig::Folding {
                name: Some("Tester".to_string()),
            })
            .add_agent_config(AgentConfig::Calling {
                name: Some("Caller".to_string()),
            })
            .build()
            .unwrap();

        // Capture stdout - we can't easily capture println!, but we can verify
        // the method doesn't panic and check the config values used
        comparison.print_configuration_summary();

        // If we get here without panic, the method executed.
        // We verify the data it would print:
        assert_eq!(comparison.agents().len(), 2);
        assert_eq!(comparison.config().players_per_table, 2);
    }

    /// Verifies that ante configuration is properly handled (positive ante vs zero ante).
    #[test]
    fn test_arena_comparison_ante_condition() {
        // When ante > 0.0, the ante line should be printed
        // When ante == 0.0, the ante line should NOT be printed
        let comparison_with_ante = ComparisonBuilder::new()
            .players_per_table(2)
            .ante(1.0)
            .add_agent_config(AgentConfig::Folding {
                name: Some("A".to_string()),
            })
            .add_agent_config(AgentConfig::Calling {
                name: Some("B".to_string()),
            })
            .build()
            .unwrap();

        let comparison_without_ante = ComparisonBuilder::new()
            .players_per_table(2)
            .ante(0.0)
            .add_agent_config(AgentConfig::Folding {
                name: Some("A".to_string()),
            })
            .add_agent_config(AgentConfig::Calling {
                name: Some("B".to_string()),
            })
            .build()
            .unwrap();

        // Verify ante values are correctly stored
        assert!(comparison_with_ante.config().ante > 0.0);
        assert_eq!(comparison_without_ante.config().ante, 0.0);

        // Print both - they should not panic and use different code paths
        comparison_with_ante.print_configuration_summary();
        comparison_without_ante.print_configuration_summary();
    }
}
