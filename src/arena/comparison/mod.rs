//! Agent comparison framework for evaluating poker agents
//!
//! This module provides a formalized way to compare poker agents across
//! all possible matchups and positions. It tracks detailed statistics
//! and produces comprehensive reports.
//!
//! # Example
//!
//! ```no_run
//! use std::path::PathBuf;
//!
//! use rs_poker::arena::comparison::{ComparisonBuilder, ComparisonError};
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), ComparisonError> {
//! // Build a comparison with agents from config files
//! let comparison = ComparisonBuilder::new()
//!     .num_games(1000)
//!     .players_per_table(3)
//!     .big_blind(10.0)
//!     .small_blind(5.0)
//!     .load_agents_from_dir("./agents/")?
//!     .seed(42)
//!     .build()?;
//!
//! // Run the comparison
//! let result = comparison.run().await?;
//!
//! // Get rankings
//! for (rank, (name, stats)) in result.get_rankings().iter().enumerate() {
//!     println!("{}. {} - {:.2} bb/game", rank + 1, name, stats.profit_per_game);
//! }
//!
//! // Generate markdown report
//! let markdown = result.to_markdown();
//! println!("{}", markdown);
//!
//! // Save results to files
//! let output_dir = PathBuf::from("./results");
//! result.save_to_dir(&output_dir)?;
//! # Ok(())
//! # }
//! ```

mod builder;
mod config;
mod error;
mod result;
mod runner;
mod stats;

pub use builder::ComparisonBuilder;
pub use config::ComparisonConfig;
pub use error::{ComparisonConfigError, ComparisonError, Result};
pub use result::ComparisonResult;
pub use runner::{ArenaComparison, PermutationResult};
pub use stats::{AgentStats, AgentStatsBuilder, PositionStats};
