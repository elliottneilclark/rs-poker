//! Agent comparison framework for evaluating poker agents
//!
//! This module provides a formalized way to compare poker agents across
//! all possible matchups and positions. It tracks detailed statistics
//! and produces comprehensive reports.
//!
//! # Example
//!
//! ```ignore
//! use rs_poker::arena::comparison::ComparisonBuilder;
//! use rs_poker::arena::agent::AgentConfig;
//!
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
//! let result = comparison.run()?;
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
//! result.save_to_dir(&output_dir)?;
//! ```

mod builder;
mod config;
mod error;
mod result;
mod runner;
mod stats;

pub use builder::ComparisonBuilder;
pub use config::ComparisonConfig;
pub use error::{ComparisonError, Result};
pub use result::ComparisonResult;
pub use runner::ArenaComparison;
pub use stats::{AgentStats, AgentStatsBuilder, PositionStats};
