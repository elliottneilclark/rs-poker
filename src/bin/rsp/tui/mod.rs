//! TUI dashboard for live poker simulation monitoring.
//!
//! ## Data flow
//!
//! ```text
//! generation thread ──► bounded sync_channel(1024) ──► TUI event loop
//! ```
//!
//! The generation thread produces `GameResult` values containing flat `SeatStats`
//! (all scalars, no heap allocations). The bounded channel applies natural
//! backpressure — the generation thread blocks when the TUI falls behind.
//!
//! ## Why `SeatStats` is flat
//!
//! Previously, each `GameResult` carried a full `StatsStorage` (40+ `Vec` fields).
//! When those heap-allocated Vecs were freed on the TUI thread (a different thread
//! than allocated them), glibc's per-thread malloc arenas could not efficiently
//! reclaim the memory, causing steady RSS growth until OOM. `SeatStats` is `Copy`
//! — zero cross-thread heap alloc/free pairs.
//!
//! ## Key components
//!
//! - **`HandStore`**: Disk-backed with in-memory byte-offset index (8 bytes/game).
//!   Hand histories are loaded on demand, not held in memory.
//! - **`FilteredGameLog`**: Virtual windowed view over the game log. Only loads
//!   the rows visible in the current scroll viewport from disk.
//! - **`TuiState`**: Bounded accumulators — profit history capped at 10K points
//!   per agent, agent stats keyed by unique agent name.

pub mod app;
pub mod effects;
pub mod event;
pub mod filtered_log;
pub mod hand_store;
pub mod screens;
pub mod state;
pub mod terminal;
pub mod theme;
pub mod widgets;

use clap::Args;

/// TUI display flags for controlling terminal UI behavior
#[derive(Args, Debug, Clone)]
pub struct TuiFlags {
    /// Force TUI dashboard display
    #[arg(long = "tui", global = true)]
    pub force_tui: bool,

    /// Disable TUI dashboard (plain log output)
    #[arg(long = "no-tui", global = true)]
    pub no_tui: bool,
}

impl TuiFlags {
    /// Determine whether to use the TUI based on flags, env, and TTY detection.
    ///
    /// Priority: --no-tui > --tui > RSP_NO_TUI env > TTY auto-detect
    pub fn should_use_tui(&self) -> bool {
        let env_no_tui = std::env::var("RSP_NO_TUI").is_ok();
        let is_tty = std::io::IsTerminal::is_terminal(&std::io::stdout());
        resolve_tui(self.no_tui, self.force_tui, env_no_tui, is_tty)
    }
}

/// Pure logic for TUI resolution, testable without env mutation.
fn resolve_tui(no_tui: bool, force_tui: bool, env_no_tui: bool, is_tty: bool) -> bool {
    if no_tui {
        return false;
    }
    if force_tui {
        return true;
    }
    if env_no_tui {
        return false;
    }
    is_tty
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_tui_flag_overrides_all() {
        assert!(!resolve_tui(true, true, false, true));
    }

    #[test]
    fn test_force_tui_flag() {
        assert!(resolve_tui(false, true, false, false));
    }

    #[test]
    fn test_no_tui_flag() {
        assert!(!resolve_tui(true, false, false, true));
    }

    #[test]
    fn test_env_var_disables_tui() {
        assert!(!resolve_tui(false, false, true, true));
    }

    #[test]
    fn test_tty_auto_detect() {
        assert!(resolve_tui(false, false, false, true));
        assert!(!resolve_tui(false, false, false, false));
    }
}
