mod basic;
mod configurable;
mod preflop_chart;
mod simple;

use crate::arena::{GameState, action::AgentAction};

use super::{CFRState, TraversalState};

pub use basic::BasicCFRActionGenerator;
pub use configurable::{ConfigurableActionConfig, ConfigurableActionGenerator, RoundActionConfig};
pub use preflop_chart::{
    PreflopChartActionConfig, PreflopChartActionGenerator, PreflopChartConfig,
};
pub use simple::SimpleActionGenerator;

/// Trait for generating possible actions in CFR.
///
/// ActionGenerators are responsible for:
/// 1. Generating all possible actions for a given game state
/// 2. Holding references to CFR state and traversal state
///
/// Action-to-index mapping and action selection are now handled by
/// `ActionIndexMapper` and `ActionPicker` respectively, which use
/// a fixed 52-action space for consistent tree traversal.
pub trait ActionGenerator {
    /// The configuration type for this action generator.
    /// Use `()` for generators that don't need configuration.
    type Config: Clone;

    /// Create a new action generator
    ///
    /// This is used by the Agent to create identical
    /// action generators for the historians it uses.
    fn new(cfr_state: CFRState, traversal_state: TraversalState, config: Self::Config) -> Self;

    /// Get a reference to the configuration
    fn config(&self) -> &Self::Config;

    /// Get a reference to the CFR state
    fn cfr_state(&self) -> &CFRState;

    /// Get a reference to the traversal state
    fn traversal_state(&self) -> &TraversalState;

    /// Generate all possible actions for the current game state.
    ///
    /// This returns a vector of valid actions that can be taken.
    /// The actions will be mapped to indices using `ActionIndexMapper`
    /// for tree traversal.
    fn gen_possible_actions(&self, game_state: &GameState) -> Vec<AgentAction>;
}
