use crate::arena::GameState;

/// This trait defines an interface for types that can generate an iterator
/// over possible game states from a given initial game state.
///
/// The trait includes support for depth-based configuration, allowing
/// different iteration counts at different recursion depths in CFR.
pub trait GameStateIteratorGen {
    /// Configuration type for this iterator generator.
    type Config: Clone;

    /// Generate an iterator over game states from the given initial state.
    fn generate(&self, game_state: &GameState) -> impl Iterator<Item = GameState>;

    /// Create a new iterator generator for a sub-simulation at the given depth.
    ///
    /// This is used during CFR reward computation to create iterator generators
    /// for sub-agents. The depth parameter indicates how deep in the recursion
    /// we are (0 = root, 1 = first sub-simulation, etc.).
    fn new(config: &Self::Config, depth: usize) -> Self;

    /// Get this iterator generator's configuration.
    fn config(&self) -> &Self::Config;

    /// Return the number of game state iterations that `generate()` would produce.
    ///
    /// This allows callers to loop without materializing cloned game states,
    /// which is more efficient when the caller already has a reference to the
    /// original game state (e.g., in `explore_all_actions`).
    fn num_iterations(&self) -> usize;

    /// Check if exploration should occur at the given depth.
    ///
    /// Returns `false` if the depth equals or exceeds the configured maximum.
    /// This prevents exponential growth in CFR sub-simulations.
    ///
    /// Default implementation returns `true` for all depths.
    fn should_explore(&self, depth: usize) -> bool {
        let _ = depth;
        true
    }
}

/// Configuration for depth-based game state iteration.
///
/// This configuration specifies how many hands to iterate at each depth
/// of CFR recursion. The `depth_hands` vector is indexed by depth, with
/// the last value used for all depths beyond the vector length.
///
/// # Example
///
/// ```
/// use rs_poker::arena::cfr::DepthBasedIteratorGenConfig;
///
/// // depth 0: 20 hands, depth 1: 5 hands, depth 2+: 1 hand
/// let config = DepthBasedIteratorGenConfig::new(vec![20, 5, 1]);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct DepthBasedIteratorGenConfig {
    /// Number of hands per depth. Last value used for all deeper depths.
    pub depth_hands: Vec<usize>,
}

impl Default for DepthBasedIteratorGenConfig {
    fn default() -> Self {
        Self {
            depth_hands: vec![10, 2, 1],
        }
    }
}

impl DepthBasedIteratorGenConfig {
    /// Create a new config with the given depth hands.
    pub fn new(depth_hands: Vec<usize>) -> Self {
        Self { depth_hands }
    }

    /// Get the number of hands for a given depth.
    ///
    /// If the depth exceeds the length of `depth_hands`, the last value is used.
    /// If `depth_hands` is empty, returns 1.
    pub fn hands_for_depth(&self, depth: usize) -> usize {
        self.depth_hands
            .get(depth)
            .or(self.depth_hands.last())
            .copied()
            .unwrap_or(1)
    }
}

/// A depth-based game state iterator generator.
///
/// This iterator generator produces clones of the input game state,
/// with the number of clones determined by the current depth in the
/// CFR recursion tree.
///
/// # Example
///
/// ```
/// use rs_poker::arena::cfr::{DepthBasedIteratorGen, DepthBasedIteratorGenConfig, GameStateIteratorGen};
/// use rs_poker::arena::GameStateBuilder;
///
/// let config = DepthBasedIteratorGenConfig::new(vec![10, 5, 1]);
///
/// // At depth 0, generates 10 game states
/// let iter_gen = DepthBasedIteratorGen::new(config.clone(), 0);
/// let game_state = GameStateBuilder::new().num_players_with_stack(2, 100.0).blinds(10.0, 5.0).build().unwrap();
/// assert_eq!(iter_gen.generate(&game_state).count(), 10);
///
/// // At depth 1, generates 5 game states
/// let iter_gen = DepthBasedIteratorGen::new(config.clone(), 1);
/// assert_eq!(iter_gen.generate(&game_state).count(), 5);
///
/// // At depth 2+, generates 1 game state
/// let iter_gen = DepthBasedIteratorGen::new(config.clone(), 5);
/// assert_eq!(iter_gen.generate(&game_state).count(), 1);
/// ```
#[derive(Clone, Debug)]
pub struct DepthBasedIteratorGen {
    num_hands: usize,
    config: DepthBasedIteratorGenConfig,
}

impl DepthBasedIteratorGen {
    /// Create a new iterator generator with the given config at the specified depth.
    pub fn new(config: DepthBasedIteratorGenConfig, depth: usize) -> Self {
        let num_hands = config.hands_for_depth(depth);
        Self { num_hands, config }
    }

    /// Get the number of hands this generator will produce.
    pub fn num_hands(&self) -> usize {
        self.num_hands
    }
}

impl Default for DepthBasedIteratorGen {
    fn default() -> Self {
        Self::new(DepthBasedIteratorGenConfig::default(), 0)
    }
}

impl GameStateIteratorGen for DepthBasedIteratorGen {
    type Config = DepthBasedIteratorGenConfig;

    fn generate(&self, game_state: &GameState) -> impl Iterator<Item = GameState> {
        (0..self.num_hands).map(|_| game_state.clone())
    }

    fn num_iterations(&self) -> usize {
        self.num_hands
    }

    fn new(config: &Self::Config, depth: usize) -> Self {
        Self::new(config.clone(), depth)
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::GameStateBuilder;

    #[test]
    fn test_config_hands_for_depth() {
        let config = DepthBasedIteratorGenConfig::new(vec![20, 5, 1]);

        assert_eq!(config.hands_for_depth(0), 20);
        assert_eq!(config.hands_for_depth(1), 5);
        assert_eq!(config.hands_for_depth(2), 1);
        // Beyond the vector, use last value
        assert_eq!(config.hands_for_depth(3), 1);
        assert_eq!(config.hands_for_depth(100), 1);
    }

    #[test]
    fn test_config_empty_depth_hands() {
        let config = DepthBasedIteratorGenConfig::new(vec![]);
        // Empty vector returns 1
        assert_eq!(config.hands_for_depth(0), 1);
        assert_eq!(config.hands_for_depth(10), 1);
    }

    #[test]
    fn test_config_default() {
        let config = DepthBasedIteratorGenConfig::default();
        assert_eq!(config.depth_hands, vec![10, 2, 1]);
    }

    #[test]
    fn test_depth_based_iterator_gen() {
        let game_state = GameStateBuilder::new()
            .num_players_with_stack(3, 100.0)
            .blinds(10.0, 5.0)
            .build()
            .unwrap();
        let config = DepthBasedIteratorGenConfig::new(vec![3, 2, 1]);

        // Depth 0: 3 hands
        let iter_gen = DepthBasedIteratorGen::new(config.clone(), 0);
        assert_eq!(iter_gen.num_hands(), 3);
        let states: Vec<_> = iter_gen.generate(&game_state).collect();
        assert_eq!(states.len(), 3);
        for state in &states {
            assert_eq!(state, &game_state);
        }

        // Depth 1: 2 hands
        let iter_gen = DepthBasedIteratorGen::new(config.clone(), 1);
        assert_eq!(iter_gen.num_hands(), 2);
        assert_eq!(iter_gen.generate(&game_state).count(), 2);

        // Depth 2: 1 hand
        let iter_gen = DepthBasedIteratorGen::new(config.clone(), 2);
        assert_eq!(iter_gen.num_hands(), 1);
        assert_eq!(iter_gen.generate(&game_state).count(), 1);

        // Depth 10: still 1 hand (last value)
        let iter_gen = DepthBasedIteratorGen::new(config.clone(), 10);
        assert_eq!(iter_gen.num_hands(), 1);
        assert_eq!(iter_gen.generate(&game_state).count(), 1);
    }

    #[test]
    fn test_depth_based_default() {
        let iter_gen = DepthBasedIteratorGen::default();
        // Default config is [10, 2, 1], default depth is 0
        assert_eq!(iter_gen.num_hands(), 10);
    }

    #[test]
    fn test_config_accessor() {
        let config = DepthBasedIteratorGenConfig::new(vec![10, 5]);
        let iter_gen = DepthBasedIteratorGen::new(config.clone(), 0);

        let retrieved_config = iter_gen.config();
        assert_eq!(retrieved_config, &config);
    }
}
