/// Depth-based CFR recursion schedule.
///
/// `depth_hands[D]` is the number of recursive sub-simulations the CFR agent
/// at depth `D` runs while computing rewards. A depth whose entry is missing
/// or `< 1` means that agent uses the cheap fast-forward reward path instead
/// of spawning recursive sub-simulations.
///
/// # Example
///
/// ```
/// use rs_poker::arena::cfr::CfrDepthConfig;
///
/// // depth 0: 20 hands, depth 1: 5 hands, depth 2: 1 hand, depth 3+: fast-forward
/// let config = CfrDepthConfig::new(vec![20, 5, 1]);
/// assert_eq!(config.hands_for_depth(0), 20);
/// assert_eq!(config.hands_for_depth(3), 0);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct CfrDepthConfig {
    pub depth_hands: Vec<usize>,
}

impl Default for CfrDepthConfig {
    fn default() -> Self {
        Self {
            depth_hands: vec![10, 2, 1],
        }
    }
}

impl CfrDepthConfig {
    pub fn new(depth_hands: Vec<usize>) -> Self {
        Self { depth_hands }
    }

    /// Hands to run at the given depth. Returns `0` when the depth is beyond
    /// the schedule — the caller should treat `0` as "use fast-forward".
    pub fn hands_for_depth(&self, depth: usize) -> usize {
        self.depth_hands.get(depth).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hands_for_depth_returns_raw_entry_or_zero() {
        let config = CfrDepthConfig::new(vec![20, 5, 1]);
        assert_eq!(config.hands_for_depth(0), 20);
        assert_eq!(config.hands_for_depth(1), 5);
        assert_eq!(config.hands_for_depth(2), 1);
        assert_eq!(config.hands_for_depth(3), 0);
        assert_eq!(config.hands_for_depth(100), 0);
    }

    #[test]
    fn hands_for_depth_empty_is_zero() {
        let config = CfrDepthConfig::new(vec![]);
        assert_eq!(config.hands_for_depth(0), 0);
        assert_eq!(config.hands_for_depth(10), 0);
    }
}
