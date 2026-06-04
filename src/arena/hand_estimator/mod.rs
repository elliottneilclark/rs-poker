//! Opponent hand-distribution estimation for CFR.
//!
//! A `HandDistributionEstimator` guesses, from the perspective of the seat
//! that is about to act, the hole-card distribution of every *other* live
//! player. The CFR engine calls it once per `act()` and then samples concrete
//! worlds from the result (see `world::sample_world`). This lets the solver
//! play against ranges instead of the pinned, fully-known hands.

use async_trait::async_trait;

use crate::arena::GameState;
use crate::arena::action::Action;

mod distribution;

pub use distribution::{HandDistribution, HoleCombo, WeightedCombos, all_hole_combos};

/// A read-only view over the actions recorded so far in the current hand,
/// passed to estimators that need history. Wraps the raw `Action` stream; kept
/// as a struct so fields can be added later without changing the trait.
pub struct GameLog<'a> {
    /// The actions of the current hand, in order.
    pub actions: &'a [Action],
}

/// One hole-card distribution per seat, from the acting agent's perspective.
/// `None` for the acting seat, folded seats, and any seat with no hidden cards.
#[derive(Debug, Clone, Default)]
pub struct OpponentRanges {
    per_seat: Vec<Option<HandDistribution>>,
}

impl OpponentRanges {
    /// Build ranges from a per-seat vector indexed by seat number.
    pub fn new(per_seat: Vec<Option<HandDistribution>>) -> Self {
        Self { per_seat }
    }

    /// The distribution for `seat`, or `None` if that seat is not re-sampled.
    pub fn get(&self, seat: usize) -> Option<&HandDistribution> {
        self.per_seat.get(seat).and_then(|o| o.as_ref())
    }
}

/// Estimates opponents' hole-card distributions. The ML model implements this.
#[async_trait]
pub trait HandDistributionEstimator: Send + Sync {
    /// Estimate the hole-card distribution of every *other* live player from
    /// the perspective of `perspective_idx` (the seat about to act). `history`
    /// is `None` unless the agent assembled a log for a `needs_history()`
    /// estimator.
    async fn estimate(
        &self,
        game_state: &GameState,
        history: Option<&GameLog<'_>>,
    ) -> OpponentRanges;

    /// Whether this estimator needs the game log. Default `false`, so the CFR
    /// agent attaches no historian and passes `history: None`.
    fn needs_history(&self) -> bool {
        false
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opponent_ranges_get_is_bounds_safe() {
        let ranges = OpponentRanges::new(vec![None, None]);
        assert!(ranges.get(0).is_none());
        assert!(ranges.get(99).is_none());
    }

    #[test]
    fn estimator_is_object_safe() {
        // Compile-time proof that the trait can be boxed as a trait object.
        fn _assert(_: std::sync::Arc<dyn HandDistributionEstimator>) {}
    }
}
