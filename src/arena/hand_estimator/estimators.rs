//! Built-in estimators: known hands (default) and uniform-random.

use async_trait::async_trait;

use crate::arena::GameState;
use crate::core::{Card, CardBitSet};

use super::{
    GameLog, HandDistribution, HandDistributionEstimator, HoleCombo, OpponentRanges,
};

/// True when `seat` still holds hidden cards in the hand (active or all-in).
fn seat_in_hand(game_state: &GameState, seat: usize) -> bool {
    game_state.player_active.get(seat) || game_state.player_all_in.get(seat)
}

/// The two hole cards for `seat`: the seat's hand minus the shared board.
fn hole_cards(game_state: &GameState, seat: usize, board: &CardBitSet) -> Option<HoleCombo> {
    let hole: Vec<Card> = game_state.hands[seat]
        .iter()
        .filter(|c| !board.contains(*c))
        .collect();
    if hole.len() == 2 {
        Some(HoleCombo::new(hole[0], hole[1]))
    } else {
        None
    }
}

/// Estimator that returns each opponent's true hand as a point mass. This is
/// the default; with it, `sample_world` reproduces the real hands exactly, so
/// CFR behavior is identical to the pre-estimator engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct KnownHandsEstimator;

#[async_trait]
impl HandDistributionEstimator for KnownHandsEstimator {
    async fn estimate(
        &self,
        game_state: &GameState,
        _history: Option<&GameLog<'_>>,
    ) -> OpponentRanges {
        let perspective_idx = game_state.to_act_idx();
        let board: CardBitSet = game_state.board.iter().copied().collect();
        let mut per_seat = Vec::with_capacity(game_state.num_players);
        for seat in 0..game_state.num_players {
            if seat == perspective_idx || !seat_in_hand(game_state, seat) {
                per_seat.push(None);
                continue;
            }
            per_seat.push(hole_cards(game_state, seat, &board).map(HandDistribution::PointMass));
        }
        OpponentRanges::new(per_seat)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::GameStateBuilder;
    use crate::core::Hand;

    fn two_player_state() -> GameState {
        // Two players, each dealt distinct hole cards, no board.
        let mut gs = GameStateBuilder::default()
            .num_players_with_stack(2, 100.0)
            .big_blind(2.0)
            .build()
            .unwrap();
        gs.hands[0] = Hand::new_with_cards(vec![Card::from(0), Card::from(1)]);
        gs.hands[1] = Hand::new_with_cards(vec![Card::from(2), Card::from(3)]);
        gs.round_data.to_act_idx = 0;
        gs
    }

    #[tokio::test]
    async fn returns_point_mass_for_opponent_only() {
        let mut gs = two_player_state();
        gs.round_data.to_act_idx = 0;
        let ranges = KnownHandsEstimator.estimate(&gs, None).await;
        assert!(ranges.get(0).is_none(), "acting seat must be None");
        match ranges.get(1) {
            Some(HandDistribution::PointMass(c)) => {
                assert_eq!(*c, HoleCombo::new(Card::from(2), Card::from(3)));
            }
            other => panic!("expected point mass, got {other:?}"),
        }
    }
}
