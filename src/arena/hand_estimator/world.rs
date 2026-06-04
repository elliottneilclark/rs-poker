//! Synchronous per-wave world sampling.

use rand::Rng;

use crate::arena::GameState;
use crate::core::{Card, CardBitSet, Hand};

use super::{OpponentRanges};

/// How many times to retry a colliding draw before falling back to a uniform
/// draw from the remaining deck.
const MAX_RESAMPLE_RETRIES: usize = 16;

/// Clone `base` and re-deal every *other* live player's hole cards from
/// `ranges`, keeping the acting agent's own cards and the board fixed. Each
/// re-dealt seat's hand is rebuilt as its new hole cards plus the shared board,
/// preserving the engine invariant that a player's `Hand` contains the board.
pub fn sample_world<R: Rng>(
    ranges: &OpponentRanges,
    base: &GameState,
    rng: &mut R,
) -> GameState {
    let mut gs = base.clone();
    let board: Vec<Card> = base.board.iter().copied().collect();

    // Dead = board + every card held by a seat we are NOT re-sampling (the
    // acting seat, folded seats, and any seat with no range). This removes
    // folded/known cards from the live deck just like the real dealer.
    let mut dead = CardBitSet::new();
    for c in &board {
        dead.insert(*c);
    }
    for seat in 0..base.num_players {
        if ranges.get(seat).is_none() {
            for c in base.hands[seat].iter() {
                dead.insert(c);
            }
        }
    }

    for seat in 0..base.num_players {
        let Some(dist) = ranges.get(seat) else {
            continue;
        };

        // Try the distribution, retrying on collision, then fall back to a
        // uniform draw from the live deck.
        let mut combo = None;
        for _ in 0..MAX_RESAMPLE_RETRIES {
            if let Some(c) = dist.sample(rng, &dead) {
                combo = Some((c.lo, c.hi));
                break;
            }
        }
        let (lo, hi) = match combo {
            Some(pair) => pair,
            None => {
                match uniform_pair_from_deck(&dead, rng) { Some(pair) => pair, None => continue }
            }
        };

        dead.insert(lo);
        dead.insert(hi);

        let mut hand = Hand::new();
        hand.insert(lo);
        hand.insert(hi);
        for c in &board {
            hand.insert(*c);
        }
        gs.hands[seat] = hand;
    }

    gs
}

/// Draw two distinct cards uniformly from the cards not in `dead`.
fn uniform_pair_from_deck<R: Rng>(dead: &CardBitSet, rng: &mut R) -> Option<(Card, Card)> {
    let mut live = CardBitSet::default(); // full 52-card deck
    for i in 0u8..52 {
        let card = Card::from(i);
        if dead.contains(card) {
            live.remove(card);
        }
    }
    let first = live.sample_one(rng)?;
    live.remove(first);
    let second = live.sample_one(rng)?;
    Some((first, second))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::GameStateBuilder;
    use crate::arena::hand_estimator::{HandDistributionEstimator, KnownHandsEstimator};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn two_player_state() -> GameState {
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
    async fn known_hands_round_trips_every_hand() {
        let mut gs = two_player_state();
        gs.round_data.to_act_idx = 0;
        let ranges = KnownHandsEstimator.estimate(&gs, None).await;
        let mut rng = StdRng::seed_from_u64(42);
        let world = sample_world(&ranges, &gs, &mut rng);
        assert_eq!(world.hands[0], gs.hands[0]);
        assert_eq!(world.hands[1], gs.hands[1]);
    }

    #[tokio::test]
    async fn sampled_world_has_no_duplicate_cards() {
        use crate::arena::hand_estimator::UniformRandomEstimator;
        let gs = two_player_state();
        let ranges = UniformRandomEstimator.estimate(&gs, None).await;
        let mut rng = StdRng::seed_from_u64(99);
        let world = sample_world(&ranges, &gs, &mut rng);
        // Acting seat unchanged.
        assert_eq!(world.hands[0], gs.hands[0]);
        // Opponent's two cards must not collide with the acting seat's cards.
        let mut seen = CardBitSet::new();
        for c in world.hands[0].iter().chain(world.hands[1].iter()) {
            assert!(!seen.contains(c), "duplicate card {c:?} across seats");
            seen.insert(c);
        }
    }
}
