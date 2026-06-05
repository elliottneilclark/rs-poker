//! The hole-card distribution type and its synchronous sampler.

use crate::core::Card;
use crate::core::CardBitSet;
use rand::RngExt;

/// A canonical, unordered pair of hole cards. `lo` is always the card with the
/// smaller `u8` index, so `HoleCombo::new(a, b) == HoleCombo::new(b, a)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HoleCombo {
    /// The card with the smaller `u8` index.
    pub lo: Card,
    /// The card with the larger `u8` index.
    pub hi: Card,
}

impl HoleCombo {
    /// Build a canonical combo from two distinct cards.
    pub fn new(a: Card, b: Card) -> Self {
        if u8::from(a) <= u8::from(b) {
            Self { lo: a, hi: b }
        } else {
            Self { lo: b, hi: a }
        }
    }
}

/// All 1326 canonical two-card combinations from a 52-card deck.
pub fn all_hole_combos() -> Vec<HoleCombo> {
    (0u8..52)
        .flat_map(|a| (a + 1..52).map(move |b| HoleCombo::new(Card::from(a), Card::from(b))))
        .collect()
}

/// Weights over hole-card combos. Stored sparsely as `(combo, weight)` pairs;
/// weights need not be normalized — [`HandDistribution::sample`] normalizes
/// over the surviving (non-dead) combos.
#[derive(Debug, Clone)]
pub struct WeightedCombos {
    /// `(combo, weight)` pairs. Zero/negative weights are ignored.
    pub weights: Vec<(HoleCombo, f32)>,
}

/// A synchronous sampler over one opponent's possible hole cards.
#[derive(Debug, Clone)]
pub enum HandDistribution {
    /// Degenerate point mass at a single known combo (used by KnownHands).
    PointMass(HoleCombo),
    /// A weighted distribution over combos (used by UniformRandom and the
    /// future ML estimator).
    Weighted(WeightedCombos),
}

impl HandDistribution {
    /// Draw one hole-card combo, avoiding every card in `dead`. Returns `None`
    /// if no legal combo remains.
    pub fn sample<R: RngExt>(&self, rng: &mut R, dead: &CardBitSet) -> Option<HoleCombo> {
        match self {
            HandDistribution::PointMass(combo) => {
                if dead.contains(combo.lo) || dead.contains(combo.hi) {
                    None
                } else {
                    Some(*combo)
                }
            }
            HandDistribution::Weighted(w) => {
                let alive = |c: &HoleCombo| !dead.contains(c.lo) && !dead.contains(c.hi);
                let total: f32 = w
                    .weights
                    .iter()
                    .filter(|(c, wt)| *wt > 0.0 && alive(c))
                    .map(|(_, wt)| *wt)
                    .sum();
                if total <= 0.0 {
                    return None;
                }
                let mut target = rng.random::<f32>() * total;
                for (combo, wt) in &w.weights {
                    if *wt > 0.0 && alive(combo) {
                        target -= *wt;
                        if target <= 0.0 {
                            return Some(*combo);
                        }
                    }
                }
                // Floating-point guard: return the last surviving combo.
                w.weights
                    .iter()
                    .rev()
                    .find(|(c, wt)| *wt > 0.0 && alive(c))
                    .map(|(c, _)| *c)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::CardBitSet;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn canonicalizes_regardless_of_order() {
        let a = Card::from(40);
        let b = Card::from(7);
        assert_eq!(HoleCombo::new(a, b), HoleCombo::new(b, a));
        assert_eq!(HoleCombo::new(a, b).lo, b);
    }

    #[test]
    fn enumerates_exactly_1326_unique_combos() {
        let combos = all_hole_combos();
        assert_eq!(combos.len(), 1326);
        let unique: std::collections::HashSet<_> = combos.iter().copied().collect();
        assert_eq!(unique.len(), 1326);
    }

    #[test]
    fn point_mass_returns_its_combo_when_alive() {
        let combo = HoleCombo::new(Card::from(3), Card::from(9));
        let dist = HandDistribution::PointMass(combo);
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(dist.sample(&mut rng, &CardBitSet::new()), Some(combo));
    }

    #[test]
    fn point_mass_returns_none_when_dead() {
        let combo = HoleCombo::new(Card::from(3), Card::from(9));
        let dist = HandDistribution::PointMass(combo);
        let mut dead = CardBitSet::new();
        dead.insert(Card::from(9));
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(dist.sample(&mut rng, &dead), None);
    }

    #[test]
    fn weighted_never_returns_a_dead_card() {
        let dist = HandDistribution::Weighted(WeightedCombos {
            weights: all_hole_combos().into_iter().map(|c| (c, 1.0)).collect(),
        });
        let mut dead = CardBitSet::new();
        for i in 0u8..50 {
            dead.insert(Card::from(i));
        }
        // Only cards 50 and 51 remain alive → the only legal combo.
        let mut rng = StdRng::seed_from_u64(7);
        let drawn = dist.sample(&mut rng, &dead).unwrap();
        assert_eq!(drawn, HoleCombo::new(Card::from(50), Card::from(51)));
    }

    #[test]
    fn weighted_returns_none_when_all_dead() {
        let dist = HandDistribution::Weighted(WeightedCombos {
            weights: all_hole_combos().into_iter().map(|c| (c, 1.0)).collect(),
        });
        let mut dead = CardBitSet::new();
        for i in 0u8..52 {
            dead.insert(Card::from(i));
        }
        let mut rng = StdRng::seed_from_u64(7);
        assert_eq!(dist.sample(&mut rng, &dead), None);
    }
}
