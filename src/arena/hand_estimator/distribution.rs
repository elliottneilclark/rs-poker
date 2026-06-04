//! The hole-card distribution type and its synchronous sampler.

use crate::core::Card;

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
    let mut combos = Vec::with_capacity(1326);
    for a in 0u8..52 {
        for b in (a + 1)..52 {
            combos.push(HoleCombo::new(Card::from(a), Card::from(b)));
        }
    }
    combos
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
