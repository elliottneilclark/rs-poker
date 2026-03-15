use super::card_bit_set::pdep;
use super::{CardBitSet, FlatDeck};

/// Iterates over all k-element subsets of a [`CardBitSet`].
///
/// Uses [Gosper's hack](https://en.wikipedia.org/wiki/Combinatorial_number_system#Applications)
/// to enumerate subsets in lexicographic order over logical bit indices,
/// then maps each subset to physical card positions via PDEP.
///
/// Zero allocation — the entire state is three `u64`-sized fields.
#[derive(Debug)]
pub struct CardIter {
    /// The card pool as a raw bitmask.
    mask: u64,
    /// Current k-subset of {0..n-1} as a bitmask. Advanced by Gosper's hack.
    combo: u64,
    /// One past the maximum valid combo value: `1 << n`.
    limit: u64,
    /// Whether the iterator is exhausted.
    done: bool,
}

impl CardIter {
    /// Create a new iterator over all `k`-element subsets of `cards`.
    ///
    /// Yields [`CardBitSet`] items, each containing exactly `k` cards
    /// chosen from `cards`. Visits exactly C(n, k) subsets where
    /// n = `cards.count()`.
    ///
    /// # Edge cases
    /// - `k == 0`: yields one empty `CardBitSet`
    /// - `k > n`: yields nothing
    pub fn new(cards: impl Into<CardBitSet>, k: usize) -> Self {
        let mask = cards.into().to_u64();
        let n = mask.count_ones() as usize;
        let k_u32 = k as u32;

        if k == 0 {
            return Self {
                mask,
                combo: 0,
                limit: 0,
                done: false,
            };
        }

        if k > n {
            return Self {
                mask,
                combo: 0,
                limit: 0,
                done: true,
            };
        }

        Self {
            mask,
            combo: (1u64 << k_u32) - 1,
            limit: 1u64 << (n as u32),
            done: false,
        }
    }
}

impl Iterator for CardIter {
    type Item = CardBitSet;

    fn next(&mut self) -> Option<CardBitSet> {
        if self.done {
            return None;
        }

        let combo = self.combo;

        // `combo` is a k-subset of the *logical* indices {0..n-1}, where n is
        // the popcount of `self.mask`. PDEP scatters these logical positions
        // into the physical bit positions of the actual cards in `mask`.
        //
        // Example: mask = cards at physical bits {1, 5, 8, 12, 20}.
        // combo = 0b01010 selects logical indices 1 and 3, and PDEP maps
        // those to physical bits 5 and 12, producing the correct CardBitSet.
        let result = CardBitSet::from_u64(pdep(combo, self.mask));

        if self.limit == 0 {
            self.done = true;
            return Some(result);
        }

        if combo == 0 {
            self.done = true;
            return Some(result);
        }

        // Gosper's hack: compute the next higher integer with the same
        // popcount as `combo`. This visits all C(n,k) subsets in order.
        //
        // Decompose combo as: ...prefix 1^m 0^n  (m ones then n trailing zeros)
        // We want:            ...prefix' 1 0^(n+1) 1^(m-1)
        // i.e., move the highest bit of the lowest run up by one position,
        // then pack the remaining m-1 bits into the bottom.
        //
        // Step 1: Smear the lowest run of ones downward through the trailing
        // zeros, producing a solid block of (m+n) ones at the bottom.
        let t = combo | (combo - 1);

        // Step 2: Adding 1 carries through that solid block, clearing all
        // (m+n) trailing ones and setting the bit just above — this is the
        // "promoted" bit that moved up one position.
        //
        // Step 3: `(!t) & (t+1)` isolates the carry-out bit (position m+n),
        // so subtracting 1 gives a mask of (m+n) ones. Shifting right by
        // (n+1) produces exactly (m-1) ones in the lowest positions — the
        // remaining bits that get packed to the bottom.
        //
        // OR-ing the two parts together yields the next k-subset.
        let next = (t + 1) | ((((!t) & (t + 1)) - 1) >> (combo.trailing_zeros() + 1));

        if next >= self.limit {
            self.done = true;
        } else {
            self.combo = next;
        }

        Some(result)
    }
}

impl IntoIterator for &FlatDeck {
    type Item = CardBitSet;
    type IntoIter = CardIter;

    fn into_iter(self) -> CardIter {
        CardIter::new(self, 5)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::{Card, Deck, FlatDeck, Suit, Value};

    use super::*;

    #[test]
    fn test_iter_zero() {
        let mut cbs = CardBitSet::new();
        cbs.insert(Card::new(Value::Two, Suit::Spade));
        cbs.insert(Card::new(Value::Three, Suit::Spade));
        let combos: Vec<_> = CardIter::new(cbs, 0).collect();
        assert_eq!(1, combos.len());
        assert_eq!(0, combos[0].count());
    }

    #[test]
    fn test_iter_k_greater_than_n() {
        let mut cbs = CardBitSet::new();
        cbs.insert(Card::new(Value::Two, Suit::Spade));
        assert_eq!(0, CardIter::new(cbs, 2).count());
    }

    #[test]
    fn test_iter_one() {
        let mut cbs = CardBitSet::new();
        cbs.insert(Card::new(Value::Two, Suit::Spade));

        for cards in CardIter::new(cbs, 1) {
            assert_eq!(1, cards.count());
        }
        assert_eq!(1, CardIter::new(cbs, 1).count());
    }

    #[test]
    fn test_iter_two() {
        let mut cbs = CardBitSet::new();
        cbs.insert(Card::new(Value::Two, Suit::Spade));
        cbs.insert(Card::new(Value::Three, Suit::Spade));
        cbs.insert(Card::new(Value::Four, Suit::Spade));

        assert_eq!(3, CardIter::new(cbs, 2).count());

        for cards in CardIter::new(cbs, 2) {
            assert_eq!(2, cards.count());
        }
    }

    #[test]
    fn test_iter_deck() {
        let d: FlatDeck = Deck::default().into();
        assert_eq!(2_598_960, d.into_iter().count());
    }

    #[test]
    fn test_iter_returns_cardbitset() {
        let mut cbs = CardBitSet::new();
        cbs.insert(Card::new(Value::Two, Suit::Spade));
        cbs.insert(Card::new(Value::Three, Suit::Spade));
        cbs.insert(Card::new(Value::Four, Suit::Spade));

        for combo in CardIter::new(cbs, 2) {
            assert_eq!(2, combo.count());

            let mut combined = combo;
            combined.insert(Card::new(Value::Five, Suit::Spade));
            assert_eq!(3, combined.count());
        }
    }

    #[test]
    fn test_iter_contains_correct_cards() {
        let mut cbs = CardBitSet::new();
        let card1 = Card::new(Value::Ace, Suit::Heart);
        let card2 = Card::new(Value::King, Suit::Heart);
        cbs.insert(card1);
        cbs.insert(card2);

        let combos: Vec<_> = CardIter::new(cbs, 2).collect();
        assert_eq!(1, combos.len());
        assert!(combos[0].contains(card1));
        assert!(combos[0].contains(card2));
        assert_eq!(2, combos[0].count());
    }

    #[test]
    fn test_iter_bitwise_or_combinations() {
        let mut cbs = CardBitSet::new();
        cbs.insert(Card::new(Value::Two, Suit::Spade));
        cbs.insert(Card::new(Value::Three, Suit::Spade));
        cbs.insert(Card::new(Value::Four, Suit::Spade));

        let base_card = Card::new(Value::Five, Suit::Heart);
        let mut base_set = CardBitSet::new();
        base_set.insert(base_card);

        for combo in CardIter::new(cbs, 2) {
            let combined = base_set | combo;
            assert_eq!(3, combined.count());
            assert!(combined.contains(base_card));
        }
    }

    /// Verify exact combination counts match C(n,k) for various n,k
    #[test]
    fn test_combination_counts() {
        let mut cbs = CardBitSet::new();
        for i in 0..7u8 {
            cbs.insert(Card::from(i));
        }
        assert_eq!(1, CardIter::new(cbs, 0).count());
        assert_eq!(7, CardIter::new(cbs, 1).count());
        assert_eq!(21, CardIter::new(cbs, 2).count());
        assert_eq!(35, CardIter::new(cbs, 3).count());
        assert_eq!(21, CardIter::new(cbs, 5).count());
        assert_eq!(1, CardIter::new(cbs, 7).count());
        assert_eq!(0, CardIter::new(cbs, 8).count());
    }

    /// Every subset produced should be a proper subset of the input
    #[test]
    fn test_all_subsets_are_subsets_of_input() {
        let mut cbs = CardBitSet::new();
        for i in 0..6u8 {
            cbs.insert(Card::from(i * 7));
        }

        for combo in CardIter::new(cbs, 3) {
            assert_eq!(3, combo.count());
            for card in combo {
                assert!(cbs.contains(card), "Card {card:?} not in input set");
            }
        }
    }

    /// No duplicate subsets
    #[test]
    fn test_no_duplicate_subsets() {
        let mut cbs = CardBitSet::new();
        for i in 0..5u8 {
            cbs.insert(Card::from(i));
        }

        let combos: Vec<CardBitSet> = CardIter::new(cbs, 3).collect();
        for (i, a) in combos.iter().enumerate() {
            for b in &combos[i + 1..] {
                assert_ne!(a, b, "Duplicate subset found");
            }
        }
    }
}
