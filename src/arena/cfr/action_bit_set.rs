/// A bit set for tracking action indices (0-51).
///
/// This is optimized for the CFR action space which has 52 possible action indices:
/// - Index 0: Fold
/// - Index 1: Call/Check
/// - Indices 2-50: Raises (logarithmic distribution)
/// - Index 51: All-in
///
/// Using a `u64` allows O(1) insert and contains operations with no heap allocation,
/// which is much faster than `HashSet<usize>` for this use case.
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub struct ActionBitSet {
    bits: u64,
}

impl ActionBitSet {
    /// Creates a new empty `ActionBitSet`.
    #[inline]
    pub fn new() -> Self {
        Self { bits: 0 }
    }

    /// Inserts an action index into the set.
    ///
    /// Returns `true` if the index was not already present (i.e., it was newly inserted),
    /// or `false` if it was already in the set.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `idx >= 52`.
    #[inline]
    pub fn insert(&mut self, idx: usize) -> bool {
        debug_assert!(idx < 52, "Action index must be < 52, got {}", idx);
        let mask = 1u64 << idx;
        let was_present = (self.bits & mask) != 0;
        self.bits |= mask;
        !was_present
    }

    /// Returns `true` if the set contains the given action index.
    #[inline]
    pub fn contains(&self, idx: usize) -> bool {
        debug_assert!(idx < 52, "Action index must be < 52, got {}", idx);
        (self.bits & (1u64 << idx)) != 0
    }

    /// Returns the number of action indices in the set.
    #[inline]
    #[allow(dead_code)]
    pub fn count(&self) -> usize {
        self.bits.count_ones() as usize
    }

    /// Returns `true` if the set is empty.
    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.bits == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_empty() {
        let set = ActionBitSet::new();
        assert!(set.is_empty());
        assert_eq!(set.count(), 0);
    }

    #[test]
    fn test_insert_and_contains() {
        let mut set = ActionBitSet::new();

        // Insert returns true for new insertions
        assert!(set.insert(0));
        assert!(set.insert(51));
        assert!(set.insert(25));

        // Insert returns false for duplicates
        assert!(!set.insert(0));
        assert!(!set.insert(51));

        // Contains works
        assert!(set.contains(0));
        assert!(set.contains(25));
        assert!(set.contains(51));
        assert!(!set.contains(1));
        assert!(!set.contains(50));
    }

    #[test]
    fn test_count() {
        let mut set = ActionBitSet::new();
        assert_eq!(set.count(), 0);

        set.insert(0);
        assert_eq!(set.count(), 1);

        set.insert(10);
        assert_eq!(set.count(), 2);

        set.insert(10); // duplicate
        assert_eq!(set.count(), 2);

        set.insert(51);
        assert_eq!(set.count(), 3);
    }

    #[test]
    fn test_all_indices() {
        let mut set = ActionBitSet::new();
        for i in 0..52 {
            assert!(set.insert(i));
        }
        assert_eq!(set.count(), 52);

        for i in 0..52 {
            assert!(set.contains(i));
        }
    }

    #[test]
    fn test_clone_copy() {
        let mut set = ActionBitSet::new();
        set.insert(10);
        set.insert(20);

        let copy = set;
        assert!(copy.contains(10));
        assert!(copy.contains(20));
        assert_eq!(copy.count(), 2);
    }
}
