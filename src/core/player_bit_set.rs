use std::{
    fmt::{Debug, Display},
    ops::BitOr,
};

/// A struct representing a bit set for players.
///
/// # Examples
///
/// ```
/// use rs_poker::core::PlayerBitSet;
/// let mut active_players = PlayerBitSet::new(9);
///
/// // Player 4 folds
/// active_players.disable(4);
/// assert_eq!(8, active_players.count());
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct PlayerBitSet {
    set: u16,
}

impl Display for PlayerBitSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;

        for idx in (0..16).rev() {
            if self.get(idx) {
                write!(f, "A")?;
            } else {
                write!(f, "_")?;
            }
        }

        write!(f, "]")
    }
}

impl PlayerBitSet {
    /// Creates a new `PlayerBitSet` with `players` number of players.
    pub fn new(players: usize) -> Self {
        let set = (1 << players) - 1;
        Self { set }
    }

    /// Returns the number of players enabled in the bit set.
    pub fn count(&self) -> usize {
        self.set.count_ones() as usize
    }

    /// Returns `true` if the bit set is empty.
    pub fn empty(&self) -> bool {
        self.set == 0
    }

    /// Enables the player at `idx` position in the bit set.
    pub fn enable(&mut self, idx: usize) {
        self.set |= 1 << idx;
    }

    /// Disables the player at `idx` position in the bit set.
    pub fn disable(&mut self, idx: usize) {
        self.set &= !(1 << idx);
    }

    /// Returns `true` if the player at `idx` position in the bit set is
    /// enabled.
    pub fn get(&self, idx: usize) -> bool {
        (self.set & (1 << idx)) != 0
    }

    /// Returns an iterator over the active players in the bit set.
    pub fn ones(self) -> ActivePlayerBitSetIter {
        ActivePlayerBitSetIter { set: self.set }
    }
}

/// Implements the BitOr trait for PlayerBitSet, allowing two PlayerBitSet
/// instances to be combined using the | operator.
///
/// # Examples
///
/// ```
/// use rs_poker::core::PlayerBitSet;
///
/// let mut a = PlayerBitSet::default();
/// a.enable(0);
/// a.enable(2);
///
/// let mut b = PlayerBitSet::default();
/// b.enable(1);
/// b.enable(2);
///
/// let c = a | b;
/// assert_eq!(c.get(0), true);
/// assert_eq!(c.get(1), true);
/// assert_eq!(c.get(2), true);
/// ```
///
/// Here, two `PlayerBitSet` instances `a` and `b` are combined using the `|`
/// operator to produce a new `PlayerBitSet` instance `c`. `c` contains all the
/// bits set in either `a` or `b`.
impl BitOr for PlayerBitSet {
    type Output = PlayerBitSet;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            set: self.set | rhs.set,
        }
    }
}

/// An iterator over the active players in a bit set.
pub struct ActivePlayerBitSetIter {
    set: u16,
}

/// # Examples
///
/// ```
/// use rs_poker::core::PlayerBitSet;
///
/// let mut set = PlayerBitSet::default();
///
/// set.enable(0);
/// set.enable(4);
/// set.enable(2);
///
/// let mut iter = set.ones();
///
/// assert_eq!(iter.next(), Some(0));
/// assert_eq!(iter.next(), Some(2));
/// assert_eq!(iter.next(), Some(4));
/// assert_eq!(iter.next(), None);
/// ```
///
/// Here, we create a `PlayerBitSet` instance `set` and set some bits in it. We
/// then create an `ActivePlayerBitSetIter` iterator from the `set` instance and
/// use it to iterate over the active players. We verify that the iterator
/// returns the correct indices of the active players.
impl Iterator for ActivePlayerBitSetIter {
    type Item = usize;

    /// Returns the next active player in the bit set, or `None` if there are no
    /// more.
    fn next(&mut self) -> Option<Self::Item> {
        if self.set == 0 {
            None
        } else {
            // Find the index of the first non-zero bit
            let idx = self.set.trailing_zeros() as usize;
            // Then set the first non-zero bit to zero
            self.set &= !(1 << idx);
            // Then emit the next active player
            Some(idx)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_count() {
        assert_eq!(7, PlayerBitSet::new(7).count());
    }

    #[test]
    fn test_default_zero_count() {
        assert_eq!(0, PlayerBitSet::default().count());
    }

    #[test]
    fn test_disable_count() {
        let mut s = PlayerBitSet::new(7);

        assert_eq!(7, s.count());
        s.disable(6);
        assert_eq!(6, s.count());
        s.disable(0);
        assert_eq!(5, s.count());
    }

    #[test]
    fn test_enable_count() {
        let mut s = PlayerBitSet::default();

        assert_eq!(0, s.count());
        s.enable(0);
        assert_eq!(1, s.count());
        s.enable(0);
        assert_eq!(1, s.count());

        s.enable(2);
        assert_eq!(2, s.count());

        s.disable(0);
        assert_eq!(1, s.count());
    }

    #[test]
    fn test_iter() {
        let s = PlayerBitSet::new(2);
        let mut iter = s.ones();

        assert_eq!(Some(0), iter.next());
        assert_eq!(Some(1), iter.next());
        assert_eq!(None, iter.next());
    }

    #[test]
    fn test_iter_with_disabled() {
        let mut s = PlayerBitSet::new(3);
        let mut iter = s.ones();

        assert_eq!(Some(0), iter.next());
        assert_eq!(Some(1), iter.next());
        assert_eq!(Some(2), iter.next());
        assert_eq!(None, iter.next());

        s.disable(0);

        let mut after_iter = s.ones();
        assert_eq!(Some(1), after_iter.next());
        assert_eq!(Some(2), after_iter.next());
        assert_eq!(None, after_iter.next());
    }

    #[test]
    fn test_iter_with_enabled() {
        let mut s = PlayerBitSet::default();
        let mut iter = s.ones();
        assert_eq!(None, iter.next());

        s.enable(3);

        let mut after_iter = s.ones();
        assert_eq!(Some(3), after_iter.next());
        assert_eq!(None, after_iter.next());
    }

    #[test]
    fn test_display() {
        let mut s = PlayerBitSet::new(6);
        s.disable(2);

        assert_eq!("[__________AAA_AA]", format!("{}", s))
    }
}
