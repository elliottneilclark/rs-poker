use std::ops::BitOr;

#[derive(Debug, Default, Clone, Copy)]
pub struct PlayerBitSet {
    set: u16,
}

impl PlayerBitSet {
    pub fn new(players: usize) -> Self {
        let set = (1 << players) - 1;
        Self { set }
    }
    pub fn count(&self) -> usize {
        self.set.count_ones() as usize
    }
    pub fn empty(&self) -> bool {
        self.set == 0
    }
    pub fn enable(&mut self, idx: usize) {
        self.set |= 1 << idx;
    }
    pub fn disable(&mut self, idx: usize) {
        self.set &= !(1 << idx);
    }
    pub fn get(&self, idx: usize) -> bool {
        (self.set & (1 << idx)) != 0
    }
    pub fn ones(self) -> ActivePlayerBitSetIter {
        ActivePlayerBitSetIter { set: self.set }
    }
}

impl BitOr for PlayerBitSet {
    type Output = PlayerBitSet;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            set: self.set | rhs.set,
        }
    }
}

pub struct ActivePlayerBitSetIter {
    set: u16,
}

impl Iterator for ActivePlayerBitSetIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.set == 0 {
            None
        } else {
            // Find the index of the first non-zero
            let idx = self.set.trailing_zeros() as usize;
            // Then set the first non-zero to zero
            self.set &= !(1 << idx);
            // Then emit the next one
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
}
