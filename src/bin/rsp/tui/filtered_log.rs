use crate::tui::{
    hand_store::HandStore,
    state::{FilterState, GameLogEntry},
};

/// Virtual game log backed by HandStore, with filter support and window caching.
///
/// Instead of keeping all game log entries in memory, this struct maintains:
/// - A list of filtered game numbers (when a filter is active)
/// - A small window cache of entries for the currently visible area
pub struct FilteredGameLog {
    /// 1-based game numbers that pass the active filter.
    filtered_indices: Vec<usize>,
    filter_active: bool,
    /// Cached entries for the visible window.
    window_cache: Vec<GameLogEntry>,
    /// Offset into the filtered list where window_cache starts.
    window_start: usize,
    /// Total games known (used when no filter is active).
    known_total: usize,
}

impl FilteredGameLog {
    pub fn new() -> Self {
        Self {
            filtered_indices: Vec::new(),
            filter_active: false,
            window_cache: Vec::new(),
            window_start: 0,
            known_total: 0,
        }
    }

    /// Total number of entries visible (filtered count or total games).
    pub fn total(&self) -> usize {
        if self.filter_active {
            self.filtered_indices.len()
        } else {
            self.known_total
        }
    }

    /// Update the known total game count.
    pub fn set_total(&mut self, total: usize) {
        self.known_total = total;
    }

    /// Rebuild filter indices by scanning all games in the HandStore.
    pub fn rebuild_filter(&mut self, filter: &FilterState, hand_store: &HandStore) {
        self.filter_active = filter.is_active();
        self.filtered_indices.clear();
        if self.filter_active {
            let total = hand_store.len();
            for game_num in 1..=total {
                if let Ok(Some(entry)) = hand_store.fetch_entry(game_num)
                    && filter.matches_entry(&entry)
                {
                    self.filtered_indices.push(game_num);
                }
            }
        }
        self.invalidate();
    }

    /// Notify that a new game has been added during live simulation.
    pub fn on_new_game(&mut self, entry: &GameLogEntry, filter: &FilterState) {
        self.known_total = entry.game_number;
        if self.filter_active && filter.matches_entry(entry) {
            self.filtered_indices.push(entry.game_number);
        }
        // Invalidate window cache since total changed
        self.window_cache.clear();
    }

    /// Invalidate the window cache, forcing a reload on next ensure_window.
    pub fn invalidate(&mut self) {
        self.window_cache.clear();
    }

    /// Resolve a filtered index to a 1-based game number.
    pub fn game_number_at(&self, filtered_idx: usize) -> Option<usize> {
        if self.filter_active {
            self.filtered_indices.get(filtered_idx).copied()
        } else if filtered_idx < self.known_total {
            Some(filtered_idx + 1)
        } else {
            None
        }
    }

    /// Ensure the window cache covers entries needed for the given scroll position.
    pub fn ensure_window(&mut self, scroll: usize, visible_rows: usize, hand_store: &HandStore) {
        let total = self.total();
        let need_start = scroll;
        let need_end = (scroll + visible_rows + 1).min(total);

        if need_end <= need_start {
            self.window_cache.clear();
            self.window_start = 0;
            return;
        }

        // Check if current cache already covers the needed range
        let cache_end = self.window_start + self.window_cache.len();
        if !self.window_cache.is_empty() && self.window_start <= need_start && cache_end >= need_end
        {
            return;
        }

        // Load with buffer for smooth scrolling
        let buffer = visible_rows;
        let load_start = need_start.saturating_sub(buffer);
        let load_end = (need_end + buffer).min(total);

        self.window_cache.clear();
        self.window_start = load_start;

        for filtered_idx in load_start..load_end {
            let game_number = if self.filter_active {
                match self.filtered_indices.get(filtered_idx) {
                    Some(&n) => n,
                    None => continue,
                }
            } else {
                filtered_idx + 1
            };
            match hand_store.fetch_entry(game_number) {
                Ok(Some(entry)) => self.window_cache.push(entry),
                _ => self.window_cache.push(GameLogEntry::new(
                    game_number,
                    vec![],
                    vec![],
                    crate::tui::state::RoundLabel::Preflop,
                    0.0,
                )),
            }
        }
    }

    /// Get the entries for the visible window at the given scroll position.
    /// Must call `ensure_window` first.
    pub fn window_entries(&self, scroll: usize, visible_rows: usize) -> &[GameLogEntry] {
        if self.window_cache.is_empty() {
            return &[];
        }
        let start = scroll.saturating_sub(self.window_start);
        let end = (start + visible_rows + 1).min(self.window_cache.len());
        let start = start.min(end);
        &self.window_cache[start..end]
    }
}

#[cfg(test)]
impl FilteredGameLog {
    /// Create with a preset filter for testing without a HandStore.
    pub fn test_with_filter(total: usize, filtered_indices: Vec<usize>) -> Self {
        Self {
            filtered_indices,
            filter_active: true,
            window_cache: Vec::new(),
            window_start: 0,
            known_total: total,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::state::RoundLabel;

    #[test]
    fn test_new_is_empty() {
        let log = FilteredGameLog::new();
        assert_eq!(log.total(), 0);
    }

    #[test]
    fn test_set_total() {
        let mut log = FilteredGameLog::new();
        log.set_total(42);
        assert_eq!(log.total(), 42);
    }

    #[test]
    fn test_game_number_at_no_filter() {
        let mut log = FilteredGameLog::new();
        log.set_total(5);
        assert_eq!(log.game_number_at(0), Some(1));
        assert_eq!(log.game_number_at(4), Some(5));
        assert_eq!(log.game_number_at(5), None);
    }

    #[test]
    fn test_game_number_at_with_filter() {
        let log = FilteredGameLog::test_with_filter(10, vec![2, 5, 8]);
        assert_eq!(log.game_number_at(0), Some(2));
        assert_eq!(log.game_number_at(1), Some(5));
        assert_eq!(log.game_number_at(2), Some(8));
        assert_eq!(log.game_number_at(3), None);
    }

    #[test]
    fn test_total_with_filter() {
        let mut log = FilteredGameLog::new();
        log.set_total(100);
        assert_eq!(log.total(), 100);

        let log = FilteredGameLog::test_with_filter(100, vec![1, 3, 5]);
        assert_eq!(log.total(), 3);
    }

    #[test]
    fn test_on_new_game_no_filter() {
        let mut log = FilteredGameLog::new();
        let filter = FilterState::default();
        let entry = GameLogEntry::new(1, vec!["A".into()], vec![1.0], RoundLabel::Preflop, 10.0);
        log.on_new_game(&entry, &filter);
        assert_eq!(log.total(), 1);
    }

    #[test]
    fn test_on_new_game_with_filter_match() {
        let mut log = FilteredGameLog::new();
        log.filter_active = true;

        let mut filter = FilterState::default();
        filter.toggle_street(RoundLabel::River);

        let entry = GameLogEntry::new(1, vec!["A".into()], vec![1.0], RoundLabel::River, 10.0);
        log.on_new_game(&entry, &filter);
        assert_eq!(log.total(), 1);
        assert_eq!(log.game_number_at(0), Some(1));
    }

    #[test]
    fn test_on_new_game_with_filter_no_match() {
        let mut log = FilteredGameLog::new();
        log.filter_active = true;

        let mut filter = FilterState::default();
        filter.toggle_street(RoundLabel::River);

        let entry = GameLogEntry::new(1, vec!["A".into()], vec![1.0], RoundLabel::Flop, 10.0);
        log.on_new_game(&entry, &filter);
        assert_eq!(log.total(), 0);
    }

    #[test]
    fn test_ensure_window_empty() {
        let mut log = FilteredGameLog::new();
        let store = HandStore::none();
        log.ensure_window(0, 10, &store);
        assert!(log.window_entries(0, 10).is_empty());
    }

    #[test]
    fn test_ensure_window_placeholder_on_fetch_failure() {
        // When fetch_entry returns Ok(None) for some indices, the cache
        // must still maintain a 1:1 mapping between logical index and cache
        // position (using placeholder entries) so that window_entries returns
        // the correct slice.
        let mut log = FilteredGameLog::new();
        log.set_total(5);
        let store = HandStore::none(); // all fetches return Ok(None)

        log.ensure_window(0, 5, &store);

        // Cache should have 5 placeholder entries, not 0
        assert_eq!(
            log.window_cache.len(),
            5,
            "window_cache should have placeholders for unfetchable entries"
        );

        // window_entries should return a non-empty slice for the visible range
        let entries = log.window_entries(0, 4);
        assert_eq!(entries.len(), 5, "should return all 5 entries (0..=4)");

        // Each placeholder should have the correct game_number
        assert_eq!(entries[0].game_number, 1);
        assert_eq!(entries[2].game_number, 3);
        assert_eq!(entries[4].game_number, 5);
    }

    #[test]
    fn test_invalidate_clears_cache() {
        let mut log = FilteredGameLog::new();
        log.window_cache.push(GameLogEntry::new(
            1,
            vec!["A".into()],
            vec![1.0],
            RoundLabel::Preflop,
            10.0,
        ));
        assert_eq!(log.window_cache.len(), 1);
        log.invalidate();
        assert!(log.window_cache.is_empty());
    }
}
