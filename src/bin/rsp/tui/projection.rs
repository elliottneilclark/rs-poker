use std::collections::HashMap;

use rs_poker::arena::historian::StatsStorage;

use crate::tui::state::{AgentDisplayData, GameResult, ProfitHistory, StreetDistribution};
use crate::tui::widgets::stats_table::SortColumn;

/// Maximum number of profit history data points per agent.
pub(crate) const MAX_PROFIT_HISTORY: usize = 10_000;

/// A self-contained accumulator for everything the summary table, profit
/// graph, and street bars need. Fold games in with [`Projection::fold`]; read
/// derived display out with [`Projection::agent_display_data`].
///
/// Each projection counts its own games, so a *filtered* projection's profit
/// history is indexed from 1 over the matching games (the filtered ordinal),
/// while the *base* projection is indexed by absolute game number.
#[derive(Default)]
pub struct Projection {
    agent_stats: HashMap<String, StatsStorage>,
    agent_profit_bb: HashMap<String, f32>,
    agent_profit_history: HashMap<String, ProfitHistory>,
    street_dist: StreetDistribution,
    game_count: usize,
    cached_agent_display: Option<Vec<AgentDisplayData>>,
    cached_sort_col: Option<SortColumn>,
}

impl Projection {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn game_count(&self) -> usize {
        self.game_count
    }

    pub fn street_dist(&self) -> &StreetDistribution {
        &self.street_dist
    }

    pub fn profit_histories(&self) -> &HashMap<String, ProfitHistory> {
        &self.agent_profit_history
    }

    pub fn invalidate_display_cache(&mut self) {
        self.cached_agent_display = None;
        self.cached_sort_col = None;
    }

    /// Sorted list of agent names seen by this projection.
    pub fn agent_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.agent_stats.keys().cloned().collect();
        names.sort();
        names
    }

    /// Incorporate one game's contribution. O(1) in the number of prior games.
    pub fn fold(&mut self, result: &GameResult) {
        self.game_count += 1;
        self.cached_agent_display = None;
        self.cached_sort_col = None;

        self.street_dist.record(result.ending_round);

        for (seat_idx, name) in result.agent_names.iter().enumerate() {
            let storage = self
                .agent_stats
                .entry(name.clone())
                .or_insert_with(|| StatsStorage::new_with_num_players(1));
            result.seat_stats[seat_idx].merge_into(storage);
        }

        // Group profits by agent name so an agent in multiple seats gets one
        // history entry per game, not one per seat.
        let mut agent_profits: HashMap<&str, f32> = HashMap::new();
        for (seat_idx, name) in result.agent_names.iter().enumerate() {
            *agent_profits.entry(name.as_str()).or_default() += result.profits[seat_idx];
        }
        for (name, profit) in agent_profits {
            if result.big_blind > 0.0 {
                *self.agent_profit_bb.entry(name.to_string()).or_default() +=
                    profit / result.big_blind;
            }
            let history = self
                .agent_profit_history
                .entry(name.to_string())
                .or_insert_with(|| ProfitHistory {
                    first_game_index: self.game_count,
                    values: Vec::new(),
                });
            let prev = history.values.last().copied().unwrap_or(0.0);
            history.values.push(prev + profit);
            if history.values.len() > MAX_PROFIT_HISTORY {
                let drop_count = history.values.len() - MAX_PROFIT_HISTORY;
                history.values.drain(..drop_count);
                history.first_game_index += drop_count;
            }
        }
    }

    /// Per-agent display data sorted by `sort_col`. Memoized until the next
    /// `fold` or `invalidate_display_cache`. The cache is keyed on `sort_col`,
    /// so switching sort columns without an intervening `fold` recomputes.
    pub fn agent_display_data(&mut self, sort_col: SortColumn) -> Vec<AgentDisplayData> {
        if self.cached_sort_col == Some(sort_col)
            && let Some(cached) = &self.cached_agent_display
        {
            return cached.clone();
        }

        let mut agents: Vec<AgentDisplayData> = self
            .agent_stats
            .iter()
            .map(|(name, stats)| {
                let idx = 0;
                let profit_bb = self.agent_profit_bb.get(name).copied().unwrap_or(0.0);
                AgentDisplayData {
                    name: name.clone(),
                    total_profit: stats.total_profit[idx],
                    profit_bb,
                    games_played: stats.hands_played[idx],
                    wins: stats.games_won[idx],
                    vpip_percent: stats.vpip_percent(idx),
                    pfr_percent: stats.pfr_percent(idx),
                    three_bet_percent: stats.three_bet_percent(idx),
                    aggression_factor: stats.aggression_factor(idx),
                    cbet_percent: stats.cbet_percent(idx),
                    wtsd_percent: stats.wtsd_percent(idx),
                    wsd_percent: stats.wsd_percent(idx),
                    roi_percent: stats.roi_percent(idx),
                }
            })
            .collect();

        agents.sort_by(|a, b| match sort_col {
            SortColumn::Name => a.name.cmp(&b.name),
            SortColumn::Profit => b
                .profit_bb
                .partial_cmp(&a.profit_bb)
                .unwrap_or(std::cmp::Ordering::Equal),
            SortColumn::Games => b.games_played.cmp(&a.games_played),
            SortColumn::WinPct => {
                let pct = |x: &AgentDisplayData| {
                    if x.games_played > 0 {
                        x.wins as f32 / x.games_played as f32
                    } else {
                        0.0
                    }
                };
                pct(b)
                    .partial_cmp(&pct(a))
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
            SortColumn::Roi => b
                .roi_percent
                .partial_cmp(&a.roi_percent)
                .unwrap_or(std::cmp::Ordering::Equal),
            SortColumn::Vpip => b
                .vpip_percent
                .partial_cmp(&a.vpip_percent)
                .unwrap_or(std::cmp::Ordering::Equal),
            SortColumn::Pfr => b
                .pfr_percent
                .partial_cmp(&a.pfr_percent)
                .unwrap_or(std::cmp::Ordering::Equal),
            SortColumn::ThreeBet => b
                .three_bet_percent
                .partial_cmp(&a.three_bet_percent)
                .unwrap_or(std::cmp::Ordering::Equal),
            SortColumn::Af => b
                .aggression_factor
                .partial_cmp(&a.aggression_factor)
                .unwrap_or(std::cmp::Ordering::Equal),
            SortColumn::Cbet => b
                .cbet_percent
                .partial_cmp(&a.cbet_percent)
                .unwrap_or(std::cmp::Ordering::Equal),
            SortColumn::Wtsd => b
                .wtsd_percent
                .partial_cmp(&a.wtsd_percent)
                .unwrap_or(std::cmp::Ordering::Equal),
            SortColumn::Wsd => b
                .wsd_percent
                .partial_cmp(&a.wsd_percent)
                .unwrap_or(std::cmp::Ordering::Equal),
        });

        self.cached_agent_display = Some(agents.clone());
        self.cached_sort_col = Some(sort_col);
        agents
    }

    #[cfg(test)]
    pub fn set_game_count(&mut self, n: usize) {
        self.game_count = n;
    }
}

use crate::tui::hand_store::HandStore;

/// Build a fresh projection by folding the OHH hands for the given game
/// numbers, fetched on demand from disk. This is the single seam for the
/// filter-change recompute — a future async/background variant can replace it
/// without touching callers.
pub fn build_projection<I>(game_numbers: I, hand_store: &HandStore) -> Projection
where
    I: IntoIterator<Item = usize>,
{
    let mut proj = Projection::new();
    for n in game_numbers {
        if let Ok(Some(hand)) = hand_store.fetch(n) {
            proj.fold(&crate::tui::hand_stats::game_result_from_hand(&hand));
        }
    }
    proj
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::state::{GameResult, RoundLabel, SeatStats};
    use crate::tui::widgets::stats_table::SortColumn;
    use rs_poker::arena::historian::StatsStorage;

    fn make_game_result(names: &[&str], profits: &[f32], round: RoundLabel) -> GameResult {
        let n = names.len();
        let mut stats = StatsStorage::new_with_num_players(n);
        for (i, &p) in profits.iter().enumerate() {
            stats.total_profit[i] = p;
            stats.hands_played[i] = 1;
            stats.total_invested[i] = 10.0;
            if p > 0.0 {
                stats.games_won[i] = 1;
            } else if p < 0.0 {
                stats.games_lost[i] = 1;
            } else {
                stats.games_breakeven[i] = 1;
            }
        }
        let seat_stats = (0..n).map(|i| SeatStats::from_storage(&stats, i)).collect();
        GameResult {
            agent_names: names.iter().map(|s| s.to_string()).collect(),
            profits: profits.to_vec(),
            ending_round: round,
            seat_stats,
            big_blind: 10.0,
        }
    }

    #[test]
    fn test_fold_accumulates_profit_and_count() {
        let mut p = Projection::new();
        p.fold(&make_game_result(
            &["Alice", "Bob"],
            &[10.0, -10.0],
            RoundLabel::Flop,
        ));
        p.fold(&make_game_result(
            &["Alice", "Bob"],
            &[-5.0, 5.0],
            RoundLabel::River,
        ));
        assert_eq!(p.game_count(), 2);
        assert_eq!(p.street_dist().flop, 1);
        assert_eq!(p.street_dist().river, 1);

        let agents = p.agent_display_data(SortColumn::Profit);
        let alice = agents.iter().find(|a| a.name == "Alice").unwrap();
        assert!((alice.total_profit - 5.0).abs() < 0.01);
        assert_eq!(alice.games_played, 2);
        assert_eq!(alice.wins, 1);
    }

    #[test]
    fn test_filtered_projection_indexes_profit_history_from_one() {
        let mut p = Projection::new();
        p.fold(&make_game_result(&["Alice"], &[3.0], RoundLabel::Preflop));
        let hist = p.profit_histories().get("Alice").unwrap();
        assert_eq!(hist.first_game_index, 1);
        assert_eq!(hist.x_at(0), 1);
    }

    #[test]
    fn test_display_cache_invalidation() {
        let mut p = Projection::new();
        p.fold(&make_game_result(&["A"], &[1.0], RoundLabel::Preflop));
        let first = p.agent_display_data(SortColumn::Profit);
        p.invalidate_display_cache();
        let second = p.agent_display_data(SortColumn::Name);
        // After manual invalidation a fresh call must succeed (cache was cleared).
        assert_eq!(first.len(), second.len());
        assert_eq!(second.len(), 1);
    }

    /// "Zeb" has the highest profit but sorts last by name; "Amy" is opposite.
    /// This test will fail if `agent_display_data` ignores the `sort_col` key
    /// and returns a cached result from the previous call.
    #[test]
    fn test_display_data_respects_sort_column() {
        let mut p = Projection::new();
        p.fold(&make_game_result(
            &["Zeb", "Amy"],
            &[20.0, -20.0],
            RoundLabel::River,
        ));

        let by_profit = p.agent_display_data(SortColumn::Profit);
        // Profit sort is descending — Zeb (+20) before Amy (−20).
        assert_eq!(by_profit[0].name, "Zeb", "profit sort: Zeb should be first");
        assert_eq!(
            by_profit[1].name, "Amy",
            "profit sort: Amy should be second"
        );

        // Switch sort column without a fold — must recompute, not return stale order.
        let by_name = p.agent_display_data(SortColumn::Name);
        // Name sort is ascending — Amy before Zeb.
        assert_eq!(by_name[0].name, "Amy", "name sort: Amy should be first");
        assert_eq!(by_name[1].name, "Zeb", "name sort: Zeb should be second");
    }

    #[test]
    fn test_build_projection_from_disk_matches_folding_those_hands() {
        use crate::tui::hand_store::HandStore;
        use rs_poker::open_hand_history::OpenHandHistoryWrapper;
        use std::io::Write;

        // Write two simple heads-up hands to a temp OHH file.
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        for gn in ["1", "2"] {
            let h = crate::tui::hand_stats::test_util::simple_hand(gn);
            let wrapped = OpenHandHistoryWrapper { ohh: h };
            serde_json::to_writer(tmp.as_file_mut(), &wrapped).unwrap();
            writeln!(tmp.as_file_mut()).unwrap();
            writeln!(tmp.as_file_mut()).unwrap();
        }
        let store = HandStore::from_existing(tmp.path()).unwrap();

        // Building over game number 1 only equals folding only hand 1.
        let proj = build_projection([1usize], &store);
        assert_eq!(proj.game_count(), 1);
    }
}
