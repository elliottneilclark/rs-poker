use std::{
    collections::{BTreeSet, HashMap, HashSet},
    fmt,
    time::{Duration, Instant},
};

use rs_poker::arena::historian::StatsStorage;
use rs_poker::open_hand_history::{Action, HandHistory};

use crate::tui::event::SimError;
use crate::tui::widgets::stats_table::SortColumn;

/// Threshold for classifying profits as win/loss/breakeven.
pub const PROFIT_EPSILON: f32 = 0.01;

/// Compute per-player profits from an OHH `HandHistory`.
///
/// Returns `(id_to_idx, profits)` where `id_to_idx` maps player IDs to seat
/// indices and `profits[i]` is the net profit for the player at seat `i`.
pub fn compute_hand_profits(hand: &HandHistory) -> (HashMap<u64, usize>, Vec<f32>) {
    let num_players = hand.players.len();

    let id_to_idx: HashMap<u64, usize> = hand
        .players
        .iter()
        .enumerate()
        .map(|(i, p)| (p.id, i))
        .collect();

    let mut wins = vec![0.0_f32; num_players];
    for pot in &hand.pots {
        for pw in &pot.player_wins {
            if let Some(&idx) = id_to_idx.get(&pw.player_id) {
                wins[idx] += pw.win_amount;
            }
        }
    }

    let mut invested = vec![0.0_f32; num_players];
    for round in &hand.rounds {
        for action in &round.actions {
            if let Some(&idx) = id_to_idx.get(&action.player_id)
                && matches!(
                    action.action,
                    Action::Bet
                        | Action::Raise
                        | Action::Call
                        | Action::PostSmallBlind
                        | Action::PostBigBlind
                        | Action::PostAnte
                        | Action::Straddle
                        | Action::PostDead
                        | Action::PostExtraBlind
                        | Action::AddedToPot
                )
            {
                invested[idx] += action.amount;
            }
        }
    }

    let mut profits = vec![0.0_f32; num_players];
    for i in 0..num_players {
        profits[i] = wins[i] - invested[i];
    }

    (id_to_idx, profits)
}

/// Determine the ending round from a completed game's stats snapshot.
///
/// After `sim.run()`, `game_state.round` is always `Complete`, so we infer
/// the last street from which street-completion counters are non-zero.
/// The deepest street any player reached is the ending round.
pub fn ending_round_from_stats(stats: &StatsStorage, num_players: usize) -> RoundLabel {
    let any = |counts: &[usize]| counts.iter().take(num_players).any(|&c| c > 0);

    if any(&stats.showdown_count) {
        return RoundLabel::Showdown;
    }
    if any(&stats.river_completes) {
        return RoundLabel::River;
    }
    if any(&stats.turn_completes) {
        return RoundLabel::Turn;
    }
    if any(&stats.flop_completes) {
        return RoundLabel::Flop;
    }
    RoundLabel::Preflop
}

/// Maximum number of profit history data points per agent.
const MAX_PROFIT_HISTORY: usize = 10_000;

/// Flat per-seat stats extracted from a `StatsStorage` snapshot.
///
/// All fields are scalars — no heap allocations. This is `Copy` so it can be
/// sent across threads without triggering cross-thread malloc fragmentation
/// that caused OOM in the old `StatsStorage`-in-`GameResult` design.
#[derive(Debug, Clone, Copy, Default)]
pub struct SeatStats {
    pub actions_count: usize,
    pub vpip_count: usize,
    pub vpip_total: f32,
    pub raise_count: usize,
    pub hands_played: usize,
    pub hands_vpip: usize,
    pub hands_pfr: usize,
    pub preflop_raise_count: usize,
    pub preflop_actions: usize,
    pub three_bet_count: usize,
    pub three_bet_opportunities: usize,
    pub call_count: usize,
    pub bet_count: usize,
    pub total_profit: f32,
    pub total_invested: f32,
    pub games_won: usize,
    pub games_lost: usize,
    pub games_breakeven: usize,
    pub preflop_wins: usize,
    pub flop_wins: usize,
    pub turn_wins: usize,
    pub river_wins: usize,
    pub preflop_completes: usize,
    pub flop_completes: usize,
    pub turn_completes: usize,
    pub river_completes: usize,
    pub cbet_opportunities: usize,
    pub cbet_count: usize,
    pub wtsd_opportunities: usize,
    pub wtsd_count: usize,
    pub showdown_count: usize,
    pub showdown_wins: usize,
    pub fold_count: usize,
    pub flop_bets: usize,
    pub flop_raises: usize,
    pub flop_calls: usize,
    pub turn_bets: usize,
    pub turn_raises: usize,
    pub turn_calls: usize,
    pub river_bets: usize,
    pub river_raises: usize,
    pub river_calls: usize,
    pub steal_opportunities: usize,
    pub steal_count: usize,
}

impl SeatStats {
    /// Extract one seat's stats from a multi-player `StatsStorage`.
    pub fn from_storage(storage: &StatsStorage, seat: usize) -> Self {
        Self {
            actions_count: storage.actions_count[seat],
            vpip_count: storage.vpip_count[seat],
            vpip_total: storage.vpip_total[seat],
            raise_count: storage.raise_count[seat],
            hands_played: storage.hands_played[seat],
            hands_vpip: storage.hands_vpip[seat],
            hands_pfr: storage.hands_pfr[seat],
            preflop_raise_count: storage.preflop_raise_count[seat],
            preflop_actions: storage.preflop_actions[seat],
            three_bet_count: storage.three_bet_count[seat],
            three_bet_opportunities: storage.three_bet_opportunities[seat],
            call_count: storage.call_count[seat],
            bet_count: storage.bet_count[seat],
            total_profit: storage.total_profit[seat],
            total_invested: storage.total_invested[seat],
            games_won: storage.games_won[seat],
            games_lost: storage.games_lost[seat],
            games_breakeven: storage.games_breakeven[seat],
            preflop_wins: storage.preflop_wins[seat],
            flop_wins: storage.flop_wins[seat],
            turn_wins: storage.turn_wins[seat],
            river_wins: storage.river_wins[seat],
            preflop_completes: storage.preflop_completes[seat],
            flop_completes: storage.flop_completes[seat],
            turn_completes: storage.turn_completes[seat],
            river_completes: storage.river_completes[seat],
            cbet_opportunities: storage.cbet_opportunities[seat],
            cbet_count: storage.cbet_count[seat],
            wtsd_opportunities: storage.wtsd_opportunities[seat],
            wtsd_count: storage.wtsd_count[seat],
            showdown_count: storage.showdown_count[seat],
            showdown_wins: storage.showdown_wins[seat],
            fold_count: storage.fold_count[seat],
            flop_bets: storage.flop_bets[seat],
            flop_raises: storage.flop_raises[seat],
            flop_calls: storage.flop_calls[seat],
            turn_bets: storage.turn_bets[seat],
            turn_raises: storage.turn_raises[seat],
            turn_calls: storage.turn_calls[seat],
            river_bets: storage.river_bets[seat],
            river_raises: storage.river_raises[seat],
            river_calls: storage.river_calls[seat],
            steal_opportunities: storage.steal_opportunities[seat],
            steal_count: storage.steal_count[seat],
        }
    }

    /// Merge this seat's stats into a single-player accumulator at index 0.
    pub fn merge_into(&self, dest: &mut StatsStorage) {
        let d = 0;
        dest.actions_count[d] += self.actions_count;
        dest.vpip_count[d] += self.vpip_count;
        dest.vpip_total[d] += self.vpip_total;
        dest.raise_count[d] += self.raise_count;
        dest.hands_played[d] += self.hands_played;
        dest.hands_vpip[d] += self.hands_vpip;
        dest.hands_pfr[d] += self.hands_pfr;
        dest.preflop_raise_count[d] += self.preflop_raise_count;
        dest.preflop_actions[d] += self.preflop_actions;
        dest.three_bet_count[d] += self.three_bet_count;
        dest.three_bet_opportunities[d] += self.three_bet_opportunities;
        dest.call_count[d] += self.call_count;
        dest.bet_count[d] += self.bet_count;
        dest.total_profit[d] += self.total_profit;
        dest.total_invested[d] += self.total_invested;
        dest.games_won[d] += self.games_won;
        dest.games_lost[d] += self.games_lost;
        dest.games_breakeven[d] += self.games_breakeven;
        dest.preflop_wins[d] += self.preflop_wins;
        dest.flop_wins[d] += self.flop_wins;
        dest.turn_wins[d] += self.turn_wins;
        dest.river_wins[d] += self.river_wins;
        dest.preflop_completes[d] += self.preflop_completes;
        dest.flop_completes[d] += self.flop_completes;
        dest.turn_completes[d] += self.turn_completes;
        dest.river_completes[d] += self.river_completes;
        dest.cbet_opportunities[d] += self.cbet_opportunities;
        dest.cbet_count[d] += self.cbet_count;
        dest.wtsd_opportunities[d] += self.wtsd_opportunities;
        dest.wtsd_count[d] += self.wtsd_count;
        dest.showdown_count[d] += self.showdown_count;
        dest.showdown_wins[d] += self.showdown_wins;
        dest.fold_count[d] += self.fold_count;
        dest.flop_bets[d] += self.flop_bets;
        dest.flop_raises[d] += self.flop_raises;
        dest.flop_calls[d] += self.flop_calls;
        dest.turn_bets[d] += self.turn_bets;
        dest.turn_raises[d] += self.turn_raises;
        dest.turn_calls[d] += self.turn_calls;
        dest.river_bets[d] += self.river_bets;
        dest.river_raises[d] += self.river_raises;
        dest.river_calls[d] += self.river_calls;
        dest.steal_opportunities[d] += self.steal_opportunities;
        dest.steal_count[d] += self.steal_count;
    }
}

/// Result from a single completed game, sent from the simulation thread.
#[derive(Debug, Clone)]
pub struct GameResult {
    pub agent_names: Vec<String>,
    pub profits: Vec<f32>,
    pub ending_round: RoundLabel,
    pub seat_stats: Vec<SeatStats>,
}

/// Simplified round label for display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoundLabel {
    Preflop,
    Flop,
    Turn,
    River,
    Showdown,
}

impl RoundLabel {
    /// Parse an OHH street name into a `RoundLabel`.
    pub fn from_street_name(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "preflop" => Self::Preflop,
            "flop" => Self::Flop,
            "turn" => Self::Turn,
            "river" => Self::River,
            "showdown" => Self::Showdown,
            _ => Self::Showdown,
        }
    }
}

impl fmt::Display for RoundLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preflop => write!(f, "preflop"),
            Self::Flop => write!(f, "flop"),
            Self::Turn => write!(f, "turn"),
            Self::River => write!(f, "river"),
            Self::Showdown => write!(f, "showdown"),
        }
    }
}

/// Pre-computed display data for one agent.
#[derive(Debug, Clone)]
pub struct AgentDisplayData {
    pub name: String,
    pub total_profit: f32,
    pub games_played: usize,
    pub wins: usize,
    pub vpip_percent: f32,
    pub pfr_percent: f32,
    pub three_bet_percent: f32,
    pub aggression_factor: f32,
    pub cbet_percent: f32,
    pub wtsd_percent: f32,
    pub wsd_percent: f32,
    pub roi_percent: f32,
}

/// Tracks how games ended across streets.
#[derive(Debug, Clone, Default)]
pub struct StreetDistribution {
    pub preflop: usize,
    pub flop: usize,
    pub turn: usize,
    pub river: usize,
    pub showdown: usize,
}

impl StreetDistribution {
    pub fn total(&self) -> usize {
        self.preflop + self.flop + self.turn + self.river + self.showdown
    }

    fn record(&mut self, round: RoundLabel) {
        match round {
            RoundLabel::Preflop => self.preflop += 1,
            RoundLabel::Flop => self.flop += 1,
            RoundLabel::Turn => self.turn += 1,
            RoundLabel::River => self.river += 1,
            RoundLabel::Showdown => self.showdown += 1,
        }
    }
}

/// Filter state for narrowing the game log and stats table.
#[derive(Debug, Clone, Default)]
pub struct FilterState {
    pub winners: HashSet<String>,
    pub participants: HashSet<String>,
    pub streets: HashSet<RoundLabel>,
    pub player_counts: HashSet<usize>,
    /// Index of the selected item in the filter panel list.
    pub selected: usize,
}

impl FilterState {
    pub fn is_active(&self) -> bool {
        !self.winners.is_empty()
            || !self.participants.is_empty()
            || !self.streets.is_empty()
            || !self.player_counts.is_empty()
    }

    pub fn clear(&mut self) {
        self.winners.clear();
        self.participants.clear();
        self.streets.clear();
        self.player_counts.clear();
    }

    pub fn toggle_winner(&mut self, name: &str) {
        if !self.winners.remove(name) {
            self.winners.insert(name.to_string());
        }
    }

    pub fn toggle_participant(&mut self, name: &str) {
        if !self.participants.remove(name) {
            self.participants.insert(name.to_string());
        }
    }

    pub fn toggle_street(&mut self, street: RoundLabel) {
        if !self.streets.remove(&street) {
            self.streets.insert(street);
        }
    }

    pub fn toggle_player_count(&mut self, count: usize) {
        if !self.player_counts.remove(&count) {
            self.player_counts.insert(count);
        }
    }

    /// Returns true if the given game log entry passes all active filters.
    /// Filters combine with AND across types: winner AND participant AND street.
    pub fn matches_entry(&self, entry: &GameLogEntry) -> bool {
        // Winner filter: at least one winner in the entry matches
        if !self.winners.is_empty() {
            let has_winner = entry
                .agent_names
                .iter()
                .zip(entry.profits.iter())
                .any(|(name, profit)| *profit > 0.0 && self.winners.contains(name));
            if !has_winner {
                return false;
            }
        }

        // Participant filter: at least one participant in the entry matches
        if !self.participants.is_empty() {
            let has_participant = entry
                .agent_names
                .iter()
                .any(|name| self.participants.contains(name));
            if !has_participant {
                return false;
            }
        }

        // Street filter: entry's ending round matches one of the selected streets
        if !self.streets.is_empty() && !self.streets.contains(&entry.ending_round) {
            return false;
        }

        // Player count filter: entry's player count matches one of the selected counts
        if !self.player_counts.is_empty() && !self.player_counts.contains(&entry.agent_names.len())
        {
            return false;
        }

        true
    }
}

/// A single entry in the game log.
#[derive(Debug, Clone)]
pub struct GameLogEntry {
    pub game_number: usize,
    pub agent_names: Vec<String>,
    pub profits: Vec<f32>,
    pub ending_round: RoundLabel,
}

impl GameLogEntry {
    /// Create a `GameLogEntry` from an OHH `HandHistory`.
    pub fn from_hand(game_number: usize, hand: &HandHistory) -> Self {
        let agent_names: Vec<String> = hand.players.iter().map(|p| p.name.clone()).collect();
        let (_id_to_idx, profits) = compute_hand_profits(hand);

        let ending_round = hand
            .rounds
            .last()
            .map(|r| RoundLabel::from_street_name(&r.street))
            .unwrap_or(RoundLabel::Preflop);

        Self {
            game_number,
            agent_names,
            profits,
            ending_round,
        }
    }
}

/// The complete TUI state, updated as game results arrive.
pub struct TuiState {
    pub games_completed: usize,
    pub games_target: Option<usize>,
    pub start_time: Instant,
    pub completed: bool,
    /// Whether this is a live simulation (true) or a static viewer (false).
    pub live: bool,
    /// Error from the simulation thread, if any.
    pub error: Option<SimError>,

    /// Per-agent accumulated stats (keyed by agent name).
    agent_stats: HashMap<String, StatsStorage>,
    /// Per-agent running profit history (cumulative).
    agent_profit_history: HashMap<String, Vec<f32>>,
    /// Street distribution tracker.
    pub street_dist: StreetDistribution,
    /// Distinct player counts observed across games.
    pub distinct_player_counts: BTreeSet<usize>,

    // Cached derived data (invalidated on update / sort change)
    cached_agent_display: Option<Vec<AgentDisplayData>>,
    cached_agent_names: Option<Vec<String>>,

    // UI state
    pub table_selected: Option<usize>,
    pub log_selected: Option<usize>,
    pub log_scroll: usize,
    pub sort_col: SortColumn,
    pub active_panel: Panel,
    pub filter: FilterState,
}

/// Which panel is currently focused for keyboard navigation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Panel {
    Table,
    GameLog,
    Filter,
}

impl Panel {
    pub fn next(self) -> Self {
        match self {
            Self::Table => Self::GameLog,
            Self::GameLog => Self::Filter,
            Self::Filter => Self::Table,
        }
    }
}

impl TuiState {
    pub fn new(games_target: Option<usize>) -> Self {
        Self {
            games_completed: 0,
            games_target,
            start_time: Instant::now(),
            completed: false,
            live: true,
            error: None,
            agent_stats: HashMap::new(),
            agent_profit_history: HashMap::new(),
            street_dist: StreetDistribution::default(),
            distinct_player_counts: BTreeSet::new(),
            cached_agent_display: None,
            cached_agent_names: None,
            table_selected: None,
            log_selected: None,
            log_scroll: 0,
            sort_col: SortColumn::Profit,
            active_panel: Panel::Table,
            filter: FilterState::default(),
        }
    }

    /// Incorporate a game result into the state.
    pub fn update(&mut self, result: GameResult) {
        self.games_completed += 1;
        self.cached_agent_display = None;
        self.cached_agent_names = None;

        // Update street distribution and player count tracking
        self.street_dist.record(result.ending_round);
        self.distinct_player_counts.insert(result.agent_names.len());

        // Merge per-seat stats into per-agent accumulators
        for (seat_idx, name) in result.agent_names.iter().enumerate() {
            let agent_storage = self
                .agent_stats
                .entry(name.clone())
                .or_insert_with(|| StatsStorage::new_with_num_players(1));
            result.seat_stats[seat_idx].merge_into(agent_storage);
        }

        // Group profits by agent name so that an agent appearing in multiple
        // seats gets a single profit history entry per game (not one per seat).
        let mut agent_profits: HashMap<&str, f32> = HashMap::new();
        for (seat_idx, name) in result.agent_names.iter().enumerate() {
            *agent_profits.entry(name.as_str()).or_default() += result.profits[seat_idx];
        }
        for (name, profit) in agent_profits {
            let history = self
                .agent_profit_history
                .entry(name.to_string())
                .or_default();
            let prev = history.last().copied().unwrap_or(0.0);
            history.push(prev + profit);
            if history.len() > MAX_PROFIT_HISTORY {
                history.drain(..history.len() - MAX_PROFIT_HISTORY);
            }
        }
    }

    /// Get agent display data, sorted by the current sort column.
    /// Results are cached and only recomputed when state changes.
    pub fn agent_display_data(&mut self) -> Vec<AgentDisplayData> {
        if let Some(cached) = &self.cached_agent_display {
            return cached.clone();
        }

        let mut agents: Vec<AgentDisplayData> = self
            .agent_stats
            .iter()
            .map(|(name, stats)| {
                let idx = 0;

                AgentDisplayData {
                    name: name.clone(),
                    total_profit: stats.total_profit[idx],
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

        agents.sort_by(|a, b| match self.sort_col {
            SortColumn::Name => a.name.cmp(&b.name),
            SortColumn::Profit => b
                .total_profit
                .partial_cmp(&a.total_profit)
                .unwrap_or(std::cmp::Ordering::Equal),
            SortColumn::Games => b.games_played.cmp(&a.games_played),
            SortColumn::WinPct => {
                let a_pct = if a.games_played > 0 {
                    a.wins as f32 / a.games_played as f32
                } else {
                    0.0
                };
                let b_pct = if b.games_played > 0 {
                    b.wins as f32 / b.games_played as f32
                } else {
                    0.0
                };
                b_pct
                    .partial_cmp(&a_pct)
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
        agents
    }

    /// Return a sorted list of all agent names seen so far.
    /// Results are cached and only recomputed when state changes.
    pub fn all_agent_names(&mut self) -> Vec<String> {
        if let Some(cached) = &self.cached_agent_names {
            return cached.clone();
        }
        let mut names: Vec<String> = self.agent_stats.keys().cloned().collect();
        names.sort();
        self.cached_agent_names = Some(names.clone());
        names
    }

    /// Invalidate cached display data (e.g., after sort column change).
    pub fn invalidate_display_cache(&mut self) {
        self.cached_agent_display = None;
    }

    /// Borrow the per-agent profit histories (avoids cloning every frame).
    pub fn profit_histories(&self) -> &HashMap<String, Vec<f32>> {
        &self.agent_profit_history
    }

    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn games_per_second(&self) -> f64 {
        let secs = self.elapsed().as_secs_f64();
        if secs > 0.0 {
            self.games_completed as f64 / secs
        } else {
            0.0
        }
    }

    pub fn eta(&self) -> Option<Duration> {
        let gps = self.games_per_second();
        let target = self.games_target?;
        if gps <= 0.0 || self.games_completed >= target {
            return None;
        }
        let remaining = target - self.games_completed;
        Some(Duration::from_secs_f64(remaining as f64 / gps))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stats(num_players: usize) -> StatsStorage {
        StatsStorage::new_with_num_players(num_players)
    }

    fn make_game_result(names: &[&str], profits: &[f32], round: RoundLabel) -> GameResult {
        let num_players = names.len();
        let mut stats = make_stats(num_players);
        for (i, &profit) in profits.iter().enumerate() {
            stats.total_profit[i] = profit;
            stats.hands_played[i] = 1;
            stats.total_invested[i] = 10.0;
            if profit > 0.0 {
                stats.games_won[i] = 1;
            } else if profit < 0.0 {
                stats.games_lost[i] = 1;
            } else {
                stats.games_breakeven[i] = 1;
            }
        }
        let seat_stats: Vec<SeatStats> = (0..num_players)
            .map(|i| SeatStats::from_storage(&stats, i))
            .collect();
        GameResult {
            agent_names: names.iter().map(|s| s.to_string()).collect(),
            profits: profits.to_vec(),
            ending_round: round,
            seat_stats,
        }
    }

    #[test]
    fn test_new_state_is_empty() {
        let mut state = TuiState::new(Some(100));
        assert_eq!(state.games_completed, 0);
        assert_eq!(state.games_target, Some(100));
        assert_eq!(state.street_dist.total(), 0);
        assert!(state.agent_display_data().is_empty());
    }

    #[test]
    fn test_update_single_game() {
        let mut state = TuiState::new(Some(10));
        let result = make_game_result(&["Alice", "Bob"], &[15.0, -15.0], RoundLabel::River);
        state.update(result);

        assert_eq!(state.games_completed, 1);
        assert_eq!(state.street_dist.river, 1);

        let agents = state.agent_display_data();
        assert_eq!(agents.len(), 2);
        // Sorted by profit desc, Alice should be first
        assert_eq!(agents[0].name, "Alice");
        assert_eq!(agents[0].total_profit, 15.0);
        assert_eq!(agents[1].name, "Bob");
        assert_eq!(agents[1].total_profit, -15.0);
    }

    #[test]
    fn test_update_multiple_games_accumulates() {
        let mut state = TuiState::new(None);
        state.update(make_game_result(
            &["Alice", "Bob"],
            &[10.0, -10.0],
            RoundLabel::Flop,
        ));
        state.update(make_game_result(
            &["Alice", "Bob"],
            &[-5.0, 5.0],
            RoundLabel::River,
        ));

        assert_eq!(state.games_completed, 2);
        let agents = state.agent_display_data();
        let alice = agents.iter().find(|a| a.name == "Alice").unwrap();
        assert!((alice.total_profit - 5.0).abs() < 0.01);
        assert_eq!(alice.games_played, 2);
        assert_eq!(alice.wins, 1);
    }

    #[test]
    fn test_agent_profit_history_tracks_running_total() {
        let mut state = TuiState::new(None);
        state.update(make_game_result(&["Alice"], &[10.0], RoundLabel::Preflop));
        state.update(make_game_result(&["Alice"], &[-3.0], RoundLabel::Preflop));
        state.update(make_game_result(&["Alice"], &[7.0], RoundLabel::Preflop));

        let histories = state.profit_histories();
        let alice_history = histories.get("Alice").unwrap();
        assert_eq!(alice_history.len(), 3);
        assert!((alice_history[0] - 10.0).abs() < 0.01);
        assert!((alice_history[1] - 7.0).abs() < 0.01);
        assert!((alice_history[2] - 14.0).abs() < 0.01);
    }

    #[test]
    fn test_street_distribution_counts_all_rounds() {
        let mut state = TuiState::new(None);
        state.update(make_game_result(&["A"], &[1.0], RoundLabel::Preflop));
        state.update(make_game_result(&["A"], &[1.0], RoundLabel::Flop));
        state.update(make_game_result(&["A"], &[1.0], RoundLabel::Turn));
        state.update(make_game_result(&["A"], &[1.0], RoundLabel::River));
        state.update(make_game_result(&["A"], &[1.0], RoundLabel::Showdown));

        assert_eq!(state.street_dist.preflop, 1);
        assert_eq!(state.street_dist.flop, 1);
        assert_eq!(state.street_dist.turn, 1);
        assert_eq!(state.street_dist.river, 1);
        assert_eq!(state.street_dist.showdown, 1);
        assert_eq!(state.street_dist.total(), 5);
    }

    #[test]
    fn test_progress_calculations() {
        let mut state = TuiState::new(Some(100));
        for _ in 0..50 {
            state.update(make_game_result(&["A"], &[1.0], RoundLabel::Preflop));
        }
        assert_eq!(state.games_completed, 50);
        assert!(state.games_per_second() > 0.0);
    }

    #[test]
    fn test_eta_returns_none_when_no_target() {
        let state = TuiState::new(None);
        assert!(state.eta().is_none());
    }

    #[test]
    fn test_eta_returns_none_when_complete() {
        let mut state = TuiState::new(Some(1));
        state.update(make_game_result(&["A"], &[1.0], RoundLabel::Preflop));
        assert!(state.eta().is_none());
    }

    #[test]
    fn test_agent_display_data_sorted_by_profit() {
        let mut state = TuiState::new(None);
        state.update(make_game_result(
            &["Worst", "Best", "Mid"],
            &[-10.0, 20.0, 5.0],
            RoundLabel::River,
        ));

        let agents = state.agent_display_data();
        assert_eq!(agents[0].name, "Best");
        assert_eq!(agents[1].name, "Mid");
        assert_eq!(agents[2].name, "Worst");
    }

    #[test]
    fn test_duplicate_agent_name_single_profit_history_entry() {
        let mut state = TuiState::new(None);
        // Same agent in both seats (random with replacement)
        state.update(make_game_result(
            &["Bot", "Bot"],
            &[10.0, -10.0],
            RoundLabel::River,
        ));

        let agents = state.agent_display_data();
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].name, "Bot");
        // Net profit: 10 + (-10) = 0
        assert!((agents[0].total_profit - 0.0).abs() < 0.01);
        // Should have exactly 1 history entry, not 2
        let histories = state.profit_histories();
        let bot_history = histories.get("Bot").unwrap();
        assert_eq!(bot_history.len(), 1);
        assert!((bot_history[0] - 0.0).abs() < 0.01);
        // Both seats count as hands played
        assert_eq!(agents[0].games_played, 2);
    }

    #[test]
    fn test_seat_stats_roundtrip() {
        let mut source = make_stats(3);
        source.total_profit[1] = 42.0;
        source.hands_played[1] = 5;
        source.games_won[1] = 3;

        let seat = SeatStats::from_storage(&source, 1);
        assert_eq!(seat.total_profit, 42.0);
        assert_eq!(seat.hands_played, 5);
        assert_eq!(seat.games_won, 3);

        let mut dest = make_stats(1);
        seat.merge_into(&mut dest);
        assert_eq!(dest.total_profit[0], 42.0);
        assert_eq!(dest.hands_played[0], 5);
        assert_eq!(dest.games_won[0], 3);
    }

    #[test]
    fn test_panel_cycles_through_three() {
        assert_eq!(Panel::Table.next(), Panel::GameLog);
        assert_eq!(Panel::GameLog.next(), Panel::Filter);
        assert_eq!(Panel::Filter.next(), Panel::Table);
    }

    #[test]
    fn test_filter_state_default_is_inactive() {
        let filter = FilterState::default();
        assert!(!filter.is_active());
    }

    #[test]
    fn test_filter_toggle_winner() {
        let mut filter = FilterState::default();
        filter.toggle_winner("Alice");
        assert!(filter.is_active());
        assert!(filter.winners.contains("Alice"));
        // Toggle off
        filter.toggle_winner("Alice");
        assert!(!filter.is_active());
    }

    #[test]
    fn test_filter_toggle_participant() {
        let mut filter = FilterState::default();
        filter.toggle_participant("Bob");
        assert!(filter.participants.contains("Bob"));
        filter.toggle_participant("Bob");
        assert!(!filter.participants.contains("Bob"));
    }

    #[test]
    fn test_filter_toggle_street() {
        let mut filter = FilterState::default();
        filter.toggle_street(RoundLabel::Flop);
        assert!(filter.streets.contains(&RoundLabel::Flop));
        filter.toggle_street(RoundLabel::Flop);
        assert!(!filter.streets.contains(&RoundLabel::Flop));
    }

    #[test]
    fn test_filter_clear() {
        let mut filter = FilterState::default();
        filter.toggle_winner("Alice");
        filter.toggle_participant("Bob");
        filter.toggle_street(RoundLabel::River);
        assert!(filter.is_active());
        filter.clear();
        assert!(!filter.is_active());
    }

    #[test]
    fn test_filter_matches_entry_no_filters() {
        let filter = FilterState::default();
        let entry = GameLogEntry {
            game_number: 1,
            agent_names: vec!["Alice".into(), "Bob".into()],
            profits: vec![10.0, -10.0],
            ending_round: RoundLabel::River,
        };
        assert!(filter.matches_entry(&entry));
    }

    #[test]
    fn test_filter_matches_winner() {
        let mut filter = FilterState::default();
        filter.toggle_winner("Alice");
        let entry = GameLogEntry {
            game_number: 1,
            agent_names: vec!["Alice".into(), "Bob".into()],
            profits: vec![10.0, -10.0],
            ending_round: RoundLabel::River,
        };
        assert!(filter.matches_entry(&entry));

        // Bob is not a winner in this entry
        let mut filter2 = FilterState::default();
        filter2.toggle_winner("Bob");
        assert!(!filter2.matches_entry(&entry));
    }

    #[test]
    fn test_filter_matches_participant() {
        let mut filter = FilterState::default();
        filter.toggle_participant("Bob");
        let entry = GameLogEntry {
            game_number: 1,
            agent_names: vec!["Alice".into(), "Bob".into()],
            profits: vec![10.0, -10.0],
            ending_round: RoundLabel::River,
        };
        assert!(filter.matches_entry(&entry));

        let mut filter2 = FilterState::default();
        filter2.toggle_participant("Charlie");
        assert!(!filter2.matches_entry(&entry));
    }

    #[test]
    fn test_filter_matches_street() {
        let mut filter = FilterState::default();
        filter.toggle_street(RoundLabel::River);
        let river_entry = GameLogEntry {
            game_number: 1,
            agent_names: vec!["A".into()],
            profits: vec![1.0],
            ending_round: RoundLabel::River,
        };
        let flop_entry = GameLogEntry {
            game_number: 2,
            agent_names: vec!["A".into()],
            profits: vec![1.0],
            ending_round: RoundLabel::Flop,
        };
        assert!(filter.matches_entry(&river_entry));
        assert!(!filter.matches_entry(&flop_entry));
    }

    #[test]
    fn test_filter_and_semantics() {
        // Winner=Alice AND Street=River: both must match
        let mut filter = FilterState::default();
        filter.toggle_winner("Alice");
        filter.toggle_street(RoundLabel::River);

        let matching = GameLogEntry {
            game_number: 1,
            agent_names: vec!["Alice".into(), "Bob".into()],
            profits: vec![10.0, -10.0],
            ending_round: RoundLabel::River,
        };
        assert!(filter.matches_entry(&matching));

        // Alice wins but wrong street
        let wrong_street = GameLogEntry {
            game_number: 2,
            agent_names: vec!["Alice".into(), "Bob".into()],
            profits: vec![10.0, -10.0],
            ending_round: RoundLabel::Flop,
        };
        assert!(!filter.matches_entry(&wrong_street));

        // Right street but Alice lost
        let alice_lost = GameLogEntry {
            game_number: 3,
            agent_names: vec!["Alice".into(), "Bob".into()],
            profits: vec![-10.0, 10.0],
            ending_round: RoundLabel::River,
        };
        assert!(!filter.matches_entry(&alice_lost));
    }

    #[test]
    fn test_all_agent_names() {
        let mut state = TuiState::new(None);
        state.update(make_game_result(
            &["Charlie", "Alice"],
            &[10.0, -10.0],
            RoundLabel::River,
        ));
        state.update(make_game_result(
            &["Bob", "Alice"],
            &[5.0, -5.0],
            RoundLabel::Flop,
        ));
        let names = state.all_agent_names();
        assert_eq!(names, vec!["Alice", "Bob", "Charlie"]);
    }
}
