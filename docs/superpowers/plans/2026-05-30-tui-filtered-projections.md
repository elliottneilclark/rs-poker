# TUI Filtered Projections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the arena `generate` / `ohh view` TUI's summary table, profit graph, and street bars re-derive from only the games matching the active filter, omit the ETA for unbounded runs, and show `{matching}/{total}` in the status bar.

**Architecture:** Lift the per-agent accumulators out of `TuiState` into a self-contained `Projection` (fold games in, read derived display out). `TuiState` holds a `base` projection (all games, maintained incrementally) and an optional `filtered` projection (rebuilt from disk on filter change). Views read the *active* projection. A single canonical `game_result_from_hand` reconstruction computes the displayed stats from an OHH `HandHistory`, used by the static viewer and by the filtered recompute, so filtered == unfiltered.

**Tech Stack:** Rust, ratatui, `insta` snapshot tests, `cargo nextest` via `mise`.

**Reference spec:** `docs/superpowers/specs/2026-05-30-tui-filtered-projections-design.md`

**Commands:**
- Single test: `mise check:test:nextest <test_name>`
- All tests: `mise check:test:nextest`
- Lint+format: `mise fix && mise check:clippy`
- Full check: `mise check`

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/bin/rsp/tui/projection.rs` | `Projection` accumulator (fold + memoized display) and `build_projection` (disk recompute) | **Create** |
| `src/bin/rsp/tui/hand_stats.rs` | Canonical `game_result_from_hand` — reconstruct displayed stats from an OHH `HandHistory` | **Create** |
| `src/bin/rsp/tui/state.rs` | `TuiState` holds `base`/`filtered` projections; delegate accessors; `update(&GameResult)` | **Modify** |
| `src/bin/rsp/tui/filtered_log.rs` | Expose `indices()` for the recompute | **Modify** |
| `src/bin/rsp/tui/app.rs` | Orchestrate append-fold + filter-change recompute | **Modify** |
| `src/bin/rsp/tui/widgets/progress_bar.rs` | ETA-when-bounded; `{matching}/{total}` count | **Modify** |
| `src/bin/rsp/tui/screens/overview.rs` | Read `state.street_dist()` accessor | **Modify** |
| `src/bin/rsp/ohh/stats.rs` | `build_state_from_hands` delegates to `hand_stats` | **Modify** |
| `src/bin/rsp/tui/mod.rs` | Register `projection` and `hand_stats` modules | **Modify** |

**Phase order (dependency-driven):**
1. **Phase 1 — Extract `Projection`** (behavior-preserving refactor; `base` only).
2. **Phase 2 — Canonical `hand_stats` reconstruction** (full parity for displayed stats; static viewer benefits immediately).
3. **Phase 3 — Filtered projection + orchestration + UI fixes**.

This order means the filtered fold (Phase 3) is wired only after the disk reconstruction (Phase 2) already matches the live stats, so there is never a shipped state with a visible filtered-vs-unfiltered discrepancy.

---

## Phase 1 — Extract `Projection`

### Task 1: Create the `Projection` accumulator

**Files:**
- Create: `src/bin/rsp/tui/projection.rs`
- Modify: `src/bin/rsp/tui/mod.rs`
- Modify: `src/bin/rsp/tui/state.rs` (make `StreetDistribution::record` and `ProfitHistory` reachable)

- [ ] **Step 1: Register the module**

In `src/bin/rsp/tui/mod.rs`, add alongside the other `mod` declarations (keep alphabetical with neighbors):

```rust
mod projection;
```

- [ ] **Step 2: Make `StreetDistribution::record` crate-visible**

In `src/bin/rsp/tui/state.rs`, change the `record` method on `StreetDistribution` from private to `pub(crate)`:

```rust
    pub(crate) fn record(&mut self, round: RoundLabel) {
```

- [ ] **Step 3: Write the failing test for `Projection::fold`**

Create `src/bin/rsp/tui/projection.rs` with only the test module first:

```rust
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
        p.fold(&make_game_result(&["Alice", "Bob"], &[10.0, -10.0], RoundLabel::Flop));
        p.fold(&make_game_result(&["Alice", "Bob"], &[-5.0, 5.0], RoundLabel::River));
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
        // Cached clone returns same data.
        let second = p.agent_display_data(SortColumn::Profit);
        assert_eq!(first.len(), second.len());
        p.invalidate_display_cache();
        let third = p.agent_display_data(SortColumn::Name);
        assert_eq!(third.len(), 1);
    }
}
```

Run: `mise check:test:nextest test_fold_accumulates_profit_and_count`
Expected: FAIL — `Projection` not found.

- [ ] **Step 4: Implement `Projection`**

Prepend to `src/bin/rsp/tui/projection.rs` (above the test module):

```rust
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
    /// `fold` or `invalidate_display_cache`.
    pub fn agent_display_data(&mut self, sort_col: SortColumn) -> Vec<AgentDisplayData> {
        if let Some(cached) = &self.cached_agent_display {
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
        agents
    }

    #[cfg(test)]
    pub fn set_game_count(&mut self, n: usize) {
        self.game_count = n;
    }
}
```

Run: `mise check:test:nextest test_fold_accumulates_profit_and_count`
Expected: PASS (and the two other new tests).

- [ ] **Step 5: Commit**

```bash
git add src/bin/rsp/tui/projection.rs src/bin/rsp/tui/mod.rs src/bin/rsp/tui/state.rs
git commit -m "feat(tui): add Projection accumulator extracted from TuiState"
```

---

### Task 2: Make `TuiState` hold a `base` `Projection` and delegate

**Files:**
- Modify: `src/bin/rsp/tui/state.rs`
- Modify: `src/bin/rsp/ohh/stats.rs` (call `update(&result)`)
- Modify: `src/bin/rsp/tui/app.rs` (call sites for `update` and `games_completed`)
- Modify: `src/bin/rsp/tui/widgets/progress_bar.rs` (`games_completed()` call sites)
- Modify: `src/bin/rsp/tui/screens/overview.rs` (`street_dist()` call site)

- [ ] **Step 1: Replace `TuiState`'s accumulator fields with a `base` projection**

In `src/bin/rsp/tui/state.rs`, in the `TuiState` struct (lines ~617-653), delete these fields:

```rust
    pub games_completed: usize,
    ...
    agent_stats: HashMap<String, StatsStorage>,
    agent_profit_bb: HashMap<String, f32>,
    agent_profit_history: HashMap<String, ProfitHistory>,
    pub street_dist: StreetDistribution,
    ...
    cached_agent_display: Option<Vec<AgentDisplayData>>,
```

and replace with:

```rust
    /// Projection over ALL games (maintained incrementally on each `update`).
    base: crate::tui::projection::Projection,
    /// Projection over only the filter-matching games, or `None` when no
    /// filter is active. Rebuilt from disk on filter change (Phase 3).
    filtered: Option<crate::tui::projection::Projection>,
```

Keep `cached_agent_names`, `distinct_player_counts`, the meta fields (`games_target`, `start_time`, `completed`, `live`, `error`), and all UI fields (`table_selected`, `log_selected`, `log_scroll`, `sort_col`, `active_panel`, `filter`).

Remove the now-unused `MAX_PROFIT_HISTORY` const from `state.rs` (it lives in `projection.rs` now, as `pub(crate)`). Remove unused imports (`StatsStorage`) if the compiler flags them — but keep `ProfitHistory`, `StreetDistribution`, `AgentDisplayData`, `SeatStats`, `GameResult` (still defined/used here).

- [ ] **Step 2: Update `TuiState::new`**

Replace the field initializers in `TuiState::new` (lines ~682-703):

```rust
    pub fn new(games_target: Option<usize>) -> Self {
        Self {
            games_target,
            start_time: Instant::now(),
            completed: false,
            live: true,
            error: None,
            base: crate::tui::projection::Projection::new(),
            filtered: None,
            distinct_player_counts: BTreeSet::new(),
            cached_agent_names: None,
            table_selected: None,
            log_selected: None,
            log_scroll: 0,
            sort_col: SortColumn::Profit,
            active_panel: Panel::Table,
            filter: FilterState::default(),
        }
    }
```

- [ ] **Step 3: Rewrite `update`, accessors, and add `games_completed()`**

Replace `TuiState::update` (lines ~706-756), `agent_display_data` (~760-848), `all_agent_names` (~852-860), `invalidate_display_cache` (~863-865), and `profit_histories` (~868-870) with:

```rust
    /// Incorporate a game result into the base projection.
    pub fn update(&mut self, result: &GameResult) {
        self.base.fold(result);
        self.distinct_player_counts.insert(result.agent_names.len());
        self.cached_agent_names = None;
    }

    /// Total games seen (the base projection's count).
    pub fn games_completed(&self) -> usize {
        self.base.game_count()
    }

    /// Number of games matching the active filter, or the total when no filter
    /// is active.
    pub fn matching_games(&self) -> usize {
        self.filtered
            .as_ref()
            .map(|f| f.game_count())
            .unwrap_or_else(|| self.base.game_count())
    }

    /// The projection the views should read: filtered if active, else base.
    fn active_projection(&self) -> &crate::tui::projection::Projection {
        self.filtered.as_ref().unwrap_or(&self.base)
    }

    fn active_projection_mut(&mut self) -> &mut crate::tui::projection::Projection {
        self.filtered.as_mut().unwrap_or(&mut self.base)
    }

    /// Replace (or clear) the filtered projection. Called on filter change.
    pub fn set_filter_projection(&mut self, proj: Option<crate::tui::projection::Projection>) {
        self.filtered = proj;
    }

    /// Fold a newly-arrived game into the filtered projection if one is active.
    pub fn fold_filtered(&mut self, result: &GameResult) {
        if let Some(f) = self.filtered.as_mut() {
            f.fold(result);
        }
    }

    /// Per-agent display data over the ACTIVE projection, sorted by `sort_col`.
    pub fn agent_display_data(&mut self) -> Vec<AgentDisplayData> {
        let sort = self.sort_col;
        self.active_projection_mut().agent_display_data(sort)
    }

    /// All agent names seen across ALL games (for the filter panel options).
    pub fn all_agent_names(&mut self) -> Vec<String> {
        if let Some(cached) = &self.cached_agent_names {
            return cached.clone();
        }
        let names = self.base.agent_names();
        self.cached_agent_names = Some(names.clone());
        names
    }

    /// Invalidate the active projection's display cache (e.g. on sort change).
    pub fn invalidate_display_cache(&mut self) {
        self.active_projection_mut().invalidate_display_cache();
    }

    /// Profit histories over the ACTIVE projection (for the graph).
    pub fn profit_histories(&self) -> &HashMap<String, ProfitHistory> {
        self.active_projection().profit_histories()
    }

    /// Street distribution over the ACTIVE projection (for the street bars).
    pub fn street_dist(&self) -> &StreetDistribution {
        self.active_projection().street_dist()
    }

    #[cfg(test)]
    pub fn set_games_completed(&mut self, n: usize) {
        self.base.set_game_count(n);
    }
```

In `eta` and `games_per_second` (lines ~876-893), replace every `self.games_completed` with `self.games_completed()`.

- [ ] **Step 4: Fix the `ohh/stats.rs` call site**

In `src/bin/rsp/ohh/stats.rs`, the `state.update(GameResult { ... })` call (lines ~89-95) now needs a reference:

```rust
        state.update(&GameResult {
            agent_names,
            profits,
            ending_round,
            seat_stats,
            big_blind: hand.big_blind_amount,
        });
```

- [ ] **Step 5: Fix `app.rs` call sites**

In `src/bin/rsp/tui/app.rs` `handle_sim_message` (lines ~491-500), replace:

```rust
            SimMessage::GameResult(result) => {
                let entry = GameLogEntry::new(
                    self.state.games_completed() + 1,
                    result.agent_names.clone(),
                    result.profits.clone(),
                    result.ending_round,
                    result.big_blind,
                );
                self.state.update(&result);
                self.filtered_log.on_new_game(&entry, &self.state.filter);
            }
```

(Phase 3 adds the filtered fold here.)

- [ ] **Step 6: Fix `progress_bar.rs` and `overview.rs` call sites**

In `src/bin/rsp/tui/widgets/progress_bar.rs`, replace each `state.games_completed` with `state.games_completed()` (lines ~57, 62, 68, 98). In the test at line ~120, replace `state.games_completed = 42;` with `state.set_games_completed(42);`.

In `src/bin/rsp/tui/screens/overview.rs` line ~95, replace `&state.street_dist` with `state.street_dist()`:

```rust
    render_street_bars(frame, left_chunks[1], state.street_dist());
```

- [ ] **Step 7: Move/adapt the accumulator tests to `Projection`**

The `state.rs` tests for profit-history eviction, duplicate-name handling, street distribution, and sorted display data now test `Projection` behavior through `TuiState`. They still compile because they call `TuiState::update`/`agent_display_data`/`profit_histories`. Apply these mechanical fixes inside `state.rs`'s `#[cfg(test)] mod tests`:

- In `make_game_result`, nothing changes.
- Every `state.update(make_game_result(...))` becomes `state.update(&make_game_result(...))`.
- `test_progress_calculations` and others reading `state.games_completed` become `state.games_completed()`.
- `test_street_distribution_counts_all_rounds` reads `state.street_dist.preflop` → `state.street_dist().preflop` (and the other streets).
- `test_profit_history_tracks_first_game_index_after_eviction` references `MAX_PROFIT_HISTORY`; add `use crate::tui::projection::MAX_PROFIT_HISTORY;` to that test module (the const now lives in `projection.rs`). The test still drives the behavior through `TuiState::update(&...)` and asserts unchanged values.

- [ ] **Step 8: Run the full TUI test module and verify behavior is preserved**

Run: `mise check:test:nextest`
Expected: PASS. This is a behavior-preserving refactor — no test assertions change values, only call syntax.

- [ ] **Step 9: Lint and commit**

```bash
mise fix && mise check:clippy
git add -A
git commit -m "refactor(tui): TuiState delegates accumulation to a base Projection"
```

---

## Phase 2 — Canonical `hand_stats` reconstruction (full parity for displayed stats)

`AgentDisplayData` surfaces: profit, profit_bb, games_played, wins, vpip%, pfr%, 3bet%, aggression_factor, cbet%, wtsd%, wsd%, roi%. This phase builds one function that reconstructs exactly those from an OHH `HandHistory`, matching `StatsTrackingHistorian` (`src/arena/historian/stats_tracking.rs`). Non-displayed `StatsStorage` fields (steal, per-street splits, per-street wins) are intentionally not reconstructed and are documented as such.

### Task 3: Scaffold `hand_stats` with profit, invested, win/loss, and ending round

**Files:**
- Create: `src/bin/rsp/tui/hand_stats.rs`
- Modify: `src/bin/rsp/tui/mod.rs`

- [ ] **Step 1: Register the module**

In `src/bin/rsp/tui/mod.rs` add:

```rust
mod hand_stats;
```

- [ ] **Step 2: Write the failing test (financials + ending round)**

Create `src/bin/rsp/tui/hand_stats.rs` with the test module and shared test helpers:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rs_poker::open_hand_history::*;

    pub(super) fn player(id: u64, seat: u64, name: &str) -> PlayerObj {
        PlayerObj {
            id,
            seat,
            name: name.to_string(),
            display: None,
            starting_stack: 1000.0,
            player_bounty: None,
            is_sitting_out: None,
        }
    }

    pub(super) fn act(player_id: u64, action: Action, amount: f32) -> ActionObj {
        ActionObj {
            action_number: 0,
            player_id,
            action,
            amount,
            is_allin: false,
            cards: None,
        }
    }

    pub(super) fn round(street: &str, actions: Vec<ActionObj>) -> RoundObj {
        RoundObj {
            id: 0,
            street: street.into(),
            cards: None,
            actions,
        }
    }

    pub(super) fn win(player_id: u64, amount: f32) -> PlayerWinsObj {
        PlayerWinsObj {
            player_id,
            win_amount: amount,
            cashout_amount: None,
            cashout_fee: None,
            bonus_amount: None,
            contributed_rake: None,
        }
    }

    pub(super) fn hand(
        players: Vec<PlayerObj>,
        rounds: Vec<RoundObj>,
        pots: Vec<PotObj>,
        dealer_seat: u64,
    ) -> HandHistory {
        HandHistory {
            spec_version: "1.4.7".into(),
            site_name: "test".into(),
            network_name: "test".into(),
            internal_version: "1.0".into(),
            tournament: false,
            tournament_info: None,
            game_number: "1".into(),
            start_date_utc: None,
            table_name: "test".into(),
            table_handle: None,
            table_skin: None,
            game_type: GameType::Holdem,
            bet_limit: None,
            table_size: players.len() as u64,
            currency: "USD".into(),
            dealer_seat,
            small_blind_amount: 5.0,
            big_blind_amount: 10.0,
            ante_amount: 0.0,
            hero_player_id: None,
            players,
            rounds,
            pots,
            tournament_bounties: None,
        }
    }

    fn pot(amount: f32, wins: Vec<PlayerWinsObj>) -> PotObj {
        PotObj {
            number: 0,
            amount,
            rake: None,
            jackpot: None,
            player_wins: wins,
        }
    }

    #[test]
    fn test_financials_and_ending_round() {
        // Alice (SB) folds preflop, Bob (BB) wins the 15 pot.
        let h = hand(
            vec![player(1, 0, "Alice"), player(2, 1, "Bob")],
            vec![round(
                "Preflop",
                vec![
                    act(1, Action::PostSmallBlind, 5.0),
                    act(2, Action::PostBigBlind, 10.0),
                    act(1, Action::Fold, 0.0),
                ],
            )],
            vec![pot(15.0, vec![win(2, 15.0)])],
            0,
        );

        let result = game_result_from_hand(&h);
        assert_eq!(result.agent_names, vec!["Alice", "Bob"]);
        assert_eq!(result.ending_round, RoundLabel::Preflop);
        // Alice invested 5, won 0 => -5. Bob invested 10, won 15 => +5.
        assert!((result.profits[0] - (-5.0)).abs() < 0.01);
        assert!((result.profits[1] - 5.0).abs() < 0.01);

        let s = &result.seat_stats;
        assert_eq!(s[0].hands_played, 1);
        assert_eq!(s[0].games_lost, 1);
        assert_eq!(s[1].games_won, 1);
        // total_invested: Alice 5, Bob 10.
        assert!((s[0].total_invested - 5.0).abs() < 0.01);
        assert!((s[1].total_invested - 10.0).abs() < 0.01);
    }
}
```

Run: `mise check:test:nextest test_financials_and_ending_round`
Expected: FAIL — `game_result_from_hand` not found.

- [ ] **Step 3: Implement the scaffold**

Prepend to `src/bin/rsp/tui/hand_stats.rs`:

```rust
//! Reconstruct the TUI's displayed per-agent statistics from an Open Hand
//! History record.
//!
//! This is the single canonical "stats from a hand" path, used by the static
//! OHH viewer (`ohh::stats::build_state_from_hands`) and by the filtered-view
//! recompute (`projection::build_projection`). It mirrors the definitions in
//! `StatsTrackingHistorian` (`src/arena/historian/stats_tracking.rs`) for every
//! metric surfaced in `AgentDisplayData`:
//!   profit, invested/ROI, win/loss, VPIP, PFR, 3-bet, aggression factor,
//!   c-bet, WTSD, WSD.
//!
//! Stats NOT surfaced by the TUI table (steal, per-street bet/raise/call
//! splits, per-street wins) are intentionally left at their defaults — the
//! viewer never reads them, so reconstructing them would be dead code.

use rs_poker::arena::historian::StatsStorage;
use rs_poker::open_hand_history::{Action, HandHistory};

use crate::tui::state::{
    GameResult, PROFIT_EPSILON, RoundLabel, SeatStats, compute_hand_profits,
};

/// Reconstruct a [`GameResult`] (profits, ending round, per-seat displayed
/// stats) from an OHH `HandHistory`.
pub fn game_result_from_hand(hand: &HandHistory) -> GameResult {
    let num_players = hand.players.len();
    let (id_to_idx, profits) = compute_hand_profits(hand);

    let mut storage = StatsStorage::new_with_num_players(num_players.max(1));

    // --- Financials, hands played, win/loss/breakeven ---
    for i in 0..num_players {
        storage.hands_played[i] = 1;
        storage.total_profit[i] = profits[i];

        // invested = sum of every chip this player put in the pot, incl. blinds.
        let mut invested = 0.0_f32;
        for r in &hand.rounds {
            for a in &r.actions {
                if id_to_idx.get(&a.player_id) == Some(&i) && is_invested_action(a.action) {
                    invested += a.amount;
                }
            }
        }
        storage.total_invested[i] = invested;

        if profits[i] > PROFIT_EPSILON {
            storage.games_won[i] = 1;
        } else if profits[i] < -PROFIT_EPSILON {
            storage.games_lost[i] = 1;
        } else {
            storage.games_breakeven[i] = 1;
        }
    }

    let ending_round = hand
        .rounds
        .last()
        .map(|r| RoundLabel::from_street_name(&r.street))
        .unwrap_or(RoundLabel::Preflop);

    // (Phase 2 tasks fill in vpip/pfr/3bet/aggression/cbet/wtsd below.)
    reconstruct_action_stats(hand, &id_to_idx, &mut storage);

    let agent_names: Vec<String> = hand.players.iter().map(|p| p.name.clone()).collect();
    let seat_stats: Vec<SeatStats> = (0..num_players)
        .map(|i| SeatStats::from_storage(&storage, i))
        .collect();

    GameResult {
        agent_names,
        profits,
        ending_round,
        seat_stats,
        big_blind: hand.big_blind_amount,
    }
}

/// Actions that move chips into the pot (used for `total_invested`).
fn is_invested_action(action: Action) -> bool {
    matches!(
        action,
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
}

/// Reconstruct the action-derived displayed stats. Filled in by later tasks.
fn reconstruct_action_stats(
    _hand: &HandHistory,
    _id_to_idx: &std::collections::HashMap<u64, usize>,
    _storage: &mut StatsStorage,
) {
}
```

Run: `mise check:test:nextest test_financials_and_ending_round`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/bin/rsp/tui/hand_stats.rs src/bin/rsp/tui/mod.rs
git commit -m "feat(tui): scaffold hand_stats reconstruction (financials, ending round)"
```

---

### Task 4: Reconstruct VPIP and PFR

**Files:**
- Modify: `src/bin/rsp/tui/hand_stats.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `hand_stats.rs`:

```rust
    #[test]
    fn test_vpip_pfr() {
        // Alice raises preflop (vpip + pfr). Bob calls (vpip only). Carol folds.
        let h = hand(
            vec![
                player(1, 0, "Alice"),
                player(2, 1, "Bob"),
                player(3, 2, "Carol"),
            ],
            vec![round(
                "Preflop",
                vec![
                    act(1, Action::PostSmallBlind, 5.0),
                    act(2, Action::PostBigBlind, 10.0),
                    act(3, Action::Raise, 30.0),
                    act(1, Action::Call, 25.0),
                    act(2, Action::Fold, 0.0),
                ],
            )],
            vec![],
            0,
        );
        // Re-map: who raised is Carol(idx2), who called is Alice(idx0).
        let r = game_result_from_hand(&h);
        let s = &r.seat_stats;
        // Carol raised => vpip + pfr.
        assert_eq!(s[2].hands_vpip, 1);
        assert_eq!(s[2].hands_pfr, 1);
        assert_eq!(s[2].preflop_raise_count, 1);
        // Alice called => vpip, not pfr.
        assert_eq!(s[0].hands_vpip, 1);
        assert_eq!(s[0].hands_pfr, 0);
        // Bob only posted BB then folded => no voluntary money => no vpip.
        assert_eq!(s[1].hands_vpip, 0);
        assert_eq!(s[1].hands_pfr, 0);
    }
```

Run: `mise check:test:nextest test_vpip_pfr`
Expected: FAIL.

- [ ] **Step 2: Implement VPIP/PFR inside `reconstruct_action_stats`**

Replace the empty `reconstruct_action_stats` body with the per-street walk that this and later tasks extend. For this task, implement only the preflop VPIP/PFR portion:

```rust
fn reconstruct_action_stats(
    hand: &HandHistory,
    id_to_idx: &std::collections::HashMap<u64, usize>,
    storage: &mut StatsStorage,
) {
    for r in &hand.rounds {
        let is_preflop = r.street.eq_ignore_ascii_case("preflop");
        for a in &r.actions {
            let Some(&idx) = id_to_idx.get(&a.player_id) else {
                continue;
            };
            match a.action {
                Action::Call => {
                    storage.call_count[idx] += 1;
                    if is_preflop {
                        storage.hands_vpip[idx] = 1;
                        storage.vpip_count[idx] += 1;
                        storage.vpip_total[idx] += a.amount;
                    }
                }
                Action::Bet => {
                    storage.bet_count[idx] += 1;
                    if is_preflop {
                        storage.hands_vpip[idx] = 1;
                        storage.hands_pfr[idx] = 1;
                        storage.vpip_count[idx] += 1;
                        storage.vpip_total[idx] += a.amount;
                    }
                }
                Action::Raise => {
                    storage.raise_count[idx] += 1;
                    if is_preflop {
                        storage.hands_vpip[idx] = 1;
                        storage.hands_pfr[idx] = 1;
                        storage.preflop_raise_count[idx] += 1;
                        storage.vpip_count[idx] += 1;
                        storage.vpip_total[idx] += a.amount;
                    }
                }
                Action::Fold => {
                    storage.fold_count[idx] += 1;
                }
                _ => {}
            }
            storage.actions_count[idx] += 1;
        }
    }
}
```

Run: `mise check:test:nextest test_vpip_pfr`
Expected: PASS. Also run `mise check:test:nextest test_financials_and_ending_round` (still PASS).

- [ ] **Step 3: Commit**

```bash
git add src/bin/rsp/tui/hand_stats.rs
git commit -m "feat(tui): reconstruct VPIP/PFR and aggression counts from OHH"
```

---

### Task 5: Reconstruct 3-bet opportunities and count

**Files:**
- Modify: `src/bin/rsp/tui/hand_stats.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
    #[test]
    fn test_three_bet() {
        // Open-raise by A; B faces raise #1 and re-raises => 3-bet.
        // C faces raise #1 then a 3-bet; C only gets a 3-bet *opportunity*
        // while facing exactly the open raise.
        let h = hand(
            vec![
                player(1, 0, "A"),
                player(2, 1, "B"),
                player(3, 2, "C"),
            ],
            vec![round(
                "Preflop",
                vec![
                    act(1, Action::PostSmallBlind, 5.0),
                    act(2, Action::PostBigBlind, 10.0),
                    act(3, Action::Raise, 30.0), // open raise (raise #1)
                    act(1, Action::Raise, 90.0), // 3-bet (faced raise #1)
                    act(2, Action::Fold, 0.0),
                    act(3, Action::Call, 60.0),
                ],
            )],
            vec![],
            0,
        );
        let r = game_result_from_hand(&h);
        let s = &r.seat_stats;
        // A (idx0) re-raised while facing exactly raise #1 => 3-bet.
        assert_eq!(s[0].three_bet_count, 1);
        assert_eq!(s[0].three_bet_opportunities, 1);
        // C (idx2) made the open raise (facing raise #0) => not a 3-bet, no opp at that point.
        assert_eq!(s[2].three_bet_count, 0);
        // B (idx1) folded while facing raise #2 (the 3-bet) => no opportunity (only raise #1 counts).
        assert_eq!(s[1].three_bet_opportunities, 0);
    }
```

Run: `mise check:test:nextest test_three_bet`
Expected: FAIL.

- [ ] **Step 2: Implement 3-bet tracking**

3-bet logic is preflop-only and depends on the running count of raises seen so far. Add a preflop raise counter to the walk. Modify `reconstruct_action_stats` so the preflop branch maintains `pf_raises` and marks opportunities/counts *before* incrementing on a raise:

Replace the `for a in &r.actions {` loop body's match with a version that, for preflop, computes 3-bet state. The cleanest structure is to track `pf_raises` per round:

```rust
    for r in &hand.rounds {
        let is_preflop = r.street.eq_ignore_ascii_case("preflop");
        let mut pf_raises: usize = 0; // number of raises seen so far this preflop
        for a in &r.actions {
            let Some(&idx) = id_to_idx.get(&a.player_id) else {
                continue;
            };

            // 3-bet opportunity: acting while facing exactly the open raise.
            if is_preflop
                && pf_raises == 1
                && matches!(a.action, Action::Fold | Action::Call | Action::Raise | Action::Bet)
            {
                storage.three_bet_opportunities[idx] += 1;
            }

            match a.action {
                Action::Call => {
                    storage.call_count[idx] += 1;
                    if is_preflop {
                        storage.hands_vpip[idx] = 1;
                        storage.vpip_count[idx] += 1;
                        storage.vpip_total[idx] += a.amount;
                    }
                }
                Action::Bet => {
                    storage.bet_count[idx] += 1;
                    if is_preflop {
                        storage.hands_vpip[idx] = 1;
                        storage.hands_pfr[idx] = 1;
                        storage.vpip_count[idx] += 1;
                        storage.vpip_total[idx] += a.amount;
                    }
                }
                Action::Raise => {
                    storage.raise_count[idx] += 1;
                    if is_preflop {
                        storage.hands_vpip[idx] = 1;
                        storage.hands_pfr[idx] = 1;
                        storage.preflop_raise_count[idx] += 1;
                        storage.vpip_count[idx] += 1;
                        storage.vpip_total[idx] += a.amount;
                        if pf_raises == 1 {
                            storage.three_bet_count[idx] += 1;
                        }
                        pf_raises += 1;
                    }
                }
                Action::Fold => {
                    storage.fold_count[idx] += 1;
                }
                _ => {}
            }
            storage.actions_count[idx] += 1;
        }
    }
```

Run: `mise check:test:nextest test_three_bet`
Expected: PASS. Re-run `test_vpip_pfr` and `test_financials_and_ending_round`: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/bin/rsp/tui/hand_stats.rs
git commit -m "feat(tui): reconstruct 3-bet opportunities and count from OHH"
```

---

### Task 6: Reconstruct C-bet (continuation bet)

**Files:**
- Modify: `src/bin/rsp/tui/hand_stats.rs`

C-bet definition (mirrors historian): the *preflop aggressor* is the last player to raise preflop. They have a c-bet **opportunity** if, on the flop, they act while no bet has yet occurred on the flop; the opportunity is **taken** if that action is a Bet or Raise.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn test_cbet() {
        // A opens preflop (last preflop raiser => aggressor). On the flop A is
        // first to act and bets => cbet opportunity taken.
        let h = hand(
            vec![player(1, 0, "A"), player(2, 1, "B")],
            vec![
                round(
                    "Preflop",
                    vec![
                        act(1, Action::PostSmallBlind, 5.0),
                        act(2, Action::PostBigBlind, 10.0),
                        act(1, Action::Raise, 30.0),
                        act(2, Action::Call, 20.0),
                    ],
                ),
                round("Flop", vec![act(1, Action::Bet, 40.0), act(2, Action::Fold, 0.0)]),
            ],
            vec![],
            0,
        );
        let r = game_result_from_hand(&h);
        let s = &r.seat_stats;
        assert_eq!(s[0].cbet_opportunities, 1);
        assert_eq!(s[0].cbet_count, 1);
        // B was not the aggressor => no cbet opportunity.
        assert_eq!(s[1].cbet_opportunities, 0);
    }

    #[test]
    fn test_cbet_opportunity_not_taken_when_checked() {
        // A opens, on flop A checks (no Bet/Raise) => opportunity, not taken.
        let h = hand(
            vec![player(1, 0, "A"), player(2, 1, "B")],
            vec![
                round(
                    "Preflop",
                    vec![
                        act(1, Action::PostSmallBlind, 5.0),
                        act(2, Action::PostBigBlind, 10.0),
                        act(1, Action::Raise, 30.0),
                        act(2, Action::Call, 20.0),
                    ],
                ),
                round("Flop", vec![act(1, Action::Check, 0.0), act(2, Action::Check, 0.0)]),
            ],
            vec![],
            0,
        );
        let r = game_result_from_hand(&h);
        assert_eq!(r.seat_stats[0].cbet_opportunities, 1);
        assert_eq!(r.seat_stats[0].cbet_count, 0);
    }
```

Run: `mise check:test:nextest test_cbet`
Expected: FAIL.

- [ ] **Step 2: Implement c-bet, after the main per-street walk**

Add a helper and call it at the end of `reconstruct_action_stats` (after the `for r in &hand.rounds` loop):

```rust
    reconstruct_cbet(hand, id_to_idx, storage);
```

Then add the helper function:

```rust
/// C-bet: the preflop aggressor (last preflop raiser) gets an opportunity if,
/// on the flop, they act with no prior flop bet; taken if that action bets/raises.
fn reconstruct_cbet(
    hand: &HandHistory,
    id_to_idx: &std::collections::HashMap<u64, usize>,
    storage: &mut StatsStorage,
) {
    // Preflop aggressor = player_id of the LAST preflop Raise.
    let mut aggressor: Option<u64> = None;
    for r in &hand.rounds {
        if r.street.eq_ignore_ascii_case("preflop") {
            for a in &r.actions {
                if a.action == Action::Raise {
                    aggressor = Some(a.player_id);
                }
            }
        }
    }
    let Some(aggressor_id) = aggressor else {
        return;
    };
    let Some(&aggressor_idx) = id_to_idx.get(&aggressor_id) else {
        return;
    };

    // Walk the flop: find the aggressor's first action before any flop bet.
    for r in &hand.rounds {
        if !r.street.eq_ignore_ascii_case("flop") {
            continue;
        }
        let mut flop_bet_occurred = false;
        for a in &r.actions {
            if a.player_id == aggressor_id && !flop_bet_occurred {
                storage.cbet_opportunities[aggressor_idx] += 1;
                if matches!(a.action, Action::Bet | Action::Raise) {
                    storage.cbet_count[aggressor_idx] += 1;
                }
                break; // only the first qualifying flop action matters
            }
            if matches!(a.action, Action::Bet | Action::Raise) {
                flop_bet_occurred = true;
            }
        }
        break; // only one flop round
    }
}
```

Run: `mise check:test:nextest test_cbet` and `test_cbet_opportunity_not_taken_when_checked`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/bin/rsp/tui/hand_stats.rs
git commit -m "feat(tui): reconstruct c-bet opportunities/count from OHH"
```

---

### Task 7: Reconstruct WTSD and WSD (showdown)

**Files:**
- Modify: `src/bin/rsp/tui/hand_stats.rs`

Definitions (mirror historian): a player *saw the flop* if they did not fold during preflop (and a flop round exists). The hand *went to showdown* if ≥2 players never folded at all. WTSD count = saw flop AND went to showdown AND never folded. Showdown count = went to showdown AND never folded. Showdown win = showdown count AND profit > epsilon.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn test_wtsd_and_wsd() {
        // A and B both reach showdown on the river; A wins.
        let h = hand(
            vec![player(1, 0, "A"), player(2, 1, "B")],
            vec![
                round(
                    "Preflop",
                    vec![
                        act(1, Action::PostSmallBlind, 5.0),
                        act(2, Action::PostBigBlind, 10.0),
                        act(1, Action::Call, 5.0),
                        act(2, Action::Check, 0.0),
                    ],
                ),
                round("Flop", vec![act(2, Action::Check, 0.0), act(1, Action::Check, 0.0)]),
                round("Turn", vec![act(2, Action::Check, 0.0), act(1, Action::Check, 0.0)]),
                round("River", vec![act(2, Action::Check, 0.0), act(1, Action::Check, 0.0)]),
            ],
            vec![pot(20.0, vec![win(1, 20.0)])],
            0,
        );
        let r = game_result_from_hand(&h);
        let s = &r.seat_stats;
        // Both saw the flop, both went to showdown.
        assert_eq!(s[0].wtsd_opportunities, 1);
        assert_eq!(s[0].wtsd_count, 1);
        assert_eq!(s[1].wtsd_opportunities, 1);
        assert_eq!(s[1].wtsd_count, 1);
        assert_eq!(s[0].showdown_count, 1);
        assert_eq!(s[1].showdown_count, 1);
        // A won at showdown (profit +10), B lost (-10).
        assert_eq!(s[0].showdown_wins, 1);
        assert_eq!(s[1].showdown_wins, 0);
    }

    #[test]
    fn test_no_showdown_when_one_player_left() {
        // A opens, B folds preflop => no showdown; A saw no flop opportunity.
        let h = hand(
            vec![player(1, 0, "A"), player(2, 1, "B")],
            vec![round(
                "Preflop",
                vec![
                    act(1, Action::PostSmallBlind, 5.0),
                    act(2, Action::PostBigBlind, 10.0),
                    act(1, Action::Raise, 30.0),
                    act(2, Action::Fold, 0.0),
                ],
            )],
            vec![pot(20.0, vec![win(1, 20.0)])],
            0,
        );
        let r = game_result_from_hand(&h);
        let s = &r.seat_stats;
        assert_eq!(s[0].showdown_count, 0);
        assert_eq!(s[0].wtsd_opportunities, 0); // no flop round
    }
```

Run: `mise check:test:nextest test_wtsd_and_wsd`
Expected: FAIL.

- [ ] **Step 2: Implement WTSD/WSD**

Showdown reconstruction needs each seat's profit, which `reconstruct_action_stats` does not currently receive. Thread `profits` through it.

First, in `game_result_from_hand`, change the call to pass `&profits`:

```rust
    reconstruct_action_stats(hand, &id_to_idx, &profits, &mut storage);
```

Then change the `reconstruct_action_stats` signature to accept `profits`:

```rust
fn reconstruct_action_stats(
    hand: &HandHistory,
    id_to_idx: &std::collections::HashMap<u64, usize>,
    profits: &[f32],
    storage: &mut StatsStorage,
) {
```

Then, at the very end of its body (after the `reconstruct_cbet(hand, id_to_idx, storage);` call added in Task 6), add:

```rust
    reconstruct_showdown(hand, id_to_idx, profits, storage);
```

Add the helper:

```rust
/// WTSD / showdown reconstruction.
fn reconstruct_showdown(
    hand: &HandHistory,
    id_to_idx: &std::collections::HashMap<u64, usize>,
    profits: &[f32],
    storage: &mut StatsStorage,
) {
    let num_players = hand.players.len();
    let has_flop = hand.rounds.iter().any(|r| r.street.eq_ignore_ascii_case("flop"));

    // folded_anywhere[i]: player folded at any point in the hand.
    let mut folded = vec![false; num_players];
    // folded_preflop[i]: player folded during the preflop round.
    let mut folded_preflop = vec![false; num_players];
    for r in &hand.rounds {
        let is_preflop = r.street.eq_ignore_ascii_case("preflop");
        for a in &r.actions {
            if a.action == Action::Fold
                && let Some(&idx) = id_to_idx.get(&a.player_id)
            {
                folded[idx] = true;
                if is_preflop {
                    folded_preflop[idx] = true;
                }
            }
        }
    }

    let survivors = (0..num_players).filter(|&i| !folded[i]).count();
    let went_to_showdown = survivors >= 2;

    for i in 0..num_players {
        // Saw the flop: a flop round exists and the player did not fold preflop.
        if has_flop && !folded_preflop[i] {
            storage.wtsd_opportunities[i] += 1;
            if went_to_showdown && !folded[i] {
                storage.wtsd_count[i] += 1;
            }
        }
        if went_to_showdown && !folded[i] {
            storage.showdown_count[i] += 1;
            if profits[i] > PROFIT_EPSILON {
                storage.showdown_wins[i] += 1;
            }
        }
    }
}
```

Run: `mise check:test:nextest test_wtsd_and_wsd` and `test_no_showdown_when_one_player_left`
Expected: PASS. Re-run the whole `hand_stats` module: `mise check:test:nextest hand_stats` — all PASS.

- [ ] **Step 3: Commit**

```bash
git add src/bin/rsp/tui/hand_stats.rs
git commit -m "feat(tui): reconstruct WTSD/WSD showdown stats from OHH"
```

---

### Task 8: Route `build_state_from_hands` through `hand_stats`

**Files:**
- Modify: `src/bin/rsp/ohh/stats.rs`

- [ ] **Step 1: Replace the partial reconstruction with delegation**

Replace the body of `build_state_from_hands` (lines ~9-99) with:

```rust
use rs_poker::open_hand_history::HandHistory;

use crate::tui::hand_stats::game_result_from_hand;
use crate::tui::state::TuiState;

/// Build a TuiState from a collection of parsed hand histories.
pub fn build_state_from_hands(hands: &[HandHistory]) -> TuiState {
    let mut state = TuiState::new(Some(hands.len()));
    for hand in hands {
        if hand.players.is_empty() {
            continue;
        }
        let result = game_result_from_hand(hand);
        state.update(&result);
    }
    state
}
```

Delete the now-unused imports at the top of `ohh/stats.rs` (`Action`, `StatsStorage`, `compute_hand_profits`, `PROFIT_EPSILON`, `SeatStats`, `RoundLabel`, `GameResult`) — keep only what the file still references. The compiler's unused-import warnings (denied by `#![deny(clippy::all)]`) will tell you exactly which to remove.

- [ ] **Step 2: Verify the existing `ohh/stats.rs` tests still pass**

The existing tests (`test_single_hand_profits`, `test_action_counting`, etc.) now exercise `hand_stats` through `build_state_from_hands` and act as regression coverage. The `test_action_counting` test asserts `b.vpip_percent > 0.0` and `b.pfr_percent > 0.0` for a preflop raiser — still satisfied.

Run: `mise check:test:nextest stats`
Expected: PASS.

- [ ] **Step 3: Add a static-viewer parity test for a new stat**

Add to the `tests` module in `ohh/stats.rs` (using its existing `make_*` helpers) a test asserting a previously-missing stat is now populated:

```rust
    #[test]
    fn test_cbet_now_reconstructed_in_static_view() {
        // Heads-up: player 1 raises preflop then c-bets the flop.
        let players = vec![make_player(1, "A"), make_player(2, "B")];
        let rounds = vec![
            make_round(
                1,
                "Preflop",
                vec![
                    make_action(1, 1, Action::PostSmallBlind, 5.0),
                    make_action(2, 2, Action::PostBigBlind, 10.0),
                    make_action(3, 1, Action::Raise, 30.0),
                    make_action(4, 2, Action::Call, 20.0),
                ],
            ),
            make_round(
                2,
                "Flop",
                vec![
                    make_action(5, 1, Action::Bet, 40.0),
                    make_action(6, 2, Action::Fold, 0.0),
                ],
            ),
        ];
        let hand = make_hand(players, rounds, vec![]);
        let mut state = build_state_from_hands(&[hand]);
        let agents = state.agent_display_data();
        let a = agents.iter().find(|x| x.name == "A").unwrap();
        assert!(a.cbet_percent > 0.0, "static viewer should now compute c-bet%");
    }
```

Run: `mise check:test:nextest test_cbet_now_reconstructed_in_static_view`
Expected: PASS.

- [ ] **Step 4: Lint and commit**

```bash
mise fix && mise check:clippy
git add src/bin/rsp/ohh/stats.rs
git commit -m "refactor(ohh): build_state_from_hands uses canonical hand_stats reconstruction"
```

---

## Phase 3 — Filtered projection, orchestration, and UI fixes

### Task 9: Expose filtered indices and add `build_projection`

**Files:**
- Modify: `src/bin/rsp/tui/filtered_log.rs`
- Modify: `src/bin/rsp/tui/projection.rs`

- [ ] **Step 1: Write the failing test for `indices()`**

Add to the `tests` module in `filtered_log.rs`:

```rust
    #[test]
    fn test_indices_exposes_filtered_game_numbers() {
        let log = FilteredGameLog::test_with_filter(10, vec![2, 5, 8]);
        assert_eq!(log.indices(), &[2, 5, 8]);
    }
```

Run: `mise check:test:nextest test_indices_exposes_filtered_game_numbers`
Expected: FAIL.

- [ ] **Step 2: Add the accessor**

In `src/bin/rsp/tui/filtered_log.rs`, add to the main `impl FilteredGameLog`:

```rust
    /// The 1-based game numbers currently passing the filter (empty when no
    /// filter is active).
    pub fn indices(&self) -> &[usize] {
        &self.filtered_indices
    }
```

Run: `mise check:test:nextest test_indices_exposes_filtered_game_numbers`
Expected: PASS.

- [ ] **Step 3: Write the failing test for `build_projection`**

Add to the `tests` module in `projection.rs`:

```rust
    #[test]
    fn test_build_projection_from_disk_matches_folding_those_hands() {
        use crate::tui::hand_store::HandStore;
        use rs_poker::open_hand_history::OpenHandHistoryWrapper;
        use std::io::Write;

        // Write two simple heads-up hands to a temp OHH file.
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        for gn in ["1", "2"] {
            let h = crate::tui::hand_stats::tests_support::simple_hand(gn);
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
```

This references a small public test fixture in `hand_stats`. Add it to `hand_stats.rs` (outside the private `tests` module so other modules' tests can use it):

```rust
#[cfg(test)]
pub mod tests_support {
    use rs_poker::open_hand_history::*;

    /// A minimal valid heads-up hand: SB folds to BB preflop.
    pub fn simple_hand(game_number: &str) -> HandHistory {
        HandHistory {
            spec_version: "1.4.7".into(),
            site_name: "test".into(),
            network_name: "test".into(),
            internal_version: "1.0".into(),
            tournament: false,
            tournament_info: None,
            game_number: game_number.into(),
            start_date_utc: None,
            table_name: "test".into(),
            table_handle: None,
            table_skin: None,
            game_type: GameType::Holdem,
            bet_limit: None,
            table_size: 2,
            currency: "USD".into(),
            dealer_seat: 0,
            small_blind_amount: 5.0,
            big_blind_amount: 10.0,
            ante_amount: 0.0,
            hero_player_id: None,
            players: vec![
                PlayerObj { id: 1, seat: 0, name: "A".into(), display: None, starting_stack: 1000.0, player_bounty: None, is_sitting_out: None },
                PlayerObj { id: 2, seat: 1, name: "B".into(), display: None, starting_stack: 1000.0, player_bounty: None, is_sitting_out: None },
            ],
            rounds: vec![RoundObj {
                id: 0,
                street: "Preflop".into(),
                cards: None,
                actions: vec![
                    ActionObj { action_number: 1, player_id: 1, action: Action::PostSmallBlind, amount: 5.0, is_allin: false, cards: None },
                    ActionObj { action_number: 2, player_id: 2, action: Action::PostBigBlind, amount: 10.0, is_allin: false, cards: None },
                    ActionObj { action_number: 3, player_id: 1, action: Action::Fold, amount: 0.0, is_allin: false, cards: None },
                ],
            }],
            pots: vec![PotObj { number: 0, amount: 15.0, rake: None, jackpot: None, player_wins: vec![PlayerWinsObj { player_id: 2, win_amount: 15.0, cashout_amount: None, cashout_fee: None, bonus_amount: None, contributed_rake: None }] }],
            tournament_bounties: None,
        }
    }
}
```

Run: `mise check:test:nextest test_build_projection_from_disk_matches_folding_those_hands`
Expected: FAIL — `build_projection` not found.

- [ ] **Step 4: Implement `build_projection`**

Add to `src/bin/rsp/tui/projection.rs` (after the `impl Projection` block):

```rust
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
```

Run: `mise check:test:nextest test_build_projection_from_disk_matches_folding_those_hands`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bin/rsp/tui/filtered_log.rs src/bin/rsp/tui/projection.rs src/bin/rsp/tui/hand_stats.rs
git commit -m "feat(tui): expose filtered indices and add build_projection disk recompute"
```

---

### Task 10: Wire filter changes and live append to the filtered projection

**Files:**
- Modify: `src/bin/rsp/tui/app.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `app.rs` (it already has TUI test scaffolding; use `App::new` and the public `state`/`filtered_log`). This test drives a live append with an active street filter and asserts the filtered projection reflects only matching games:

```rust
    #[test]
    fn test_filtered_projection_reflects_only_matching_games() {
        use crate::tui::state::{GameResult, RoundLabel, SeatStats};
        use rs_poker::arena::historian::StatsStorage;

        fn gr(name: &str, profit: f32, round: RoundLabel) -> GameResult {
            let mut s = StatsStorage::new_with_num_players(1);
            s.total_profit[0] = profit;
            s.hands_played[0] = 1;
            if profit > 0.0 { s.games_won[0] = 1; } else { s.games_lost[0] = 1; }
            GameResult {
                agent_names: vec![name.into()],
                profits: vec![profit],
                ending_round: round,
                seat_stats: vec![SeatStats::from_storage(&s, 0)],
                big_blind: 10.0,
            }
        }

        let mut app = App::new(None);
        // Activate a River-only filter directly, then rebuild (no disk: base
        // remains the source of incremental matching during live append).
        app.state.filter.toggle_street(RoundLabel::River);
        app.apply_filter_change();

        // A flop game (no match) then a river game (match).
        app.handle_sim_message(crate::tui::event::SimMessage::GameResult(gr("A", 5.0, RoundLabel::Flop)));
        app.handle_sim_message(crate::tui::event::SimMessage::GameResult(gr("A", 7.0, RoundLabel::River)));

        // Base saw both; matching (filtered) saw only the river game.
        assert_eq!(app.state.games_completed(), 2);
        assert_eq!(app.state.matching_games(), 1);
    }
```

Run: `mise check:test:nextest test_filtered_projection_reflects_only_matching_games`
Expected: FAIL — `apply_filter_change` not found.

- [ ] **Step 2: Add `apply_filter_change` and call it from filter mutations**

In `src/bin/rsp/tui/app.rs`, add a method on `impl App` that rebuilds both the filtered index list and the filtered projection:

```rust
    /// Rebuild the filtered game-log indices and the filtered stats projection
    /// after the filter changes. Synchronous full recompute from disk; only
    /// games matching the filter are parsed for stats.
    pub fn apply_filter_change(&mut self) {
        self.filtered_log
            .rebuild_filter(&self.state.filter, &self.hand_store);
        if self.state.filter.is_active() {
            let proj = crate::tui::projection::build_projection(
                self.filtered_log.indices().iter().copied(),
                &self.hand_store,
            );
            self.state.set_filter_projection(Some(proj));
        } else {
            self.state.set_filter_projection(None);
        }
    }
```

Replace `reset_log_selection` (lines ~407-412) to call it:

```rust
    fn reset_log_selection(&mut self) {
        self.state.log_selected = None;
        self.state.log_scroll = 0;
        self.apply_filter_change();
    }
```

- [ ] **Step 3: Fold matching live games into the filtered projection**

Update `handle_sim_message`'s `GameResult` arm (from Task 2 Step 5) to also fold matching games into the filtered projection:

```rust
            SimMessage::GameResult(result) => {
                let entry = GameLogEntry::new(
                    self.state.games_completed() + 1,
                    result.agent_names.clone(),
                    result.profits.clone(),
                    result.ending_round,
                    result.big_blind,
                );
                self.state.update(&result);
                if self.state.filter.is_active() && self.state.filter.matches_entry(&entry) {
                    self.state.fold_filtered(&result);
                }
                self.filtered_log.on_new_game(&entry, &self.state.filter);
            }
```

Run: `mise check:test:nextest test_filtered_projection_reflects_only_matching_games`
Expected: PASS.

- [ ] **Step 4: Run the full app test module**

Run: `mise check:test:nextest`
Expected: PASS (existing app/filter/navigation tests unaffected — `reset_log_selection` still rebuilds the index list, now also the projection).

- [ ] **Step 5: Commit**

```bash
git add src/bin/rsp/tui/app.rs
git commit -m "feat(tui): rebuild filtered projection on filter change and live append"
```

---

### Task 11: Hide the ETA for unbounded runs

**Files:**
- Modify: `src/bin/rsp/tui/widgets/progress_bar.rs`

- [ ] **Step 1: Write the failing snapshot test**

Add to the `tests` module in `progress_bar.rs`:

```rust
    #[test]
    fn test_render_live_progress_unbounded_omits_eta() {
        let backend = TestBackend::new(100, 1);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut state = TuiState::new(None); // unbounded => no ETA
        state.set_games_completed(5);
        terminal
            .draw(|frame| {
                render_progress(frame, frame.area(), &state);
            })
            .unwrap();
        let buf = format!("{:?}", terminal.backend().buffer());
        assert!(!buf.contains("ETA"), "unbounded runs must not render an ETA");
    }
```

Run: `mise check:test:nextest test_render_live_progress_unbounded_omits_eta`
Expected: FAIL — current code always renders `ETA --:--`.

- [ ] **Step 2: Conditionally build the ETA segment**

In `render_live_progress` (lines ~78-89), replace the unconditional `eta_str` + `spans` construction with:

```rust
    let mut detail = format!(" │ {} │ ", elapsed_str);
    if let Some(eta) = state.eta() {
        detail = format!(
            " │ {} │ ETA {}:{:02} │ ",
            elapsed_str,
            eta.as_secs() / 60,
            eta.as_secs() % 60
        );
    }

    let mut spans = vec![
        Span::styled(format!(" {:.1} g/s", gps), Style::default().fg(SKY)),
        Span::styled(detail, Style::default().fg(SUBTEXT0)),
    ];
    spans.extend(keybinding_hints());
```

Note: `state.eta()` already returns `None` for unbounded runs and for completed runs, so a bounded run still shows `ETA m:ss` and an unbounded run shows just the elapsed time.

Run: `mise check:test:nextest test_render_live_progress_unbounded_omits_eta`
Expected: PASS.

- [ ] **Step 3: Review/accept any changed snapshots and commit**

Bounded-run snapshots that contained `ETA --:--` (if any) need review:

```bash
mise check:test:nextest progress_bar
cargo insta review   # accept intended snapshot changes, if the repo uses insta review
```

```bash
git add src/bin/rsp/tui/widgets/progress_bar.rs
git add -A   # include any updated .snap files
git commit -m "feat(tui): omit ETA for unbounded generation"
```

---

### Task 12: Show `{matching} / {total}` in the status bar under a filter

**Files:**
- Modify: `src/bin/rsp/tui/widgets/progress_bar.rs`

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `progress_bar.rs`:

```rust
    #[test]
    fn test_static_status_shows_matching_over_total_when_filtered() {
        use crate::tui::state::RoundLabel;
        let backend = TestBackend::new(120, 1);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut state = TuiState::new(Some(50));
        state.live = false;
        state.set_games_completed(50);
        // Simulate an active filter with a filtered projection of 7 games.
        state.filter.toggle_street(RoundLabel::River);
        let mut proj = crate::tui::projection::Projection::new();
        proj.set_game_count(7);
        state.set_filter_projection(Some(proj));

        terminal
            .draw(|frame| render_progress(frame, frame.area(), &state))
            .unwrap();
        let buf = format!("{:?}", terminal.backend().buffer());
        assert!(buf.contains("7 / 50"), "filtered static status should show matching/total");
    }

    #[test]
    fn test_static_status_shows_total_when_unfiltered() {
        let backend = TestBackend::new(120, 1);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut state = TuiState::new(Some(50));
        state.live = false;
        state.set_games_completed(42);
        terminal
            .draw(|frame| render_progress(frame, frame.area(), &state))
            .unwrap();
        let buf = format!("{:?}", terminal.backend().buffer());
        assert!(buf.contains("42 games loaded"));
    }
```

`Projection::set_game_count` is `#[cfg(test)]` (added in Task 1) — make it visible to this test by ensuring it is `pub` under cfg(test) (it is). 

Run: `mise check:test:nextest test_static_status_shows_matching_over_total_when_filtered`
Expected: FAIL.

- [ ] **Step 2: Implement the count in `render_static_status` and `render_live_progress`**

Replace `render_static_status` (lines ~95-106):

```rust
fn render_static_status(frame: &mut Frame, area: Rect, state: &TuiState) {
    let total = state.games_completed();
    let label = if state.filter.is_active() {
        format!(" {} / {} games ", state.matching_games(), total)
    } else {
        format!(" {} games loaded ", total)
    };
    let mut spans = vec![
        Span::styled(label, Style::default().fg(SKY)),
        Span::styled("│ ", Style::default().fg(SUBTEXT0)),
    ];
    spans.extend(keybinding_hints());
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}
```

In `render_live_progress`, append a matching indicator to the left counter/gauge when a filter is active. After the existing `if let Some(target) = state.games_target { ... } else { ... }` block (lines ~54-71), add a filter note into the right-hand `detail` string built in Task 11. Change the `detail` initialization there to include matching info:

```rust
    let mut detail = if state.filter.is_active() {
        format!(" │ {} │ {} match │ ", elapsed_str, state.matching_games())
    } else {
        format!(" │ {} │ ", elapsed_str)
    };
    if let Some(eta) = state.eta() {
        let eta_str = format!("{}:{:02}", eta.as_secs() / 60, eta.as_secs() % 60);
        detail = if state.filter.is_active() {
            format!(
                " │ {} │ ETA {} │ {} match │ ",
                elapsed_str,
                eta_str,
                state.matching_games()
            )
        } else {
            format!(" │ {} │ ETA {} │ ", elapsed_str, eta_str)
        };
    }
```

(Live mode shows total in the gauge/counter on the left and `{matching} match` on the right; static mode shows `{matching} / {total}`. Both satisfy "matching out of total".)

Run: `mise check:test:nextest test_static_status_shows_matching_over_total_when_filtered` and `test_static_status_shows_total_when_unfiltered`
Expected: PASS.

- [ ] **Step 3: Run all progress_bar tests, review snapshots, commit**

```bash
mise check:test:nextest progress_bar
cargo insta review   # if snapshots changed
git add -A
git commit -m "feat(tui): show matching/total game count when a filter is active"
```

---

## Final Verification

- [ ] **Step 1: Full check**

Run: `mise check`
Expected: format, clippy, tests, and TOML lint all PASS.

- [ ] **Step 2: Manual smoke (optional, recommended)**

Build and run a short bounded live generation, then an unbounded one, and an `ohh view` over a generated file. Confirm:
- Toggling a participant/street/size filter updates the summary table, the profit graph (filtered ordinal x-axis), and the street bars to reflect only matching games.
- Clearing the filter restores the full view.
- Unbounded generation shows no `ETA`.
- The status bar shows `{matching} / {total}` (static) or `{matching} match` (live) under a filter.

Use the `run` skill or:
```bash
cargo run --bin rsp -- arena generate ...   # bounded and unbounded
cargo run --bin rsp -- ohh view <path>
```

- [ ] **Step 3: Secondary fuzz validation (optional)**

Per CLAUDE.md, after `mise check` passes:
```bash
mise fuzz replay_agent
mise fuzz config_agent
```

---

## Self-Review Notes (addressed)

- **Spec coverage:** summary filtered (Task 2/10), graph filtered + ordinal (Task 1 `fold` indexing + Task 10 wiring), street bars filtered (Task 2 `street_dist()`), ETA omitted (Task 11), matching/total (Task 12), recompute-from-disk (Task 9/10), full displayed-stat parity (Tasks 3-8), single canonical reconstruction (Task 8).
- **Scoped deviation from spec:** "full parity" is implemented for the metrics the TUI actually displays (`AgentDisplayData`); steal and per-street splits are intentionally not reconstructed (the viewer never reads them). Documented in `hand_stats.rs` module docs.
- **Type consistency:** `game_result_from_hand`, `build_projection`, `apply_filter_change`, `matching_games`, `games_completed()`, `set_filter_projection`, `fold_filtered`, `street_dist()`, `indices()` are used consistently across tasks.
- **Equivalence guard:** the live-fold path (full historian `GameResult`) and the disk-recompute path (`game_result_from_hand`) agree on displayed stats because Phase 2 lands before the filtered fold is wired (Phase 3), and the per-metric reconstruction tests encode the historian definitions.
