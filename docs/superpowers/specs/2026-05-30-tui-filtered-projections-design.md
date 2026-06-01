# TUI Filtered Projections — Design

**Date:** 2026-05-30
**Branch:** `elliott/tui_filtering`
**Status:** Design (pre-implementation)

## Problem

The arena `generate` / `ohh view` TUI has a game-log list, a facet filter, a
per-agent summary table, a cumulative-profit graph, and a street-distribution
panel. Today the **filter only narrows the game-log list**. The summary, the
graph, and the street bars are global accumulators that ignore the filter.
Three smaller UI issues compound this:

1. The graph shows all games regardless of the active filter.
2. The summary (stats table + street bars) shows all games, not the filtered set.
3. Unbounded generation (`--num-games 0`, `games_target == None`) still renders
   `ETA --:--` instead of omitting the ETA.
4. The bottom status shows only a total count; under a filter it should show
   "matching / total".

## Goals

- When a filter is active, the **summary, graph, and street bars re-derive from
  only the matching games**; clearing the filter restores the global view.
- The graph, under a filter, plots cumulative profit over the **filtered ordinal
  1..N** (dense, continuous), not the absolute game number.
- The bottom status shows **`{matching} / {total}`** when a filter is active.
- Unbounded generation **omits the ETA** segment entirely.
- One canonical definition of "stats from a game", used by live generation, the
  static OHH viewer, and filtered recompute — so the filtered summary matches
  the unfiltered summary exactly (stat parity), and the static viewer gains the
  full stat set it currently lacks.

## Non-goals

- No change to the filter facets themselves (winners/losers/participants/
  streets/win-size/loss-size/player-count) or their AND semantics.
- No async/background recompute (filter recompute is synchronous — see
  Decisions). The recompute is isolated behind one function so async can be
  added later without touching call sites.
- No new persistence format; the OHH file on disk remains the source of truth.

## Decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Where filtered per-game data comes from | **Recompute from disk** (`HandStore`) | Matches the codebase's disk-backed, memory-bounded design; no per-game in-memory log. |
| Graph x-axis under filter | **Re-index to filtered ordinal 1..N** | Clean continuous line answering "how did agents do across matching games". |
| Recompute timing | **Synchronous** on filter change | Simplest, deterministic, testable; filter toggles are discrete (not per-keystroke). Behind a single seam for a future async swap. |
| Stat parity | **Enrich `from_hand` to full parity** | The "more correct" option: filtered == unfiltered in live mode, and the static viewer gains cbet/wtsd/3bet/steal/per-street stats it lacks today. |

## Architecture

The design follows the Elm/Flux shape the TUI already uses (Model → update →
view) and layers it with a CQRS-style read/write split and the standard
incremental-on-append / full-recompute-on-filter hybrid for view maintenance.

```
 Canonical store (disk)         Filter (predicate)        Projections (read models)
 ┌──────────────────┐          ┌───────────────┐         ┌─────────────────────────┐
 │ HandStore (OHH)   │  ──┐     │ FilterState   │   ──▶   │ base: Projection (all)  │  ← summary/graph
 │ + FilteredGameLog │    ├────▶│ matches_entry │         │ filtered: Option<Proj.> │     read the
 │   (indices)       │  ──┘     └───────────────┘         └─────────────────────────┘     ACTIVE one
 └──────────────────┘
```

### Layers and their one job

- **Canonical store** — `HandStore` (full hands on disk) + `FilteredGameLog`
  (the matching game numbers). Unchanged in spirit; gains an accessor to iterate
  matching indices.
- **Filter** — `FilterState` + `matches_entry` (unchanged).
- **Projection** (NEW, `tui/projection.rs`) — a self-contained accumulator: fold
  games in, read derived display out. No UI/meta state. This is the unit that
  was previously tangled inside `TuiState`.
- **Model** — `TuiState` keeps UI/meta state and holds `base: Projection` plus
  `filtered: Option<Projection>`. Views read the **active** projection.

### `Projection` (tui/projection.rs)

Holds exactly what the summary, graph, and street bars need — the accumulator
fields lifted out of `TuiState`:

```rust
pub struct Projection {
    agent_stats: HashMap<String, StatsStorage>,
    agent_profit_bb: HashMap<String, f32>,
    agent_profit_history: HashMap<String, ProfitHistory>, // indexed 1..game_count
    street_dist: StreetDistribution,
    game_count: usize,
    cached_agent_display: Option<Vec<AgentDisplayData>>,   // memoized
}

impl Projection {
    pub fn fold(&mut self, result: &GameResult);                       // O(1) incremental
    pub fn agent_display_data(&mut self, sort: SortColumn) -> Vec<AgentDisplayData>; // memoized
    pub fn profit_histories(&self) -> &HashMap<String, ProfitHistory>;
    pub fn street_dist(&self) -> &StreetDistribution;
    pub fn game_count(&self) -> usize;
    pub fn invalidate_display_cache(&mut self);
}
```

- `fold` is the current body of `TuiState::update`, minus the meta bookkeeping.
  Because each projection counts its own `game_count`, the filtered projection's
  `ProfitHistory` naturally indexes from 1 (the filtered ordinal) without extra
  logic.
- Memoization stays per-projection (each caches its own `agent_display_data`),
  keyed implicitly by invalidation on `fold`/sort change.

### `TuiState` (tui/state.rs, refactored)

Keeps UI + meta state and the two projections:

```rust
pub struct TuiState {
    pub games_target: Option<usize>,
    pub start_time: Instant,
    pub completed: bool,
    pub live: bool,
    pub error: Option<SimError>,

    base: Projection,             // ALL games — maintained incrementally
    filtered: Option<Projection>, // Some(..) iff filter.is_active()
    pub distinct_player_counts: BTreeSet<usize>, // from ALL games (filter-panel options)

    // UI state (unchanged): table_selected, log_selected, log_scroll,
    // sort_col, active_panel, filter, cached_agent_names
}
```

Key accessors:

- `active_projection_mut()` → `self.filtered.as_mut().unwrap_or(&mut self.base)`.
- `agent_display_data()`, `profit_histories()`, `street_dist()` delegate to the
  **active** projection — so the summary/graph/street bars reflect the filter.
- `all_agent_names()`, `distinct_player_counts` read **base** — so the filter
  panel's available facets stay stable regardless of the active filter.
- `games_completed()` returns `base.game_count()` (single source of truth; the
  free-standing `games_completed` counter is removed to avoid drift).

### Data flow per event

**Append (`GameArrived(GameResult)` — live generation):**
1. `base.fold(&result)` — always.
2. `filtered_log.on_new_game(&entry, &filter)` — push index if it matches.
3. If `filter.is_active()` and `filter.matches_entry(&entry)`:
   `filtered.fold(&result)` — keeps the filtered summary/graph live with no disk
   hit.

**Filter change (toggle in the filter panel):**
1. `filtered_log.rebuild_filter(&filter, &hand_store)` — rebuild matching indices
   (cheap; uses lightweight `fetch_entry`).
2. If active: `state.filtered = Some(build_projection(filtered_log.indices(),
   &hand_store))` — full recompute. Only **matching** games get a full-hand
   parse (`hand_store.fetch`); non-matching games are never parsed. If inactive:
   `state.filtered = None`.

**Render (`view`):** pure — reads the active projection's memoized
`agent_display_data` and borrowed `profit_histories`/`street_dist`. No folding or
filtering in the render path.

```rust
// tui/projection.rs — the single recompute seam (future async swap point)
pub fn build_projection<I: IntoIterator<Item = usize>>(
    game_numbers: I,
    hand_store: &HandStore,
) -> Projection {
    let mut proj = Projection::new();
    for n in game_numbers {
        if let Ok(Some(hand)) = hand_store.fetch(n) {
            proj.fold(&game_result_from_hand(&hand));
        }
    }
    proj
}
```

### Canonical stat reconstruction (tui/hand_stats.rs, NEW) — stat parity

Today there are two stat paths that disagree:

- Live: `GameResult` from the simulation `StatsTrackingHistorian` — **full** stat
  set.
- Disk/static: `ohh::stats::build_state_from_hands` — **partial** (vpip, pfr,
  fold/call/bet/raise counts, profit only).

We unify on **one** reconstruction that mirrors the historian's definitions:

```rust
// tui/hand_stats.rs
pub fn seat_stats_from_hand(hand: &HandHistory) -> Vec<SeatStats>;
pub fn game_result_from_hand(hand: &HandHistory) -> GameResult;
```

`build_state_from_hands` and `build_projection` both call this, so live, static,
and filtered views compute identical numbers.

Each metric is reconstructed to match `StatsTrackingHistorian`
(`src/arena/historian/stats_tracking.rs`). All are reconstructable from the OHH
`HandHistory`; the implementation notes below capture the definitions and the
reconstruction caveats verified during design:

| Metric | Definition (historian) | OHH reconstruction notes |
|---|---|---|
| VPIP | Voluntary preflop money in (Bet/Raise/Call), binary per hand | Exclude forced blinds (`PostSmallBlind`/`PostBigBlind`/`PostAnte`/…); count only voluntary preflop actions |
| PFR | ≥1 preflop `Raise`, binary per hand | `Raise` actions are explicit in OHH |
| 3-bet | Re-raise of the open-raise (raise #2); opp = facing raise #1 | Track preflop raise sequence number |
| C-bet | Preflop aggressor + first to act on flop with no prior bet; opp marked on bet **or** fold | Preflop aggressor = last preflop raiser; order from `dealer_seat`+seats |
| WTSD | Saw flop (active into flop); count = active at showdown w/ ≥2 players | "active" = never folded; ≥2 players in final `player_wins` |
| Showdown / WSD | At `Complete` with ≥2 active; win = profit > 0.01 | From folds + `pots[].player_wins` |
| Steal | In CO/BTN/SB, folded to, raised; opp = in position & folded to | Position from `dealer_seat`+`table_size`; folded-to from preflop fold sequence |
| AF + per-street | (bets + raises) / calls, overall and per street | `Bet`/`Raise`/`Call` explicit; group by round street |
| Per-street completes/wins | Completes per round reached; win recorded at ending street | Infer from round sequence present in the hand |
| Profit / invested / W-L-BE | profit from rewards; invested = all money in incl. forced; W/L/BE by sign (±0.01) | `total_invested` = sum of bet/raise/call/all-in + forced bets; profit from `player_wins` − invested |

### The four UI fixes

- **Graph filtered** — falls out: `render_profit_chart` reads
  `state.profit_histories()` → active projection. Filtered ordinal is automatic
  (filtered projection counts from 1).
- **Summary + street bars filtered** — `render_stats_table` /
  `render_street_bars` read `state.agent_display_data()` / `state.street_dist()`
  → active projection.
- **ETA omitted when unbounded** — in `render_live_progress`, build the ETA span
  only when `games_target.is_some()`; otherwise omit the `ETA …` segment (no
  `--:--`).
- **Count `{matching} / {total}`** — when `filter.is_active()`, the status bar
  shows matching vs total in both live and static modes:
  - Live, bounded: gauge label `{completed} / {target}` unchanged; append
    `{matching} match` on the right when filtered.
  - Live, unbounded: `{total} games` → `{total} games · {matching} match`.
  - Static: `{total} games loaded` → `{matching} / {total} games`.

## Files changed

| File | Change |
|---|---|
| `tui/projection.rs` | **NEW** — `Projection`, `fold`, memoized display, `build_projection`. |
| `tui/hand_stats.rs` | **NEW** — `seat_stats_from_hand` / `game_result_from_hand` (full parity). |
| `tui/state.rs` | Refactor: `TuiState` holds `base`/`filtered`; delegate accessors; drop standalone `games_completed`; move accumulator logic into `Projection`. |
| `tui/filtered_log.rs` | Add `indices()` accessor for the recompute. |
| `tui/app.rs` | Orchestrate append-fold and filter-change recompute. |
| `tui/widgets/progress_bar.rs` | ETA-when-bounded; `{matching}/{total}` count. |
| `ohh/stats.rs` | `build_state_from_hands` delegates to `hand_stats`. |
| `tui/screens/overview.rs` & widgets | Read via active-projection accessors (mechanical). |

## Testing (TDD)

Tests are written before the code they cover. Conventions follow the existing
TUI tests (`#[test]` unit tests; `insta` snapshots for widgets;
`HandStore::none()` and hand builders for fixtures).

1. **`Projection` unit tests** — fold N `GameResult`s, assert accumulated
   stats/profit/street_dist; memoized `agent_display_data` returns cached result
   and invalidates on sort change.
2. **IVM drift insurance** — `fold`-incrementally over a list equals
   `build_projection` over the same list (the classic delta-vs-truth guard).
3. **Stat-parity equivalence (key correctness test)** — run a seeded simulation
   with `StatsTrackingHistorian`, capture its `GameResult`; also write the hand
   to OHH and reconstruct via `game_result_from_hand`; assert the two `SeatStats`
   are equal field-by-field. Covers each metric group; documents any approved
   tolerance.
4. **Filter recompute** — given a `HandStore` fixture + a filter, `build_projection`
   over the matching indices equals folding only the matching games; clearing the
   filter restores `base`.
5. **Active-projection routing** — with a filter active, `agent_display_data` /
   `profit_histories` / `street_dist` reflect the filtered set; `all_agent_names`
   / `distinct_player_counts` still reflect all games.
6. **Graph ordinal** — filtered projection's `ProfitHistory.first_game_index`
   starts at 1 and x-axis is the filtered ordinal.
7. **Render** — ETA span absent when `games_target == None`; present when
   bounded. Count string snapshots for filtered/unfiltered × live/static.

## Risks & mitigations

- **Reconstruction parity edge cases** (multi-way preflop aggressor for cbet,
  position mapping for steals, all-in shorthand, rake in `win_amount`). Mitigated
  by the equivalence test (#3) run over varied seeded hands; any unavoidable
  tolerance is documented there.
- **Recompute latency on large files with broad filters** — full-hand parse of
  every matching game is synchronous. Accepted per decision; isolated behind
  `build_projection` so an async/background variant can be slotted in later. The
  lightweight `fetch_entry` pre-filter ensures non-matching games are never
  parsed.
- **Refactor blast radius** — lifting accumulators out of `TuiState` touches many
  read sites. Mitigated by keeping `TuiState`'s public accessor names stable
  (delegating to the active projection), so most call sites are unchanged.

## Out of scope / future

- Async/background recompute with a generation token (swap behind `build_projection`).
- Caching filtered projections per filter-id (LRU) to make toggling back and
  forth instant.
